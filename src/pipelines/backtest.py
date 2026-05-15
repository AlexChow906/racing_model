"""
Backtester with matplotlib graphs.

Simple mode — score date range with pre-trained model:
    python -m src.pipelines.backtest --start 2025-01-01 --end 2025-12-31
    python -m src.pipelines.backtest --month 2025-06
    python -m src.pipelines.backtest --date 2025-04-12

Walk-forward mode — train fresh models per window:
    python -m src.pipelines.backtest --walk-forward

Options:
    --category flat|jumps
    --min-edge 0.05
    --params tuned|default
    --output bets.csv
    --no-graphs
"""
from __future__ import annotations

import argparse
import calendar
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db
from constants.features import EXCLUDE, FLAT_DROP, JUMPS_DROP, FLAT_V2_FEATURES
from constants.windows import WALK_FORWARD_WINDOWS
from constants.params import TUNED_PARAMS, DEFAULT_PARAMS, CATBOOST_FLAT_PARAMS

DB_PATH = os.environ.get("RACING_DB", str(ROOT / "racing.duckdb"))
MODELS_DIR = ROOT / "models"
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"flat": "#2196F3", "jumps": "#4CAF50", "combined": "#FF9800"}


# ── Scoring ──────────────────────────────────────────────────────────

def load_model(category, params="tuned"):
    model_dir = MODELS_DIR / params / category
    cbm_path = model_dir / "model.cbm"
    if cbm_path.exists():
        from catboost import CatBoostRanker
        model = CatBoostRanker()
        model.load_model(str(cbm_path))
        return model, None
    model = lgb.Booster(model_file=str(model_dir / "model.lgbm"))
    calibrator = None
    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
    return model, calibrator


def race_softmax(scores, race_ids):
    out = np.zeros_like(scores, dtype=float)
    start = 0
    n = len(scores)
    while start < n:
        rid = race_ids[start]
        end = start + 1
        while end < n and race_ids[end] == rid:
            end += 1
        chunk = scores[start:end]
        chunk = chunk - np.max(chunk)
        exps = np.exp(chunk)
        out[start:end] = exps / np.sum(exps)
        start = end
    return out


def renormalize(probs, race_ids):
    out = np.zeros_like(probs)
    start = 0
    n = len(probs)
    while start < n:
        rid = race_ids[start]
        end = start + 1
        while end < n and race_ids[end] == rid:
            end += 1
        chunk = probs[start:end]
        s = chunk.sum()
        out[start:end] = chunk / s if s > 0 else 1.0 / (end - start)
        start = end
    return out


def top_pick_win_rate(probs, race_ids, y_true):
    frame = pd.DataFrame({"race_id": race_ids, "prob": probs, "target": y_true})
    top = frame.sort_values(["race_id", "prob"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    return float(top["target"].mean())


def score_category(start_date, end_date, category, params="tuned"):
    """Score a date range with a pre-trained model. Returns bets DataFrame."""
    from modeling.train_split import load_data

    db = get_db(DB_PATH)
    type_filter = "AND ra.race_type = 'Flat'" if category == "flat" else "AND ra.race_type IN ('Chase', 'Hurdle', 'NH Flat')"

    df = db.execute(f"""
        SELECT fs.*, ra.race_type, ra.course_name, ra.going_code,
            ru.horse_name, res.sp_decimal
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        JOIN runners ru ON fs.runner_id = ru.runner_id
        JOIN results res ON fs.runner_id = res.runner_id
        WHERE fs.race_date >= '{start_date}' AND fs.race_date <= '{end_date}'
        AND fs.target IS NOT NULL
        AND res.sp_decimal IS NOT NULL
        {type_filter}
        ORDER BY fs.race_date, ra.scheduled_off_utc, fs.race_id
    """).df()
    db.close()

    if len(df) == 0:
        return None

    model, calibrator = load_model(category, params)
    expected_features = model.feature_name()

    meta_cols = ["race_type", "course_name", "going_code", "horse_name", "sp_decimal"]
    drop_list = FLAT_DROP if category == "flat" else JUMPS_DROP
    drop_cols = [c for c in EXCLUDE + drop_list + meta_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    non_numeric = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    for col in non_numeric:
        X[col] = pd.Categorical(X[col].astype(str)).codes

    for col in expected_features:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected_features]

    race_ids = df["race_id"].to_numpy()
    raw_scores = model.predict(X)
    probs = race_softmax(raw_scores, race_ids)

    if calibrator:
        probs = calibrator.transform(probs)
        probs = np.nan_to_num(probs, nan=1e-6)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = renormalize(probs, race_ids)

    sp = df["sp_decimal"].values
    implied = np.where((sp > 1) & ~np.isnan(sp), 1.0 / sp, np.nan)

    bets = pd.DataFrame({
        "date": pd.to_datetime(df["race_date"].values),
        "race_id": df["race_id"].values,
        "course": df["course_name"].values,
        "going": df["going_code"].values,
        "horse": df["horse_name"].values,
        "category": category,
        "model_prob": probs,
        "sp": sp,
        "implied": implied,
        "edge": probs - implied,
        "won": df["target"].values,
        "profit": df["target"].values * (sp - 1) - (1 - df["target"].values),
    })

    return bets


# ── Walk-forward ─────────────────────────────────────────────────────

def run_walk_forward(categories, params_name="tuned", flat_v2=False, flat_v2_engine="catboost"):
    """Run full walk-forward: train per window, return combined bets."""
    from modeling.train_split import load_data, load_flat_v2_data

    db = get_db(DB_PATH)
    incomplete = db.execute("""
        SELECT COUNT(*) FROM feature_store fs
        WHERE fs.runner_id IN (
            SELECT runner_id FROM results
            WHERE won = FALSE AND sp_decimal IS NULL AND finishing_position IS NULL
        )
    """).fetchone()[0]
    if incomplete > 0:
        print(f"  Removing {incomplete:,} incomplete runners from feature_store", flush=True)
        db.execute("""
            DELETE FROM feature_store
            WHERE runner_id IN (
                SELECT runner_id FROM results
                WHERE won = FALSE AND sp_decimal IS NULL AND finishing_position IS NULL
            )
        """)
        remaining = db.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        print(f"  Feature store: {remaining:,} rows remaining", flush=True)
    db.close()

    params_dict = TUNED_PARAMS if params_name == "tuned" else {"flat": DEFAULT_PARAMS, "jumps": DEFAULT_PARAMS}
    all_bets = []
    use_catboost = flat_v2 and flat_v2_engine == "catboost"

    for w in WALK_FORWARD_WINDOWS:
        print(f"\n  --- {w['label']} ---", flush=True)
        for category in categories:
            is_flat_v2 = flat_v2 and category == "flat"

            if is_flat_v2:
                X_train, y_train, g_train, df_train, feat_names = load_flat_v2_data("2015-01-01", w["train_end"])
                X_cal, y_cal, g_cal, df_cal, _ = load_flat_v2_data(w["cal_start"], w["cal_end"])
                X_test, y_test, g_test, df_test, _ = load_flat_v2_data(w["test_start"], w["test_end"])

                if len(X_test) == 0:
                    continue

                for extra_df in [X_cal, X_test]:
                    for col in feat_names:
                        if col not in extra_df.columns:
                            extra_df[col] = np.nan
                    drop_extra = [c for c in extra_df.columns if c not in feat_names]
                    extra_df.drop(columns=drop_extra, inplace=True, errors="ignore")
                X_cal = X_cal[feat_names]
                X_test = X_test[feat_names]

                if use_catboost:
                    from catboost import CatBoost, Pool
                    train_pool = Pool(X_train, label=y_train, group_id=np.repeat(
                        np.arange(len(g_train)), g_train))
                    cal_pool = Pool(X_cal, label=y_cal, group_id=np.repeat(
                        np.arange(len(g_cal)), g_cal))

                    model = CatBoost(CATBOOST_FLAT_PARAMS)
                    model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=200)
                    test_scores = model.predict(X_test)
                else:
                    model = lgb.LGBMRanker(
                        objective="lambdarank", n_estimators=3000,
                        learning_rate=0.03, num_leaves=50, min_child_samples=40,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, max_depth=6,
                        random_state=42, n_jobs=-1,
                    )
                    model.fit(
                        X_train, y_train, group=g_train,
                        eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1],
                        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(0)],
                    )
                    test_scores = model.predict(X_test, num_iteration=model.best_iteration_)

                test_ids = df_test["race_id"].to_numpy()
                test_probs = race_softmax(test_scores, test_ids)
            else:
                X_train, y_train, g_train, df_train = load_data("2015-01-01", w["train_end"], category)
                X_cal, y_cal, g_cal, df_cal = load_data(w["cal_start"], w["cal_end"], category)
                X_test, y_test, g_test, df_test = load_data(w["test_start"], w["test_end"], category)

                if len(X_test) == 0:
                    continue

                non_numeric = [c for c in X_train.columns
                               if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
                for col in non_numeric:
                    cats = pd.Index(pd.concat([X_train[col], X_cal[col], X_test[col]]).astype(str).astype("category").cat.categories)
                    X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
                    X_cal[col] = pd.Categorical(X_cal[col].astype(str), categories=cats).codes
                    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

                p = params_dict[category] if isinstance(params_dict, dict) and category in params_dict else params_dict
                model = lgb.LGBMRanker(**p)
                model.fit(X_train, y_train, group=g_train,
                          eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1],
                          callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(0)])

                cal_ids = df_cal["race_id"].to_numpy()
                cal_probs = race_softmax(model.predict(X_cal, num_iteration=model.best_iteration_), cal_ids)
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cal_probs, y_cal)

                test_ids = df_test["race_id"].to_numpy()
                calibrated = calibrator.transform(race_softmax(model.predict(X_test, num_iteration=model.best_iteration_), test_ids))
                calibrated = np.nan_to_num(calibrated, nan=1e-6)
                calibrated = np.clip(calibrated, 1e-6, 1.0)
                test_probs = renormalize(calibrated, test_ids)

            db = get_db(DB_PATH)
            sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1").df()
            race_info = db.execute("SELECT race_id, course_name, going_code FROM races").df()
            db.close()

            a = df_test[["race_id", "runner_id", "race_date"]].copy()
            a["horse_name"] = ""
            a["model_prob"] = test_probs
            a["target"] = y_test
            a = a.merge(sp_df, on="runner_id", how="left")
            a = a.merge(race_info, on="race_id", how="left")
            a["implied"] = np.where(a["sp_decimal"] > 1, 1.0 / a["sp_decimal"], np.nan)
            a["edge"] = a["model_prob"] - a["implied"]
            a["profit"] = a["target"] * (a["sp_decimal"] - 1) - (1 - a["target"])

            bets = pd.DataFrame({
                "date": pd.to_datetime(a["race_date"].values),
                "race_id": a["race_id"].values,
                "course": a["course_name"].values,
                "going": a["going_code"].values,
                "horse": a["horse_name"].values,
                "category": category,
                "model_prob": a["model_prob"].values,
                "sp": a["sp_decimal"].values,
                "implied": a["implied"].values,
                "edge": a["edge"].values,
                "won": a["target"].values,
                "profit": a["profit"].values,
                "window": w["label"],
            })
            all_bets.append(bets)
            min_edge = 0.08 if is_flat_v2 else 0.05
            vb = bets[(bets["edge"] > min_edge) & bets["sp"].notna()]
            roi = vb["profit"].mean() if len(vb) > 0 else 0
            label = f"{category}" + (" (v2)" if is_flat_v2 else "")
            print(f"    {label}: {len(vb)} bets (edge>{min_edge:.0%}), ROI={roi:+.2%}", flush=True)

    return pd.concat(all_bets) if all_bets else pd.DataFrame()


# ── Graphs ───────────────────────────────────────────────────────────

def generate_graphs(bets: pd.DataFrame, out_dir: Path, min_edge: float, is_walk_forward: bool):
    """Generate all matplotlib graphs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vb = bets[(bets["edge"] > min_edge) & bets["sp"].notna()].copy().sort_values("date")

    if len(vb) == 0:
        print("  No value bets to graph", flush=True)
        return

    vb["cum_pnl"] = vb["profit"].cumsum()
    vb["month"] = vb["date"].dt.to_period("M")

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    for cat in vb["category"].unique():
        sub = vb[vb["category"] == cat].copy()
        sub["cum"] = sub["profit"].cumsum()
        ax.plot(sub["date"], sub["cum"], label=cat.title(), color=COLORS.get(cat, "#999"), linewidth=1.5)
    if len(vb["category"].unique()) > 1:
        ax.plot(vb["date"], vb["cum_pnl"], label="Combined", color=COLORS["combined"], linewidth=2, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative P&L (£1 Level Stakes)")
    ax.set_ylabel("Profit (£)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    # 2. Monthly ROI
    monthly = vb.groupby("month").agg(bets=("profit", "size"), pnl=("profit", "sum")).reset_index()
    monthly["roi"] = monthly["pnl"] / monthly["bets"]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in monthly["roi"]]
    ax.bar(range(len(monthly)), monthly["roi"] * 100, color=colors)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly["month"]], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Monthly ROI (%)")
    ax.set_ylabel("ROI (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_roi.png", dpi=150)
    plt.close(fig)

    # 3. Drawdown
    peak = vb["cum_pnl"].cummax()
    drawdown = vb["cum_pnl"] - peak
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(vb["date"], drawdown, 0, color="#F44336", alpha=0.4)
    ax.plot(vb["date"], drawdown, color="#F44336", linewidth=0.8)
    ax.set_title("Drawdown from Peak")
    ax.set_ylabel("Drawdown (£)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close(fig)

    # 4. ROI by course (top 20) — strip date suffixes from Betfair course names
    import re
    vb["venue"] = vb["course"].apply(lambda c: re.sub(r'\s+\d+\w*\s+\w+$', '', str(c)).strip())
    course_stats = vb.groupby("venue").agg(bets=("profit", "size"), pnl=("profit", "sum")).reset_index()
    course_stats["roi"] = course_stats["pnl"] / course_stats["bets"]
    course_stats = course_stats[course_stats["bets"] >= 20].nlargest(20, "bets").sort_values("roi")
    if len(course_stats) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in course_stats["roi"]]
        labels = [str(c)[:15] for c in course_stats["venue"]]
        ax.barh(range(len(course_stats)), course_stats["roi"] * 100, color=colors)
        ax.set_yticks(range(len(course_stats)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("ROI by Course (min 20 bets)")
        ax.set_xlabel("ROI (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "roi_by_course.png", dpi=150)
        plt.close(fig)

    # 5. Flat vs Jumps
    cats = vb["category"].unique()
    if len(cats) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, cat in zip(axes, ["flat", "jumps"]):
            sub = vb[vb["category"] == cat].copy()
            if len(sub) == 0:
                continue
            sub["cum"] = sub["profit"].cumsum()
            ax.plot(sub["date"], sub["cum"], color=COLORS[cat], linewidth=1.5)
            ax.fill_between(sub["date"], sub["cum"], 0, alpha=0.2, color=COLORS[cat])
            ax.axhline(0, color="black", linewidth=0.5)
            roi = sub["profit"].mean()
            ax.set_title(f"{cat.title()} — {len(sub):,} bets, ROI {roi:+.1%}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "flat_vs_jumps.png", dpi=150)
        plt.close(fig)

    # 6. ROI by SP range
    sp_bins = [(1, 3, "1-3"), (3, 6, "3-6"), (6, 15, "6-15"), (15, 50, "15-50"), (50, 1000, "50+")]
    sp_data = []
    for lo, hi, label in sp_bins:
        sub = vb[(vb["sp"] >= lo) & (vb["sp"] < hi)]
        if len(sub) > 0:
            sp_data.append({"label": label, "bets": len(sub), "roi": sub["profit"].mean()})
    if sp_data:
        sp_df = pd.DataFrame(sp_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in sp_df["roi"]]
        bars = ax.bar(sp_df["label"], sp_df["roi"] * 100, color=colors)
        for bar, row in zip(bars, sp_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={row["bets"]}', ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("ROI by SP Range")
        ax.set_xlabel("SP Odds Range")
        ax.set_ylabel("ROI (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "roi_by_sp_range.png", dpi=150)
        plt.close(fig)

    # 7. ROI by edge bucket
    edge_bins = [(0.03, 0.05, "3-5%"), (0.05, 0.08, "5-8%"), (0.08, 0.12, "8-12%"), (0.12, 1.0, "12%+")]
    edge_data = []
    for lo, hi, label in edge_bins:
        sub = vb[(vb["edge"] >= lo) & (vb["edge"] < hi)]
        if len(sub) > 0:
            edge_data.append({"label": label, "bets": len(sub), "roi": sub["profit"].mean()})
    if edge_data:
        edge_df = pd.DataFrame(edge_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in edge_df["roi"]]
        bars = ax.bar(edge_df["label"], edge_df["roi"] * 100, color=colors)
        for bar, row in zip(bars, edge_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={row["bets"]}', ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("ROI by Edge Bucket")
        ax.set_xlabel("Model Edge")
        ax.set_ylabel("ROI (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "roi_by_edge.png", dpi=150)
        plt.close(fig)

    # 8. Calibration plot
    all_runners = bets[bets["sp"].notna()].copy()
    if len(all_runners) > 100:
        all_runners["prob_bin"] = pd.qcut(all_runners["model_prob"], q=10, duplicates="drop")
        cal = all_runners.groupby("prob_bin", observed=False).agg(
            avg_pred=("model_prob", "mean"), actual=("won", "mean"), count=("won", "size")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 0.5], [0, 0.5], "k--", linewidth=0.8, label="Perfect calibration")
        ax.scatter(cal["avg_pred"], cal["actual"], s=cal["count"] / 10, c="#2196F3", alpha=0.7)
        ax.plot(cal["avg_pred"], cal["actual"], color="#2196F3", linewidth=1.5, label="Model")
        ax.set_title("Calibration Plot")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.legend()
        ax.set_xlim(0, max(0.4, cal["avg_pred"].max() * 1.1))
        ax.set_ylim(0, max(0.4, cal["actual"].max() * 1.1))
        fig.tight_layout()
        fig.savefig(out_dir / "calibration.png", dpi=150)
        plt.close(fig)

    # 9. Rolling 90-day ROI
    if len(vb) > 100:
        vb_sorted = vb.sort_values("date").copy()
        vb_sorted["rolling_roi"] = vb_sorted["profit"].rolling(window=min(500, len(vb_sorted) // 3), min_periods=50).mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(vb_sorted["date"], vb_sorted["rolling_roi"] * 100, color="#2196F3", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.fill_between(vb_sorted["date"], vb_sorted["rolling_roi"] * 100, 0,
                         where=vb_sorted["rolling_roi"] >= 0, alpha=0.2, color="#4CAF50")
        ax.fill_between(vb_sorted["date"], vb_sorted["rolling_roi"] * 100, 0,
                         where=vb_sorted["rolling_roi"] < 0, alpha=0.2, color="#F44336")
        ax.set_title("Rolling ROI (500-bet window)")
        ax.set_ylabel("ROI (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "rolling_roi.png", dpi=150)
        plt.close(fig)

    # 10. Strike rate vs odds
    sp_strike_bins = [(1, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 35), (35, 60), (60, 150)]
    strike_data = []
    for lo, hi in sp_strike_bins:
        sub = vb[(vb["sp"] >= lo) & (vb["sp"] < hi)]
        if len(sub) >= 10:
            strike_data.append({"sp_mid": (lo + hi) / 2, "strike": sub["won"].mean(), "bets": len(sub),
                                "expected": 1.0 / ((lo + hi) / 2)})
    if strike_data:
        sd = pd.DataFrame(strike_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sd["sp_mid"], sd["strike"] * 100, s=sd["bets"] / 5, c="#2196F3", alpha=0.7, label="Actual")
        ax.plot(sd["sp_mid"], sd["expected"] * 100, "k--", linewidth=0.8, label="Market implied")
        ax.set_title("Strike Rate vs SP Odds")
        ax.set_xlabel("SP Odds")
        ax.set_ylabel("Strike Rate (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "strike_vs_odds.png", dpi=150)
        plt.close(fig)

    # 11. Walk-forward comparison (only if walk-forward data)
    if is_walk_forward and "window" in vb.columns:
        wf_data = []
        for window in vb["window"].unique():
            for cat in vb["category"].unique():
                sub = vb[(vb["window"] == window) & (vb["category"] == cat)]
                if len(sub) > 0:
                    wf_data.append({"window": window, "category": cat, "roi": sub["profit"].mean(), "bets": len(sub)})
        if wf_data:
            wf_df = pd.DataFrame(wf_data)
            windows = sorted(wf_df["window"].unique())
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35
            for i, cat in enumerate(["flat", "jumps"]):
                sub = wf_df[wf_df["category"] == cat]
                if len(sub) == 0:
                    continue
                x = [windows.index(w) for w in sub["window"]]
                ax.bar([xi + i * width for xi in x], sub["roi"] * 100, width,
                       label=cat.title(), color=COLORS[cat], alpha=0.8)
            ax.set_xticks([i + width / 2 for i in range(len(windows))])
            ax.set_xticklabels(windows)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title("Walk-Forward ROI by Window")
            ax.set_ylabel("ROI (%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "walk_forward_comparison.png", dpi=150)
            plt.close(fig)

    print(f"  Graphs saved to {out_dir}/", flush=True)


# ── Text summary ─────────────────────────────────────────────────────

def print_summary(bets, min_edge, out_dir=None):
    """Print and optionally save text summary."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s, flush=True)

    vb = bets[(bets["edge"] > min_edge) & bets["sp"].notna()]

    for cat in sorted(bets["category"].unique()):
        cat_all = bets[bets["category"] == cat]
        cat_vb = vb[vb["category"] == cat]
        race_ids = cat_all["race_id"].to_numpy()
        tpwr = top_pick_win_rate(cat_all["model_prob"].values, race_ids, cat_all["won"].values)

        p(f"\n{'='*65}")
        p(f"  {cat.upper()} RESULTS")
        p(f"{'='*65}")
        p(f"  Races: {cat_all['race_id'].nunique():,}  |  Runners: {len(cat_all):,}  |  Top Pick: {tpwr:.1%}")

        if len(cat_vb) > 0:
            p(f"\n  Value Bets (edge>{min_edge:.0%}):")
            p(f"  Bets: {len(cat_vb):,}")
            p(f"  Winners: {int(cat_vb['won'].sum()):,} ({cat_vb['won'].mean():.1%})")
            p(f"  Avg SP: {cat_vb['sp'].mean():.1f}")
            p(f"  P&L: £{cat_vb['profit'].sum():+,.2f}")
            p(f"  ROI: {cat_vb['profit'].mean():+.2%}")

            monthly = cat_vb.groupby(cat_vb["date"].dt.to_period("M")).agg(
                bets=("profit", "size"), wins=("won", "sum"), pnl=("profit", "sum")).reset_index()
            monthly["roi"] = monthly["pnl"] / monthly["bets"]
            monthly["cum"] = monthly["pnl"].cumsum()

            if len(monthly) > 1:
                p(f"\n  {'Month':<10} {'Bets':>6} {'Wins':>5} {'P&L':>9} {'ROI':>7} {'Cumulative':>11}")
                p(f"  {'-'*52}")
                for _, row in monthly.iterrows():
                    p(f"  {str(row['date']):<10} {row['bets']:>6} {int(row['wins']):>5} £{row['pnl']:>+8.0f} {row['roi']:>+6.1%} £{row['cum']:>+10.0f}")

    # Combined
    if len(vb["category"].unique()) > 1 and len(vb) > 0:
        p(f"\n{'='*65}")
        p(f"  COMBINED: {len(vb):,} bets, ROI={vb['profit'].mean():+.2%}, P&L=£{vb['profit'].sum():+,.0f}")
        p(f"{'='*65}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.txt").write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest with graphs")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--month", type=str, default=None)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--params", type=str, default="tuned", choices=["tuned", "default"])
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--category", type=str, default=None, choices=["flat", "jumps"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--flat-v2", action="store_true", help="Use CatBoost flat v2 model for flat races")
    parser.add_argument("--flat-v2-lgbm", action="store_true", help="Use LightGBM flat v2 model (no calibration) for comparison")
    args = parser.parse_args()

    categories = [args.category] if args.category else ["flat", "jumps"]
    if args.flat_v2 and args.min_edge == 0.05:
        args.min_edge = 0.08
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = ROOT / "reports" / f"backtest_{timestamp}"

    if args.walk_forward:
        print("Running walk-forward backtest...", flush=True)
        engine = "lgbm" if args.flat_v2_lgbm else "catboost"
        bets = run_walk_forward(categories, args.params, flat_v2=args.flat_v2 or args.flat_v2_lgbm, flat_v2_engine=engine)
    else:
        if args.date:
            start_date = end_date = args.date
        elif args.month:
            year, month = args.month.split("-")
            start_date = f"{year}-{month}-01"
            last_day = calendar.monthrange(int(year), int(month))[1]
            end_date = f"{year}-{month}-{last_day}"
        elif args.start and args.end:
            start_date = args.start
            end_date = args.end
        else:
            parser.error("Specify --date, --month, --start/--end, or --walk-forward")

        print(f"Backtesting {start_date} to {end_date}", flush=True)
        all_bets = []
        for category in categories:
            result = score_category(start_date, end_date, category, args.params)
            if result is not None:
                all_bets.append(result)
        bets = pd.concat(all_bets) if all_bets else pd.DataFrame()

    if len(bets) == 0:
        print("No data for this period.")
        return

    print_summary(bets, args.min_edge, out_dir if not args.no_graphs else None)

    if not args.no_graphs:
        print("\nGenerating graphs...", flush=True)
        generate_graphs(bets, out_dir, args.min_edge, args.walk_forward)

    if args.output:
        vb = bets[(bets["edge"] > args.min_edge) & bets["sp"].notna()]
        vb.to_csv(args.output, index=False)
        print(f"Saved {len(vb):,} bets to {args.output}")

    # Save all bets CSV
    if not args.no_graphs:
        bets.to_csv(out_dir / "all_bets.csv", index=False)


if __name__ == "__main__":
    main()
