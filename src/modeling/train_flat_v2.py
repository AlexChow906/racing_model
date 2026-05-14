"""
Flat v2 model: CatBoost ranker, minimal features, raw softmax (no isotonic calibration).

Single window:
    python -m src.modeling.train_flat_v2

Walk-forward validation across all 4 windows:
    python -m src.modeling.train_flat_v2 --walk-forward

Options:
    --min-edge 0.08   (default 0.08)
    --also-lgbm       (train LightGBM alongside for comparison)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import brier_score_loss, log_loss

from src.ingestion.db_connect import get_db
from src.constants.features import EXCLUDE, FLAT_V2_FEATURES
from src.constants.params import CATBOOST_FLAT_PARAMS
from src.constants.windows import (
    TRAIN_END, CAL_START, CAL_END, TEST_START, TEST_END,
    WALK_FORWARD_WINDOWS,
)

ROOT = Path(__file__).resolve().parents[2]


def load_flat_data(start, end):
    db = get_db(str(ROOT / "racing.duckdb"))
    df = db.execute(f"""
        SELECT fs.*, ra.race_type
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        WHERE fs.race_date >= '{start}' AND fs.race_date < '{end}'
        AND fs.target IS NOT NULL
        AND ra.race_type = 'Flat'
        ORDER BY fs.race_date, fs.race_id
    """).df()
    db.close()

    df = df[df["race_date"] >= "2015-01-01"].copy()
    df = df.drop(columns=["race_type"], errors="ignore")

    features = [f for f in FLAT_V2_FEATURES if f in df.columns]
    missing = [f for f in FLAT_V2_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: missing features: {missing}", flush=True)

    groups = df.groupby("race_id", sort=False)["runner_id"].count().values
    y = df["target"].astype(int).values
    X = df[features].copy()

    return X, y, groups, df, features


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


def top_pick_win_rate(probs, race_ids, y_true):
    frame = pd.DataFrame({"race_id": race_ids, "prob": probs, "target": y_true})
    top = frame.sort_values(["race_id", "prob"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    return float(top["target"].mean())


def value_analysis(df_test, probs, y_test, thresholds):
    db = get_db(str(ROOT / "racing.duckdb"))
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal IS NOT NULL AND sp_decimal > 1").df()
    race_info = db.execute("SELECT race_id, is_handicap FROM races").df()
    db.close()

    analysis = df_test[["race_id", "runner_id", "race_date"]].copy()
    analysis["prob"] = probs
    analysis["target"] = y_test
    analysis = analysis.merge(sp_df, on="runner_id", how="left")
    analysis = analysis.merge(race_info, on="race_id", how="left")
    analysis["implied"] = 1.0 / analysis["sp_decimal"]
    analysis["edge"] = analysis["prob"] - analysis["implied"]
    analysis["profit"] = analysis["target"] * (analysis["sp_decimal"] - 1) - (1 - analysis["target"])

    has_sp = analysis[analysis["sp_decimal"].notna()]

    results = {}
    for thresh in thresholds:
        vb = has_sp[has_sp["edge"] > thresh]
        if len(vb) > 0:
            roi = vb["profit"].mean()
            strike = vb["target"].mean()
            avg_sp = vb["sp_decimal"].mean()
            pnl = vb["profit"].sum()
            results[thresh] = {"bets": len(vb), "strike": strike, "roi": roi, "pnl": pnl, "avg_sp": avg_sp}

            hcap = vb[vb["is_handicap"] == True]
            non_hcap = vb[vb["is_handicap"] == False]
            hcap_roi = hcap["profit"].mean() if len(hcap) > 0 else 0
            non_hcap_roi = non_hcap["profit"].mean() if len(non_hcap) > 0 else 0
            results[thresh]["handicap_bets"] = len(hcap)
            results[thresh]["handicap_roi"] = hcap_roi
            results[thresh]["non_handicap_bets"] = len(non_hcap)
            results[thresh]["non_handicap_roi"] = non_hcap_roi

    return results, has_sp


def train_catboost_window(X_train, y_train, g_train, X_cal, y_cal, g_cal, features):
    group_ids_train = np.repeat(range(len(g_train)), g_train)
    group_ids_cal = np.repeat(range(len(g_cal)), g_cal)

    train_pool = Pool(data=X_train, label=y_train, group_id=group_ids_train)
    cal_pool = Pool(data=X_cal, label=y_cal, group_id=group_ids_cal)

    params = dict(CATBOOST_FLAT_PARAMS)
    params["early_stopping_rounds"] = 200

    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=cal_pool)

    return model


def run_single_window(train_start, train_end, cal_start, cal_end, test_start, test_end, label, min_edge):
    print(f"\n  Train: {train_start} → {train_end}", flush=True)
    print(f"  Cal:   {cal_start} → {cal_end}", flush=True)
    print(f"  Test:  {test_start} → {test_end}", flush=True)

    X_train, y_train, g_train, df_train, features = load_flat_data(train_start, train_end)
    X_cal, y_cal, g_cal, df_cal, _ = load_flat_data(cal_start, cal_end)
    X_test, y_test, g_test, df_test, _ = load_flat_data(test_start, test_end)

    if len(X_test) == 0:
        print("  No test data", flush=True)
        return None

    print(f"  Train: {len(X_train):,} rows, {len(g_train):,} races, {len(features)} features", flush=True)
    print(f"  Cal:   {len(X_cal):,} rows, {len(g_cal):,} races", flush=True)
    print(f"  Test:  {len(X_test):,} rows, {len(g_test):,} races", flush=True)

    model = train_catboost_window(X_train, y_train, g_train, X_cal, y_cal, g_cal, features)

    test_ids = df_test["race_id"].to_numpy()
    raw_scores = model.predict(X_test)
    probs = race_softmax(raw_scores, test_ids)

    tpwr = top_pick_win_rate(probs, test_ids, y_test)
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, np.clip(probs, 1e-15, 1 - 1e-15), labels=[0, 1])

    print(f"\n  {label}: TopPick={tpwr:.1%}  Brier={brier:.5f}  LogLoss={ll:.5f}  Trees={model.best_iteration_}", flush=True)

    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12]
    val_results, has_sp = value_analysis(df_test, probs, y_test, thresholds)

    print(f"\n  {'Threshold':<12} {'Bets':>6} {'Strike':>8} {'AvgSP':>7} {'ROI':>8} {'P&L':>9}  | {'Hcap':>5} {'H-ROI':>7} {'Non-H':>5} {'NH-ROI':>7}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for thresh in thresholds:
        if thresh in val_results:
            v = val_results[thresh]
            print(f"  edge>{thresh:<5.0%}  {v['bets']:>6} {v['strike']:>7.3f} {v['avg_sp']:>7.1f} {v['roi']:>+7.2%} £{v['pnl']:>+8.0f}  | {v['handicap_bets']:>5} {v['handicap_roi']:>+6.2%} {v['non_handicap_bets']:>5} {v['non_handicap_roi']:>+6.2%}", flush=True)

    sp_bins = [(1, 3), (3, 6), (6, 15), (15, 50), (50, 1000)]
    vb = has_sp[has_sp["edge"] > min_edge]
    if len(vb) > 0:
        print(f"\n  ROI by SP range (edge>{min_edge:.0%}):", flush=True)
        for lo, hi in sp_bins:
            sub = vb[(vb["sp_decimal"] >= lo) & (vb["sp_decimal"] < hi)]
            if len(sub) > 0:
                print(f"    SP {lo}-{hi}: {len(sub):>5} bets, ROI={sub['profit'].mean():>+7.2%}", flush=True)

    return {
        "label": label,
        "features": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "top_pick": float(tpwr),
        "brier": float(brier),
        "log_loss": float(ll),
        "best_iter": int(model.best_iteration_),
        "value": {f"edge_{int(t*100)}": v for t, v in val_results.items()},
        "model": model,
        "probs": probs,
        "df_test": df_test,
        "y_test": y_test,
    }


def run_walk_forward(min_edge):
    print(f"\n{'='*80}", flush=True)
    print(f"  FLAT V2 — CATBOOST WALK-FORWARD VALIDATION", flush=True)
    print(f"  Features: {len(FLAT_V2_FEATURES)} | Calibration: raw softmax | Min edge: {min_edge:.0%}", flush=True)
    print(f"{'='*80}", flush=True)

    all_results = []
    for w in WALK_FORWARD_WINDOWS:
        print(f"\n{'='*60}", flush=True)
        print(f"  {w['label']}", flush=True)
        print(f"{'='*60}", flush=True)

        result = run_single_window(
            "2015-01-01", w["train_end"],
            w["cal_start"], w["cal_end"],
            w["test_start"], w["test_end"],
            w["label"], min_edge,
        )
        if result:
            all_results.append(result)

    print(f"\n{'='*80}", flush=True)
    print(f"  WALK-FORWARD SUMMARY (CatBoost flat v2, edge>{min_edge:.0%})", flush=True)
    print(f"{'='*80}", flush=True)

    edge_key = f"edge_{int(min_edge * 100)}"
    print(f"  {'Window':<14} {'TopPick':>8} {'Brier':>8} {'Bets':>7} {'ROI':>8} {'P&L':>9}", flush=True)
    print(f"  {'-'*58}", flush=True)

    total_bets = 0
    total_pnl = 0.0
    for r in all_results:
        v = r["value"].get(edge_key, {})
        bets = v.get("bets", 0)
        roi = v.get("roi", 0)
        pnl = v.get("pnl", 0)
        total_bets += bets
        total_pnl += pnl
        print(f"  {r['label']:<14} {r['top_pick']:>7.1%} {r['brier']:>8.5f} {bets:>7,} {roi:>+7.2%} £{pnl:>+8.0f}", flush=True)

    avg_roi = total_pnl / total_bets if total_bets > 0 else 0
    print(f"  {'TOTAL':<14} {'':>8} {'':>8} {total_bets:>7,} {avg_roi:>+7.2%} £{total_pnl:>+8.0f}", flush=True)

    for alt_edge in [0.05, 0.08, 0.10, 0.12]:
        alt_key = f"edge_{int(alt_edge * 100)}"
        alt_bets = sum(r["value"].get(alt_key, {}).get("bets", 0) for r in all_results)
        alt_pnl = sum(r["value"].get(alt_key, {}).get("pnl", 0) for r in all_results)
        alt_roi = alt_pnl / alt_bets if alt_bets > 0 else 0
        print(f"  (alt edge>{alt_edge:.0%}: {alt_bets:,} bets, ROI={alt_roi:+.2%}, P&L=£{alt_pnl:+,.0f})", flush=True)

    meta = {
        "model": "catboost_flat_v2",
        "features": FLAT_V2_FEATURES,
        "min_edge": min_edge,
        "calibration": "raw_softmax",
        "windows": [{
            "label": r["label"],
            "top_pick": r["top_pick"],
            "brier": r["brier"],
            "log_loss": r["log_loss"],
            "best_iter": r["best_iter"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "value": r["value"],
        } for r in all_results],
        "total_bets": total_bets,
        "total_pnl": total_pnl,
        "total_roi": avg_roi,
    }
    out_path = ROOT / "experiments" / f"flat_v2_walkforward_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}", flush=True)

    return all_results


def run_single(min_edge):
    print(f"\n{'='*80}", flush=True)
    print(f"  FLAT V2 — CATBOOST (single split)", flush=True)
    print(f"{'='*80}", flush=True)

    result = run_single_window(
        "2015-01-01", TRAIN_END,
        CAL_START, CAL_END,
        TEST_START, TEST_END,
        "Primary split", min_edge,
    )

    if result:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_flat_v2"
        model_path = ROOT / "models" / f"{run_id}.cbm"
        result["model"].save_model(str(model_path))
        print(f"\n  Saved model: {model_path}", flush=True)

        meta = {
            "run_id": run_id,
            "model": "catboost_flat_v2",
            "features": result["features"],
            "calibration": "raw_softmax",
            "min_edge": min_edge,
            "top_pick": result["top_pick"],
            "brier": result["brier"],
            "log_loss": result["log_loss"],
            "best_iter": result["best_iter"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "value": result["value"],
        }
        meta_path = ROOT / "experiments" / f"{run_id}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Flat v2 model: CatBoost + raw softmax")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--min-edge", type=float, default=0.08)
    parser.add_argument("--also-lgbm", action="store_true", help="Also train LightGBM for comparison")
    args = parser.parse_args()

    if args.walk_forward:
        results = run_walk_forward(args.min_edge)

        if args.also_lgbm:
            print(f"\n\n{'='*80}", flush=True)
            print(f"  LIGHTGBM COMPARISON (same features, raw softmax, no isotonic)", flush=True)
            print(f"{'='*80}", flush=True)
            run_lgbm_comparison(args.min_edge)
    else:
        run_single(args.min_edge)


def run_lgbm_comparison(min_edge):
    import lightgbm as lgb

    all_results = []
    for w in WALK_FORWARD_WINDOWS:
        print(f"\n  --- {w['label']} ---", flush=True)

        X_train, y_train, g_train, df_train, features = load_flat_data("2015-01-01", w["train_end"])
        X_cal, y_cal, g_cal, df_cal, _ = load_flat_data(w["cal_start"], w["cal_end"])
        X_test, y_test, g_test, df_test, _ = load_flat_data(w["test_start"], w["test_end"])

        if len(X_test) == 0:
            continue

        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=3000, learning_rate=0.03,
            num_leaves=63, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(
            X_train, y_train, group=g_train,
            eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1],
            callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(0)],
        )

        test_ids = df_test["race_id"].to_numpy()
        probs = race_softmax(model.predict(X_test, num_iteration=model.best_iteration_), test_ids)
        tpwr = top_pick_win_rate(probs, test_ids, y_test)

        val_results, _ = value_analysis(df_test, probs, y_test, [min_edge])
        v = val_results.get(min_edge, {})
        roi = v.get("roi", 0)
        bets = v.get("bets", 0)
        pnl = v.get("pnl", 0)
        print(f"    LightGBM: TopPick={tpwr:.1%}, {bets} bets, ROI={roi:+.2%}, P&L=£{pnl:+,.0f}", flush=True)
        all_results.append({"label": w["label"], "bets": bets, "roi": roi, "pnl": pnl})

    total_bets = sum(r["bets"] for r in all_results)
    total_pnl = sum(r["pnl"] for r in all_results)
    avg_roi = total_pnl / total_bets if total_bets > 0 else 0
    print(f"\n  LightGBM total: {total_bets:,} bets, ROI={avg_roi:+.2%}, P&L=£{total_pnl:+,.0f}", flush=True)


if __name__ == "__main__":
    main()
