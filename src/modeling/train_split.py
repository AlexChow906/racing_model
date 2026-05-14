import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.ingestion.db_connect import get_db
from src.constants.features import EXCLUDE, FLAT_DROP, JUMPS_DROP, FLAT_V2_FEATURES
from src.constants.params import CATBOOST_FLAT_PARAMS
from src.constants.windows import (
    TRAIN_END, CAL_START, CAL_END, TEST_START, TEST_END,
    WALK_FORWARD_WINDOWS,
)


def load_data(start, end, race_category):
    db = get_db("racing.duckdb")
    if race_category == "flat":
        type_filter = "AND ra.race_type = 'Flat'"
    else:
        type_filter = "AND ra.race_type IN ('Chase', 'Hurdle', 'NH Flat')"

    # Check if horse_history has non_completion column
    has_nc = False
    try:
        db.execute("SELECT non_completion FROM horse_history LIMIT 1")
        has_nc = True
    except Exception:
        pass

    if has_nc:
        nc_join = """
        LEFT JOIN (
            SELECT ru.runner_id,
                CASE WHEN COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat')) > 0
                    THEN 1.0 - COUNT(*) FILTER (WHERE hh.non_completion IS NOT NULL)::DOUBLE
                        / COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat'))
                    ELSE NULL END as horse_completion_rate,
                CASE WHEN COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat')) > 0
                    THEN COUNT(*) FILTER (WHERE hh.non_completion IN ('F','UR','BD','SU'))::DOUBLE
                        / COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat'))
                    ELSE NULL END as horse_fall_rate,
                CASE WHEN COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat')) > 0
                    THEN COUNT(*) FILTER (WHERE hh.non_completion = 'PU')::DOUBLE
                        / COUNT(*) FILTER (WHERE ra2.race_type IN ('Chase','Hurdle','NH Flat'))
                    ELSE NULL END as horse_pu_rate,
                COUNT(*) FILTER (WHERE hh.non_completion IS NOT NULL
                    AND hh.rn_desc <= 5) as horse_nc_last_5
            FROM runners ru
            JOIN races ra3 ON ra3.race_id = ru.race_id
            JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY horse_id ORDER BY scheduled_off_utc DESC) as rn_desc
                FROM horse_history
            ) hh ON hh.horse_id = ru.horse_id AND hh.scheduled_off_utc < ra3.decision_cutoff_utc
            JOIN races ra2 ON ra2.race_id = hh.race_id
            GROUP BY 1
        ) nc ON nc.runner_id = fs.runner_id"""
        nc_cols = "nc.horse_completion_rate, nc.horse_fall_rate, nc.horse_pu_rate, nc.horse_nc_last_5,"
    else:
        nc_join = ""
        nc_cols = ""

    df = db.execute(f"""
        SELECT fs.*,
            ra.race_type,
            {nc_cols}
            1 as _dummy
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        {nc_join}
        WHERE fs.race_date >= '{start}' AND fs.race_date < '{end}'
        AND fs.target IS NOT NULL
        {type_filter}
        ORDER BY fs.race_date, fs.race_id
    """).df()
    df = df.drop(columns=["_dummy"], errors="ignore")

    df = df[df["race_date"] >= "2015-01-01"].copy()
    df = df.drop(columns=["race_type"], errors="ignore")

    groups = df.groupby("race_id", sort=False)["runner_id"].count().values
    y = df["target"].astype(int).values

    drop_list = FLAT_DROP if race_category == "flat" else JUMPS_DROP
    drop_cols = [c for c in EXCLUDE + drop_list if c in df.columns]
    X = df.drop(columns=drop_cols)

    return X, y, groups, df


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
        out[start:end] = chunk / chunk.sum()
        start = end
    return out


def top_pick_win_rate(probs, race_ids, y_true):
    frame = pd.DataFrame({"race_id": race_ids, "prob": probs, "target": y_true})
    top = frame.sort_values(["race_id", "prob"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    return float(top["target"].mean())


def calibration_table(probs, y_true, bins=10):
    frame = pd.DataFrame({"pred": probs, "target": y_true})
    frame["bin"] = pd.qcut(frame["pred"], q=bins, duplicates="drop")
    return frame.groupby("bin", observed=False).agg(
        count=("target", "size"), avg_pred=("pred", "mean"), empirical=("target", "mean")
    ).reset_index()


def train_model(category, train_start="2015-01-01"):
    print(f"\n{'='*60}", flush=True)
    print(f"  {category.upper()} MODEL (3-way split)", flush=True)
    print(f"  Train: {train_start} to {TRAIN_END}  (base model)", flush=True)
    print(f"  Cal:   {CAL_START} to {CAL_END}  (early stopping + calibration)", flush=True)
    print(f"  Test:  {TEST_START} to {TEST_END}  (final evaluation)", flush=True)
    print(f"{'='*60}", flush=True)

    X_train, y_train, g_train, df_train = load_data(train_start, TRAIN_END, category)
    X_cal, y_cal, g_cal, df_cal = load_data(CAL_START, CAL_END, category)
    X_test, y_test, g_test, df_test = load_data(TEST_START, TEST_END, category)

    non_numeric = [c for c in X_train.columns
                   if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
    for col in non_numeric:
        cats = pd.Index(pd.concat([X_train[col], X_cal[col], X_test[col]]).astype(str).astype("category").cat.categories)
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
        X_cal[col] = pd.Categorical(X_cal[col].astype(str), categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

    feature_names = X_train.columns.tolist()
    print(f"Train: {len(X_train):,} rows, {len(g_train):,} races, {len(feature_names)} features", flush=True)
    print(f"Cal:   {len(X_cal):,} rows, {len(g_cal):,} races", flush=True)
    print(f"Test:  {len(X_test):,} rows, {len(g_test):,} races", flush=True)

    # Train base model, early-stop on CAL set
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1, 3],
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)],
    )

    # Calibrate on CAL set (out-of-sample for base model)
    cal_race_ids = df_cal["race_id"].to_numpy()
    cal_scores = model.predict(X_cal, num_iteration=model.best_iteration_)
    cal_probs_raw = race_softmax(cal_scores, cal_race_ids)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_probs_raw, y_cal)

    # Evaluate on TEST set (fully out-of-sample for both model and calibrator)
    test_race_ids = df_test["race_id"].to_numpy()
    test_scores = model.predict(X_test, num_iteration=model.best_iteration_)
    test_probs_raw = race_softmax(test_scores, test_race_ids)
    test_probs_calibrated = calibrator.transform(test_probs_raw)
    test_probs = renormalize(test_probs_calibrated, test_race_ids)

    brier_raw = brier_score_loss(y_test, test_probs_raw)
    brier = brier_score_loss(y_test, test_probs)
    ll_raw = log_loss(y_test, np.clip(test_probs_raw, 1e-15, 1 - 1e-15), labels=[0, 1])
    ll = log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15), labels=[0, 1])
    tpwr_raw = top_pick_win_rate(test_probs_raw, test_race_ids, y_test)
    tpwr = top_pick_win_rate(test_probs, test_race_ids, y_test)

    evals = model.evals_result_
    ndcg1 = evals.get("valid_0", {}).get("ndcg@1", [None])[-1]

    # Value analysis on TEST
    db2 = get_db("racing.duckdb")
    sp_df = db2.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal IS NOT NULL AND sp_decimal > 1").df()
    test_analysis = df_test[["race_id", "runner_id"]].copy()
    test_analysis["prob"] = test_probs
    test_analysis["target"] = y_test
    test_analysis = test_analysis.merge(sp_df, on="runner_id", how="left")
    test_analysis["implied"] = 1.0 / test_analysis["sp_decimal"]
    test_analysis["edge"] = test_analysis["prob"] - test_analysis["implied"]

    for threshold in [0.03, 0.05, 0.10]:
        vb = test_analysis[(test_analysis["edge"] > threshold) & (test_analysis["sp_decimal"].notna())]
        roi = 0.0
        strike = 0.0
        if len(vb) > 0:
            roi = (vb["target"] * (vb["sp_decimal"] - 1) - (1 - vb["target"])).mean()
            strike = vb["target"].mean()
        print(f"  Value edge>{threshold:.0%}: {len(vb):>6,} bets, strike={strike:.3f}, avgSP={vb['sp_decimal'].mean() if len(vb)>0 else 0:.1f}, ROI={roi:+.2%}", flush=True)

    print(f"\nResults ({category}) on TEST set 2024:", flush=True)
    print(f"  Best iteration: {model.best_iteration_}", flush=True)
    print(f"  NDCG@1 (cal set): {ndcg1:.4f}", flush=True)
    print(f"  Before calibration:  Brier={brier_raw:.5f}  LogLoss={ll_raw:.5f}  TopPick={tpwr_raw:.4f}", flush=True)
    print(f"  After calibration:   Brier={brier:.5f}  LogLoss={ll:.5f}  TopPick={tpwr:.4f}", flush=True)

    cal_table = calibration_table(test_probs, y_test)
    print(f"\nCalibration (TEST set):", flush=True)
    print(cal_table.to_string(index=False), flush=True)

    importance = model.feature_importances_
    fi = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    print(f"\nTop 15 features:", flush=True)
    for feat, imp in fi[:15]:
        print(f"  {feat:<35} {imp}", flush=True)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{category}_v3"
    model.booster_.save_model(f"models/{run_id}.lgbm")
    with open(f"models/{run_id}_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    meta = {
        "run_id": run_id, "category": category,
        "split": {"train": [train_start, TRAIN_END], "cal": [CAL_START, CAL_END], "test": [TEST_START, TEST_END]},
        "features": feature_names, "n_train": len(X_train), "n_cal": len(X_cal), "n_test": len(X_test),
        "best_iter": model.best_iteration_,
        "test_brier_raw": float(brier_raw), "test_brier": float(brier),
        "test_ll_raw": float(ll_raw), "test_ll": float(ll),
        "test_top_pick_raw": float(tpwr_raw), "test_top_pick": float(tpwr),
    }
    with open(f"experiments/{run_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: models/{run_id}.lgbm", flush=True)

    run_value_analysis(df_test, test_probs, y_test, category)

    return meta


def run_value_analysis(df_test, test_probs, y_test, category):
    """Run value betting analysis — no filters for flat, smart filters for jumps."""
    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal IS NOT NULL AND sp_decimal > 1").df()
    race_info = db.execute("SELECT race_id, is_handicap, race_class, field_size, going_code, race_type FROM races").df()
    cr = db.execute("SELECT runner_id, career_runs FROM feature_store").df()
    db.close()

    analysis = df_test[["race_id", "runner_id"]].copy()
    analysis["prob"] = test_probs
    analysis["target"] = y_test
    analysis = analysis.merge(sp_df, on="runner_id", how="left")
    analysis = analysis.merge(race_info, on="race_id", how="left")
    analysis = analysis.merge(cr, on="runner_id", how="left")
    analysis["implied"] = 1.0 / analysis["sp_decimal"]
    analysis["edge"] = analysis["prob"] - analysis["implied"]
    analysis["profit"] = analysis["target"] * (analysis["sp_decimal"] - 1) - (1 - analysis["target"])

    has_sp = analysis[analysis["sp_decimal"].notna()].copy()

    print(f"\n  {'='*70}", flush=True)
    print(f"  {category.upper()} VALUE BETTING ANALYSIS", flush=True)
    print(f"  {'='*70}", flush=True)

    # Unfiltered baseline
    for thresh in [0.03, 0.05, 0.10]:
        vb = has_sp[has_sp["edge"] > thresh]
        if len(vb) == 0: continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        print(f"  Unfiltered edge>{thresh:.0%}: {len(vb):>5} bets, strike={strike:.3f}, ROI={roi:>+7.2%}", flush=True)

    if category == "flat":
        # Flat: no filters, just edge threshold
        best = has_sp[has_sp["edge"] > 0.05]
        best_label = "FLAT — unfiltered, edge>5%"
    else:
        # Jumps: filter out novice 1-3 career runs at long odds, and outsiders 50+
        jumps_filter = (
            (has_sp["edge"] > 0.05) &
            ~((has_sp["career_runs"] >= 1) & (has_sp["career_runs"] <= 3) & (has_sp["sp_decimal"] > 20)) &
            (has_sp["sp_decimal"] < 50)
        )
        best = has_sp[jumps_filter]
        best_label = "JUMPS — no novice longshots, no outsiders 50+, edge>5%"

        # Also show progressive filters for jumps
        filters = {
            "No outsiders (SP<50)": (has_sp["edge"]>0.05) & (has_sp["sp_decimal"]<50),
            "+ No novice longshots": (has_sp["edge"]>0.05) & (has_sp["sp_decimal"]<50) & ~((has_sp["career_runs"]>=1) & (has_sp["career_runs"]<=3) & (has_sp["sp_decimal"]>20)),
            "+ Good going only": (has_sp["edge"]>0.05) & (has_sp["sp_decimal"]<50) & ~((has_sp["career_runs"]>=1) & (has_sp["career_runs"]<=3) & (has_sp["sp_decimal"]>20)) & (has_sp["going_code"].isin(["Good","Good To Firm","Yielding","Standard"])),
        }
        print(f"\n  --- Jumps progressive filters ---", flush=True)
        print(f"  {'Filter':<40} {'Bets':>6} {'Strike':>8} {'AvgSP':>7} {'ROI':>8}", flush=True)
        print(f"  {'-'*72}", flush=True)
        for label, mask in filters.items():
            vb = has_sp[mask]
            if len(vb) == 0: continue
            roi = vb["profit"].mean()
            strike = vb["target"].mean()
            avg_sp = vb["sp_decimal"].mean()
            print(f"  {label:<40} {len(vb):>6} {strike:>7.3f} {avg_sp:>7.1f} {roi:>+7.2%}", flush=True)

    if len(best) > 0:
        roi = best["profit"].mean()
        strike = best["target"].mean()
        total_profit = best["profit"].sum()
        avg_sp = best["sp_decimal"].mean()

        print(f"\n  --- {best_label} ---", flush=True)
        print(f"  Bets: {len(best):,}", flush=True)
        print(f"  Strike rate: {strike:.1%}", flush=True)
        print(f"  Avg SP: {avg_sp:.1f}", flush=True)
        print(f"  ROI per bet: {roi:+.2%}", flush=True)
        print(f"  Total profit (to £1 stakes): £{total_profit:+,.0f}", flush=True)

        best_with_date = best.merge(df_test[["runner_id", "race_date"]].drop_duplicates("runner_id"), on="runner_id", how="left")
        if "race_date" in best_with_date.columns:
            best_with_date["month"] = pd.to_datetime(best_with_date["race_date"]).dt.to_period("M")
            monthly = best_with_date.groupby("month").agg(
                bets=("profit", "size"), wins=("target", "sum"), profit=("profit", "sum"),
            ).reset_index()
            monthly["roi"] = monthly["profit"] / monthly["bets"]
            monthly["cum_profit"] = monthly["profit"].cumsum()

            print(f"\n  --- MONTHLY P&L (£1 stakes) ---", flush=True)
            print(f"  {'Month':<10} {'Bets':>5} {'Wins':>5} {'P&L':>8} {'ROI':>7} {'Cumulative':>11}", flush=True)
            for _, row in monthly.iterrows():
                print(f"  {str(row['month']):<10} {row['bets']:>5} {int(row['wins']):>5} £{row['profit']:>+7.0f} {row['roi']:>+6.1%} £{row['cum_profit']:>+10.0f}", flush=True)

    return analysis


def load_flat_v2_data(start, end, handicap_filter=None, feature_list=None):
    db = get_db("racing.duckdb")

    if handicap_filter == "handicap":
        hcap_sql = "AND ra.is_handicap = TRUE"
    elif handicap_filter == "nonhandicap":
        hcap_sql = "AND (ra.is_handicap = FALSE OR ra.is_handicap IS NULL)"
    else:
        hcap_sql = ""

    df = db.execute(f"""
        SELECT fs.*,
            ra.race_type,
            1 as _dummy
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        WHERE fs.race_date >= '{start}' AND fs.race_date < '{end}'
        AND fs.target IS NOT NULL
        AND ra.race_type = 'Flat'
        {hcap_sql}
        ORDER BY fs.race_date, fs.race_id
    """).df()
    db.close()
    df = df.drop(columns=["_dummy"], errors="ignore")
    df = df[df["race_date"] >= "2015-01-01"].copy()
    df = df.drop(columns=["race_type"], errors="ignore")

    groups = df.groupby("race_id", sort=False)["runner_id"].count().values
    y = df["target"].astype(int).values

    feat_source = feature_list if feature_list else FLAT_V2_FEATURES
    available = [f for f in feat_source if f in df.columns]
    missing = [f for f in feat_source if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features (will skip): {missing}", flush=True)

    X = df[available].copy()

    return X, y, groups, df, available


def train_flat_v2(train_start="2015-01-01", use_catboost=True):
    print(f"\n{'='*60}", flush=True)
    print(f"  FLAT V2 MODEL ({'CatBoost' if use_catboost else 'LightGBM'}, no isotonic cal)", flush=True)
    print(f"  Train: {train_start} to {TRAIN_END}", flush=True)
    print(f"  Cal:   {CAL_START} to {CAL_END}  (early stopping only)", flush=True)
    print(f"  Test:  {TEST_START} to {TEST_END}", flush=True)
    print(f"{'='*60}", flush=True)

    X_train, y_train, g_train, df_train, feat_names = load_flat_v2_data(train_start, TRAIN_END)
    X_cal, y_cal, g_cal, df_cal, _ = load_flat_v2_data(CAL_START, CAL_END)
    X_test, y_test, g_test, df_test, _ = load_flat_v2_data(TEST_START, TEST_END)

    for extra_df in [X_cal, X_test]:
        for col in feat_names:
            if col not in extra_df.columns:
                extra_df[col] = np.nan
        drop_extra = [c for c in extra_df.columns if c not in feat_names]
        extra_df.drop(columns=drop_extra, inplace=True, errors="ignore")
    X_cal = X_cal[feat_names]
    X_test = X_test[feat_names]

    print(f"  Features ({len(feat_names)}): {feat_names}", flush=True)
    print(f"  Train: {len(X_train):,} rows, {len(g_train):,} races", flush=True)
    print(f"  Cal:   {len(X_cal):,} rows, {len(g_cal):,} races", flush=True)
    print(f"  Test:  {len(X_test):,} rows, {len(g_test):,} races", flush=True)

    if use_catboost:
        from catboost import CatBoost, Pool
        train_pool = Pool(X_train, label=y_train, group_id=np.repeat(
            np.arange(len(g_train)), g_train))
        cal_pool = Pool(X_cal, label=y_cal, group_id=np.repeat(
            np.arange(len(g_cal)), g_cal))

        model = CatBoost(CATBOOST_FLAT_PARAMS)
        model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=200)
        best_iter = model.best_iteration_

        cal_scores = model.predict(X_cal)
        test_scores = model.predict(X_test)
    else:
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=50,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train, group=g_train,
            eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1, 3],
            callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)],
        )
        best_iter = model.best_iteration_
        cal_scores = model.predict(X_cal, num_iteration=best_iter)
        test_scores = model.predict(X_test, num_iteration=best_iter)

    cal_ids = df_cal["race_id"].to_numpy()
    test_ids = df_test["race_id"].to_numpy()
    test_probs = race_softmax(test_scores, test_ids)

    brier = brier_score_loss(y_test, test_probs)
    ll = log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15), labels=[0, 1])
    tpwr = top_pick_win_rate(test_probs, test_ids, y_test)

    print(f"\n  Best iteration: {best_iter}", flush=True)
    print(f"  Brier={brier:.5f}  LogLoss={ll:.5f}  TopPick={tpwr:.1%}", flush=True)

    cal_table = calibration_table(test_probs, y_test)
    print(f"\n  Calibration (raw softmax on TEST):", flush=True)
    print(cal_table.to_string(index=False), flush=True)

    db2 = get_db("racing.duckdb")
    sp_df = db2.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal IS NOT NULL AND sp_decimal > 1").df()
    race_info = db2.execute("SELECT race_id, is_handicap, race_class FROM races").df()
    db2.close()

    analysis = df_test[["race_id", "runner_id", "race_date"]].copy()
    analysis["prob"] = test_probs
    analysis["target"] = y_test
    analysis = analysis.merge(sp_df, on="runner_id", how="left")
    analysis = analysis.merge(race_info, on="race_id", how="left")
    analysis["implied"] = 1.0 / analysis["sp_decimal"]
    analysis["edge"] = analysis["prob"] - analysis["implied"]
    analysis["profit"] = analysis["target"] * (analysis["sp_decimal"] - 1) - (1 - analysis["target"])

    has_sp = analysis[analysis["sp_decimal"].notna()].copy()

    print(f"\n  {'='*70}", flush=True)
    print(f"  FLAT V2 VALUE ANALYSIS", flush=True)
    print(f"  {'='*70}", flush=True)

    for thresh in [0.03, 0.05, 0.08, 0.10, 0.12]:
        vb = has_sp[has_sp["edge"] > thresh]
        if len(vb) == 0:
            continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        avg_sp = vb["sp_decimal"].mean()
        print(f"  Edge>{thresh:.0%}: {len(vb):>5} bets, strike={strike:.3f}, avgSP={avg_sp:.1f}, ROI={roi:>+7.2%}", flush=True)

    for sp_lo, sp_hi, label in [(1, 6, "SP 1-6"), (6, 15, "SP 6-15"), (15, 50, "SP 15-50")]:
        for thresh in [0.05, 0.08, 0.10]:
            vb = has_sp[(has_sp["edge"] > thresh) & (has_sp["sp_decimal"] >= sp_lo) & (has_sp["sp_decimal"] < sp_hi)]
            if len(vb) == 0:
                continue
            roi = vb["profit"].mean()
            strike = vb["target"].mean()
            print(f"  {label} edge>{thresh:.0%}: {len(vb):>5} bets, strike={strike:.3f}, ROI={roi:>+7.2%}", flush=True)

    for is_h, label in [(True, "Handicap"), (False, "Non-handicap")]:
        for thresh in [0.05, 0.08, 0.10]:
            vb = has_sp[(has_sp["edge"] > thresh) & (has_sp["is_handicap"] == is_h)]
            if len(vb) == 0:
                continue
            roi = vb["profit"].mean()
            strike = vb["target"].mean()
            print(f"  {label} edge>{thresh:.0%}: {len(vb):>5} bets, strike={strike:.3f}, ROI={roi:>+7.2%}", flush=True)

    if use_catboost:
        fi = sorted(zip(feat_names, model.get_feature_importance(type="PredictionValuesChange")), key=lambda x: x[1], reverse=True)
    else:
        fi = sorted(zip(feat_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importance:", flush=True)
    for feat, imp in fi:
        print(f"    {feat:<35} {imp:.1f}", flush=True)

    engine = "catboost" if use_catboost else "lgbm"
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_flat_v2_{engine}"
    if use_catboost:
        model.save_model(f"models/{run_id}.cbm")
        Path("models/tuned/flat").mkdir(parents=True, exist_ok=True)
        model.save_model("models/tuned/flat/model.cbm")
    else:
        model.booster_.save_model(f"models/{run_id}.lgbm")

    meta = {
        "run_id": run_id, "category": "flat_v2", "engine": engine,
        "calibration": "none",
        "features": feat_names,
        "split": {"train": [train_start, TRAIN_END], "cal": [CAL_START, CAL_END], "test": [TEST_START, TEST_END]},
        "n_train": len(X_train), "n_cal": len(X_cal), "n_test": len(X_test),
        "best_iter": int(best_iter),
        "test_brier": float(brier), "test_ll": float(ll), "test_top_pick": float(tpwr),
    }
    with open(f"experiments/{run_id}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved: models/tuned/flat/model.cbm (production)", flush=True)
    print(f"  Saved: models/{run_id}.{'cbm' if use_catboost else 'lgbm'} (archive)", flush=True)

    return meta


def walk_forward_flat_v2(use_catboost=True):
    print(f"\n{'='*70}", flush=True)
    print(f"  FLAT V2 WALK-FORWARD ({'CatBoost' if use_catboost else 'LightGBM'}, no isotonic cal)", flush=True)
    print(f"{'='*70}", flush=True)

    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal IS NOT NULL AND sp_decimal > 1").df()
    race_info = db.execute("SELECT race_id, is_handicap, race_class FROM races").df()
    db.close()

    all_window_results = []

    for w in WALK_FORWARD_WINDOWS:
        print(f"\n  --- {w['label']} ---", flush=True)

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

        a = df_test[["race_id", "runner_id", "race_date"]].copy()
        a["prob"] = test_probs
        a["target"] = y_test
        a = a.merge(sp_df, on="runner_id", how="left")
        a = a.merge(race_info, on="race_id", how="left")
        a["implied"] = np.where(a["sp_decimal"] > 1, 1.0 / a["sp_decimal"], np.nan)
        a["edge"] = a["prob"] - a["implied"]
        a["profit"] = a["target"] * (a["sp_decimal"] - 1) - (1 - a["target"])
        a["window"] = w["label"]

        has_sp = a[a["sp_decimal"].notna()].copy()

        for thresh in [0.05, 0.08, 0.10]:
            vb = has_sp[has_sp["edge"] > thresh]
            if len(vb) == 0:
                continue
            roi = vb["profit"].mean()
            strike = vb["target"].mean()
            avg_sp = vb["sp_decimal"].mean()
            print(f"    edge>{thresh:.0%}: {len(vb):>5} bets, strike={strike:.3f}, avgSP={avg_sp:.1f}, ROI={roi:>+7.2%}", flush=True)

        all_window_results.append(has_sp)

    if not all_window_results:
        print("  No data across walk-forward windows", flush=True)
        return

    combined = pd.concat(all_window_results)

    print(f"\n  {'='*70}", flush=True)
    print(f"  COMBINED WALK-FORWARD RESULTS", flush=True)
    print(f"  {'='*70}", flush=True)

    for thresh in [0.05, 0.08, 0.10, 0.12]:
        vb = combined[combined["edge"] > thresh]
        if len(vb) == 0:
            continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        avg_sp = vb["sp_decimal"].mean()
        total_pnl = vb["profit"].sum()
        print(f"  edge>{thresh:.0%}: {len(vb):>6} bets, strike={strike:.3f}, avgSP={avg_sp:.1f}, ROI={roi:>+7.2%}, P&L=£{total_pnl:>+,.0f}", flush=True)

    print(f"\n  Per-window breakdown (edge>8%):", flush=True)
    print(f"  {'Window':<15} {'Bets':>6} {'Strike':>8} {'AvgSP':>7} {'ROI':>8} {'P&L':>10}", flush=True)
    print(f"  {'-'*58}", flush=True)
    for window in [w["label"] for w in WALK_FORWARD_WINDOWS]:
        vb = combined[(combined["window"] == window) & (combined["edge"] > 0.08)]
        if len(vb) == 0:
            continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        avg_sp = vb["sp_decimal"].mean()
        pnl = vb["profit"].sum()
        print(f"  {window:<15} {len(vb):>6} {strike:>7.3f} {avg_sp:>7.1f} {roi:>+7.2%} £{pnl:>+9.0f}", flush=True)

    for sp_lo, sp_hi, label in [(1, 6, "SP 1-6"), (6, 15, "SP 6-15"), (15, 50, "SP 15-50")]:
        vb = combined[(combined["edge"] > 0.08) & (combined["sp_decimal"] >= sp_lo) & (combined["sp_decimal"] < sp_hi)]
        if len(vb) == 0:
            continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        print(f"  {label} edge>8%: {len(vb):>5} bets, strike={strike:.3f}, ROI={roi:>+7.2%}", flush=True)

    for is_h, label in [(True, "Handicap"), (False, "Non-handicap")]:
        vb = combined[(combined["edge"] > 0.08) & (combined["is_handicap"] == is_h)]
        if len(vb) == 0:
            continue
        roi = vb["profit"].mean()
        strike = vb["target"].mean()
        print(f"  {label} edge>8%: {len(vb):>5} bets, strike={strike:.3f}, ROI={roi:>+7.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train racing models")
    parser.add_argument("--flat-v2", action="store_true", help="Train flat v2 (CatBoost, no calibration)")
    parser.add_argument("--flat-v2-lgbm", action="store_true", help="Train flat v2 with LightGBM for comparison")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation instead of single split")
    parser.add_argument("--original", action="store_true", help="Train original flat+jumps models")
    args = parser.parse_args()

    if args.flat_v2 or args.flat_v2_lgbm:
        use_catboost = args.flat_v2
        if args.walk_forward:
            walk_forward_flat_v2(use_catboost=use_catboost)
        else:
            train_flat_v2(use_catboost=use_catboost)
        return

    if args.original or not any([args.flat_v2, args.flat_v2_lgbm]):
        results = {}
        for category in ["flat", "jumps"]:
            meta = train_model(category)
            results[category] = meta

        print(f"\n{'='*70}", flush=True)
        print(f"  FINAL SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        for cat, m in results.items():
            print(f"  {cat.upper()}: TopPick={m['test_top_pick']:.1%}  Brier={m['test_brier']:.5f}  LogLoss={m['test_ll']:.5f}", flush=True)


if __name__ == "__main__":
    main()
