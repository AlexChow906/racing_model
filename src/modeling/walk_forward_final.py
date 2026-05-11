import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.modeling.train_split import load_data, race_softmax, renormalize, top_pick_win_rate
from src.ingestion.db_connect import get_db

WINDOWS = [
    {"train_end": "2021-01-01", "cal_start": "2021-01-01", "cal_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2023-01-01", "label": "Test 2022"},
    {"train_end": "2022-01-01", "cal_start": "2022-01-01", "cal_end": "2023-01-01", "test_start": "2023-01-01", "test_end": "2024-01-01", "label": "Test 2023"},
    {"train_end": "2023-01-01", "cal_start": "2023-01-01", "cal_end": "2024-01-01", "test_start": "2024-01-01", "test_end": "2025-01-01", "label": "Test 2024"},
    {"train_end": "2024-01-01", "cal_start": "2024-01-01", "cal_end": "2025-01-01", "test_start": "2025-01-01", "test_end": "2027-01-01", "label": "Test 2025-26"},
]

PARAMS = {
    "flat": {
        "objective": "lambdarank", "n_estimators": 3000,
        "learning_rate": 0.04369, "num_leaves": 57, "min_child_samples": 69,
        "subsample": 0.8805, "colsample_bytree": 0.9574,
        "reg_alpha": 2.935e-06, "reg_lambda": 0.001326,
        "max_depth": 7, "min_split_gain": 0.02777,
        "random_state": 42, "n_jobs": -1,
    },
    "jumps": {
        "objective": "lambdarank", "n_estimators": 3000,
        "learning_rate": 0.04460, "num_leaves": 100, "min_child_samples": 63,
        "subsample": 0.7438, "colsample_bytree": 0.5903,
        "reg_alpha": 0.2719, "reg_lambda": 0.004538,
        "max_depth": 9, "min_split_gain": 0.06107,
        "random_state": 42, "n_jobs": -1,
    },
}


def run_window(category, window):
    X_train, y_train, g_train, df_train = load_data("2015-01-01", window["train_end"], category)
    X_cal, y_cal, g_cal, df_cal = load_data(window["cal_start"], window["cal_end"], category)
    X_test, y_test, g_test, df_test = load_data(window["test_start"], window["test_end"], category)

    if len(X_test) == 0:
        return None

    non_numeric = [c for c in X_train.columns
                   if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
    for col in non_numeric:
        cats = pd.Index(pd.concat([X_train[col], X_cal[col], X_test[col]]).astype(str).astype("category").cat.categories)
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
        X_cal[col] = pd.Categorical(X_cal[col].astype(str), categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

    model = lgb.LGBMRanker(**PARAMS[category])
    model.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1],
        callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(0)],
    )

    cal_ids = df_cal["race_id"].to_numpy()
    cal_probs = race_softmax(model.predict(X_cal, num_iteration=model.best_iteration_), cal_ids)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_probs, y_cal)

    test_ids = df_test["race_id"].to_numpy()
    calibrated = calibrator.transform(race_softmax(model.predict(X_test, num_iteration=model.best_iteration_), test_ids))
    calibrated = np.nan_to_num(calibrated, nan=1e-6)
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    test_probs = renormalize(calibrated, test_ids)
    test_probs = np.nan_to_num(test_probs, nan=1e-6)

    tpwr = top_pick_win_rate(test_probs, test_ids, y_test)
    brier = brier_score_loss(y_test, np.clip(test_probs, 0, 1))

    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1").df()
    db.close()

    a = df_test[["race_id", "runner_id", "race_date"]].copy()
    a["prob"] = test_probs; a["target"] = y_test
    a = a.merge(sp_df, on="runner_id", how="left")
    a["implied"] = 1.0 / a["sp_decimal"]
    a["edge"] = a["prob"] - a["implied"]
    a["profit"] = a["target"] * (a["sp_decimal"] - 1) - (1 - a["target"])
    has_sp = a[a["sp_decimal"].notna()]

    results = {}
    for thresh in [0.03, 0.05, 0.10]:
        vb = has_sp[has_sp["edge"] > thresh]
        if len(vb) > 0:
            results[f"edge_{int(thresh*100)}"] = {
                "bets": len(vb), "strike": float(vb["target"].mean()),
                "roi": float(vb["profit"].mean()), "pnl": float(vb["profit"].sum()),
            }

    return {
        "label": window["label"],
        "train_rows": len(X_train), "test_rows": len(X_test), "test_races": len(g_test),
        "top_pick": float(tpwr), "brier": float(brier), "best_iter": model.best_iteration_,
        "value": results,
    }


def main():
    all_results = {}

    for category in ["flat", "jumps"]:
        print(f"\n{'='*70}", flush=True)
        print(f"  {category.upper()} — FINAL WALK-FORWARD (TUNED PARAMS)", flush=True)
        print(f"{'='*70}", flush=True)

        cat_results = []
        for w in WINDOWS:
            print(f"  Training {w['label']}...", end="", flush=True)
            result = run_window(category, w)
            if result is None:
                print(" skipped", flush=True)
                continue
            cat_results.append(result)
            v5 = result["value"].get("edge_5", {})
            print(f" TopPick={result['top_pick']:.1%}  ROI(5%)={v5.get('roi',0):+.2%}  P&L=£{v5.get('pnl',0):+,.0f}  ({result['best_iter']} trees)", flush=True)

        all_results[category] = cat_results

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL RESULTS — TUNED PARAMS", flush=True)
    print(f"{'='*70}", flush=True)

    grand_total_bets = 0
    grand_total_pnl = 0.0

    for category in ["flat", "jumps"]:
        print(f"\n  {category.upper()}:", flush=True)
        print(f"  {'Window':<14} {'TopPick':>8} {'Brier':>7}  {'Bets(3%)':>9} {'ROI(3%)':>8} {'P&L(3%)':>9}  {'Bets(5%)':>9} {'ROI(5%)':>8} {'P&L(5%)':>9}  {'Bets(10%)':>10} {'ROI(10%)':>9} {'P&L(10%)':>10}", flush=True)
        print(f"  {'-'*120}", flush=True)

        total = {"bets3": 0, "pnl3": 0, "bets5": 0, "pnl5": 0, "bets10": 0, "pnl10": 0}

        for r in all_results[category]:
            v3 = r["value"].get("edge_3", {})
            v5 = r["value"].get("edge_5", {})
            v10 = r["value"].get("edge_10", {})
            total["bets3"] += v3.get("bets", 0); total["pnl3"] += v3.get("pnl", 0)
            total["bets5"] += v5.get("bets", 0); total["pnl5"] += v5.get("pnl", 0)
            total["bets10"] += v10.get("bets", 0); total["pnl10"] += v10.get("pnl", 0)

            print(f"  {r['label']:<14} {r['top_pick']:>7.1%} {r['brier']:>7.5f}"
                  f"  {v3.get('bets',0):>9,} {v3.get('roi',0):>+7.2%} £{v3.get('pnl',0):>+8.0f}"
                  f"  {v5.get('bets',0):>9,} {v5.get('roi',0):>+7.2%} £{v5.get('pnl',0):>+8.0f}"
                  f"  {v10.get('bets',0):>10,} {v10.get('roi',0):>+7.2%} £{v10.get('pnl',0):>+9.0f}", flush=True)

        r3 = total["pnl3"]/total["bets3"] if total["bets3"] else 0
        r5 = total["pnl5"]/total["bets5"] if total["bets5"] else 0
        r10 = total["pnl10"]/total["bets10"] if total["bets10"] else 0
        print(f"  {'TOTAL':<14} {'':>8} {'':>7}"
              f"  {total['bets3']:>9,} {r3:>+7.2%} £{total['pnl3']:>+8.0f}"
              f"  {total['bets5']:>9,} {r5:>+7.2%} £{total['pnl5']:>+8.0f}"
              f"  {total['bets10']:>10,} {r10:>+7.2%} £{total['pnl10']:>+9.0f}", flush=True)

        grand_total_bets += total["bets5"]
        grand_total_pnl += total["pnl5"]

    print(f"\n  {'='*70}", flush=True)
    grand_roi = grand_total_pnl / grand_total_bets if grand_total_bets else 0
    print(f"  GRAND TOTAL (edge>5%): {grand_total_bets:,} bets, ROI={grand_roi:+.2%}, P&L=£{grand_total_pnl:+,.0f}", flush=True)

    with open("experiments/walk_forward_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: experiments/walk_forward_final.json", flush=True)


if __name__ == "__main__":
    main()
