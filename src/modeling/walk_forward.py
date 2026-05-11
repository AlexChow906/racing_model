import json
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.modeling.train_split import load_data, race_softmax, renormalize, top_pick_win_rate, FLAT_DROP, JUMPS_DROP, EXCLUDE
from src.ingestion.db_connect import get_db

WINDOWS = [
    {"train_end": "2021-01-01", "cal_start": "2021-01-01", "cal_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2023-01-01", "label": "Test 2022"},
    {"train_end": "2022-01-01", "cal_start": "2022-01-01", "cal_end": "2023-01-01", "test_start": "2023-01-01", "test_end": "2024-01-01", "label": "Test 2023"},
    {"train_end": "2023-01-01", "cal_start": "2023-01-01", "cal_end": "2024-01-01", "test_start": "2024-01-01", "test_end": "2025-01-01", "label": "Test 2024"},
    {"train_end": "2024-01-01", "cal_start": "2024-01-01", "cal_end": "2025-01-01", "test_start": "2025-01-01", "test_end": "2027-01-01", "label": "Test 2025-26"},
]


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

    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=3000, learning_rate=0.01,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1, 3],
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(500)],
    )

    cal_ids = df_cal["race_id"].to_numpy()
    cal_probs = race_softmax(model.predict(X_cal, num_iteration=model.best_iteration_), cal_ids)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_probs, y_cal)

    test_ids = df_test["race_id"].to_numpy()
    test_scores = model.predict(X_test, num_iteration=model.best_iteration_)
    test_probs = renormalize(calibrator.transform(race_softmax(test_scores, test_ids)), test_ids)

    tpwr = top_pick_win_rate(test_probs, test_ids, y_test)
    brier = brier_score_loss(y_test, test_probs)

    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1").df()
    db.close()

    a = df_test[["race_id", "runner_id"]].copy()
    a["prob"] = test_probs
    a["target"] = y_test
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
                "bets": len(vb),
                "strike": float(vb["target"].mean()),
                "roi": float(vb["profit"].mean()),
                "pnl": float(vb["profit"].sum()),
            }

    return {
        "label": window["label"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_races": len(g_test),
        "top_pick": float(tpwr),
        "brier": float(brier),
        "best_iter": model.best_iteration_,
        "value": results,
    }


def main():
    all_results = {}

    for category in ["flat", "jumps"]:
        print(f"\n{'='*70}", flush=True)
        print(f"  {category.upper()} — WALK-FORWARD VALIDATION", flush=True)
        print(f"{'='*70}", flush=True)

        cat_results = []
        for w in WINDOWS:
            print(f"\n  --- {w['label']} (train to {w['train_end']}) ---", flush=True)
            result = run_window(category, w)
            if result is None:
                continue
            cat_results.append(result)

            print(f"  Train: {result['train_rows']:,} rows | Test: {result['test_rows']:,} rows, {result['test_races']:,} races", flush=True)
            print(f"  TopPick: {result['top_pick']:.1%} | Brier: {result['brier']:.5f} | Trees: {result['best_iter']}", flush=True)
            for key, v in result["value"].items():
                print(f"  {key}: {v['bets']:>5} bets, strike={v['strike']:.3f}, ROI={v['roi']:>+7.2%}, P&L=£{v['pnl']:>+7.0f}", flush=True)

        all_results[category] = cat_results

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"  WALK-FORWARD SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    for category in ["flat", "jumps"]:
        print(f"\n  {category.upper()}:", flush=True)
        print(f"  {'Window':<14} {'TopPick':>8} {'Brier':>7} {'Bets(5%)':>9} {'ROI(5%)':>8} {'P&L(5%)':>9}", flush=True)
        print(f"  {'-'*58}", flush=True)

        total_bets = 0
        total_pnl = 0.0

        for r in all_results[category]:
            v5 = r["value"].get("edge_5", {})
            bets = v5.get("bets", 0)
            roi = v5.get("roi", 0)
            pnl = v5.get("pnl", 0)
            total_bets += bets
            total_pnl += pnl
            print(f"  {r['label']:<14} {r['top_pick']:>7.1%} {r['brier']:>7.5f} {bets:>9,} {roi:>+7.2%} £{pnl:>+8.0f}", flush=True)

        avg_roi = total_pnl / total_bets if total_bets > 0 else 0
        print(f"  {'TOTAL':<14} {'':>8} {'':>7} {total_bets:>9,} {avg_roi:>+7.2%} £{total_pnl:>+8.0f}", flush=True)

    # Save
    with open("experiments/walk_forward_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: experiments/walk_forward_results.json", flush=True)


if __name__ == "__main__":
    main()
