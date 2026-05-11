import json
import lightgbm as lgb
import numpy as np
import pandas as pd
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

TUNED_PARAMS = {
    "flat": {
        "objective": "lambdarank",
        "n_estimators": 3000,
        "learning_rate": 0.04369,
        "num_leaves": 57,
        "min_child_samples": 69,
        "subsample": 0.8805,
        "colsample_bytree": 0.9574,
        "reg_alpha": 2.935e-06,
        "reg_lambda": 0.001326,
        "max_depth": 7,
        "min_split_gain": 0.02777,
        "random_state": 42,
        "n_jobs": -1,
    },
    "jumps": {
        "objective": "lambdarank",
        "n_estimators": 3000,
        "learning_rate": 0.04460,
        "num_leaves": 100,
        "min_child_samples": 63,
        "subsample": 0.7438,
        "colsample_bytree": 0.5903,
        "reg_alpha": 0.2719,
        "reg_lambda": 0.004538,
        "max_depth": 9,
        "min_split_gain": 0.06107,
        "random_state": 42,
        "n_jobs": -1,
    },
}

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def run_window(category, window, params):
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

    model = lgb.LGBMRanker(**params)
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
    raw_softmax = race_softmax(model.predict(X_test, num_iteration=model.best_iteration_), test_ids)
    calibrated = calibrator.transform(raw_softmax)
    calibrated = np.nan_to_num(calibrated, nan=1e-6)
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    test_probs = renormalize(calibrated, test_ids)
    test_probs = np.nan_to_num(test_probs, nan=1e-6)

    tpwr = top_pick_win_rate(test_probs, test_ids, y_test)
    brier = brier_score_loss(y_test, np.clip(test_probs, 0, 1))

    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1").df()
    db.close()

    a = df_test[["race_id", "runner_id"]].copy()
    a["prob"] = test_probs; a["target"] = y_test
    a = a.merge(sp_df, on="runner_id", how="left")
    a["implied"] = 1.0 / a["sp_decimal"]
    a["edge"] = a["prob"] - a["implied"]
    a["profit"] = a["target"] * (a["sp_decimal"] - 1) - (1 - a["target"])
    has_sp = a[a["sp_decimal"].notna()]

    vb = has_sp[has_sp["edge"] > 0.05]
    roi = float(vb["profit"].mean()) if len(vb) > 0 else 0
    pnl = float(vb["profit"].sum()) if len(vb) > 0 else 0
    n_bets = len(vb)

    return {
        "label": window["label"],
        "top_pick": float(tpwr),
        "brier": float(brier),
        "n_bets": n_bets,
        "roi": roi,
        "pnl": pnl,
        "best_iter": model.best_iteration_,
    }


def main():
    for category in ["flat", "jumps"]:
        print(f"\n{'='*75}", flush=True)
        print(f"  {category.upper()} — WALK-FORWARD: DEFAULT vs TUNED", flush=True)
        print(f"{'='*75}", flush=True)

        print(f"\n  {'Window':<14} {'':^3} {'TopPick':>8} {'Bets':>7} {'ROI':>8} {'P&L':>9} {'Trees':>6}", flush=True)
        print(f"  {'-'*60}", flush=True)

        total_default = {"bets": 0, "pnl": 0.0}
        total_tuned = {"bets": 0, "pnl": 0.0}

        for w in WINDOWS:
            rd = run_window(category, w, DEFAULT_PARAMS)
            rt = run_window(category, w, TUNED_PARAMS[category])

            if rd and rt:
                total_default["bets"] += rd["n_bets"]; total_default["pnl"] += rd["pnl"]
                total_tuned["bets"] += rt["n_bets"]; total_tuned["pnl"] += rt["pnl"]

                print(f"  {w['label']:<14} DEF {rd['top_pick']:>7.1%} {rd['n_bets']:>7,} {rd['roi']:>+7.2%} £{rd['pnl']:>+8.0f} {rd['best_iter']:>5}", flush=True)
                print(f"  {'':14} TUN {rt['top_pick']:>7.1%} {rt['n_bets']:>7,} {rt['roi']:>+7.2%} £{rt['pnl']:>+8.0f} {rt['best_iter']:>5}", flush=True)

                diff = rt["roi"] - rd["roi"]
                marker = ">>>" if diff > 0 else "<<<" if diff < -0.01 else "==="
                print(f"  {'':14} {marker} ROI diff: {diff:>+.2%}", flush=True)
                print(flush=True)

        d_roi = total_default["pnl"] / total_default["bets"] if total_default["bets"] > 0 else 0
        t_roi = total_tuned["pnl"] / total_tuned["bets"] if total_tuned["bets"] > 0 else 0
        print(f"  {'TOTAL':<14} DEF {total_default['bets']:>22,} {d_roi:>+7.2%} £{total_default['pnl']:>+8.0f}", flush=True)
        print(f"  {'':14} TUN {total_tuned['bets']:>22,} {t_roi:>+7.2%} £{total_tuned['pnl']:>+8.0f}", flush=True)


if __name__ == "__main__":
    main()
