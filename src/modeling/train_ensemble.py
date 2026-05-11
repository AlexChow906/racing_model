import json
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

from src.ingestion.db_connect import get_db

TRAIN_START = "2015-01-01"
TRAIN_END = "2021-01-01"
STACK_START = "2021-01-01"
STACK_END = "2022-01-01"
TEST_START = "2022-01-01"
TEST_END = "2023-01-01"

EXCLUDE = [
    "runner_id", "race_id", "race_date", "decision_cutoff_utc",
    "target", "event_timestamp_utc",
]

DROP_LOW_IMPORTANCE = [
    "weight_change_lbs", "collateral_franked_winners", "trainer_runs_90d",
    "race_class_encoded", "career_win_rate", "draw_bias_coefficient",
    "horse_course_affinity", "jockey_trainer_combo_runs", "horse_distance_affinity",
    "horse_win_rate_last_10", "draw_course_going_win_rate", "pace_front_runners",
    "draw_position", "horse_going_group_affinity", "pace_hold_up_horses",
    "field_size", "race_month", "race_type_encoded", "going_encoded",
    "horse_first_time_headgear", "race_day_of_week", "horse_course_runs",
    "surface_encoded", "horse_wins_last_5", "is_jumps", "draw_is_null",
]


def load_split(start, end):
    db = get_db("racing.duckdb")
    df = db.execute(f"""
        SELECT * FROM feature_store
        WHERE race_date >= '{start}' AND race_date < '{end}' AND target IS NOT NULL
        ORDER BY race_date, race_id
    """).df()
    df = df[df["race_date"] >= "2015-01-01"].copy()
    groups = df.groupby("race_id", sort=False)["runner_id"].count().values
    y = df["target"].astype(int).values
    drop_cols = [c for c in EXCLUDE + DROP_LOW_IMPORTANCE if c in df.columns]
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


def top_pick_win_rate(probs, race_ids, y_true):
    frame = pd.DataFrame({"race_id": race_ids, "prob": probs, "target": y_true})
    top = frame.sort_values(["race_id", "prob"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    return float(top["target"].mean())


def encode_non_numeric(X_train, X_val):
    non_numeric = [c for c in X_train.columns
                   if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
    for col in non_numeric:
        cats = pd.Index(pd.concat([X_train[col], X_val[col]]).astype(str).astype("category").cat.categories)
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats).codes
    return X_train, X_val


def train_lgb(X_train, y_train, g_train, X_eval, y_eval, g_eval):
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=3000, learning_rate=0.01,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_eval, y_eval)], eval_group=[g_eval], eval_at=[1, 3],
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(200)],
    )
    return model


def train_xgb(X_train, y_train, g_train, X_eval, y_eval, g_eval):
    model = xgb.XGBRanker(
        objective="rank:ndcg", n_estimators=3000, learning_rate=0.01,
        max_depth=6, min_child_weight=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        tree_method="hist", eval_metric="ndcg@1",
        callbacks=[xgb.callback.EarlyStopping(rounds=200, metric_name="ndcg@1", maximize=True)],
    )
    model.fit(
        X_train, y_train, group=g_train,
        eval_set=[(X_eval, y_eval)], eval_group=[g_eval],
        verbose=200,
    )
    return model


def train_cb(X_train, y_train, g_train, X_eval, y_eval, g_eval):
    train_pool = cb.Pool(X_train, label=y_train, group_id=np.repeat(np.arange(len(g_train)), g_train))
    eval_pool = cb.Pool(X_eval, label=y_eval, group_id=np.repeat(np.arange(len(g_eval)), g_eval))
    model = cb.CatBoost({
        "loss_function": "YetiRank", "iterations": 3000, "learning_rate": 0.01,
        "depth": 6, "random_seed": 42, "verbose": 200,
        "early_stopping_rounds": 200, "task_type": "CPU",
    })
    model.fit(train_pool, eval_set=eval_pool)
    return model


def predict_probs(model, X, race_ids, model_type):
    if model_type == "lgb":
        scores = model.predict(X, num_iteration=model.best_iteration_)
    elif model_type == "xgb":
        scores = model.predict(X)
    elif model_type == "cb":
        scores = model.predict(cb.Pool(X))
    return race_softmax(scores, race_ids)


def renormalize_per_race(probs, race_ids):
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


def main():
    print("Loading data (3-way split)...", flush=True)
    X_train, y_train, g_train, df_train = load_split(TRAIN_START, TRAIN_END)
    X_stack, y_stack, g_stack, df_stack = load_split(STACK_START, STACK_END)
    X_test, y_test, g_test, df_test = load_split(TEST_START, TEST_END)

    non_numeric = [c for c in X_train.columns
                   if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
    for col in non_numeric:
        cats = pd.Index(pd.concat([X_train[col], X_stack[col], X_test[col]]).astype(str).astype("category").cat.categories)
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
        X_stack[col] = pd.Categorical(X_stack[col].astype(str), categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

    feature_names = X_train.columns.tolist()
    print(f"Train (base models):  {len(X_train):,} rows, {len(g_train):,} races  [{TRAIN_START} to {TRAIN_END}]", flush=True)
    print(f"Stack (meta-learner): {len(X_stack):,} rows, {len(g_stack):,} races  [{STACK_START} to {STACK_END}]", flush=True)
    print(f"Test  (evaluation):   {len(X_test):,} rows, {len(g_test):,} races  [{TEST_START} to {TEST_END}]", flush=True)
    print(f"Features: {len(feature_names)}", flush=True)

    # === Train base models on TRAIN, early-stop on STACK ===
    print("\n--- Training LightGBM ---", flush=True)
    lgb_model = train_lgb(X_train, y_train, g_train, X_stack, y_stack, g_stack)
    print(f"  best iter: {lgb_model.best_iteration_}", flush=True)

    print("\n--- Training XGBoost ---", flush=True)
    xgb_model = train_xgb(X_train, y_train, g_train, X_stack, y_stack, g_stack)
    print(f"  best iter: {xgb_model.best_iteration}", flush=True)

    print("\n--- Training CatBoost ---", flush=True)
    cb_model = train_cb(X_train, y_train, g_train, X_stack, y_stack, g_stack)
    print(f"  best iter: {cb_model.best_iteration_}", flush=True)

    # === Generate stacking features on STACK set (out-of-sample for meta-learner) ===
    print("\n--- Generating stacking features ---", flush=True)
    stack_race_ids = df_stack["race_id"].to_numpy()
    lgb_stack_probs = predict_probs(lgb_model, X_stack, stack_race_ids, "lgb")
    xgb_stack_probs = predict_probs(xgb_model, X_stack, stack_race_ids, "xgb")
    cb_stack_probs = predict_probs(cb_model, X_stack, stack_race_ids, "cb")

    stack_features = np.column_stack([lgb_stack_probs, xgb_stack_probs, cb_stack_probs])

    # === Train meta-learner (Platt calibration) on STACK set ===
    print("--- Training meta-learner ---", flush=True)
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(stack_features, y_stack)
    print(f"  Meta weights: LGB={meta_model.coef_[0][0]:.3f}, XGB={meta_model.coef_[0][1]:.3f}, CB={meta_model.coef_[0][2]:.3f}", flush=True)

    # === Evaluate everything on TEST set (fully out-of-sample) ===
    print("\n--- Evaluating on TEST set ---", flush=True)
    test_race_ids = df_test["race_id"].to_numpy()
    lgb_test_probs = predict_probs(lgb_model, X_test, test_race_ids, "lgb")
    xgb_test_probs = predict_probs(xgb_model, X_test, test_race_ids, "xgb")
    cb_test_probs = predict_probs(cb_model, X_test, test_race_ids, "cb")

    test_stack_features = np.column_stack([lgb_test_probs, xgb_test_probs, cb_test_probs])
    ensemble_raw = meta_model.predict_proba(test_stack_features)[:, 1]
    ensemble_probs = renormalize_per_race(ensemble_raw, test_race_ids)

    # Also simple average as baseline
    avg_probs = renormalize_per_race((lgb_test_probs + xgb_test_probs + cb_test_probs) / 3, test_race_ids)

    print(f"\n=== TEST SET RESULTS ({TEST_START} to {TEST_END}) ===", flush=True)
    models = {
        "LightGBM": lgb_test_probs,
        "XGBoost": xgb_test_probs,
        "CatBoost": cb_test_probs,
        "SimpleAvg": avg_probs,
        "StackedEnsemble": ensemble_probs,
    }
    for name, probs in models.items():
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, np.clip(probs, 1e-15, 1 - 1e-15), labels=[0, 1])
        tpwr = top_pick_win_rate(probs, test_race_ids, y_test)
        print(f"  {name:<18} Brier={brier:.5f}  LogLoss={ll:.5f}  TopPick={tpwr:.4f}", flush=True)

    # === Save ===
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_ensemble_v2"
    Path("models").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    lgb_model.booster_.save_model(f"models/{run_id}_lgb.lgbm")
    xgb_model.save_model(f"models/{run_id}_xgb.json")
    cb_model.save_model(f"models/{run_id}_cb.cbm")
    with open(f"models/{run_id}_meta.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    best_probs = ensemble_probs
    best_name = "StackedEnsemble"
    meta = {
        "run_id": run_id,
        "split": {"train": [TRAIN_START, TRAIN_END], "stack": [STACK_START, STACK_END], "test": [TEST_START, TEST_END]},
        "features": feature_names,
        "n_train": len(X_train), "n_stack": len(X_stack), "n_test": len(X_test),
        "lgb_best_iter": lgb_model.best_iteration_,
        "xgb_best_iter": int(xgb_model.best_iteration),
        "cb_best_iter": int(cb_model.best_iteration_),
        "meta_weights": {"lgb": float(meta_model.coef_[0][0]), "xgb": float(meta_model.coef_[0][1]), "cb": float(meta_model.coef_[0][2])},
        "test_metrics": {
            name: {
                "brier": float(brier_score_loss(y_test, probs)),
                "log_loss": float(log_loss(y_test, np.clip(probs, 1e-15, 1 - 1e-15), labels=[0, 1])),
                "top_pick_win_rate": float(top_pick_win_rate(probs, test_race_ids, y_test)),
            }
            for name, probs in models.items()
        },
    }
    with open(f"experiments/{run_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: models/{run_id}_*.* and experiments/{run_id}.json", flush=True)


if __name__ == "__main__":
    main()
