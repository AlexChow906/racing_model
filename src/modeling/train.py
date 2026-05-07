import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.ingestion.db_connect import get_db

# Walk-forward Window A
TRAIN_START = "2015-01-01"
TRAIN_END = "2022-01-01"
VAL_START = "2022-01-01"
VAL_END = "2023-01-01"

EXCLUDE = [
    "runner_id",
    "race_id",
    "race_date",
    "decision_cutoff_utc",
    "target",
    "event_timestamp_utc",
]


def load_split(start, end):
    db = get_db("racing.duckdb")
    df = db.execute(
        f"""
        SELECT * FROM feature_store
        WHERE race_date >= '{start}'
        AND race_date <  '{end}'
        AND target IS NOT NULL
        ORDER BY race_date, race_id
    """
    ).df()

    # Drop the known 2014 tail globally from modeling.
    df = df[df["race_date"] >= "2015-01-01"].copy()

    groups = df.groupby("race_id", sort=False)["runner_id"].count().values
    y = df["target"].astype(int).values
    drop_cols = [c for c in EXCLUDE if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y, groups, df


def race_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
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


def calibration_table(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"pred": probs, "target": y_true})
    frame["bin"] = pd.qcut(frame["pred"], q=bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=False)
        .agg(count=("target", "size"), avg_pred=("pred", "mean"), empirical_win_rate=("target", "mean"))
        .reset_index()
    )
    return table


def top_pick_win_rate(probs: np.ndarray, race_ids: np.ndarray, y_true: np.ndarray) -> float:
    frame = pd.DataFrame({"race_id": race_ids, "prob": probs, "target": y_true})
    top = frame.sort_values(["race_id", "prob"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    return float(top["target"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank on feature_store")
    parser.add_argument("--window", default="window_a", choices=["window_a"])
    args = parser.parse_args()

    X_train, y_train, g_train, df_train = load_split(TRAIN_START, TRAIN_END)
    X_val, y_val, g_val, df_val = load_split(VAL_START, VAL_END)

    # Sanity checks before training.
    print("Target distribution in training set:")
    print(pd.Series(y_train).value_counts())
    print(f"Win rate: {y_train.mean():.4f}")
    print(f"Total groups (races): {len(g_train)}")
    print(f"Total rows: {len(y_train)}")
    print(f"Average field size: {len(y_train)/len(g_train):.1f}")
    print(f"Min group size: {g_train.min()}")
    print(f"Max group size: {g_train.max()}")

    # LightGBM requires numeric/boolean dtypes. Encode object-like columns
    # with shared categories across train/val to keep mappings aligned.
    non_numeric_cols = [
        c for c in X_train.columns
        if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))
    ]
    for col in non_numeric_cols:
        categories = pd.Index(
            pd.concat([X_train[col], X_val[col]], axis=0)
            .astype(str)
            .astype("category")
            .cat.categories
        )
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=categories).codes
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=categories).codes

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
        X_train,
        y_train,
        group=g_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        eval_at=[1, 3],
        callbacks=[
            lgb.early_stopping(200, first_metric_only=True),
            lgb.log_evaluation(100),
        ],
    )

    # Save model
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_lgbm_window_a"
    Path("models").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    model.booster_.save_model(f"models/{run_id}.lgbm")

    # Validation predictions and metrics.
    val_scores = model.predict(X_val, num_iteration=model.best_iteration_)
    val_probs = race_softmax(val_scores, df_val["race_id"].to_numpy())

    brier = float(brier_score_loss(y_val, val_probs))
    ll = float(log_loss(y_val, np.clip(val_probs, 1e-15, 1.0 - 1e-15), labels=[0, 1]))
    top1 = top_pick_win_rate(val_probs, df_val["race_id"].to_numpy(), y_val)

    evals = model.evals_result_
    valid_metrics = evals.get("valid_0", {})
    ndcg1 = valid_metrics.get("ndcg@1", [None])[-1]
    ndcg3 = valid_metrics.get("ndcg@3", [None])[-1]

    cal = calibration_table(val_probs, y_val, bins=10)

    # Log experiment
    meta = {
        "run_id": run_id,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "n_estimators_used": model.best_iteration_,
        "features": X_train.columns.tolist(),
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "val_ndcg_at_1": ndcg1,
        "val_ndcg_at_3": ndcg3,
        "val_brier": brier,
        "val_log_loss": ll,
        "val_top_pick_win_rate": top1,
    }
    with open(f"experiments/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved: models/{run_id}.lgbm")
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Final validation NDCG@1: {ndcg1}")
    print(f"Final validation NDCG@3: {ndcg3}")
    print(f"Brier score: {brier}")
    print(f"Log loss: {ll}")
    print(f"Top pick win rate: {top1}")
    print("Calibration table:")
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
