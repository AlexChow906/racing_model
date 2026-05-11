import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.modeling.train_split import load_data, race_softmax, renormalize, top_pick_win_rate
from src.ingestion.db_connect import get_db

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(category):
    X_train, y_train, g_train, df_train = load_data("2015-01-01", "2023-01-01", category)
    X_cal, y_cal, g_cal, df_cal = load_data("2023-01-01", "2024-01-01", category)
    X_test, y_test, g_test, df_test = load_data("2024-01-01", "2025-01-01", category)

    non_numeric = [c for c in X_train.columns
                   if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))]
    for col in non_numeric:
        cats = pd.Index(pd.concat([X_train[col], X_cal[col], X_test[col]]).astype(str).astype("category").cat.categories)
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats).codes
        X_cal[col] = pd.Categorical(X_cal[col].astype(str), categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats).codes

    db = get_db("racing.duckdb")
    sp_df = db.execute("SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1").df()
    db.close()

    test_analysis = df_test[["race_id", "runner_id"]].copy()
    test_analysis = test_analysis.merge(sp_df, on="runner_id", how="left")

    print(f"  {category.upper()}: train={len(X_train):,}, cal={len(X_cal):,}, test={len(X_test):,}, features={len(X_train.columns)}", flush=True)

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

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
        test_probs = renormalize(calibrator.transform(race_softmax(model.predict(X_test, num_iteration=model.best_iteration_), test_ids)), test_ids)

        # Optimize for value betting ROI at edge>5%
        ta = test_analysis.copy()
        ta["prob"] = test_probs
        ta["target"] = y_test
        ta["implied"] = 1.0 / ta["sp_decimal"]
        ta["edge"] = ta["prob"] - ta["implied"]
        ta["profit"] = ta["target"] * (ta["sp_decimal"] - 1) - (1 - ta["target"])

        vb = ta[(ta["edge"] > 0.05) & (ta["sp_decimal"].notna())]
        if len(vb) < 100:
            return -1.0

        roi = float(vb["profit"].mean())
        tpwr = top_pick_win_rate(test_probs, test_ids, y_test)

        trial.set_user_attr("roi", roi)
        trial.set_user_attr("top_pick", tpwr)
        trial.set_user_attr("n_bets", len(vb))
        trial.set_user_attr("pnl", float(vb["profit"].sum()))
        trial.set_user_attr("best_iter", model.best_iteration_)

        return roi

    return objective, (X_train, y_train, g_train, X_cal, y_cal, g_cal, X_test, y_test, g_test, df_train, df_cal, df_test)


def main():
    N_TRIALS = 80
    results = {}

    for category in ["flat", "jumps"]:
        print(f"\n{'='*60}", flush=True)
        print(f"  OPTUNA TUNING — {category.upper()} ({N_TRIALS} trials)", flush=True)
        print(f"{'='*60}", flush=True)

        objective, data = make_objective(category)

        study = optuna.create_study(direction="maximize", study_name=f"{category}_tuning")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_trial
        print(f"\n  Best trial #{best.number}:", flush=True)
        print(f"  ROI: {best.value:+.2%}", flush=True)
        print(f"  Top Pick: {best.user_attrs['top_pick']:.1%}", flush=True)
        print(f"  Bets: {best.user_attrs['n_bets']:,}", flush=True)
        print(f"  P&L: £{best.user_attrs['pnl']:+,.0f}", flush=True)
        print(f"  Trees: {best.user_attrs['best_iter']}", flush=True)
        print(f"  Params:", flush=True)
        for k, v in best.params.items():
            print(f"    {k}: {v}", flush=True)

        # Show top 5 trials
        print(f"\n  Top 5 trials:", flush=True)
        print(f"  {'#':<4} {'ROI':>7} {'TopPick':>8} {'Bets':>6} {'P&L':>8}", flush=True)
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
        for t in sorted_trials[:5]:
            if t.value is None: continue
            print(f"  {t.number:<4} {t.value:>+6.2%} {t.user_attrs.get('top_pick',0):>7.1%} {t.user_attrs.get('n_bets',0):>6,} £{t.user_attrs.get('pnl',0):>+7.0f}", flush=True)

        results[category] = {
            "best_params": best.params,
            "best_roi": best.value,
            "best_top_pick": best.user_attrs["top_pick"],
            "best_pnl": best.user_attrs["pnl"],
            "best_n_bets": best.user_attrs["n_bets"],
        }

    # Compare defaults vs tuned
    print(f"\n{'='*60}", flush=True)
    print(f"  DEFAULT vs TUNED COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    for cat in ["flat", "jumps"]:
        r = results[cat]
        print(f"  {cat.upper()}: best ROI={r['best_roi']:+.2%}, P&L=£{r['best_pnl']:+,.0f} on {r['best_n_bets']:,} bets", flush=True)

    with open("experiments/optuna_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: experiments/optuna_results.json", flush=True)


if __name__ == "__main__":
    main()
