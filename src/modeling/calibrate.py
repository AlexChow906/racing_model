import numpy as np
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
import pandas as pd


def scores_to_probs(df_val, raw_scores):
    """Convert raw LightGBM scores to race-level probabilities."""
    df_val = df_val.copy()
    df_val["raw_score"] = raw_scores

    # Softmax per race — enforces sum to 1.0
    df_val["raw_prob"] = df_val.groupby("race_id")["raw_score"].transform(
        lambda x: softmax(x.values)
    )
    return df_val


def calibrate(df_with_probs):
    """Fit isotonic regression on out-of-fold predictions."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(
        df_with_probs["raw_prob"].values,
        df_with_probs["target"].values
    )

    df_with_probs["calibrated_prob"] = calibrator.transform(
        df_with_probs["raw_prob"].values
    )

    # Re-normalise per race after calibration
    df_with_probs["calibrated_prob"] = (
        df_with_probs.groupby("race_id")["calibrated_prob"]
        .transform(lambda x: x / x.sum())
    )
    return df_with_probs, calibrator


def verify_sum_to_one(df_val):
    race_sums = df_val.groupby("race_id")["calibrated_prob"].sum()
    assert (race_sums - 1.0).abs().max() < 1e-6, "Probabilities do not sum to 1"
    print("Sum-to-one check: PASS")
