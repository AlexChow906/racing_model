import duckdb
import pandas as pd
from src.ingestion.db_connect import get_db

db = get_db("racing.duckdb")

WINDOWS = [
    {
        "name": "window_a",
        "train_start": "2015-01-01",
        "train_end": "2022-01-01",
        "val_start": "2022-01-01",
        "val_end": "2023-01-01",
        "test_start": "2023-01-01",
        "test_end": "2024-01-01",
    },
    {
        "name": "window_b",
        "train_start": "2015-01-01",
        "train_end": "2023-01-01",
        "val_start": "2023-01-01",
        "val_end": "2024-01-01",
        "test_start": "2024-01-01",
        "test_end": "2025-01-01",
    },
    {
        "name": "window_c",
        "train_start": "2015-01-01",
        "train_end": "2024-01-01",
        "val_start": "2024-01-01",
        "val_end": "2025-01-01",
        "test_start": "2025-01-01",
        "test_end": "2026-01-01",
    },
]


def make_split(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame[(frame["race_date"] >= start) & (frame["race_date"] < end)].copy()


df_all = db.execute("""
    SELECT *
    FROM feature_store
    WHERE target IS NOT NULL
    ORDER BY race_date, race_id
""").df()

# Drop pre-2015 artifact rows from all windows.
df_all = df_all[df_all["race_date"] >= "2015-01-01"].copy()

# Use window_a train split as the canonical training dataset printout.
window_a = WINDOWS[0]
df = make_split(df_all, window_a["train_start"], window_a["train_end"])

# Group sizes — number of runners per race, in race_date order
# LightGBM requires groups in the same order as rows
group_sizes = (
    df.groupby("race_id", sort=False)["runner_id"]
    .count()
    .values
)

# Target — 1 for winner, 0 for all others
# LightGBM lambdarank wants relevance scores
# Use 1/0 — winner is most relevant
y = df["target"].astype(int).values

# Features — drop all non-feature columns
EXCLUDE = [
    "runner_id", "race_id", "race_date",
    "decision_cutoff_utc", "target",
    "event_timestamp_utc"
]
feature_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feature_cols]

print(f"Training rows:    {len(df):,}")
print(f"Races:            {len(group_sizes):,}")
print(f"Features:         {len(feature_cols)}")
print(f"Winners in train: {y.sum():,}")
print(f"Feature columns:\n{feature_cols}")

print("\nWalk-forward windows:")
for window in WINDOWS:
    train_df = make_split(df_all, window["train_start"], window["train_end"])
    val_df = make_split(df_all, window["val_start"], window["val_end"])
    test_df = make_split(df_all, window["test_start"], window["test_end"])
    print(
        f"{window['name']}: "
        f"train_rows={len(train_df):,} train_races={train_df['race_id'].nunique():,} | "
        f"val_rows={len(val_df):,} val_races={val_df['race_id'].nunique():,} | "
        f"test_rows={len(test_df):,} test_races={test_df['race_id'].nunique():,}"
    )
