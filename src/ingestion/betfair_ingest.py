from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


RAW_ROOT = Path("data/raw/odds_snapshots")


def _month_partition(ts: datetime) -> Path:
    ts = ts.astimezone(timezone.utc)
    return RAW_ROOT / f"{ts.year:04d}" / f"{ts.month:02d}" / f"{ts.day:02d}"


def save_race_snapshot_to_parquet(
    race_id: str,
    rows: Iterable[dict],
    snapshot_time_utc: datetime,
    snapshot_label: str,
) -> Path:
    """Persist one race odds snapshot to the required parquet layout.

        Layout:
            data/raw/odds_snapshots/YYYY/MM/DD/<snapshot_label>_<race_id>.parquet
    """
    partition = _month_partition(snapshot_time_utc)
    partition.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("No rows provided for snapshot")

    snapshot_utc = snapshot_time_utc.astimezone(timezone.utc)
    df["snapshot_timestamp_utc"] = snapshot_utc
    df["snapshot_label"] = snapshot_label
    if "ingest_timestamp_utc" not in df.columns:
        df["ingest_timestamp_utc"] = datetime.now(timezone.utc)

    out_path = partition / f"{snapshot_label}_{race_id}.parquet"

    # We append within the same partition file if re-run for idempotency.
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        dedupe_cols = [
            "runner_id",
            "snapshot_timestamp_utc",
            "source",
            "market_type",
        ]
        available_dedupe_cols = [c for c in dedupe_cols if c in combined.columns]
        if available_dedupe_cols:
            combined = combined.drop_duplicates(subset=available_dedupe_cols, keep="last")
        combined.to_parquet(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    return out_path


def normalize_betfair_price(row: dict) -> dict:
    """Map source payload keys to canonical odds schema keys."""
    snapshot_ts = row.get("snapshot_timestamp_utc")
    snapshot_label = row.get("snapshot_label")
    race_id = row.get("race_id")
    runner_id = row.get("runner_id")
    source = row.get("source", "betfair_exchange")
    market_type = row.get("market_type", "WIN")

    snapshot_id = f"{race_id}_{runner_id}_{source}_{market_type}_{snapshot_ts}_{snapshot_label}"
    return {
        "snapshot_id": snapshot_id,
        "race_id": race_id,
        "runner_id": runner_id,
        "runner_name": row.get("runner_name"),
        "source": source,
        "market_type": market_type,
        "minutes_to_off": row.get("minutes_to_off"),
        "snapshot_timestamp_utc": snapshot_ts,
        "snapshot_label": snapshot_label,
        "decimal_odds": row.get("decimal_odds"),
        "implied_prob_raw": 1.0 / row.get("decimal_odds") if row.get("decimal_odds") else None,
        "traded_volume_gbp": row.get("traded_volume_gbp"),
        "market_status": row.get("market_status"),
        "event_timestamp_utc": snapshot_ts,
        "decision_cutoff_utc": row.get("decision_cutoff_utc") or snapshot_ts,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }


def insert_odds_snapshots_to_duckdb(db_path: Path | str, rows: Iterable[dict]) -> int:
    """Insert normalized odds rows into odds_snapshots, ignoring duplicates by PK."""
    records = list(rows)
    if not records:
        return 0

    df = pd.DataFrame(records)
    required_cols = [
        "snapshot_id",
        "race_id",
        "runner_id",
        "runner_name",
        "source",
        "market_type",
        "snapshot_timestamp_utc",
        "snapshot_label",
        "minutes_to_off",
        "decimal_odds",
        "implied_prob_raw",
        "traded_volume_gbp",
        "market_status",
        "event_timestamp_utc",
        "decision_cutoff_utc",
        "ingest_timestamp_utc",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    con = duckdb.connect(str(db_path))
    try:
        before_count = int(con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0])
        con.register("incoming_odds", df[required_cols])
        con.execute(
            """
            INSERT OR IGNORE INTO odds_snapshots (
                snapshot_id,
                race_id,
                runner_id,
                runner_name,
                source,
                market_type,
                snapshot_timestamp_utc,
                snapshot_label,
                minutes_to_off,
                decimal_odds,
                implied_prob_raw,
                traded_volume_gbp,
                market_status,
                event_timestamp_utc,
                decision_cutoff_utc,
                ingest_timestamp_utc
            )
            SELECT
                snapshot_id,
                race_id,
                runner_id,
                runner_name,
                source,
                market_type,
                snapshot_timestamp_utc,
                snapshot_label,
                minutes_to_off,
                decimal_odds,
                implied_prob_raw,
                traded_volume_gbp,
                market_status,
                event_timestamp_utc,
                decision_cutoff_utc,
                ingest_timestamp_utc
            FROM incoming_odds
            """
        )
        after_count = int(con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0])
    finally:
        con.close()
    return max(0, after_count - before_count)


def write_morning_alerts_csv(db_path: Path | str, target_date_utc: datetime) -> Path:
    alerts_dir = Path("logs")
    alerts_dir.mkdir(parents=True, exist_ok=True)
    out_path = alerts_dir / f"morning_alerts_{target_date_utc.strftime('%Y%m%d')}.csv"

    con = duckdb.connect(str(db_path))
    try:
        query = """
            WITH evening AS (
                SELECT
                    race_id,
                    runner_id,
                    decimal_odds AS evening_price
                FROM odds_snapshots
                WHERE snapshot_label = 'evening_21h'
            ),
            morning AS (
                SELECT
                    race_id,
                    runner_id,
                    decimal_odds AS morning_price
                FROM odds_snapshots
                WHERE snapshot_label = 'morning_09h'
            )
            SELECT
                m.race_id,
                m.runner_id,
                e.evening_price,
                m.morning_price,
                CASE
                    WHEN e.evening_price IS NULL OR e.evening_price = 0 THEN NULL
                    ELSE (m.morning_price - e.evening_price) / e.evening_price
                END AS price_move_pct
            FROM morning m
            LEFT JOIN evening e
                ON m.race_id = e.race_id
               AND m.runner_id = e.runner_id
            WHERE e.evening_price IS NOT NULL
              AND ABS((m.morning_price - e.evening_price) / e.evening_price) > 0.20
            ORDER BY m.race_id, m.runner_id
        """
        df = con.execute(query).df()
    finally:
        con.close()

    df.to_csv(out_path, index=False)
    return out_path
