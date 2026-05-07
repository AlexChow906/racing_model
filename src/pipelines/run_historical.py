from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.betfair_historical import main as historical_main
from ingestion.rematch import run_rematch
from ingestion.rpscrape_enrich import enrich_from_rpscrape
from quality.checks import ensure_standard_race_flag, run_all_checks
from quality.leakage_guard import check_no_leakage
DB_PATH = ROOT / "racing.duckdb"


def _count_snapshot(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    tables = ["races", "runners", "results", "horse_history", "trainer_history", "jockey_history", "odds_snapshots"]
    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def backfill_trainer_history(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(
        """
        INSERT OR IGNORE INTO trainer_history (
            history_id,
            trainer_id,
            trainer_name,
            race_id,
            race_date,
            scheduled_off_utc,
            course_id,
            race_type,
            going_code,
            distance_furlongs,
            race_class,
            days_since_last_run,
            won,
            finishing_position,
            field_size,
            event_timestamp_utc,
            decision_cutoff_utc,
            ingest_timestamp_utc
        )
        SELECT
            r.runner_id || '_tr' AS history_id,
            r.trainer_id,
            COALESCE(r.trainer_name, 'unknown') AS trainer_name,
            r.race_id,
            ra.race_date,
            ra.scheduled_off_utc,
            ra.course_id,
            ra.race_type,
            ra.going_code,
            ra.distance_furlongs,
            ra.race_class,
            r.days_since_last_run,
            res.won,
            res.finishing_position,
            ra.field_size,
            ra.scheduled_off_utc AS event_timestamp_utc,
            ra.decision_cutoff_utc,
            NOW() AS ingest_timestamp_utc
        FROM runners r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN results res ON r.runner_id = res.runner_id
                WHERE r.trainer_id IS NOT NULL
                    AND ra.is_standard_race
        """
    )
    return int(con.execute("SELECT COUNT(*) FROM trainer_history").fetchone()[0])


def backfill_jockey_history(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(
        """
        INSERT OR IGNORE INTO jockey_history (
            history_id,
            jockey_id,
            jockey_name,
            trainer_id,
            race_id,
            race_date,
            scheduled_off_utc,
            course_id,
            race_type,
            going_code,
            won,
            finishing_position,
            field_size,
            event_timestamp_utc,
            decision_cutoff_utc,
            ingest_timestamp_utc
        )
        SELECT
            r.runner_id || '_jk' AS history_id,
            r.jockey_id,
            COALESCE(r.jockey_name, 'unknown') AS jockey_name,
            r.trainer_id,
            r.race_id,
            ra.race_date,
            ra.scheduled_off_utc,
            ra.course_id,
            ra.race_type,
            ra.going_code,
            res.won,
            res.finishing_position,
            ra.field_size,
            ra.scheduled_off_utc AS event_timestamp_utc,
            ra.decision_cutoff_utc,
            NOW() AS ingest_timestamp_utc
        FROM runners r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN results res ON r.runner_id = res.runner_id
                WHERE r.jockey_id IS NOT NULL
                    AND ra.is_standard_race
        """
    )
    return int(con.execute("SELECT COUNT(*) FROM jockey_history").fetchone()[0])


def run_pipeline() -> None:
    started = time.time()
    historical_main()
    rps_enrich_summary = enrich_from_rpscrape(db_path=DB_PATH, countries=("GB", "IE"))
    rematch_summary = run_rematch(db_path=DB_PATH, input_glob="data/raw/rpscrape/**/*.csv", dry_run=False)

    con = duckdb.connect(str(DB_PATH))
    try:
        ensure_standard_race_flag(con)
        before = _count_snapshot(con)
        trainer_rows = backfill_trainer_history(con)
        jockey_rows = backfill_jockey_history(con)
        after = _count_snapshot(con)
    finally:
        con.close()

    run_all_checks(DB_PATH)
    # horse_history is post-race by design; enforce leakage guard on odds snapshots where
    # decision_cutoff_utc is expected to align with snapshot-time ingestion contracts.
    check_no_leakage("odds_snapshots", "decision_cutoff_utc", DB_PATH)

    con2 = duckdb.connect(str(DB_PATH))
    try:
        rerun_before = _count_snapshot(con2)
        backfill_trainer_history(con2)
        backfill_jockey_history(con2)
        rerun_after = _count_snapshot(con2)
    finally:
        con2.close()

    print("Historical pipeline summary")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"duration_sec={round(time.time() - started, 3)}")
    print(f"rps_enrich_summary={rps_enrich_summary}")
    print(f"rematch_summary={rematch_summary}")
    print(f"trainer_history_rows={trainer_rows}")
    print(f"jockey_history_rows={jockey_rows}")
    print(f"counts_before={before}")
    print(f"counts_after={after}")
    print(f"idempotency_before={rerun_before}")
    print(f"idempotency_after={rerun_after}")


if __name__ == "__main__":
    run_pipeline()
