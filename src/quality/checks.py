from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import duckdb
import yaml

from ingestion.db_connect import get_db


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "racing.duckdb"
LOG_DIR = ROOT / "logs"
SETTINGS_PATH = ROOT / "configs" / "settings.yaml"
STANDARD_RACE_SQL_PATH = ROOT / "sql" / "schema" / "update_standard_race_flags.sql"

MAX_MISSING_TRAINER_PCT = 5.0
MAX_MISSING_JOCKEY_PCT = 5.0


def _load_thresholds_from_settings() -> dict[str, float]:
    if not SETTINGS_PATH.exists():
        return {
            "pre_2017_trainer": 5.0,
            "pre_2017_jockey": 5.0,
            "pre_2017_zero_coverage": 2000.0,
            "from_2017_trainer": MAX_MISSING_TRAINER_PCT,
            "from_2017_jockey": MAX_MISSING_JOCKEY_PCT,
            "from_2017_zero_coverage": 100.0,
            "from_2019_trainer": 3.0,
            "from_2019_jockey": 3.0,
            "from_2019_zero_coverage": 0.0,
        }
    try:
        payload = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}

    dq = payload.get("dq_thresholds", {}) if isinstance(payload, dict) else {}
    pre = dq.get("pre_2017", {}) if isinstance(dq, dict) else {}
    post_2017 = dq.get("from_2017", {}) if isinstance(dq, dict) else {}
    post_2019 = dq.get("from_2019", {}) if isinstance(dq, dict) else {}
    return {
        "pre_2017_trainer": float(pre.get("max_missing_trainer_pct", 5.0)),
        "pre_2017_jockey": float(pre.get("max_missing_jockey_pct", 5.0)),
        "pre_2017_zero_coverage": float(pre.get("max_zero_coverage_standard_races", 2000.0)),
        "from_2017_trainer": float(post_2017.get("max_missing_trainer_pct", MAX_MISSING_TRAINER_PCT)),
        "from_2017_jockey": float(post_2017.get("max_missing_jockey_pct", MAX_MISSING_JOCKEY_PCT)),
        "from_2017_zero_coverage": float(post_2017.get("max_zero_coverage_standard_races", 100.0)),
        "from_2019_trainer": float(post_2019.get("max_missing_trainer_pct", 3.0)),
        "from_2019_jockey": float(post_2019.get("max_missing_jockey_pct", 3.0)),
        "from_2019_zero_coverage": float(post_2019.get("max_zero_coverage_standard_races", 0.0)),
    }


def _thresholds_for_window(min_race_date: date | None, max_race_date: date | None) -> tuple[float, float]:
    cfg = _load_thresholds_from_settings()
    year_ref = max_race_date.year if max_race_date is not None else (min_race_date.year if min_race_date is not None else None)
    if year_ref is not None and year_ref >= 2019:
        return cfg["from_2019_trainer"], cfg["from_2019_jockey"]
    if year_ref is not None and year_ref < 2017:
        return cfg["pre_2017_trainer"], cfg["pre_2017_jockey"]
    return cfg["from_2017_trainer"], cfg["from_2017_jockey"]


def _zero_coverage_threshold_for_window(min_race_date: date | None, max_race_date: date | None) -> float:
    cfg = _load_thresholds_from_settings()
    year_ref = max_race_date.year if max_race_date is not None else (min_race_date.year if min_race_date is not None else None)
    if year_ref is not None and year_ref >= 2019:
        return cfg["from_2019_zero_coverage"]
    if year_ref is not None and year_ref < 2017:
        return cfg["pre_2017_zero_coverage"]
    return cfg["from_2017_zero_coverage"]


def ensure_standard_race_flag(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute("ALTER TABLE races ADD COLUMN IF NOT EXISTS is_standard_race BOOLEAN")
    if not STANDARD_RACE_SQL_PATH.exists():
        raise FileNotFoundError(f"Missing standard-race SQL: {STANDARD_RACE_SQL_PATH}")

    con.execute(STANDARD_RACE_SQL_PATH.read_text(encoding="utf-8"))
    restored = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM races
        WHERE is_standard_race IS NULL
        """,
    )
    con.execute(
        """
        UPDATE races
        SET is_standard_race = TRUE
        WHERE is_standard_race IS NULL
        """
    )
    standard_races = _single_count(con, "SELECT COUNT(*) FROM races WHERE is_standard_race")
    non_standard_races = _single_count(con, "SELECT COUNT(*) FROM races WHERE NOT is_standard_race")
    return {
        "excluded_updates": non_standard_races,
        "restored_updates": restored,
        "standard_races": standard_races,
        "non_standard_races": non_standard_races,
    }


def _single_count(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def check_no_missing_keys(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT race_id AS key_col FROM races WHERE race_id IS NULL OR race_id = ''
            UNION ALL
            SELECT runner_id FROM runners WHERE runner_id IS NULL OR runner_id = ''
            UNION ALL
            SELECT race_id FROM results WHERE race_id IS NULL OR race_id = ''
            UNION ALL
            SELECT runner_id FROM results WHERE runner_id IS NULL OR runner_id = ''
        ) keys
        """,
    )
    return {"check_name": "check_no_missing_keys", "passed": failing_count == 0, "failing_count": failing_count, "details": "Missing race_id/runner_id across core tables"}


def check_timestamp_range(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT event_timestamp_utc AS ts FROM races
            UNION ALL SELECT event_timestamp_utc FROM runners
            UNION ALL SELECT event_timestamp_utc FROM results
            UNION ALL SELECT event_timestamp_utc FROM odds_snapshots
            UNION ALL SELECT event_timestamp_utc FROM horse_history
            UNION ALL SELECT event_timestamp_utc FROM trainer_history
            UNION ALL SELECT event_timestamp_utc FROM jockey_history
        ) t
        WHERE ts < TIMESTAMPTZ '2010-01-01 00:00:00+00'
           OR ts > TIMESTAMPTZ '2030-01-01 00:00:00+00'
           OR ts IS NULL
        """,
    )
    return {"check_name": "check_timestamp_range", "passed": failing_count == 0, "failing_count": failing_count, "details": "event_timestamp_utc outside expected range"}


def check_referential_integrity(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM results res
        LEFT JOIN races ra ON res.race_id = ra.race_id
        WHERE ra.race_id IS NULL
        """,
    )
    return {"check_name": "check_referential_integrity", "passed": failing_count == 0, "failing_count": failing_count, "details": "results without races parent"}


def check_runner_integrity(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM results res
        LEFT JOIN runners r ON res.runner_id = r.runner_id
        WHERE r.runner_id IS NULL
        """,
    )
    return {"check_name": "check_runner_integrity", "passed": failing_count == 0, "failing_count": failing_count, "details": "results without runners parent"}


def check_single_winner_per_race(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                race_id,
                SUM(CASE WHEN won THEN 1 ELSE 0 END) AS winner_count
            FROM results
            GROUP BY race_id
            HAVING winner_count > 2
        ) x
        """,
    )
    return {
        "check_name": "check_single_winner_per_race",
        "passed": failing_count == 0,
        "failing_count": failing_count,
        "details": "races with implausible winner counts (>2); allows dead-heats",
    }


def check_races_have_runners(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM races ra
        LEFT JOIN (
            SELECT race_id, COUNT(*) AS n_runners
            FROM runners
            GROUP BY race_id
        ) rr ON ra.race_id = rr.race_id
        WHERE COALESCE(rr.n_runners, 0) = 0
        """,
    )
    return {"check_name": "check_races_have_runners", "passed": failing_count == 0, "failing_count": failing_count, "details": "races with zero runners"}


def check_odds_range(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM odds_snapshots
        WHERE decimal_odds <= 1.01 OR decimal_odds > 1000 OR decimal_odds IS NULL
        """,
    )
    return {"check_name": "check_odds_range", "passed": failing_count == 0, "failing_count": failing_count, "details": "odds outside [1.01, 1000]"}


def check_field_size_consistency(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT ra.race_id, ra.field_size, COUNT(r.runner_id) AS actual_runners
            FROM races ra
            LEFT JOIN runners r ON ra.race_id = r.race_id
            WHERE ra.field_size IS NOT NULL
            GROUP BY ra.race_id, ra.field_size
            HAVING ra.field_size <> COUNT(r.runner_id)
        ) mismatches
        """,
    )
    return {"check_name": "check_field_size_consistency", "passed": failing_count == 0, "failing_count": failing_count, "details": "field_size differs from runner count"}


def check_no_duplicate_races(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM (
            SELECT race_id, COUNT(*) AS c
            FROM races
            GROUP BY race_id
            HAVING COUNT(*) > 1
        ) d
        """,
    )
    return {"check_name": "check_no_duplicate_races", "passed": failing_count == 0, "failing_count": failing_count, "details": "duplicate race_id values in races"}


def check_history_timestamps(con: duckdb.DuckDBPyConnection) -> dict:
    failing_count = _single_count(
        con,
        """
        SELECT COUNT(*)
        FROM horse_history hh
        JOIN races ra ON hh.race_id = ra.race_id
        WHERE hh.event_timestamp_utc <> ra.scheduled_off_utc
        """,
    )
    return {"check_name": "check_history_timestamps", "passed": failing_count == 0, "failing_count": failing_count, "details": "horse_history.event_timestamp_utc must equal races.scheduled_off_utc"}


def _race_date_window_sql(min_race_date: date | None, max_race_date: date | None) -> str:
    filters: list[str] = []
    if min_race_date is not None:
        filters.append(f"ra.race_date >= DATE '{min_race_date.isoformat()}'")
    if max_race_date is not None:
        filters.append(f"ra.race_date <= DATE '{max_race_date.isoformat()}'")
    if not filters:
        return ""
    return " AND " + " AND ".join(filters)


def check_trainer_jockey_missing_rate(
    con: duckdb.DuckDBPyConnection,
    min_race_date: date | None = None,
    max_race_date: date | None = None,
) -> dict:
    trainer_threshold, jockey_threshold = _thresholds_for_window(
        min_race_date=min_race_date,
        max_race_date=max_race_date,
    )
    date_clause = _race_date_window_sql(min_race_date=min_race_date, max_race_date=max_race_date)
    total_runners, missing_trainer, missing_jockey = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_runners,
            SUM(CASE WHEN COALESCE(TRIM(r.trainer_name), '') = '' THEN 1 ELSE 0 END) AS missing_trainer,
            SUM(CASE WHEN COALESCE(TRIM(r.jockey_name), '') = '' THEN 1 ELSE 0 END) AS missing_jockey
        FROM runners r
        JOIN races ra ON r.race_id = ra.race_id
        WHERE ra.is_standard_race
        {date_clause}
        """
    ).fetchone()

    total_runners = int(total_runners or 0)
    missing_trainer = int(missing_trainer or 0)
    missing_jockey = int(missing_jockey or 0)
    trainer_pct = (missing_trainer / total_runners * 100.0) if total_runners else 0.0
    jockey_pct = (missing_jockey / total_runners * 100.0) if total_runners else 0.0

    passed = trainer_pct <= trainer_threshold and jockey_pct <= jockey_threshold
    details = (
        f"standard_runners={total_runners}; "
        f"missing_trainer={missing_trainer} ({trainer_pct:.2f}% <= {trainer_threshold:.2f}%); "
        f"missing_jockey={missing_jockey} ({jockey_pct:.2f}% <= {jockey_threshold:.2f}%)"
    )
    failing_count = 0 if passed else int(missing_trainer + missing_jockey)
    return {
        "check_name": "check_trainer_jockey_missing_rate",
        "passed": passed,
        "failing_count": failing_count,
        "details": details,
    }


def check_zero_coverage_standard_races(
    con: duckdb.DuckDBPyConnection,
    min_race_date: date | None = None,
    max_race_date: date | None = None,
) -> dict:
    zero_threshold = _zero_coverage_threshold_for_window(
        min_race_date=min_race_date,
        max_race_date=max_race_date,
    )
    date_clause = _race_date_window_sql(min_race_date=min_race_date, max_race_date=max_race_date)
    failing_rows = con.execute(
        f"""
        SELECT ra.race_id
        FROM races ra
        JOIN runners r ON ra.race_id = r.race_id
        WHERE ra.is_standard_race
          {date_clause}
        GROUP BY ra.race_id
        HAVING SUM(CASE WHEN COALESCE(TRIM(r.trainer_name), '') <> '' OR COALESCE(TRIM(r.jockey_name), '') <> '' THEN 1 ELSE 0 END) = 0
        """
    ).fetchall()
    race_ids = [str(row[0]) for row in failing_rows]
    preview = ",".join(race_ids[:20])
    details = f"standard races with zero trainer+jockey coverage <= {int(zero_threshold)}"
    if race_ids:
        details = f"{details}; race_ids={preview}"
        if len(race_ids) > 20:
            details = f"{details}...(+{len(race_ids) - 20} more)"
    return {
        "check_name": "check_zero_coverage_standard_races",
        "passed": len(race_ids) <= int(zero_threshold),
        "failing_count": len(race_ids),
        "details": details,
    }


def run_all_checks(
    db_path: Path | None = None,
    min_race_date: date | None = None,
    max_race_date: date | None = None,
) -> list[dict]:
    chosen_db = db_path or DB_PATH
    con = get_db(chosen_db)
    try:
        standard_summary = ensure_standard_race_flag(con)
        print(
            "standard_race_filter "
            f"standard={standard_summary['standard_races']} "
            f"excluded={standard_summary['non_standard_races']} "
            f"updated_false={standard_summary['excluded_updates']} "
            f"updated_true={standard_summary['restored_updates']}"
        )
        checks: list[Callable[[duckdb.DuckDBPyConnection], dict]] = [
            check_no_missing_keys,
            check_timestamp_range,
            check_referential_integrity,
            check_runner_integrity,
            check_single_winner_per_race,
            check_races_have_runners,
            check_odds_range,
            check_field_size_consistency,
            check_no_duplicate_races,
            check_history_timestamps,
            lambda db_con: check_trainer_jockey_missing_rate(db_con, min_race_date=min_race_date, max_race_date=max_race_date),
            lambda db_con: check_zero_coverage_standard_races(db_con, min_race_date=min_race_date, max_race_date=max_race_date),
        ]
        results = [fn(con) for fn in checks]
    finally:
        con.close()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / f"dq_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"
    lines = ["Data Quality Report", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", ""]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"{status} | {result['check_name']} | failing_count={result['failing_count']} | {result['details']}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\nSaved report: {report_path}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        raise RuntimeError(f"DQ checks failed: {len(failures)} failing checks")
    return results


if __name__ == "__main__":
    run_all_checks()
