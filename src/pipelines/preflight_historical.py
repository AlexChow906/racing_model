from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.rpscrape_enrich import enrich_from_rpscrape
from ingestion.db_connect import get_db
from ingestion.rematch import run_rematch
from quality.checks import run_all_checks

DB_PATH = ROOT / "racing.duckdb"
LOG_DIR = ROOT / "logs"


@dataclass
class Thresholds:
    min_rps_match_rate: float
    max_rps_unmatched_rate: float
    max_rps_ambiguous_rate: float


def _country_tokens(countries: tuple[str, ...]) -> tuple[str, ...]:
    mapping = {
        "GB": "pricesukwin",
        "IE": "pricesirewin",
    }
    tokens: list[str] = []
    for country in countries:
        token = mapping.get(country.upper())
        if token:
            tokens.append(token)
    return tuple(tokens)


def _run_sp_ingest(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    countries: tuple[str, ...],
    parse_existing_only: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.ingestion.betfair_historical",
        "--use-sp-history",
        "--start-year",
        str(start_year),
        "--start-month",
        str(start_month),
        "--end-year",
        str(end_year),
        "--end-month",
        str(end_month),
        "--sp-include",
        ",".join(_country_tokens(countries)),
    ]

    if parse_existing_only:
        cmd.extend(["--parse-only", "--scan-existing-sp-csvs"])

    completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _coverage_metrics(
    db_path: Path,
    countries: tuple[str, ...],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    con = get_db(db_path)
    try:
        country_csv = ",".join([f"'{c.upper()}'" for c in countries])
        where_sql = f"country IN ({country_csv}) AND EXTRACT(year FROM race_date) BETWEEN {start_year} AND {end_year}"

        races = int(con.execute(f"SELECT COUNT(*) FROM races WHERE {where_sql}").fetchone()[0])
        runners = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM runners r
                JOIN races ra ON r.race_id = ra.race_id
                WHERE {where_sql}
                """
            ).fetchone()[0]
        )
        results = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM results res
                JOIN races ra ON res.race_id = ra.race_id
                WHERE {where_sql}
                """
            ).fetchone()[0]
        )

        trainer_pop = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM runners r
                JOIN races ra ON r.race_id = ra.race_id
                WHERE {where_sql}
                  AND r.trainer_id IS NOT NULL
                """
            ).fetchone()[0]
        )
        jockey_pop = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM runners r
                JOIN races ra ON r.race_id = ra.race_id
                WHERE {where_sql}
                  AND r.jockey_id IS NOT NULL
                """
            ).fetchone()[0]
        )
        draw_pop = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM runners r
                JOIN races ra ON r.race_id = ra.race_id
                WHERE {where_sql}
                  AND r.draw IS NOT NULL
                """
            ).fetchone()[0]
        )

        return {
            "races": races,
            "runners": runners,
            "results": results,
            "trainer_coverage": round(trainer_pop / runners, 4) if runners else 0.0,
            "jockey_coverage": round(jockey_pop / runners, 4) if runners else 0.0,
            "draw_coverage": round(draw_pop / runners, 4) if runners else 0.0,
        }
    finally:
        con.close()


def _evaluate_gates(
    rps_summary: dict[str, Any],
    dq_ok: bool,
    thresholds: Thresholds,
    require_rps: bool,
) -> dict[str, Any]:
    rows_parsed = int(rps_summary.get("rows_parsed", 0) or 0)
    rows_matched = int(rps_summary.get("rows_matched", 0) or 0)
    rows_unmatched = int(rps_summary.get("rows_unmatched", 0) or 0)
    rows_ambiguous = int(rps_summary.get("rows_ambiguous", 0) or 0)

    match_rate = (rows_matched / rows_parsed) if rows_parsed else 0.0
    unmatched_rate = (rows_unmatched / rows_parsed) if rows_parsed else 0.0
    ambiguous_rate = (rows_ambiguous / rows_parsed) if rows_parsed else 0.0

    rps_status = str(rps_summary.get("status", ""))
    rps_available = rps_status not in {"no_input_files", "no_parseable_rows"}

    checks = {
        "dq_ok": dq_ok,
        "rps_available": (rps_available or (not require_rps)),
        "match_rate_ok": (match_rate >= thresholds.min_rps_match_rate) if rps_available else (not require_rps),
        "unmatched_rate_ok": (unmatched_rate <= thresholds.max_rps_unmatched_rate) if rps_available else (not require_rps),
        "ambiguous_rate_ok": (ambiguous_rate <= thresholds.max_rps_ambiguous_rate) if rps_available else (not require_rps),
    }

    return {
        "checks": checks,
        "rates": {
            "match_rate": round(match_rate, 4),
            "unmatched_rate": round(unmatched_rate, 4),
            "ambiguous_rate": round(ambiguous_rate, 4),
        },
        "ready": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight gate for GB/IE historical backfill")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--end-month", type=int, default=1)
    parser.add_argument("--countries", type=str, default="GB,IE")
    parser.add_argument("--rps-input-glob", type=str, default="data/raw/rpscrape/**/*.csv")
    parser.add_argument("--min-confidence", type=str, choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--parse-existing-only", action="store_true")
    parser.add_argument("--require-rps", action="store_true")
    parser.add_argument("--apply-rps", action="store_true", help="Apply enrichment updates instead of dry-run")
    parser.add_argument("--skip-rematch", action="store_true", help="Skip second-pass rematch before quality gates")
    parser.add_argument("--min-rps-match-rate", type=float, default=0.90)
    parser.add_argument("--max-rps-unmatched-rate", type=float, default=0.07)
    parser.add_argument("--max-rps-ambiguous-rate", type=float, default=0.03)
    args = parser.parse_args()

    countries = tuple(x.strip().upper() for x in args.countries.split(",") if x.strip())
    thresholds = Thresholds(
        min_rps_match_rate=args.min_rps_match_rate,
        max_rps_unmatched_rate=args.max_rps_unmatched_rate,
        max_rps_ambiguous_rate=args.max_rps_ambiguous_rate,
    )

    sp_summary = _run_sp_ingest(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        countries=countries,
        parse_existing_only=args.parse_existing_only,
    )

    rps_summary = enrich_from_rpscrape(
        db_path=Path(args.db_path),
        input_glob=args.rps_input_glob,
        min_confidence=args.min_confidence,
        countries=countries,
        dry_run=(not args.apply_rps),
    )

    rematch_summary: dict[str, Any] | None = None
    if not args.skip_rematch:
        rematch_summary = run_rematch(
            db_path=Path(args.db_path),
            input_glob=args.rps_input_glob,
            dry_run=(not args.apply_rps),
        )

    dq_ok = True
    dq_error: str | None = None
    try:
        window_start = date(args.start_year, args.start_month, 1)
        if args.end_month == 12:
            window_end = date(args.end_year + 1, 1, 1)
        else:
            window_end = date(args.end_year, args.end_month + 1, 1)
        # Inclusive max date for SQL filters.
        window_end = window_end - timedelta(days=1)

        run_all_checks(
            Path(args.db_path),
            min_race_date=window_start,
            max_race_date=window_end,
        )
    except Exception as exc:
        dq_ok = False
        dq_error = str(exc)

    coverage = _coverage_metrics(
        db_path=Path(args.db_path),
        countries=countries,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    gate = _evaluate_gates(rps_summary, dq_ok, thresholds, require_rps=args.require_rps)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "countries": list(countries),
        "window": {
            "start_year": args.start_year,
            "start_month": args.start_month,
            "end_year": args.end_year,
            "end_month": args.end_month,
        },
        "sp_summary": sp_summary,
        "rps_summary": rps_summary,
        "rematch_summary": rematch_summary,
        "dq": {
            "ok": dq_ok,
            "error": dq_error,
        },
        "coverage": coverage,
        "thresholds": {
            "min_rps_match_rate": thresholds.min_rps_match_rate,
            "max_rps_unmatched_rate": thresholds.max_rps_unmatched_rate,
            "max_rps_ambiguous_rate": thresholds.max_rps_ambiguous_rate,
        },
        "gate": gate,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"preflight_historical_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"preflight_log={log_path}")

    if not gate["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
