from __future__ import annotations

import argparse
import sys
from datetime import date
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quality.checks import run_all_checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full baseline DQ report with an explicit race-date scope"
    )
    parser.add_argument("--db-path", type=str, default=str(ROOT / "racing.duckdb"))
    parser.add_argument("--min-year", type=int, default=2015)
    parser.add_argument("--min-month", type=int, default=1)
    parser.add_argument("--min-day", type=int, default=1)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    min_race_date = date(args.min_year, args.min_month, args.min_day)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    source_report = ROOT / "logs" / f"dq_report_{stamp}.txt"
    baseline_report = ROOT / "logs" / f"dq_report_full_baseline_{stamp}.txt"

    status = "PASS"
    error_text = ""
    try:
        run_all_checks(db_path=db_path, min_race_date=min_race_date, max_race_date=None)
    except Exception as exc:
        status = "FAIL"
        error_text = str(exc)

    if source_report.exists():
        body = source_report.read_text(encoding="utf-8")
    else:
        body = "Data Quality Report\n\nNo source report was generated.\n"

    header = [
        "FULL BASELINE SNAPSHOT",
        f"Status: {status}",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Scope: races on/after {min_race_date.isoformat()}",
        f"Source: {source_report.relative_to(ROOT)}",
    ]
    if error_text:
        header.append(f"Error: {error_text}")
    header.append("")

    baseline_report.write_text("\n".join(header) + body, encoding="utf-8")

    print(f"BASELINE_REPORT={baseline_report}")
    print(f"BASELINE_STATUS={status}")
    print(f"BASELINE_SCOPE_START={min_race_date.isoformat()}")


if __name__ == "__main__":
    main()
