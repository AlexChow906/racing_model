from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.rpscrape_enrich import enrich_from_rpscrape


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich SP spine for a single race_date")
    parser.add_argument("--date", type=str, required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--db-path", type=str, default="racing.duckdb")
    parser.add_argument("--input-glob", type=str, default="data/raw/rpscrape_repo/data/region/**/*.csv")
    parser.add_argument("--min-confidence", type=str, choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--countries", type=str, default="GB,IE", help="Comma-separated race countries to enrich")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_date = _parse_date(args.date)
    countries = tuple(x.strip().upper() for x in args.countries.split(",") if x.strip())

    result = enrich_from_rpscrape(
        db_path=Path(args.db_path),
        input_glob=args.input_glob,
        min_confidence=args.min_confidence,
        countries=countries,
        dry_run=args.dry_run,
        target_date=target_date,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
