from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.nightly_odds_snapshot import run  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Betfair odds snapshots and persist to Parquet + DuckDB")
    parser.add_argument("--hours-ahead", type=int, default=6, help="How far ahead to fetch WIN markets")
    parser.add_argument(
        "--snapshot-label",
        type=str,
        default="evening_21h",
        choices=["evening_21h", "morning_09h", "near_off"],
        help="Snapshot label to persist",
    )
    args = parser.parse_args()

    run(snapshot_label=args.snapshot_label, hours_ahead=args.hours_ahead)
