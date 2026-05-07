from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.betfair_ingest import write_morning_alerts_csv
from pipelines.nightly_odds_snapshot import DB_PATH, run


if __name__ == "__main__":
    run(snapshot_label="morning_09h", hours_ahead=24)
    alerts = write_morning_alerts_csv(DB_PATH, datetime.now(timezone.utc))
    print(f"Morning alerts written: {alerts}")
