from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.betfair_client import BetfairClient, BetfairCredentials  # noqa: E402
from ingestion.betfair_ingest import (  # noqa: E402
    insert_odds_snapshots_to_duckdb,
    normalize_betfair_price,
    save_race_snapshot_to_parquet,
)


DB_PATH = Path(os.getenv("DB_PATH", ROOT / "racing.duckdb"))
LOG_DIR = ROOT / "logs"


def _best_back_price(runner_book: dict) -> float | None:
    ex = runner_book.get("ex", {})
    backs = ex.get("availableToBack", [])
    if not backs:
        return None
    return float(backs[0].get("price"))


def _traded_volume(runner_book: dict) -> float:
    ex = runner_book.get("ex", {})
    traded = ex.get("tradedVolume", [])
    return float(sum(level.get("size", 0.0) for level in traded))


def _load_client() -> BetfairClient:
    creds = BetfairCredentials(
        app_key=os.environ["BETFAIR_APP_KEY"],
        username=os.environ["BETFAIR_USERNAME"],
        password=os.environ["BETFAIR_PASSWORD"],
        cert_file=Path(os.environ["BETFAIR_CERT_FILE"]).expanduser(),
        key_file=Path(os.environ["BETFAIR_KEY_FILE"]).expanduser(),
    )
    return BetfairClient(creds)


def _fetch_market_books_safe(client: BetfairClient, market_ids: list[str]) -> dict[str, dict]:
    """Fetch market books with automatic batch splitting on TOO_MUCH_DATA errors."""
    if not market_ids:
        return {}

    try:
        books = client.list_market_book(market_ids)
        return {book["marketId"]: book for book in books}
    except RuntimeError as exc:
        message = str(exc)
        if "TOO_MUCH_DATA" not in message or len(market_ids) == 1:
            raise

        mid = len(market_ids) // 2
        left = _fetch_market_books_safe(client, market_ids[:mid])
        right = _fetch_market_books_safe(client, market_ids[mid:])
        left.update(right)
        return left


def run(snapshot_label: str, hours_ahead: int = 24) -> dict:
    started = time.time()
    client = _load_client()
    snapshot_utc = datetime.now(timezone.utc)

    markets = client.list_win_markets(from_hours=0, to_hours=hours_ahead)
    market_ids = [m["marketId"] for m in markets]

    total_markets = 0
    total_rows = 0
    inserted_rows = 0
    skipped_rows = 0
    failed_rows = 0

    for i in range(0, len(market_ids), 40):
        batch_ids = market_ids[i:i + 40]
        market_books = _fetch_market_books_safe(client, batch_ids)

        for market in markets[i:i + 40]:
            market_id = market["marketId"]
            book = market_books.get(market_id)
            if not book:
                continue

            selection_to_runner = {runner["selectionId"]: runner for runner in market.get("runners", [])}
            market_start = datetime.fromisoformat(market["marketStartTime"].replace("Z", "+00:00"))
            minutes_to_off = int((market_start - snapshot_utc).total_seconds() // 60)

            rows = []
            for rb in book.get("runners", []):
                try:
                    odds = _best_back_price(rb)
                    if odds is None or odds <= 1.01:
                        skipped_rows += 1
                        continue
                    selection_id = rb["selectionId"]
                    meta = selection_to_runner.get(selection_id, {})
                    rows.append(
                        {
                            "race_id": f"bf_{market_id}",
                            "runner_id": f"bf_{market_id}_{selection_id}",
                            "runner_name": meta.get("runnerName"),
                            "source": "betfair_exchange",
                            "market_type": "WIN",
                            "snapshot_timestamp_utc": snapshot_utc,
                            "snapshot_label": snapshot_label,
                            "minutes_to_off": minutes_to_off,
                            "decimal_odds": odds,
                            "traded_volume_gbp": _traded_volume(rb),
                            "market_status": book.get("status"),
                            "decision_cutoff_utc": snapshot_utc,
                        }
                    )
                except Exception:
                    failed_rows += 1

            if not rows:
                continue

            save_race_snapshot_to_parquet(
                race_id=f"bf_{market_id}",
                rows=rows,
                snapshot_time_utc=snapshot_utc,
                snapshot_label=snapshot_label,
            )
            normalized = [normalize_betfair_price(row) for row in rows]
            inserted = insert_odds_snapshots_to_duckdb(DB_PATH, normalized)
            inserted_rows += inserted
            skipped_rows += max(0, len(rows) - inserted)
            total_markets += 1
            total_rows += len(rows)

    duration = round(time.time() - started, 3)
    result = {
        "snapshot_label": snapshot_label,
        "markets_captured": total_markets,
        "rows_captured": total_rows,
        "rows_inserted": inserted_rows,
        "rows_skipped": skipped_rows,
        "rows_failed": failed_rows,
        "duration_sec": duration,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_log_path = LOG_DIR / f"odds_snapshot_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    run_log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["log_file"] = str(run_log_path)
    print(result)
    return result


if __name__ == "__main__":
    run(snapshot_label="evening_21h", hours_ahead=24)
