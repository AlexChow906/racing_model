"""
Collect settled race results from Betfair public SP CSVs.

Downloads yesterday's (or a specified date's) SP CSV from promo.betfair.com,
parses it, and UPDATEs existing placeholder result rows in the DB with actual
sp_decimal, won, and finishing_position values.

Usage:
    python -m src.pipelines.collect_results --date yesterday
    python -m src.pipelines.collect_results --date 2026-04-03
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db
from ingestion.normalise import slugify
from pipelines.helpers import resolve_date

SP_INDEX_URL = "https://promo.betfair.com/betfairsp/prices"
SP_FILE_PATTERN = re.compile(r'href="(/betfairsp/prices/([^"]+\.csv))"', re.IGNORECASE)
SP_DATE_PATTERN = re.compile(r"(\d{2})(\d{2})(\d{4})\.csv$", re.IGNORECASE)
INCLUDE_TOKENS = ("pricesukwin", "pricesirewin")
_ALPHA_ONLY = re.compile(r"[^a-z0-9]")


def _parse_date_from_filename(filename: str) -> date | None:
    match = SP_DATE_PATTERN.search(filename)
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _download_text(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def find_sp_csvs_for_date(target_date: date) -> list[dict[str, str]]:
    file_date_needed = target_date + timedelta(days=1)
    html = _download_text(SP_INDEX_URL)
    matches = []
    seen: set[str] = set()
    for rel_path, filename in SP_FILE_PATTERN.findall(html):
        if filename in seen:
            continue
        seen.add(filename)
        file_date = _parse_date_from_filename(filename)
        if file_date != file_date_needed:
            continue
        if not any(tok in filename.lower() for tok in INCLUDE_TOKENS):
            continue
        matches.append({
            "filename": filename,
            "url": f"https://promo.betfair.com{rel_path}",
        })
    return matches


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _compact(name: str) -> str:
    return _ALPHA_ONLY.sub("", name.lower())


def parse_sp_csv(csv_text: str, source_filename: str) -> list[dict[str, Any]]:
    normalised = csv_text.replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.DictReader(io.StringIO(normalised))
    rows = []
    for raw in reader:
        if not raw:
            continue
        lowered = {str(k).lower(): v for k, v in raw.items()}
        selection_name = str(lowered.get("selection_name") or "").strip()
        if not selection_name:
            continue

        if "place" in source_filename.lower():
            continue

        horse_id = slugify(selection_name)
        if not horse_id:
            continue

        won = str(lowered.get("win_lose", "0")).strip() == "1"
        bsp = _safe_float(lowered.get("bsp"))

        rows.append({
            "compact_horse": _compact(horse_id),
            "sp_decimal": bsp,
            "won": won,
            "finishing_position": 1 if won else None,
        })
    return rows


def update_results_in_db(rows: list[dict[str, Any]], target_date: date, db_path: str | Path) -> dict[str, int]:
    if not rows:
        return {"matched": 0, "updated": 0, "not_found": 0}

    con = get_db(str(db_path))

    db_results = con.execute(
        """SELECT res.result_id, res.horse_id
           FROM results res
           JOIN races ra ON res.race_id = ra.race_id
           WHERE ra.race_date = ?""",
        [target_date],
    ).fetchall()

    lookup: dict[str, str] = {}
    for result_id, horse_id in db_results:
        lookup[_compact(horse_id)] = result_id

    updated = 0
    not_found = 0
    already = set()

    for row in rows:
        key = row["compact_horse"]
        result_id = lookup.get(key)

        if result_id and result_id not in already:
            con.execute(
                """UPDATE results
                   SET sp_decimal = ?,
                       won = ?,
                       finishing_position = COALESCE(?, finishing_position),
                       finishing_position_raw = ?
                   WHERE result_id = ?""",
                [
                    row["sp_decimal"],
                    row["won"],
                    row["finishing_position"],
                    "WIN" if row["won"] else "LOSE",
                    result_id,
                ],
            )
            updated += 1
            already.add(result_id)
        else:
            not_found += 1

    con.close()
    return {"matched": len(rows), "updated": updated, "not_found": not_found}


def main():
    parser = argparse.ArgumentParser(description="Collect settled results from Betfair SP CSVs")
    parser.add_argument("--date", type=str, required=True, help="Date: YYYY-MM-DD, 'yesterday', or 'today'")
    parser.add_argument("--db", type=str, default=None, help="DB path override")
    args = parser.parse_args()

    target_date = resolve_date(args.date)
    db_path = args.db or str(ROOT / "racing.duckdb")
    print(f"Collecting results for {target_date}", flush=True)

    print("Fetching SP CSV index from promo.betfair.com...", flush=True)
    csv_files = find_sp_csvs_for_date(target_date)

    if not csv_files:
        print(f"  No SP CSVs found for {target_date} (may not be available yet)")
        return

    print(f"  Found {len(csv_files)} file(s): {[f['filename'] for f in csv_files]}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for file_info in csv_files:
        print(f"  Downloading {file_info['filename']}...", flush=True)
        csv_text = _download_text(file_info["url"])
        rows = parse_sp_csv(csv_text, file_info["filename"])
        print(f"    Parsed {len(rows)} runners", flush=True)
        all_rows.extend(rows)

    print(f"Updating DB with {len(all_rows)} results...", flush=True)
    stats = update_results_in_db(all_rows, target_date, db_path)
    print(f"  Updated: {stats['updated']}, Not in DB: {stats['not_found']}", flush=True)


if __name__ == "__main__":
    main()
