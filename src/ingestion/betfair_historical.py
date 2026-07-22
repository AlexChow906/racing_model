from __future__ import annotations

import argparse
import csv
import calendar
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import requests
import betfairlightweight
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.betfair_client import BetfairClient, BetfairCredentials
from ingestion.normalise import (
    canonical_race_id_from_market,
    canonical_runner_id,
    decision_cutoff_for_off_time,
    parse_utc,
    slugify,
    upsert_ignore,
)

RAW_DIR = ROOT / "data" / "raw" / "betfair_historical"
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "racing.duckdb"))
SP_HISTORY_INDEX_URL = "https://promo.betfair.com/betfairsp/prices"
SP_FILE_PATTERN = re.compile(r'href="(/betfairsp/prices/([^"]+\.csv))"', re.IGNORECASE)
SP_DATE_PATTERN = re.compile(r"(\d{2})(\d{2})(\d{4})\.csv$", re.IGNORECASE)


def month_iter(start_year: int, start_month: int, end_year: int, end_month: int) -> Iterable[tuple[int, int]]:
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            month = 1
            year += 1


def monthly_zip_path(year: int, month: int) -> Path:
    return RAW_DIR / f"{year:04d}" / f"{month:02d}" / f"betfair_historical_{year:04d}_{month:02d}.zip"


def _historic_client_from_env() -> betfairlightweight.APIClient:
    cert_file = Path(os.environ.get("BETFAIR_CERT_FILE", "")).expanduser()
    cert_dir = cert_file.parent
    client = betfairlightweight.APIClient(
        username=os.environ.get("BETFAIR_USERNAME", ""),
        password=os.environ.get("BETFAIR_PASSWORD", ""),
        app_key=os.environ.get("BETFAIR_APP_KEY", ""),
        certs=str(cert_dir),
    )
    client.login()
    return client


def _fetch_historic_file_entries(year: int, month: int) -> list[dict[str, Any]]:
    client = _historic_client_from_env()
    last_day = calendar.monthrange(year, month)[1]
    response = client.historic.get_file_list(
        sport="Horse Racing",
        plan="Basic Plan",
        from_day=f"{1:02d}",
        from_month=f"{month:02d}",
        from_year=f"{year:04d}",
        to_day=f"{last_day:02d}",
        to_month=f"{month:02d}",
        to_year=f"{year:04d}",
        market_types_collection="WIN",
        countries_collection="GB,IE",
    )

    if isinstance(response, dict):
        for key in ["files", "FileList", "result", "Results"]:
            value = response.get(key)
            if isinstance(value, list):
                return value
        if "filePath" in response:
            return [response]
    if isinstance(response, list):
        return response
    return []


def _download_historic_entries(year: int, month: int, entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "year": year,
            "month": month,
            "status": "no_remote_files",
            "downloaded": 0,
        }

    client = _historic_client_from_env()
    target_dir = RAW_DIR / f"{year:04d}" / f"{month:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for entry in entries:
        file_path = entry.get("filePath") or entry.get("path") or entry.get("file")
        if not file_path:
            failed += 1
            continue

        expected_name = Path(str(file_path)).name
        if (target_dir / expected_name).exists():
            skipped += 1
            continue

        try:
            client.historic.download_file(str(file_path), store_directory=str(target_dir))
            downloaded += 1
        except Exception:
            failed += 1

    return {
        "year": year,
        "month": month,
        "status": "downloaded_via_historic_api",
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=60, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    out.write(chunk)


def download_month(year: int, month: int, base_url: str) -> dict[str, Any]:
    try:
        entries = _fetch_historic_file_entries(year, month)
        return _download_historic_entries(year, month, entries)
    except Exception as historic_exc:
        # Fallback to direct URL pattern if historic endpoint errors for this account/session.
        print(f"historic_api_fallback: {year:04d}-{month:02d} reason={historic_exc}")

    zip_path = monthly_zip_path(year, month)
    if zip_path.exists():
        return {
            "year": year,
            "month": month,
            "status": "skipped_existing",
            "file": str(zip_path),
            "bytes": zip_path.stat().st_size,
        }

    url = f"{base_url.rstrip('/')}/{year:04d}/{month:02d}.zip"
    started = time.time()
    _download_file(url, zip_path)
    duration = time.time() - started

    return {
        "year": year,
        "month": month,
        "status": "downloaded",
        "file": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "duration_sec": round(duration, 3),
    }


def _load_records_from_zip(zip_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith(".json") or lower.endswith(".jsonl"):
                with zf.open(name) as fh:
                    for raw in fh:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                            records.append(payload)
                        except json.JSONDecodeError:
                            continue
            elif lower.endswith(".csv"):
                with zf.open(name) as fh:
                    text = fh.read().decode("utf-8", errors="ignore").splitlines()
                    for row in csv.DictReader(text):
                        records.append(row)
    return records


def _date_range_from_args(start_year: int, start_month: int, end_year: int, end_month: int) -> tuple[date, date]:
    start_date = date(start_year, start_month, 1)
    end_day = calendar.monthrange(end_year, end_month)[1]
    end_date = date(end_year, end_month, end_day)
    return start_date, end_date


def _parse_sp_date_from_filename(filename: str) -> date | None:
    match = SP_DATE_PATTERN.search(filename)
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _list_sp_history_files() -> list[dict[str, Any]]:
    response = requests.get(SP_HISTORY_INDEX_URL, timeout=60)
    response.raise_for_status()
    html = response.text

    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    for rel_path, filename in SP_FILE_PATTERN.findall(html):
        if filename in seen:
            continue
        seen.add(filename)
        file_date = _parse_sp_date_from_filename(filename)
        if file_date is None:
            continue
        files.append(
            {
                "filename": filename,
                "relative_path": rel_path,
                "url": f"https://promo.betfair.com{rel_path}",
                "file_date": file_date,
            }
        )
    return files


def _download_sp_history_files(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    include_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    start_date, end_date = _date_range_from_args(start_year, start_month, end_year, end_month)
    include_tokens_lc = tuple(token.lower() for token in include_tokens)

    candidates = _list_sp_history_files()
    selected = [
        item
        for item in candidates
        if start_date <= item["file_date"] <= end_date
        and any(token in item["filename"].lower() for token in include_tokens_lc)
    ]

    results: list[dict[str, Any]] = []
    for item in sorted(selected, key=lambda x: (x["file_date"], x["filename"])):
        file_date: date = item["file_date"]
        dest = RAW_DIR / "sp_history" / f"{file_date.year:04d}" / f"{file_date.month:02d}" / item["filename"]
        if dest.exists():
            results.append(
                {
                    "status": "skipped_existing",
                    "file": str(dest),
                    "source_url": item["url"],
                }
            )
            continue

        started = time.time()
        _download_file(item["url"], dest)
        results.append(
            {
                "status": "downloaded",
                "file": str(dest),
                "source_url": item["url"],
                "bytes": dest.stat().st_size,
                "duration_sec": round(time.time() - started, 3),
            }
        )
    return results


def _parse_sp_event_dt(value: Any) -> datetime:
    text = str(value or "").strip()
    # Betfair SP CSV uses dd-mm-YYYY HH:MM and no timezone; treat as UTC for now.
    dt = datetime.strptime(text, "%d-%m-%Y %H:%M")
    return dt.replace(tzinfo=timezone.utc)


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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _country_from_filename(filename: str) -> str:
    name = filename.lower()
    if "pricesuk" in name:
        return "GB"
    if "pricesire" in name:
        return "IE"
    if "pricesaus" in name:
        return "AUS"
    if "pricesusa" in name:
        return "USA"
    return "GB"


def _market_type_from_filename(filename: str) -> str:
    name = filename.lower()
    if "place" in name:
        return "PLACE"
    return "WIN"


def _sp_row_to_records(raw: dict[str, Any], source_filename: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    event_id = str(raw.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("missing event_id")

    selection_id = str(raw.get("selection_id") or "").strip()
    selection_name = str(raw.get("selection_name") or "").strip()
    if not selection_id or not selection_name:
        raise ValueError("missing selection data")

    market_type = _market_type_from_filename(source_filename)
    # SP CSVs omit the "1." exchange prefix that the Betfair API includes.
    if not event_id.startswith("1."):
        event_id = f"1.{event_id}"
    race_id = f"bfsp_{event_id}_{market_type.lower()}"
    scheduled_off_utc = _parse_sp_event_dt(raw.get("event_dt"))
    decision_cutoff_utc = decision_cutoff_for_off_time(scheduled_off_utc)
    race_date = scheduled_off_utc.date()

    menu_hint = str(raw.get("menu_hint") or "unknown_course").strip()
    course_id = slugify(menu_hint) or "unknown_course"
    horse_id = slugify(selection_name) or f"sel_{selection_id}"
    runner_id = canonical_runner_id(race_id, horse_id)

    finishing_position = 1 if str(raw.get("win_lose", "0")).strip() == "1" else None
    won = finishing_position == 1

    race = {
        "race_id": race_id,
        "source_race_id": event_id,
        "course_id": course_id,
        "course_name": menu_hint,
        "race_date": race_date,
        "scheduled_off_utc": scheduled_off_utc,
        "distance_furlongs": None,
        "going_code": None,
        "going_description": None,
        "race_type": str(raw.get("event_name") or "").strip() or None,
        "race_class": None,
        "race_grade": None,
        "prize_money_gbp": None,
        "field_size": None,
        "is_handicap": None,
        "surface": None,
        "country": _country_from_filename(source_filename),
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    runner = {
        "runner_id": runner_id,
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_name": selection_name,
        "trainer_id": None,
        "trainer_name": None,
        "jockey_id": None,
        "jockey_name": None,
        "draw": None,
        "weight_lbs": None,
        "age": None,
        "official_rating": None,
        "headgear": None,
        "headgear_first_time": False,
        "days_since_last_run": None,
        "career_runs": None,
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    result = {
        "result_id": f"{runner_id}_res",
        "race_id": race_id,
        "runner_id": runner_id,
        "horse_id": horse_id,
        "finishing_position": finishing_position,
        "finishing_position_raw": "WIN" if won else "LOSE",
        "btn_lengths": None,
        "official_time_secs": None,
        "sp_decimal": _safe_float(raw.get("bsp")),
        "won": won,
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    history = {
        "history_id": f"{runner_id}_hist",
        "horse_id": horse_id,
        "horse_name": selection_name,
        "race_id": race_id,
        "race_date": race_date,
        "scheduled_off_utc": scheduled_off_utc,
        "course_id": course_id,
        "distance_furlongs": None,
        "going_code": None,
        "race_type": race["race_type"],
        "race_class": None,
        "is_handicap": None,
        "field_size": None,
        "finishing_position": finishing_position,
        "won": won,
        "btn_lengths": None,
        "trainer_id": None,
        "jockey_id": None,
        "weight_lbs": None,
        "official_rating": None,
        "headgear": None,
        "days_since_prev_run": None,
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }
    return race, runner, result, history


def parse_sp_csv_to_duckdb(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"status": "missing_csv", "file": str(csv_path)}

    done_marker = csv_path.with_suffix(".csv.done")
    if done_marker.exists():
        return {"status": "already_parsed", "file": str(csv_path)}

    started = time.time()
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.DictReader(io.StringIO(normalised))

    races: list[dict[str, Any]] = []
    runners: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    horse_history: list[dict[str, Any]] = []
    failed = 0

    for raw in reader:
        if not raw:
            continue
        try:
            lowered = {str(k).lower(): v for k, v in raw.items()}
            race, runner, result, history = _sp_row_to_records(lowered, csv_path.name)
            races.append(race)
            runners.append(runner)
            results.append(result)
            horse_history.append(history)
        except Exception:
            failed += 1

    if not runners:
        return {
            "file": str(csv_path),
            "status": "parse_failed_no_rows",
            "raw_records": 0,
            "failed_records": failed,
            "inserted": {
                "races": 0,
                "runners": 0,
                "results": 0,
                "horse_history": 0,
            },
            "duration_sec": round(time.time() - started, 3),
        }

    field_sizes: dict[str, int] = {}
    for runner in runners:
        rid = runner["race_id"]
        field_sizes[rid] = field_sizes.get(rid, 0) + 1
    for race in races:
        race["field_size"] = field_sizes.get(race["race_id"])
    for hist in horse_history:
        hist["field_size"] = field_sizes.get(hist["race_id"])

    con = duckdb.connect(str(DB_PATH))
    try:
        inserted = {
            "races": upsert_ignore(con, "races", races, [
                "race_id", "source_race_id", "course_id", "course_name", "race_date", "scheduled_off_utc",
                "distance_furlongs", "going_code", "going_description", "race_type", "race_class", "race_grade",
                "prize_money_gbp", "field_size", "is_handicap", "surface", "country", "event_timestamp_utc",
                "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "runners": upsert_ignore(con, "runners", runners, [
                "runner_id", "race_id", "horse_id", "horse_name", "trainer_id", "trainer_name", "jockey_id",
                "jockey_name", "draw", "weight_lbs", "age", "official_rating", "headgear", "headgear_first_time",
                "days_since_last_run", "career_runs", "event_timestamp_utc", "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "results": upsert_ignore(con, "results", results, [
                "result_id", "race_id", "runner_id", "horse_id", "finishing_position", "finishing_position_raw",
                "btn_lengths", "official_time_secs", "sp_decimal", "won", "event_timestamp_utc",
                "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "horse_history": upsert_ignore(con, "horse_history", horse_history, [
                "history_id", "horse_id", "horse_name", "race_id", "race_date", "scheduled_off_utc", "course_id",
                "distance_furlongs", "going_code", "race_type", "race_class", "is_handicap", "field_size",
                "finishing_position", "won", "btn_lengths", "trainer_id", "jockey_id", "weight_lbs",
                "official_rating", "headgear", "days_since_prev_run", "event_timestamp_utc", "decision_cutoff_utc",
                "ingest_timestamp_utc"
            ]),
        }
    finally:
        con.close()

    done_marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return {
        "file": str(csv_path),
        "status": "parsed",
        "raw_records": len(runners),
        "failed_records": failed,
        "inserted": inserted,
        "duration_sec": round(time.time() - started, 3),
    }


def _to_race_runner_result_records(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    market_id = str(raw.get("marketId") or raw.get("market_id") or raw.get("id") or "")
    if not market_id:
        raise ValueError("missing market id")

    race_id = canonical_race_id_from_market(market_id)
    scheduled_off_utc = parse_utc(raw.get("marketStartTime") or raw.get("scheduled_off_utc") or raw.get("off_time"))
    decision_cutoff_utc = decision_cutoff_for_off_time(scheduled_off_utc)
    race_date = scheduled_off_utc.date()

    course_name = str(raw.get("venue") or raw.get("course_name") or raw.get("eventName") or "unknown")
    course_id = slugify(course_name) or "unknown_course"
    horse_name = str(raw.get("runnerName") or raw.get("horse_name") or "unknown_horse")
    horse_id = slugify(horse_name) or "unknown_horse"
    runner_id = canonical_runner_id(race_id, horse_id)

    trainer_name = raw.get("trainer") or raw.get("trainer_name")
    jockey_name = raw.get("jockey") or raw.get("jockey_name")

    finishing_position = raw.get("finishingPosition") or raw.get("finishing_position")
    finishing_position_raw = str(raw.get("status") or finishing_position or "")
    pos_int = int(finishing_position) if str(finishing_position).isdigit() else None
    won = pos_int == 1

    race = {
        "race_id": race_id,
        "source_race_id": market_id,
        "course_id": course_id,
        "course_name": course_name,
        "race_date": race_date,
        "scheduled_off_utc": scheduled_off_utc,
        "distance_furlongs": raw.get("distance_furlongs"),
        "going_code": raw.get("going_code"),
        "going_description": raw.get("going_description"),
        "race_type": raw.get("race_type"),
        "race_class": raw.get("race_class"),
        "race_grade": raw.get("race_grade"),
        "prize_money_gbp": raw.get("prize_money_gbp"),
        "field_size": raw.get("numberOfRunners") or raw.get("field_size"),
        "is_handicap": raw.get("is_handicap"),
        "surface": raw.get("surface"),
        "country": raw.get("country") or "GB",
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    runner = {
        "runner_id": runner_id,
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_name": horse_name,
        "trainer_id": slugify(str(trainer_name)) if trainer_name else None,
        "trainer_name": trainer_name,
        "jockey_id": slugify(str(jockey_name)) if jockey_name else None,
        "jockey_name": jockey_name,
        "draw": raw.get("draw"),
        "weight_lbs": raw.get("weight_lbs"),
        "age": raw.get("age"),
        "official_rating": raw.get("official_rating"),
        "headgear": raw.get("headgear"),
        "headgear_first_time": bool(raw.get("headgear_first_time", False)),
        "days_since_last_run": raw.get("days_since_last_run"),
        "career_runs": raw.get("career_runs"),
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    result = {
        "result_id": f"{runner_id}_res",
        "race_id": race_id,
        "runner_id": runner_id,
        "horse_id": horse_id,
        "finishing_position": pos_int,
        "finishing_position_raw": finishing_position_raw,
        "btn_lengths": raw.get("btn_lengths"),
        "official_time_secs": raw.get("official_time_secs"),
        "sp_decimal": raw.get("sp_decimal") or raw.get("bsp") or raw.get("sp"),
        "won": won,
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    history = {
        "history_id": f"{runner_id}_hist",
        "horse_id": horse_id,
        "horse_name": horse_name,
        "race_id": race_id,
        "race_date": race_date,
        "scheduled_off_utc": scheduled_off_utc,
        "course_id": course_id,
        "distance_furlongs": raw.get("distance_furlongs"),
        "going_code": raw.get("going_code"),
        "race_type": raw.get("race_type"),
        "race_class": raw.get("race_class"),
        "is_handicap": raw.get("is_handicap"),
        "field_size": raw.get("numberOfRunners") or raw.get("field_size"),
        "finishing_position": pos_int,
        "won": won,
        "btn_lengths": raw.get("btn_lengths"),
        "trainer_id": slugify(str(trainer_name)) if trainer_name else None,
        "jockey_id": slugify(str(jockey_name)) if jockey_name else None,
        "weight_lbs": raw.get("weight_lbs"),
        "official_rating": raw.get("official_rating"),
        "headgear": raw.get("headgear"),
        "days_since_prev_run": raw.get("days_since_last_run"),
        "event_timestamp_utc": scheduled_off_utc,
        "decision_cutoff_utc": decision_cutoff_utc,
        "ingest_timestamp_utc": datetime.now(timezone.utc),
    }

    return race, runner, result, history


def parse_month_to_duckdb(year: int, month: int) -> dict[str, Any]:
    zip_path = monthly_zip_path(year, month)
    return parse_zip_to_duckdb(zip_path)


def parse_zip_to_duckdb(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists():
        return {"status": "missing_zip", "file": str(zip_path)}

    done_marker = zip_path.with_suffix(".done")
    if done_marker.exists():
        return {"status": "already_parsed", "file": str(zip_path)}

    started = time.time()
    records = _load_records_from_zip(zip_path)

    races: list[dict[str, Any]] = []
    runners: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    horse_history: list[dict[str, Any]] = []
    failed = 0

    for raw in records:
        try:
            race, runner, result, history = _to_race_runner_result_records(raw)
            races.append(race)
            runners.append(runner)
            results.append(result)
            if history:
                horse_history.append(history)
        except Exception:
            failed += 1

    con = duckdb.connect(str(DB_PATH))
    try:
        inserted = {
            "races": upsert_ignore(con, "races", races, [
                "race_id", "source_race_id", "course_id", "course_name", "race_date", "scheduled_off_utc",
                "distance_furlongs", "going_code", "going_description", "race_type", "race_class", "race_grade",
                "prize_money_gbp", "field_size", "is_handicap", "surface", "country", "event_timestamp_utc",
                "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "runners": upsert_ignore(con, "runners", runners, [
                "runner_id", "race_id", "horse_id", "horse_name", "trainer_id", "trainer_name", "jockey_id",
                "jockey_name", "draw", "weight_lbs", "age", "official_rating", "headgear", "headgear_first_time",
                "days_since_last_run", "career_runs", "event_timestamp_utc", "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "results": upsert_ignore(con, "results", results, [
                "result_id", "race_id", "runner_id", "horse_id", "finishing_position", "finishing_position_raw",
                "btn_lengths", "official_time_secs", "sp_decimal", "won", "event_timestamp_utc",
                "decision_cutoff_utc", "ingest_timestamp_utc"
            ]),
            "horse_history": upsert_ignore(con, "horse_history", horse_history, [
                "history_id", "horse_id", "horse_name", "race_id", "race_date", "scheduled_off_utc", "course_id",
                "distance_furlongs", "going_code", "race_type", "race_class", "is_handicap", "field_size",
                "finishing_position", "won", "btn_lengths", "trainer_id", "jockey_id", "weight_lbs",
                "official_rating", "headgear", "days_since_prev_run", "event_timestamp_utc", "decision_cutoff_utc",
                "ingest_timestamp_utc"
            ]),
        }
    finally:
        con.close()

    done_marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    duration = time.time() - started
    return {
        "file": str(zip_path),
        "status": "parsed",
        "raw_records": len(records),
        "failed_records": failed,
        "inserted": inserted,
        "duration_sec": round(duration, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse Betfair historical data")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--download-base-url", type=str, default="https://historicdata.betfair.com")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument(
        "--use-sp-history",
        action="store_true",
        help="Use public Betfair SP CSV history instead of historic ZIP endpoints",
    )
    parser.add_argument(
        "--sp-include",
        type=str,
        default="pricesukwin,pricesirewin",
        help="Comma-separated filename tokens used to select SP files",
    )
    parser.add_argument(
        "--scan-existing-sp-csvs",
        action="store_true",
        help="Parse all existing SP CSV files under data/raw/betfair_historical/sp_history recursively",
    )
    parser.add_argument(
        "--scan-existing-zips",
        action="store_true",
        help="Parse all existing ZIP files under data/raw/betfair_historical recursively",
    )
    parser.add_argument(
        "--strict-download",
        action="store_true",
        help="Fail fast on first download error instead of continuing to remaining months",
    )
    args = parser.parse_args()

    if not args.use_sp_history:
        # Require a valid Betfair session before non-public historical processing starts.
        creds = BetfairCredentials(
            app_key=os.environ.get("BETFAIR_APP_KEY", ""),
            username=os.environ.get("BETFAIR_USERNAME", ""),
            password=os.environ.get("BETFAIR_PASSWORD", ""),
            cert_file=Path(os.environ.get("BETFAIR_CERT_FILE", "")).expanduser(),
            key_file=Path(os.environ.get("BETFAIR_KEY_FILE", "")).expanduser(),
        )
        if not all([creds.app_key, creds.username, creds.password, str(creds.cert_file), str(creds.key_file)]):
            raise RuntimeError("Missing BETFAIR_* credentials for historical ingestion")

        auth_client = BetfairClient(creds)
        auth_client.login()
        print("Betfair session validation: SUCCESS")
    else:
        print("Using public Betfair SP history mode (no API auth required)")

    run_started = time.time()
    download_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []

    if args.use_sp_history:
        include_tokens = tuple(token.strip() for token in args.sp_include.split(",") if token.strip())
        if args.scan_existing_sp_csvs and not args.download_only:
            csv_files = sorted((RAW_DIR / "sp_history").rglob("*.csv"))
            for csv_file in csv_files:
                parse_info = parse_sp_csv_to_duckdb(csv_file)
                parse_rows.append(parse_info)
                print(f"parse {csv_file.name}: {parse_info['status']}")
        else:
            if not args.parse_only:
                download_rows = _download_sp_history_files(
                    args.start_year,
                    args.start_month,
                    args.end_year,
                    args.end_month,
                    include_tokens=include_tokens,
                )
                for row in download_rows:
                    print(f"download sp: {row['status']} {Path(row['file']).name}")

            if not args.download_only:
                csv_files = sorted((RAW_DIR / "sp_history").rglob("*.csv"))
                start_date, end_date = _date_range_from_args(
                    args.start_year, args.start_month, args.end_year, args.end_month
                )
                for csv_file in csv_files:
                    csv_date = _parse_sp_date_from_filename(csv_file.name)
                    if csv_date is None or csv_date < start_date or csv_date > end_date:
                        continue
                    if include_tokens and not any(token.lower() in csv_file.name.lower() for token in include_tokens):
                        continue
                    parse_info = parse_sp_csv_to_duckdb(csv_file)
                    parse_rows.append(parse_info)
                    print(f"parse {csv_file.name}: {parse_info['status']}")
    elif args.scan_existing_zips and not args.download_only:
        zip_files = sorted(RAW_DIR.rglob("*.zip"))
        for zip_file in zip_files:
            parse_info = parse_zip_to_duckdb(zip_file)
            parse_rows.append(parse_info)
            print(f"parse {zip_file.name}: {parse_info['status']}")
    else:
        for year, month in month_iter(args.start_year, args.start_month, args.end_year, args.end_month):
            if not args.parse_only:
                try:
                    download_info = download_month(year, month, args.download_base_url)
                except Exception as exc:
                    download_info = {
                        "year": year,
                        "month": month,
                        "status": "download_failed",
                        "error": str(exc),
                    }
                    if args.strict_download:
                        raise
                download_rows.append(download_info)
                print(f"download {year:04d}-{month:02d}: {download_info['status']}")

            if not args.download_only:
                parse_info = parse_month_to_duckdb(year, month)
                parse_rows.append(parse_info)
                print(f"parse {year:04d}-{month:02d}: {parse_info['status']}")

    log_path = ROOT / "logs" / f"historical_ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - run_started, 3),
        "download": download_rows,
        "parse": parse_rows,
    }
    log_path.write_text(json.dumps(log_payload, indent=2, default=str), encoding="utf-8")
    print(f"log written: {log_path}")


if __name__ == "__main__":
    main()
