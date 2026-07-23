#!/usr/bin/env python
"""
Scrape today's racecards from Racing Post for model enrichment.

Extracts going, distance, class, prize money, handicap status, surface,
race grade, and runner sex from RP's __NEXT_DATA__ JSON.

Usage:
    python scripts/scrape_racecards.py --date today
    python scripts/scrape_racecards.py --date 2026-07-24
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RPSCRAPE_SCRIPTS = PROJECT_DIR / "data" / "raw" / "rpscrape_repo" / "scripts"
sys.path.insert(0, str(RPSCRAPE_SCRIPTS))

from dotenv import load_dotenv
from lxml import html
from utils.network import NetworkClient

RACECARDS_DIR = PROJECT_DIR / "data" / "raw" / "rpscrape_repo" / "racecards"
BASE_URL = "https://www.racingpost.com"

AW_GOINGS = {"slow", "standard", "standard to fast", "standard to slow"}


def _resolve_date(date_str: str) -> date:
    if date_str == "today":
        return date.today()
    if date_str == "tomorrow":
        return date.today() + timedelta(days=1)
    return date.fromisoformat(date_str)


def _parse_going(going_details: str) -> str | None:
    if not going_details:
        return None
    text = going_details.split("(")[0].strip()
    return text.title() if text else None


def _parse_distance_furlongs(display_distance: str) -> float | None:
    if not display_distance:
        return None
    m = re.match(r"(?:(\d+)m\s*)?(?:(\d+)f)?", display_distance)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    miles = int(m.group(1) or 0)
    furlongs = int(m.group(2) or 0)
    total = miles * 8 + furlongs
    return float(total) if total > 0 else None


def _get_surface(going: str | None) -> str | None:
    if not going:
        return None
    return "AW" if going.lower() in AW_GOINGS else "Turf"


def _parse_grade(race_title: str) -> str | None:
    m = re.search(r"\b((?:Group|Grade)\s+\d)\b", race_title, re.IGNORECASE)
    if m:
        return m.group(1).title()
    if re.search(r"\bListed\b", race_title, re.IGNORECASE):
        return "Listed"
    return None


def _is_handicap(race: dict) -> bool:
    if race.get("ratingBand"):
        return True
    title = (race.get("raceTitle") or "").lower()
    return "handicap" in title or "hcap" in title


def _extract_next_data(content: bytes) -> dict | None:
    doc = html.fromstring(content)
    scripts = doc.xpath('//script[@id="__NEXT_DATA__"]')
    if not scripts:
        return None
    return json.loads(scripts[0].text_content())


def _fetch_with_retry(
    client: NetworkClient, url: str, max_retries: int = 3, base_delay: float = 5.0,
) -> bytes | None:
    for attempt in range(1, max_retries + 1):
        status, response = client.get(url)
        if status == 200:
            return response.content
        if status == 429:
            retry_after = base_delay * attempt
            try:
                retry_after = float(response.headers.get("Retry-After", retry_after))
            except (ValueError, TypeError):
                pass
            print(f"  429 rate limited, waiting {retry_after:.0f}s "
                  f"(attempt {attempt}/{max_retries})", flush=True)
            time.sleep(retry_after)
            continue
        print(f"  HTTP {status} for {url}", flush=True)
        if status in (403, 401):
            print("  Auth may have expired — refresh RP cookies in .env", flush=True)
            return None
        if attempt < max_retries:
            time.sleep(base_delay)
    print(f"  Failed after {max_retries} retries: {url}", flush=True)
    return None


def scrape_racecards(target_date: date, client: NetworkClient) -> list[dict]:
    listing_url = f"{BASE_URL}/racecards/{target_date}"
    print(f"  Fetching listing: {listing_url}", flush=True)
    content = _fetch_with_retry(client, listing_url)
    if not content:
        return []

    data = _extract_next_data(content)
    if not data:
        print("  No __NEXT_DATA__ found in listing page", flush=True)
        return []

    meetings = (data.get("props", {}).get("pageProps", {})
                .get("initialState", {}).get("raceCards", {}).get("meetings", []))
    if not meetings:
        print("  No meetings found", flush=True)
        return []

    races_out: list[dict] = []
    race_urls: list[str] = []

    for meeting in meetings:
        course = meeting.get("courseName", "")
        going_raw = meeting.get("goingDetails", "")
        going = _parse_going(going_raw)
        surface = _get_surface(going)

        for race in meeting.get("races", []):
            if race.get("isAbandoned"):
                continue

            raw_class = race.get("raceClass")
            race_entry = {
                "course": course,
                "off_time": race.get("raceStart", ""),
                "going": going,
                "distance_f": _parse_distance_furlongs(race.get("displayDistance", "")),
                "race_class": int(raw_class) if raw_class is not None else None,
                "is_handicap": _is_handicap(race),
                "surface": surface,
                "race_grade": _parse_grade(race.get("raceTitle", "")),
                "prize_money_gbp": None,
                "runners": [],
            }
            races_out.append(race_entry)
            race_urls.append(race.get("raceUrl", ""))

    print(f"  Found {len(races_out)} races across {len(meetings)} meetings", flush=True)
    print(f"  Fetching individual race pages for prize money + runners...", flush=True)

    for i, (race_entry, race_url) in enumerate(zip(races_out, race_urls)):
        if not race_url:
            continue

        time.sleep(1)
        content = _fetch_with_retry(client, f"{BASE_URL}{race_url}")
        if not content:
            continue

        page_data = _extract_next_data(content)
        if not page_data:
            continue

        race_page = (page_data.get("props", {}).get("pageProps", {})
                     .get("initialState", {}).get("racePage", {}).get("data", {}))

        race_info = race_page.get("race", {})
        prizes = race_info.get("prizes", [])
        if prizes:
            winner_prize = prizes[0].get("prize_sterling")
            if winner_prize is not None:
                race_entry["prize_money_gbp"] = float(winner_prize)

        runners_data = race_page.get("runners", {})
        if isinstance(runners_data, dict):
            runners_data = list(runners_data.values())
        for runner in runners_data:
            if not isinstance(runner, dict) or runner.get("nonRunner"):
                continue
            sex_code = runner.get("horseSexCode") or runner.get("sexCode")
            if not sex_code:
                color_sex = runner.get("colorSex", "")
                parts = color_sex.strip().split()
                if len(parts) >= 2:
                    sex_code = parts[-1].upper()
            race_entry["runners"].append({
                "name": runner.get("horseName", ""),
                "sex_code": sex_code,
            })

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(races_out)} done", flush=True)

    print(f"  Done: {len(races_out)} races scraped", flush=True)
    return races_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape racecards from Racing Post")
    parser.add_argument("--date", type=str, required=True,
                        help="YYYY-MM-DD, 'today', or 'tomorrow'")
    args = parser.parse_args()

    target_date = _resolve_date(args.date)

    load_dotenv(str(PROJECT_DIR / "data" / "raw" / "rpscrape_repo" / ".env"))
    load_dotenv(str(PROJECT_DIR / ".env"))

    client = NetworkClient(
        email=os.getenv("EMAIL"),
        auth_state=os.getenv("AUTH_STATE"),
        access_token=os.getenv("ACCESS_TOKEN"),
    )

    print(f"Scraping racecards for {target_date}", flush=True)
    races = scrape_racecards(target_date, client)
    if not races:
        print("No races scraped")
        return

    RACECARDS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RACECARDS_DIR / f"{target_date}.json"
    with open(output_path, "w") as f:
        json.dump(races, f)
    print(f"  Saved {len(races)} races to {output_path}", flush=True)


if __name__ == "__main__":
    main()
