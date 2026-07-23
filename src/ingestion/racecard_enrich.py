"""
Enrich today's races with metadata from Racing Post racecards.

Reads racecard JSON (scraped by scripts/scrape_racecards.py) and UPDATEs
today's races and runners in the DB with going, distance, class, prize money,
handicap, surface, race grade, and sex — columns the Betfair API doesn't
provide.

Usage:
    python -m src.ingestion.racecard_enrich --date today
    python -m src.ingestion.racecard_enrich --date 2026-07-23 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db
from ingestion.rpscrape_enrich import (
    _compact_name_key,
    _normalize_course_name,
    _normalize_horse_name,
)

RACECARD_DIR = ROOT / "data" / "raw" / "rpscrape_repo" / "racecards"


def _to_uk_time(dt: pd.Timestamp) -> str:
    try:
        import zoneinfo
        uk = zoneinfo.ZoneInfo("Europe/London")
    except ImportError:
        from dateutil import tz
        uk = tz.gettz("Europe/London")
    try:
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.astimezone(uk).strftime("%H:%M")
    except Exception:
        return str(dt)[:5]


def _time_diff_minutes(t1: str, t2: str) -> int:
    h1, m1 = int(t1[:2]), int(t1[3:5])
    h2, m2 = int(t2[:2]), int(t2[3:5])
    return abs((h1 * 60 + m1) - (h2 * 60 + m2))


_RaceEntry = dict[str, str]
_RunnerEntry = dict[str, str]
_RaceIndex = dict[tuple[str, str], _RaceEntry]
_CourseIndex = dict[str, list[_RaceEntry]]
_RunnerIndex = dict[str, list[_RunnerEntry]]


def _build_db_index(
    db_path: str, target_date: date,
) -> tuple[_RaceIndex, _CourseIndex, _RunnerIndex]:
    con = get_db(db_path)
    races_df = con.execute(
        """SELECT race_id, course_name, scheduled_off_utc
           FROM races WHERE race_date = ?""",
        [target_date],
    ).df()
    runners_df = con.execute(
        """SELECT runner_id, race_id, horse_name
           FROM runners
           WHERE race_id IN (SELECT race_id FROM races WHERE race_date = ?)""",
        [target_date],
    ).df()
    con.close()

    race_index: _RaceIndex = {}
    course_index: _CourseIndex = defaultdict(list)

    for _, row in races_df.iterrows():
        slug = _normalize_course_name(str(row["course_name"]))
        uk_time = _to_uk_time(row["scheduled_off_utc"])
        entry: _RaceEntry = {"race_id": row["race_id"], "slug": slug, "uk_time": uk_time}
        race_index[(slug, uk_time)] = entry
        course_index[slug].append(entry)

    runner_index: _RunnerIndex = defaultdict(list)
    for _, row in runners_df.iterrows():
        name = str(row.get("horse_name", "") or "")
        runner_index[row["race_id"]].append({
            "runner_id": row["runner_id"],
            "name_norm": _normalize_horse_name(name),
            "name_compact": _compact_name_key(name),
        })

    return race_index, course_index, runner_index


def _match_race(
    rc_slug: str, off_time: str,
    race_index: _RaceIndex,
    course_index: _CourseIndex,
) -> str | None:
    hit = race_index.get((rc_slug, off_time))
    if hit:
        return hit["race_id"]

    for cand in course_index.get(rc_slug, []):
        if _time_diff_minutes(off_time, cand["uk_time"]) <= 5:
            return cand["race_id"]

    return None


def _match_runner(
    horse_name: str, race_runners: list[_RunnerEntry],
) -> str | None:
    norm = _normalize_horse_name(horse_name)
    compact = _compact_name_key(horse_name)

    for r in race_runners:
        if r["name_norm"] == norm:
            return r["runner_id"]

    for r in race_runners:
        if r["name_compact"] == compact:
            return r["runner_id"]

    return None


def enrich_from_racecards(
    db_path: Path,
    racecard_json_path: Path,
    target_date: date,
    dry_run: bool = False,
) -> dict[str, int]:
    with open(racecard_json_path, "r") as f:
        races_data: list[dict] = json.load(f)

    if not races_data:
        print("  No races found in racecard JSON", flush=True)
        return {"racecard_races": 0, "matched_races": 0, "matched_runners": 0}

    race_index, course_index, runner_index = _build_db_index(str(db_path), target_date)

    race_updates: list[dict[str, object]] = []
    runner_updates: list[dict[str, str | None]] = []
    unmatched_races: list[tuple[str, str]] = []

    for rc in races_data:
        course_name = rc.get("course", "")
        off_time = rc.get("off_time", "")
        rc_slug = _normalize_course_name(course_name)
        race_id = _match_race(rc_slug, off_time, race_index, course_index)

        if not race_id:
            unmatched_races.append((course_name, off_time))
            continue

        race_updates.append({
            "race_id": race_id,
            "going_code": rc.get("going"),
            "distance_furlongs": rc.get("distance_f"),
            "race_class": rc.get("race_class"),
            "prize_money_gbp": rc.get("prize_money_gbp"),
            "is_handicap": rc.get("is_handicap"),
            "surface": rc.get("surface"),
            "race_grade": rc.get("race_grade"),
        })

        for rc_runner in rc.get("runners", []):
            horse_name = rc_runner.get("name", "")
            runner_id = _match_runner(horse_name, runner_index.get(race_id, []))
            if runner_id:
                runner_updates.append({
                    "runner_id": runner_id,
                    "sex": rc_runner.get("sex_code"),
                })

    summary = {
        "racecard_races": len(races_data),
        "matched_races": len(race_updates),
        "matched_runners": len(runner_updates),
        "unmatched_races": len(unmatched_races),
    }

    if unmatched_races:
        print(f"  Unmatched races: {unmatched_races[:5]}"
              f"{'...' if len(unmatched_races) > 5 else ''}", flush=True)

    if dry_run:
        print(f"  DRY RUN: would update {len(race_updates)} races, "
              f"{len(runner_updates)} runners", flush=True)
        return summary

    con = get_db(str(db_path))
    try:
        if race_updates:
            race_frame = pd.DataFrame(race_updates).drop_duplicates(subset=["race_id"])
            con.register("tmp_rc_race_updates", race_frame)
            con.execute(
                """
                UPDATE races
                SET
                    going_code = COALESCE(tmp.going_code, races.going_code),
                    distance_furlongs = COALESCE(tmp.distance_furlongs, races.distance_furlongs),
                    race_class = COALESCE(tmp.race_class, races.race_class),
                    prize_money_gbp = COALESCE(tmp.prize_money_gbp, races.prize_money_gbp),
                    is_handicap = COALESCE(tmp.is_handicap, races.is_handicap),
                    surface = COALESCE(tmp.surface, races.surface),
                    race_grade = COALESCE(tmp.race_grade, races.race_grade)
                FROM tmp_rc_race_updates tmp
                WHERE races.race_id = tmp.race_id
                """
            )

        if runner_updates:
            runner_frame = pd.DataFrame(runner_updates).drop_duplicates(subset=["runner_id"])
            con.register("tmp_rc_runner_updates", runner_frame)
            try:
                con.execute("ALTER TABLE runners ADD COLUMN sex VARCHAR")
            except Exception:
                pass
            con.execute(
                """
                UPDATE runners
                SET sex = COALESCE(tmp.sex, runners.sex)
                FROM tmp_rc_runner_updates tmp
                WHERE runners.runner_id = tmp.runner_id
                """
            )
    finally:
        con.close()

    print(f"  Enriched {len(race_updates)} races, {len(runner_updates)} runners "
          f"from racecards", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich DB races with racecard metadata")
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD, 'today', or 'tomorrow'")
    parser.add_argument("--db-path", type=str, default=str(ROOT / "racing.duckdb"))
    parser.add_argument("--racecard-dir", type=str, default=str(RACECARD_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from pipelines.helpers import resolve_date
    target_date = resolve_date(args.date)

    racecard_json = Path(args.racecard_dir) / f"{target_date}.json"
    if not racecard_json.exists():
        print(f"No racecard JSON found at {racecard_json}")
        return

    print(f"Enriching races for {target_date} from {racecard_json}", flush=True)
    summary = enrich_from_racecards(
        db_path=Path(args.db_path),
        racecard_json_path=racecard_json,
        target_date=target_date,
        dry_run=args.dry_run,
    )
    print(f"  {summary['matched_races']}/{summary['racecard_races']} races matched, "
          f"{summary['matched_runners']} runners enriched", flush=True)


if __name__ == "__main__":
    main()
