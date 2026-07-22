from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dateutil import parser as dtparser

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db
from ingestion.normalise import normalise_course, slugify


DB_PATH = ROOT / "racing.duckdb"
LOG_DIR = ROOT / "logs"


@dataclass
class NormalizedRow:
    race_date: datetime.date
    course_name: str
    course_norm: str
    horse_name: str
    horse_norm: str
    horse_compact: str
    off_time_utc: datetime | None
    trainer_name: str | None
    jockey_name: str | None
    draw: int | None
    weight_lbs: float | None
    age: int | None
    official_rating: int | None
    headgear: str | None
    race_type: str | None
    race_class: int | None
    is_handicap: bool | None
    distance_furlongs: float | None
    surface: str | None
    going_code: str | None
    prize_money_gbp: float | None
    finishing_position: int | None
    won: bool | None
    btn_lengths: float | None
    official_time_secs: float | None
    rpr: float | None
    non_completion: str | None
    sex: str | None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def _coerce_class(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _coerce_is_handicap(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "hcap" in text or "handicap" in text:
        return True
    return False


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = dtparser.parse(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_date(value: Any) -> datetime.date | None:
    dt = _parse_datetime(value)
    if dt is None:
        return None
    return dt.date()


def _normalize_course_name(name: str) -> str:
    canonical = normalise_course(name)
    return slugify(canonical) or ""


def _normalize_horse_name(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""

    # Handle RP artifacts where a trailing "I" is injected before country suffixes.
    # Examples: "Mr MoonshineI (IRE)", "Westaway I (IRE)", "Castle ViewI (IRE)".
    text = re.sub(r"(?<=[A-Za-z])I(?=\s*\()", "", text)
    text = re.sub(r"\bI\b(?=\s*\()", "", text)

    # Remove trailing breeding/country suffixes commonly represented in brackets.
    text = re.sub(r"\([^)]*\)", "", text)
    # Some feeds include trailing country tags without brackets, e.g. "Horse Name IRE".
    text = re.sub(r"\b(?:IRE|GB|FR|USA|AUS|GER|ITY)\b$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return slugify(text) or ""


def _compact_name_key(name: str) -> str:
    # Build compact key from already-normalized horse name to avoid country-tag drift.
    text = _normalize_horse_name(name)
    # Keep alphanumeric only and collapse everything else.
    return re.sub(r"[^a-z0-9]", "", text)


def _column_value(row: dict[str, Any], aliases: list[str]) -> Any:
    lower_map = {k.lower(): v for k, v in row.items()}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _normalize_input_row(row: dict[str, Any]) -> NormalizedRow | None:
    race_date_raw = _column_value(row, ["race_date", "date", "racedate"])
    race_date = _parse_date(race_date_raw)
    if race_date is None:
        return None

    course_name = str(_column_value(row, ["course", "course_name", "track", "venue", "meeting"]) or "").strip()
    horse_name = str(_column_value(row, ["horse", "horse_name", "runner", "selection_name"]) or "").strip()
    if not course_name or not horse_name:
        return None

    off_time_utc = _parse_datetime(_column_value(row, ["off_time", "off", "race_time", "scheduled_off", "event_dt"]))

    finishing_position = _coerce_int(_column_value(row, ["finishing_position", "position", "pos", "finish_pos"]))
    pos_raw = str(_column_value(row, ["pos", "finishing_position", "position"]) or "").strip().upper()
    non_completion = pos_raw if pos_raw and not pos_raw.replace(".", "").isdigit() and pos_raw not in ("", "NONE", "NAN") else None
    won_raw = _column_value(row, ["won", "win", "win_flag", "win_lose"])
    won: bool | None = None
    if won_raw is not None and str(won_raw).strip() != "":
        won = str(won_raw).strip().lower() in {"1", "true", "win", "w"}
    elif finishing_position is not None:
        won = finishing_position == 1

    return NormalizedRow(
        race_date=race_date,
        course_name=course_name,
        course_norm=_normalize_course_name(course_name),
        horse_name=horse_name,
        horse_norm=_normalize_horse_name(horse_name),
        horse_compact=_compact_name_key(horse_name),
        off_time_utc=off_time_utc,
        trainer_name=str(_column_value(row, ["trainer", "trainer_name"]) or "").strip() or None,
        jockey_name=str(_column_value(row, ["jockey", "jockey_name", "rider"]) or "").strip() or None,
        draw=_coerce_int(_column_value(row, ["draw", "stall", "stall_draw"])),
        weight_lbs=_coerce_float(_column_value(row, ["weight_lbs", "weight", "lbs"])),
        age=_coerce_int(_column_value(row, ["age"])),
        official_rating=_coerce_int(_column_value(row, ["official_rating", "or", "rating"])),
        headgear=str(_column_value(row, ["headgear", "hg"]) or "").strip() or None,
        race_type=str(_column_value(row, ["race_type", "type"]) or "").strip() or None,
        race_class=_coerce_class(_column_value(row, ["race_class", "class", "class_band"])),
        is_handicap=_coerce_is_handicap(
            _column_value(row, ["race_name", "race_type", "type"])
        ),
        distance_furlongs=_coerce_float(_column_value(row, ["dist_f", "distance_furlongs", "distance", "trip", "dist"])),
        surface=str(_column_value(row, ["surface"]) or "").strip() or None,
        going_code=str(_column_value(row, ["going", "going_code"]) or "").strip() or None,
        prize_money_gbp=_coerce_float(_column_value(row, ["prize_money", "prize_money_gbp", "prize"])),
        finishing_position=finishing_position,
        won=won,
        btn_lengths=_coerce_float(_column_value(row, ["btn", "btn_lengths", "ovr_btn"])),
        official_time_secs=_coerce_float(_column_value(row, ["secs", "official_time_secs", "time_secs"])),
        rpr=_coerce_float(_column_value(row, ["rpr"])),
        non_completion=non_completion,
        sex=str(_column_value(row, ["sex"]) or "").strip().upper() or None,
    )


def _load_rpscrape_rows(csv_paths: list[Path]) -> list[NormalizedRow]:
    rows: list[NormalizedRow] = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        for record in frame.to_dict(orient="records"):
            normalized = _normalize_input_row(record)
            if normalized is not None:
                rows.append(normalized)
    return rows


def _confidence_rank(value: str) -> int:
    ranks = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return ranks.get(value, 0)


def enrich_from_rpscrape(
    db_path: Path,
    input_glob: str = "data/raw/rpscrape/**/*.csv",
    min_confidence: str = "medium",
    countries: tuple[str, ...] = ("GB", "IE"),
    dry_run: bool = False,
    target_date: datetime.date | None = None,
) -> dict[str, Any]:
    csv_paths = sorted(ROOT.glob(input_glob))
    if not csv_paths:
        return {
            "status": "no_input_files",
            "input_glob": input_glob,
            "matched_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
        }

    source_rows = _load_rpscrape_rows(csv_paths)
    if target_date:
        source_rows = [row for row in source_rows if row.race_date == target_date]
    if not source_rows:
        return {
            "status": "no_parseable_rows",
            "files": len(csv_paths),
            "matched_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
        }

    con = get_db(db_path)
    try:
        races_df = con.execute(
            """
            SELECT race_id, race_date, scheduled_off_utc, course_name, race_type, race_class, going_code, prize_money_gbp, country
            FROM races
            """
        ).df()
        runners_df = con.execute(
            """
            SELECT runner_id, race_id, horse_name, trainer_name, jockey_name
            FROM runners
            """
        ).df()
    finally:
        con.close()

    if races_df.empty or runners_df.empty:
        return {
            "status": "empty_target_tables",
            "matched_rows": 0,
            "unmatched_rows": len(source_rows),
            "ambiguous_rows": 0,
        }

    races_df["race_date"] = pd.to_datetime(races_df["race_date"]).dt.date
    if target_date:
        races_df = races_df[races_df["race_date"] == target_date]
        if races_df.empty:
            return {
                "status": "no_target_races_for_date",
                "target_date": str(target_date),
                "matched_rows": 0,
                "unmatched_rows": len(source_rows),
                "ambiguous_rows": 0,
            }
    country_set = {c.strip().upper() for c in countries if c.strip()}
    if country_set:
        races_df["country"] = races_df["country"].fillna("").astype(str).str.upper()
        races_df = races_df[races_df["country"].isin(country_set)]
        if races_df.empty:
            return {
                "status": "no_target_races_for_countries",
                "countries": sorted(country_set),
                "matched_rows": 0,
                "unmatched_rows": len(source_rows),
                "ambiguous_rows": 0,
            }

    races_df["course_norm"] = races_df["course_name"].fillna("").map(_normalize_course_name)
    races_df["scheduled_off_utc"] = pd.to_datetime(races_df["scheduled_off_utc"], utc=True, errors="coerce")

    runners_df["horse_norm"] = runners_df["horse_name"].fillna("").map(lambda x: _normalize_horse_name(str(x)))
    runners_df["horse_compact"] = runners_df["horse_name"].fillna("").map(lambda x: _compact_name_key(str(x)))

    races_by_key: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    races_by_date: dict[Any, list[dict[str, Any]]] = {}
    for row in races_df.to_dict(orient="records"):
        key = (row["race_date"], row["course_norm"])
        races_by_key.setdefault(key, []).append(row)
        races_by_date.setdefault(row["race_date"], []).append(row)

    runners_by_race: dict[str, list[dict[str, Any]]] = {}
    for row in runners_df.to_dict(orient="records"):
        runners_by_race.setdefault(row["race_id"], []).append(row)

    race_horse_index: dict[str, set[str]] = {}
    race_horse_compact_index: dict[str, set[str]] = {}
    for race_id, rows in runners_by_race.items():
        race_horse_index[race_id] = {str(r.get("horse_norm") or "") for r in rows}
        race_horse_compact_index[race_id] = {str(r.get("horse_compact") or "") for r in rows}

    race_updates: list[dict[str, Any]] = []
    runner_updates: list[dict[str, Any]] = []
    result_updates: list[dict[str, Any]] = []
    history_updates: list[dict[str, Any]] = []

    unmatched_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []

    for src in source_rows:
        race_candidates = races_by_key.get((src.race_date, src.course_norm), [])
        if not race_candidates:
            # Fallback for differing course naming conventions between sources.
            date_candidates = races_by_date.get(src.race_date, [])
            race_candidates = [
                candidate
                for candidate in date_candidates
                if (
                    src.horse_norm in race_horse_index.get(str(candidate["race_id"]), set())
                    or src.horse_compact in race_horse_compact_index.get(str(candidate["race_id"]), set())
                )
            ]
            if not race_candidates:
                unmatched_rows.append(
                    {
                        "reason": "race_not_found",
                        "race_date": str(src.race_date),
                        "course_name": src.course_name,
                        "horse_name": src.horse_name,
                    }
                )
                continue

        # Prioritize races that actually contain this horse on the card.
        horse_filtered_candidates = [
            candidate
            for candidate in race_candidates
            if (
                src.horse_norm in race_horse_index.get(str(candidate["race_id"]), set())
                or src.horse_compact in race_horse_compact_index.get(str(candidate["race_id"]), set())
            )
        ]
        if horse_filtered_candidates:
            race_candidates = horse_filtered_candidates

        confidence = "low"
        chosen_race = race_candidates[0]
        if len(race_candidates) == 1:
            confidence = "high"
        elif src.off_time_utc is not None:
            scored = []
            for candidate in race_candidates:
                off = candidate.get("scheduled_off_utc")
                if off is None or pd.isna(off):
                    continue
                delta_minutes = abs((off.to_pydatetime() - src.off_time_utc).total_seconds() / 60.0)
                scored.append((delta_minutes, candidate))
            if scored:
                scored.sort(key=lambda x: x[0])
                chosen_race = scored[0][1]
                confidence = "high" if scored[0][0] <= 25 else "medium"
            else:
                confidence = "medium"
        else:
            confidence = "low"

        race_id = str(chosen_race["race_id"])
        runner_candidates = [
            r
            for r in runners_by_race.get(race_id, [])
            if (
                r.get("horse_norm") == src.horse_norm
                or r.get("horse_compact") == src.horse_compact
            )
        ]

        if not runner_candidates:
            unmatched_rows.append(
                {
                    "reason": "runner_not_found",
                    "race_id": race_id,
                    "race_date": str(src.race_date),
                    "course_name": src.course_name,
                    "horse_name": src.horse_name,
                }
            )
            continue

        if len(runner_candidates) > 1:
            ambiguous_rows.append(
                {
                    "reason": "runner_ambiguous",
                    "race_id": race_id,
                    "horse_name": src.horse_name,
                    "candidate_count": len(runner_candidates),
                }
            )
            continue

        runner = runner_candidates[0]
        runner_id = str(runner["runner_id"])

        if _confidence_rank(confidence) < _confidence_rank(min_confidence):
            ambiguous_rows.append(
                {
                    "reason": "confidence_too_low",
                    "race_id": race_id,
                    "runner_id": runner_id,
                    "horse_name": src.horse_name,
                    "confidence": confidence,
                }
            )
            continue

        trainer_id = slugify(src.trainer_name) if src.trainer_name else None
        jockey_id = slugify(src.jockey_name) if src.jockey_name else None

        race_updates.append(
            {
                "race_id": race_id,
                "race_type": src.race_type,
                "race_class": src.race_class,
                "is_handicap": src.is_handicap,
                "distance_furlongs": src.distance_furlongs,
                "surface": src.surface,
                "going_code": src.going_code,
                "prize_money_gbp": src.prize_money_gbp,
            }
        )
        runner_updates.append(
            {
                "runner_id": runner_id,
                "trainer_name": src.trainer_name,
                "trainer_id": trainer_id,
                "jockey_name": src.jockey_name,
                "jockey_id": jockey_id,
                "draw": src.draw,
                "weight_lbs": src.weight_lbs,
                "age": src.age,
                "official_rating": src.official_rating,
                "headgear": src.headgear,
                "sex": src.sex,
            }
        )
        result_updates.append(
            {
                "runner_id": runner_id,
                "race_id": race_id,
                "finishing_position": src.finishing_position,
                "finishing_position_raw": str(src.finishing_position) if src.finishing_position is not None else None,
                "btn_lengths": src.btn_lengths,
                "official_time_secs": src.official_time_secs,
                "rpr": src.rpr,
                "non_completion": src.non_completion,
            }
        )
        history_updates.append(
            {
                "history_id": f"{runner_id}_hist",
                "trainer_id": trainer_id,
                "jockey_id": jockey_id,
                "weight_lbs": src.weight_lbs,
                "official_rating": src.official_rating,
                "headgear": src.headgear,
                "finishing_position": src.finishing_position,
                "btn_lengths": src.btn_lengths,
            }
        )

    applied = {
        "races": 0,
        "runners": 0,
        "results": 0,
        "horse_history": 0,
    }

    if not dry_run:
        con = get_db(db_path)
        try:
            if race_updates:
                race_frame = pd.DataFrame(race_updates).drop_duplicates(subset=["race_id"])
                con.register("tmp_race_updates", race_frame)
                con.execute(
                    """
                    UPDATE races
                    SET
                        race_type = COALESCE(tmp.race_type, races.race_type),
                        race_class = COALESCE(tmp.race_class, races.race_class),
                        is_handicap = COALESCE(tmp.is_handicap, races.is_handicap),
                        distance_furlongs = COALESCE(tmp.distance_furlongs, races.distance_furlongs),
                        surface = COALESCE(tmp.surface, races.surface),
                        going_code = COALESCE(tmp.going_code, races.going_code),
                        prize_money_gbp = COALESCE(tmp.prize_money_gbp, races.prize_money_gbp)
                    FROM tmp_race_updates tmp
                    WHERE races.race_id = tmp.race_id
                    """
                )
                applied["races"] = len(race_frame)

            if runner_updates:
                runner_frame = pd.DataFrame(runner_updates).drop_duplicates(subset=["runner_id"])
                con.register("tmp_runner_updates", runner_frame)
                con.execute(
                    """
                    UPDATE runners
                    SET
                        trainer_name = COALESCE(tmp.trainer_name, runners.trainer_name),
                        trainer_id = COALESCE(tmp.trainer_id, runners.trainer_id),
                        jockey_name = COALESCE(tmp.jockey_name, runners.jockey_name),
                        jockey_id = COALESCE(tmp.jockey_id, runners.jockey_id),
                        draw = COALESCE(tmp.draw, runners.draw),
                        weight_lbs = COALESCE(tmp.weight_lbs, runners.weight_lbs),
                        age = COALESCE(tmp.age, runners.age),
                        official_rating = COALESCE(tmp.official_rating, runners.official_rating),
                        headgear = COALESCE(tmp.headgear, runners.headgear),
                        sex = COALESCE(tmp.sex, runners.sex)
                    FROM tmp_runner_updates tmp
                    WHERE runners.runner_id = tmp.runner_id
                    """
                )
                applied["runners"] = len(runner_frame)

            if result_updates:
                result_frame = pd.DataFrame(result_updates).drop_duplicates(subset=["runner_id"])
                con.register("tmp_result_updates", result_frame)
                con.execute(
                    """
                    UPDATE results
                    SET
                        finishing_position = COALESCE(tmp.finishing_position, results.finishing_position),
                        finishing_position_raw = COALESCE(tmp.finishing_position_raw, results.finishing_position_raw),
                        btn_lengths = COALESCE(tmp.btn_lengths, results.btn_lengths),
                        official_time_secs = COALESCE(tmp.official_time_secs, results.official_time_secs),
                        rpr = COALESCE(tmp.rpr, results.rpr),
                        non_completion = COALESCE(tmp.non_completion, results.non_completion)
                    FROM tmp_result_updates tmp
                    WHERE results.runner_id = tmp.runner_id
                      AND results.race_id = tmp.race_id
                    """
                )
                applied["results"] = len(result_frame)

            if history_updates:
                history_frame = pd.DataFrame(history_updates).drop_duplicates(subset=["history_id"])
                con.register("tmp_history_updates", history_frame)
                con.execute(
                    """
                    UPDATE horse_history
                    SET
                        trainer_id = COALESCE(tmp.trainer_id, horse_history.trainer_id),
                        jockey_id = COALESCE(tmp.jockey_id, horse_history.jockey_id),
                        weight_lbs = COALESCE(tmp.weight_lbs, horse_history.weight_lbs),
                        official_rating = COALESCE(tmp.official_rating, horse_history.official_rating),
                        headgear = COALESCE(tmp.headgear, horse_history.headgear),
                        finishing_position = COALESCE(tmp.finishing_position, horse_history.finishing_position),
                        btn_lengths = COALESCE(tmp.btn_lengths, horse_history.btn_lengths)
                    FROM tmp_history_updates tmp
                    WHERE horse_history.history_id = tmp.history_id
                    """
                )
                applied["horse_history"] = len(history_frame)
        finally:
            con.close()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unmatched_path = LOG_DIR / f"rpscrape_unmatched_{stamp}.csv"
    ambiguous_path = LOG_DIR / f"rpscrape_ambiguous_{stamp}.csv"
    summary_path = LOG_DIR / f"rpscrape_enrich_{stamp}.json"

    with unmatched_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["reason", "race_id", "race_date", "course_name", "horse_name"])
        writer.writeheader()
        for row in unmatched_rows:
            writer.writerow(row)

    with ambiguous_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["reason", "race_id", "runner_id", "horse_name", "confidence", "candidate_count"])
        writer.writeheader()
        for row in ambiguous_rows:
            writer.writerow(row)

    summary = {
        "status": "ok",
        "files_scanned": len(csv_paths),
        "rows_parsed": len(source_rows),
        "rows_matched": len(runner_updates),
        "rows_unmatched": len(unmatched_rows),
        "rows_ambiguous": len(ambiguous_rows),
        "dry_run": dry_run,
        "applied": applied,
        "unmatched_log": str(unmatched_path),
        "ambiguous_log": str(ambiguous_path),
        "input_glob": input_glob,
        "min_confidence": min_confidence,
        "countries": sorted(country_set),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_log"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich SP spine with rpscrape fundamentals")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--input-glob", type=str, default="data/raw/rpscrape_repo/data/region/**/*.csv")
    parser.add_argument("--min-confidence", type=str, choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--countries", type=str, default="GB,IE", help="Comma-separated race countries to enrich")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = enrich_from_rpscrape(
        db_path=Path(args.db_path),
        input_glob=args.input_glob,
        min_confidence=args.min_confidence,
        countries=tuple(x.strip().upper() for x in args.countries.split(",") if x.strip()),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
