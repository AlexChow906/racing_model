from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.normalise import normalise_course, normalise_horse, slugify
from ingestion.db_connect import get_db
from ingestion.rpscrape_enrich import _load_rpscrape_rows
from quality.checks import ensure_standard_race_flag

DB_PATH = ROOT / "racing.duckdb"
LOG_DIR = ROOT / "logs"


@dataclass
class TargetRunner:
    runner_id: str
    race_id: str
    race_date: date
    course_name: str
    horse_name: str
    horse_norm: str
    course_relaxed: str


@dataclass
class RpsRow:
    race_date: date
    course_name: str
    horse_name: str
    horse_norm: str
    course_relaxed: str
    trainer_name: str | None
    jockey_name: str | None


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date"):
        try:
            converted = value.date()
            if isinstance(converted, date):
                return converted
        except Exception:
            pass
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1

    transpositions = t / 2.0
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    jaro = _jaro_similarity(s1, s2)
    if jaro <= 0.0:
        return 0.0
    prefix = 0
    max_prefix = 4
    for a, b in zip(s1, s2):
        if a == b:
            prefix += 1
            if prefix == max_prefix:
                break
        else:
            break
    return jaro + (prefix * 0.1 * (1.0 - jaro))


def _load_target_runners(con: duckdb.DuckDBPyConnection, year: int | None = None) -> list[TargetRunner]:
    where_year = ""
    if year is not None:
        where_year = f" AND EXTRACT(year FROM ra.race_date) = {int(year)}"
    rows = con.execute(
        f"""
        SELECT
            ru.runner_id,
            ru.race_id,
            ra.race_date,
            ra.course_name,
            ru.horse_name
        FROM runners ru
        JOIN races ra ON ru.race_id = ra.race_id
        WHERE ra.is_standard_race
          AND COALESCE(TRIM(ru.trainer_name), '') = ''
          AND COALESCE(TRIM(ru.jockey_name), '') = ''
                    {where_year}
                """
    ).fetchall()

    targets: list[TargetRunner] = []
    for row in rows:
        horse_name = str(row[4] or "")
        course_name = str(row[3] or "")
        race_date = _to_date(row[2])
        targets.append(
            TargetRunner(
                runner_id=str(row[0]),
                race_id=str(row[1]),
                race_date=race_date,
                course_name=course_name,
                horse_name=horse_name,
                horse_norm=normalise_horse(horse_name),
                course_relaxed=normalise_course(course_name),
            )
        )
    return targets


def _load_rps_rows(input_glob: str, year: int | None = None) -> tuple[list[RpsRow], dict[str, int]]:
    csv_paths = sorted(ROOT.glob(input_glob))
    parsed_rows = _load_rpscrape_rows(csv_paths)

    rows: list[RpsRow] = []
    for row in parsed_rows:
        # Keep rows that can contribute enrichment.
        if not row.trainer_name and not row.jockey_name:
            continue
        race_date = _to_date(row.race_date)
        if year is not None and race_date.year != int(year):
            continue
        rows.append(
            RpsRow(
                race_date=race_date,
                course_name=row.course_name,
                horse_name=row.horse_name,
                horse_norm=normalise_horse(row.horse_name),
                course_relaxed=normalise_course(row.course_name),
                trainer_name=row.trainer_name,
                jockey_name=row.jockey_name,
            )
        )

    return rows, {"files_scanned": len(csv_paths), "rows_parsed": len(parsed_rows), "rows_with_people": len(rows)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def run_rematch(db_path: Path, input_glob: str, dry_run: bool = False, year: int | None = None) -> dict[str, Any]:
    con = get_db(db_path)
    try:
        standard_stats = ensure_standard_race_flag(con)
        targets = _load_target_runners(con, year=year)
        where_year = ""
        if year is not None:
            where_year = f" AND EXTRACT(year FROM ra.race_date) = {int(year)}"

        race_card_rows = con.execute(
            f"""
            SELECT ru.race_id, ra.race_date, ra.course_name, ru.horse_name
            FROM runners ru
            JOIN races ra ON ra.race_id = ru.race_id
            WHERE ra.is_standard_race
            {where_year}
            """
        ).fetchall()
    finally:
        con.close()

    rps_rows, src_stats = _load_rps_rows(input_glob, year=year)

    by_key: dict[tuple[date, str, str], list[int]] = defaultdict(list)
    by_card: dict[tuple[date, str], list[int]] = defaultdict(list)
    rps_card_sets: dict[tuple[date, str], set[str]] = defaultdict(set)
    for i, row in enumerate(rps_rows):
        key = (row.race_date, row.course_relaxed, row.horse_norm)
        card_key = (row.race_date, row.course_relaxed)
        by_key[key].append(i)
        by_card[card_key].append(i)
        rps_card_sets[card_key].add(row.horse_norm)

    betfair_card_sets: dict[tuple[date, str], set[str]] = defaultdict(set)
    for _, race_date, course_name, horse_name in race_card_rows:
        card_key = (_to_date(race_date), normalise_course(str(course_name or "")))
        betfair_card_sets[card_key].add(normalise_horse(str(horse_name or "")))

    updates: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def pick_best(indices: list[int]) -> int | None:
        if not indices:
            return None
        # Prefer rows with both trainer and jockey, then stable earliest index.
        scored = []
        for idx in indices:
            r = rps_rows[idx]
            score = int(bool(r.trainer_name)) + int(bool(r.jockey_name))
            scored.append((score, -idx, idx))
        scored.sort(reverse=True)
        return scored[0][2]

    for target in targets:
        matched_idx: int | None = None
        match_type = "unmatched"
        confidence = 0.0
        fuzzy_score: float | None = None
        card_jaccard: float | None = None

        # Strategy A: relaxed course + exact date + horse
        key = (target.race_date, target.course_relaxed, target.horse_norm)
        exact_candidates = by_key.get(key, [])
        if exact_candidates:
            matched_idx = pick_best(exact_candidates)
            match_type = "relaxed_course"
            confidence = 1.0

        # Strategy B: date +/- 1 day on same relaxed course + horse
        if matched_idx is None:
            for offset in (1, -1):
                shifted = target.race_date + timedelta(days=offset)
                shifted_candidates = by_key.get((shifted, target.course_relaxed, target.horse_norm), [])
                if shifted_candidates:
                    matched_idx = pick_best(shifted_candidates)
                    match_type = "date_shifted"
                    confidence = 0.98
                    break

        # Strategy C: card-level overlap > 70%
        if matched_idx is None:
            card_key = (target.race_date, target.course_relaxed)
            race_card = betfair_card_sets.get(card_key, set())
            rps_card = rps_card_sets.get(card_key, set())
            jaccard = _jaccard(race_card, rps_card)
            if jaccard > 0.70:
                candidates = [
                    idx for idx in by_card.get(card_key, [])
                    if rps_rows[idx].horse_norm == target.horse_norm
                ]
                if candidates:
                    matched_idx = pick_best(candidates)
                    match_type = "card_overlap"
                    card_jaccard = jaccard
                    confidence = min(0.99, max(0.93, jaccard))

        # Strategy D: Jaro-Winkler fuzzy, only when date+course match exactly.
        if matched_idx is None:
            card_key = (target.race_date, target.course_relaxed)
            fuzzy_candidates: list[tuple[float, int]] = []
            for idx in by_card.get(card_key, []):
                score = _jaro_winkler_similarity(target.horse_norm, rps_rows[idx].horse_norm)
                if score >= 0.93:
                    fuzzy_candidates.append((score, idx))
            if fuzzy_candidates:
                fuzzy_candidates.sort(key=lambda x: x[0], reverse=True)
                fuzzy_score, matched_idx = fuzzy_candidates[0]
                match_type = "fuzzy_name"
                confidence = float(fuzzy_score)

        matched_rps: RpsRow | None = rps_rows[matched_idx] if matched_idx is not None else None

        if matched_rps is not None:
            trainer_name = matched_rps.trainer_name
            jockey_name = matched_rps.jockey_name
            trainer_id = slugify(trainer_name) if trainer_name else None
            jockey_id = slugify(jockey_name) if jockey_name else None
            updates.append(
                {
                    "runner_id": target.runner_id,
                    "trainer_name": trainer_name,
                    "trainer_id": trainer_id,
                    "jockey_name": jockey_name,
                    "jockey_id": jockey_id,
                }
            )
            audit_rows.append(
                {
                    "runner_id": target.runner_id,
                    "betfair_race_id": target.race_id,
                    "source": "rpscrape",
                    "race_date": str(target.race_date),
                    "course_name": target.course_name,
                    "horse_name": target.horse_name,
                    "match_type": match_type,
                    "match_confidence": round(confidence, 4),
                    "matched_rps_date": str(matched_rps.race_date),
                    "matched_rps_course": matched_rps.course_name,
                    "matched_rps_horse": matched_rps.horse_name,
                    "trainer_name": trainer_name,
                    "jockey_name": jockey_name,
                    "fuzzy_score": round(float(fuzzy_score), 4) if fuzzy_score is not None else None,
                    "card_jaccard": round(float(card_jaccard), 4) if card_jaccard is not None else None,
                }
            )
        else:
            audit_rows.append(
                {
                    "runner_id": target.runner_id,
                    "betfair_race_id": None,
                    "source": "rpscrape_only",
                    "race_date": str(target.race_date),
                    "course_name": target.course_name,
                    "horse_name": target.horse_name,
                    "match_type": "unmatched",
                    "match_confidence": 0.0,
                    "matched_rps_date": None,
                    "matched_rps_course": None,
                    "matched_rps_horse": None,
                    "trainer_name": None,
                    "jockey_name": None,
                    "fuzzy_score": None,
                    "card_jaccard": None,
                }
            )

    # Preserve one update per runner.
    dedup_updates: dict[str, dict[str, Any]] = {}
    for row in updates:
        dedup_updates[row["runner_id"]] = row

    unmatched_runner_ids = sorted(
        {
            str(row["runner_id"])
            for row in audit_rows
            if row.get("match_type") == "unmatched" and row.get("runner_id")
        }
    )

    applied_updates = 0
    tagged_coverage_gaps = 0
    if not dry_run and (dedup_updates or audit_rows):
        con = get_db(db_path)
        try:
            con.execute("ALTER TABLE runners ADD COLUMN IF NOT EXISTS match_type VARCHAR")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS rematch_audit (
                    audit_id BIGINT,
                    audit_timestamp_utc TIMESTAMPTZ NOT NULL,
                    runner_id VARCHAR,
                    betfair_race_id VARCHAR,
                    source VARCHAR,
                    race_date DATE,
                    course_name VARCHAR,
                    horse_name VARCHAR,
                    match_type VARCHAR,
                    match_confidence FLOAT,
                    matched_rps_date DATE,
                    matched_rps_course VARCHAR,
                    matched_rps_horse VARCHAR,
                    trainer_name VARCHAR,
                    jockey_name VARCHAR,
                    fuzzy_score FLOAT,
                    card_jaccard FLOAT
                )
                """
            )

            if dedup_updates:
                update_frame = [dedup_updates[k] for k in sorted(dedup_updates.keys())]
                update_df = pd.DataFrame(update_frame)
                con.register("tmp_rematch_updates", update_df)
                con.execute(
                    """
                    UPDATE runners r
                    SET
                        trainer_name = COALESCE(tmp.trainer_name, r.trainer_name),
                        trainer_id = COALESCE(tmp.trainer_id, r.trainer_id),
                        jockey_name = COALESCE(tmp.jockey_name, r.jockey_name),
                        jockey_id = COALESCE(tmp.jockey_id, r.jockey_id)
                    FROM tmp_rematch_updates tmp
                    WHERE r.runner_id = tmp.runner_id
                      AND COALESCE(TRIM(r.trainer_name), '') = ''
                      AND COALESCE(TRIM(r.jockey_name), '') = ''
                    """
                )
                applied_updates = int(con.execute("SELECT COUNT(*) FROM tmp_rematch_updates").fetchone()[0])

            if unmatched_runner_ids:
                unmatched_df = pd.DataFrame(
                    {
                        "runner_id": unmatched_runner_ids,
                        "match_type": ["unmatched"] * len(unmatched_runner_ids),
                    }
                )
                con.register("unmatched_runners", unmatched_df)
                con.execute(
                    """
                    UPDATE runners
                    SET match_type = 'rpscrape_coverage_gap',
                        trainer_name = NULL,
                        jockey_name = NULL
                    WHERE runner_id IN (
                        SELECT runner_id
                        FROM unmatched_runners
                        WHERE match_type = 'unmatched'
                    )
                    """
                )
                tagged_coverage_gaps = int(
                    con.execute(
                        """
                        SELECT COUNT(*)
                        FROM runners
                        WHERE runner_id IN (SELECT runner_id FROM unmatched_runners)
                          AND match_type = 'rpscrape_coverage_gap'
                        """
                    ).fetchone()[0]
                )

            audit_payload = []
            now_utc = datetime.now(timezone.utc)
            for i, row in enumerate(audit_rows, start=1):
                payload = dict(row)
                payload["audit_id"] = i
                payload["audit_timestamp_utc"] = now_utc
                audit_payload.append(payload)
            audit_df = pd.DataFrame(audit_payload)
            con.register("tmp_rematch_audit", audit_df)
            con.execute(
                """
                INSERT INTO rematch_audit (
                    audit_id,
                    audit_timestamp_utc,
                    runner_id,
                    betfair_race_id,
                    source,
                    race_date,
                    course_name,
                    horse_name,
                    match_type,
                    match_confidence,
                    matched_rps_date,
                    matched_rps_course,
                    matched_rps_horse,
                    trainer_name,
                    jockey_name,
                    fuzzy_score,
                    card_jaccard
                )
                SELECT
                    audit_id,
                    audit_timestamp_utc,
                    runner_id,
                    betfair_race_id,
                    source,
                    race_date,
                    course_name,
                    horse_name,
                    match_type,
                    match_confidence,
                    matched_rps_date,
                    matched_rps_course,
                    matched_rps_horse,
                    trainer_name,
                    jockey_name,
                    fuzzy_score,
                    card_jaccard
                FROM tmp_rematch_audit
                """
            )
        finally:
            con.close()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = LOG_DIR / f"rematch_results_{stamp}.csv"
    fuzzy_path = LOG_DIR / f"rematch_fuzzy_sample_{stamp}.csv"

    fieldnames = [
        "runner_id",
        "betfair_race_id",
        "source",
        "race_date",
        "course_name",
        "horse_name",
        "match_type",
        "match_confidence",
        "matched_rps_date",
        "matched_rps_course",
        "matched_rps_horse",
        "trainer_name",
        "jockey_name",
        "fuzzy_score",
        "card_jaccard",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    fuzzy_rows = [row for row in audit_rows if row["match_type"] == "fuzzy_name"]
    with fuzzy_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in fuzzy_rows[:20]:
            writer.writerow(row)

    counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        counts[str(row["match_type"])] += 1

    return {
        "status": "ok",
        "dry_run": dry_run,
        "db_path": str(db_path),
        "year": year,
        "target_runners": len(targets),
        "matched_runners": len(dedup_updates),
        "unmatched_runners": counts.get("unmatched", 0),
        "applied_updates": applied_updates,
        "tagged_coverage_gaps": tagged_coverage_gaps,
        "standard_race_summary": standard_stats,
        "source_stats": src_stats,
        "match_type_counts": dict(sorted(counts.items())),
        "results_log": str(out_path),
        "fuzzy_sample_log": str(fuzzy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-pass trainer/jockey rematch against rpscrape rows")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--input-glob", type=str, default="data/raw/rpscrape/**/*.csv")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_rematch(
        db_path=Path(args.db_path),
        input_glob=args.input_glob,
        dry_run=args.dry_run,
        year=args.year,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
