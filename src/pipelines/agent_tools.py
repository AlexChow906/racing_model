"""
Tools for the race analysis agent.

Each tool is a DuckDB-backed lookup the LLM can call to investigate a value bet.
The agent decides which tools to call and in what order; the dispatcher executes
them against the racing database and returns plain dicts the LLM can read.

All history tables only contain prior runs (the feature store enforces decision
cutoffs upstream), so there is no leakage concern reading them here for context.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db

DB_PATH = str(ROOT / "racing.duckdb")


def _rows_to_dicts(cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_horse_history(db_path: str, horse_id: str, limit: int = 10) -> dict[str, Any]:
    """Last N runs for a horse: position, course, going, race type, SP, beaten lengths."""
    con = get_db(db_path)
    try:
        cur = con.execute(
            """
            SELECT race_date, course_id, going_code, race_type, race_class,
                   finishing_position, field_size, btn_lengths, rpr, non_completion
            FROM horse_history
            WHERE horse_id = ?
            ORDER BY scheduled_off_utc DESC
            LIMIT ?
            """,
            [horse_id, limit],
        )
        runs = _rows_to_dicts(cur)
    finally:
        con.close()

    if not runs:
        return {"horse_id": horse_id, "runs": [], "note": "no prior runs found"}

    # A run with no finishing position and no non-completion code is a real start
    # whose result is simply missing from our data — the race has been run and has
    # a result in reality, we just haven't ingested it. Flag it, never drop it.
    for r in runs:
        r["result_missing"] = (
            r.get("finishing_position") is None and not r.get("non_completion")
        )

    resulted = [r for r in runs if not r["result_missing"]]
    wins = sum(1 for r in resulted if r.get("finishing_position") == 1)
    placed = sum(1 for r in resulted if r.get("finishing_position") in (1, 2, 3))
    non_completions = sum(1 for r in runs if r.get("non_completion"))
    return {
        "horse_id": horse_id,
        "runs_shown": len(runs),
        "wins": wins,
        "placed": placed,
        "non_completions": non_completions,
        "missing_results": sum(1 for r in runs if r["result_missing"]),
        "runs": runs,
    }


def get_trainer_form(db_path: str, trainer_id: str, course_id: str | None = None,
                     going_code: str | None = None) -> dict[str, Any]:
    """Trainer win rates: overall, at this course, and on this going."""
    con = get_db(db_path)
    try:
        overall = con.execute(
            "SELECT COUNT(*) AS runs, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins "
            "FROM trainer_history WHERE trainer_id = ?",
            [trainer_id],
        ).fetchone()
        course = None
        if course_id:
            course = con.execute(
                "SELECT COUNT(*) AS runs, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins "
                "FROM trainer_history WHERE trainer_id = ? AND course_id = ?",
                [trainer_id, course_id],
            ).fetchone()
        going = None
        if going_code:
            going = con.execute(
                "SELECT COUNT(*) AS runs, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins "
                "FROM trainer_history WHERE trainer_id = ? AND going_code = ?",
                [trainer_id, going_code],
            ).fetchone()
    finally:
        con.close()

    def _rate(row):
        if not row or not row[0]:
            return None
        return {"runs": int(row[0]), "wins": int(row[1] or 0),
                "win_rate": round((row[1] or 0) / row[0], 3)}

    return {
        "trainer_id": trainer_id,
        "overall": _rate(overall),
        "at_course": _rate(course),
        "on_going": _rate(going),
    }


def get_jockey_form(db_path: str, jockey_id: str, course_id: str | None = None) -> dict[str, Any]:
    """Jockey win rates: overall and at this course."""
    con = get_db(db_path)
    try:
        overall = con.execute(
            "SELECT COUNT(*) AS runs, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins "
            "FROM jockey_history WHERE jockey_id = ?",
            [jockey_id],
        ).fetchone()
        course = None
        if course_id:
            course = con.execute(
                "SELECT COUNT(*) AS runs, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins "
                "FROM jockey_history WHERE jockey_id = ? AND course_id = ?",
                [jockey_id, course_id],
            ).fetchone()
    finally:
        con.close()

    def _rate(row):
        if not row or not row[0]:
            return None
        return {"runs": int(row[0]), "wins": int(row[1] or 0),
                "win_rate": round((row[1] or 0) / row[0], 3)}

    return {"jockey_id": jockey_id, "overall": _rate(overall), "at_course": _rate(course)}


def get_going_record(db_path: str, horse_id: str, going_code: str) -> dict[str, Any]:
    """How the horse has performed on a specific going."""
    con = get_db(db_path)
    try:
        cur = con.execute(
            """
            SELECT race_date, course_id, race_type, finishing_position, field_size, btn_lengths
            FROM horse_history
            WHERE horse_id = ? AND LOWER(going_code) = LOWER(?)
            ORDER BY scheduled_off_utc DESC
            """,
            [horse_id, going_code],
        )
        runs = _rows_to_dicts(cur)
    finally:
        con.close()

    wins = sum(1 for r in runs if r.get("finishing_position") == 1)
    placed = sum(1 for r in runs if r.get("finishing_position") in (1, 2, 3))
    return {
        "horse_id": horse_id,
        "going_code": going_code,
        "runs": len(runs),
        "wins": wins,
        "placed": placed,
        "detail": runs[:8],
    }


def get_course_record(db_path: str, horse_id: str, course_id: str) -> dict[str, Any]:
    """How the horse has performed at a specific course.

    Betfair course_ids embed the meeting date (e.g. 'limerick_31st_mar'), so we
    match on the venue prefix by stripping the trailing '_<day>_<month>' suffix
    (same normalisation the feature SQL uses).
    """
    venue = re.sub(r"_\d+\w*_\w+$", "", course_id)
    con = get_db(db_path)
    try:
        cur = con.execute(
            r"""
            SELECT race_date, going_code, race_type, finishing_position, field_size, btn_lengths
            FROM horse_history
            WHERE horse_id = ?
              AND REGEXP_REPLACE(course_id, '_\d+\w*_\w+$', '') = ?
            ORDER BY scheduled_off_utc DESC
            """,
            [horse_id, venue],
        )
        runs = _rows_to_dicts(cur)
    finally:
        con.close()

    wins = sum(1 for r in runs if r.get("finishing_position") == 1)
    placed = sum(1 for r in runs if r.get("finishing_position") in (1, 2, 3))
    return {
        "horse_id": horse_id,
        "course_id": course_id,
        "runs": len(runs),
        "wins": wins,
        "placed": placed,
        "detail": runs[:8],
    }


def get_race_field(db_path: str, race_id: str) -> dict[str, Any]:
    """Other runners in the race: names, official ratings, age, draw."""
    con = get_db(db_path)
    try:
        cur = con.execute(
            """
            SELECT horse_name, official_rating, age, draw, trainer_name, jockey_name
            FROM runners
            WHERE race_id = ?
            ORDER BY official_rating DESC NULLS LAST
            """,
            [race_id],
        )
        runners = _rows_to_dicts(cur)
    finally:
        con.close()

    return {"race_id": race_id, "field_size": len(runners), "runners": runners}


# ── Tool schemas (OpenAI function-calling format, works on Groq/Claude/OpenAI) ──

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_horse_history",
            "description": "Get the horse's last 10 runs: finishing position, course, going, race type, RPR, beaten lengths, non-completions. A run flagged result_missing=true is a real start whose result our data does not have (the race has run) — include it when counting recent starts, but make no claim about how it finished.",
            "parameters": {
                "type": "object",
                "properties": {"horse_id": {"type": "string", "description": "The horse's ID"}},
                "required": ["horse_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trainer_form",
            "description": "Get the trainer's win rates overall, at this course, and on this going.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trainer_id": {"type": "string"},
                    "course_id": {"type": "string", "description": "Optional, for course-specific rate"},
                    "going_code": {"type": "string", "description": "Optional, for going-specific rate"},
                },
                "required": ["trainer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_jockey_form",
            "description": "Get the jockey's win rates overall and at this course.",
            "parameters": {
                "type": "object",
                "properties": {
                    "jockey_id": {"type": "string"},
                    "course_id": {"type": "string", "description": "Optional, for course-specific rate"},
                },
                "required": ["jockey_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_going_record",
            "description": "Get how the horse has performed on a specific going (ground condition).",
            "parameters": {
                "type": "object",
                "properties": {
                    "horse_id": {"type": "string"},
                    "going_code": {"type": "string", "description": "e.g. Good, Soft, Heavy"},
                },
                "required": ["horse_id", "going_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_record",
            "description": "Get how the horse has performed at a specific course.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horse_id": {"type": "string"},
                    "course_id": {"type": "string"},
                },
                "required": ["horse_id", "course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_field",
            "description": "Get the other runners in the race (names, official ratings, age) to gauge the strength of the field.",
            "parameters": {
                "type": "object",
                "properties": {"race_id": {"type": "string"}},
                "required": ["race_id"],
            },
        },
    },
]

_DISPATCH = {
    "get_horse_history": get_horse_history,
    "get_trainer_form": get_trainer_form,
    "get_jockey_form": get_jockey_form,
    "get_going_record": get_going_record,
    "get_course_record": get_course_record,
    "get_race_field": get_race_field,
}


def dispatch(tool_name: str, args: dict[str, Any], db_path: str = DB_PATH) -> dict[str, Any]:
    """Execute a tool by name with the given args. Returns a dict (or an error dict)."""
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        return fn(db_path, **args)
    except Exception as exc:  # defensive: never crash the agent loop on a bad call
        return {"error": f"{tool_name} failed: {exc}"}
