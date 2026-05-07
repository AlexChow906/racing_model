from __future__ import annotations

import re
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
ALIASES_PATH = ROOT / "configs" / "course_aliases.yaml"


def _load_course_aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(ALIASES_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    aliases: dict[str, str] = {}
    for key, value in payload.items():
        if key is None or value is None:
            continue
        aliases[str(key).strip().lower()] = str(value).strip()
    return aliases


_ALIASES = _load_course_aliases()
_MISSING_ALIAS_LOGGED: set[str] = set()
_BETFAIR_EVENT_RE = re.compile(r"^(?:gb|ire|ie|fr)\s*/\s*([a-z]+)", re.IGNORECASE)
_BETFAIR_VENUE_TOKENS: dict[str, str] = {
    "leop": "Leopardstown",
    "curr": "Curragh",
    "gowr": "Gowran Park",
    "gowp": "Gowran Park",
    "navan": "Navan",
    "punc": "Punchestown",
    "punch": "Punchestown",
    "fair": "Fairyhouse",
    "tipp": "Tipperary",
    "cork": "Cork",
    "kill": "Killarney",
    "down": "Down Royal",
    "naas": "Naas",
    "limk": "Limerick",
    "galw": "Galway",
    "sligo": "Sligo",
    "tramore": "Tramore",
    "chelt": "Cheltenham",
    "ascot": "Ascot",
    "good": "Goodwood",
    "newm": "Newmarket",
    "donc": "Doncaster",
    "hayd": "Haydock",
    "newb": "Newbury",
    "sand": "Sandown",
    "wind": "Windsor",
    "leic": "Leicester",
    "nott": "Nottingham",
    "wolv": "Wolverhampton",
    "kemp": "Kempton",
    "chms": "Chelmsford",
    "ling": "Lingfield",
    "redc": "Redcar",
    "ayr": "Ayr",
    "carl": "Carlisle",
    "muss": "Musselburgh",
    "ches": "Chester",
    "bev": "Beverley",
    "ripo": "Ripon",
    "thir": "Thirsk",
    "pont": "Pontefract",
    "catt": "Catterick",
    "brig": "Brighton",
    "epso": "Epsom",
    "bath": "Bath",
    "sali": "Salisbury",
    "yarm": "Yarmouth",
    "nwcs": "Newcastle",
    "newc": "Newcastle",
    "sthl": "Southwell",
    "chep": "Chepstow",
    "wrcr": "Worcester",
    "worc": "Worcester",
    "ludl": "Ludlow",
    "uttx": "Uttoxeter",
    "warw": "Warwick",
    "stfd": "Stratford",
    "plum": "Plumpton",
    "font": "Fontwell",
    "wins": "Wincanton",
    "towc": "Towcester",
    "hexm": "Hexham",
    "kelso": "Kelso",
    "perth": "Perth",
    "bang": "Bangor",
    "cart": "Cartmel",
    "here": "Hereford",
    "mark": "Market Rasen",
    "mket": "Market Rasen",
    "hunt": "Huntingdon",
    "exet": "Exeter",
    "taun": "Taunton",
    "nwtn": "Newton Abbot",
    "nton": "Newton Abbot",
}


def slugify(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_") or None


def normalise_course(name: str | None) -> str:
    if not name:
        return ""
    raw = str(name).strip()
    event_match = _BETFAIR_EVENT_RE.match(raw)
    if event_match:
        token = event_match.group(1).lower()
        resolved = _BETFAIR_VENUE_TOKENS.get(token)
        if resolved:
            return resolved
        if token in _ALIASES:
            return _ALIASES[token]
        if token not in _MISSING_ALIAS_LOGGED:
            _MISSING_ALIAS_LOGGED.add(token)
            LOGGER.warning("unknown_betfair_venue_token token=%s raw=%s", token, raw)
        return token

    cleaned = (
        raw
        .lower()
        .replace("(aw)", "")
        .replace("(ire)", "")
        .replace("(gb)", "")
        .replace("tapeta", "")
        .replace("polytrack", "")
    )
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"^(gb|ire|ie)\s*/\s*", "", cleaned)
    cleaned = re.sub(r"\b\d{1,2}(st|nd|rd|th)?\b", "", cleaned)
    cleaned = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    token = cleaned.split(" ")[0] if cleaned else ""
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    if token in _ALIASES:
        return _ALIASES[token]

    log_key = token or cleaned
    if log_key and log_key not in _MISSING_ALIAS_LOGGED:
        _MISSING_ALIAS_LOGGED.add(log_key)
        LOGGER.warning("course_alias_missing cleaned=%s token=%s", cleaned, token)
    return cleaned


def normalise_horse(name: str | None) -> str:
    if not name:
        return ""
    text = str(name).strip()
    if not text:
        return ""

    text = re.sub(r"(?<=[A-Za-z])I(?=\s*\()", "", text)
    text = re.sub(r"\bI\b(?=\s*\()", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(?:IRE|GB|FR|USA|AUS|GER|ITY)\b$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return slugify(text) or ""


def parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if value is None:
        return datetime.now(timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def decision_cutoff_for_off_time(scheduled_off_utc: datetime) -> datetime:
    off = parse_utc(scheduled_off_utc)
    evening_before = (off - timedelta(days=1)).date()
    return datetime(evening_before.year, evening_before.month, evening_before.day, 21, 0, 0, tzinfo=timezone.utc)


def canonical_race_id_from_market(market_id: str) -> str:
    return f"bf_{market_id}"


def canonical_runner_id(race_id: str, horse_id: str) -> str:
    return f"{race_id}_{horse_id}"


def upsert_ignore(con: duckdb.DuckDBPyConnection, table_name: str, records: list[dict], columns: list[str]) -> int:
    if not records:
        return 0

    frame = pd.DataFrame(records)
    for col in columns:
        if col not in frame.columns:
            frame[col] = None

    temp_name = f"tmp_{table_name}"
    con.register(temp_name, frame[columns])
    column_csv = ", ".join(columns)
    con.execute(
        f"""
        INSERT OR IGNORE INTO {table_name} ({column_csv})
        SELECT {column_csv}
        FROM {temp_name}
        """
    )
    return len(records)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
