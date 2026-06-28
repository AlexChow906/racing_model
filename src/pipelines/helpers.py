"""Shared helpers for the daily pipeline modules.

Small, dependency-light utilities that several pipeline scripts need (date
resolution, the daily-log CSV schema, and the append/de-dup write used for the
dated log files). Kept free of model/Betfair/Discord imports so importing it is
cheap and never risks a circular import.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BETS_LOG_COLS = [
    "date", "race_id", "runner_id", "horse", "course", "time",
    "category", "model_prob", "back_odds", "edge", "stake", "model_signals",
]


def resolve_date(date_str: str) -> date:
    """Resolve a CLI date argument to a date.

    Accepts the relative keywords 'today', 'tomorrow' and 'yesterday', or an
    explicit YYYY-MM-DD string.
    """
    if date_str == "today":
        return date.today()
    if date_str == "tomorrow":
        return date.today() + timedelta(days=1)
    if date_str == "yesterday":
        return date.today() - timedelta(days=1)
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def append_dated_csv(
    log_path: Path,
    new_rows: pd.DataFrame,
    target_date: date,
    cols: list[str],
    refresh: bool = False,
    label: str = "rows",
) -> bool:
    """Append new_rows to a CSV keyed by a 'date' column, de-duplicating by date.

    Pandas-based so the schema auto-migrates: reindexing to `cols` backfills any
    columns missing from an older CSV rather than corrupting on append. A date
    already present is skipped unless refresh=True, which replaces that date's
    rows. `label` is only used in the printed status lines. Returns True if it
    wrote, False if it skipped.
    """
    if new_rows.empty:
        return False
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        existing = pd.read_csv(log_path)
        already = (pd.to_datetime(existing["date"]).dt.date == target_date).any()
        if already and not refresh:
            print(f"  {label.capitalize()} for {target_date} already logged (skipping)", flush=True)
            return False
        if already and refresh:
            existing = existing[pd.to_datetime(existing["date"]).dt.date != target_date]
            print(f"  Refreshing {label} for {target_date}", flush=True)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.reindex(columns=cols).to_csv(log_path, index=False)
    print(f"  Logged {len(new_rows)} {label} to {log_path}", flush=True)
    return True
