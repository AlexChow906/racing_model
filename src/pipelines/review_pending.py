"""
Review FADE picks the analyst agent held back.

For each pending pick:
1. Re-fetch live odds (the model_prob is fixed — price-blind — only odds/edge move)
2. Recompute edge with the fresh odds
3. Prompt accept / reject:
   - accept (edge still >= threshold) -> post to #daily-picks, add back to daily_bets.csv
   - accept (edge below threshold)     -> warn value is gone, default to drop (override allowed)
   - reject                            -> drop (stays out of daily_bets.csv)

No re-scoring: model_prob is read from the pending row. Only the Betfair odds are
re-fetched (a quick market-book call), so this runs in seconds.

Usage:
    python -m src.pipelines.review_pending
    python -m src.pipelines.review_pending --min-edge 0.15
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(ROOT / ".env")

from pipelines.daily_predictions import fetch_race_cards, fetch_live_odds
from pipelines.ai_agents import generate_reviewed_pick_note
from pipelines import publish_discord
from pipelines.helpers import BETS_LOG_COLS

BETS_LOG = ROOT / "logs" / "daily_bets.csv"
PENDING_REVIEW = ROOT / "logs" / "pending_review.csv"


def _market_id_from_race_id(race_id: str) -> str:
    """bfsp_1.258261792_win -> 1.258261792"""
    return race_id.replace("bfsp_", "").rsplit("_win", 1)[0]


def _build_selection_map(target_date: date) -> dict[str, int]:
    """runner_id -> selection_id, fetched fresh from the Betfair catalogue."""
    _, runners = fetch_race_cards(target_date)
    return {r["runner_id"]: r["selection_id"] for r in runners if r.get("selection_id")}


def _refetch_odds(pending: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each pending runner_id, return {runner_id: {back, selection_id}} with fresh odds."""
    out: dict[str, dict[str, Any]] = {}
    for target_date in sorted(pending["date"].unique()):
        day = pd.to_datetime(target_date).date()
        day_rows = pending[pending["date"] == target_date]
        sel_map = _build_selection_map(day)
        market_ids = sorted({_market_id_from_race_id(rid) for rid in day_rows["race_id"]})
        odds = fetch_live_odds(market_ids)
        for _, row in day_rows.iterrows():
            sel = sel_map.get(row["runner_id"])
            back = odds.get(sel, {}).get("back") if sel else None
            out[row["runner_id"]] = {"selection_id": sel, "back": back}
    return out


def _append_to_bets_log(row: dict[str, Any]) -> None:
    BETS_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BETS_LOG.exists()
    with open(BETS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BETS_LOG_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in BETS_LOG_COLS})


def _rewrite_pending(remaining: pd.DataFrame) -> None:
    if remaining.empty:
        PENDING_REVIEW.unlink(missing_ok=True)
    else:
        remaining.to_csv(PENDING_REVIEW, index=False)


def main():
    parser = argparse.ArgumentParser(description="Review FADE picks")
    parser.add_argument("--min-edge", type=float, default=0.15)
    args = parser.parse_args()

    if not PENDING_REVIEW.exists():
        print("No pending picks to review.")
        return

    pending = pd.read_csv(PENDING_REVIEW)
    if pending.empty:
        print("No pending picks to review.")
        PENDING_REVIEW.unlink(missing_ok=True)
        return

    print(f"Re-fetching live odds for {len(pending)} pending pick(s)...", flush=True)
    try:
        fresh = _refetch_odds(pending)
    except Exception as exc:
        print(f"Could not fetch live odds: {exc}")
        return

    picks_channel = os.environ["DISCORD_PICKS_CHANNEL_ID"]
    resolved_rows = set()  # runner_ids that have been decided

    for _, row in pending.iterrows():
        rid = row["runner_id"]
        model_prob = float(row["model_prob"])
        old_odds = float(row["back_odds"])
        fresh_info = fresh.get(rid, {})
        new_back = fresh_info.get("back")

        print("\n" + "=" * 64)
        print(f"  {str(row['horse']).upper()} — {row['time']} {row['course']} "
              f"[{str(row['category']).upper()}]")
        print(f"  Model: {model_prob:.0%} | Original odds: {old_odds} "
              f"(edge +{float(row['edge']):.0%})")
        print(f"  Analyst FADE reasoning: {row.get('analysis', '')}")

        if not new_back:
            print("  ⚠️  No live odds (non-runner or market closed). Skipping — stays pending.")
            continue

        new_edge = model_prob - 1.0 / new_back
        moved = "shortened" if new_back < old_odds else "drifted"
        print(f"  Fresh odds: {new_back} ({moved} from {old_odds}) | New edge: {new_edge:+.0%}")

        if new_edge < args.min_edge:
            print(f"  ⚠️  Value gone — edge {new_edge:+.0%} below {args.min_edge:.0%} threshold.")

        choice = input("  [a]ccept / [r]eject / [s]kip? ").strip().lower()

        if choice == "a":
            if new_edge < args.min_edge:
                override = input("  Edge below threshold. Post anyway? [y/N] ").strip().lower()
                if override != "y":
                    print("  Dropped (value gone).")
                    resolved_rows.add(rid)
                    continue
            accepted = row.to_dict()
            accepted["back_odds"] = round(new_back, 2)
            accepted["edge"] = round(new_edge, 4)
            accepted["race_time"] = row["time"]
            # The stored analysis is the analyst's FADE critique. This pick was
            # human-approved, so post a fresh neutral note instead of recycling it.
            try:
                accepted["analysis_text"] = generate_reviewed_pick_note(accepted)
            except Exception as exc:
                print(f"  (couldn't generate a fresh note: {exc})")
                accepted["analysis_text"] = "Reviewed and approved as a late addition."
            _append_to_bets_log(accepted)
            msg = publish_discord.format_picks_message(
                pd.to_datetime(row["date"]).date(), [accepted],
                "Late addition — reviewed and approved.",
            )
            publish_discord.discord_post(picks_channel, msg)
            print("  ✅ Accepted — posted to #daily-picks, added to bet log.")
            resolved_rows.add(rid)
        elif choice == "r":
            print("  ❌ Rejected — dropped (won't count toward P&L).")
            resolved_rows.add(rid)
        else:
            print("  Skipped — stays pending.")

    remaining = pending[~pending["runner_id"].isin(resolved_rows)]
    _rewrite_pending(remaining)
    print(f"\nDone. {len(resolved_rows)} resolved, {len(remaining)} still pending.")


if __name__ == "__main__":
    main()
