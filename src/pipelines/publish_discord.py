"""
Publish daily picks and results to Discord.

Flow (run as Step 4 of daily_run.sh, after picks are generated):
1. Read today's value bets from logs/daily_bets.csv
2. Run the race analysis agent on each (tool-using loop) -> verdict + analysis
3. CONFIRM/NEUTRAL  -> post to #daily-picks now
   FADE            -> queue to logs/pending_review.csv + ping #review
4. Generate + post a morning preview
5. If yesterday's bets have settled, post results + post-race summary to #results

Posting uses the Discord REST API directly (requests) — no persistent gateway
connection needed for one-shot posting. Re-uses track_pnl for settled results.

Usage:
    python -m src.pipelines.publish_discord --date today
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(ROOT / ".env")

from ingestion.db_connect import get_db
from pipelines.ai_agents import (
    run_race_analysis_agent,
    generate_morning_preview,
    generate_post_race_summary,
)
from pipelines import track_pnl

DB_PATH = str(ROOT / "racing.duckdb")
BETS_LOG = ROOT / "logs" / "daily_bets.csv"
PENDING_REVIEW = ROOT / "logs" / "pending_review.csv"
DISCORD_API = "https://discord.com/api/v10"

PENDING_COLS = [
    "date", "race_id", "runner_id", "horse", "course", "time", "category",
    "model_prob", "back_odds", "edge", "stake", "model_signals",
    "verdict", "confidence", "analysis",
]


def resolve_date(date_str: str) -> date:
    if date_str == "today":
        return date.today()
    if date_str == "yesterday":
        return date.today() - timedelta(days=1)
    return datetime.strptime(date_str, "%Y-%m-%d").date()


# ── Discord posting (REST, one-shot) ─────────────────────────────────

def discord_post(channel_id: str, content: str) -> None:
    """Post a message to a channel, chunking to respect the 2000-char limit."""
    token = os.environ["DISCORD_BOT_TOKEN"]
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    url = f"{DISCORD_API}/channels/{channel_id}/messages"

    for chunk in _chunk_message(content):
        resp = requests.post(url, headers=headers, json={"content": chunk}, timeout=30)
        if resp.status_code == 429:  # rate limited
            retry_after = resp.json().get("retry_after", 1)
            time.sleep(float(retry_after) + 0.5)
            resp = requests.post(url, headers=headers, json={"content": chunk}, timeout=30)
        resp.raise_for_status()
        time.sleep(0.4)  # gentle pacing between chunks


def _chunk_message(content: str, limit: int = 1900) -> list[str]:
    """Split on line boundaries so a message never exceeds Discord's limit."""
    if len(content) <= limit:
        return [content]
    chunks, current = [], ""
    for line in content.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


# ── Bet enrichment (look up IDs the agent needs) ─────────────────────

def enrich_bets_with_ids(bets: pd.DataFrame, db_path: str) -> list[dict[str, Any]]:
    """Attach horse_id, trainer_id, jockey_id, course_id, going_code, race_type per bet."""
    con = get_db(db_path)
    enriched = []
    try:
        for _, row in bets.iterrows():
            runner_id = row["runner_id"]
            race_id = row["race_id"]
            meta = con.execute(
                """
                SELECT ru.horse_id, ru.trainer_id, ru.jockey_id,
                       ra.course_id, ra.going_code, ra.race_type
                FROM runners ru
                JOIN races ra ON ru.race_id = ra.race_id
                WHERE ru.runner_id = ?
                """,
                [runner_id],
            ).fetchone()
            bet = row.to_dict()
            bet["race_time"] = row.get("time")
            if meta:
                bet.update({
                    "horse_id": meta[0], "trainer_id": meta[1], "jockey_id": meta[2],
                    "course_id": meta[3], "going_code": meta[4], "race_type": meta[5],
                })
            enriched.append(bet)
    finally:
        con.close()
    return enriched


# ── P&L context for the commentary agent ─────────────────────────────

def pnl_context_string() -> str:
    """Short running-P&L summary from logs/pnl_tracker.csv (best-effort)."""
    path = ROOT / "logs" / "pnl_tracker.csv"
    if not path.exists():
        return "No P&L history yet."
    try:
        df = pd.read_csv(path)
        if df.empty:
            return "No P&L history yet."
        last = df.iloc[-1]
        return (
            f"Running P&L {last['cumulative_profit']:+.1f}u over "
            f"{int(last['cumulative_staked'])} bets, "
            f"cumulative ROI {last['cumulative_roi_pct']:+.1f}%."
        )
    except Exception:
        return "P&L history unavailable."


# ── Formatting ───────────────────────────────────────────────────────

CATEGORY_EMOJI = {"flat": "🟢", "chase": "🟠", "hurdle": "🔵"}


def format_picks_message(target_date: date, posted: list[dict], preview: str) -> str:
    n = len(posted)
    by_cat: dict[str, int] = {}
    for b in posted:
        by_cat[b.get("category", "?")] = by_cat.get(b.get("category", "?"), 0) + 1
    cat_summary = ", ".join(f"{v} {k}" for k, v in by_cat.items())

    lines = [
        f"🏇 **DAILY PICKS — {target_date:%d %b %Y}**",
        "",
        f"📊 {n} value bet{'s' if n != 1 else ''} today ({cat_summary})",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    for b in sorted(posted, key=lambda x: x.get("edge", 0), reverse=True):
        emoji = CATEGORY_EMOJI.get(b.get("category"), "🏇")
        lines += [
            "",
            f"{emoji} **{b.get('race_time')} {b.get('course')}** [{b.get('category', '').upper()}]",
            f"**{str(b.get('horse', '')).upper()}** — Back {b.get('back_odds')} | "
            f"Model {b.get('model_prob', 0):.0%} | Edge +{b.get('edge', 0):.0%}",
            f"🔍 Analyst: {b.get('analysis_text', '')}",
        ]
    lines += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "", f"💬 {preview}"]
    return "\n".join(lines)


def format_review_message(target_date: date, faded: list[dict]) -> str:
    lines = [
        f"⚠️ **{len(faded)} pick(s) FADED by the analyst — review needed** ({target_date:%d %b %Y})",
        "Run `python -m src.pipelines.review_pending` to accept or reject.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    for b in faded:
        lines += [
            "",
            f"🚫 **{str(b.get('horse', '')).upper()}** — {b.get('race_time')} {b.get('course')} "
            f"[{b.get('category', '').upper()}]",
            f"Back {b.get('back_odds')} | Model {b.get('model_prob', 0):.0%} | "
            f"Edge +{b.get('edge', 0):.0%} | Confidence: {b.get('confidence', '?')}",
        ]
        signals = b.get("model_signals")
        if isinstance(signals, str) and signals.strip():
            lines.append(f"📈 Model rated it on: {signals.strip()}")
        lines.append(f"🔍 Analyst (FADE): {b.get('analysis_text', '')}")
    return "\n".join(lines)


def format_results_message(target_date: date, settled: list[dict], summary: str) -> str:
    total_staked = sum(r.get("stake", 1.0) for r in settled)
    total_profit = sum(r.get("profit", 0.0) for r in settled)
    total_profit_bsp = sum(r.get("profit_bsp", 0.0) for r in settled)
    winners = sum(1 for r in settled if r.get("won"))
    roi = total_profit / total_staked * 100 if total_staked else 0
    roi_bsp = total_profit_bsp / total_staked * 100 if total_staked else 0

    lines = [f"📊 **RESULTS — {target_date:%d %b %Y}**", ""]
    for r in sorted(settled, key=lambda x: str(x.get("time", ""))):
        mark = "✅" if r.get("won") else "❌"
        outcome = "WON" if r.get("won") else "LOST"
        sp = f"{r.get('sp'):.1f}" if r.get("sp") else "-"
        back = f"{r.get('back'):.1f}" if r.get("back") else "-"
        lines.append(
            f"{mark} {r.get('horse')} ({r.get('course')} {r.get('time')}) — "
            f"Back {back} (SP {sp}) — {outcome} {r.get('profit', 0):+.2f}u"
        )
    lines += [
        "",
        f"Day: {len(settled)} bets | {winners}W | P&L: {total_profit:+.2f}u | ROI: {roi:+.1f}% "
        f"(BSP {total_profit_bsp:+.2f}u / {roi_bsp:+.1f}%)",
        "",
        f"💬 {summary}",
    ]
    return "\n".join(lines)


# ── Pending review queue ─────────────────────────────────────────────

def remove_from_bets_log(faded: list[dict], target_date: date) -> None:
    """Remove FADE rows from daily_bets.csv so they don't count toward P&L until accepted.

    Matched by (date, runner_id). They live in pending_review.csv until reviewed;
    review_pending.py re-adds them on accept.
    """
    if not BETS_LOG.exists() or not faded:
        return
    faded_keys = {(str(target_date), b.get("runner_id")) for b in faded}
    df = pd.read_csv(BETS_LOG)
    keep = ~df.apply(
        lambda r: (str(pd.to_datetime(r["date"]).date()), r["runner_id"]) in faded_keys,
        axis=1,
    )
    df[keep].to_csv(BETS_LOG, index=False)


def append_pending(faded: list[dict]) -> None:
    PENDING_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PENDING_REVIEW.exists()
    with open(PENDING_REVIEW, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PENDING_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for b in faded:
            writer.writerow({
                "date": b.get("date"), "race_id": b.get("race_id"),
                "runner_id": b.get("runner_id"), "horse": b.get("horse"),
                "course": b.get("course"), "time": b.get("race_time") or b.get("time"),
                "category": b.get("category"), "model_prob": b.get("model_prob"),
                "back_odds": b.get("back_odds"), "edge": b.get("edge"),
                "stake": b.get("stake", 1.0), "model_signals": b.get("model_signals", ""),
                "verdict": b.get("verdict"),
                "confidence": b.get("confidence"), "analysis": b.get("analysis_text"),
            })


# ── Results (yesterday) ──────────────────────────────────────────────

def build_settled_results(target_date: date, db_path: str) -> list[dict]:
    """Reuse track_pnl to get settled bets for a date as a list of dicts."""
    bets = track_pnl.load_bets(target_date, target_date)
    if bets.empty:
        return []
    bets = track_pnl.apply_race_type_categories(bets, db_path)
    results = track_pnl.fetch_results(
        bets["runner_id"].tolist(), bets["runner_id_norm"].tolist(), db_path
    )
    pnl = track_pnl.compute_pnl(bets, results)
    settled = pnl[pnl["settled"]]
    out = []
    for _, r in settled.iterrows():
        out.append({
            "horse": r.get("horse"), "course": r.get("course"), "time": r.get("time"),
            "sp": r.get("sp"),
            "back": float(r["back_odds"]) if pd.notna(r.get("back_odds")) else None,
            "won": bool(r.get("won_actual")),
            "profit": float(r.get("profit", 0)), "profit_bsp": float(r.get("profit_bsp", 0)),
            "stake": float(r.get("stake", 1.0)),
        })
    return out


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Publish picks and results to Discord")
    parser.add_argument("--date", type=str, default="today")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--skip-results", action="store_true", help="Don't post yesterday's results")
    args = parser.parse_args()

    db_path = args.db or DB_PATH
    target_date = resolve_date(args.date)

    picks_channel = os.environ["DISCORD_PICKS_CHANNEL_ID"]
    results_channel = os.environ["DISCORD_RESULTS_CHANNEL_ID"]
    review_channel = os.environ.get("DISCORD_REVIEW_CHANNEL_ID")

    # ── Today's picks ──
    bets_df = track_pnl.load_bets(target_date, target_date)
    if bets_df.empty:
        print(f"No bets logged for {target_date}; nothing to post.", flush=True)
    else:
        print(f"Analysing {len(bets_df)} bets for {target_date}...", flush=True)
        enriched = enrich_bets_with_ids(bets_df, db_path)

        posted, faded = [], []
        for bet in enriched:
            verdict = run_race_analysis_agent(bet, db_path)
            bet.update(verdict)
            print(f"  {bet.get('horse')}: {verdict['verdict']} "
                  f"(tools: {', '.join(verdict['tools_used']) or 'none'})", flush=True)
            if verdict["verdict"] == "FADE":
                faded.append(bet)
            else:
                posted.append(bet)

        if posted:
            preview = generate_morning_preview(posted, pnl_context_string())
            discord_post(picks_channel, format_picks_message(target_date, posted, preview))
            print(f"  Posted {len(posted)} picks to #daily-picks", flush=True)

        if faded:
            append_pending(faded)
            remove_from_bets_log(faded, target_date)
            if review_channel:
                discord_post(review_channel, format_review_message(target_date, faded))
            print(f"  Queued {len(faded)} FADE picks for review (removed from bet log)", flush=True)

    # ── Yesterday's results ──
    if not args.skip_results:
        yesterday = target_date - timedelta(days=1)
        settled = build_settled_results(yesterday, db_path)
        if settled:
            summary = generate_post_race_summary(settled, pnl_context_string())
            discord_post(results_channel, format_results_message(yesterday, settled, summary))
            print(f"  Posted {len(settled)} results for {yesterday} to #results", flush=True)
        else:
            print(f"  No settled results for {yesterday} to post", flush=True)


if __name__ == "__main__":
    main()
