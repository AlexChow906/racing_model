"""
Track P&L from daily value bets against settled results.

Reads the bet log (logs/daily_bets.csv), joins with actual results from the DB,
and computes P&L assuming 1-unit level stakes at BSP.

Usage:
    python -m src.pipelines.track_pnl                          # all-time summary
    python -m src.pipelines.track_pnl --from 2026-05-01        # everything from May 1
    python -m src.pipelines.track_pnl --from 2026-05-01 --to 2026-05-07  # one week
    python -m src.pipelines.track_pnl --date 2026-05-15        # single day (bet details)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db

BETS_LOG = ROOT / "logs" / "daily_bets.csv"
PNL_OUTPUT = ROOT / "logs" / "pnl_tracker.csv"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_bets(date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    if not BETS_LOG.exists():
        return pd.DataFrame()

    df = pd.read_csv(BETS_LOG)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    return df


def fetch_results(runner_ids: list[str], db_path: str) -> pd.DataFrame:
    if not runner_ids:
        return pd.DataFrame()

    con = get_db(db_path)
    placeholders = ",".join(["?"] * len(runner_ids))
    df = con.execute(
        f"""SELECT runner_id, sp_decimal, won, finishing_position
            FROM results
            WHERE runner_id IN ({placeholders})""",
        runner_ids,
    ).df()
    con.close()
    return df


def compute_pnl(bets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    merged = bets.merge(results, on="runner_id", how="left", suffixes=("", "_actual"))

    merged["settled"] = merged["sp_decimal"].notna()
    merged["won_actual"] = merged["won"].fillna(False)
    merged["sp"] = merged["sp_decimal"].fillna(0)

    merged["profit"] = merged.apply(
        lambda r: (r["sp"] - 1) * r["stake"] if r["won_actual"] else -r["stake"]
        if r["settled"] else 0,
        axis=1,
    )

    return merged


def print_single_day(pnl: pd.DataFrame, target_date: date):
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print(f"No settled bets for {target_date}.")
        return

    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    winners = int(settled["won_actual"].sum())
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0

    print(f"\n  {target_date}")
    print(f"  {'-'*62}")
    print(f"  {'Horse':<22} {'Course':<12} {'Time':<6} {'Back':>5} {'SP':>6} {'W':>3} {'P&L':>7}")
    print(f"  {'-'*62}")

    for _, r in settled.sort_values("time").iterrows():
        horse = str(r.get("horse", "?"))[:21]
        course = str(r.get("course", ""))[:11]
        time_str = str(r.get("time", ""))[:5]
        back = f"{r['back_odds']:.1f}" if pd.notna(r.get("back_odds")) else "-"
        sp = f"{r['sp']:.1f}" if r["sp"] > 0 else "-"
        won_str = "Y" if r["won_actual"] else ""
        pnl_str = f"{r['profit']:+.2f}"
        print(f"  {horse:<22} {course:<12} {time_str:<6} {back:>5} {sp:>6} {won_str:>3} {pnl_str:>7}")

    print(f"  {'-'*62}")
    print(f"  {len(settled)} bets | {winners}W | P&L: {total_profit:+.2f}u | ROI: {roi:+.1f}%")


def print_range(pnl: pd.DataFrame, date_from: date | None, date_to: date | None):
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print("No settled bets in this range.")
        return

    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    winners = int(settled["won_actual"].sum())
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0

    label = "All-time"
    if date_from and date_to:
        label = f"{date_from} to {date_to}"
    elif date_from:
        label = f"From {date_from}"
    elif date_to:
        label = f"Up to {date_to}"

    print(f"\n  {label}")
    print(f"  Bets: {len(settled)}  |  P&L: {total_profit:+.2f}u  |  ROI: {roi:+.1f}%")

    pending = pnl[~pnl["settled"]]
    if not pending.empty:
        print(f"  ({len(pending)} bets still pending results)")


def main():
    parser = argparse.ArgumentParser(description="Track P&L from daily value bets")
    parser.add_argument("--date", type=str, default=None, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--from", type=str, default=None, dest="date_from", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", type=str, default=None, dest="date_to", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    if args.date:
        date_from = date_to = _parse_date(args.date)
    else:
        date_from = _parse_date(args.date_from) if args.date_from else None
        date_to = _parse_date(args.date_to) if args.date_to else None

    db_path = args.db or str(ROOT / "racing.duckdb")
    single_day = args.date is not None

    bets = load_bets(date_from, date_to)
    if bets.empty:
        print("No bets found in log. Run daily_predictions first.")
        return

    results = fetch_results(bets["runner_id"].tolist(), db_path)
    pnl = compute_pnl(bets, results)

    if pnl.empty or not pnl["settled"].any():
        print("No settled bets yet. Run collect_results to fetch actual SPs.")
        return

    if single_day:
        print_single_day(pnl, date_from)
    else:
        print_range(pnl, date_from, date_to)

        settled = pnl[pnl["settled"]].copy()
        daily = settled.groupby("date").agg(
            bets=("runner_id", "count"),
            winners=("won_actual", "sum"),
            total_staked=("stake", "sum"),
            total_profit=("profit", "sum"),
        ).reset_index()
        daily["roi_pct"] = (daily["total_profit"] / daily["total_staked"] * 100).round(1)
        daily["cumulative_profit"] = daily["total_profit"].cumsum().round(2)
        daily["cumulative_staked"] = daily["total_staked"].cumsum()
        daily["cumulative_roi_pct"] = (daily["cumulative_profit"] / daily["cumulative_staked"] * 100).round(1)

        PNL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(PNL_OUTPUT, index=False)


if __name__ == "__main__":
    main()
