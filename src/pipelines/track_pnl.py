"""
Track P&L from daily value bets against settled results.

Reads the bet log (logs/daily_bets.csv), joins with actual results from the DB,
and computes P&L at the backed price (the scraped odds the edge was judged on),
keeping BSP alongside as a benchmark. 1-unit level stakes.

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


def _normalize_bfsp_id(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return value
    text = str(value)
    return text.replace("bfsp_1.", "bfsp_")


def load_bets(date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    if not BETS_LOG.exists():
        return pd.DataFrame()

    df = pd.read_csv(BETS_LOG)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    df["race_id_norm"] = df["race_id"].apply(_normalize_bfsp_id)
    df["runner_id_norm"] = df["runner_id"].apply(_normalize_bfsp_id)
    return df


def fetch_results(runner_ids: list[str], runner_ids_norm: list[str], db_path: str) -> pd.DataFrame:
    if not runner_ids and not runner_ids_norm:
        return pd.DataFrame()

    con = get_db(db_path)
    all_ids = [rid for rid in runner_ids if rid] + [rid for rid in runner_ids_norm if rid]
    placeholders = ",".join(["?"] * len(all_ids))
    df = con.execute(
        f"""SELECT runner_id, sp_decimal, won, finishing_position
            FROM results
            WHERE runner_id IN ({placeholders})""",
        all_ids,
    ).df()
    con.close()
    if df.empty:
        return df

    df["runner_id_norm"] = df["runner_id"].apply(_normalize_bfsp_id)
    df = df.sort_values(by=["sp_decimal"], ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["runner_id_norm"], keep="first")
    return df


def apply_race_type_categories(bets: pd.DataFrame, db_path: str) -> pd.DataFrame:
    if bets.empty:
        return bets

    race_ids = bets["race_id"].dropna().unique().tolist()
    race_ids_norm = bets["race_id_norm"].dropna().unique().tolist()
    if not race_ids:
        return bets

    con = get_db(db_path)
    all_ids = [rid for rid in race_ids if rid] + [rid for rid in race_ids_norm if rid]
    placeholders = ",".join(["?"] * len(all_ids))
    races = con.execute(
        f"""SELECT race_id, race_type
            FROM races
            WHERE race_id IN ({placeholders})""",
        all_ids,
    ).df()
    con.close()

    if races.empty:
        return bets

    races["race_id_norm"] = races["race_id"].apply(_normalize_bfsp_id)
    race_type_map = dict(zip(races["race_id_norm"], races["race_type"]))

    def _map_category(race_id: str, fallback: str | None) -> str | None:
        race_type = race_type_map.get(race_id)
        if race_type == "Chase":
            return "chase"
        if race_type in ("Hurdle", "NH Flat"):
            return "hurdle"
        if race_type == "Flat":
            return "flat"
        return fallback

    bets = bets.copy()
    bets["category_orig"] = bets["category"]
    bets["category"] = [
        _map_category(race_id, cat)
        for race_id, cat in zip(bets["race_id_norm"], bets["category"], strict=False)
    ]
    return bets


def _settle_profit(odds: float, won: bool, stake: float) -> float:
    """Profit for a settled bet at decimal `odds`: (odds-1)*stake on a win, else -stake."""
    return (odds - 1) * stake if won else -stake


def compute_pnl(bets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    merged = bets.merge(results, on="runner_id_norm", how="left", suffixes=("", "_actual"))

    merged["settled"] = merged["sp_decimal"].notna()
    merged["won_actual"] = merged["won"].fillna(False)
    merged["sp"] = merged["sp_decimal"].fillna(0)

    # Headline P&L settles at the price we actually backed (the scraped odds the edge
    # was judged on); profit_bsp keeps the BSP figure as the backtest benchmark. Fall
    # back to SP only if a back price is missing, so no settled bet is silently dropped.
    def _profit(r, odds_col):
        if not r["settled"]:
            return 0.0
        odds = r[odds_col]
        if pd.isna(odds) or odds <= 0:
            odds = r["sp"]
        return _settle_profit(odds, r["won_actual"], r["stake"])

    merged["profit"] = merged.apply(lambda r: _profit(r, "back_odds"), axis=1)
    merged["profit_bsp"] = merged.apply(lambda r: _profit(r, "sp"), axis=1)

    return merged


def _print_category_summary(settled: pd.DataFrame, label: str, indent: int = 0):
    staked = settled["stake"].sum()
    profit = settled["profit"].sum()
    profit_bsp = settled["profit_bsp"].sum()
    roi = profit / staked * 100 if staked > 0 else 0
    roi_bsp = profit_bsp / staked * 100 if staked > 0 else 0
    prefix = " " * indent
    print(f"  {prefix}{label:<10} Bets: {len(settled):<5} |  "
          f"P&L: {profit:+.2f}u (ROI {roi:+.1f}%)  |  BSP: {profit_bsp:+.2f}u (ROI {roi_bsp:+.1f}%)")


def print_single_day(pnl: pd.DataFrame, target_date: date):
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print(f"No settled bets for {target_date}.")
        return

    print(f"\n  {target_date}")

    for cat in sorted(settled["category"].dropna().unique()):
        cat_settled = settled[settled["category"] == cat]
        print(f"  {'-'*62}")
        print(f"  {cat.upper()}")
        print(f"  {'Horse':<22} {'Course':<12} {'Time':<6} {'Back':>5} {'SP':>6} {'W':>3} {'P&L':>8} {'BSP':>8}")
        print(f"  {'-'*70}")

        for _, r in cat_settled.sort_values("time").iterrows():
            horse = str(r.get("horse", "?"))[:21]
            course = str(r.get("course", ""))[:11]
            time_str = str(r.get("time", ""))[:5]
            back = f"{r['back_odds']:.1f}" if pd.notna(r.get("back_odds")) else "-"
            sp = f"{r['sp']:.1f}" if r["sp"] > 0 else "-"
            won_str = "Y" if r["won_actual"] else ""
            pnl_str = f"{r['profit']:+.2f}"
            pnl_bsp_str = f"{r['profit_bsp']:+.2f}"
            print(f"  {horse:<22} {course:<12} {time_str:<6} {back:>5} {sp:>6} {won_str:>3} {pnl_str:>8} {pnl_bsp_str:>8}")

        _print_category_summary(cat_settled, cat.upper())

    print(f"  {'='*70}")
    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    total_profit_bsp = settled["profit_bsp"].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    roi_bsp = total_profit_bsp / total_staked * 100 if total_staked > 0 else 0
    print(f"  {'TOTAL':<10} Bets: {len(settled):<5} |  "
          f"P&L: {total_profit:+.2f}u (ROI {roi:+.1f}%)  |  BSP: {total_profit_bsp:+.2f}u (ROI {roi_bsp:+.1f}%)")


def print_range(pnl: pd.DataFrame, date_from: date | None, date_to: date | None):
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print("No settled bets in this range.")
        return

    label = "All-time"
    if date_from and date_to:
        label = f"{date_from} to {date_to}"
    elif date_from:
        label = f"From {date_from}"
    elif date_to:
        label = f"Up to {date_to}"

    print(f"\n  {label}")

    flat = settled[settled["category"] == "flat"]
    hurdle = settled[settled["category"] == "hurdle"]
    chase = settled[settled["category"] == "chase"]
    jumps_extra = settled[settled["category"] == "jumps"]
    jumps = pd.concat([hurdle, chase, jumps_extra], ignore_index=True)

    _print_category_summary(flat, "FLAT")
    _print_category_summary(jumps, "JUMPS")
    _print_category_summary(hurdle, "HURDLE", indent=2)
    _print_category_summary(chase, "CHASE", indent=2)

    known = {"flat", "hurdle", "chase", "jumps"}
    for cat in sorted(set(settled["category"].dropna().unique()) - known):
        _print_category_summary(settled[settled["category"] == cat], cat.upper())

    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    total_profit_bsp = settled["profit_bsp"].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    roi_bsp = total_profit_bsp / total_staked * 100 if total_staked > 0 else 0
    print(f"  {'TOTAL':<10} Bets: {len(settled):<5} |  "
          f"P&L: {total_profit:+.2f}u (ROI {roi:+.1f}%)  |  BSP: {total_profit_bsp:+.2f}u (ROI {roi_bsp:+.1f}%)")

    pending = pnl[~pnl["settled"]]
    if not pending.empty:
        print(f"  ({len(pending)} bets still pending results)")


def main():
    parser = argparse.ArgumentParser(description="Track P&L from daily value bets")
    parser.add_argument("--date", type=str, default=None, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--from", type=str, default=None, dest="date_from", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", type=str, default=None, dest="date_to", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--split-jumps",
        action="store_true",
        help="Reclassify jumps bets into hurdle/chase using races.race_type",
    )
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

    if args.split_jumps:
        bets = apply_race_type_categories(bets, db_path)

    results = fetch_results(
        bets["runner_id"].tolist(),
        bets["runner_id_norm"].tolist(),
        db_path,
    )
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
            total_profit_bsp=("profit_bsp", "sum"),
        ).reset_index()
        daily["roi_pct"] = (daily["total_profit"] / daily["total_staked"] * 100).round(1)
        daily["cumulative_profit"] = daily["total_profit"].cumsum().round(2)
        daily["cumulative_staked"] = daily["total_staked"].cumsum()
        daily["cumulative_roi_pct"] = (daily["cumulative_profit"] / daily["cumulative_staked"] * 100).round(1)
        daily["cumulative_profit_bsp"] = daily["total_profit_bsp"].cumsum().round(2)
        daily["cumulative_roi_bsp_pct"] = (daily["cumulative_profit_bsp"] / daily["cumulative_staked"] * 100).round(1)
        daily["total_profit"] = daily["total_profit"].round(2)
        daily["total_profit_bsp"] = daily["total_profit_bsp"].round(2)

        PNL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(PNL_OUTPUT, index=False)


if __name__ == "__main__":
    main()
