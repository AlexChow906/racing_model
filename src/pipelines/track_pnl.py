"""
Track P&L from daily value bets and top picks against settled results.

Reads the bet log (logs/daily_bets.csv) and top picks log (logs/daily_top_picks.csv),
joins with actual results from the DB, and prints both side by side.

Value bets settle at the backed price (BSP as benchmark). Top picks settle at BSP
(1-unit level stakes on the model's #1-rated horse per race).

Usage:
    python -m src.pipelines.track_pnl                          # all-time summary
    python -m src.pipelines.track_pnl --from 2026-05-01        # everything from May 1
    python -m src.pipelines.track_pnl --from 2026-05-01 --to 2026-05-07  # one week
    python -m src.pipelines.track_pnl --date 2026-05-15        # single day (bet details)
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db

BETS_LOG = ROOT / "logs" / "daily_bets.csv"
TOP_PICKS_LOG = ROOT / "logs" / "daily_top_picks.csv"
PNL_OUTPUT = ROOT / "logs" / "pnl_tracker.csv"
TOP_PICKS_PNL_OUTPUT = ROOT / "logs" / "top_picks_pnl_tracker.csv"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _normalize_bfsp_id(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return value
    text = str(value)
    return text.replace("bfsp_1.", "bfsp_")


def _load_dated_csv(
    log_path: Path, date_from: date | None = None, date_to: date | None = None,
) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(log_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    df["race_id_norm"] = df["race_id"].apply(_normalize_bfsp_id)
    df["runner_id_norm"] = df["runner_id"].apply(_normalize_bfsp_id)
    return df


def load_bets(date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    return _load_dated_csv(BETS_LOG, date_from, date_to)


def load_top_picks(date_from: date | None = None, date_to: date | None = None) -> pd.DataFrame:
    df = _load_dated_csv(TOP_PICKS_LOG, date_from, date_to)
    if not df.empty:
        df["stake"] = 1.0
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
    return (odds - 1) * stake if won else -stake


def _merge_with_results(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(results, on="runner_id_norm", how="left", suffixes=("", "_actual"))
    merged["settled"] = merged["sp_decimal"].notna()
    merged["won_actual"] = merged["won"].fillna(False)
    merged["sp"] = merged["sp_decimal"].fillna(0)
    return merged


def compute_pnl(bets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    merged = _merge_with_results(bets, results)

    def _profit(r: pd.Series, odds_col: str) -> float:
        if not r["settled"]:
            return 0.0
        odds = r[odds_col]
        if pd.isna(odds) or odds <= 0:
            odds = r["sp"]
        return _settle_profit(odds, r["won_actual"], r["stake"])

    merged["profit"] = merged.apply(lambda r: _profit(r, "back_odds"), axis=1)
    merged["profit_bsp"] = merged.apply(lambda r: _profit(r, "sp"), axis=1)

    return merged


def compute_top_picks_pnl(picks: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()

    merged = _merge_with_results(picks, results)

    merged["profit"] = merged.apply(
        lambda r: _settle_profit(r["sp"], r["won_actual"], r["stake"]) if r["settled"] else 0.0,
        axis=1,
    )

    return merged


# ── Printing ───────────────────────────────────────────────────────────


def _print_category_summary(settled: pd.DataFrame, label: str, indent: int = 0) -> None:
    staked = settled["stake"].sum()
    profit = settled["profit"].sum()
    profit_bsp = settled["profit_bsp"].sum()
    roi = profit / staked * 100 if staked > 0 else 0
    roi_bsp = profit_bsp / staked * 100 if staked > 0 else 0
    prefix = " " * indent
    print(f"  {prefix}{label:<10} Bets: {len(settled):<5} |  "
          f"P&L: {profit:+.2f}u (ROI {roi:+.1f}%)  |  BSP: {profit_bsp:+.2f}u (ROI {roi_bsp:+.1f}%)")


def _print_tp_category_summary(settled: pd.DataFrame, label: str, indent: int = 0) -> None:
    staked = settled["stake"].sum()
    profit = settled["profit"].sum()
    roi = profit / staked * 100 if staked > 0 else 0
    winners = int(settled["won_actual"].sum())
    sr = winners / len(settled) * 100 if len(settled) > 0 else 0
    prefix = " " * indent
    print(f"  {prefix}{label:<10} Picks: {len(settled):<5} W: {winners:<4} SR: {sr:.0f}%  |  "
          f"P&L: {profit:+.2f}u (ROI {roi:+.1f}%)")


def _range_label(date_from: date | None, date_to: date | None) -> str:
    if date_from and date_to:
        return f"{date_from} to {date_to}"
    if date_from:
        return f"From {date_from}"
    if date_to:
        return f"Up to {date_to}"
    return "All-time"


def _print_category_breakdown(
    settled: pd.DataFrame, summary_fn: Callable[[pd.DataFrame, str, int], None],
) -> None:
    flat = settled[settled["category"] == "flat"]
    hurdle = settled[settled["category"] == "hurdle"]
    chase = settled[settled["category"] == "chase"]
    jumps_extra = settled[settled["category"] == "jumps"]
    jumps = pd.concat([hurdle, chase, jumps_extra], ignore_index=True)

    summary_fn(flat, "FLAT")
    summary_fn(jumps, "JUMPS")
    summary_fn(hurdle, "HURDLE", indent=2)
    summary_fn(chase, "CHASE", indent=2)

    known = {"flat", "hurdle", "chase", "jumps"}
    for cat in sorted(set(settled["category"].dropna().unique()) - known):
        summary_fn(settled[settled["category"] == cat], cat.upper())


def print_single_day(pnl: pd.DataFrame, target_date: date) -> None:
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
    _print_category_summary(settled, "TOTAL")


def print_range(pnl: pd.DataFrame, date_from: date | None, date_to: date | None) -> None:
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print("No settled bets in this range.")
        return

    print(f"\n  {_range_label(date_from, date_to)}")
    _print_category_breakdown(settled, _print_category_summary)
    _print_category_summary(settled, "TOTAL")

    pending = pnl[~pnl["settled"]]
    if not pending.empty:
        print(f"  ({len(pending)} bets still pending results)")


def print_top_picks_single_day(pnl: pd.DataFrame, target_date: date) -> None:
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print(f"No settled top picks for {target_date}.")
        return

    print(f"\n  {target_date}")

    for cat in sorted(settled["category"].dropna().unique()):
        cat_settled = settled[settled["category"] == cat]
        print(f"  {'-'*55}")
        print(f"  {cat.upper()}")
        print(f"  {'Horse':<22} {'Course':<12} {'Time':<6} {'SP':>6} {'W':>3} {'P&L':>8}")
        print(f"  {'-'*55}")

        for _, r in cat_settled.sort_values("time").iterrows():
            horse = str(r.get("horse", "?"))[:21]
            course = str(r.get("course", ""))[:11]
            time_str = str(r.get("time", ""))[:5]
            sp = f"{r['sp']:.1f}" if r["sp"] > 0 else "-"
            won_str = "Y" if r["won_actual"] else ""
            pnl_str = f"{r['profit']:+.2f}"
            print(f"  {horse:<22} {course:<12} {time_str:<6} {sp:>6} {won_str:>3} {pnl_str:>8}")

        _print_tp_category_summary(cat_settled, cat.upper())

    print(f"  {'='*55}")
    _print_tp_category_summary(settled, "TOTAL")


def print_top_picks_range(pnl: pd.DataFrame, date_from: date | None, date_to: date | None) -> None:
    settled = pnl[pnl["settled"]].copy()
    if settled.empty:
        print("No settled top picks in this range.")
        return

    print(f"\n  {_range_label(date_from, date_to)}")
    _print_category_breakdown(settled, _print_tp_category_summary)
    _print_tp_category_summary(settled, "TOTAL")

    pending = pnl[~pnl["settled"]]
    if not pending.empty:
        print(f"  ({len(pending)} picks still pending results)")


# ── Tracker CSV ────────────────────────────────────────────────────────


def _write_daily_tracker(settled: pd.DataFrame, output_path: Path, has_bsp: bool = True) -> None:
    agg_dict: dict[str, tuple[str, str]] = {
        "bets": ("runner_id", "count"),
        "winners": ("won_actual", "sum"),
        "total_staked": ("stake", "sum"),
        "total_profit": ("profit", "sum"),
    }
    if has_bsp:
        agg_dict["total_profit_bsp"] = ("profit_bsp", "sum")

    daily = settled.groupby("date").agg(**agg_dict).reset_index()
    daily["roi_pct"] = (daily["total_profit"] / daily["total_staked"] * 100).round(1)
    daily["cumulative_profit"] = daily["total_profit"].cumsum().round(2)
    daily["cumulative_staked"] = daily["total_staked"].cumsum()
    daily["cumulative_roi_pct"] = (daily["cumulative_profit"] / daily["cumulative_staked"] * 100).round(1)
    daily["total_profit"] = daily["total_profit"].round(2)

    if has_bsp:
        daily["cumulative_profit_bsp"] = daily["total_profit_bsp"].cumsum().round(2)
        daily["cumulative_roi_bsp_pct"] = (daily["cumulative_profit_bsp"] / daily["cumulative_staked"] * 100).round(1)
        daily["total_profit_bsp"] = daily["total_profit_bsp"].round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track P&L from daily value bets and top picks")
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
    picks = load_top_picks(date_from, date_to)

    if bets.empty and picks.empty:
        print("No bets or top picks found. Run daily_predictions first.")
        return

    if args.split_jumps:
        if not bets.empty:
            bets = apply_race_type_categories(bets, db_path)
        if not picks.empty:
            picks = apply_race_type_categories(picks, db_path)

    all_runner_ids: list[str] = []
    all_runner_ids_norm: list[str] = []
    for df in [bets, picks]:
        if not df.empty:
            all_runner_ids.extend(df["runner_id"].tolist())
            all_runner_ids_norm.extend(df["runner_id_norm"].tolist())

    results = fetch_results(all_runner_ids, all_runner_ids_norm, db_path)

    has_any_settled = False

    if not bets.empty:
        pnl = compute_pnl(bets, results)
        if not pnl.empty and pnl["settled"].any():
            has_any_settled = True
            print(f"\n  {'='*70}")
            print(f"  VALUE BETS  (edge-filtered, settled at back odds)")
            print(f"  {'='*70}")
            if single_day:
                print_single_day(pnl, date_from)
            else:
                print_range(pnl, date_from, date_to)
                _write_daily_tracker(pnl[pnl["settled"]].copy(), PNL_OUTPUT, has_bsp=True)

    if not picks.empty:
        tp_pnl = compute_top_picks_pnl(picks, results)
        if not tp_pnl.empty and tp_pnl["settled"].any():
            has_any_settled = True
            print(f"\n  {'='*70}")
            print(f"  TOP PICKS  (model #1 per race, settled at BSP)")
            print(f"  {'='*70}")
            if single_day:
                print_top_picks_single_day(tp_pnl, date_from)
            else:
                print_top_picks_range(tp_pnl, date_from, date_to)
                _write_daily_tracker(tp_pnl[tp_pnl["settled"]].copy(), TOP_PICKS_PNL_OUTPUT, has_bsp=False)

    if not has_any_settled:
        print("No settled bets or picks yet. Run collect_results to fetch actual SPs.")


if __name__ == "__main__":
    main()
