from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "racing.duckdb"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
LOG_PATH = ROOT / "logs" / "paper_trades.csv"


def ensure_layout() -> None:
    required_dirs = [
        ROOT / "betfair_odds_raw" / "2022" / "01",
        ROOT / "models",
        ROOT / "logs",
        ROOT / "sql",
        ROOT / "configs",
        ROOT / "src" / "ingestion",
        ROOT / "src" / "pipelines",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)



def init_db() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(sql)
    finally:
        con.close()



def init_logs() -> None:
    if LOG_PATH.exists():
        return
    LOG_PATH.write_text(
        "trade_id,race_id,runner_id,placed_timestamp_utc,bookmaker_source,price_taken,model_prob,market_prob_fair,edge,stake,result_win_flag,pnl\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    ensure_layout()
    init_db()
    init_logs()
    print(f"Initialized project at {ROOT}")
    print(f"DuckDB: {DB_PATH}")
    print(f"Log CSV: {LOG_PATH}")
