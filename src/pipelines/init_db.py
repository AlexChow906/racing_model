from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "racing.duckdb"
SCHEMA_PATH = ROOT / "sql" / "schema" / "001_create_tables.sql"


def get_db_path() -> Path:
    env_path = os.getenv("DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DB_PATH


def init_db(reset: bool = False) -> None:
    db_path = get_db_path()
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    con = duckdb.connect(str(db_path))
    try:
        if reset:
            drop_order = [
                "results",
                "runners",
                "odds_snapshots",
                "horse_history",
                "trainer_history",
                "jockey_history",
                "races",
            ]
            for table in drop_order:
                con.execute(f"DROP TABLE IF EXISTS {table}")
            print("Dropped existing Phase 1 tables (reset mode enabled)")

        con.execute(sql_text)
        tables = [
            "races",
            "runners",
            "results",
            "odds_snapshots",
            "horse_history",
            "trainer_history",
            "jockey_history",
        ]
        for table in tables:
            print(f"Created/verified table: {table}")

        print("\\nRow counts:")
        for table in tables:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"- {table}: {count}")
    finally:
        con.close()

    print(f"\\nDuckDB initialised at: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize DuckDB schema")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate Phase 1 tables")
    args = parser.parse_args()
    init_db(reset=args.reset)
