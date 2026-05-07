from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "racing.duckdb"


def check_no_leakage(table_name: str, decision_cutoff_col: str = "decision_cutoff_utc", db_path: Path | None = None) -> dict:
    chosen_db = db_path or DB_PATH
    con = duckdb.connect(str(chosen_db))
    try:
        query = f"""
            SELECT
                COUNT(*) AS leaking_count
            FROM {table_name}
            WHERE event_timestamp_utc > {decision_cutoff_col}
        """
        leaking_count = int(con.execute(query).fetchone()[0])

        sample_query = f"""
            SELECT
                CAST(event_timestamp_utc AS VARCHAR) AS event_timestamp_utc,
                CAST({decision_cutoff_col} AS VARCHAR) AS decision_cutoff_utc
            FROM {table_name}
            WHERE event_timestamp_utc > {decision_cutoff_col}
            LIMIT 5
        """
        sample_rows = con.execute(sample_query).fetchall()
    finally:
        con.close()

    result = {
        "table_name": table_name,
        "leaking_count": leaking_count,
        "sample_rows": sample_rows,
    }

    if leaking_count > 0:
        raise RuntimeError(f"Leakage guard failed for {table_name}: {leaking_count} rows with event_timestamp_utc > {decision_cutoff_col}")

    print(f"Leakage guard passed for {table_name}: leaking_count=0")
    return result


if __name__ == "__main__":
    check_no_leakage("horse_history")
