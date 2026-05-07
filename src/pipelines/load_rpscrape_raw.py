from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ingestion.db_connect import get_db


DB_PATH = ROOT / "racing.duckdb"


def load_rpscrape_raw(db_path: Path, input_glob: str) -> dict[str, int]:
    files = sorted(ROOT.glob(input_glob))
    if not files:
        return {"files_scanned": 0, "rows_loaded": 0}

    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frame = pd.read_csv(file_path, low_memory=False)
        except Exception:
            continue

        cols = {c.lower(): c for c in frame.columns}
        required = ["horse", "date", "course", "trainer", "jockey"]
        if not all(col in cols for col in required):
            continue

        view = pd.DataFrame(
            {
                "horse": frame[cols["horse"]].astype(str),
                "date": frame[cols["date"]].astype(str),
                "course": frame[cols["course"]].astype(str),
                "trainer": frame[cols["trainer"]].astype(str),
                "jockey": frame[cols["jockey"]].astype(str),
                "source_file": str(file_path.relative_to(ROOT)),
                "loaded_at_utc": datetime.now(timezone.utc),
            }
        )
        frames.append(view)

    if not frames:
        return {"files_scanned": len(files), "rows_loaded": 0}

    payload = pd.concat(frames, ignore_index=True)

    con = get_db(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS rpscrape_raw")
        con.execute(
            """
            CREATE TABLE rpscrape_raw (
                horse VARCHAR,
                date VARCHAR,
                course VARCHAR,
                trainer VARCHAR,
                jockey VARCHAR,
                source_file VARCHAR,
                loaded_at_utc TIMESTAMPTZ
            )
            """
        )
        con.register("tmp_rpscrape_raw", payload)
        con.execute(
            """
            INSERT INTO rpscrape_raw (horse, date, course, trainer, jockey, source_file, loaded_at_utc)
            SELECT horse, date, course, trainer, jockey, source_file, loaded_at_utc
            FROM tmp_rpscrape_raw
            """
        )
        rows_loaded = int(con.execute("SELECT COUNT(*) FROM rpscrape_raw").fetchone()[0])
    finally:
        con.close()

    return {"files_scanned": len(files), "rows_loaded": rows_loaded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load raw rpscrape CSV rows into persistent rpscrape_raw table")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--input-glob", type=str, default="data/raw/rpscrape_repo/data/region/**/*.csv")
    args = parser.parse_args()

    stats = load_rpscrape_raw(db_path=Path(args.db_path), input_glob=args.input_glob)
    print(stats)


if __name__ == "__main__":
    main()
