from __future__ import annotations

from pathlib import Path

import duckdb

from .normalise import normalise_course, normalise_horse


def get_db(path: str | Path) -> duckdb.DuckDBPyConnection:
    db = duckdb.connect(str(path))
    try:
        db.create_function("normalise_course", normalise_course, [str], str)
    except duckdb.CatalogException:
        pass
    try:
        db.create_function("normalise_horse", normalise_horse, [str], str)
    except duckdb.CatalogException:
        pass
    return db
