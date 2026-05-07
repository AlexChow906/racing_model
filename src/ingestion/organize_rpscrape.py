from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_RPS_DIR = ROOT / "data" / "raw" / "rpscrape"
DEST_ROOT = RAW_RPS_DIR / "by_year_month"
DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")
YEAR_FILE_PATTERN = re.compile(r"^(\d{4})\.csv$", re.IGNORECASE)


@dataclass
class OrganizeStats:
    scanned: int = 0
    copied: int = 0
    moved: int = 0
    skipped: int = 0
    failed: int = 0


def _extract_year_month_day(file_name: str) -> tuple[str, str, str] | None:
    match = DATE_PATTERN.search(file_name)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _country_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "region" in parts:
        idx = parts.index("region")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _destination_dir(dest_root: Path, country: str, year: str, month: str, include_region: bool) -> Path:
    if include_region:
        return dest_root / country / year / month
    return dest_root / year / month


def organize_rpscrape_files(
    source_root: Path,
    dest_root: Path,
    move_files: bool = False,
    include_region: bool = False,
) -> OrganizeStats:
    stats = OrganizeStats()

    candidates = sorted(
        [p for p in source_root.rglob("*.csv") if p.is_file() and "by_year_month" not in p.parts]
    )

    for src in candidates:
        stats.scanned += 1

        ymd = _extract_year_month_day(src.name)
        country = _country_from_path(src)

        if ymd is not None:
            year, month, day = ymd
            dest_dir = _destination_dir(dest_root=dest_root, country=country, year=year, month=month, include_region=include_region)
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_name = f"{country}_{year}_{month}_{day}{src.suffix}"
            dest = dest_dir / dest_name

            if dest.exists():
                stats.skipped += 1
                continue

            try:
                if move_files:
                    shutil.move(str(src), str(dest))
                    stats.moved += 1
                else:
                    shutil.copy2(src, dest)
                    stats.copied += 1
            except Exception:
                stats.failed += 1
            continue

        year_match = YEAR_FILE_PATTERN.match(src.name)
        if year_match is None:
            stats.skipped += 1
            continue

        try:
            with src.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "date" not in reader.fieldnames:
                    stats.skipped += 1
                    continue

                month_rows: dict[str, list[dict[str, str]]] = {}
                for row in reader:
                    date_value = (row.get("date") or "").strip()
                    if len(date_value) < 7:
                        continue
                    year = date_value[:4]
                    month = date_value[5:7]
                    if not (year.isdigit() and month.isdigit() and 1 <= int(month) <= 12):
                        continue
                    key = f"{year}_{month}"
                    month_rows.setdefault(key, []).append(row)

                for key, rows in month_rows.items():
                    y, m = key.split("_")
                    dest_dir = _destination_dir(dest_root=dest_root, country=country, year=y, month=m, include_region=include_region)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    source_type = src.parent.name.lower()
                    dest = dest_dir / f"{country}_{source_type}_{y}_{m}.csv"

                    if dest.exists():
                        stats.skipped += 1
                        continue

                    with dest.open("w", newline="", encoding="utf-8") as out_fh:
                        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    stats.copied += 1
        except Exception:
            stats.failed += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize rpscrape CSV files into YYYY/MM partitions")
    parser.add_argument("--source-root", type=str, default=str(RAW_RPS_DIR))
    parser.add_argument("--dest-root", type=str, default=str(DEST_ROOT))
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument(
        "--include-region",
        action="store_true",
        help="Include region folder in destination path: <dest>/<region>/<year>/<month>",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()

    stats = organize_rpscrape_files(
        source_root=source_root,
        dest_root=dest_root,
        move_files=args.move,
        include_region=args.include_region,
    )
    print(
        {
            "source_root": str(source_root),
            "dest_root": str(dest_root),
            "include_region": args.include_region,
            "mode": "move" if args.move else "copy",
            "scanned": stats.scanned,
            "copied": stats.copied,
            "moved": stats.moved,
            "skipped": stats.skipped,
            "failed": stats.failed,
        }
    )


if __name__ == "__main__":
    main()
