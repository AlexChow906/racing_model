from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "data" / "raw" / "betfair_historical"


def infer_year_month(file_name: str) -> tuple[int, int] | None:
    patterns = [
        r"(20\d{2})[-_]?([01]\d)",
        r"([01]\d)[-_]?(20\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, file_name)
        if not match:
            continue
        a = int(match.group(1))
        b = int(match.group(2))

        if 2000 <= a <= 2100 and 1 <= b <= 12:
            return a, b
        if 1 <= a <= 12 and 2000 <= b <= 2100:
            return b, a

    return None


def stage_zips(source_dir: Path, dry_run: bool = False) -> dict[str, int]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    unknown = 0

    for zip_path in sorted(source_dir.glob("*.zip")):
        name_lower = zip_path.name.lower()
        if "betfair" not in name_lower and "horse" not in name_lower and "racing" not in name_lower:
            skipped += 1
            continue

        inferred = infer_year_month(zip_path.stem)
        if inferred is None:
            unknown += 1
            print(f"unknown_date_pattern: {zip_path.name}")
            continue

        year, month = inferred
        target_dir = TARGET_ROOT / f"{year:04d}" / f"{month:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / zip_path.name

        if target_file.exists():
            skipped += 1
            print(f"already_exists: {target_file}")
            continue

        print(f"stage: {zip_path} -> {target_file}")
        if not dry_run:
            shutil.copy2(zip_path, target_file)
        moved += 1

    return {
        "moved": moved,
        "skipped": skipped,
        "unknown": unknown,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage Betfair historical ZIP files into data/raw/betfair_historical/YYYY/MM")
    parser.add_argument("--source-dir", type=str, default="~/Downloads", help="Directory containing downloaded ZIP files")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only; do not copy files")
    args = parser.parse_args()

    result = stage_zips(Path(args.source_dir), dry_run=args.dry_run)
    print(result)
