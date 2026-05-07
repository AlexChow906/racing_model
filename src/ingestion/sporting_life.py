from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "sporting_life"
UNMATCHED_LOG = ROOT / "logs" / "unmatched_sporting_life.csv"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    }
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    if response.status_code == 429:
        raise RuntimeError("Rate limited")
    return response.text


def save_raw_html(url: str, html: str) -> Path:
    date_dir = RAW_ROOT / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.rstrip("/").split("/")[-1] or "index"
    out_file = date_dir / f"{file_name}.html"
    out_file.write_text(html, encoding="utf-8")
    return out_file


def parse_race_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title else ""
    return {
        "title": title,
        "runner_rows": [],
        "course_name": None,
        "race_date": None,
        "distance_furlongs": None,
    }


def scrape_results(urls: list[str], max_pages: int) -> None:
    processed = 0
    for url in urls:
        if processed >= max_pages:
            break
        html = fetch_page(url)
        save_raw_html(url, html)
        parse_race_page(html)
        processed += 1
        time.sleep(random.uniform(2, 3))


def log_unmatched_rows(rows: list[dict]) -> None:
    UNMATCHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = UNMATCHED_LOG.exists()
    with UNMATCHED_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["race_date", "course_name", "horse_name", "reason"])
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sporting Life scrape utility")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()
    if not args.url:
        print("No URLs provided. Use --url multiple times.")
        return
    scrape_results(args.url, args.max_pages)
    print(f"scraped_pages={min(len(args.url), args.max_pages)}")


if __name__ == "__main__":
    main()
