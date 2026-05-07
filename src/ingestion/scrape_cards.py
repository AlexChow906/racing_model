from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class FetchResult:
    source: str
    url: str
    ok: bool
    status_code: int
    html: Optional[str]


def fetch_html(url: str, source: str, timeout: int = 20) -> FetchResult:
    """Basic fetch helper for race-card pages.

    Respect site terms, robots.txt, and local regulations before scraping.
    """
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "racing-model-research/1.0"})
    return FetchResult(
        source=source,
        url=url,
        ok=resp.ok,
        status_code=resp.status_code,
        html=resp.text if resp.ok else None,
    )


def parse_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.title.text.strip() if soup.title else ""
