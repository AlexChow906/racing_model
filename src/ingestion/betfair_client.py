from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


IDENTITY_CERT_URL = "https://identitysso-cert.betfair.com/api/certlogin"
BETTING_URL_UK = "https://api.betfair.com/exchange/betting/json-rpc/v1"


@dataclass
class BetfairCredentials:
    app_key: str
    username: str
    password: str
    cert_file: Path
    key_file: Path


class BetfairClient:
    def __init__(self, credentials: BetfairCredentials, timeout: int = 20) -> None:
        self._creds = credentials
        self._timeout = timeout
        self._session = requests.Session()
        self._session_token: str | None = None

    def login(self) -> str:
        data = {
            "username": self._creds.username,
            "password": self._creds.password,
        }
        headers = {"X-Application": self._creds.app_key}
        response = self._session.post(
            IDENTITY_CERT_URL,
            data=data,
            headers=headers,
            cert=(str(self._creds.cert_file), str(self._creds.key_file)),
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("loginStatus") != "SUCCESS":
            raise RuntimeError(f"Betfair login failed: {payload}")

        token = payload.get("sessionToken")
        if not token:
            raise RuntimeError("Betfair login succeeded but no sessionToken returned")

        self._session_token = token
        return token

    def list_win_markets(self, from_hours: int = 0, to_hours: int = 6) -> list[dict[str, Any]]:
        self._ensure_logged_in()

        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=from_hours)
        end = now + timedelta(hours=to_hours)

        params = {
            "filter": {
                "eventTypeIds": ["7"],  # Horse Racing
                "marketCountries": ["GB", "IE"],
                "marketTypeCodes": ["WIN"],
                "marketStartTime": {
                    "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
            "sort": "FIRST_TO_START",
            "maxResults": "200",
        }
        return self._rpc_call("SportsAPING/v1.0/listMarketCatalogue", params)

    def list_market_book(self, market_ids: list[str]) -> list[dict[str, Any]]:
        self._ensure_logged_in()
        if not market_ids:
            return []

        params = {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                "virtualise": True,
                "exBestOffersOverrides": {"bestPricesDepth": 3},
            },
            "orderProjection": "ALL",
            "matchProjection": "ROLLED_UP_BY_PRICE",
        }
        return self._rpc_call("SportsAPING/v1.0/listMarketBook", params)

    def _rpc_call(self, method: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        headers = {
            "X-Application": self._creds.app_key,
            "X-Authentication": self._session_token or "",
            "Content-Type": "application/json",
        }
        payload = [{"jsonrpc": "2.0", "method": method, "params": params, "id": 1}]
        response = self._session.post(
            BETTING_URL_UK,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()

        rpc_responses = response.json()
        if not rpc_responses:
            return []
        message = rpc_responses[0]
        if "error" in message:
            raise RuntimeError(f"Betfair RPC error: {message['error']}")
        return message.get("result", [])

    def _ensure_logged_in(self) -> None:
        if not self._session_token:
            self.login()
