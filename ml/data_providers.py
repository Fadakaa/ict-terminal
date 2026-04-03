"""
Unified candle data provider abstracting Twelve Data and OANDA APIs.

Both providers return identical format:
    [{"datetime": "2026-03-24 16:00:00", "open": float, "high": float,
      "low": float, "close": float, "volume": float}, ...]
"""

import httpx
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class CandleProvider(ABC):
    """Abstract base class for candle data providers."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        count: int | None = None,
    ) -> list[dict]:
        """Fetch OHLCV candle data for a symbol.

        Args:
            symbol: Trading pair, e.g. "XAU/USD", "DXY".
            interval: One of "5min", "15min", "1h", "4h", "1day".
            start_date: ISO date string "YYYY-MM-DD".
            end_date: ISO date string "YYYY-MM-DD".
            count: Optional max number of candles to return.

        Returns:
            List of dicts with keys: datetime, open, high, low, close, volume.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...


class TwelveDataProvider(CandleProvider):
    """Candle provider using the Twelve Data REST API."""

    BASE_URL = "https://api.twelvedata.com/time_series"

    VALID_INTERVALS = {"5min", "15min", "1h", "4h", "1day"}

    def __init__(self, api_key: str):
        self.api_key = api_key

    def name(self) -> str:
        return "twelvedata"

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        count: int | None = None,
    ) -> list[dict]:
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of {self.VALID_INTERVALS}"
            )

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC",
        }
        if count is not None:
            params["outputsize"] = count

        response = httpx.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "values" not in data:
            logger.warning("Twelve Data returned no values: %s", data)
            return []

        candles = []
        for v in data["values"]:
            candles.append(
                {
                    "datetime": v["datetime"],
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": float(v["close"]),
                    "volume": float(v.get("volume", 0)),
                }
            )
        return candles


class OandaProvider(CandleProvider):
    """Candle provider using OANDA v20 REST API. Free demo account. No daily limits."""

    SYMBOL_MAP = {"XAU/USD": "XAU_USD", "DXY": "EUR_USD", "US10Y": "USB10Y_USD"}
    GRAN_MAP = {"5min": "M5", "15min": "M15", "30min": "M30", "1h": "H1", "4h": "H4", "1day": "D", "1week": "W"}

    def __init__(self, account_id: str, access_token: str):
        self.account_id = account_id
        self.access_token = access_token
        self.base_url = "https://api-fxtrade.oanda.com"

    def name(self) -> str:
        return "oanda"

    def fetch_candles(self, symbol, interval, start_date, end_date, count=None):
        instrument = self.SYMBOL_MAP.get(symbol, symbol.replace("/", "_"))
        granularity = self.GRAN_MAP.get(interval)
        if not granularity:
            raise ValueError(f"Unsupported interval: {interval}")

        # Accept both datetime objects and strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date[:19], "%Y-%m-%d" if len(start_date) <= 10 else "%Y-%m-%dT%H:%M:%S")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date[:19], "%Y-%m-%d" if len(end_date) <= 10 else "%Y-%m-%dT%H:%M:%S")

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        all_candles = []
        current_from = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        max_per_request = 4999  # OANDA max is 5000; use 4999 to stay safe

        while True:
            # Use count instead of 'to' to avoid "Maximum value for count exceeded" error.
            # OANDA rejects from+to when the range exceeds 5000 candles.
            params = {
                "granularity": granularity,
                "from": current_from,
                "count": max_per_request,
                "price": "M",  # midpoint
            }

            resp = httpx.get(
                f"{self.base_url}/v3/instruments/{instrument}/candles",
                headers=headers,
                params=params,
                timeout=60,
            )

            if resp.status_code != 200:
                logger.error("OANDA API error %d: %s", resp.status_code, resp.text[:200])
                break

            data = resp.json()
            raw_candles = data.get("candles", [])

            if not raw_candles:
                break

            for c in raw_candles:
                if not c.get("complete", False):
                    continue
                mid = c.get("mid", {})
                ts = c["time"][:19].replace("T", " ")  # "2025-09-15T07:00:00" -> "2025-09-15 07:00:00"
                # Stop if we've gone past end_date
                if ts.replace(" ", "T") + "Z" > end_str:
                    break
                all_candles.append({
                    "datetime": ts,
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })

            if len(raw_candles) < max_per_request:
                break  # Got all data

            # Paginate: use last candle time as new 'from'
            last_time = raw_candles[-1]["time"]
            current_from = last_time
            time.sleep(1)  # polite delay

        # Deduplicate by timestamp
        seen = set()
        deduped = []
        for c in all_candles:
            if c["datetime"] not in seen:
                seen.add(c["datetime"])
                deduped.append(c)

        return deduped


def get_provider(source: str, config: dict) -> CandleProvider:
    """Factory function to create a CandleProvider by name.

    Args:
        source: "oanda" or "twelvedata".
        config: Dict containing at minimum the relevant API key,
                e.g. {"twelvedata_api_key": "...", "oanda_account_id": "...", "oanda_access_token": "..."}.

    Returns:
        A CandleProvider instance.
    """
    source = source.lower()
    if source == "oanda":
        return OandaProvider(
            account_id=config.get("oanda_account_id", ""),
            access_token=config.get("oanda_access_token", ""),
        )
    elif source in ("twelvedata", "td"):
        return TwelveDataProvider(api_key=config.get("td_api_key") or config.get("twelvedata_api_key", ""))
    else:
        raise ValueError(f"Unknown provider '{source}'. Use 'oanda' or 'twelvedata'.")
