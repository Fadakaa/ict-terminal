"""Tests for ml.data_providers — all API calls mocked."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ml.data_providers import (
    CandleProvider,
    OandaProvider,
    TwelveDataProvider,
    get_provider,
)


# ---------------------------------------------------------------------------
# Helpers to build mock responses
# ---------------------------------------------------------------------------

def _twelvedata_response(candles: list[dict]) -> dict:
    """Build a Twelve Data JSON response with a 'values' key."""
    return {
        "values": [
            {
                "datetime": c["datetime"],
                "open": str(c["open"]),
                "high": str(c["high"]),
                "low": str(c["low"]),
                "close": str(c["close"]),
                "volume": str(c["volume"]),
            }
            for c in candles
        ]
    }


def _oanda_response(candles: list[dict], all_complete: bool = True) -> dict:
    """Build an OANDA v20 candle response from a list of standardized candles."""
    return {
        "candles": [
            {
                "time": c["datetime"].replace(" ", "T") + ".000000000Z",
                "mid": {
                    "o": str(c["open"]),
                    "h": str(c["high"]),
                    "l": str(c["low"]),
                    "c": str(c["close"]),
                },
                "volume": int(c["volume"]),
                "complete": c.get("complete", all_complete),
            }
            for c in candles
        ]
    }


def _make_candle(dt_str: str, o: float, h: float, l: float, c: float, v: float, complete: bool = True) -> dict:
    return {
        "datetime": dt_str,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "complete": complete,
    }


# ---------------------------------------------------------------------------
# 1. TwelveDataProvider returns standardized format
# ---------------------------------------------------------------------------

class TestTwelveDataProvider:
    @patch("ml.data_providers.httpx.get")
    def test_returns_standardized_format(self, mock_get):
        candles = [
            _make_candle("2026-03-24 09:00:00", 2010.0, 2015.0, 2008.0, 2012.0, 1500.0),
            _make_candle("2026-03-24 09:05:00", 2012.0, 2018.0, 2011.0, 2016.0, 1200.0),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _twelvedata_response(candles)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = TwelveDataProvider(api_key="test_key")
        result = provider.fetch_candles("XAU/USD", "5min", "2026-03-24", "2026-03-25")

        assert len(result) == 2
        for r in result:
            assert set(r.keys()) == {"datetime", "open", "high", "low", "close", "volume"}
            assert isinstance(r["open"], float)
            assert isinstance(r["high"], float)
            assert isinstance(r["low"], float)
            assert isinstance(r["close"], float)
            assert isinstance(r["volume"], float)

        assert result[0]["datetime"] == "2026-03-24 09:00:00"
        assert result[0]["open"] == 2010.0

    @patch("ml.data_providers.httpx.get")
    def test_name(self, mock_get):
        provider = TwelveDataProvider(api_key="k")
        assert provider.name() == "twelvedata"

    @patch("ml.data_providers.httpx.get")
    def test_invalid_interval_raises(self, mock_get):
        provider = TwelveDataProvider(api_key="k")
        with pytest.raises(ValueError, match="Invalid interval"):
            provider.fetch_candles("XAU/USD", "3min", "2026-03-24", "2026-03-25")

    @patch("ml.data_providers.httpx.get")
    def test_resolution_valid_intervals(self, mock_get):
        """All five expected intervals are accepted without error."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"values": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        provider = TwelveDataProvider(api_key="k")
        for interval in ("5min", "15min", "1h", "4h", "1day"):
            result = provider.fetch_candles("XAU/USD", interval, "2026-03-24", "2026-03-25")
            assert result == []


# ---------------------------------------------------------------------------
# 2. OandaProvider returns standardized format
# ---------------------------------------------------------------------------

class TestOandaProvider:
    @patch("ml.data_providers.httpx.get")
    def test_returns_standardized_format(self, mock_get):
        candles = [
            _make_candle("2025-09-15 07:00:00", 2580.500, 2585.200, 2578.100, 2583.400, 1234),
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _oanda_response(candles)
        mock_get.return_value = mock_resp

        provider = OandaProvider(account_id="101-001-123", access_token="test_token")
        result = provider.fetch_candles(
            "XAU/USD", "1h",
            datetime(2025, 9, 15), datetime(2025, 9, 16),
        )

        assert len(result) == 1
        r = result[0]
        assert set(r.keys()) == {"datetime", "open", "high", "low", "close", "volume"}
        assert isinstance(r["open"], float)
        assert r["open"] == 2580.5
        assert r["high"] == 2585.2
        assert r["low"] == 2578.1
        assert r["close"] == 2583.4
        assert r["volume"] == 1234
        assert r["datetime"] == "2025-09-15 07:00:00"

    def test_name(self):
        provider = OandaProvider(account_id="x", access_token="y")
        assert provider.name() == "oanda"


# ---------------------------------------------------------------------------
# 3. Symbol mapping
# ---------------------------------------------------------------------------

class TestOandaSymbolMapping:
    def test_xau_usd_mapped(self):
        provider = OandaProvider(account_id="x", access_token="y")
        assert provider.SYMBOL_MAP["XAU/USD"] == "XAU_USD"

    def test_dxy_mapped(self):
        provider = OandaProvider(account_id="x", access_token="y")
        assert provider.SYMBOL_MAP["DXY"] == "EUR_USD"  # Inverted proxy

    def test_us10y_mapped(self):
        provider = OandaProvider(account_id="x", access_token="y")
        assert provider.SYMBOL_MAP["US10Y"] == "USB10Y_USD"

    @patch("ml.data_providers.httpx.get")
    def test_unknown_symbol_uses_slash_replacement(self, mock_get):
        """Unknown symbols get / replaced with _."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candles": []}
        mock_get.return_value = mock_resp

        provider = OandaProvider(account_id="x", access_token="y")
        provider.fetch_candles("EUR/USD", "1h", datetime(2025, 9, 15), datetime(2025, 9, 16))

        # Verify the URL used EUR_USD
        call_url = mock_get.call_args[0][0]
        assert "EUR_USD" in call_url


# ---------------------------------------------------------------------------
# 4. Granularity mapping
# ---------------------------------------------------------------------------

class TestOandaGranularityMapping:
    def test_granularity_mappings(self):
        provider = OandaProvider(account_id="x", access_token="y")
        assert provider.GRAN_MAP["1h"] == "H1"
        assert provider.GRAN_MAP["4h"] == "H4"
        assert provider.GRAN_MAP["1day"] == "D"
        assert provider.GRAN_MAP["5min"] == "M5"
        assert provider.GRAN_MAP["15min"] == "M15"
        assert provider.GRAN_MAP["30min"] == "M30"

    @patch("ml.data_providers.httpx.get")
    def test_unsupported_interval_raises(self, mock_get):
        provider = OandaProvider(account_id="x", access_token="y")
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.fetch_candles("XAU/USD", "3min", datetime(2025, 9, 15), datetime(2025, 9, 16))


# ---------------------------------------------------------------------------
# 5. Pagination — 5000 candles triggers a second request
# ---------------------------------------------------------------------------

class TestOandaPagination:
    @patch("ml.data_providers.time.sleep")
    @patch("ml.data_providers.httpx.get")
    def test_pagination_on_5000_candles(self, mock_get, mock_sleep):
        """When first response has 5000 candles, a second request is made."""
        # Build 5000 candles for page 1
        page1_candles = []
        for i in range(5000):
            page1_candles.append({
                "time": f"2025-09-15T07:{i:04d}:00.000000000Z",
                "mid": {"o": "2580.0", "h": "2585.0", "l": "2578.0", "c": "2583.0"},
                "volume": 100,
                "complete": True,
            })

        # Page 2: fewer than 5000
        page2_candles = [
            {
                "time": "2025-09-16T08:00:00.000000000Z",
                "mid": {"o": "2590.0", "h": "2595.0", "l": "2588.0", "c": "2593.0"},
                "volume": 200,
                "complete": True,
            }
        ]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"candles": page1_candles}

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"candles": page2_candles}

        mock_get.side_effect = [resp1, resp2]

        provider = OandaProvider(account_id="x", access_token="y")
        result = provider.fetch_candles("XAU/USD", "1h", datetime(2025, 9, 15), datetime(2025, 9, 17))

        # Two API calls made
        assert mock_get.call_count == 2
        # Polite delay called between pages
        mock_sleep.assert_called_once_with(1)

    @patch("ml.data_providers.httpx.get")
    def test_no_pagination_under_5000(self, mock_get):
        """When response has fewer than 5000 candles, no second request."""
        candles = [
            _make_candle("2025-09-15 07:00:00", 2580.0, 2585.0, 2578.0, 2583.0, 100),
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _oanda_response(candles)
        mock_get.return_value = mock_resp

        provider = OandaProvider(account_id="x", access_token="y")
        provider.fetch_candles("XAU/USD", "1h", datetime(2025, 9, 15), datetime(2025, 9, 16))

        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# 6. Only complete candles included
# ---------------------------------------------------------------------------

class TestOandaCompleteFilter:
    @patch("ml.data_providers.httpx.get")
    def test_incomplete_candles_filtered_out(self, mock_get):
        """Candles with complete=false are excluded."""
        raw = {
            "candles": [
                {
                    "time": "2025-09-15T07:00:00.000000000Z",
                    "mid": {"o": "2580.0", "h": "2585.0", "l": "2578.0", "c": "2583.0"},
                    "volume": 100,
                    "complete": True,
                },
                {
                    "time": "2025-09-15T08:00:00.000000000Z",
                    "mid": {"o": "2590.0", "h": "2595.0", "l": "2588.0", "c": "2593.0"},
                    "volume": 200,
                    "complete": False,
                },
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = raw
        mock_get.return_value = mock_resp

        provider = OandaProvider(account_id="x", access_token="y")
        result = provider.fetch_candles("XAU/USD", "1h", datetime(2025, 9, 15), datetime(2025, 9, 16))

        assert len(result) == 1
        assert result[0]["datetime"] == "2025-09-15 07:00:00"


# ---------------------------------------------------------------------------
# 7. Deduplication
# ---------------------------------------------------------------------------

class TestOandaDeduplication:
    @patch("ml.data_providers.time.sleep")
    @patch("ml.data_providers.httpx.get")
    def test_duplicate_timestamps_deduplicated(self, mock_get, mock_sleep):
        """If pagination returns overlapping candles, they are deduplicated."""
        candle_data = {
            "time": "2025-09-15T07:00:00.000000000Z",
            "mid": {"o": "2580.0", "h": "2585.0", "l": "2578.0", "c": "2583.0"},
            "volume": 100,
            "complete": True,
        }

        # First page: 5000 identical candles (to trigger pagination)
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"candles": [candle_data] * 5000}

        # Second page: same candle again
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"candles": [candle_data]}

        mock_get.side_effect = [resp1, resp2]

        provider = OandaProvider(account_id="x", access_token="y")
        result = provider.fetch_candles("XAU/USD", "1h", datetime(2025, 9, 15), datetime(2025, 9, 16))

        # All duplicates collapsed to 1
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 8. get_provider() factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_returns_oanda(self):
        config = {"oanda_account_id": "101-001-123", "oanda_access_token": "tok123"}
        provider = get_provider("oanda", config)
        assert isinstance(provider, OandaProvider)
        assert provider.account_id == "101-001-123"
        assert provider.access_token == "tok123"

    def test_returns_twelvedata(self):
        config = {"twelvedata_api_key": "td_key"}
        provider = get_provider("twelvedata", config)
        assert isinstance(provider, TwelveDataProvider)
        assert provider.api_key == "td_key"

    def test_td_alias(self):
        config = {"td_api_key": "td_key2"}
        provider = get_provider("td", config)
        assert isinstance(provider, TwelveDataProvider)
        assert provider.api_key == "td_key2"

    def test_case_insensitive(self):
        config = {"twelvedata_api_key": "k"}
        provider = get_provider("TwelveData", config)
        assert isinstance(provider, TwelveDataProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("binance", {})

    def test_returns_candle_provider_subclass(self):
        config = {"twelvedata_api_key": "k", "oanda_account_id": "x", "oanda_access_token": "y"}
        for source in ("twelvedata", "oanda"):
            provider = get_provider(source, config)
            assert isinstance(provider, CandleProvider)


# ---------------------------------------------------------------------------
# 9. TwelveData valid intervals constant
# ---------------------------------------------------------------------------

class TestTwelveDataIntervals:
    def test_twelvedata_valid_intervals(self):
        assert TwelveDataProvider.VALID_INTERVALS == {"5min", "15min", "1h", "4h", "1day"}
