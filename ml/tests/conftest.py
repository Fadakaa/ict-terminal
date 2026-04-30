"""Shared fixtures for ML tests."""
import pytest
import math
from ml.config import make_test_config


def pytest_configure(config):
    """Register custom markers used by the calendar backfill cross-source check."""
    config.addinivalue_line(
        "markers",
        "integration: network/live-data tests; run manually with -m integration",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.integration`` tests unless explicitly requested.

    The default ``pytest`` run is hermetic; integration tests hit live
    services and require network plus optional packages (``market-calendar-tool``).
    Run them manually with ``-m integration``.
    """
    if config.getoption("-m") and "integration" in config.getoption("-m"):
        return
    skip_integration = pytest.mark.skip(
        reason="integration test — run manually with -m integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def test_config(tmp_path):
    return make_test_config(
        db_path=str(tmp_path / "test_trades.db"),
        model_dir=str(tmp_path / "models"),
    )


@pytest.fixture
def sample_analysis():
    """Full ICT analysis JSON matching the frontend schema."""
    return {
        "bias": "bullish",
        "summary": "Strong bullish structure with OB confluence",
        "orderBlocks": [
            {"type": "bullish", "high": 2650.0, "low": 2645.0, "candleIndex": 80, "strength": "strong", "note": "Key demand zone"},
            {"type": "bearish", "high": 2680.0, "low": 2675.0, "candleIndex": 60, "strength": "moderate", "note": "Supply zone"},
        ],
        "fvgs": [
            {"type": "bullish", "high": 2660.0, "low": 2655.0, "startIndex": 85, "filled": False, "note": "Unfilled gap"},
            {"type": "bearish", "high": 2670.0, "low": 2668.0, "startIndex": 70, "filled": True, "note": "Filled"},
        ],
        "liquidity": [
            {"type": "buyside", "price": 2690.0, "candleIndex": 50, "note": "Swing high"},
            {"type": "sellside", "price": 2630.0, "candleIndex": 55, "note": "Swing low"},
        ],
        "entry": {"price": 2650.0, "direction": "long", "rationale": "Entry at bull OB"},
        "stopLoss": {"price": 2643.0, "rationale": "Below OB low"},
        "takeProfits": [
            {"price": 2680.0, "rationale": "First target", "rr": 4.3},
            {"price": 2690.0, "rationale": "BSL sweep", "rr": 5.7},
        ],
        "killzone": "London Open",
        "confluences": ["Bullish OB + FVG overlap", "Higher TF bullish", "Session timing"],
    }


@pytest.fixture
def sample_candles():
    """100 synthetic XAU/USD candles for testing feature extraction."""
    candles = []
    base = 2600.0
    for i in range(100):
        o = base + i * 0.5
        h = o + 3.0
        l = o - 2.0
        c = o + 1.0 if i % 2 == 0 else o - 0.5
        candles.append({
            "datetime": f"2026-03-{10 + i // 24:02d} {i % 24:02d}:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
        })
    return candles


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_trades.db")


@pytest.fixture
def wfo_candles():
    """700+ synthetic candles with occasional displacements for WFO testing.

    Creates a realistic-looking series with:
    - Gradual uptrend overall
    - Occasional large displacement candles (for OB/FVG detection)
    - Varied price action for market structure detection
    """
    candles = []
    base = 2600.0
    for i in range(750):
        # Trend + noise
        trend = i * 0.1
        noise = math.sin(i * 0.3) * 5 + math.cos(i * 0.17) * 3

        o = base + trend + noise
        h_offset = 3.0 + abs(math.sin(i * 0.1)) * 2
        l_offset = 2.5 + abs(math.cos(i * 0.1)) * 2

        # Displacement candles every ~70 bars
        if i % 70 == 35 and i > 30:
            h_offset = 15.0  # Big candle
            c = o + 12.0  # Strong bullish
        elif i % 70 == 55 and i > 30:
            l_offset = 14.0  # Big bearish
            c = o - 11.0
        elif i % 2 == 0:
            c = o + 1.5
        else:
            c = o - 0.8

        h = max(o, c) + h_offset
        l = min(o, c) - l_offset

        # London/NY AM hours for killzone testing
        hour = i % 24
        candles.append({
            "datetime": f"2026-01-{1 + i // 24:02d} {hour:02d}:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
        })
    return candles
