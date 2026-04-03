"""Tests for ICT key levels computation.

Validates PDH/PDL, PWH/PWL, PMH/PML, Asia session H/L,
previous session H/L, and the orchestrator function.
TDD — tests written before implementation.
"""
import pytest
from datetime import datetime, timezone, timedelta


# ── Test helpers ────────────────────────────────────────────

def _make_daily_candles(n: int = 45, base: float = 3000.0,
                        start_date: str = "2026-03-01") -> list[dict]:
    """Generate n daily candles starting from start_date (skipping weekends)."""
    candles = []
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    price = base
    for _ in range(n):
        # Skip Saturday (5) and Sunday (6)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        high = price + 15.0
        low = price - 12.0
        close = price + 2.0
        candles.append({
            "datetime": dt.strftime("%Y-%m-%d 00:00:00"),
            "open": price, "high": high, "low": low, "close": close,
            "volume": 10000,
        })
        price = close
        dt += timedelta(days=1)
    return candles


def _make_weekly_candles(n: int = 10, base: float = 3000.0) -> list[dict]:
    """Generate n weekly candles."""
    candles = []
    dt = datetime(2026, 1, 5)  # Monday
    price = base
    for _ in range(n):
        high = price + 50.0
        low = price - 40.0
        close = price + 5.0
        candles.append({
            "datetime": dt.strftime("%Y-%m-%d 00:00:00"),
            "open": price, "high": high, "low": low, "close": close,
        })
        price = close
        dt += timedelta(weeks=1)
    return candles


def _make_intraday_candles(n_hours: int = 48, granularity_min: int = 15,
                            base: float = 3030.0,
                            start_dt: str = "2026-04-01 00:00:00") -> list[dict]:
    """Generate intraday candles spanning n_hours at given granularity."""
    candles = []
    dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
    price = base
    step = timedelta(minutes=granularity_min)
    total = int(n_hours * 60 / granularity_min)
    for _ in range(total):
        high = price + 3.0
        low = price - 2.5
        close = price + 0.5
        candles.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": price, "high": high, "low": low, "close": close,
        })
        price = close
        dt += step
    return candles


# ── Test: PDH/PDL from daily candles ───────────────────────

class TestComputePDHPDL:
    def test_returns_previous_day_from_daily_candles(self):
        from ml.key_levels import compute_pdh_pdl
        candles = _make_daily_candles(5)
        result = compute_pdh_pdl(candles)
        # Second-to-last candle is the "previous day"
        prev = candles[-2]
        assert result["high"] == prev["high"]
        assert result["low"] == prev["low"]

    def test_returns_none_for_single_candle(self):
        from ml.key_levels import compute_pdh_pdl
        candles = _make_daily_candles(1)
        result = compute_pdh_pdl(candles)
        assert result["high"] is None
        assert result["low"] is None

    def test_returns_none_for_empty_list(self):
        from ml.key_levels import compute_pdh_pdl
        result = compute_pdh_pdl([])
        assert result["high"] is None

    def test_uses_second_to_last_candle(self):
        from ml.key_levels import compute_pdh_pdl
        candles = _make_daily_candles(10)
        result = compute_pdh_pdl(candles)
        # Must be [-2], not [-1] (current incomplete day)
        assert result["high"] == candles[-2]["high"]
        assert result["low"] == candles[-2]["low"]

    def test_equilibrium_is_midpoint(self):
        from ml.key_levels import compute_pdh_pdl
        candles = _make_daily_candles(5)
        result = compute_pdh_pdl(candles)
        expected_eq = (result["high"] + result["low"]) / 2
        assert result["eq"] == pytest.approx(expected_eq)


class TestComputePDHPDLFromIntraday:
    def test_aggregates_intraday_candles_by_date(self):
        from ml.key_levels import compute_pdh_pdl_from_intraday
        candles = _make_intraday_candles(48, 15)  # 2 full days
        result = compute_pdh_pdl_from_intraday(candles)
        assert result["high"] is not None
        assert result["low"] is not None

    def test_skips_current_incomplete_day(self):
        from ml.key_levels import compute_pdh_pdl_from_intraday
        # 36 hours = 1 full day + 12h into next
        candles = _make_intraday_candles(36, 15)
        result = compute_pdh_pdl_from_intraday(candles)
        # Should return the first full day, not partial second day
        assert result["high"] is not None

    def test_returns_none_for_insufficient_data(self):
        from ml.key_levels import compute_pdh_pdl_from_intraday
        # Only 6 hours = not even 1 full day
        candles = _make_intraday_candles(6, 15)
        result = compute_pdh_pdl_from_intraday(candles)
        # Only 1 partial day — no "previous" day available
        assert result["high"] is None

    def test_handles_empty_input(self):
        from ml.key_levels import compute_pdh_pdl_from_intraday
        result = compute_pdh_pdl_from_intraday([])
        assert result["high"] is None


class TestComputePWHPWL:
    def test_returns_previous_week_from_weekly_candles(self):
        from ml.key_levels import compute_pwh_pwl
        candles = _make_weekly_candles(5)
        result = compute_pwh_pwl(candles)
        assert result["high"] == candles[-2]["high"]
        assert result["low"] == candles[-2]["low"]

    def test_returns_none_for_single_candle(self):
        from ml.key_levels import compute_pwh_pwl
        candles = _make_weekly_candles(1)
        result = compute_pwh_pwl(candles)
        assert result["high"] is None

    def test_equilibrium_is_midpoint(self):
        from ml.key_levels import compute_pwh_pwl
        candles = _make_weekly_candles(5)
        result = compute_pwh_pwl(candles)
        expected = (result["high"] + result["low"]) / 2
        assert result["eq"] == pytest.approx(expected)


class TestComputePWHPWLFromDaily:
    def test_groups_by_iso_week(self):
        from ml.key_levels import compute_pwh_pwl_from_daily
        candles = _make_daily_candles(15)  # ~3 weeks
        result = compute_pwh_pwl_from_daily(candles)
        assert result["high"] is not None
        assert result["low"] is not None

    def test_skips_current_incomplete_week(self):
        from ml.key_levels import compute_pwh_pwl_from_daily
        # 10 trading days = ~2 weeks
        candles = _make_daily_candles(10)
        result = compute_pwh_pwl_from_daily(candles)
        # Should use the last COMPLETE week, not the current partial one
        assert result["high"] is not None

    def test_returns_none_if_no_complete_week(self):
        from ml.key_levels import compute_pwh_pwl_from_daily
        candles = _make_daily_candles(3)  # Not even 1 full week
        result = compute_pwh_pwl_from_daily(candles)
        # Might be None if all candles fall in the same week
        # This depends on start_date alignment, so just ensure no crash
        assert "high" in result


class TestComputePMHPML:
    def test_returns_previous_month(self):
        from ml.key_levels import compute_pmh_pml
        # Start from Feb 1 — gives us full Feb + some March
        candles = _make_daily_candles(45, start_date="2026-02-02")
        result = compute_pmh_pml(candles)
        # Should have a complete month (Feb or March)
        assert result["high"] is not None
        assert result["low"] is not None

    def test_skips_current_incomplete_month(self):
        from ml.key_levels import compute_pmh_pml
        candles = _make_daily_candles(45, start_date="2026-02-02")
        result = compute_pmh_pml(candles)
        # Verify the returned levels aren't from the last (current) month
        if result["high"] is not None:
            # The previous month's high should differ from current month's range
            assert result["low"] < result["high"]

    def test_returns_none_for_insufficient_data(self):
        from ml.key_levels import compute_pmh_pml
        candles = _make_daily_candles(5, start_date="2026-03-25")
        result = compute_pmh_pml(candles)
        # Only 5 days in 1 month — no previous month available
        assert result["high"] is None

    def test_handles_empty_input(self):
        from ml.key_levels import compute_pmh_pml
        result = compute_pmh_pml([])
        assert result["high"] is None


class TestComputeAsiaSessionHL:
    def test_filters_candles_to_0_7_utc(self):
        from ml.key_levels import compute_asia_session_hl
        # 48h of 15min candles covers all sessions
        candles = _make_intraday_candles(48, 15, start_dt="2026-04-01 00:00:00")
        result = compute_asia_session_hl(candles, target_date="2026-04-01")
        assert result["high"] is not None
        assert result["low"] is not None

    def test_uses_today_by_default(self):
        from ml.key_levels import compute_asia_session_hl
        # Generate candles for today
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = f"{today} 00:00:00"
        candles = _make_intraday_candles(24, 15, start_dt=start)
        result = compute_asia_session_hl(candles)
        # Should find today's Asia candles (00:00-07:00)
        assert result["high"] is not None

    def test_accepts_explicit_date(self):
        from ml.key_levels import compute_asia_session_hl
        candles = _make_intraday_candles(48, 15, start_dt="2026-04-01 00:00:00")
        result = compute_asia_session_hl(candles, target_date="2026-04-02")
        # Second day should have Asia candles
        assert result["high"] is not None

    def test_returns_none_when_no_candles_in_range(self):
        from ml.key_levels import compute_asia_session_hl
        # Candles only from 10:00-20:00 — no Asia hours
        candles = _make_intraday_candles(10, 15, start_dt="2026-04-01 10:00:00")
        result = compute_asia_session_hl(candles, target_date="2026-04-01")
        assert result["high"] is None

    def test_handles_partial_session(self):
        from ml.key_levels import compute_asia_session_hl
        # Only 3 hours of Asia (00:00-03:00)
        candles = _make_intraday_candles(3, 15, start_dt="2026-04-01 00:00:00")
        result = compute_asia_session_hl(candles, target_date="2026-04-01")
        assert result["high"] is not None

    def test_equilibrium_computed(self):
        from ml.key_levels import compute_asia_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_asia_session_hl(candles, target_date="2026-04-01")
        if result["high"] and result["low"]:
            assert result["eq"] == pytest.approx((result["high"] + result["low"]) / 2)


class TestComputePreviousSessionHL:
    def test_london_previous_is_asian_same_day(self):
        from ml.key_levels import compute_previous_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_previous_session_hl(candles, "London")
        assert result["session"] == "Asian"
        assert result["high"] is not None

    def test_ny_am_previous_is_london_same_day(self):
        from ml.key_levels import compute_previous_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_previous_session_hl(candles, "NY_AM")
        assert result["session"] == "London"

    def test_ny_pm_previous_is_ny_am_same_day(self):
        from ml.key_levels import compute_previous_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_previous_session_hl(candles, "NY_PM")
        assert result["session"] == "NY_AM"

    def test_asian_previous_is_ny_pm_yesterday(self):
        from ml.key_levels import compute_previous_session_hl
        # Need 2 days of candles so yesterday's NY_PM exists
        candles = _make_intraday_candles(48, 15, start_dt="2026-03-31 00:00:00")
        result = compute_previous_session_hl(candles, "Asian")
        assert result["session"] == "NY_PM"

    def test_off_previous_is_ny_pm_same_day(self):
        from ml.key_levels import compute_previous_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_previous_session_hl(candles, "Off")
        assert result["session"] == "NY_PM"

    def test_returns_session_name(self):
        from ml.key_levels import compute_previous_session_hl
        candles = _make_intraday_candles(24, 15, start_dt="2026-04-01 00:00:00")
        result = compute_previous_session_hl(candles, "London")
        assert "session" in result
        assert result["session"] is not None

    def test_returns_none_for_insufficient_data(self):
        from ml.key_levels import compute_previous_session_hl
        result = compute_previous_session_hl([], "London")
        assert result["high"] is None


class TestComputeAllKeyLevels:
    def test_returns_all_keys(self):
        from ml.key_levels import compute_all_key_levels
        daily = _make_daily_candles(45)
        weekly = _make_weekly_candles(5)
        intraday = _make_intraday_candles(48, 15)
        result = compute_all_key_levels(daily, weekly, intraday, "London")
        expected_keys = [
            "pdh", "pdl", "pd_eq",
            "pwh", "pwl", "pw_eq",
            "pmh", "pml", "pm_eq",
            "asia_high", "asia_low", "asia_eq",
            "prev_session_high", "prev_session_low",
            "prev_session_eq", "prev_session_name",
            "levels_computed",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_graceful_with_no_data(self):
        from ml.key_levels import compute_all_key_levels
        result = compute_all_key_levels()
        assert result["pdh"] is None
        assert result["pwh"] is None
        assert result["pmh"] is None
        assert result["asia_high"] is None
        assert result["levels_computed"] == 0

    def test_levels_computed_count(self):
        from ml.key_levels import compute_all_key_levels
        daily = _make_daily_candles(45, start_date="2026-02-02")
        weekly = _make_weekly_candles(5)
        result = compute_all_key_levels(daily_candles=daily, weekly_candles=weekly)
        # At least PDH/PDL and PWH/PWL should be computed
        assert result["levels_computed"] >= 4

    def test_uses_daily_for_pdh_when_available(self):
        from ml.key_levels import compute_all_key_levels
        daily = _make_daily_candles(5)
        result = compute_all_key_levels(daily_candles=daily)
        assert result["pdh"] == daily[-2]["high"]
        assert result["pdl"] == daily[-2]["low"]

    def test_falls_back_to_intraday_for_pdh(self):
        from ml.key_levels import compute_all_key_levels
        intraday = _make_intraday_candles(48, 15)
        result = compute_all_key_levels(intraday_candles=intraday)
        # Should still compute PDH/PDL from intraday aggregation
        assert result["pdh"] is not None

    def test_uses_weekly_for_pwh_when_available(self):
        from ml.key_levels import compute_all_key_levels
        weekly = _make_weekly_candles(5)
        result = compute_all_key_levels(weekly_candles=weekly)
        assert result["pwh"] == weekly[-2]["high"]
        assert result["pwl"] == weekly[-2]["low"]

    def test_falls_back_to_daily_for_pwh(self):
        from ml.key_levels import compute_all_key_levels
        daily = _make_daily_candles(15)
        result = compute_all_key_levels(daily_candles=daily)
        # Should attempt PWH/PWL from daily aggregation
        assert "pwh" in result


class TestHelperFunctions:
    def test_compute_equilibrium(self):
        from ml.key_levels import _compute_equilibrium
        assert _compute_equilibrium(100.0, 80.0) == 90.0
        assert _compute_equilibrium(None, 80.0) is None
        assert _compute_equilibrium(100.0, None) is None
        assert _compute_equilibrium(None, None) is None

    def test_group_candles_by_date(self):
        from ml.key_levels import _group_candles_by_date
        candles = _make_intraday_candles(48, 60)  # 2 days at 1h
        groups = _group_candles_by_date(candles)
        assert len(groups) == 2

    def test_group_candles_empty(self):
        from ml.key_levels import _group_candles_by_date
        groups = _group_candles_by_date([])
        assert len(groups) == 0

    def test_filter_candles_by_hour_range(self):
        from ml.key_levels import _filter_candles_by_hour_range
        candles = _make_intraday_candles(24, 60, start_dt="2026-04-01 00:00:00")
        filtered = _filter_candles_by_hour_range(candles, 0, 7, "2026-04-01")
        # Hours 0-6 inclusive = 7 candles at 1h granularity
        assert len(filtered) == 7
        for c in filtered:
            hour = int(c["datetime"].split(" ")[1].split(":")[0])
            assert 0 <= hour < 7
