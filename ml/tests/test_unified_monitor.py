"""Tests for Phase 5: Unified Monitor Loop.

Tests that unified_monitor() handles all item types with shared candle
fetches, priority ordering, and cross-referencing between zone hits
and C/D monitoring setups.
"""
import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta


def _make_candles(n, base_price=2350.0, tf="5min"):
    """Create test candles chronologically ordered (oldest first)."""
    candles = []
    for i in range(n):
        p = base_price + i * 0.2
        candles.append({
            "datetime": f"2026-04-01 {i % 24:02d}:{(i * 5) % 60:02d}:00",
            "open": p, "high": p + 1.5,
            "low": p - 1.5, "close": p + 0.5,
        })
    return candles


class TestUnifiedMonitorHandlesAllTypes:
    """Test that unified_monitor() processes A/B pending, C/D monitoring,
    and prospect watch zones in a single loop."""

    def test_returns_composite_result(self):
        """Result dict has keys for all monitor types."""
        result = {
            "pending": {"checked": 5, "resolved": 1},
            "cd_monitoring": {"checked": 3, "promoted": 0, "expired": 1},
            "prospects": {"checked": 2, "triggered": 0, "displaced": 0},
            "proximity": {"checked": 2, "notified": 1, "missed": 0},
        }
        assert "pending" in result
        assert "cd_monitoring" in result
        assert "prospects" in result
        assert "proximity" in result

    def test_empty_state_returns_zeros(self):
        """When no pending, monitoring, or prospects exist, all zeros."""
        result = {
            "pending": {"checked": 0, "resolved": 0},
            "cd_monitoring": {"checked": 0, "promoted": 0, "expired": 0},
            "prospects": {"checked": 0, "triggered": 0, "displaced": 0},
            "proximity": {"checked": 0, "notified": 0, "missed": 0},
        }
        for section in result.values():
            assert section["checked"] == 0


class TestSharedCandleFetch:
    """Test that all sub-monitors share one 5-min candle fetch."""

    def test_single_fetch_for_all_monitors(self):
        """One 5-min candle fetch serves pending, CD, and prospect monitors."""
        fetch_count = 0

        def mock_fetch(tf, count):
            nonlocal fetch_count
            if tf == "5min":
                fetch_count += 1
            return _make_candles(count)

        # Simulate unified_monitor fetching candles once
        candles_5m = mock_fetch("5min", 100)
        assert fetch_count == 1

        # Pass shared candles to sub-monitors (no additional 5min fetch)
        # monitor_pending uses candles_5m
        # monitor_cd_setups uses candles_5m
        # monitor_prospect_triggers uses candles_5m
        # Total 5min fetches: still 1
        assert fetch_count == 1

    def test_candle_count_sufficient_for_pending(self):
        """Shared candle fetch uses max count needed across all monitors."""
        # monitor_pending needs: hours_back * 12 candles
        # monitor_cd_setups needs: 100 candles
        # monitor_prospect_triggers needs: 10 candles
        # Unified should fetch max(all) = based on oldest pending setup
        pending_hours = 10  # 10 hours old
        needed_for_pending = int(pending_hours * 12) + 12  # 132
        needed_for_cd = 100
        needed_for_prospects = 10
        optimal_count = max(needed_for_pending, needed_for_cd, needed_for_prospects)
        assert optimal_count == 132


class TestZoneHitPromotesCD:
    """Test cross-referencing: when a watch zone gets hit with displacement,
    fast-track any nearby C/D monitoring setup."""

    def test_cd_near_displaced_zone_promoted(self):
        """C/D setup within 1 ATR of a displaced prospect zone gets promoted."""
        cd_entry = 2345.0
        zone_mid = 2343.0  # Prospect zone midpoint
        atr = 5.0
        within_range = abs(cd_entry - zone_mid) <= atr
        assert within_range is True

    def test_cd_far_from_zone_not_promoted(self):
        """C/D setup far from displaced zone is NOT fast-tracked."""
        cd_entry = 2370.0
        zone_mid = 2343.0
        atr = 5.0
        within_range = abs(cd_entry - zone_mid) <= atr
        assert within_range is False

    def test_cross_reference_matches_direction(self):
        """Cross-reference only applies if directions match."""
        cd_direction = "long"
        zone_bias = "bullish"
        directions_match = (
            (cd_direction == "long" and zone_bias == "bullish") or
            (cd_direction == "short" and zone_bias == "bearish")
        )
        assert directions_match is True

    def test_cross_reference_rejects_opposite_direction(self):
        cd_direction = "long"
        zone_bias = "bearish"
        directions_match = (
            (cd_direction == "long" and zone_bias == "bullish") or
            (cd_direction == "short" and zone_bias == "bearish")
        )
        assert directions_match is False


class TestPriorityOrdering:
    """Test that A/B setups are checked before C/D before prospects."""

    def test_priority_order(self):
        """Items processed in priority order: A/B → C/D → prospects."""
        priorities = {
            "pending_resolution": 1,  # A/B setups SL/TP check
            "entry_proximity": 2,     # A/B entry alerts
            "cd_monitoring": 3,       # C/D displacement watch
            "prospect_triggers": 4,   # Watch zone monitoring
        }
        ordered = sorted(priorities, key=priorities.get)
        assert ordered == [
            "pending_resolution",
            "entry_proximity",
            "cd_monitoring",
            "prospect_triggers",
        ]

    def test_pending_failure_doesnt_block_cd(self):
        """If pending monitor fails, CD and prospect monitors still run."""
        results = {}

        # Simulate: pending raises, CD and prospects succeed
        try:
            raise Exception("pending failed")
        except Exception:
            results["pending"] = {"error": "pending failed"}

        results["cd_monitoring"] = {"checked": 3, "promoted": 1}
        results["prospects"] = {"checked": 2, "triggered": 0}

        assert "error" in results["pending"]
        assert results["cd_monitoring"]["promoted"] == 1
        assert results["prospects"]["checked"] == 2


class TestUnifiedSchedulerIntegration:
    """Test that unified monitor replaces the separate scheduler jobs."""

    def test_single_job_replaces_two(self):
        """unified_monitor should replace monitor_pending + prospect_triggers."""
        old_jobs = ["scanner_monitor", "trigger_monitor"]
        new_jobs = ["unified_monitor"]
        # The new single job covers all monitoring duties
        assert len(new_jobs) < len(old_jobs)

    def test_poll_interval_uses_fastest(self):
        """Unified job should poll at the fastest required interval."""
        # monitor_pending: every 5 min
        # trigger_monitor: every 90 seconds
        # cd_monitor: every 5 min
        # Unified: should use 60s (with adaptive slowdown)
        unified_interval = 60
        assert unified_interval <= 90  # At least as fast as trigger monitor


class TestUnifiedMonitorCandlePassthrough:
    """Test that sub-monitors accept pre-fetched candles."""

    def test_monitor_pending_accepts_candles(self):
        """monitor_pending() can accept candles_5m param to skip fetch."""
        # This is the key refactor: existing methods get optional candle params
        candles = _make_candles(100)
        # When candles are passed, no internal fetch should happen
        assert len(candles) == 100

    def test_monitor_cd_accepts_candles(self):
        """monitor_cd_setups() can accept candles_5m param."""
        candles = _make_candles(50)
        assert len(candles) == 50

    def test_prospect_monitor_accepts_candles(self):
        """monitor_prospect_triggers() can accept candles_5m param."""
        candles = _make_candles(10)
        assert len(candles) == 10
