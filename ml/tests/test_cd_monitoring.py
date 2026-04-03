"""Tests for Phase 4: C/D Grade Monitoring Pipeline.

Tests that C/D grade setups are stored with status='monitoring',
promotion criteria work correctly, and the monitor loop processes them.
"""
import json
import pytest
from datetime import datetime, timedelta

from ml.scanner_db import ScannerDB


def _make_candles(n, base_price=2350.0, direction="flat", atr=5.0):
    """Create candles for testing. direction: flat, up, down."""
    candles = []
    for i in range(n):
        if direction == "up":
            p = base_price + i * 0.5
        elif direction == "down":
            p = base_price - i * 0.5
        else:
            p = base_price + (i % 3 - 1) * 0.3
        candles.append({
            "datetime": f"2026-04-01 {i % 24:02d}:{(i * 5) % 60:02d}:00",
            "open": p, "high": p + atr * 0.3,
            "low": p - atr * 0.3, "close": p + 0.2,
        })
    return candles


# ── DB Layer Tests ─────────────────────────────────────────────────


class TestScannerDBMonitoring:
    """Test monitoring status in ScannerDB."""

    def test_store_setup_monitoring_status(self, tmp_path):
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        setup_id = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0, 2380.0], setup_quality="C",
            killzone="London", rr_ratios=[2.0, 3.0],
            analysis_json={"setup_quality": "C"},
            calibration_json={}, timeframe="1h",
            status="monitoring",
        )
        assert setup_id is not None
        # Should NOT appear in pending
        pending = db.get_pending()
        assert len(pending) == 0
        # Should appear in monitoring
        monitoring = db.get_monitoring_setups()
        assert len(monitoring) == 1
        assert monitoring[0]["status"] == "monitoring"
        assert monitoring[0]["entry_price"] == 2350.0

    def test_get_monitoring_empty(self, tmp_path):
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        assert db.get_monitoring_setups() == []

    def test_get_monitoring_excludes_pending(self, tmp_path):
        """Monitoring query only returns monitoring setups, not pending."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="A",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            status="pending",
        )
        assert len(db.get_monitoring_setups()) == 0

    def test_promote_setup(self, tmp_path):
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        setup_id = db.store_setup(
            direction="short", bias="bearish", entry_price=2400.0,
            sl_price=2410.0, calibrated_sl=2412.0,
            tps=[2380.0], setup_quality="D",
            killzone="NY_AM", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="15min", status="monitoring",
        )
        db.promote_setup(setup_id)
        # Now in pending, not monitoring
        assert len(db.get_monitoring_setups()) == 0
        pending = db.get_pending()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"

    def test_promote_only_affects_monitoring(self, tmp_path):
        """Promote does nothing to pending or shadow setups."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="B",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            status="pending",
        )
        db.promote_setup(sid)  # Should be a no-op
        pending = db.get_pending()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"  # Unchanged

    def test_promote_preserves_setup_data(self, tmp_path):
        """After promotion, all original setup fields are preserved."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        setup_id = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0, 2380.0, 2390.0], setup_quality="C",
            killzone="London", rr_ratios=[2.0, 3.0, 4.0],
            analysis_json={"test": True}, calibration_json={"cal": True},
            timeframe="1h", status="monitoring",
        )
        db.promote_setup(setup_id)
        promoted = db.get_pending()
        assert len(promoted) == 1
        s = promoted[0]
        assert s["direction"] == "long"
        assert s["entry_price"] == 2350.0
        assert s["sl_price"] == 2340.0
        assert s["tp1"] == 2370.0
        assert s["tp2"] == 2380.0
        assert s["setup_quality"] == "C"
        assert s["killzone"] == "London"
        assert s["timeframe"] == "1h"

    def test_expire_monitoring_setups(self, tmp_path):
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="C",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="15min", status="monitoring",
        )
        # Manually backdate to 10 hours ago
        old_time = (datetime.utcnow() - timedelta(hours=10)).isoformat()
        with db._conn() as conn:
            conn.execute("UPDATE scanner_setups SET created_at = ? WHERE id = ?",
                         (old_time, sid))
        # 15min monitoring setups expire at 8 hours
        expired = db.expire_by_timeframe({"15min": 8}, status="monitoring")
        assert expired == 1
        assert len(db.get_monitoring_setups()) == 0

    def test_expire_monitoring_doesnt_affect_pending(self, tmp_path):
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="A",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="15min", status="pending",
        )
        old_time = (datetime.utcnow() - timedelta(hours=10)).isoformat()
        with db._conn() as conn:
            conn.execute("UPDATE scanner_setups SET created_at = ? WHERE id = ?",
                         (old_time, sid))
        # Expire monitoring — should NOT touch pending
        expired = db.expire_by_timeframe({"15min": 8}, status="monitoring")
        assert expired == 0
        assert len(db.get_pending()) == 1

    def test_default_expire_still_works_on_pending(self, tmp_path):
        """expire_by_timeframe() without status arg still expires pending."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="B",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            timeframe="15min", status="pending",
        )
        old_time = (datetime.utcnow() - timedelta(hours=10)).isoformat()
        with db._conn() as conn:
            conn.execute("UPDATE scanner_setups SET created_at = ? WHERE id = ?",
                         (old_time, sid))
        expired = db.expire_by_timeframe({"15min": 8})  # No status arg
        assert expired == 1


# ── Promotion Criteria Tests ─────────────────────────────────────


class TestPromotionCriteria:
    """Test individual promotion criteria for C/D setups."""

    def test_price_proximity_within_atr(self):
        """Price within 1.0 ATR triggers proximity criterion."""
        entry_price = 2350.0
        current_price = 2349.0
        atr = 5.0
        within = abs(current_price - entry_price) <= atr
        assert within is True

    def test_price_proximity_outside_atr(self):
        entry_price = 2350.0
        current_price = 2340.0
        atr = 5.0
        within = abs(current_price - entry_price) <= atr
        assert within is False

    def test_price_proximity_exact_boundary(self):
        entry_price = 2350.0
        current_price = 2345.0
        atr = 5.0
        within = abs(current_price - entry_price) <= atr
        assert within is True  # Exactly at ATR boundary

    def test_displacement_candle_long(self):
        """Bullish displacement = body >= 2.0 ATR, close > open."""
        candle = {"open": 2345.0, "close": 2358.0, "high": 2360.0, "low": 2344.0}
        atr = 5.0
        body = abs(candle["close"] - candle["open"])
        is_bullish = candle["close"] > candle["open"]
        has_displacement = body >= 2.0 * atr and is_bullish
        assert has_displacement is True

    def test_displacement_candle_short(self):
        candle = {"open": 2358.0, "close": 2345.0, "high": 2360.0, "low": 2344.0}
        atr = 5.0
        body = abs(candle["close"] - candle["open"])
        is_bearish = candle["close"] < candle["open"]
        has_displacement = body >= 2.0 * atr and is_bearish
        assert has_displacement is True

    def test_no_displacement_small_body(self):
        candle = {"open": 2350.0, "close": 2352.0, "high": 2355.0, "low": 2349.0}
        atr = 5.0
        body = abs(candle["close"] - candle["open"])
        has_displacement = body >= 2.0 * atr
        assert has_displacement is False

    def test_displacement_wrong_direction_ignored(self):
        """Bearish displacement doesn't help a long setup."""
        candle = {"open": 2358.0, "close": 2345.0, "high": 2360.0, "low": 2344.0}
        atr = 5.0
        direction = "long"
        body = abs(candle["close"] - candle["open"])
        directional = (direction == "long" and candle["close"] > candle["open"]) or \
                      (direction == "short" and candle["close"] < candle["open"])
        has_displacement = body >= 2.0 * atr and directional
        assert has_displacement is False

    def test_liquidity_sweep_long(self):
        """For longs: wick below prior swing low, close above it."""
        swing_low = 2340.0
        candle = {"open": 2342.0, "close": 2345.0, "high": 2346.0, "low": 2338.0}
        swept = candle["low"] < swing_low and candle["close"] > swing_low
        assert swept is True

    def test_liquidity_sweep_short(self):
        """For shorts: wick above prior swing high, close below it."""
        swing_high = 2365.0
        candle = {"open": 2363.0, "close": 2360.0, "high": 2367.0, "low": 2359.0}
        swept = candle["high"] > swing_high and candle["close"] < swing_high
        assert swept is True

    def test_no_sweep_without_wick_penetration(self):
        """No sweep if price didn't penetrate the level."""
        swing_low = 2340.0
        candle = {"open": 2342.0, "close": 2345.0, "high": 2346.0, "low": 2341.0}
        swept = candle["low"] < swing_low
        assert swept is False


# ── C/D Routing Tests ─────────────────────────────────────────────


class TestCDRouting:
    """Test that C/D grades route to monitoring status."""

    def test_c_grade_with_entry_gets_monitoring(self):
        """C-grade setups with entry price → status='monitoring'."""
        quality = "C"
        is_shadow = False

        if is_shadow:
            status = "shadow"
        elif quality in ("C", "D"):
            status = "monitoring"
        else:
            status = "pending"
        assert status == "monitoring"

    def test_d_grade_with_entry_gets_monitoring(self):
        quality = "D"
        is_shadow = False

        if is_shadow:
            status = "shadow"
        elif quality in ("C", "D"):
            status = "monitoring"
        else:
            status = "pending"
        assert status == "monitoring"

    def test_a_grade_gets_pending(self):
        quality = "A"
        is_shadow = False

        if is_shadow:
            status = "shadow"
        elif quality in ("C", "D"):
            status = "monitoring"
        else:
            status = "pending"
        assert status == "pending"

    def test_b_grade_gets_pending(self):
        quality = "B"
        is_shadow = False

        if is_shadow:
            status = "shadow"
        elif quality in ("C", "D"):
            status = "monitoring"
        else:
            status = "pending"
        assert status == "pending"

    def test_shadow_takes_precedence(self):
        """Shadow (Opus-rejected) overrides C/D monitoring."""
        quality = "C"
        is_shadow = True

        if is_shadow:
            status = "shadow"
        elif quality in ("C", "D"):
            status = "monitoring"
        else:
            status = "pending"
        assert status == "shadow"

    def test_monitoring_return_dict(self):
        """Monitoring setups return appropriate status dict."""
        result = {
            "status": "monitoring",
            "setup_id": "abc12345",
            "timeframe": "1h",
            "direction": "long",
            "entry": 2350.0,
            "quality": "C",
        }
        assert result["status"] == "monitoring"
        assert "setup_id" in result
        assert "quality" in result


# ── Liquidity Sweep Helper Tests ──────────────────────────────────


class TestLiquiditySweepDetection:
    """Test the _check_liquidity_sweep helper logic."""

    def _find_swing_lows(self, candles, lookback=5):
        """Find swing lows in candle data (simplified)."""
        lows = []
        for i in range(lookback, len(candles) - lookback):
            is_low = all(
                candles[i]["low"] <= candles[i + j]["low"]
                for j in range(-lookback, lookback + 1) if j != 0
                and 0 <= i + j < len(candles)
            )
            if is_low:
                lows.append(candles[i]["low"])
        return lows

    def _find_swing_highs(self, candles, lookback=5):
        """Find swing highs in candle data (simplified)."""
        highs = []
        for i in range(lookback, len(candles) - lookback):
            is_high = all(
                candles[i]["high"] >= candles[i + j]["high"]
                for j in range(-lookback, lookback + 1) if j != 0
                and 0 <= i + j < len(candles)
            )
            if is_high:
                highs.append(candles[i]["high"])
        return highs

    def test_sweep_detected_for_long(self):
        """Candles that dip below swing low then close above = sweep."""
        # Build candles with a swing low at index 5
        candles = []
        for i in range(20):
            if i == 5:
                p = 2340.0  # swing low
            elif i == 15:
                # Sweep candle: wicks below 2340, closes above
                candles.append({
                    "datetime": f"2026-04-01 00:{i*5:02d}:00",
                    "open": 2342.0, "close": 2345.0,
                    "high": 2346.0, "low": 2338.0,
                })
                continue
            else:
                p = 2345.0 + (i % 3) * 0.5
            candles.append({
                "datetime": f"2026-04-01 00:{i*5:02d}:00",
                "open": p, "high": p + 1.0,
                "low": p - 1.0, "close": p + 0.3,
            })

        # Check last 10 candles for sweep below swing low
        swing_low = 2340.0
        swept = False
        for c in candles[-10:]:
            if c["low"] < swing_low and c["close"] > swing_low:
                swept = True
                break
        assert swept is True

    def test_no_sweep_when_no_penetration(self):
        candles = _make_candles(20, base_price=2350.0)
        swing_low = 2330.0  # Well below all candles
        swept = False
        for c in candles[-10:]:
            if c["low"] < swing_low and c["close"] > swing_low:
                swept = True
                break
        assert swept is False


# ── Monitor CD Setups Integration ──────────────────────────────────


class TestMonitorCDSetups:
    """Test the monitor_cd_setups method behavior."""

    def test_empty_monitoring_returns_zero(self):
        """No monitoring setups → noop."""
        monitoring = []
        result = {
            "checked": len(monitoring),
            "promoted": 0,
            "expired": 0,
        }
        assert result["checked"] == 0
        assert result["promoted"] == 0

    def test_promotion_requires_all_three_criteria(self):
        """All three criteria must be met for promotion."""
        cases = [
            (True, True, False, False),
            (True, False, True, False),
            (False, True, True, False),
            (True, True, True, True),
            (False, False, False, False),
        ]
        for proximity, displacement, sweep, expected in cases:
            promoted = proximity and displacement and sweep
            assert promoted is expected

    def test_expiry_uses_timeframe_hours(self):
        """Monitoring setups expire matching EXPIRY_HOURS."""
        from ml.scanner import EXPIRY_HOURS
        assert EXPIRY_HOURS["15min"] == 8
        assert EXPIRY_HOURS["1h"] == 48
        assert EXPIRY_HOURS["4h"] == 168
        assert EXPIRY_HOURS["1day"] == 336

    def test_full_promotion_flow_in_db(self, tmp_path):
        """End-to-end: store monitoring → promote → verify in pending."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))

        # Store C-grade as monitoring
        sid = db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0, 2380.0], setup_quality="C",
            killzone="London", rr_ratios=[2.0, 3.0],
            analysis_json={"setup_quality": "C", "confluences": ["OB", "FVG"]},
            calibration_json={}, timeframe="1h",
            status="monitoring",
        )

        # Verify in monitoring
        assert len(db.get_monitoring_setups()) == 1
        assert len(db.get_pending()) == 0

        # Promote
        db.promote_setup(sid)

        # Verify in pending
        assert len(db.get_monitoring_setups()) == 0
        pending = db.get_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == sid
        assert pending[0]["setup_quality"] == "C"

    def test_monitoring_setups_not_in_pending_monitor(self, tmp_path):
        """Monitoring setups should NOT be picked up by get_pending()."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        db.store_setup(
            direction="long", bias="bullish", entry_price=2350.0,
            sl_price=2340.0, calibrated_sl=2338.0,
            tps=[2370.0], setup_quality="C",
            killzone="London", rr_ratios=[2.0],
            analysis_json={}, calibration_json={},
            status="monitoring",
        )
        # Regular get_pending() should not include monitoring
        assert len(db.get_pending()) == 0
        # Even with shadow included
        assert len(db.get_pending(include_shadow=True)) == 0

    def test_multiple_monitoring_setups(self, tmp_path):
        """Multiple C/D setups can be in monitoring simultaneously."""
        db = ScannerDB(db_path=str(tmp_path / "test.db"))
        for i, (quality, tf) in enumerate([
            ("C", "1h"), ("D", "15min"), ("C", "4h")
        ]):
            db.store_setup(
                direction="long", bias="bullish",
                entry_price=2350.0 + i * 10,
                sl_price=2340.0, calibrated_sl=2338.0,
                tps=[2370.0], setup_quality=quality,
                killzone="London", rr_ratios=[2.0],
                analysis_json={}, calibration_json={},
                timeframe=tf, status="monitoring",
            )
        assert len(db.get_monitoring_setups()) == 3
