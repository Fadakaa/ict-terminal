"""Tests for unified notification lifecycle system.

Tests all 6 stages, dedup, threading, config gates, and backward
compatibility wrappers.
"""
import os
import sqlite3
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from ml.notifications import (
    notify_lifecycle, STAGE_NAMES, STAGE_SOUNDS, STAGE_EMOJIS,
    _build_lifecycle_message, _get_current_thesis,
)
from ml.scanner_db import ScannerDB


@pytest.fixture
def db(tmp_path):
    """Create a ScannerDB with notification_lifecycle table."""
    db_path = str(tmp_path / "test_lifecycle.db")
    return ScannerDB(db_path=db_path)


def _make_narrative(bias="bullish", confidence=0.75, thesis="Test thesis",
                    scan_count=1, key_levels=None, p3_phase="manipulation",
                    invalidation=None, watching_for=None):
    """Create a narrative state dict for testing."""
    return {
        "id": "ns-test-1",
        "directional_bias": bias,
        "bias_confidence": confidence,
        "thesis": thesis,
        "scan_count": scan_count,
        "p3_phase": p3_phase,
        "key_levels": key_levels or [2338.0, 2348.0],
        "invalidation": invalidation or {"price_level": 2335.0, "direction": "below"},
        "watching_for": watching_for or ["displacement below 2338"],
        "expected_next_move": "Bullish reversal expected",
    }


def _make_setup(direction="long", entry=2340.5, sl=2337.0, tp1=2348.0,
                grade="A", kz="London", tf="1h"):
    """Create a setup data dict for testing."""
    return {
        "id": "setup-test-1",
        "direction": direction,
        "entry_price": entry,
        "sl_price": sl,
        "calibrated_sl": sl - 0.5,
        "tp1": tp1,
        "tp2": tp1 + 7,
        "tp3": tp1 + 14,
        "tps": [tp1, tp1 + 7, tp1 + 14],
        "setup_quality": grade,
        "killzone": kz,
        "timeframe": tf,
        "current_price": entry + 1.5,
    }


def _make_resolution(outcome="stopped_out", price=2337.0, rr=-1.0):
    """Create a resolution result dict for testing."""
    return {
        "outcome": outcome,
        "price": price,
        "rr": rr,
        "gross_rr": rr,
        "cost_rr": 0.05,
        "mfe_atr": 0.5,
        "mae_atr": 1.2,
    }


# ── Test: Stage message builders ────────────────────────────────────

class TestStageMessageBuilders:
    """Test that all 6 stages produce correct message content."""

    def test_stage_1_thesis_forming(self):
        ns = _make_narrative(confidence=0.65, scan_count=1)
        title, body = _build_lifecycle_message(
            1, "\U0001f4ad", "THESIS_FORMING", "1h", ns)
        assert "THESIS FORMING" in title
        assert "1h" in title
        assert "BULLISH" in title
        assert "65%" in body
        assert "manipulation" in body

    def test_stage_2_thesis_confirmed(self):
        ns = _make_narrative(confidence=0.8, scan_count=3)
        title, body = _build_lifecycle_message(
            2, "\U0001f52c", "THESIS_CONFIRMED", "1h", ns)
        assert "THESIS CONFIRMED" in title
        assert "80%" in title
        assert "Scan 3" in body

    def test_stage_3_setup_detected(self):
        ns = _make_narrative(confidence=0.85, scan_count=3)
        setup = _make_setup()
        title, body = _build_lifecycle_message(
            3, "\U0001f3af", "SETUP_DETECTED", "1h", ns, setup_data=setup)
        assert "SETUP DETECTED" in title
        assert "LONG" in title
        assert "Grade A" in title
        assert "2340.50" in body
        assert "2337.00" in body

    def test_stage_4_entry_ready(self):
        ns = _make_narrative(confidence=0.85, scan_count=3)
        setup = _make_setup()
        title, body = _build_lifecycle_message(
            4, "\u2705", "ENTRY_READY", "1h", ns, setup_data=setup)
        assert "ENTRY READY" in title
        assert "LONG" in title
        assert "LOT SIZE" in body
        assert "Risk:" in body

    def test_stage_5_trade_resolved_win(self):
        setup = _make_setup()
        result = _make_resolution("tp1", price=2348.0, rr=2.0)
        title, body = _build_lifecycle_message(
            5, "\U0001f4ca", "TRADE_RESOLVED", "1h", {},
            setup_data=setup, resolution_data=result)
        assert "TP1" in title
        assert "LONG" in title
        assert "+2.0R" in body

    def test_stage_5_trade_resolved_loss(self):
        setup = _make_setup()
        result = _make_resolution("stopped_out", price=2337.0, rr=-1.0)
        title, body = _build_lifecycle_message(
            5, "\U0001f4ca", "TRADE_RESOLVED", "1h", {},
            setup_data=setup, resolution_data=result)
        assert "STOPPED OUT" in title
        assert "-1.0R" in body

    def test_stage_5_with_post_resolution_thesis(self):
        setup = _make_setup()
        result = _make_resolution("stopped_out", price=2337.0, rr=-1.0)
        post_thesis = _make_narrative(bias="bearish", confidence=0.6,
                                       thesis="Bearish reversal forming")
        title, body = _build_lifecycle_message(
            5, "\U0001f4ca", "TRADE_RESOLVED", "1h", {},
            setup_data=setup, resolution_data=result,
            post_thesis=post_thesis)
        assert "POST-RESOLUTION" in body
        assert "BEARISH" in body
        assert "Bearish reversal" in body

    def test_stage_6_thesis_revised(self):
        ns = _make_narrative(bias="bearish", thesis="Bearish reversal forming",
                              invalidation={"price_level": 2335, "direction": "below"})
        setup = _make_setup()
        title, body = _build_lifecycle_message(
            6, "\u26a0\ufe0f", "THESIS_REVISED", "1h", ns, setup_data=setup)
        assert "THESIS REVISED" in title
        assert "BEARISH" in title
        assert "Active setup affected" in body
        assert "LONG" in body

    def test_stage_6_without_setup(self):
        ns = _make_narrative(bias="bearish", thesis="Bearish reversal forming")
        title, body = _build_lifecycle_message(
            6, "\u26a0\ufe0f", "THESIS_REVISED", "1h", ns)
        assert "THESIS REVISED" in title
        assert "Active setup affected" not in body


# ── Test: Deduplication ─────────────────────────────────────────────

class TestDeduplication:

    def test_duplicate_stage_skipped(self, db):
        """Same thesis_id + stage should not send twice (using push stage 2)."""
        ns = _make_narrative(scan_count=3, confidence=0.8)
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg123"
            notify_lifecycle(2, "thesis-001", "1h", ns, db=db)
            notify_lifecycle(2, "thesis-001", "1h", ns, db=db)

        # First call sends, second is deduped
        assert mock_tg.call_count == 1

    def test_different_stages_both_sent(self, db):
        """Different push stages for same thesis should both send."""
        ns = _make_narrative(scan_count=3, confidence=0.8)
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg123"
            notify_lifecycle(2, "thesis-002", "1h", ns, db=db)
            notify_lifecycle(4, "thesis-002", "1h", ns,
                             setup_data=setup, db=db)

        assert mock_tg.call_count == 2

    def test_lower_stage_skipped_when_higher_sent(self, db):
        """If stage 4 already sent, don't send late stage 3."""
        ns = _make_narrative()
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg123"
            notify_lifecycle(4, "thesis-003", "1h", ns,
                             setup_data=setup, db=db)
            notify_lifecycle(3, "thesis-003", "1h", ns,
                             setup_data=setup, db=db)

        # Stage 4 sent, stage 3 skipped
        assert mock_tg.call_count == 1

    def test_stage_5_always_allowed(self, db):
        """Stage 5 (TRADE_RESOLVED) should always fire even with higher dedup."""
        ns = _make_narrative()
        setup = _make_setup()
        result = _make_resolution()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg123"
            # Send stage 4 first
            notify_lifecycle(4, "thesis-004", "1h", ns,
                             setup_data=setup, db=db)
            # Stage 5 should still send
            notify_lifecycle(5, "thesis-004", "1h", ns,
                             setup_data=setup, resolution_data=result, db=db)

        assert mock_tg.call_count == 2

    def test_stage_6_always_allowed(self, db):
        """Stage 6 (THESIS_REVISED) should always fire."""
        ns_bullish = _make_narrative()
        ns_bearish = _make_narrative(bias="bearish")
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg123"
            notify_lifecycle(4, "thesis-005", "1h", ns_bullish,
                             setup_data=setup, db=db)
            notify_lifecycle(6, "thesis-005", "1h", ns_bearish,
                             setup_data=setup, db=db)

        assert mock_tg.call_count == 2


# ── Test: Config gates ──────────────────────────────────────────────

class TestConfigGates:

    def test_thesis_forming_gate(self, db):
        """Stage 1 skipped when notify_thesis_forming=False."""
        ns = _make_narrative()
        with patch("ml.notifications.get_config") as mock_cfg, \
             patch("ml.notifications._send_telegram_html") as mock_tg:
            mock_cfg.return_value = {"notify_thesis_forming": False,
                                     "notify_telegram": True}
            notify_lifecycle(1, "thesis-010", "1h", ns, db=db)

        mock_tg.assert_not_called()

    def test_thesis_confirmed_gate(self, db):
        """Stage 2 skipped when notify_thesis_confirmed=False."""
        ns = _make_narrative(scan_count=3, confidence=0.8)
        with patch("ml.notifications.get_config") as mock_cfg, \
             patch("ml.notifications._send_telegram_html") as mock_tg:
            mock_cfg.return_value = {"notify_thesis_confirmed": False,
                                     "notify_telegram": True}
            notify_lifecycle(2, "thesis-011", "1h", ns, db=db)

        mock_tg.assert_not_called()

    def test_thesis_revised_gate(self, db):
        """Stage 6 skipped when notify_thesis_revised=False."""
        ns = _make_narrative(bias="bearish")
        with patch("ml.notifications.get_config") as mock_cfg, \
             patch("ml.notifications._send_telegram_html") as mock_tg:
            mock_cfg.return_value = {"notify_thesis_revised": False,
                                     "notify_telegram": True}
            notify_lifecycle(6, "thesis-012", "1h", ns, db=db)

        mock_tg.assert_not_called()


# ── Test: Transport routing ─────────────────────────────────────────

class TestTransportRouting:

    def test_stage_1_log_only(self, db):
        """Stage 1 (THESIS_FORMING) is log-only after Phase 6 simplification."""
        ns = _make_narrative()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos") as mock_macos:
            mock_tg.return_value = "msg123"
            notify_lifecycle(1, "thesis-020", "1h", ns, db=db)

        # Stage 1 no longer pushes to Telegram — Phase 6
        mock_tg.assert_not_called()
        mock_macos.assert_not_called()

    def test_stage_3_log_only(self, db):
        """Stage 3 (SETUP_DETECTED) is log-only after Phase 6 simplification."""
        ns = _make_narrative()
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos") as mock_macos:
            mock_tg.return_value = "msg123"
            notify_lifecycle(3, "thesis-021", "1h", ns,
                             setup_data=setup, db=db)

        # Stage 3 is no longer pushed — Phase 6 notification simplification
        mock_tg.assert_not_called()
        mock_macos.assert_not_called()

    def test_stage_6_warning_sound(self, db):
        """Stage 6 should use Basso warning sound."""
        ns = _make_narrative(bias="bearish")
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos") as mock_macos:
            mock_tg.return_value = "msg123"
            notify_lifecycle(6, "thesis-022", "1h", ns,
                             setup_data=setup, db=db)

        mock_macos.assert_called_once()
        call_args = mock_macos.call_args
        assert call_args.kwargs.get("sound") == "Basso" or call_args[2] == "Basso"


# ── Test: DB lifecycle tracking ─────────────────────────────────────

class TestDBLifecycleTracking:

    def test_notification_recorded(self, db):
        """Sent notifications are recorded in the DB."""
        ns = _make_narrative()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg456"
            notify_lifecycle(1, "thesis-030", "1h", ns, db=db)

        history = db.get_lifecycle_history("thesis-030")
        assert len(history) == 1
        assert history[0]["stage"] == 1
        assert history[0]["stage_name"] == "THESIS_FORMING"
        assert history[0]["thesis_id"] == "thesis-030"
        assert history[0]["timeframe"] == "1h"

    def test_thread_msg_id_stored(self, db):
        """Telegram message_id is stored for threading (uses push stage)."""
        ns = _make_narrative()
        ns["scan_count"] = 3
        ns["bias_confidence"] = 0.8
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg789"
            notify_lifecycle(2, "thesis-031", "1h", ns, db=db)

        thread_id = db.get_lifecycle_thread_msg_id("thesis-031")
        assert thread_id == "msg789"

    def test_already_sent_check(self, db):
        """already_sent returns True after recording."""
        db.record_lifecycle_notification(
            "thesis-032", "1h", 1, "THESIS_FORMING")
        assert db.lifecycle_already_sent("thesis-032", 1) is True
        assert db.lifecycle_already_sent("thesis-032", 2) is False

    def test_max_stage_sent(self, db):
        """max_stage_sent returns the highest stage recorded."""
        db.record_lifecycle_notification(
            "thesis-033", "1h", 1, "THESIS_FORMING")
        db.record_lifecycle_notification(
            "thesis-033", "1h", 3, "SETUP_DETECTED")
        assert db.lifecycle_max_stage_sent("thesis-033") == 3

    def test_max_stage_sent_empty(self, db):
        """max_stage_sent returns 0 when nothing sent."""
        assert db.lifecycle_max_stage_sent("thesis-nonexistent") == 0


# ── Test: Telegram threading ────────────────────────────────────────

class TestTelegramThreading:

    def test_reply_to_used_for_second_stage(self, db):
        """Second push stage should reply to the first push stage."""
        ns = _make_narrative(scan_count=3, confidence=0.8)
        setup = _make_setup()
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = "msg100"
            # Stage 2 (THESIS_CONFIRMED) — push stage
            notify_lifecycle(2, "thesis-040", "1h", ns, db=db)

            mock_tg.return_value = "msg101"
            # Stage 4 (ENTRY_READY) — push stage, should thread
            notify_lifecycle(4, "thesis-040", "1h", ns,
                             setup_data=setup, db=db)

        # Second call should have reply_to_message_id
        assert len(mock_tg.call_args_list) >= 2
        second_call = mock_tg.call_args_list[1]
        assert second_call.kwargs.get("reply_to_message_id") == "msg100"


# ── Test: Notify without DB (graceful) ──────────────────────────────

class TestWithoutDB:

    def test_notify_lifecycle_works_without_db(self):
        """notify_lifecycle should work even without a DB (no dedup).
        Uses push stage 2 since stage 1 is log-only after Phase 6."""
        ns = _make_narrative(scan_count=3, confidence=0.8)
        with patch("ml.notifications._send_telegram_html") as mock_tg, \
             patch("ml.notifications._send_macos"):
            mock_tg.return_value = None
            notify_lifecycle(2, "thesis-050", "1h", ns)

        mock_tg.assert_called_once()


# ── Test: Stage constants ───────────────────────────────────────────

class TestConstants:

    def test_all_stages_have_names(self):
        for stage in range(1, 7):
            assert stage in STAGE_NAMES

    def test_all_stages_have_sounds(self):
        for stage in range(1, 7):
            assert stage in STAGE_SOUNDS

    def test_all_stages_have_emojis(self):
        for stage in range(1, 7):
            assert stage in STAGE_EMOJIS

    def test_stage_1_no_sound(self):
        assert STAGE_SOUNDS[1] is None

    def test_stage_6_basso(self):
        assert STAGE_SOUNDS[6] == "Basso"
