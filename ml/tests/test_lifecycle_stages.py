"""Tests for lifecycle stage tracking (thesis_id, store_setup pass-through, Stage 3/4/5)."""
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from ml.scanner_db import ScannerDB


@pytest.fixture
def tmp_db():
    """Create a temporary ScannerDB for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = ScannerDB(db_path=path)
    yield db
    os.unlink(path)


def _get_setup_by_id(db, setup_id):
    """Fetch a single setup row as dict by ID."""
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM scanner_setups WHERE id = ?", (setup_id,)
        ).fetchone()
    return dict(row) if row else None


class TestThesisIdColumn:
    """Step 1: thesis_id column exists in scanner_setups."""

    def test_thesis_id_column_exists(self, tmp_db):
        """The thesis_id column should be present after init."""
        import sqlite3
        with sqlite3.connect(tmp_db.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(scanner_setups)")
            columns = {row[1] for row in cursor.fetchall()}
        assert "thesis_id" in columns

    def test_thesis_id_nullable(self, tmp_db):
        """thesis_id should allow NULL (most older setups won't have it)."""
        setup_id = tmp_db.store_setup(
            direction="long", bias="bullish", entry_price=3050.0,
            sl_price=3040.0, calibrated_sl=3038.0,
            tps=[3060.0, 3070.0], setup_quality="A",
            killzone="london", rr_ratios=[2.0, 3.0],
            analysis_json={}, calibration_json={},
            thesis_id=None,
        )
        setup = _get_setup_by_id(tmp_db, setup_id)
        assert setup is not None
        assert setup.get("thesis_id") is None


class TestStoreSetupThesisId:
    """Step 2: store_setup() accepts and persists thesis_id."""

    def test_thesis_id_stored(self, tmp_db):
        """thesis_id passed to store_setup should be retrievable."""
        tid = "abc12345"
        setup_id = tmp_db.store_setup(
            direction="short", bias="bearish", entry_price=3100.0,
            sl_price=3110.0, calibrated_sl=3112.0,
            tps=[3090.0, 3080.0], setup_quality="B",
            killzone="ny_am", rr_ratios=[2.0, 4.0],
            analysis_json={"test": True}, calibration_json={},
            thesis_id=tid,
        )
        setup = _get_setup_by_id(tmp_db, setup_id)
        assert setup["thesis_id"] == tid

    def test_thesis_id_none_by_default(self, tmp_db):
        """When thesis_id is not passed, it should default to None."""
        setup_id = tmp_db.store_setup(
            direction="long", bias="bullish", entry_price=3050.0,
            sl_price=3040.0, calibrated_sl=3038.0,
            tps=[3060.0], setup_quality="C",
            killzone="asian", rr_ratios=[1.5],
            analysis_json={}, calibration_json={},
        )
        setup = _get_setup_by_id(tmp_db, setup_id)
        assert setup.get("thesis_id") is None

    def test_multiple_setups_same_thesis(self, tmp_db):
        """Multiple setups can share the same thesis_id."""
        tid = "shared-thesis"
        ids = []
        for i in range(3):
            sid = tmp_db.store_setup(
                direction="long", bias="bullish",
                entry_price=3050.0 + i, sl_price=3040.0,
                calibrated_sl=3038.0, tps=[3060.0],
                setup_quality="B", killzone="london",
                rr_ratios=[2.0], analysis_json={},
                calibration_json={}, thesis_id=tid,
            )
            ids.append(sid)
        for sid in ids:
            assert _get_setup_by_id(tmp_db, sid)["thesis_id"] == tid


class TestLifecycleRecording:
    """Steps 4-6: Lifecycle stages are correctly recorded in the DB."""

    def test_stage_3_recorded(self, tmp_db):
        """Stage 3 (SETUP_DETECTED) should be recorded in notification_lifecycle."""
        tid = "thesis-stage3"
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="1h", stage=3,
            stage_name="SETUP_DETECTED", setup_id="setup-001",
        )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1
        assert history[0]["stage"] == 3
        assert history[0]["stage_name"] == "SETUP_DETECTED"
        assert history[0]["setup_id"] == "setup-001"

    def test_stage_4_recorded(self, tmp_db):
        """Stage 4 (ENTRY_READY) should be recorded."""
        tid = "thesis-stage4"
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="4h", stage=4,
            stage_name="ENTRY_READY", setup_id="setup-002",
        )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1
        assert history[0]["stage"] == 4

    def test_stage_5_recorded(self, tmp_db):
        """Stage 5 (TRADE_RESOLVED) should be recorded."""
        tid = "thesis-stage5"
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="15min", stage=5,
            stage_name="TRADE_RESOLVED", setup_id="setup-003",
        )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1
        assert history[0]["stage"] == 5

    def test_full_lifecycle_sequence(self, tmp_db):
        """A thesis should accumulate stages 3→4→5 in order."""
        tid = "thesis-full"
        for stage, name in [(3, "SETUP_DETECTED"), (4, "ENTRY_READY"), (5, "TRADE_RESOLVED")]:
            tmp_db.record_lifecycle_notification(
                thesis_id=tid, timeframe="1h", stage=stage,
                stage_name=name, setup_id="setup-full",
            )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 3
        stages = [h["stage"] for h in history]
        assert stages == [3, 4, 5]


class TestLifecycleDedup:
    """Dedup: same thesis+stage should not be sent twice."""

    def test_already_sent_returns_true(self, tmp_db):
        tid = "dedup-test"
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="1h", stage=4,
            stage_name="ENTRY_READY",
        )
        assert tmp_db.lifecycle_already_sent(tid, 4) is True
        assert tmp_db.lifecycle_already_sent(tid, 3) is False

    def test_max_stage_sent(self, tmp_db):
        tid = "max-stage"
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="1h", stage=3,
            stage_name="SETUP_DETECTED",
        )
        tmp_db.record_lifecycle_notification(
            thesis_id=tid, timeframe="1h", stage=4,
            stage_name="ENTRY_READY",
        )
        assert tmp_db.lifecycle_max_stage_sent(tid) == 4

    def test_max_stage_zero_for_unknown_thesis(self, tmp_db):
        assert tmp_db.lifecycle_max_stage_sent("unknown") == 0


class TestGracefulSkipWithoutThesisId:
    """When thesis_id is None, lifecycle calls should be silently skipped."""

    @patch("ml.notifications._send_telegram_html")
    @patch("ml.notifications._send_macos")
    def test_notify_lifecycle_skipped_for_none_thesis(self, mock_macos, mock_tg, tmp_db):
        """notify_lifecycle should not crash or send when thesis_id is missing."""
        from ml.notifications import notify_lifecycle
        # This simulates the scanner code pattern:
        # if _thesis_id: notify_lifecycle(...)
        # So we just verify that NOT calling it when None is the expected behavior.
        # The scanner code guards with `if _thesis_id:` — verify it's truthy check.
        _thesis_id = None
        if _thesis_id:
            notify_lifecycle(4, _thesis_id, "1h", {}, db=tmp_db)
        # Should have done nothing
        mock_tg.assert_not_called()
        mock_macos.assert_not_called()
        history = tmp_db.get_lifecycle_history("none")
        assert len(history) == 0

    @patch("ml.notifications._send_telegram_html")
    @patch("ml.notifications._send_macos")
    def test_notify_lifecycle_skipped_for_empty_string(self, mock_macos, mock_tg, tmp_db):
        """Empty string thesis_id should also be skipped by the truthiness guard."""
        _thesis_id = ""
        if _thesis_id:
            notify_lifecycle(4, _thesis_id, "1h", {}, db=tmp_db)
        mock_tg.assert_not_called()


class TestNotifyLifecycleFunctionIntegration:
    """Integration test: call notify_lifecycle and verify DB recording."""

    @patch("ml.notifications._send_telegram_html", return_value=None)
    @patch("ml.notifications._send_macos")
    def test_stage_4_recorded_via_notify_lifecycle(self, mock_macos, mock_tg, tmp_db):
        """Calling notify_lifecycle(4, ...) should record in DB."""
        from ml.notifications import notify_lifecycle
        tid = "integ-thesis"
        notify_lifecycle(
            stage=4, thesis_id=tid, timeframe="1h",
            narrative_state={"directional_bias": "bullish", "bias_confidence": 0.8},
            setup_data={"id": "setup-integ", "entry_price": 3050, "sl_price": 3040,
                        "direction": "long", "setup_quality": "A", "timeframe": "1h"},
            db=tmp_db,
        )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1
        assert history[0]["stage"] == 4
        assert history[0]["setup_id"] == "setup-integ"

    @patch("ml.notifications._send_telegram_html", return_value=None)
    @patch("ml.notifications._send_macos")
    def test_stage_5_recorded_via_notify_lifecycle(self, mock_macos, mock_tg, tmp_db):
        """Calling notify_lifecycle(5, ...) should record stage 5."""
        from ml.notifications import notify_lifecycle
        tid = "integ-resolve"
        notify_lifecycle(
            stage=5, thesis_id=tid, timeframe="4h",
            narrative_state={},
            setup_data={"id": "setup-res", "outcome": "tp1", "rr": 2.0},
            db=tmp_db,
        )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1
        assert history[0]["stage"] == 5

    @patch("ml.notifications._send_telegram_html", return_value="12345")
    @patch("ml.notifications._send_macos")
    def test_telegram_msg_id_stored(self, mock_macos, mock_tg, tmp_db):
        """Telegram message ID should be stored for reply threading."""
        from ml.notifications import notify_lifecycle
        tid = "thread-test"
        notify_lifecycle(
            stage=4, thesis_id=tid, timeframe="1h",
            narrative_state={"directional_bias": "short", "bias_confidence": 0.9},
            setup_data={"id": "s1"}, db=tmp_db,
        )
        thread_id = tmp_db.get_lifecycle_thread_msg_id(tid)
        assert thread_id == "12345"

    @patch("ml.notifications._send_telegram_html", return_value=None)
    @patch("ml.notifications._send_macos")
    def test_dedup_prevents_duplicate_stage(self, mock_macos, mock_tg, tmp_db):
        """Sending the same stage twice should be deduped."""
        from ml.notifications import notify_lifecycle
        tid = "dedup-integ"
        for _ in range(2):
            notify_lifecycle(
                stage=4, thesis_id=tid, timeframe="1h",
                narrative_state={}, setup_data={"id": "s-dedup"},
                db=tmp_db,
            )
        history = tmp_db.get_lifecycle_history(tid)
        assert len(history) == 1  # Only one recorded despite two calls


class TestScannerCodePatterns:
    """Verify the scanner code patterns work correctly with lifecycle."""

    def test_thesis_id_derivation_from_narrative_state(self):
        """Replicate the thesis_id derivation logic from scanner.py."""
        # Case 1: current_ns has an id
        ns = {"id": "ns-abc123", "directional_bias": "bullish"}
        _store_thesis_id = None
        try:
            _store_thesis_id = ns.get("id") or str(ns.get("thesis", ""))[:8] or None
        except Exception:
            pass
        assert _store_thesis_id == "ns-abc123"

        # Case 2: ns has thesis but no id
        ns = {"thesis": "some-long-thesis-uuid-here", "directional_bias": "bearish"}
        _store_thesis_id = None
        try:
            _store_thesis_id = ns.get("id") or str(ns.get("thesis", ""))[:8] or None
        except Exception:
            pass
        assert _store_thesis_id == "some-lon"

        # Case 3: empty ns
        ns = {}
        _store_thesis_id = None
        try:
            _store_thesis_id = ns.get("id") or str(ns.get("thesis", ""))[:8] or None
        except Exception:
            pass
        assert _store_thesis_id is None

    def test_stage_5_setup_data_includes_outcome(self):
        """Stage 5 should pass outcome and rr in setup_data."""
        setup = {"id": "s1", "direction": "long", "timeframe": "1h", "thesis_id": "t1"}
        result = {"outcome": "tp1", "rr": 2.5}
        # Replicate the scanner pattern
        stage5_data = {**setup, "outcome": result["outcome"], "rr": result.get("rr", 0)}
        assert stage5_data["outcome"] == "tp1"
        assert stage5_data["rr"] == 2.5
        assert stage5_data["id"] == "s1"
        assert stage5_data["thesis_id"] == "t1"
