"""Tests for system evolution snapshot recorder."""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from ml.system_snapshot import SystemSnapshotRecorder


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DB with the snapshot table + scanner_setups."""
    db_path = str(tmp_path / "test_scanner.db")
    conn = sqlite3.connect(db_path)
    # Scanner setups table (minimal for backfill testing)
    conn.execute("""
        CREATE TABLE scanner_setups (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            timeframe TEXT NOT NULL DEFAULT '1h',
            direction TEXT NOT NULL,
            entry_price REAL,
            outcome TEXT,
            pnl_rr REAL,
            mfe_atr REAL,
            mae_atr REAL,
            analysis_json TEXT,
            calibration_json TEXT,
            killzone TEXT,
            setup_quality TEXT
        )
    """)
    # Narrative states table (for thesis accuracy)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS narrative_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            thesis TEXT,
            prediction_hit INTEGER,
            is_revision INTEGER DEFAULT 0,
            invalidation_hit INTEGER,
            thesis_age_minutes REAL,
            status TEXT DEFAULT 'active'
        )
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def recorder(tmp_db):
    return SystemSnapshotRecorder(db_path=tmp_db)


class TestTakeSnapshot:
    """Test core snapshot recording."""

    def test_take_snapshot_returns_dict(self, recorder):
        snap = recorder.take_snapshot(trigger="test")
        assert isinstance(snap, dict)
        assert "timestamp" in snap
        assert snap["trigger"] == "test"

    def test_take_snapshot_stores_in_db(self, recorder):
        recorder.take_snapshot(trigger="test")
        count = recorder.get_snapshot_count()
        assert count == 1

    def test_take_multiple_snapshots(self, recorder):
        recorder.take_snapshot(trigger="first")
        recorder.take_snapshot(trigger="second")
        assert recorder.get_snapshot_count() == 2

    def test_snapshot_has_expected_keys(self, recorder):
        snap = recorder.take_snapshot()
        expected_keys = {"timestamp", "trigger", "narrative_weights",
                        "bayesian", "bandit", "setup_profiles",
                        "thesis_accuracy", "loss_types",
                        "learned_rules_count"}
        assert expected_keys.issubset(snap.keys())

    def test_snapshot_trigger_stored(self, recorder):
        recorder.take_snapshot(trigger="manual")
        snaps = recorder.get_snapshots(days=1)
        assert snaps[0]["trigger"] == "manual"


class TestMaybeSnapshot:
    """Test throttled snapshot logic."""

    def test_first_snapshot_always_succeeds(self, recorder):
        result = recorder.maybe_take_snapshot(trigger="trade_resolved")
        assert result is not None

    def test_second_snapshot_within_hour_returns_none(self, recorder):
        recorder.take_snapshot(trigger="first")
        result = recorder.maybe_take_snapshot(trigger="second")
        assert result is None

    def test_snapshot_after_interval_succeeds(self, recorder, tmp_db):
        """Manually insert an old snapshot, then verify new one is allowed."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                "VALUES (?, ?, ?)",
                (old_ts, json.dumps({"timestamp": old_ts}), "old")
            )
        result = recorder.maybe_take_snapshot()
        assert result is not None


class TestGetSnapshots:
    """Test snapshot retrieval."""

    def test_get_snapshots_empty(self, recorder):
        snaps = recorder.get_snapshots(days=7)
        assert snaps == []

    def test_get_snapshots_returns_newest_first(self, recorder):
        recorder.take_snapshot(trigger="first")
        recorder.take_snapshot(trigger="second")
        snaps = recorder.get_snapshots(days=1)
        assert len(snaps) == 2
        assert snaps[0]["trigger"] == "second"

    def test_get_snapshots_respects_days_limit(self, recorder, tmp_db):
        """Old snapshots outside the window are excluded."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                "VALUES (?, ?, ?)",
                (old_ts, json.dumps({"timestamp": old_ts, "trigger": "ancient"}), "ancient")
            )
        recorder.take_snapshot(trigger="recent")
        snaps = recorder.get_snapshots(days=7)
        assert len(snaps) == 1
        assert snaps[0]["trigger"] == "recent"


class TestComputeTrends:
    """Test trend computation."""

    def test_insufficient_data_returns_status(self, recorder):
        recorder.take_snapshot()
        trends = recorder.compute_trends(days=14)
        assert trends["status"] == "insufficient_data"

    def test_trends_with_enough_data(self, recorder, tmp_db):
        """Insert 6 snapshots with varying weights to get a trend."""
        base_ts = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(6):
            ts = (base_ts + timedelta(days=i)).isoformat()
            weight = 0.3 + (i * 0.08)  # 0.30 → 0.70
            snap = {
                "timestamp": ts,
                "trigger": "test",
                "narrative_weights": {
                    "directional_bias": {"weight": weight, "total": 20},
                },
                "bayesian": {"win_rate_mean": 0.5 + (i * 0.02)},
                "loss_types": {
                    "total_losses": 10,
                    "type1_wrong_narrative": 5 - i,
                    "type2_execution_failure": 5 + i,
                    "ambiguous": 0,
                },
                "setup_profiles": {"total": 50 + i, "wins": 30 + i,
                                   "losses": 20, "win_rate": (30 + i) / (50 + i)},
            }
            with sqlite3.connect(tmp_db) as conn:
                conn.execute(
                    "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                    "VALUES (?, ?, ?)",
                    (ts, json.dumps(snap), "test")
                )

        trends = recorder.compute_trends(days=14)
        assert "narrative_directional_bias" in trends
        assert trends["narrative_directional_bias"]["direction"] == "improving"

    def test_stable_trends(self, recorder, tmp_db):
        """Constant weights should produce 'stable' direction."""
        base_ts = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(6):
            ts = (base_ts + timedelta(days=i)).isoformat()
            snap = {
                "timestamp": ts, "trigger": "test",
                "narrative_weights": {
                    "directional_bias": {"weight": 0.55, "total": 20},
                },
                "bayesian": {}, "loss_types": {"total_losses": 0,
                    "type1_wrong_narrative": 0, "type2_execution_failure": 0,
                    "ambiguous": 0},
                "setup_profiles": {},
            }
            with sqlite3.connect(tmp_db) as conn:
                conn.execute(
                    "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                    "VALUES (?, ?, ?)",
                    (ts, json.dumps(snap), "test")
                )
        trends = recorder.compute_trends(days=14)
        assert trends["narrative_directional_bias"]["direction"] == "stable"


class TestBuildPromptContext:
    """Test prompt context generation."""

    def test_empty_returns_empty_string(self, recorder):
        ctx = recorder.build_prompt_context(days=14)
        assert ctx == ""

    def test_with_data_returns_string(self, recorder, tmp_db):
        """Insert enough snapshots to generate a trend context."""
        base_ts = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(6):
            ts = (base_ts + timedelta(days=i)).isoformat()
            snap = {
                "timestamp": ts, "trigger": "test",
                "narrative_weights": {
                    "directional_bias": {"weight": 0.3 + (i * 0.1), "total": 20},
                    "p3_phase": {"weight": 0.4, "total": 20},
                },
                "bayesian": {"win_rate_mean": 0.55},
                "loss_types": {"total_losses": 10,
                    "type1_wrong_narrative": 5,
                    "type2_execution_failure": 5, "ambiguous": 0},
                "setup_profiles": {"total": 50, "wins": 30,
                                   "losses": 20, "win_rate": 0.6},
            }
            with sqlite3.connect(tmp_db) as conn:
                conn.execute(
                    "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                    "VALUES (?, ?, ?)",
                    (ts, json.dumps(snap), "test")
                )
        ctx = recorder.build_prompt_context(days=14)
        assert "SYSTEM LEARNING STATUS" in ctx
        assert "Directional bias" in ctx


class TestBackfillFromTrades:
    """Test historical snapshot backfilling."""

    def test_backfill_empty_db(self, recorder, tmp_db):
        count = recorder.backfill_from_trades(db_path=tmp_db)
        assert count == 0

    def test_backfill_creates_daily_snapshots(self, recorder, tmp_db):
        """Insert resolved trades across 3 days, verify 3 snapshots created."""
        with sqlite3.connect(tmp_db) as conn:
            for i in range(6):
                day_offset = i // 2  # 2 trades per day, 3 days
                day = f"2026-03-{25 + day_offset:02d}"
                outcome = "tp1" if i % 3 != 0 else "stopped_out"
                conn.execute(
                    "INSERT INTO scanner_setups "
                    "(id, created_at, resolved_at, status, direction, outcome, pnl_rr, mfe_atr) "
                    "VALUES (?, ?, ?, 'resolved', 'long', ?, ?, ?)",
                    (f"trade_{i}", f"{day}T10:00:00+00:00",
                     f"{day}T14:00:00+00:00", outcome,
                     1.5 if outcome == "tp1" else -1.0,
                     2.0 if outcome == "tp1" else 0.3)
                )

        count = recorder.backfill_from_trades(db_path=tmp_db)
        assert count == 3  # 3 distinct days

    def test_backfill_bayesian_evolves(self, recorder, tmp_db):
        """Verify that backfilled snapshots show evolving Bayesian beliefs."""
        with sqlite3.connect(tmp_db) as conn:
            # 5 wins then 5 losses
            for i in range(10):
                day = f"2026-03-{20 + i:02d}"
                outcome = "tp1" if i < 5 else "stopped_out"
                conn.execute(
                    "INSERT INTO scanner_setups "
                    "(id, created_at, resolved_at, status, direction, outcome, pnl_rr, mfe_atr) "
                    "VALUES (?, ?, ?, 'resolved', 'long', ?, ?, ?)",
                    (f"t_{i}", f"{day}T10:00:00", f"{day}T14:00:00",
                     outcome, 1.0 if outcome == "tp1" else -1.0,
                     2.0 if outcome == "tp1" else 0.3)
                )

        recorder.backfill_from_trades(db_path=tmp_db)
        snaps = recorder.get_snapshots(days=30)
        assert len(snaps) == 10

        # First snapshot (earliest day) should show higher win rate than last
        earliest = snaps[-1]  # oldest
        latest = snaps[0]     # newest
        assert earliest["bayesian"]["win_rate_mean"] > latest["bayesian"]["win_rate_mean"]

    def test_backfill_loss_type_classification(self, recorder, tmp_db):
        """Verify MFE-based loss type counting in backfill."""
        with sqlite3.connect(tmp_db) as conn:
            # Type 1 loss: low MFE
            conn.execute(
                "INSERT INTO scanner_setups "
                "(id, created_at, resolved_at, status, direction, outcome, pnl_rr, mfe_atr) "
                "VALUES ('t1', '2026-03-25T10:00:00', '2026-03-25T14:00:00', "
                "'resolved', 'long', 'stopped_out', -1.0, 0.2)")
            # Type 2 loss: high MFE
            conn.execute(
                "INSERT INTO scanner_setups "
                "(id, created_at, resolved_at, status, direction, outcome, pnl_rr, mfe_atr) "
                "VALUES ('t2', '2026-03-25T15:00:00', '2026-03-25T18:00:00', "
                "'resolved', 'long', 'stopped_out', -1.0, 2.5)")

        recorder.backfill_from_trades(db_path=tmp_db)
        snaps = recorder.get_snapshots(days=30)
        assert len(snaps) == 1
        lt = snaps[0]["loss_types"]
        assert lt["type1_wrong_narrative"] == 1
        assert lt["type2_execution_failure"] == 1


class TestWeeklyReport:
    """Test weekly report generation."""

    def test_no_data_report(self, recorder):
        report = recorder.generate_weekly_report()
        assert report["status"] == "no_data"

    def test_report_with_data(self, recorder, tmp_db):
        """Insert snapshots and verify report structure."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            ts = (now - timedelta(days=i)).isoformat()
            snap = {
                "timestamp": ts, "trigger": "test",
                "narrative_weights": {
                    "directional_bias": {"weight": 0.6 + (i * 0.02), "total": 20},
                },
                "bayesian": {"win_rate_mean": 0.55, "total_trades": 30 + i,
                             "total_wins": 16, "cumulative_pnl": 5.0,
                             "max_drawdown": 2.0, "win_rate_lower_95": 0.4, "win_rate_upper_95": 0.7},
                "bandit": {"total_trades": 30, "num_arms": 3,
                          "best_arm_id": "arm_2", "best_arm_win_rate": 0.6,
                          "is_active": False},
                "setup_profiles": {"total": 50, "wins": 30, "losses": 20,
                                   "win_rate": 0.6, "avg_mfe": 1.5},
                "loss_types": {"total_losses": 20, "type1_wrong_narrative": 8,
                              "type2_execution_failure": 10, "ambiguous": 2},
                "thesis_accuracy": {"prediction_accuracy": 0.65,
                                    "predictions_scored": 40,
                                    "thesis_stability": 0.8, "revisions": 5},
                "learned_rules_count": 3,
            }
            with sqlite3.connect(tmp_db) as conn:
                conn.execute(
                    "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                    "VALUES (?, ?, ?)",
                    (ts, json.dumps(snap), "test")
                )

        report = recorder.generate_weekly_report()
        assert "components" in report
        assert "bayesian" in report["components"]
        assert "narrative_weights" in report["components"]
        assert "loss_types" in report["components"]
        assert "setup_profiles" in report["components"]
        assert "bandit" in report["components"]
        assert "overall_interpretation" in report

    def test_report_loss_type_interpretation(self, recorder, tmp_db):
        """Verify correct interpretation based on loss type ratio."""
        ts = datetime.now(timezone.utc).isoformat()
        snap = {
            "timestamp": ts, "trigger": "test",
            "narrative_weights": {},
            "bayesian": {"win_rate_mean": 0.5},
            "bandit": {},
            "setup_profiles": {"total": 30, "wins": 15, "losses": 15, "win_rate": 0.5},
            "loss_types": {"total_losses": 15, "type1_wrong_narrative": 3,
                          "type2_execution_failure": 10, "ambiguous": 2},
            "thesis_accuracy": {},
            "learned_rules_count": 0,
        }
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO system_snapshots (timestamp, snapshot_json, trigger) "
                "VALUES (?, ?, ?)",
                (ts, json.dumps(snap), "test")
            )
        report = recorder.generate_weekly_report()
        interp = report["components"]["loss_types"]["interpretation"]
        assert "execution" in interp.lower()
