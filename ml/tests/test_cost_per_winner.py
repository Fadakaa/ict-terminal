"""Tests for Priority 8: Cost-Per-Winner Optimization."""
import json
import os
import pytest
from unittest.mock import MagicMock

from ml.cost_per_winner import (
    CostPerWinnerTracker,
    MIN_SEGMENT_SAMPLES,
    CPW_WARNING_USD,
    CPW_CRITICAL_USD,
    ROI_MIN_THRESHOLD,
    FREQ_REDUCE,
    FREQ_NORMAL,
    FREQ_BOOST,
    WIN_OUTCOMES,
    DATA_WINDOW_DAYS,
)
from ml.cost_tracker import CostTracker, _compute_cost
from ml.config import make_test_config


@pytest.fixture
def tmp_tracker(tmp_path):
    """CostPerWinnerTracker with a temp model_dir."""
    cfg = make_test_config(model_dir=str(tmp_path))
    return CostPerWinnerTracker(config=cfg)


def _make_setup(timeframe="1h", killzone="london", outcome="tp1",
                pnl_rr=2.5, api_cost_usd=0.25, quality="B"):
    return {
        "timeframe": timeframe,
        "killzone": killzone,
        "setup_quality": quality,
        "outcome": outcome,
        "pnl_rr": pnl_rr,
        "api_cost_usd": api_cost_usd,
    }


# ---------------------------------------------------------------------------
# Segment key generation
# ---------------------------------------------------------------------------

class TestSegmentKey:
    def test_basic(self, tmp_tracker):
        assert tmp_tracker._segment_key("1h", "london") == "1h_london"

    def test_lowercased(self, tmp_tracker):
        assert tmp_tracker._segment_key("4H", "New_York") == "4h_new_york"

    def test_missing_values(self, tmp_tracker):
        assert tmp_tracker._segment_key(None, None) == "unknown_unknown"


# ---------------------------------------------------------------------------
# Single trade ingestion
# ---------------------------------------------------------------------------

class TestIngestTrade:
    def test_win_updates_segment(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="tp1", pnl_rr=3.0, api_cost_usd=0.20))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert seg["total_trades"] == 1
        assert seg["wins"] == 1
        assert seg["losses"] == 0
        assert seg["total_api_cost_usd"] == 0.20
        assert seg["total_pnl_rr"] == 3.0

    def test_loss_updates_segment(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="stopped_out", pnl_rr=-1.0, api_cost_usd=0.20))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert seg["total_trades"] == 1
        assert seg["wins"] == 0
        assert seg["losses"] == 1
        assert seg["total_pnl_rr"] == -1.0

    def test_skip_if_no_cost(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(api_cost_usd=None))
        assert len(tmp_tracker._stats["segments"]) == 0

    def test_skip_if_zero_cost(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(api_cost_usd=0))
        assert len(tmp_tracker._stats["segments"]) == 0

    def test_multiple_trades_accumulate(self, tmp_tracker):
        for i in range(5):
            tmp_tracker.ingest_trade(_make_setup(
                outcome="tp1" if i % 2 == 0 else "stopped_out",
                pnl_rr=2.0 if i % 2 == 0 else -1.0,
                api_cost_usd=0.20,
            ))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert seg["total_trades"] == 5
        assert seg["wins"] == 3
        assert seg["losses"] == 2
        assert abs(seg["total_api_cost_usd"] - 1.0) < 1e-6
        assert abs(seg["total_pnl_rr"] - 4.0) < 1e-4  # 3*2 + 2*(-1) = 4

    def test_global_stats_updated(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="tp2", pnl_rr=4.0, api_cost_usd=0.30))
        tmp_tracker.ingest_trade(_make_setup(timeframe="4h", outcome="stopped_out",
                                              pnl_rr=-1.0, api_cost_usd=0.15))
        g = tmp_tracker._stats["global"]
        assert g["total_trades"] == 2
        assert g["total_wins"] == 1
        assert abs(g["total_api_cost_usd"] - 0.45) < 1e-6
        assert abs(g["total_pnl_rr"] - 3.0) < 1e-4

    def test_different_segments_tracked_separately(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(timeframe="1h", killzone="london"))
        tmp_tracker.ingest_trade(_make_setup(timeframe="4h", killzone="new_york"))
        assert "1h_london" in tmp_tracker._stats["segments"]
        assert "4h_new_york" in tmp_tracker._stats["segments"]


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

class TestDerivedMetrics:
    def test_cost_per_winner(self, tmp_tracker):
        # 3 wins, $0.60 total cost → $0.20 per winner
        for _ in range(3):
            tmp_tracker.ingest_trade(_make_setup(outcome="tp1", api_cost_usd=0.10))
        for _ in range(2):
            tmp_tracker.ingest_trade(_make_setup(outcome="stopped_out", api_cost_usd=0.15))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        # total cost = 3*0.10 + 2*0.15 = 0.60, wins = 3
        assert abs(seg["cost_per_winner_usd"] - 0.20) < 1e-4

    def test_cost_per_trade(self, tmp_tracker):
        for _ in range(4):
            tmp_tracker.ingest_trade(_make_setup(api_cost_usd=0.25))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert abs(seg["cost_per_trade_usd"] - 0.25) < 1e-6

    def test_roi_per_dollar(self, tmp_tracker):
        # $0.50 spent, 5.0R earned → 10.0 R/$
        tmp_tracker.ingest_trade(_make_setup(outcome="tp1", pnl_rr=5.0, api_cost_usd=0.50))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert abs(seg["roi_per_dollar"] - 10.0) < 1e-2

    def test_win_rate(self, tmp_tracker):
        for i in range(10):
            tmp_tracker.ingest_trade(_make_setup(
                outcome="tp1" if i < 6 else "stopped_out",
                pnl_rr=2.0 if i < 6 else -1.0,
            ))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert abs(seg["win_rate"] - 0.6) < 1e-4

    def test_no_wins_cpw_is_none(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="stopped_out", pnl_rr=-1.0))
        seg = tmp_tracker._stats["segments"]["1h_london"]
        assert seg["cost_per_winner_usd"] is None

    def test_global_metrics(self, tmp_tracker):
        for _ in range(4):
            tmp_tracker.ingest_trade(_make_setup(outcome="tp1", pnl_rr=2.0, api_cost_usd=0.20))
        g = tmp_tracker._stats["global"]
        assert g["avg_cost_per_winner_usd"] is not None
        assert abs(g["avg_cost_per_winner_usd"] - 0.20) < 1e-6
        assert g["overall_roi_per_dollar"] is not None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def _fill_segment(self, tracker, n=MIN_SEGMENT_SAMPLES, cpw_target=0.50,
                      win_rate=0.5, timeframe="1h", killzone="london"):
        """Helper: create N trades with approximate target CPW and WR."""
        wins = int(n * win_rate)
        losses = n - wins
        # CPW = total_cost / wins → total_cost = CPW * wins
        total_cost = cpw_target * wins if wins > 0 else cpw_target * n
        cost_per_trade = total_cost / n

        for i in range(n):
            tracker.ingest_trade(_make_setup(
                timeframe=timeframe,
                killzone=killzone,
                outcome="tp1" if i < wins else "stopped_out",
                pnl_rr=2.5 if i < wins else -1.0,
                api_cost_usd=cost_per_trade,
            ))

    def test_insufficient_data_normal(self, tmp_tracker):
        # Only 3 trades — not enough for a recommendation
        for _ in range(3):
            tmp_tracker.ingest_trade(_make_setup())
        recs = tmp_tracker.get_recommendations()
        assert recs["1h_london"]["action"] == FREQ_NORMAL
        assert "Insufficient" in recs["1h_london"]["reason"]

    def test_critical_cpw_reduces(self, tmp_tracker):
        # CPW = $4.00 — way above critical threshold
        self._fill_segment(tmp_tracker, cpw_target=CPW_CRITICAL_USD + 1.0, win_rate=0.3)
        recs = tmp_tracker.get_recommendations()
        assert recs["1h_london"]["action"] == FREQ_REDUCE
        assert "critical" in recs["1h_london"]["reason"].lower()

    def test_low_roi_reduces(self, tmp_tracker):
        # Win rate 10%, ROI will be very negative
        self._fill_segment(tmp_tracker, cpw_target=1.0, win_rate=0.1)
        recs = tmp_tracker.get_recommendations()
        rec = recs["1h_london"]
        # Either reduce due to low ROI or critical CPW
        assert rec["action"] == FREQ_REDUCE

    def test_high_roi_boosts(self, tmp_tracker):
        # High WR, cheap scans → great ROI
        self._fill_segment(tmp_tracker, cpw_target=0.10, win_rate=0.8)
        recs = tmp_tracker.get_recommendations()
        assert recs["1h_london"]["action"] == FREQ_BOOST

    def test_normal_range(self, tmp_tracker):
        # CPW=$0.40 (below warning), ROI=1.25 (above min, below boost threshold)
        wins = int(MIN_SEGMENT_SAMPLES * 0.5)
        for i in range(MIN_SEGMENT_SAMPLES):
            tmp_tracker.ingest_trade(_make_setup(
                outcome="tp1" if i < wins else "stopped_out",
                pnl_rr=1.5 if i < wins else -1.0,
                api_cost_usd=0.20,
            ))
        recs = tmp_tracker.get_recommendations()
        assert recs["1h_london"]["action"] == FREQ_NORMAL

    def test_warning_cpw_stays_normal(self, tmp_tracker):
        # CPW between warning and critical
        self._fill_segment(tmp_tracker, cpw_target=CPW_WARNING_USD + 0.5, win_rate=0.5)
        recs = tmp_tracker.get_recommendations()
        rec = recs["1h_london"]
        # Should be normal (warning, not reduce) unless ROI also bad
        assert rec["action"] in (FREQ_NORMAL, FREQ_REDUCE)


# ---------------------------------------------------------------------------
# should_reduce_scan / should_boost_scan
# ---------------------------------------------------------------------------

class TestScanDecisions:
    def test_reduce_when_recommended(self, tmp_tracker):
        tmp_tracker._stats["recommendations"] = {
            "1h_london": {"action": FREQ_REDUCE, "reason": "test"}
        }
        assert tmp_tracker.should_reduce_scan("1h", "london") is True
        assert tmp_tracker.should_boost_scan("1h", "london") is False

    def test_boost_when_recommended(self, tmp_tracker):
        tmp_tracker._stats["recommendations"] = {
            "1h_london": {"action": FREQ_BOOST, "reason": "test"}
        }
        assert tmp_tracker.should_boost_scan("1h", "london") is True
        assert tmp_tracker.should_reduce_scan("1h", "london") is False

    def test_no_recommendation_defaults_false(self, tmp_tracker):
        assert tmp_tracker.should_reduce_scan("1h", "london") is False
        assert tmp_tracker.should_boost_scan("1h", "london") is False


# ---------------------------------------------------------------------------
# Recompute from DB
# ---------------------------------------------------------------------------

class TestRecomputeFromDB:
    def test_recompute_populates_segments(self, tmp_tracker):
        mock_db = MagicMock()
        mock_db.get_resolved_with_costs.return_value = [
            {"timeframe": "1h", "killzone": "london", "outcome": "tp1",
             "pnl_rr": 2.5, "api_cost_usd": 0.25, "resolved_at": "2026-03-28T10:00:00",
             "setup_quality": "B"},
            {"timeframe": "1h", "killzone": "london", "outcome": "stopped_out",
             "pnl_rr": -1.0, "api_cost_usd": 0.25, "resolved_at": "2026-03-28T12:00:00",
             "setup_quality": "B"},
        ]
        result = tmp_tracker.recompute_from_db(mock_db)
        assert "1h_london" in result["segments"]
        assert result["segments"]["1h_london"]["total_trades"] == 2
        assert result["global"]["total_trades"] == 2

    def test_recompute_excludes_old_data(self, tmp_tracker):
        mock_db = MagicMock()
        mock_db.get_resolved_with_costs.return_value = [
            # Old setup — outside DATA_WINDOW_DAYS
            {"timeframe": "1h", "killzone": "london", "outcome": "tp1",
             "pnl_rr": 2.5, "api_cost_usd": 0.25, "resolved_at": "2025-01-01T10:00:00",
             "setup_quality": "B"},
            # Recent setup
            {"timeframe": "1h", "killzone": "london", "outcome": "tp1",
             "pnl_rr": 2.5, "api_cost_usd": 0.25, "resolved_at": "2026-03-28T10:00:00",
             "setup_quality": "B"},
        ]
        result = tmp_tracker.recompute_from_db(mock_db)
        assert result["segments"]["1h_london"]["total_trades"] == 1

    def test_recompute_generates_recommendations(self, tmp_tracker):
        mock_db = MagicMock()
        setups = []
        for i in range(MIN_SEGMENT_SAMPLES):
            setups.append({
                "timeframe": "1h", "killzone": "london",
                "outcome": "tp1" if i % 2 == 0 else "stopped_out",
                "pnl_rr": 2.0 if i % 2 == 0 else -1.0,
                "api_cost_usd": 0.20,
                "resolved_at": "2026-03-28T10:00:00",
                "setup_quality": "B",
            })
        mock_db.get_resolved_with_costs.return_value = setups
        result = tmp_tracker.recompute_from_db(mock_db)
        assert "1h_london" in result["recommendations"]


# ---------------------------------------------------------------------------
# Segment ranking
# ---------------------------------------------------------------------------

class TestSegmentRanking:
    def test_ranking_sorted_by_roi(self, tmp_tracker):
        # Good segment
        for _ in range(MIN_SEGMENT_SAMPLES):
            tmp_tracker.ingest_trade(_make_setup(
                timeframe="15min", killzone="london",
                outcome="tp1", pnl_rr=3.0, api_cost_usd=0.10))
        # Bad segment
        for _ in range(MIN_SEGMENT_SAMPLES):
            tmp_tracker.ingest_trade(_make_setup(
                timeframe="4h", killzone="asian",
                outcome="stopped_out", pnl_rr=-1.0, api_cost_usd=0.50))

        ranking = tmp_tracker.get_segment_ranking()
        assert len(ranking) == 2
        assert ranking[0]["segment"] == "15min_london"
        assert ranking[1]["segment"] == "4h_asian"

    def test_ranking_excludes_insufficient_data(self, tmp_tracker):
        for _ in range(3):  # Less than MIN_SEGMENT_SAMPLES
            tmp_tracker.ingest_trade(_make_setup())
        ranking = tmp_tracker.get_segment_ranking()
        assert len(ranking) == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        cfg = make_test_config(model_dir=str(tmp_path))
        tracker1 = CostPerWinnerTracker(config=cfg)
        tracker1.ingest_trade(_make_setup(outcome="tp1", pnl_rr=3.0, api_cost_usd=0.25))

        # New instance should load saved data
        tracker2 = CostPerWinnerTracker(config=cfg)
        assert tracker2._stats["global"]["total_trades"] == 1
        assert "1h_london" in tracker2._stats["segments"]


# ---------------------------------------------------------------------------
# CostTracker setup_id tagging
# ---------------------------------------------------------------------------

class TestCostTrackerSetupId:
    def test_log_call_with_setup_id(self, tmp_path):
        cfg = make_test_config(model_dir=str(tmp_path))
        ct = CostTracker(config=cfg)
        ct._path = str(tmp_path / "cost_log.json")

        ct.log_call("sonnet", 5000, 500, "analysis", setup_id="abc123")
        ct.log_call("haiku", 1500, 200, "screen", setup_id="abc123")
        ct.log_call("sonnet", 5000, 500, "analysis")  # No setup_id

        result = ct.get_setup_cost("abc123")
        assert result["setup_id"] == "abc123"
        assert result["call_count"] == 2
        assert result["total_usd"] > 0
        assert "analysis" in result["by_purpose"]
        assert "screen" in result["by_purpose"]

    def test_get_setup_cost_empty(self, tmp_path):
        cfg = make_test_config(model_dir=str(tmp_path))
        ct = CostTracker(config=cfg)
        ct._path = str(tmp_path / "cost_log.json")

        result = ct.get_setup_cost("nonexistent")
        assert result["call_count"] == 0
        assert result["total_usd"] == 0


# ---------------------------------------------------------------------------
# WIN_OUTCOMES consistency
# ---------------------------------------------------------------------------

class TestWinOutcomes:
    def test_tp1_is_win(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="tp1"))
        assert tmp_tracker._stats["segments"]["1h_london"]["wins"] == 1

    def test_tp2_is_win(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="tp2"))
        assert tmp_tracker._stats["segments"]["1h_london"]["wins"] == 1

    def test_tp3_is_win(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="tp3"))
        assert tmp_tracker._stats["segments"]["1h_london"]["wins"] == 1

    def test_runner_is_win(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="runner"))
        assert tmp_tracker._stats["segments"]["1h_london"]["wins"] == 1

    def test_stopped_out_is_loss(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="stopped_out"))
        assert tmp_tracker._stats["segments"]["1h_london"]["losses"] == 1

    def test_expired_is_loss(self, tmp_tracker):
        tmp_tracker.ingest_trade(_make_setup(outcome="expired"))
        assert tmp_tracker._stats["segments"]["1h_london"]["losses"] == 1
