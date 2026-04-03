"""Tests for prediction engine — cold start + inference."""
import json
import os

import pytest
from unittest.mock import MagicMock, patch
from ml.prediction import predict, _build_calibration, _apply_wfo_filter
from ml.database import TradeLogger
from ml.wfo import WFOReport


class TestPredict:
    def test_cold_start_returns_correct_status(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["model_status"] == "cold_start"

    def test_cold_start_confidence_zero(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["confidence"] == 0

    def test_cold_start_classification_empty(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["classification"] == {}

    def test_cold_start_training_samples(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["training_samples"] == 0

    def test_returns_valid_schema_keys(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        # Must include both original and extended ML pipeline keys
        required_keys = {
            "confidence", "classification", "suggested_sl", "suggested_tp1",
            "suggested_tp2", "model_status", "training_samples", "feature_importances",
            "grade", "blended_confidence", "conservative_sl",
            "volatility_regime", "bayesian_win_rate", "session", "reasoning",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_insufficient_data_status(self, tmp_db, test_config, sample_analysis, sample_candles):
        """With some but not enough trades, status should be insufficient_data."""
        import json
        db = TradeLogger(db_path=tmp_db, config=test_config)
        # Log 5 trades (below 30 minimum)
        for i in range(5):
            sid = f"s-{i}"
            db.log_setup(sid, {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650, "sl_price": 2643, "tp1_price": 2680,
            })
            db.log_outcome(sid, "tp1_hit", 30, 5, 25)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["model_status"] == "insufficient_data"
        assert result["training_samples"] == 5

    def test_no_suggested_levels_in_cold_start(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["suggested_sl"] is None
        assert result["suggested_tp1"] is None
        assert result["suggested_tp2"] is None

    def test_feature_importances_empty_in_cold_start(self, tmp_db, test_config, sample_analysis, sample_candles):
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["feature_importances"] == {}

    def test_consensus_fields_none_in_cold_start(self, tmp_db, test_config, sample_analysis, sample_candles):
        """Consensus fields should be None when no data (cold start)."""
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["grade"] is None
        assert result["blended_confidence"] is None
        assert result["bayesian_win_rate"] is None
        assert result["volatility_regime"] is None
        assert result["reasoning"] is None

    def test_consensus_fields_none_in_insufficient_data(self, tmp_db, test_config, sample_analysis, sample_candles):
        """Consensus fields None when insufficient data (no model confidence)."""
        import json
        db = TradeLogger(db_path=tmp_db, config=test_config)
        for i in range(5):
            sid = f"s-{i}"
            db.log_setup(sid, {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650, "sl_price": 2643, "tp1_price": 2680,
            })
            db.log_outcome(sid, "tp1_hit", 30, 5, 25)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        # Insufficient data → confidence=0 → consensus fields stay None
        assert result["grade"] is None
        assert result["blended_confidence"] is None


class TestBuildCalibration:
    """Step 9: Tests for _build_calibration and take_trade logic."""

    def test_take_trade_false_when_no_trade_high(self):
        """take_trade=False when no_trade class probability > 0.4."""
        result = {
            "confidence": 0.6,
            "classification": {"no_trade": 0.5, "tp1_hit": 0.3, "stopped_out": 0.2},
        }
        cal = _build_calibration(result, None, None, {})
        assert cal["take_trade"] is False

    def test_take_trade_true_when_no_trade_low(self):
        """take_trade=True when no_trade probability <= 0.4."""
        result = {
            "confidence": 0.7,
            "classification": {"no_trade": 0.1, "tp1_hit": 0.6, "stopped_out": 0.3},
        }
        cal = _build_calibration(result, None, None, {})
        assert cal["take_trade"] is True

    def test_take_trade_none_when_no_classification(self):
        """take_trade=None when no classification available."""
        result = {"confidence": 0, "classification": {}}
        cal = _build_calibration(result, None, None, {})
        assert cal["take_trade"] is None

    def test_dataset_backing_from_dm(self):
        """dataset_backing populated from dataset_manager.get_stats()."""
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 100, "wfo_count": 80, "live_count": 20,
            "regime_distribution": {"trending_up": 40, "ranging": 30, "low_volatility": 30},
            "outcome_distribution": {},
        }
        result = {"confidence": 0.7, "classification": {"tp1_hit": 0.7, "stopped_out": 0.3}}
        cal = _build_calibration(result, dm, None, {})
        assert cal["dataset_backing"]["total"] == 100
        assert cal["dataset_backing"]["wfo_count"] == 80
        assert cal["dataset_backing"]["live_count"] == 20

    def test_data_maturity_early(self):
        """data_maturity='early' when live_count < 10."""
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 50, "wfo_count": 45, "live_count": 5,
            "regime_distribution": {"trending_up": 25, "ranging": 25},
            "outcome_distribution": {},
        }
        result = {"confidence": 0.7, "classification": {"tp1_hit": 0.7}}
        cal = _build_calibration(result, dm, None, {})
        assert cal["data_maturity"] == "early"

    def test_data_maturity_mature(self):
        """data_maturity='mature' when live_count >= 50."""
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 200, "wfo_count": 100, "live_count": 100,
            "regime_distribution": {"trending_up": 50, "ranging": 50,
                                     "low_volatility": 50, "high_volatility": 50},
            "outcome_distribution": {},
        }
        result = {"confidence": 0.7, "classification": {"tp1_hit": 0.7}}
        cal = _build_calibration(result, dm, None, {})
        assert cal["data_maturity"] == "mature"

    def test_defensive_mode_low_regime_coverage(self):
        """defensive_mode=True when a regime has < 10% representation."""
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 100, "wfo_count": 90, "live_count": 10,
            "regime_distribution": {"trending_up": 90, "high_volatility": 5, "ranging": 5},
            "outcome_distribution": {},
        }
        result = {"confidence": 0.7, "classification": {"tp1_hit": 0.7}}
        cal = _build_calibration(result, dm, None, {})
        assert cal["defensive_mode"] is True
        assert cal["regime_coverage"] == "low"
        assert cal["regime_adjustment"] == 0.7

    def test_no_dm_returns_none_fields(self):
        """Without dataset_manager, dataset fields stay None."""
        result = {"confidence": 0.5, "classification": {"tp1_hit": 0.5}}
        cal = _build_calibration(result, None, None, {})
        assert cal["dataset_backing"] is None
        assert cal["data_maturity"] is None
        assert cal["defensive_mode"] is None

    def test_cold_start_includes_calibration_keys(self, tmp_db, test_config, sample_analysis, sample_candles):
        """Cold start predictions include extended calibration keys."""
        db = TradeLogger(db_path=tmp_db, config=test_config)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        # These should exist (as None)
        assert "take_trade" in result
        assert "confidence_raw" in result
        assert "data_maturity" in result


class TestEffectiveCount:
    """predict() should use effective_count = max(db_count, dataset_count)."""

    def test_dataset_trades_count_towards_readiness(self, tmp_db, test_config, sample_analysis, sample_candles):
        """When dataset_manager has 50 trades but db has 0, status should NOT be cold_start."""
        db = TradeLogger(db_path=tmp_db, config=test_config)
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 50, "wfo_count": 50, "live_count": 0,
            "regime_distribution": {"trending_up": 25, "ranging": 25},
            "outcome_distribution": {"tp1_hit": 30, "stopped_out": 20},
        }
        result = predict(sample_analysis, sample_candles, "1h", db=db,
                         config=test_config, dataset_manager=dm)
        # Should NOT be cold_start — dataset has 50 trades
        assert result["model_status"] != "cold_start"
        assert result["training_samples"] >= 50

    def test_zero_everywhere_is_cold_start(self, tmp_db, test_config, sample_analysis, sample_candles):
        """When both db and dataset_manager have 0 trades, status is cold_start."""
        db = TradeLogger(db_path=tmp_db, config=test_config)
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 0, "wfo_count": 0, "live_count": 0,
            "regime_distribution": {}, "outcome_distribution": {},
        }
        result = predict(sample_analysis, sample_candles, "1h", db=db,
                         config=test_config, dataset_manager=dm)
        assert result["model_status"] == "cold_start"
        assert result["training_samples"] == 0

    def test_db_trades_still_work_without_dm(self, tmp_db, test_config, sample_analysis, sample_candles):
        """Without dataset_manager, falls back to db count only (backwards compat)."""
        import json
        db = TradeLogger(db_path=tmp_db, config=test_config)
        for i in range(5):
            sid = f"s-{i}"
            db.log_setup(sid, {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650, "sl_price": 2643, "tp1_price": 2680,
            })
            db.log_outcome(sid, "tp1_hit", 30, 5, 25)
        result = predict(sample_analysis, sample_candles, "1h", db=db, config=test_config)
        assert result["training_samples"] == 5
        assert result["model_status"] == "insufficient_data"

    def test_effective_count_uses_max(self, tmp_db, test_config, sample_analysis, sample_candles):
        """effective_count = max(db_count, dm_total) so whichever is higher wins."""
        import json
        db = TradeLogger(db_path=tmp_db, config=test_config)
        # 5 trades in DB
        for i in range(5):
            sid = f"s-{i}"
            db.log_setup(sid, {"ob_count": 1}, json.dumps(sample_analysis), "[]", {
                "timeframe": "1h", "bias": "bullish", "direction": "long",
                "entry_price": 2650, "sl_price": 2643, "tp1_price": 2680,
            })
            db.log_outcome(sid, "tp1_hit", 30, 5, 25)
        # 100 trades in dataset
        dm = MagicMock()
        dm.get_stats.return_value = {
            "total": 100, "wfo_count": 95, "live_count": 5,
            "regime_distribution": {"trending_up": 50, "ranging": 50},
            "outcome_distribution": {},
        }
        result = predict(sample_analysis, sample_candles, "1h", db=db,
                         config=test_config, dataset_manager=dm)
        # Should use 100 (dm total), not 5 (db count)
        assert result["training_samples"] >= 100


# ═══════════════════════════════════════════════════════════════════════
# WFO Filter Tests
# ═══════════════════════════════════════════════════════════════════════


class TestApplyWfoFilter:
    """Test _apply_wfo_filter gating logic."""

    @pytest.fixture
    def bull_analysis(self):
        """Bull FVG + structure analysis."""
        return {
            "entry": {"direction": "long"},
            "orderBlocks": [],
            "fvgs": [{"type": "bullish", "high": 102, "low": 101}],
            "confluences": {"structureAlignment": True},
            "killzone": "",
        }

    @pytest.fixture
    def wfo_report_path(self, tmp_path):
        """Path to temp WFO report file."""
        return str(tmp_path / "wfo_report.json")

    def _save_report(self, path, setup_type_stats):
        """Helper to save a WFO report with given stats."""
        report = WFOReport(
            total_oos_trades=100, oos_win_rate=0.45, oos_avg_rr=2.0,
            oos_profit_factor=1.3, oos_sharpe=0.8, oos_max_drawdown=3.5,
            regime_stability=0.85, recommended_sl_atr=1.5,
            recommended_tp_atr=[1.0, 2.0, 3.5], grade="B", folds=[],
            fold_count=5, skipped_folds=0, setup_type_breakdown={},
            timestamp="2026-03-13T00:00:00Z",
            setup_type_stats=setup_type_stats,
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(report.to_dict()))

    def test_validated_passes_confidence(self, bull_analysis, wfo_report_path):
        """Profitable setup type → confidence unchanged, status=validated."""
        stats = {
            "bull_fvg_structure": {"wins": 10, "total": 20, "win_rate": 0.50},
        }
        self._save_report(wfo_report_path, stats)
        cfg = {"wfo_report_path": wfo_report_path}
        result = {"confidence": 0.75, "reasoning": []}

        result = _apply_wfo_filter(result, bull_analysis, [], "1h", cfg)
        assert result["confidence"] == 0.75  # unchanged
        assert result["wfo_filter"]["status"] == "validated"
        assert result["wfo_filter"]["action"] == "pass"

    def test_unprofitable_halves_confidence(self, bull_analysis, wfo_report_path):
        """Unprofitable setup type → confidence × 0.5, status=unprofitable."""
        stats = {
            "bull_fvg_structure": {"wins": 3, "total": 25, "win_rate": 0.12},
        }
        self._save_report(wfo_report_path, stats)
        cfg = {"wfo_report_path": wfo_report_path}
        result = {"confidence": 0.80, "reasoning": []}

        result = _apply_wfo_filter(result, bull_analysis, [], "1h", cfg)
        assert result["confidence"] == pytest.approx(0.40, abs=0.01)
        assert result["wfo_filter"]["status"] == "unprofitable"
        assert result["wfo_filter"]["action"] == "downgrade"
        assert any("confidence halved" in r for r in result["reasoning"])

    def test_unvalidated_reduces_confidence_20pct(self, bull_analysis, wfo_report_path):
        """Unknown setup type (not in WFO stats) → confidence × 0.8."""
        # Stats exist but don't include bull_fvg_structure
        stats = {
            "bear_ob": {"wins": 5, "total": 10, "win_rate": 0.50},
        }
        self._save_report(wfo_report_path, stats)
        cfg = {"wfo_report_path": wfo_report_path}
        result = {"confidence": 1.0, "reasoning": []}

        result = _apply_wfo_filter(result, bull_analysis, [], "1h", cfg)
        assert result["confidence"] == pytest.approx(0.80, abs=0.01)
        assert result["wfo_filter"]["status"] == "unvalidated"
        assert result["wfo_filter"]["action"] == "caution"

    def test_no_report_returns_no_report_status(self, bull_analysis, tmp_path):
        """No saved WFO report → wfo_filter.status = 'no_report'."""
        cfg = {"wfo_report_path": str(tmp_path / "nonexistent.json")}
        result = {"confidence": 0.75, "reasoning": []}

        result = _apply_wfo_filter(result, bull_analysis, [], "1h", cfg)
        assert result["confidence"] == 0.75  # unchanged
        assert result["wfo_filter"]["status"] == "no_report"

    def test_fuzzy_match_partial_overlap(self, wfo_report_path):
        """Partial tag match → status=partial_match when overlap sufficient."""
        # Analysis has: fvg + structure + ob → bull_fvg_ob_structure
        analysis = {
            "entry": {"direction": "long"},
            "orderBlocks": [{"type": "bullish", "high": 100, "low": 99}],
            "fvgs": [{"type": "bullish", "high": 102, "low": 101}],
            "confluences": {"structureAlignment": True},
        }
        # WFO only has bull_fvg_structure (no ob)
        stats = {
            "bull_fvg_structure": {"wins": 10, "total": 18, "win_rate": 0.5556},
        }
        self._save_report(wfo_report_path, stats)
        cfg = {"wfo_report_path": wfo_report_path}
        result = {"confidence": 0.80, "reasoning": []}

        result = _apply_wfo_filter(result, analysis, [], "1h", cfg)
        # bull_fvg_ob_structure has 3 tags, bull_fvg_structure has 2 tags
        # overlap = {fvg, structure} = 2, which >= len(tags) - 1 = 2
        assert result["wfo_filter"]["status"] == "partial_match"
        assert result["wfo_filter"]["matched_to"] == "bull_fvg_structure"

    def test_empty_stats_returns_no_report(self, bull_analysis, wfo_report_path):
        """WFO report exists but has empty setup_type_stats → no_report."""
        self._save_report(wfo_report_path, {})
        cfg = {"wfo_report_path": wfo_report_path}
        result = {"confidence": 0.75, "reasoning": []}

        result = _apply_wfo_filter(result, bull_analysis, [], "1h", cfg)
        assert result["wfo_filter"]["status"] == "no_report"
