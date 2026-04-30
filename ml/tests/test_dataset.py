"""Tests for TrainingDatasetManager, RegimeBalancer, negative examples, and Bayesian safeguards."""
import os
import math
import pytest
import pandas as pd

from ml.dataset import (
    TrainingDatasetManager, RegimeBalancer,
    generate_negative_examples, PriorValidator, DriftAlarm,
)
from ml.config import make_test_config


# ─── TrainingDatasetManager ─────────────────────────────────────────


class TestTrainingDatasetManager:
    @pytest.fixture
    def dm(self, tmp_path):
        cfg = make_test_config(
            dataset_parquet_path=str(tmp_path / "test_dataset.parquet"),
            live_weight_multiplier=5.0,
            wfo_weight_decay_rate=200,
        )
        return TrainingDatasetManager(config=cfg)

    @pytest.fixture
    def sample_wfo_trades(self):
        return [
            {"ob_count": 2, "fvg_count": 1, "outcome": "tp1_hit",
             "direction": "long", "regime": "trending_up",
             "max_favorable_atr": 2.0, "max_drawdown_atr": 0.5},
            {"ob_count": 1, "fvg_count": 0, "outcome": "stopped_out",
             "direction": "short", "regime": "ranging",
             "max_favorable_atr": 0.3, "max_drawdown_atr": 1.5},
            {"ob_count": 3, "fvg_count": 2, "outcome": "tp2_hit",
             "direction": "long", "regime": "trending_up",
             "max_favorable_atr": 3.0, "max_drawdown_atr": 0.8},
        ]

    def test_dataset_ingests_calendar_features(self, dm):
        """Calendar columns round-trip through ingest_live_trade → parquet
        → get_blended_dataset (Task 13)."""
        from ml.feature_schema import FEATURE_COLUMNS
        baseline = {c: 0.0 for c in FEATURE_COLUMNS}
        baseline["event_is_fomc"] = 1
        baseline["calendar_proximity_imminent"] = 1
        baseline["calendar_proximity_clear"] = 0
        baseline["mins_to_next_high_impact"] = 15
        baseline["news_density_24h"] = 3

        dm.ingest_live_trade(
            features=baseline, outcome="tp1_hit",
            mfe=3.0, mae=0.5, pnl=2.5,
        )
        df = dm.get_blended_dataset(live_only=True)
        assert "event_is_fomc" in df.columns
        assert "calendar_proximity_imminent" in df.columns
        assert df.iloc[-1]["event_is_fomc"] == 1
        assert df.iloc[-1]["calendar_proximity_imminent"] == 1
        assert df.iloc[-1]["mins_to_next_high_impact"] == 15

    def test_ingest_wfo_trades(self, dm, sample_wfo_trades):
        count = dm.ingest_wfo_trades(sample_wfo_trades)
        assert count == 3
        stats = dm.get_stats()
        assert stats["wfo_count"] == 3
        assert stats["live_count"] == 0
        assert stats["total"] == 3

    def test_ingest_wfo_clears_old(self, dm, sample_wfo_trades):
        dm.ingest_wfo_trades(sample_wfo_trades)
        dm.ingest_wfo_trades(sample_wfo_trades[:1])
        stats = dm.get_stats()
        assert stats["wfo_count"] == 1
        assert stats["total"] == 1

    def test_ingest_live_trade(self, dm):
        dm.ingest_live_trade(
            features={"ob_count": 2, "fvg_count": 1},
            outcome="tp1_hit", mfe=3.0, mae=0.5, pnl=2.5,
        )
        stats = dm.get_stats()
        assert stats["live_count"] == 1

    def test_ingest_wfo_preserves_live(self, dm, sample_wfo_trades):
        dm.ingest_live_trade(
            features={"ob_count": 2, "fvg_count": 1},
            outcome="tp1_hit", mfe=3.0, mae=0.5, pnl=2.5,
        )
        dm.ingest_wfo_trades(sample_wfo_trades)
        stats = dm.get_stats()
        assert stats["live_count"] == 1
        assert stats["wfo_count"] == 3
        assert stats["total"] == 4

    def test_get_blended_dataset_weights(self, dm, sample_wfo_trades):
        dm.ingest_wfo_trades(sample_wfo_trades)
        # Provide enough features for "full" quality (>= RICH_FEATURE_THRESHOLD)
        full_features = {
            "ob_count": 2, "ob_bullish_count": 1, "ob_bearish_count": 1,
            "ob_strongest_strength": 3, "ob_nearest_distance_atr": 0.5,
            "ob_avg_size_atr": 0.3, "ob_alignment": 1,
            "fvg_count": 1, "fvg_unfilled_count": 1, "fvg_nearest_distance_atr": 0.2,
            "fvg_avg_size_atr": 0.1, "fvg_alignment": 1,
            "liq_buyside_count": 1, "liq_sellside_count": 1,
            "liq_nearest_target_distance_atr": 1.0, "liq_nearest_threat_distance_atr": 2.0,
            "risk_reward_tp1": 3.0, "risk_reward_tp2": 5.0,
            "sl_distance_atr": 1.5, "tp1_distance_atr": 4.5,
            "entry_direction": 1, "bias_direction_match": 1,
            "num_confluences": 3, "has_ob_fvg_overlap": 1,
        }
        dm.ingest_live_trade(
            features=full_features,
            outcome="tp1_hit", mfe=3.0, mae=0.5, pnl=2.5,
        )
        df = dm.get_blended_dataset()
        assert "sample_weight" in df.columns

        live_rows = df[df["source"] == "live"]
        wfo_rows = df[df["source"] == "wfo"]
        assert all(live_rows["sample_weight"] == 5.0)
        # WFO weight should be close to 1.0 with only 1 live trade
        assert all(wfo_rows["sample_weight"] > 0.99)

    def test_wfo_weight_decay(self, dm, sample_wfo_trades, tmp_path):
        """WFO weight should decrease as live trade count grows."""
        dm.ingest_wfo_trades(sample_wfo_trades)
        # Add 100 live trades
        for i in range(100):
            dm.ingest_live_trade(
                features={"ob_count": 1, "fvg_count": 1},
                outcome="tp1_hit", mfe=1.0, mae=0.5, pnl=0.5,
            )
        df = dm.get_blended_dataset()
        wfo_weights = df[df["source"] == "wfo"]["sample_weight"]
        # With 100 live trades and decay_rate=200: weight = max(0.2, 1.0 - 100/200) = 0.5
        assert all(abs(w - 0.5) < 0.01 for w in wfo_weights)

    def test_get_stats_regime_distribution(self, dm, sample_wfo_trades):
        dm.ingest_wfo_trades(sample_wfo_trades)
        stats = dm.get_stats()
        assert stats["regime_distribution"]["trending_up"] == 2
        assert stats["regime_distribution"]["ranging"] == 1

    def test_get_stats_empty(self, dm):
        stats = dm.get_stats()
        assert stats["total"] == 0

    def test_parquet_persistence(self, tmp_path, sample_wfo_trades):
        cfg = make_test_config(
            dataset_parquet_path=str(tmp_path / "persist.parquet"),
        )
        dm1 = TrainingDatasetManager(config=cfg)
        dm1.ingest_wfo_trades(sample_wfo_trades)

        dm2 = TrainingDatasetManager(config=cfg)
        assert dm2.get_stats()["total"] == 3

    def test_ingest_empty_wfo(self, dm):
        count = dm.ingest_wfo_trades([])
        assert count == 0

    def test_get_blended_dataset_live_only(self, dm, sample_wfo_trades):
        """live_only=True should exclude WFO rows entirely."""
        dm.ingest_wfo_trades(sample_wfo_trades)
        full_features = {
            "ob_count": 2, "ob_bullish_count": 1, "ob_bearish_count": 1,
            "ob_strongest_strength": 3, "ob_nearest_distance_atr": 0.5,
            "ob_avg_size_atr": 0.3, "ob_alignment": 1,
            "fvg_count": 1, "fvg_unfilled_count": 1, "fvg_nearest_distance_atr": 0.2,
            "fvg_avg_size_atr": 0.1, "fvg_alignment": 1,
            "liq_buyside_count": 1, "liq_sellside_count": 1,
            "liq_nearest_target_distance_atr": 1.0, "liq_nearest_threat_distance_atr": 2.0,
            "risk_reward_tp1": 3.0, "risk_reward_tp2": 5.0,
            "sl_distance_atr": 1.5, "tp1_distance_atr": 4.5,
            "entry_direction": 1, "bias_direction_match": 1,
            "num_confluences": 3, "has_ob_fvg_overlap": 1,
        }
        dm.ingest_live_trade(
            features=full_features,
            outcome="tp1_hit", mfe=3.0, mae=0.5, pnl=2.5,
        )
        dm.ingest_live_trade(
            features=full_features,
            outcome="stopped_out", mfe=0.5, mae=2.0, pnl=-1.0,
        )

        # Default: includes both WFO and live
        df_all = dm.get_blended_dataset()
        assert len(df_all[df_all["source"] == "wfo"]) == 3
        assert len(df_all[df_all["source"] == "live"]) == 2

        # live_only: excludes WFO
        df_live = dm.get_blended_dataset(live_only=True)
        assert len(df_live) == 2
        assert all(df_live["source"] == "live")
        assert "sample_weight" in df_live.columns

    def test_get_blended_dataset_live_only_empty(self, dm, sample_wfo_trades):
        """live_only=True with no live trades returns empty DataFrame."""
        dm.ingest_wfo_trades(sample_wfo_trades)
        df = dm.get_blended_dataset(live_only=True)
        assert len(df) == 0


# ─── Negative Example Generation ────────────────────────────────────


class TestNegativeExamples:
    def test_generates_no_trade_examples(self, wfo_candles):
        positive_trades = [
            {"candle_index": 100, "direction": "long"},
            {"candle_index": 200, "direction": "short"},
            {"candle_index": 300, "direction": "long"},
        ]
        negs = generate_negative_examples(wfo_candles, positive_trades,
                                          target_ratio=0.3)
        assert len(negs) > 0
        assert all(n["outcome"] == "no_trade" for n in negs)

    def test_negative_indices_not_near_positives(self, wfo_candles):
        positive_trades = [
            {"candle_index": 100, "direction": "long"},
            {"candle_index": 200, "direction": "short"},
        ]
        negs = generate_negative_examples(wfo_candles, positive_trades)
        pos_indices = {t["candle_index"] for t in positive_trades}
        for neg in negs:
            for pi in pos_indices:
                assert abs(neg["candle_index"] - pi) > 5

    def test_empty_positives(self, wfo_candles):
        assert generate_negative_examples(wfo_candles, []) == []

    def test_short_candles(self):
        assert generate_negative_examples([], [{"candle_index": 1}]) == []

    def test_has_38_features(self, wfo_candles):
        positive_trades = [{"candle_index": 100, "direction": "long"}]
        negs = generate_negative_examples(wfo_candles, positive_trades)
        if negs:
            # Should have at least ob_count, fvg_count, etc.
            assert "ob_count" in negs[0]
            assert "fvg_count" in negs[0]
            assert "direction" in negs[0]


# ─── RegimeBalancer ──────────────────────────────────────────────────


class TestRegimeBalancer:
    @pytest.fixture
    def balancer(self):
        return RegimeBalancer()

    def test_balance_upsamples_minority(self, balancer):
        data = (
            [{"regime": "trending_up", "ob_count": 1, "fvg_count": 1, "outcome": "tp1_hit"}] * 20 +
            [{"regime": "ranging", "ob_count": 1, "fvg_count": 0, "outcome": "stopped_out"}] * 5
        )
        df = pd.DataFrame(data)
        balanced = balancer.balance(df)
        regime_counts = balanced["regime"].value_counts()
        # Minority should be upsampled closer to majority
        assert regime_counts.get("ranging", 0) > 5

    def test_balance_downsamples_majority(self, balancer):
        data = (
            [{"regime": "trending_up", "ob_count": 1, "fvg_count": 1, "outcome": "tp1_hit"}] * 50 +
            [{"regime": "ranging", "ob_count": 1, "fvg_count": 0, "outcome": "stopped_out"}] * 5
        )
        df = pd.DataFrame(data)
        balanced = balancer.balance(df)
        regime_counts = balanced["regime"].value_counts()
        assert regime_counts.get("trending_up", 0) < 50

    def test_balance_empty_df(self, balancer):
        result = balancer.balance(pd.DataFrame())
        assert result.empty

    def test_get_regime_coverage(self, balancer):
        data = [
            {"regime": "trending_up", "ob_count": 1, "fvg_count": 1, "outcome": "tp1_hit"},
            {"regime": "ranging", "ob_count": 1, "fvg_count": 0, "outcome": "stopped_out"},
        ]
        balancer.balance(pd.DataFrame(data))
        coverage = balancer.get_regime_coverage()
        assert "trending_up" in coverage
        assert "ranging" in coverage

    def test_defensive_adjustment_underrepresented(self, balancer):
        # Simulate regime counts where one regime is tiny
        balancer._regime_counts = {
            "trending_up": 50, "ranging": 2, "high_volatility": 30,
        }
        adj = balancer.get_defensive_adjustment("ranging")
        assert adj < 1.0

    def test_defensive_adjustment_well_represented(self, balancer):
        balancer._regime_counts = {
            "trending_up": 30, "ranging": 30, "high_volatility": 30,
        }
        adj = balancer.get_defensive_adjustment("trending_up")
        assert adj == 1.0

    def test_defensive_adjustment_empty(self, balancer):
        assert balancer.get_defensive_adjustment("any") == 1.0


# ─── PriorValidator ──────────────────────────────────────────────────


class TestPriorValidator:
    @pytest.fixture
    def validator(self):
        return PriorValidator()

    def test_robust_priors_pass(self, validator):
        state = {"alpha": 20.0, "beta_param": 15.0}
        result = validator.stress_test_priors(state)
        assert result["passed"] is True

    def test_fragile_priors_fail(self, validator):
        state = {"alpha": 1.0, "beta_param": 1.0}
        result = validator.stress_test_priors(state)
        # Beta(1,1) + 50 losses → mean drops quickly
        assert "breakdown_after_n_losses" in result

    def test_validate_before_anchoring_small_oos(self, validator):
        class FakeReport:
            total_oos_trades = 10
            oos_win_rate = 0.5
        assert validator.validate_before_anchoring(FakeReport(), None) is True

    def test_validate_before_anchoring_robust(self, validator):
        class FakeReport:
            total_oos_trades = 100
            oos_win_rate = 0.55
        assert validator.validate_before_anchoring(FakeReport(), None) is True


# ─── DriftAlarm ──────────────────────────────────────────────────────


class TestDriftAlarm:
    @pytest.fixture
    def alarm(self):
        cfg = make_test_config(
            drift_significant_threshold=2.0,
            drift_critical_threshold=3.0,
            kappa_cap=30,
        )
        return DriftAlarm(config=cfg)

    def test_no_drift_for_fresh_state(self, alarm):
        state = {"alpha": 1.0, "beta_param": 1.0}
        result = alarm.check_drift(state)
        assert result["level"] == "none"

    def test_significant_drift(self, alarm):
        state = {"alpha": 30.0, "beta_param": 5.0}
        ref = {"alpha": 1.0, "beta_param": 1.0}
        result = alarm.check_drift(state, ref)
        assert result["drift_sd"] > 0

    def test_none_state(self, alarm):
        result = alarm.check_drift(None)
        assert result["level"] == "none"

    def test_cap_kappa_no_change_below_cap(self, alarm):
        state = {"alpha": 10.0, "beta_param": 10.0}
        result = alarm.cap_kappa(state)
        assert result["alpha"] == 10.0
        assert result["beta_param"] == 10.0

    def test_cap_kappa_scales_down(self, alarm):
        state = {"alpha": 30.0, "beta_param": 30.0}
        result = alarm.cap_kappa(state)
        assert result["alpha"] + result["beta_param"] <= 30.0
        # Proportions preserved
        assert abs(result["alpha"] - result["beta_param"]) < 0.01

    def test_cap_kappa_preserves_ratio(self, alarm):
        state = {"alpha": 40.0, "beta_param": 20.0}
        result = alarm.cap_kappa(state)
        total = result["alpha"] + result["beta_param"]
        assert total <= 30.0
        # Ratio should be 2:1
        assert abs(result["alpha"] / result["beta_param"] - 2.0) < 0.01

    def test_cap_kappa_none_state(self, alarm):
        assert alarm.cap_kappa(None) is None
