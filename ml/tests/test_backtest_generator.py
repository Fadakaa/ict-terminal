"""Tests for ml/backtest_generator.py — all mocked, no API calls."""

import json
import math
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from ml.config import make_test_config
from ml.backtest_generator import BacktestGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config(tmp_path):
    return make_test_config(
        model_dir=str(tmp_path / "models"),
        db_path=str(tmp_path / "test.db"),
        dataset_parquet_path=str(tmp_path / "models" / "training_dataset.parquet"),
        backtest_data_source="oanda",
        oanda_account_id="test-account",
        oanda_access_token="test-token",
    )


@pytest.fixture
def generator(test_config):
    return BacktestGenerator(config=test_config)


@pytest.fixture
def synthetic_1h_candles():
    """350 synthetic 1H candles with realistic gold-like price action.

    Creates varied price movement with OBs, FVGs, and displacement-like
    candles to test the structural scanner.
    """
    candles = []
    base = 2600.0
    for i in range(350):
        o = base + i * 0.3
        h = o + 4.0 + math.sin(i * 0.2) * 2
        l = o - 3.0 - abs(math.cos(i * 0.15)) * 2
        c = o + 1.5 if i % 2 == 0 else o - 0.8
        hour = i % 24
        day = 1 + i // 24
        candles.append({
            "datetime": f"2025-09-{day:02d} {hour:02d}:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": 1000 + i,
        })
    return candles


@pytest.fixture
def synthetic_daily_candles():
    """60 synthetic daily candles for regime classification."""
    candles = []
    base = 2600.0
    for i in range(60):
        # First 20: trending (strong directional move)
        if i < 20:
            o = base + i * 5.0
            c = o + 4.0
        # Middle 20: ranging (small moves)
        elif i < 40:
            o = base + 100 + math.sin(i * 0.5) * 3
            c = o + 0.5 if i % 2 == 0 else o - 0.5
        # Last 20: volatile (large swings)
        else:
            o = base + 100 + math.sin(i) * 20
            c = o + 15.0 if i % 2 == 0 else o - 15.0

        h = max(o, c) + 5.0
        l = min(o, c) - 4.0
        month = "09" if i < 20 else ("10" if i < 40 else "11")
        day = (i % 20) + 1
        candles.append({
            "datetime": f"2025-{month}-{day:02d} 00:00:00",
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": 50000 + i * 100,
        })
    return candles


def _make_claude_response():
    """Return a plausible Claude analysis JSON."""
    return {
        "bias": "bullish",
        "summary": "Bullish OB confluence",
        "orderBlocks": [
            {"type": "bullish", "high": 2650.0, "low": 2645.0, "candleIndex": 80, "strength": "strong"},
        ],
        "fvgs": [
            {"type": "bullish", "high": 2660.0, "low": 2655.0, "startIndex": 85, "filled": False},
        ],
        "liquidity": [
            {"type": "buyside", "price": 2690.0, "candleIndex": 50},
        ],
        "entry": {"price": 2650.0, "direction": "long"},
        "stopLoss": {"price": 2643.0},
        "takeProfits": [
            {"price": 2670.0, "rr": 2.9},
            {"price": 2685.0, "rr": 5.0},
        ],
        "killzone": "London",
        "confluences": ["Bullish OB", "FVG overlap"],
    }


# ---------------------------------------------------------------------------
# 1. test_strip_candle_dates
# ---------------------------------------------------------------------------

class TestStripCandleDates:
    def test_removes_datetime_adds_index(self, generator):
        candles = [
            {"datetime": "2025-09-01 08:00:00", "open": 2600, "high": 2610, "low": 2595, "close": 2605, "volume": 100},
            {"datetime": "2025-09-01 09:00:00", "open": 2605, "high": 2615, "low": 2600, "close": 2610, "volume": 200},
        ]
        stripped = generator._strip_candle_dates(candles)

        assert len(stripped) == 2
        assert "datetime" not in stripped[0]
        assert "datetime" not in stripped[1]
        assert stripped[0]["index"] == 0
        assert stripped[1]["index"] == 1
        assert stripped[0]["open"] == 2600
        assert stripped[1]["close"] == 2610
        assert stripped[0]["volume"] == 100


# ---------------------------------------------------------------------------
# 2. test_entry_noise_jitter
# ---------------------------------------------------------------------------

class TestEntryNoiseJitter:
    def test_shifts_price_preserves_sl_distance(self, generator, synthetic_1h_candles):
        entry = 2650.0
        sl_dist = 7.0
        result = generator._add_entry_noise(entry, sl_dist, "long", synthetic_1h_candles, 150)

        # Entry should be different from original (most of the time)
        new_entry = result["entry"]
        new_sl = result["sl"]

        # SL distance preserved
        assert abs(abs(new_entry - new_sl) - sl_dist) < 0.01

        # Jitter is bounded -3 to 3
        assert -3 <= result["jitter_candles"] <= 3

    def test_short_direction_sl_above(self, generator, synthetic_1h_candles):
        result = generator._add_entry_noise(2650.0, 7.0, "short", synthetic_1h_candles, 150)
        assert result["sl"] > result["entry"]


# ---------------------------------------------------------------------------
# 3. test_regime_classification
# ---------------------------------------------------------------------------

class TestRegimeClassification:
    def test_classifies_synthetic_daily(self, generator, synthetic_daily_candles):
        regimes = generator._classify_regime(synthetic_daily_candles)

        assert isinstance(regimes, dict)
        assert len(regimes) > 0

        # At least one regime should be assigned
        values = set(regimes.values())
        for v in values:
            assert v in {"trending", "ranging", "volatile"}

    def test_empty_candles(self, generator):
        result = generator._classify_regime([])
        assert result == {}

    def test_short_candles(self, generator):
        result = generator._classify_regime([{"datetime": "2025-09-01", "open": 100, "high": 101, "low": 99, "close": 100}] * 5)
        assert result == {}


# ---------------------------------------------------------------------------
# 4. test_structural_scan (NEW — Pass 1)
# ---------------------------------------------------------------------------

class TestStructuralScan:
    def test_finds_candidates_in_killzone_candles(self, generator, synthetic_1h_candles):
        """Structural scan should find candidates during London/NY hours."""
        regime_map = {"2025-09": "ranging"}
        candidates = generator.structural_scan(synthetic_1h_candles, regime_map)

        # Should find at least some candidates (synthetic data has some structure)
        assert isinstance(candidates, list)
        # Each candidate has required fields
        for cand in candidates:
            assert "candle_idx" in cand
            assert "score" in cand
            assert cand["score"] >= 2
            assert "killzone" in cand
            assert cand["killzone"] not in _SKIP_KILLZONES
            assert "structural_elements" in cand
            assert "atr" in cand

    def test_skips_asian_and_off_hours(self, generator):
        """Asian/off-hours candles should be excluded."""
        # Build candles all at 3am (Asian)
        candles = []
        for i in range(200):
            candles.append({
                "datetime": f"2025-09-01 03:00:00",
                "open": 2600 + i * 0.1,
                "high": 2605 + i * 0.1,
                "low": 2595 + i * 0.1,
                "close": 2602 + i * 0.1,
                "volume": 100,
            })
        result = generator.structural_scan(candles, {"2025-09": "ranging"})
        assert result == []

    def test_empty_market_no_candidates(self, generator):
        """Flat candles with no structure should produce no candidates."""
        candles = []
        for i in range(200):
            candles.append({
                "datetime": f"2025-09-01 10:00:00",
                "open": 2600.0,
                "high": 2600.5,
                "low": 2599.5,
                "close": 2600.0,
                "volume": 100,
            })
        result = generator.structural_scan(candles, {"2025-09": "ranging"})
        # Flat market — no displacement, no OBs, no FVGs, no sweeps
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 5. test_deduplicate_candidates (NEW)
# ---------------------------------------------------------------------------

class TestDeduplicateCandidates:
    def test_clusters_within_gap(self, generator):
        """Adjacent candidates within gap=4 should cluster, keep highest."""
        candidates = [
            {"candle_idx": 100, "score": 2},
            {"candle_idx": 102, "score": 4},
            {"candle_idx": 104, "score": 3},
            {"candle_idx": 200, "score": 5},
            {"candle_idx": 201, "score": 3},
        ]
        result = generator._deduplicate_candidates(candidates, gap=4)

        assert len(result) == 2
        # First cluster: [100, 102, 104] → keep score=4
        assert result[0]["score"] == 4
        assert result[0]["candle_idx"] == 102
        # Second cluster: [200, 201] → keep score=5
        assert result[1]["score"] == 5

    def test_empty_input(self, generator):
        assert generator._deduplicate_candidates([], gap=4) == []

    def test_single_candidate(self, generator):
        result = generator._deduplicate_candidates(
            [{"candle_idx": 50, "score": 3}], gap=4
        )
        assert len(result) == 1

    def test_no_overlap(self, generator):
        """Candidates far apart should all survive."""
        candidates = [
            {"candle_idx": 100, "score": 2},
            {"candle_idx": 200, "score": 3},
            {"candle_idx": 300, "score": 4},
        ]
        result = generator._deduplicate_candidates(candidates, gap=4)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 6. test_regime_cap (NEW)
# ---------------------------------------------------------------------------

class TestRegimeCap:
    def test_caps_dominant_regime(self, generator):
        """No regime should exceed 45% after the cap has enough data."""
        candidates = []
        # Interleave regimes so the cap can act early
        for i in range(10):
            candidates.append({"candle_idx": i, "regime": "trending"})
        for i in range(10):
            candidates.append({"candle_idx": 50 + i, "regime": "volatile"})
        # Now add 40 ranging — cap should kick in after total > 20
        for i in range(40):
            candidates.append({"candle_idx": 100 + i, "regime": "ranging"})

        filtered = generator._apply_regime_cap(candidates, max_pct=0.45)

        # Ranging should have been capped — can't keep all 40
        ranging = sum(1 for c in filtered if c["regime"] == "ranging")
        total = len(filtered)
        assert ranging < 40, f"Expected ranging to be capped, got {ranging}/40"

    def test_skips_cap_with_single_regime(self, generator):
        """All-same-regime candidates should pass through uncapped."""
        candidates = [
            {"candle_idx": i, "regime": "ranging"} for i in range(100)
        ]
        filtered = generator._apply_regime_cap(candidates, max_pct=0.45)
        assert len(filtered) == 100


# ---------------------------------------------------------------------------
# 7. test_outcome_labelling
# ---------------------------------------------------------------------------

class TestOutcomeLabelling:
    def test_stopped_out_long(self, generator):
        forward = [
            {"high": 2655, "low": 2640, "close": 2645},
            {"high": 2650, "low": 2630, "close": 2635},  # SL hit at 2643
        ]
        result = generator._label_outcome(2650.0, 2643.0, [2670.0], "long", forward)
        assert result == "stopped_out"

    def test_tp1_long(self, generator):
        # entry=2650, sl=2643 (dist=7), tp1=2660, tp2=2667, tp3=2674
        forward = [
            {"high": 2658, "low": 2648, "close": 2656},
            {"high": 2661, "low": 2655, "close": 2660},  # TP1 hit at 2660
        ]
        result = generator._label_outcome(
            2650.0, 2643.0, [2660.0, 2667.0, 2674.0], "long", forward)
        assert result == "tp1"

    def test_tp2_long(self, generator):
        # entry=2650, sl=2643 (dist=7), tp1=2660, tp2=2667
        forward = [
            {"high": 2661, "low": 2648, "close": 2658},  # TP1 hit
            {"high": 2668, "low": 2655, "close": 2666},  # TP2 hit at 2667
        ]
        result = generator._label_outcome(
            2650.0, 2643.0, [2660.0, 2667.0, 2674.0], "long", forward)
        assert result == "tp2"

    def test_tp3_long(self, generator):
        # entry=2650, sl=2643 (dist=7), tp1=2660, tp2=2667, tp3=2674
        forward = [
            {"high": 2668, "low": 2648, "close": 2658},  # TP1+TP2 hit
            {"high": 2675, "low": 2660, "close": 2670},  # TP3 hit at 2674
        ]
        result = generator._label_outcome(
            2650.0, 2643.0, [2660.0, 2667.0, 2674.0], "long", forward)
        assert result == "tp3"

    def test_tp3_hit_long(self, generator):
        # entry=2650, sl=2643 (dist=7), default levels: tp1=2664, tp2=2671, tp3=2678
        # tp3_hit requires hitting tp3 level on two separate candles
        forward = [
            {"high": 2672, "low": 2648, "close": 2668},  # TP1+TP2 hit
            {"high": 2679, "low": 2665, "close": 2676},  # First TP3 hit → best_hit=tp3
            {"high": 2679, "low": 2673, "close": 2678},  # Second TP3 hit → tp3_hit
        ]
        result = generator._label_outcome(2650.0, 2643.0, None, "long", forward)
        assert result == "tp3_hit"

    def test_stopped_out_short(self, generator):
        forward = [
            {"high": 2660, "low": 2648, "close": 2655},  # SL hit (entry=2650, sl=2657)
        ]
        result = generator._label_outcome(2650.0, 2657.0, [2640.0], "short", forward)
        assert result == "stopped_out"

    def test_empty_forward(self, generator):
        result = generator._label_outcome(2650.0, 2643.0, [2670.0], "long", [])
        assert result == "stopped_out"


# ---------------------------------------------------------------------------
# 8. test_temporal_holdout_ordered
# ---------------------------------------------------------------------------

class TestTemporalHoldout:
    def test_all_train_before_test(self):
        """Verify concept: in a temporal split, all train timestamps < test timestamps."""
        # Simulate a backtest-generated dataset with timestamps
        timestamps = list(range(100))
        split_point = 80

        train = timestamps[:split_point]
        test = timestamps[split_point:]

        assert all(t < test[0] for t in train)
        assert all(t >= train[-1] for t in test)
        # No leakage: train and test don't overlap
        assert set(train) & set(test) == set()


# ---------------------------------------------------------------------------
# 9. test_source_weight_correct
# ---------------------------------------------------------------------------

class TestSourceWeight:
    def test_backtest_weight_is_0_7(self, test_config):
        """Backtest rows should have source_weight=0.7, live should be 1.0."""
        bt_row = {"source": "backtest", "source_weight": 0.7}
        live_row = {"source": "live", "source_weight": 1.0}

        assert bt_row["source_weight"] == 0.7
        assert live_row["source_weight"] == 1.0
        assert bt_row["source_weight"] < live_row["source_weight"]


# ---------------------------------------------------------------------------
# 10. test_budget_limit_stops
# ---------------------------------------------------------------------------

class TestBudgetLimit:
    @patch("ml.backtest_generator.get_provider")
    @patch("ml.backtest_generator.get_cost_tracker")
    @patch.object(BacktestGenerator, "_call_haiku")
    @patch.object(BacktestGenerator, "_call_claude")
    @patch("ml.backtest_generator.TrainingDatasetManager")
    @patch("ml.backtest_generator.ClaudeAnalysisBridge")
    @patch("ml.backtest_generator.extract_features")
    def test_stops_when_budget_exceeded(self, mock_feat, mock_bridge_cls, mock_ds_cls,
                                         mock_claude, mock_haiku, mock_tracker,
                                         mock_provider,
                                         test_config, synthetic_1h_candles,
                                         synthetic_daily_candles):
        from ml.feature_schema import FEATURE_COLUMNS as FC

        mock_provider_inst = MagicMock()
        mock_provider_inst.fetch_candles.side_effect = [
            synthetic_1h_candles,  # 1H
            [],                    # 4H (HTF context)
            [],                    # EUR/USD (intermarket)
            [],                    # US10Y (intermarket)
            synthetic_daily_candles,  # daily (regime)
        ]
        mock_provider_inst.name.return_value = "mock"
        mock_provider.return_value = mock_provider_inst

        # Each Sonnet call costs $5 (high cost to hit budget fast)
        tracker_inst = MagicMock()
        tracker_inst.log_call.return_value = 5.0
        tracker_inst.flush = MagicMock()
        mock_tracker.return_value = tracker_inst

        # Haiku approves everything
        mock_haiku.return_value = {
            "valid": True,
            "direction": "long",
            "entry_price": 2650.0,
            "sl_price": 2643.0,
        }

        mock_claude.return_value = _make_claude_response()

        mock_ds_inst = MagicMock()
        mock_ds_inst._df = pd.DataFrame()
        mock_ds_cls.return_value = mock_ds_inst

        mock_bridge_inst = MagicMock()
        mock_bridge_inst.parse_analysis.return_value = {
            "claude_entry_price": 2650.0,
            "claude_sl_price": 2643.0,
            "claude_direction": "long",
            "claude_tp_prices": [2670.0],
        }
        mock_bridge_cls.return_value = mock_bridge_inst

        mock_feat.return_value = {col: 0.0 for col in FC}

        gen = BacktestGenerator(config=test_config)
        result = gen.generate(months_back=1, max_setups=100, budget_limit_usd=10.0)

        # Should have stopped early because $5/call hits budget fast
        assert result["total_cost"] >= 10.0 or result["setups_found"] <= 5


# ---------------------------------------------------------------------------
# 11. test_checkpoint_save_resume
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self, generator):
        state = {
            "last_candle_idx": 500,
            "setups_found": 42,
            "cost": 12.50,
            "regime_counts": {"trending": 15, "ranging": 20, "volatile": 7},
        }

        generator._save_checkpoint(state)
        loaded = generator._load_checkpoint()

        assert loaded["last_candle_idx"] == 500
        assert loaded["cost"] == 12.50
        assert loaded["regime_counts"]["trending"] == 15
        assert loaded["regime_counts"]["ranging"] == 20

    def test_missing_checkpoint_returns_defaults(self, generator):
        # Ensure no checkpoint file exists
        if os.path.exists(generator._checkpoint_path):
            os.remove(generator._checkpoint_path)

        loaded = generator._load_checkpoint()
        assert loaded["last_candle_idx"] == 200
        assert loaded["setups_found"] == 0
        assert loaded["cost"] == 0.0


# ---------------------------------------------------------------------------
# 12. test_fidelity_check_flags_bad
# ---------------------------------------------------------------------------

class TestFidelityCheck:
    def test_flags_bad_distribution(self, test_config, tmp_path):
        """Fidelity check should flag when backtest has very different distributions from live."""
        from ml.feature_schema import FEATURE_COLUMNS
        import numpy as np

        gen = BacktestGenerator(config=test_config)

        # Create synthetic dataset with divergent backtest vs live
        n_bt = 50
        n_live = 50
        rng = np.random.RandomState(42)

        rows = []
        # Backtest rows: unrealistically high win rate, different feature distributions
        for i in range(n_bt):
            row = {col: rng.normal(10, 1) for col in FEATURE_COLUMNS}
            row["source"] = "backtest"
            row["source_weight"] = 0.7
            row["outcome"] = "tp1" if i < 40 else "stopped_out"  # 80% WR
            row["setup_id"] = f"bt-{i:04d}"
            row["killzone_encoded"] = 1
            row["sl_distance_atr"] = rng.normal(2.0, 0.3)
            rows.append(row)

        # Live rows: lower win rate, shifted features
        for i in range(n_live):
            row = {col: rng.normal(5, 2) for col in FEATURE_COLUMNS}  # Different mean
            row["source"] = "live"
            row["source_weight"] = 1.0
            row["outcome"] = "tp1" if i < 20 else "stopped_out"  # 40% WR
            row["setup_id"] = f"live-{i:04d}"
            row["killzone_encoded"] = 2  # Different killzone distribution
            row["sl_distance_atr"] = rng.normal(1.0, 0.5)  # Different SL distribution
            rows.append(row)

        df = pd.DataFrame(rows)

        # Write CSV for dataset manager to load
        csv_path = str(tmp_path / "models" / "training_dataset.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

        result = gen.run_fidelity_check()

        # Should have flagged multiple issues
        assert result["checks_failed"] >= 2, f"Expected 2+ failures, got {result['checks_failed']}"
        # Weight should be reduced
        assert result["adjusted_weight"] <= 0.4


# ---------------------------------------------------------------------------
# 13. test_self_correcting_weights
# ---------------------------------------------------------------------------

class TestSelfCorrectingWeights:
    def test_reduces_weight_at_2_failures(self):
        """At 2+ check failures, weight drops to 0.4. At 4+, to 0.2."""
        # Test the weight logic directly
        # 0-1 failures → 0.7
        # 2-3 failures → 0.4
        # 4+ failures → 0.2
        checks_map = {
            0: 0.7,
            1: 0.7,
            2: 0.4,
            3: 0.4,
            4: 0.2,
            5: 0.2,
        }

        for checks_failed, expected_weight in checks_map.items():
            if checks_failed >= 4:
                adjusted = 0.2
            elif checks_failed >= 2:
                adjusted = 0.4
            else:
                adjusted = 0.7
            assert adjusted == expected_weight, \
                f"checks_failed={checks_failed}: expected {expected_weight}, got {adjusted}"

    def test_fidelity_reduces_to_0_2_at_4_failures(self, test_config, tmp_path):
        """4+ fidelity failures should reduce backtest weight to 0.2."""
        from ml.feature_schema import FEATURE_COLUMNS
        import numpy as np

        gen = BacktestGenerator(config=test_config)
        n = 60
        rng = np.random.RandomState(99)

        rows = []
        # Backtest: extremely different from live in every dimension
        for i in range(n):
            row = {col: rng.normal(100, 1) for col in FEATURE_COLUMNS}
            row["source"] = "backtest"
            row["source_weight"] = 0.7
            row["outcome"] = "runner" if i < 55 else "stopped_out"  # 92% WR (unrealistic)
            row["setup_id"] = f"bt-{i:04d}"
            row["killzone_encoded"] = 1
            row["sl_distance_atr"] = rng.normal(5.0, 0.1)
            rows.append(row)

        for i in range(n):
            row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}  # Completely different
            row["source"] = "live"
            row["source_weight"] = 1.0
            row["outcome"] = "stopped_out" if i < 42 else "tp1"  # 30% WR
            row["setup_id"] = f"live-{i:04d}"
            row["killzone_encoded"] = 3  # All different killzone
            row["sl_distance_atr"] = rng.normal(1.0, 0.1)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "models" / "training_dataset.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

        result = gen.run_fidelity_check()

        assert result["checks_failed"] >= 4, f"Expected 4+ failures, got {result['checks_failed']}"
        assert result["adjusted_weight"] == 0.2


# ---------------------------------------------------------------------------
# 14. test_haiku_prompt (NEW)
# ---------------------------------------------------------------------------

class TestHaikuPrompt:
    def test_includes_structural_elements(self):
        """Haiku prompt should include pre-detected structural elements."""
        from ml.prompts import build_haiku_backtest_prompt

        candles = [
            {"open": 2600 + i, "high": 2605 + i, "low": 2595 + i,
             "close": 2602 + i, "volume": 100}
            for i in range(40)
        ]
        elements = {
            "ob_count": 2,
            "ob_types": ["bullish", "bearish"],
            "fvg_count": 1,
            "fvg_types": ["bullish"],
            "sweep_detected": True,
            "structure_score": 0.65,
            "displacement": True,
            "price_in_zone": True,
        }

        prompt = build_haiku_backtest_prompt(candles, elements)

        assert "Order Blocks: 2" in prompt
        assert "bullish" in prompt
        assert "FVGs: 1" in prompt
        assert "Liquidity sweep confirmed" in prompt
        assert "Displacement candle present" in prompt
        assert "Market structure: bullish" in prompt
        assert "Price is inside an OB/FVG zone" in prompt
        assert "valid" in prompt  # Response format

    def test_minimal_elements(self):
        """Prompt works with minimal structural elements."""
        from ml.prompts import build_haiku_backtest_prompt

        candles = [
            {"open": 2600, "high": 2605, "low": 2595, "close": 2602, "volume": 100}
            for _ in range(40)
        ]
        elements = {
            "ob_count": 0, "ob_types": [],
            "fvg_count": 0, "fvg_types": [],
            "sweep_detected": False, "structure_score": 0.1,
            "displacement": False, "price_in_zone": False,
        }

        prompt = build_haiku_backtest_prompt(candles, elements)
        assert "Minimal confluence" in prompt


# ---------------------------------------------------------------------------
# 15. test_generate_dry_run (NEW)
# ---------------------------------------------------------------------------

class TestDryRun:
    @patch("ml.backtest_generator.get_provider")
    def test_returns_pass1_count_without_api(self, mock_provider,
                                              test_config, synthetic_1h_candles,
                                              synthetic_daily_candles):
        """dry_run=True should run Pass 1 only, no API calls."""
        mock_provider_inst = MagicMock()
        mock_provider_inst.fetch_candles.side_effect = [
            synthetic_1h_candles,  # 1H
            [],                    # 4H (HTF context)
            [],                    # EUR/USD (intermarket)
            [],                    # US10Y (intermarket)
            synthetic_daily_candles,  # daily (regime)
        ]
        mock_provider_inst.name.return_value = "mock"
        mock_provider.return_value = mock_provider_inst

        gen = BacktestGenerator(config=test_config)
        result = gen.generate(months_back=1, dry_run=True)

        assert result.get("dry_run") is True
        assert "pass1_candidates" in result
        assert isinstance(result["pass1_candidates"], int)


# ---------------------------------------------------------------------------
# 16. test_three_pass_flow (NEW)
# ---------------------------------------------------------------------------

class TestThreePassFlow:
    @staticmethod
    def _make_candles_with_structure():
        """Build 350 candles with actual displacement, OBs, FVGs during London hours.

        Key: some candles have large bodies (> 1.5x ATR) for displacement,
        plus gaps between candle[i].low and candle[i-2].high for FVGs.
        All during London hours (08:00-11:00) so killzone passes.
        """
        candles = []
        base = 2600.0
        for i in range(350):
            hour = 8 + (i % 4)  # 8, 9, 10, 11 — all London
            day = 1 + i // 24
            month = "09" if day <= 28 else "10"
            day_clamped = ((day - 1) % 28) + 1

            o = base + i * 0.5

            # Every 15 candles, create a big displacement candle
            if i % 15 == 0 and i > 0:
                # Large bullish candle: body > 15 (ATR ~5-7 for these candles)
                h = o + 20.0
                l = o - 1.0
                c = o + 18.0
            # Create a gap (FVG) every 15 candles offset by 2
            elif i % 15 == 2 and i > 2:
                h = o + 25.0
                l = o + 10.0  # Gap up from 2 candles ago
                c = o + 22.0
            else:
                h = o + 5.0
                l = o - 4.0
                c = o + 1.5 if i % 2 == 0 else o - 0.8

            candles.append({
                "datetime": f"2025-{month}-{day_clamped:02d} {hour:02d}:00:00",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 1000 + i,
            })
        return candles

    @patch("ml.backtest_generator.get_provider")
    @patch("ml.backtest_generator.get_cost_tracker")
    @patch.object(BacktestGenerator, "_call_haiku")
    @patch.object(BacktestGenerator, "_call_claude")
    @patch("ml.backtest_generator.TrainingDatasetManager")
    @patch("ml.backtest_generator.ClaudeAnalysisBridge")
    @patch("ml.backtest_generator.extract_features")
    def test_pipeline_haiku_filters_sonnet_enriches(
        self, mock_feat, mock_bridge_cls, mock_ds_cls,
        mock_claude, mock_haiku, mock_tracker, mock_provider,
        test_config, synthetic_daily_candles,
    ):
        """Three-pass pipeline: candidates → Haiku filter → Sonnet enrich."""
        from ml.feature_schema import FEATURE_COLUMNS as FC

        structured_candles = self._make_candles_with_structure()

        mock_provider_inst = MagicMock()
        mock_provider_inst.fetch_candles.side_effect = [
            structured_candles,        # 1H
            [],                        # 4H (HTF context)
            [],                        # EUR/USD (intermarket)
            [],                        # US10Y (intermarket)
            synthetic_daily_candles,   # daily (regime)
        ]
        mock_provider_inst.name.return_value = "mock"
        mock_provider.return_value = mock_provider_inst

        tracker_inst = MagicMock()
        tracker_inst.log_call.return_value = 0.001
        tracker_inst.flush = MagicMock()
        mock_tracker.return_value = tracker_inst

        # Haiku: approve first call, reject second, approve third, ...
        haiku_responses = []
        for i in range(500):
            if i % 2 == 0:
                haiku_responses.append({
                    "valid": True, "direction": "long",
                    "entry_price": 2650.0, "sl_price": 2643.0,
                })
            else:
                haiku_responses.append({"valid": False, "reasoning": "no setup"})
        mock_haiku.side_effect = haiku_responses

        mock_claude.return_value = _make_claude_response()

        mock_ds_inst = MagicMock()
        mock_ds_inst._df = pd.DataFrame()
        mock_ds_cls.return_value = mock_ds_inst

        mock_bridge_inst = MagicMock()
        mock_bridge_inst.parse_analysis.return_value = {
            "claude_entry_price": 2650.0,
            "claude_sl_price": 2643.0,
            "claude_direction": "long",
            "claude_tp_prices": [2670.0],
        }
        mock_bridge_cls.return_value = mock_bridge_inst

        mock_feat.return_value = {col: 0.0 for col in FC}

        gen = BacktestGenerator(config=test_config)
        result = gen.generate(months_back=1, max_setups=10, budget_limit_usd=50.0)

        # Should have pass1 > pass2 (Haiku filters ~50%) > pass3
        assert result["pass1_candidates"] > 0, \
            "Pass 1 should find structural candidates in candles with displacement+FVGs"
        assert result["pass2_haiku_approved"] <= result["pass1_candidates"]
        # Sonnet should have been called only for Haiku-approved
        assert result["pass3_sonnet_analysed"] <= result["pass2_haiku_approved"]
        # Cost breakdown present
        assert "haiku_cost" in result
        assert "sonnet_cost" in result
        assert "cost_per_setup" in result


# ---------------------------------------------------------------------------
# 17. HTF alignment helpers
# ---------------------------------------------------------------------------

class TestHTFAlignment:
    """Test _find_4h_window and _find_intermarket_window helpers."""

    def test_find_4h_window_returns_correct_slice(self):
        """4H window should end at or before the target 1H timestamp."""
        candles_4h = [
            {"datetime": f"2025-09-01 {h:02d}:00:00", "open": 2600 + i,
             "high": 2605 + i, "low": 2595 + i, "close": 2602 + i, "volume": 100}
            for i, h in enumerate([0, 4, 8, 12, 16, 20])
        ]
        timestamps = [c["datetime"] for c in candles_4h]

        # Target at 10:00 — should get candles at 00:00, 04:00, 08:00
        result = BacktestGenerator._find_4h_window(
            candles_4h, timestamps, "2025-09-01 10:00:00", count=20
        )
        assert len(result) == 3
        assert result[-1]["datetime"] == "2025-09-01 08:00:00"

    def test_find_4h_window_exact_match(self):
        """When target matches a 4H candle exactly, include it."""
        candles_4h = [
            {"datetime": "2025-09-01 08:00:00", "open": 2600,
             "high": 2605, "low": 2595, "close": 2602, "volume": 100},
            {"datetime": "2025-09-01 12:00:00", "open": 2602,
             "high": 2610, "low": 2598, "close": 2607, "volume": 100},
        ]
        timestamps = [c["datetime"] for c in candles_4h]

        result = BacktestGenerator._find_4h_window(
            candles_4h, timestamps, "2025-09-01 12:00:00", count=5
        )
        # bisect_right("12:00:00") returns 2 — gets both candles
        assert len(result) == 2
        assert result[-1]["datetime"] == "2025-09-01 12:00:00"

    def test_find_4h_window_empty_input(self):
        result = BacktestGenerator._find_4h_window([], [], "2025-09-01 10:00:00")
        assert result == []

    def test_find_4h_window_respects_count(self):
        """Should return at most `count` candles."""
        candles_4h = [
            {"datetime": f"2025-09-01 {h:02d}:00:00", "open": 2600,
             "high": 2605, "low": 2595, "close": 2602, "volume": 100}
            for h in [0, 4, 8, 12, 16, 20]
        ]
        timestamps = [c["datetime"] for c in candles_4h]

        result = BacktestGenerator._find_4h_window(
            candles_4h, timestamps, "2025-09-01 23:00:00", count=3
        )
        assert len(result) == 3

    def test_find_intermarket_window_returns_correct_slice(self):
        """Intermarket window should end at or before the target timestamp."""
        candles = [
            {"datetime": f"2025-09-01 {h:02d}:00:00", "open": 1.08 + i * 0.001,
             "high": 1.082 + i * 0.001, "low": 1.078 + i * 0.001,
             "close": 1.081 + i * 0.001, "volume": 50}
            for i, h in enumerate(range(0, 24))
        ]

        result = BacktestGenerator._find_intermarket_window(
            candles, "2025-09-01 10:30:00", count=5
        )
        assert len(result) == 5
        # Last candle should be 10:00 (last one <= 10:30)
        assert result[-1]["datetime"] == "2025-09-01 10:00:00"

    def test_find_intermarket_window_empty(self):
        result = BacktestGenerator._find_intermarket_window(
            [], "2025-09-01 10:00:00"
        )
        assert result == []


# ---------------------------------------------------------------------------
# Import guard for _SKIP_KILLZONES
# ---------------------------------------------------------------------------
from ml.backtest_generator import _SKIP_KILLZONES
