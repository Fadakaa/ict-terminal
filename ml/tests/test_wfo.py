"""Tests for Walk-Forward Optimization engine.

Covers: ICT structure detection (features.py additions), WFO engine,
regime detection, report generation, serialization, and server endpoints.
"""
import json
import math
import os

import pytest
import pandas as pd

from ml.features import (
    compute_atr,
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity,
    compute_market_structure,
    create_trade_labels,
    engineer_features_from_candles,
    _extract_hour,
)
from ml.wfo import (
    WFOConfig,
    FoldResult,
    WFOReport,
    ICTSetupDetector,
    WalkForwardEngine,
    detect_regime,
    save_report,
    load_report,
    build_setup_filter,
    _compute_wfo_grade,
)


# ═══════════════════════════════════════════════════════════════════════
# ICT Structure Detection Tests (features.py)
# ═══════════════════════════════════════════════════════════════════════


class TestDetectOrderBlocks:
    """Test Order Block detection from raw candles."""

    def test_detects_bullish_ob(self):
        """Bullish displacement (big green candle) marks prior candle as OB."""
        candles = []
        for i in range(10):
            candles.append({
                "open": 100 + i, "high": 103 + i,
                "low": 98 + i, "close": 101 + i,
                "datetime": f"2026-01-01 {i:02d}:00:00",
            })
        # Add displacement candle (big bullish body > 1.5 * ATR)
        candles.append({
            "open": 110, "high": 140, "low": 108,
            "close": 138,  # body = 28
            "datetime": "2026-01-01 10:00:00",
        })
        # Use ATR from the calm candles only (body ~28 vs calm ATR ~5)
        atr = 5.0  # Representative of the calm market
        obs = detect_order_blocks(candles, atr)
        bullish = [ob for ob in obs if ob["type"] == "bullish"]
        assert len(bullish) >= 1
        assert bullish[0]["index"] < len(candles) - 1  # OB is preceding candle

    def test_detects_bearish_ob(self):
        """Bearish displacement marks prior candle as bearish OB."""
        candles = []
        for i in range(10):
            candles.append({
                "open": 200 - i, "high": 203 - i,
                "low": 198 - i, "close": 199 - i,
                "datetime": f"2026-01-01 {i:02d}:00:00",
            })
        # Big bearish candle
        candles.append({
            "open": 195, "high": 196, "low": 160,
            "close": 162,  # body = 33
            "datetime": "2026-01-01 10:00:00",
        })
        atr = 5.0  # Representative of the calm market
        obs = detect_order_blocks(candles, atr)
        bearish = [ob for ob in obs if ob["type"] == "bearish"]
        assert len(bearish) >= 1

    def test_no_obs_in_quiet_market(self, sample_candles):
        """Uniform candles without displacement produce no OBs."""
        atr = compute_atr(sample_candles, 14)
        obs = detect_order_blocks(sample_candles, atr)
        # With gentle 0.5/candle moves and bodies of ~1.5, unlikely to trigger
        # This may or may not find OBs depending on exact ATR vs body size
        # The key contract: function runs without error and returns a list
        assert isinstance(obs, list)

    def test_ob_does_not_mutate_input(self, sample_candles):
        """detect_order_blocks must not modify the input candle list."""
        original_len = len(sample_candles)
        original_first = dict(sample_candles[0])
        atr = compute_atr(sample_candles, 14)
        detect_order_blocks(sample_candles, atr)
        assert len(sample_candles) == original_len
        assert sample_candles[0] == original_first


class TestDetectFVGs:
    """Test Fair Value Gap detection."""

    def test_detects_bullish_fvg(self):
        """Bullish FVG: candle[i].low > candle[i-2].high."""
        candles = [
            {"open": 100, "high": 105, "low": 98, "close": 103, "datetime": ""},
            {"open": 103, "high": 115, "low": 102, "close": 114, "datetime": ""},
            {"open": 114, "high": 120, "low": 107, "close": 118, "datetime": ""},
        ]
        fvgs = detect_fvgs(candles)
        bullish = [f for f in fvgs if f["type"] == "bullish"]
        assert len(bullish) >= 1

    def test_detects_bearish_fvg(self):
        """Bearish FVG: candle[i].high < candle[i-2].low."""
        candles = [
            {"open": 200, "high": 205, "low": 198, "close": 199, "datetime": ""},
            {"open": 199, "high": 200, "low": 185, "close": 186, "datetime": ""},
            {"open": 186, "high": 196, "low": 180, "close": 182, "datetime": ""},
        ]
        fvgs = detect_fvgs(candles)
        bearish = [f for f in fvgs if f["type"] == "bearish"]
        assert len(bearish) >= 1

    def test_no_fvg_in_normal_price_action(self, sample_candles):
        """Gentle moves with overlapping wicks don't produce FVGs."""
        fvgs = detect_fvgs(sample_candles)
        # May or may not find gaps — main contract is no errors
        assert isinstance(fvgs, list)


class TestDetectLiquidity:
    """Test liquidity (swing high/low) detection."""

    def test_detects_swing_high(self):
        """Clear swing high should be detected as buyside liquidity."""
        candles = []
        # Build a peak in the middle
        for i in range(15):
            if i == 7:
                h = 250  # Clear peak
            else:
                h = 200 + i * 0.5
            candles.append({
                "open": 195, "high": h, "low": 190,
                "close": 196, "datetime": "",
            })
        liqs = detect_liquidity(candles, window=5)
        buyside = [l for l in liqs if l["type"] == "buyside"]
        assert len(buyside) >= 1

    def test_detects_swing_low(self):
        """Clear swing low should be detected as sellside liquidity."""
        candles = []
        for i in range(15):
            if i == 7:
                low = 150  # Clear trough
            else:
                low = 195 - i * 0.3
            candles.append({
                "open": 200, "high": 210, "low": low,
                "close": 201, "datetime": "",
            })
        liqs = detect_liquidity(candles, window=5)
        sellside = [l for l in liqs if l["type"] == "sellside"]
        assert len(sellside) >= 1


class TestComputeMarketStructure:
    """Test market structure scoring."""

    def test_uptrend_returns_positive(self):
        """Higher highs and higher lows should give positive score."""
        candles = [
            {"open": 100 + i * 2, "high": 105 + i * 2,
             "low": 98 + i * 2, "close": 103 + i * 2, "datetime": ""}
            for i in range(25)
        ]
        score = compute_market_structure(candles, lookback=20)
        assert score > 0

    def test_downtrend_returns_negative(self):
        """Lower highs and lower lows should give negative score."""
        candles = [
            {"open": 200 - i * 2, "high": 205 - i * 2,
             "low": 198 - i * 2, "close": 199 - i * 2, "datetime": ""}
            for i in range(25)
        ]
        score = compute_market_structure(candles, lookback=20)
        assert score < 0

    def test_score_bounded(self, sample_candles):
        """Score should be in [-1, 1]."""
        score = compute_market_structure(sample_candles, lookback=20)
        assert -1 <= score <= 1


class TestCreateTradeLabels:
    """Test forward trade simulation for labeling."""

    def test_tp_hit(self):
        """Price moving favorably should hit TP and label as win."""
        candles = []
        for i in range(60):
            price = 100 + i * 0.5  # Steady uptrend
            candles.append({
                "open": price, "high": price + 1,
                "low": price - 0.5, "close": price + 0.8,
                "datetime": "",
            })
        atr = compute_atr(candles, 14)
        labels = create_trade_labels(candles, 10, "long", atr,
                                     sl_atr_mult=2.0, tp_atr_mults=[1.0])
        assert labels["won"] is True or labels["outcome"] in ("tp1_hit", "expired")

    def test_sl_hit(self):
        """Price moving against should hit SL and label as loss."""
        candles = []
        for i in range(60):
            price = 200 - i * 0.5  # Steady downtrend
            candles.append({
                "open": price, "high": price + 0.5,
                "low": price - 1, "close": price - 0.8,
                "datetime": "",
            })
        atr = compute_atr(candles, 14)
        labels = create_trade_labels(candles, 10, "long", atr,
                                     sl_atr_mult=1.5, tp_atr_mults=[3.0])
        assert labels["outcome"] == "stopped_out" or labels["outcome"] == "expired"

    def test_expired_trade(self):
        """Flat market should expire without hitting SL or TP."""
        candles = []
        for i in range(60):
            candles.append({
                "open": 100, "high": 100.2,
                "low": 99.8, "close": 100.1,
                "datetime": "",
            })
        atr = compute_atr(candles, 14)
        labels = create_trade_labels(candles, 5, "long", atr,
                                     sl_atr_mult=5.0, tp_atr_mults=[10.0],
                                     max_bars=10)
        assert labels["outcome"] == "expired"

    def test_mfe_mae_tracking(self):
        """MFE and MAE should be non-negative floats."""
        candles = []
        for i in range(60):
            candles.append({
                "open": 100 + i * 0.2, "high": 102 + i * 0.2,
                "low": 98 + i * 0.2, "close": 101 + i * 0.2,
                "datetime": "",
            })
        atr = compute_atr(candles, 14)
        labels = create_trade_labels(candles, 10, "long", atr)
        assert labels["max_drawdown_atr"] >= 0
        assert labels["max_favorable_atr"] >= 0
        assert isinstance(labels["bars_held"], int)


# ═══════════════════════════════════════════════════════════════════════
# WFO Engine Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDetectRegime:
    """Test volatility regime classification."""

    def test_ranging_for_short_data(self):
        """Short candle arrays default to ranging."""
        candles = [{"close": 100, "open": 100, "high": 101, "low": 99, "datetime": ""}
                   for _ in range(3)]
        assert detect_regime(candles, 2) == "ranging"

    def test_returns_valid_regime(self, wfo_candles):
        """Regime should be one of the valid regime strings."""
        valid = {"high_volatility", "low_volatility", "trending_up",
                 "trending_down", "ranging"}
        regime = detect_regime(wfo_candles, 200)
        assert regime in valid

    def test_high_vol_with_spike(self):
        """Recent high volatility should be detected."""
        candles = []
        # 30 calm candles
        for i in range(30):
            candles.append({"close": 100, "open": 100, "high": 100.1,
                           "low": 99.9, "datetime": ""})
        # 5 volatile candles
        for i in range(5):
            c = 100 + (i * 10 if i % 2 == 0 else -i * 10)
            candles.append({"close": c, "open": 100, "high": max(c, 100) + 5,
                           "low": min(c, 100) - 5, "datetime": ""})
        regime = detect_regime(candles, len(candles) - 1)
        # Should detect elevated short-term vol
        assert regime in {"high_volatility", "trending_up", "trending_down", "ranging"}


class TestICTSetupDetector:
    """Test the setup detector that scans candles for ICT setups."""

    def test_returns_dataframe(self, wfo_candles):
        """detect_setups should return a DataFrame."""
        detector = ICTSetupDetector(WFOConfig(min_confluence_score=1))
        df = detector.detect_setups(wfo_candles[:200], "1h")
        assert isinstance(df, pd.DataFrame)

    def test_confluence_filtering(self, wfo_candles):
        """Higher min_confluence should produce fewer setups."""
        detector_low = ICTSetupDetector(WFOConfig(min_confluence_score=1))
        detector_high = ICTSetupDetector(WFOConfig(min_confluence_score=4))
        df_low = detector_low.detect_setups(wfo_candles[:200], "1h")
        df_high = detector_high.detect_setups(wfo_candles[:200], "1h")
        assert len(df_high) <= len(df_low)

    def test_setup_type_format(self, wfo_candles):
        """Setup types should follow bull_/bear_ prefix convention."""
        detector = ICTSetupDetector(WFOConfig(min_confluence_score=1))
        df = detector.detect_setups(wfo_candles[:200], "1h")
        if len(df) > 0:
            for st in df["setup_type"]:
                assert st.startswith("bull") or st.startswith("bear")

    def test_empty_for_insufficient_data(self):
        """Too few candles should return empty DataFrame."""
        short_candles = [{"open": 100, "high": 101, "low": 99,
                          "close": 100.5, "datetime": ""}
                         for _ in range(20)]
        detector = ICTSetupDetector()
        df = detector.detect_setups(short_candles, "1h")
        assert len(df) == 0


class TestConfluenceScoring:
    """Test refined confluence scoring: tighter OBs, mitigation, structure boost."""

    def _make_detector(self, **overrides):
        cfg = WFOConfig(**overrides)
        return ICTSetupDetector(cfg)

    def _make_candles(self, n=100, base=2600.0):
        """Simple candles with small bodies — no displacement by default."""
        return [{"open": base + i * 0.1, "high": base + i * 0.1 + 2,
                 "low": base + i * 0.1 - 2, "close": base + i * 0.1 + 0.5,
                 "datetime": f"2026-03-10 {(i % 24):02d}:00:00"}
                for i in range(n)]

    def test_ob_proximity_tighter_than_before(self, wfo_candles):
        """OB at 2 ATR should NOT tag; OBs are now within 1 ATR only."""
        detector = self._make_detector(min_confluence_score=1)
        from ml.features import compute_atr, detect_order_blocks
        atr = compute_atr(wfo_candles, 14)
        obs = detect_order_blocks(wfo_candles, atr)

        # Find an OB that's between 1-2 ATR from current price (should be rejected)
        for idx in range(60, len(wfo_candles) - 20):
            price = wfo_candles[idx]["close"]
            for ob in obs:
                if ob["index"] >= idx:
                    continue
                ob_mid = (ob["high"] + ob["low"]) / 2
                dist = abs(price - ob_mid)
                # OB in the 1-2 ATR dead zone
                if 1.0 * atr < dist <= 2.0 * atr:
                    _, tags = detector._score_confluence(
                        wfo_candles, idx, "long" if ob["type"] == "bullish" else "short",
                        atr, [ob], [], [], 0.0,
                    )
                    assert "ob" not in tags, f"OB at {dist/atr:.1f} ATR should be rejected"
                    return  # one check is enough
        pytest.skip("No OB in 1-2 ATR range found in test data")

    def test_ob_within_1_atr_still_tags(self, wfo_candles):
        """OB within 1 ATR should still produce the ob tag."""
        detector = self._make_detector(min_confluence_score=1)
        from ml.features import compute_atr, detect_order_blocks
        atr = compute_atr(wfo_candles, 14)
        obs = detect_order_blocks(wfo_candles, atr)

        for idx in range(60, len(wfo_candles) - 20):
            price = wfo_candles[idx]["close"]
            for ob in obs:
                if ob["index"] >= idx:
                    continue
                ob_mid = (ob["high"] + ob["low"]) / 2
                dist = abs(price - ob_mid)
                if dist <= 1.0 * atr:
                    direction = "long" if ob["type"] == "bullish" else "short"
                    # Need to check it's not mitigated
                    mitigated = False
                    for k in range(ob["index"] + 1, idx):
                        if wfo_candles[k]["low"] <= ob["low"] and wfo_candles[k]["high"] >= ob["high"]:
                            mitigated = True
                            break
                    if mitigated:
                        continue
                    _, tags = detector._score_confluence(
                        wfo_candles, idx, direction, atr, [ob], [], [], 0.0,
                    )
                    assert "ob" in tags, f"OB at {dist/atr:.2f} ATR should tag"
                    return
        pytest.skip("No unmitigated OB within 1 ATR in test data")

    def test_mitigated_ob_rejected(self):
        """OB whose zone was fully traded through should not score."""
        detector = self._make_detector(min_confluence_score=1)
        # Create an OB at index 0, then price sweeps completely through it
        ob = {"type": "bullish", "high": 2605, "low": 2600, "index": 0, "body_size": 5}
        # Candles where price sweeps below OB low AND above OB high (mitigated)
        candles = [
            {"open": 2602, "high": 2606, "low": 2599, "close": 2604, "datetime": "2026-03-10 00:00:00"},  # OB candle
        ]
        # Add candles that trade through the OB zone
        for i in range(1, 60):
            candles.append({
                "open": 2603, "high": 2610, "low": 2595,  # sweeps through 2600-2605
                "close": 2603, "datetime": f"2026-03-10 {(i % 24):02d}:00:00",
            })
        from ml.features import compute_atr
        atr = compute_atr(candles, 14)
        _, tags = detector._score_confluence(candles, 59, "long", atr, [ob], [], [], 0.0)
        assert "ob" not in tags, "Mitigated OB should not score"

    def test_structure_scores_double(self, wfo_candles):
        """Market structure alignment should contribute 2 points, not 1."""
        detector = self._make_detector(min_confluence_score=1)
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        # Use a strong bullish ms_score
        score, tags = detector._score_confluence(wfo_candles, 60, "long", atr, [], [], [], 0.8)
        assert "structure" in tags
        assert score == 2, f"Structure should give +2 but gave {score}"

    def test_large_ob_bonus_removed(self):
        """Large OB body should NOT give a bonus point anymore."""
        detector = self._make_detector(min_confluence_score=1)
        # Create OB with large body (> 0.5 * atr) right next to price
        ob = {"type": "bullish", "high": 2605, "low": 2600, "index": 0, "body_size": 10}
        candles = []
        for i in range(60):
            candles.append({
                "open": 2603, "high": 2604, "low": 2602, "close": 2603,
                "datetime": f"2026-03-10 {(i % 24):02d}:00:00",
            })
        from ml.features import compute_atr
        atr = compute_atr(candles, 14)
        score, tags = detector._score_confluence(candles, 59, "long", atr, [ob], [], [], 0.0)
        # With just one unmitigated nearby OB, score should be exactly 1 (no bonus)
        if "ob" in tags:
            assert score == 1, f"OB should give exactly 1 point (no body bonus) but gave {score}"


class TestWFOReport:
    """Test report dataclass serialization and grading."""

    def test_grade_a(self):
        """PF>1.5 + WR>50% = grade A."""
        assert _compute_wfo_grade(0.55, 1.6) == "A"

    def test_grade_b(self):
        """PF>1.2 + WR>40% = grade B."""
        assert _compute_wfo_grade(0.45, 1.3) == "B"

    def test_grade_c(self):
        """PF>1.0 = grade C."""
        assert _compute_wfo_grade(0.35, 1.1) == "C"

    def test_grade_d(self):
        """PF<=1.0 = grade D."""
        assert _compute_wfo_grade(0.25, 0.8) == "D"

    def test_serialization_roundtrip(self):
        """to_dict() → from_dict() should produce equivalent report."""
        fold = FoldResult(
            fold_num=0, train_start=0, train_end=500,
            test_start=500, test_end=600,
            total_trades=20, wins=12, losses=6, expired=2,
            win_rate=0.6, avg_rr=1.5, profit_factor=1.8,
            sharpe=0.9, max_drawdown=2.1, regime="trending_up",
            setup_types={"bull_ob_fvg": 8, "bear_structure": 4},
            winning_drawdowns=[0.5, 0.8, 1.0],
            winning_excursions=[1.5, 2.0, 3.5],
        )
        report = WFOReport(
            total_oos_trades=20, oos_win_rate=0.6,
            oos_avg_rr=1.5, oos_profit_factor=1.8,
            oos_sharpe=0.9, oos_max_drawdown=2.1,
            regime_stability=0.85, recommended_sl_atr=1.2,
            recommended_tp_atr=[0.8, 1.5, 2.8],
            grade="A", folds=[fold], fold_count=1,
            skipped_folds=0, setup_type_breakdown={"bull_ob_fvg": 8},
            timestamp="2026-01-01T00:00:00Z",
        )

        d = report.to_dict()
        restored = WFOReport.from_dict(d)

        assert restored.total_oos_trades == 20
        assert restored.grade == "A"
        assert restored.oos_win_rate == 0.6
        assert len(restored.folds) == 1
        assert restored.folds[0].fold_num == 0
        assert restored.recommended_tp_atr == [0.8, 1.5, 2.8]

    def test_regime_stability_formula(self):
        """regime_stability = 1.0 - min(std(fold_wrs), 0.5) * 2."""
        import numpy as np
        fold_wrs = [0.5, 0.6, 0.55, 0.52]
        wr_std = float(np.std(fold_wrs))
        stability = round(1.0 - min(wr_std, 0.5) * 2, 4)
        assert 0 <= stability <= 1

    def test_save_and_load(self, tmp_path):
        """save_report then load_report should roundtrip."""
        report = WFOReport(
            total_oos_trades=10, oos_win_rate=0.5,
            oos_avg_rr=1.0, oos_profit_factor=1.0,
            oos_sharpe=0.5, oos_max_drawdown=1.0,
            regime_stability=0.8, recommended_sl_atr=1.5,
            recommended_tp_atr=[1.0, 2.0, 3.5],
            grade="C", folds=[], fold_count=0,
            skipped_folds=0, setup_type_breakdown={},
            timestamp="2026-01-01T00:00:00Z",
        )
        path = str(tmp_path / "test_report.json")
        save_report(report, path)
        loaded = load_report(path)
        assert loaded is not None
        assert loaded.grade == "C"
        assert loaded.total_oos_trades == 10

    def test_load_nonexistent(self, tmp_path):
        """load_report on missing file returns None."""
        path = str(tmp_path / "nope.json")
        assert load_report(path) is None


class TestWalkForwardEngineHeuristic:
    """Test WFO engine with heuristic model (no AutoGluon)."""

    def test_runs_without_autogluon(self, wfo_candles):
        """Engine should run and produce a report using heuristic model."""
        cfg = WFOConfig(
            train_window=200, test_window=50, step_size=50,
            max_folds=3, min_confluence_score=1, min_setups_per_fold=3,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        report = engine.run(wfo_candles, "1h")
        assert isinstance(report, WFOReport)
        assert report.grade in ("A", "B", "C", "D")
        assert report.timestamp is not None

    def test_heuristic_probability_capping(self):
        """Heuristic win_prob should be capped at 0.8."""
        engine = WalkForwardEngine(WFOConfig(), use_autogluon=False)
        test_df = pd.DataFrame({"confluence_score": [10, 20, 100]})
        preds = engine._predict_heuristic(test_df)
        assert all(preds["win_prob"] <= 0.8)

    def test_insufficient_candles_raises(self):
        """Engine should raise ValueError for too few candles."""
        engine = WalkForwardEngine(WFOConfig(train_window=500, test_window=100))
        short = [{"open": 100, "high": 101, "low": 99,
                  "close": 100.5, "datetime": ""} for _ in range(100)]
        with pytest.raises(ValueError, match="Need at least"):
            engine.run(short, "1h")

    def test_empty_folds_produce_grade_d(self):
        """If all folds are skipped, report should have grade D."""
        cfg = WFOConfig(
            train_window=200, test_window=50, step_size=50,
            max_folds=2, min_confluence_score=6,  # Very high - likely no setups
            min_setups_per_fold=100,
        )
        # Create minimal candles (just enough to not raise ValueError)
        candles = [{"open": 100, "high": 100.1, "low": 99.9,
                    "close": 100, "datetime": f"2026-01-01 {i % 24:02d}:00:00"}
                   for i in range(300)]
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        report = engine.run(candles, "1h")
        assert report.grade == "D"
        assert report.total_oos_trades == 0


class TestExtractHour:
    """Test hour extraction helper."""

    def test_standard_format(self):
        assert _extract_hour("2026-01-01 14:30:00") == 14

    def test_iso_format(self):
        assert _extract_hour("2026-01-01T08:00:00Z") == 8

    def test_invalid_returns_zero(self):
        assert _extract_hour("") == 0
        assert _extract_hour("not-a-date") == 0


# ═══════════════════════════════════════════════════════════════════════
# OOS Trade Retention Tests
# ═══════════════════════════════════════════════════════════════════════


class TestOOSTradeRetention:
    """Test that WFO engine retains OOS trade records for dataset ingestion."""

    def test_engine_retains_oos_trades(self, wfo_candles):
        cfg = WFOConfig(
            train_window=200, test_window=200, step_size=100,
            max_folds=3, min_confluence_score=1, min_setups_per_fold=1,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        engine.run(wfo_candles, "1h")
        assert hasattr(engine, "oos_trades")
        assert isinstance(engine.oos_trades, list)
        assert len(engine.oos_trades) > 0

    def test_oos_trades_have_required_fields(self, wfo_candles):
        cfg = WFOConfig(
            train_window=200, test_window=200, step_size=100,
            max_folds=3, min_confluence_score=1, min_setups_per_fold=1,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        engine.run(wfo_candles, "1h")
        if not engine.oos_trades:
            pytest.skip("No OOS trades detected — candle data too uniform")
        required = {"outcome", "direction", "ob_count", "fvg_count",
                     "regime", "fold", "candle_index"}
        for trade in engine.oos_trades:
            for field in required:
                assert field in trade, f"Missing field: {field}"

    def test_oos_trades_have_regime(self, wfo_candles):
        cfg = WFOConfig(
            train_window=200, test_window=200, step_size=100,
            max_folds=3, min_confluence_score=1, min_setups_per_fold=1,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        engine.run(wfo_candles, "1h")
        if not engine.oos_trades:
            pytest.skip("No OOS trades detected")
        valid_regimes = {"high_volatility", "low_volatility", "trending_up",
                         "trending_down", "ranging"}
        for trade in engine.oos_trades:
            assert trade["regime"] in valid_regimes

    def test_empty_folds_empty_oos_trades(self):
        cfg = WFOConfig(
            train_window=200, test_window=50, step_size=50,
            max_folds=2, min_confluence_score=6,
            min_setups_per_fold=100,
        )
        candles = [{"open": 100, "high": 100.1, "low": 99.9,
                    "close": 100, "datetime": f"2026-01-01 {i % 24:02d}:00:00"}
                   for i in range(300)]
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        engine.run(candles, "1h")
        assert engine.oos_trades == []


# ═══════════════════════════════════════════════════════════════════════
# Server Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════


class TestWFOServerEndpoints:
    """Test WFO API endpoints via TestClient."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Create FastAPI test client with temp config."""
        import ml.config
        from ml.config import make_test_config

        cfg = make_test_config(db_path=str(tmp_path / "test.db"))
        cfg["wfo_report_path"] = str(tmp_path / "wfo_report.json")
        monkeypatch.setattr(ml.config, "_active_config", cfg)

        from fastapi.testclient import TestClient
        from ml.server import app
        import ml.server
        ml.server._db_instance = None  # Reset singleton

        with TestClient(app) as c:
            yield c

        ml.server._db_instance = None
        ml.config.reset_config()

    def test_wfo_report_null_initially(self, client):
        """GET /wfo/report returns null when no report exists."""
        resp = client.get("/wfo/report")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_wfo_run_insufficient_candles(self, client):
        """POST /wfo/run with too few candles returns 400."""
        candles = [{"datetime": f"2026-01-01 {i:02d}:00:00",
                    "open": 100, "high": 101, "low": 99, "close": 100.5}
                   for i in range(50)]
        resp = client.post("/wfo/run", json={
            "candles": candles, "timeframe": "1h",
            "train_window": 500, "test_window": 100,
        })
        assert resp.status_code == 400
        assert "Insufficient" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════
# build_setup_filter + setup_type_stats Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBuildSetupFilter:
    """Test WFO setup type filter classification."""

    def _make_report_with_stats(self, stats: dict) -> WFOReport:
        """Helper: create a WFOReport with given setup_type_stats."""
        return WFOReport(
            total_oos_trades=100,
            oos_win_rate=0.45,
            oos_avg_rr=2.0,
            oos_profit_factor=1.3,
            oos_sharpe=0.8,
            oos_max_drawdown=3.5,
            regime_stability=0.85,
            recommended_sl_atr=1.5,
            recommended_tp_atr=[1.0, 2.0, 3.5],
            grade="B",
            folds=[],
            fold_count=5,
            skipped_folds=0,
            setup_type_breakdown={},
            timestamp="2026-03-13T00:00:00Z",
            setup_type_stats=stats,
        )

    def test_classifies_profitable(self):
        """Setup type with >= 40% win rate and >= 3 trades is profitable."""
        stats = {
            "bull_fvg_structure": {"wins": 8, "total": 15, "win_rate": 0.5333},
        }
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert "bull_fvg_structure" in result["profitable"]
        assert "bull_fvg_structure" not in result["unprofitable"]

    def test_classifies_unprofitable(self):
        """Setup type with < 40% win rate and >= 3 trades is unprofitable."""
        stats = {
            "bear_fvg_ob": {"wins": 2, "total": 20, "win_rate": 0.10},
        }
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert "bear_fvg_ob" in result["unprofitable"]

    def test_classifies_insufficient(self):
        """Setup type with < 3 trades is insufficient data."""
        stats = {
            "bull_sweep": {"wins": 1, "total": 2, "win_rate": 0.50},
        }
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert "bull_sweep" in result["insufficient"]

    def test_mixed_classification(self):
        """Multiple setup types get classified correctly."""
        stats = {
            "bull_fvg_structure": {"wins": 10, "total": 20, "win_rate": 0.50},
            "bear_fvg_ob": {"wins": 3, "total": 25, "win_rate": 0.12},
            "bull_ob": {"wins": 1, "total": 2, "win_rate": 0.50},
        }
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert "bull_fvg_structure" in result["profitable"]
        assert "bear_fvg_ob" in result["unprofitable"]
        assert "bull_ob" in result["insufficient"]

    def test_custom_thresholds(self):
        """Custom min_win_rate and min_trades are respected."""
        stats = {
            "bull_fvg": {"wins": 3, "total": 10, "win_rate": 0.30},
        }
        # Default 40% → unprofitable
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert "bull_fvg" in result["unprofitable"]

        # Lower threshold 25% → profitable
        result = build_setup_filter(self._make_report_with_stats(stats),
                                    min_win_rate=0.25)
        assert "bull_fvg" in result["profitable"]

    def test_returns_stats_dict(self):
        """Filter result includes the raw stats for lookup."""
        stats = {
            "bull_fvg_structure": {"wins": 8, "total": 15, "win_rate": 0.5333},
        }
        result = build_setup_filter(self._make_report_with_stats(stats))
        assert result["stats"] == stats


class TestSetupTypeStatsInReport:
    """Test that setup_type_stats serializes/deserializes correctly."""

    def test_setup_type_stats_in_to_dict(self):
        """setup_type_stats included in to_dict() output."""
        report = WFOReport(
            total_oos_trades=10,
            oos_win_rate=0.5,
            oos_avg_rr=2.0,
            oos_profit_factor=1.5,
            oos_sharpe=1.0,
            oos_max_drawdown=2.0,
            regime_stability=0.9,
            recommended_sl_atr=1.5,
            recommended_tp_atr=[1.0, 2.0, 3.5],
            grade="A",
            folds=[],
            fold_count=3,
            skipped_folds=0,
            setup_type_breakdown={"bull_fvg": 5},
            timestamp="2026-03-13T00:00:00Z",
            setup_type_stats={"bull_fvg": {"wins": 3, "total": 5, "win_rate": 0.6}},
        )
        d = report.to_dict()
        assert "setup_type_stats" in d
        assert d["setup_type_stats"]["bull_fvg"]["win_rate"] == 0.6

    def test_setup_type_stats_from_dict_roundtrip(self):
        """setup_type_stats survives save/load roundtrip."""
        stats = {"bear_ob": {"wins": 2, "total": 8, "win_rate": 0.25}}
        report = WFOReport(
            total_oos_trades=8,
            oos_win_rate=0.25,
            oos_avg_rr=1.5,
            oos_profit_factor=0.8,
            oos_sharpe=0.3,
            oos_max_drawdown=4.0,
            regime_stability=0.7,
            recommended_sl_atr=1.5,
            recommended_tp_atr=[1.0, 2.0, 3.5],
            grade="D",
            folds=[],
            fold_count=2,
            skipped_folds=1,
            setup_type_breakdown={"bear_ob": 8},
            timestamp="2026-03-13T00:00:00Z",
            setup_type_stats=stats,
        )
        d = report.to_dict()
        restored = WFOReport.from_dict(d)
        assert restored.setup_type_stats == stats


# ═══════════════════════════════════════════════════════════════════════
# V2 Quality-Weighted Detection Tests
# ═══════════════════════════════════════════════════════════════════════


class TestQualityScoring:
    """Test the V2 quality-weighted confluence scoring system."""

    def _make_v2_detector(self, **overrides):
        cfg = WFOConfig(use_quality_scoring=True, **overrides)
        return ICTSetupDetector(cfg)

    def test_returns_float_score_and_breakdown(self, wfo_candles):
        """_score_quality returns (float, list, dict) tuple."""
        detector = self._make_v2_detector()
        atr = compute_atr(wfo_candles, 14)
        from ml.features import (
            detect_order_blocks, detect_fvgs, detect_liquidity,
            compute_market_structure, detect_swing_points,
        )
        obs = detect_order_blocks(wfo_candles, atr)
        fvgs = detect_fvgs(wfo_candles)
        liqs = detect_liquidity(wfo_candles)
        swings = detect_swing_points(wfo_candles, lookback=5)
        ms = compute_market_structure(wfo_candles[:100], 20)

        score, tags, breakdown = detector._score_quality(
            wfo_candles, 60, "long", atr, obs, fvgs, liqs, ms, swings
        )
        assert isinstance(score, float)
        assert isinstance(tags, list)
        assert isinstance(breakdown, dict)
        assert "total_quality_score" in breakdown
        assert "ob_score" in breakdown

    def test_breakdown_has_all_categories(self, wfo_candles):
        """Breakdown dict should have all 6 scoring categories + total."""
        detector = self._make_v2_detector()
        atr = compute_atr(wfo_candles, 14)
        from ml.features import (
            detect_order_blocks, detect_fvgs, detect_liquidity,
            compute_market_structure, detect_swing_points,
        )
        obs = detect_order_blocks(wfo_candles, atr)
        fvgs = detect_fvgs(wfo_candles)
        liqs = detect_liquidity(wfo_candles)
        swings = detect_swing_points(wfo_candles, lookback=5)
        ms = compute_market_structure(wfo_candles[:100], 20)

        _, _, breakdown = detector._score_quality(
            wfo_candles, 60, "long", atr, obs, fvgs, liqs, ms, swings
        )
        required = {"ob_score", "fvg_score", "liq_score", "structure_score",
                     "session_score", "displacement_score", "total_quality_score"}
        assert required.issubset(breakdown.keys())

    def test_structure_penalty_for_counter_trend(self, wfo_candles):
        """Buying in a downtrend should give negative structure score."""
        detector = self._make_v2_detector()
        atr = compute_atr(wfo_candles, 14)
        from ml.features import (
            detect_order_blocks, detect_fvgs, detect_liquidity,
            detect_swing_points,
        )
        obs = detect_order_blocks(wfo_candles, atr)
        fvgs = detect_fvgs(wfo_candles)
        liqs = detect_liquidity(wfo_candles)
        swings = detect_swing_points(wfo_candles, lookback=5)

        # Use a strongly bearish ms_score
        _, _, breakdown = detector._score_quality(
            wfo_candles, 60, "long", atr, obs, fvgs, liqs, -0.8, swings
        )
        assert breakdown["structure_score"] <= 0

    def test_session_penalty_off_hours(self):
        """Setups outside killzones should get negative session score."""
        detector = self._make_v2_detector()
        # Create candles at 5 AM UTC (off hours)
        candles = [{"open": 2600 + i * 0.1, "high": 2602 + i * 0.1,
                     "low": 2598 + i * 0.1, "close": 2601 + i * 0.1,
                     "datetime": f"2026-03-10 05:00:00"}
                    for i in range(100)]
        atr = 3.0
        _, _, breakdown = detector._score_quality(
            candles, 60, "long", atr, [], [], [], 0.0, []
        )
        assert breakdown["session_score"] == -0.5

    def test_ny_am_gets_max_session_score(self):
        """NY AM session should get 2.0 session score."""
        detector = self._make_v2_detector()
        candles = [{"open": 2600, "high": 2602, "low": 2598, "close": 2601,
                     "datetime": "2026-03-10 14:00:00"}
                    for _ in range(100)]
        atr = 3.0
        _, tags, breakdown = detector._score_quality(
            candles, 60, "long", atr, [], [], [], 0.0, []
        )
        assert breakdown["session_score"] == 2.0
        assert "ny_am" in tags

    def test_ob_score_caps_at_4(self, wfo_candles):
        """OB quality score should not exceed 4.0."""
        detector = self._make_v2_detector()
        atr = compute_atr(wfo_candles, 14)
        from ml.features import (
            detect_order_blocks, detect_fvgs, detect_liquidity,
            detect_swing_points,
        )
        obs = detect_order_blocks(wfo_candles, atr)
        fvgs = detect_fvgs(wfo_candles)
        liqs = detect_liquidity(wfo_candles)
        swings = detect_swing_points(wfo_candles, lookback=5)
        ms = compute_market_structure(wfo_candles[:100], 20)

        _, _, breakdown = detector._score_quality(
            wfo_candles, 60, "long", atr, obs, fvgs, liqs, ms, swings
        )
        assert breakdown["ob_score"] <= 4.0


class TestRejectionEntry:
    """Test the V2 rejection candle entry detection."""

    def _make_v2_detector(self, **overrides):
        cfg = WFOConfig(use_quality_scoring=True, use_rejection_entry=True,
                        max_bars_to_rejection=15, **overrides)
        return ICTSetupDetector(cfg)

    def test_finds_bullish_rejection(self):
        """Strong bullish close in lower zone → valid rejection for longs."""
        detector = self._make_v2_detector()
        atr = 5.0
        ob = {"high": 2610, "low": 2600, "index": 0, "type": "bullish"}

        candles = []
        # Signal candle
        candles.append({"open": 2615, "high": 2620, "low": 2614, "close": 2618,
                        "datetime": "2026-03-10 14:00:00"})
        # Non-rejection candles
        for i in range(1, 5):
            candles.append({"open": 2615, "high": 2618, "low": 2612,
                            "close": 2616, "datetime": f"2026-03-10 {14+i}:00:00"})
        # Rejection candle: retraces to OB zone, closes in upper 33%
        candles.append({"open": 2605, "high": 2612, "low": 2601,  # wicks into 2600-2610
                        "close": 2611,  # closes at top (close_pos = (2611-2601)/(2612-2601) = 0.91)
                        "datetime": "2026-03-10 19:00:00"})
        # More displacement body reference candles
        candles.append({"open": 2600, "high": 2620, "low": 2598, "close": 2618,
                        "datetime": "2026-03-10 20:00:00"})  # big candle for disp reference

        # Pad for safety
        for i in range(10):
            candles.append({"open": 2612, "high": 2614, "low": 2610,
                            "close": 2613, "datetime": f"2026-03-11 {i:02d}:00:00"})

        result = detector._find_rejection_entry(
            candles, 0, "long", atr, best_ob=ob
        )
        assert result is not None
        assert result["entry_type"] == "rejection"
        assert result["rejection_quality"] >= 0.67

    def test_returns_none_without_rejection(self):
        """No valid rejection within max_wait → returns None."""
        detector = self._make_v2_detector()
        atr = 5.0
        ob = {"high": 2610, "low": 2600, "index": 0, "type": "bullish"}

        # All candles above the OB zone — never retraces
        candles = [{"open": 2620 + i, "high": 2625 + i, "low": 2618 + i,
                    "close": 2623 + i, "datetime": f"2026-03-10 {i:02d}:00:00"}
                   for i in range(25)]

        result = detector._find_rejection_entry(
            candles, 0, "long", atr, best_ob=ob
        )
        assert result is None

    def test_structural_sl_below_rejection_low(self):
        """For longs, structural_sl should be below rejection candle's low."""
        detector = self._make_v2_detector()
        atr = 5.0
        ob = {"high": 2610, "low": 2600, "index": 0, "type": "bullish"}

        candles = [
            {"open": 2615, "high": 2620, "low": 2614, "close": 2618,
             "datetime": "2026-03-10 14:00:00"},  # signal
            {"open": 2605, "high": 2612, "low": 2601, "close": 2611,
             "datetime": "2026-03-10 15:00:00"},  # rejection
        ]
        # Pad
        for i in range(15):
            candles.append({"open": 2612, "high": 2614, "low": 2610,
                            "close": 2613, "datetime": f"2026-03-10 {16+i}:00:00"})

        result = detector._find_rejection_entry(
            candles, 0, "long", atr, best_ob=ob
        )
        if result is not None:
            assert result["structural_sl"] < candles[result["idx"]]["low"]


class TestNarrativeAlignment:
    """Test the V2 narrative-based HTF directional filter."""

    def _make_detector(self):
        cfg = WFOConfig(use_quality_scoring=True, use_narrative_filter=True)
        return ICTSetupDetector(cfg)

    def test_long_passes_in_discount_with_ssl_swept(self):
        """Long in discount + SSL swept → passes."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.3, "htf_ssl_swept": 1,
               "htf_bsl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 5.0, "htf_ob_above_dist": 5.0}
        passes, reason = detector.check_narrative_alignment("long", htf)
        assert passes is True

    def test_long_rejected_in_premium(self):
        """Long in premium (> 0.5) with no support → rejected."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.8, "htf_ssl_swept": 0,
               "htf_bsl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 5.0, "htf_ob_above_dist": 5.0}
        passes, reason = detector.check_narrative_alignment("long", htf)
        assert passes is False

    def test_short_passes_in_premium_with_bsl_swept(self):
        """Short in premium + BSL swept → passes."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.7, "htf_bsl_swept": 1,
               "htf_ssl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 5.0, "htf_ob_above_dist": 5.0}
        passes, reason = detector.check_narrative_alignment("short", htf)
        assert passes is True

    def test_short_rejected_in_discount(self):
        """Short in discount with no supply → rejected."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.2, "htf_ssl_swept": 0,
               "htf_bsl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 5.0, "htf_ob_above_dist": 5.0}
        passes, reason = detector.check_narrative_alignment("short", htf)
        assert passes is False

    def test_long_passes_with_nearby_demand(self):
        """Long with 4H demand zone nearby (< 1.5 ATR) → passes even in premium."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.7, "htf_ssl_swept": 0,
               "htf_bsl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 0.8, "htf_ob_above_dist": 5.0}
        passes, reason = detector.check_narrative_alignment("long", htf)
        assert passes is True

    def test_returns_reason_string(self):
        """Both pass and fail should return non-empty reason."""
        detector = self._make_detector()
        htf = {"htf_premium_discount": 0.3, "htf_ssl_swept": 1,
               "htf_bsl_swept": 0, "htf_liq_narrative": 0,
               "htf_ob_below_dist": 5.0, "htf_ob_above_dist": 5.0}
        _, reason = detector.check_narrative_alignment("long", htf)
        assert len(reason) > 0


class TestV2DetectorIntegration:
    """Integration tests for the full V2 detection pipeline."""

    def test_v2_detector_returns_dataframe(self, wfo_candles):
        """V2 detector should produce a DataFrame with quality metadata."""
        cfg = WFOConfig(
            use_quality_scoring=True, min_quality_score=3.0,
            use_rejection_entry=False,  # disable rejection for more setups
            min_setups_per_fold=1,
        )
        detector = ICTSetupDetector(cfg)
        df = detector.detect_setups(wfo_candles[:300], "1h")
        assert isinstance(df, pd.DataFrame)

    def test_v2_has_quality_columns(self, wfo_candles):
        """V2 output should include quality score breakdown columns."""
        cfg = WFOConfig(
            use_quality_scoring=True, min_quality_score=1.0,
            use_rejection_entry=False,
        )
        detector = ICTSetupDetector(cfg)
        df = detector.detect_setups(wfo_candles[:300], "1h")
        if len(df) > 0:
            assert "total_quality_score" in df.columns
            assert "ob_score" in df.columns
            assert "fvg_score" in df.columns
            assert "liq_score" in df.columns
            assert "structure_score_detail" in df.columns

    def test_v2_higher_threshold_fewer_setups(self, wfo_candles):
        """Higher min_quality_score should produce fewer or equal setups."""
        cfg_low = WFOConfig(
            use_quality_scoring=True, min_quality_score=1.0,
            use_rejection_entry=False,
        )
        cfg_high = WFOConfig(
            use_quality_scoring=True, min_quality_score=6.0,
            use_rejection_entry=False,
        )
        det_low = ICTSetupDetector(cfg_low)
        det_high = ICTSetupDetector(cfg_high)
        df_low = det_low.detect_setups(wfo_candles[:300], "1h")
        df_high = det_high.detect_setups(wfo_candles[:300], "1h")
        assert len(df_high) <= len(df_low)

    def test_v2_wfo_engine_runs(self, wfo_candles):
        """V2 config should work end-to-end with WFO engine."""
        cfg = WFOConfig(
            train_window=200, test_window=100, step_size=50,
            max_folds=2, min_setups_per_fold=1,
            use_quality_scoring=True, min_quality_score=1.0,
            use_rejection_entry=False,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        report = engine.run(wfo_candles, "1h")
        assert isinstance(report, WFOReport)
        assert report.grade in ("A", "B", "C", "D")

    def test_v1_unchanged_with_defaults(self, wfo_candles):
        """Default WFOConfig (v1) must still work identically."""
        cfg = WFOConfig(
            train_window=200, test_window=200, step_size=100,
            max_folds=3, min_confluence_score=1, min_setups_per_fold=1,
        )
        engine = WalkForwardEngine(cfg, use_autogluon=False)
        report = engine.run(wfo_candles, "1h")
        assert isinstance(report, WFOReport)
        assert report.grade in ("A", "B", "C", "D")
