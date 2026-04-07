"""ML Calibration Engine — session-aware, multi-layer SL/TP calibration.

Takes Claude's parsed ICT analysis and runs it through 6 calibration layers:
  1. Volatility Calibration (ATR + session + regime scaling)
  2. V1 Session Statistics (baseline drawdown distributions)
  3. Bayesian Beliefs (sequential win rate + drawdown learning)
  4. AutoGluon Quantile Prediction (if models trained)
  5. Historical Pattern Matching (similar setups by volatility profile)
  6. Consensus (widest SL, median TPs, composite confidence)

All functions are pure where possible — side effects (file I/O) isolated to __init__.
"""
import json
import math
import os
import statistics

import numpy as np
import pandas as pd

from ml.config import get_config
from ml.features import compute_atr, engineer_features_from_candles
from ml.features import (
    detect_order_blocks, detect_fvgs, detect_liquidity,
    compute_market_structure, _extract_hour,
)
from ml.volatility import calibrate_volatility, detect_session
from ml.bayesian import get_default_prior, get_beliefs
from ml.wfo import detect_regime


class MLCalibrator:
    """Multi-layer SL/TP calibrator using V1 data + Bayesian + volatility."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self._v1_session_stats = self._load_v1_session_stats()
        self._v1_priors = self._load_v1_priors()
        self._autogluon_available = self._check_autogluon()

    def calibrate_trade(self, parsed_analysis: dict,
                        candles_df) -> dict:
        """Core calibration method — run parsed analysis through all layers.

        Args:
            parsed_analysis: Output from ClaudeAnalysisBridge.parse_analysis()
            candles_df: DataFrame or list of candle dicts

        Returns:
            Full calibration result dict
        """
        candles = self._to_candle_list(candles_df)
        if not candles:
            return self._empty_result(parsed_analysis)

        entry_price = parsed_analysis.get("claude_entry_price", 0)
        sl_price = parsed_analysis.get("claude_sl_price", 0)
        tp_prices = parsed_analysis.get("claude_tp_prices", [])
        direction = parsed_analysis.get("claude_direction", "long")
        killzone = parsed_analysis.get("claude_killzone", "")
        confluence_count = parsed_analysis.get("claude_confluence_count", 0)

        if not entry_price:
            return self._empty_result(parsed_analysis)

        atr = compute_atr(candles, 14)
        if atr <= 0:
            atr = 1.0

        is_long = direction == "long"
        claude_sl_dist = abs(entry_price - sl_price) if sl_price else atr * 1.5
        claude_sl_atr = claude_sl_dist / atr

        warnings = []
        sl_candidates = {}  # source -> (sl_price, sl_atr_distance)
        tp_candidates = {}  # level -> [list of tp prices from sources]
        sl_floor_atr = self.cfg.get("sl_floor_atr", 3.0)

        # Always include Claude's SL
        sl_candidates["claude"] = (sl_price, claude_sl_atr)

        # ── Layer 1: Volatility Calibration ──────────────────────
        vol_cal = calibrate_volatility(candles, "1h", config=self.cfg)
        effective_atr = vol_cal["calibrated_vol"]
        session = vol_cal["session"]
        session_scale = vol_cal["session_factor"]
        regime_label = vol_cal["regime"]
        regime_scale = vol_cal["regime_multiplier"]

        # 5-state structural regime — overrides the old 3-state multiplier
        structural_regime = vol_cal.get("structural_regime", "RANGING")
        structural_regime_confidence = vol_cal.get("structural_regime_confidence", 0)
        struct_sl_mult = vol_cal.get("structural_sl_multiplier", 1.0)
        struct_tp_mult = vol_cal.get("structural_tp_multiplier", 1.0)

        vol_sl_atr = 2.5 * struct_sl_mult  # regime-adjusted SL width
        vol_sl_dist = vol_sl_atr * effective_atr
        if is_long:
            vol_sl_price = entry_price - vol_sl_dist
        else:
            vol_sl_price = entry_price + vol_sl_dist
        sl_candidates["volatility"] = (vol_sl_price, vol_sl_dist / atr)

        vol_tp_atr_mults = [1.0 * struct_tp_mult, 2.0 * struct_tp_mult,
                            3.5 * struct_tp_mult]
        for level, mult in enumerate(vol_tp_atr_mults):
            tp_dist = mult * effective_atr
            if is_long:
                tp_price = entry_price + tp_dist
            else:
                tp_price = entry_price - tp_dist
            tp_candidates.setdefault(level, []).append(tp_price)

        if claude_sl_atr < vol_sl_atr * 0.8:
            warnings.append(
                f"Claude's SL at {claude_sl_atr:.1f} ATR is tighter than "
                f"volatility-calibrated minimum ({vol_sl_atr:.1f} ATR)"
            )

        # ── V1 Decay: reduce V1 influence as live data grows ─────
        # Full influence at 0 live trades, 50% at 150, 20% at 400+
        live_count = 0
        try:
            from ml.dataset import TrainingDatasetManager
            _dm = TrainingDatasetManager(config=self.cfg)
            _ds_stats = _dm.get_stats()
            live_count = _ds_stats.get("live_count", 0)
        except Exception:
            pass
        v1_decay = max(0.2, 1.0 - live_count / 400)

        # ── Layer 2: V1 Session Statistics (DISABLED) ─────────────
        # V1 session drawdown SL disabled — 16.8% survival rate across
        # 337 trades, avg SL 1.08 ATR is inside gold's noise band.
        # Stats still loaded for win rate (used by confidence scoring)
        # and session_context in the return dict.
        mapped_session = self._map_session(session)
        v1_stats = self._v1_session_stats.get(mapped_session, {})
        v1_median_dd = v1_stats.get("median_drawdown", 0)
        v1_p95_dd = v1_stats.get("p95_drawdown", 0)
        v1_median_fav = v1_stats.get("median_favorable", 0)
        v1_session_wr = v1_stats.get("win_rate", 0.3)
        v1_session_trades = v1_stats.get("trades", 0)
        # No longer contributes to sl_candidates

        # ── Layer 3: Bayesian Beliefs ────────────────────────────
        bayesian_wr = 0.5
        bayesian_trades = 0
        bayesian_dd = 0
        bayesian_fav = 0
        bayesian_sl_atr = 0

        if self._v1_priors:
            bayesian_wr = self._v1_priors.get("overall_win_rate", 0.5)
            bayesian_trades = self._v1_priors.get("total_trades", 0)
            bayesian_dd = self._v1_priors.get("drawdown_mu", 1.0)
            bayesian_fav = self._v1_priors.get("favorable_mu", 1.5)
            # Blend Bayesian drawdown toward floor as live data grows
            bayesian_sl_atr = bayesian_dd * 2.0 * v1_decay + sl_floor_atr * (1 - v1_decay)

            if bayesian_sl_atr > 0:
                b_sl_dist = bayesian_sl_atr * atr
                if is_long:
                    b_sl_price = entry_price - b_sl_dist
                else:
                    b_sl_price = entry_price + b_sl_dist
                sl_candidates["bayesian"] = (b_sl_price, bayesian_sl_atr)

        # Also try to load live Bayesian state from DB
        try:
            from ml.database import TradeLogger
            db = TradeLogger(config=self.cfg)
            live_state = db.get_bayesian_state()
            if live_state and live_state.get("total_trades", 0) > 5:
                beliefs = get_beliefs(live_state)
                bayesian_wr = beliefs["win_rate_mean"]
                bayesian_trades = live_state["total_trades"]
        except Exception:
            pass

        # Bayesian TPs (from favorable excursion)
        if bayesian_fav > 0:
            for level, mult in enumerate([0.7, 1.0, 1.5]):
                tp_dist = bayesian_fav * mult * atr
                if is_long:
                    tp_price = entry_price + tp_dist
                else:
                    tp_price = entry_price - tp_dist
                tp_candidates.setdefault(level, []).append(tp_price)

        # ── Layer 4: AutoGluon Quantile Prediction ───────────────
        ag_win_prob = None
        ag_predicted_dd = None
        ag_predicted_fav = None

        if self._autogluon_available:
            try:
                ag_result = self._run_autogluon(parsed_analysis, candles)
                if ag_result:
                    ag_win_prob = ag_result.get("win_probability")
                    ag_predicted_dd = ag_result.get("predicted_p95_drawdown_atr")
                    ag_predicted_fav = ag_result.get("predicted_favorable_atr")

                    if ag_predicted_dd and ag_predicted_dd > 0:
                        ag_sl_dist = ag_predicted_dd * atr
                        if is_long:
                            ag_sl_price = entry_price - ag_sl_dist
                        else:
                            ag_sl_price = entry_price + ag_sl_dist
                        sl_candidates["autogluon"] = (ag_sl_price, ag_predicted_dd)
            except Exception:
                pass

        # ── Layer 5: Historical Pattern Matching ─────────────────
        hist_matches = 0
        hist_wr = None
        hist_median_dd = None
        hist_median_fav = None

        try:
            hist_result = self._find_historical_matches(
                parsed_analysis, candles, mapped_session
            )
            hist_matches = hist_result.get("match_count", 0)
            hist_wr = hist_result.get("win_rate")
            hist_median_dd = hist_result.get("median_drawdown_atr")
            hist_median_fav = hist_result.get("median_favorable_atr")

            if hist_matches >= 10 and hist_median_dd:
                hist_sl_atr = hist_median_dd * 1.3
                hist_sl_dist = hist_sl_atr * atr
                if is_long:
                    hist_sl_price = entry_price - hist_sl_dist
                else:
                    hist_sl_price = entry_price + hist_sl_dist
                sl_candidates["historical"] = (hist_sl_price, hist_sl_atr)
            elif hist_matches < 10:
                warnings.append(
                    f"Low historical precedent: only {hist_matches} similar "
                    f"setups found in dataset"
                )
        except Exception:
            pass

        # ── SL Floor: Gold's noise floor ──────────────────────────
        # Data: 72% of trades draw down past 1.3 ATR, 48% past 3.0 ATR.
        # Claude's median SL is 1.3 ATR — inside the noise band.
        # Use adaptive floor if enough segment data exists
        try:
            from ml.layer_performance import LayerPerformanceTracker
            _perf = LayerPerformanceTracker(config=self.cfg)
            _grade = parsed_analysis.get("claude_setup_grade", "")
            sl_floor_atr = _perf.get_adaptive_floor(
                grade=_grade, killzone=killzone, default=sl_floor_atr,
            )
        except Exception:
            pass

        if atr > 0 and claude_sl_atr < sl_floor_atr:
            floor_sl_dist = sl_floor_atr * atr
            if is_long:
                floor_sl_price = entry_price - floor_sl_dist
            else:
                floor_sl_price = entry_price + floor_sl_dist
            sl_candidates["floor"] = (floor_sl_price, sl_floor_atr)
            warnings.append(
                f"Claude's SL at {claude_sl_atr:.1f} ATR is inside gold's "
                f"noise band. Floor enforced at {sl_floor_atr:.1f} ATR "
                f"(52% of trades survive at this level)."
            )

        # ── Layer 6: Consensus ───────────────────────────────────

        # Final SL: widest (most conservative)
        final_sl_price = sl_price
        final_sl_source = "claude"
        final_sl_atr = claude_sl_atr

        for source, (sl_p, sl_a) in sl_candidates.items():
            if sl_p == 0:
                continue
            # "Widest" means furthest from entry
            candidate_dist = abs(entry_price - sl_p)
            current_dist = abs(entry_price - final_sl_price)
            if candidate_dist > current_dist:
                final_sl_price = sl_p
                final_sl_source = source
                final_sl_atr = sl_a

        # Final TPs: median across sources per level
        final_tps = []
        for level in sorted(tp_candidates.keys()):
            candidates = [p for p in tp_candidates[level] if p and p != 0]
            # Also include Claude's TPs if available
            if level < len(tp_prices) and tp_prices[level]:
                candidates.append(tp_prices[level])
            if candidates:
                final_tps.append(round(statistics.median(candidates), 2))

        # Fill remaining TPs from Claude if fewer consensus TPs
        while len(final_tps) < len(tp_prices):
            final_tps.append(tp_prices[len(final_tps)])

        # TP distances
        final_tp_distances_atr = [
            round(abs(entry_price - tp) / atr, 4) for tp in final_tps
        ]
        risk = abs(entry_price - final_sl_price)
        final_rr_ratios = [
            round(abs(entry_price - tp) / risk, 4) if risk > 0 else 0
            for tp in final_tps
        ]

        # ── Confidence Score ─────────────────────────────────────
        confidence = 1.0

        # Claude signal strength
        claude_factor = min(1.0, confluence_count / 6)
        confidence *= (0.3 + 0.7 * claude_factor)

        # Bayesian win rate
        confidence *= (0.4 + 0.6 * bayesian_wr)

        # AutoGluon (if available)
        if ag_win_prob is not None:
            confidence *= (0.5 + 0.5 * ag_win_prob)

        # Historical match penalty
        if hist_matches < 5:
            confidence *= 0.5
        elif hist_matches < 10:
            confidence *= 0.7

        # Session quality
        confidence *= (0.5 + 0.5 * v1_session_wr)

        confidence = max(0.05, min(0.95, confidence))

        # ── Grade ────────────────────────────────────────────────
        # Check SL agreement
        sl_values = [v[1] for v in sl_candidates.values() if v[1] > 0]
        sl_agree_count = 0
        if sl_values:
            for sv in sl_values:
                if abs(sv - final_sl_atr) < 0.5:
                    sl_agree_count += 1

        critical_warning = any("tighter than" in w for w in warnings)

        if confidence > 0.65 and sl_agree_count >= 3:
            grade = "A"
        elif confidence > 0.50:
            grade = "B"
        elif confidence > 0.35:
            grade = "C"
        elif confidence > 0.20:
            grade = "D"
        else:
            grade = "F"

        if critical_warning and grade in ("A", "B"):
            grade = "C"

        # ── Adjustments ──────────────────────────────────────────
        sl_widened = abs(entry_price - final_sl_price) > abs(entry_price - sl_price) + 0.01
        sl_widened_by = abs(entry_price - final_sl_price) - abs(entry_price - sl_price) if sl_widened else 0
        sl_widened_by_atr = sl_widened_by / atr if atr > 0 else 0

        # Determine TP adjustment direction
        if final_tps and tp_prices:
            avg_final = sum(abs(entry_price - tp) for tp in final_tps) / len(final_tps)
            avg_claude = sum(abs(entry_price - tp) for tp in tp_prices) / len(tp_prices)
            if avg_final > avg_claude * 1.02:
                tp_adj_dir = "widened"
            elif avg_final < avg_claude * 0.98:
                tp_adj_dir = "narrowed"
            else:
                tp_adj_dir = "unchanged"
        else:
            tp_adj_dir = "unchanged"

        tp_adjusted = tp_adj_dir != "unchanged"

        sl_reason = ""
        if sl_widened:
            if final_sl_source == "v1_session":
                sl_reason = (
                    f"V1 session data shows 95th pctile drawdown is "
                    f"{v1_p95_dd:.1f} ATR during {mapped_session}"
                )
            elif final_sl_source == "volatility":
                sl_reason = (
                    f"Volatility calibration (ATR×session×regime = "
                    f"{effective_atr:.2f}) requires wider stop"
                )
            elif final_sl_source == "bayesian":
                sl_reason = (
                    f"Bayesian median drawdown of {bayesian_dd:.1f} ATR "
                    f"suggests wider stop (from {bayesian_trades} trades)"
                )
            elif final_sl_source == "autogluon":
                sl_reason = (
                    f"AutoGluon predicts 95th pctile drawdown of "
                    f"{ag_predicted_dd:.1f} ATR"
                )
            elif final_sl_source == "historical":
                sl_reason = (
                    f"Historical matches ({hist_matches} trades) show "
                    f"median drawdown of {hist_median_dd:.1f} ATR"
                )
            elif final_sl_source == "floor":
                sl_reason = (
                    f"SL floor enforced at {sl_floor_atr:.1f} ATR — "
                    f"Claude's {claude_sl_atr:.1f} ATR is inside gold's noise band"
                )

        # ── Recommendation String ────────────────────────────────
        recommendation = self._build_recommendation(
            sl_widened=sl_widened,
            sl_widened_by=sl_widened_by,
            final_sl_price=final_sl_price,
            sl_price=sl_price,
            final_sl_source=final_sl_source,
            claude_sl_atr=claude_sl_atr,
            v1_p95_dd=v1_p95_dd,
            mapped_session=mapped_session,
            bayesian_wr=bayesian_wr,
            bayesian_trades=bayesian_trades,
            hist_matches=hist_matches,
            hist_wr=hist_wr,
            ag_win_prob=ag_win_prob,
            grade=grade,
            confidence=confidence,
            setup_type=self._get_setup_type(parsed_analysis),
            warnings=warnings,
        )

        return {
            "claude_original": {
                "entry": entry_price,
                "sl": sl_price,
                "tps": tp_prices,
                "direction": direction,
                "rr_ratios": parsed_analysis.get("claude_rr_ratios", []),
            },
            "calibrated": {
                "entry": entry_price,  # never override entry
                "sl": round(final_sl_price, 2),
                "sl_source": final_sl_source,
                "sl_distance_atr": round(final_sl_atr, 4),
                "tps": final_tps,
                "tp_distances_atr": final_tp_distances_atr,
                "rr_ratios": final_rr_ratios,
                "risk_amount": round(abs(entry_price - final_sl_price), 2),
            },
            "adjustments": {
                "sl_widened": sl_widened,
                "sl_widened_by": round(sl_widened_by, 2),
                "sl_widened_by_atr": round(sl_widened_by_atr, 4),
                "sl_widened_reason": sl_reason,
                "tp_adjusted": tp_adjusted,
                "tp_adjustment_direction": tp_adj_dir,
            },
            "confidence": {
                "score": round(confidence, 4),
                "grade": grade,
                "claude_signal_strength": round(claude_factor, 4),
                "bayesian_win_rate": round(bayesian_wr, 4),
                "bayesian_trades_for_type": bayesian_trades,
                "autogluon_win_prob": round(ag_win_prob, 4) if ag_win_prob is not None else None,
                "historical_match_count": hist_matches,
                "historical_match_win_rate": round(hist_wr, 4) if hist_wr is not None else None,
                "session_win_rate": round(v1_session_wr, 4),
            },
            "session_context": {
                "session": mapped_session,
                "v1_median_drawdown": round(v1_median_dd, 4),
                "v1_p95_drawdown": round(v1_p95_dd, 4),
                "v1_median_favorable": round(v1_median_fav, 4),
                "v1_session_win_rate": round(v1_session_wr, 4),
                "v1_session_trades": v1_session_trades,
            },
            "volatility_context": {
                "atr_14": round(atr, 4),
                "effective_atr": round(effective_atr, 4),
                "regime": regime_label,
                "session_scale": session_scale,
                "regime_scale": regime_scale,
                "structural_regime": structural_regime,
                "structural_regime_confidence": structural_regime_confidence,
                "structural_sl_multiplier": struct_sl_mult,
                "structural_tp_multiplier": struct_tp_mult,
            },
            "warnings": warnings,
            "recommendation": recommendation,
            "layer_candidates": {
                src: {"sl_price": round(sl_p, 2), "sl_distance_atr": round(sl_a, 4)}
                for src, (sl_p, sl_a) in sl_candidates.items() if sl_p != 0
            },
        }

        # ── Multi-3 enhancements (only when active) ────────────
        if ag_result and ag_result.get("optimal_split"):
            calibration_result["multi3"] = {
                "class_probabilities": ag_result.get("class_probabilities"),
                "optimal_split": ag_result["optimal_split"],
                "ev": ag_result.get("ev"),
                "adjusted_ev_per_unit": ag_result.get("adjusted_ev_per_unit"),
                "position_multiplier": ag_result.get("position_multiplier", 1.0),
            }

            if ag_result.get("low_ev_warning"):
                warnings.append(ag_result["low_ev_warning"])

        return calibration_result

    # ── Pre-analysis ML context (Phase 2: feeds INTO Sonnet's prompt) ──

    def build_ml_context(self, thesis_type: str | None, timeframe: str,
                         killzone: str, candles: list,
                         setup_dna_pattern: dict | None = None) -> dict:
        """Build prompt-ready ML context block for Sonnet enrichment.

        Runs the statistical memory layers and returns structured data
        for injection into Sonnet's prompt — NOT competing SL/TP values.

        Each layer is independently try/excepted so a single failure
        doesn't prevent the rest from contributing.

        Args:
            thesis_type: e.g. "bullish_accumulation" from narrative engine
            timeframe: "15min", "1h", "4h", "1day"
            killzone: "Asian", "London", "NY_AM", "NY_PM", "Off"
            candles: OHLC candle list for ATR/regime computation
            setup_dna_pattern: optional DNA dict for pattern matching

        Returns:
            dict with keys: regime, sl_floor_atr, mae_percentile_80,
            dna_win_rate, dna_avg_rr, dna_sample_size, bayesian_wr,
            bayesian_trend, intermarket_quality, entry_placement, etc.
        """
        ctx = {}

        # 1. Regime + volatility floor
        try:
            candle_list = self._to_candle_list(candles)
        except Exception:
            candle_list = None
        if candle_list and len(candle_list) >= 14:
            try:
                from ml.volatility import calibrate_volatility, detect_session
                from ml.features import compute_atr
                atr = compute_atr(candle_list)
                vol = calibrate_volatility(candle_list, timeframe)
                ctx["regime"] = vol.get("regime", "NORMAL")
                ctx["vol_ratio"] = vol.get("vol_ratio", 1.0)
                ctx["sl_floor_atr"] = vol.get("min_sl_atr", 3.0)
                # MAE from V1 session stats
                session = detect_session(candle_list)
                mapped_session = self._map_session(session)
                session_stats = self._v1_session_stats.get(mapped_session, {})
                ctx["mae_percentile_80"] = session_stats.get("mae_p80_atr", 4.0)
            except Exception:
                ctx.setdefault("regime", "UNKNOWN")
                ctx.setdefault("sl_floor_atr", 3.0)
                ctx.setdefault("mae_percentile_80", 4.0)
        else:
            ctx["regime"] = "UNKNOWN"
            ctx["sl_floor_atr"] = 3.0
            ctx["mae_percentile_80"] = 4.0

        # 2. Setup DNA pattern match
        if setup_dna_pattern:
            try:
                from ml.setup_profiles import SetupProfileStore
                store = SetupProfileStore()
                stats = store.get_conditional_stats(setup_dna_pattern)
                if stats and stats.get("sample_size", 0) >= 15:
                    ctx["dna_win_rate"] = stats["win_rate"]
                    ctx["dna_avg_rr"] = stats.get("avg_rr", 0)
                    ctx["dna_sample_size"] = stats["sample_size"]
            except Exception:
                pass

        # 3. Bayesian beliefs
        try:
            from ml.bayesian import get_beliefs
            from ml.database import TradeLogger
            db = TradeLogger()
            state = db.get_bayesian_state()
            if state:
                beliefs = get_beliefs(state)
                ctx["bayesian_wr"] = beliefs.get("win_rate_mean", 0.5)
                # Compute trend: compare recent vs overall
                total = beliefs.get("total_trades", 0)
                if total > 20:
                    # Simple trend: current WR vs prior midpoint
                    prior_mean = 0.446  # V1 seed prior
                    ctx["bayesian_trend"] = round(
                        (beliefs["win_rate_mean"] - prior_mean) * 100, 1)
                else:
                    ctx["bayesian_trend"] = 0
        except Exception:
            pass

        # 4. Intermarket signal quality
        try:
            from ml.intermarket_validator import IntermarketValidator
            iv = IntermarketValidator()
            result = iv.get_last_result()
            if result:
                ctx["intermarket_quality"] = result.get("recommendation", "unknown")
        except Exception:
            pass

        # 5. Opus accuracy by narrative type
        if thesis_type:
            try:
                from ml.claude_bridge import ClaudeAnalysisBridge
                bridge = ClaudeAnalysisBridge()
                tracker = bridge._accuracy.get("narrative_tracker", {})
                if tracker:
                    # Overall narrative alignment accuracy
                    aligned = tracker.get("aligned", {})
                    total_aligned = aligned.get("total", 0)
                    if total_aligned >= 10:
                        ctx["opus_accuracy"] = aligned.get("wins", 0) / total_aligned
            except Exception:
                pass

        # 6. Entry placement guidance
        try:
            from ml.entry_placement import EntryPlacementAnalyzer
            guidance = EntryPlacementAnalyzer().get_placement_guidance()
            if guidance and guidance.get("status") == "active":
                ctx["entry_placement"] = guidance.get("best_zone", "OB midpoint")
                ctx["entry_placement_delta_rr"] = guidance.get("improvement_rr", 0)
        except Exception:
            pass

        return ctx

    # ── Private helpers ──────────────────────────────────────────

    def _load_v1_session_stats(self) -> dict:
        """Load V1 session stats from disk."""
        path = os.path.join(
            self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
            "v1_session_stats.json"
        )
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_v1_priors(self) -> dict:
        """Load V1 Bayesian priors from disk."""
        path = os.path.join(
            self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models")),
            "v1_bayesian_priors.json"
        )
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _check_autogluon(self) -> bool:
        """Check if AutoGluon models exist."""
        model_dir = self.cfg.get("model_dir", "models/")
        classifier_path = os.path.join(model_dir, "classifier")
        return os.path.exists(classifier_path)

    def _is_model_trustworthy(self) -> bool:
        """Check if the AutoGluon model has demonstrated genuine predictive power."""
        eval_path = os.path.join(self.cfg.get("model_dir", "models"),
                                 "classifier_evaluation.json")
        if not os.path.exists(eval_path):
            return False
        try:
            with open(eval_path) as f:
                data = json.load(f)
            return (
                data.get("oos_accuracy", 0) > self.cfg.get("oos_min_accuracy", 0.55)
                and data.get("test_trades", 0) >= self.cfg.get("oos_min_test_trades", 30)
            )
        except Exception:
            return False

    def _run_autogluon(self, parsed: dict, candles: list) -> dict | None:
        """Run AutoGluon prediction if models available and trustworthy."""
        # Gate: only use AG predictions when model has proven OOS accuracy
        if not self._is_model_trustworthy():
            return None

        try:
            from ml.prediction import predict
            from ml.database import TradeLogger
            from ml.dataset import TrainingDatasetManager
            analysis = {
                "bias": parsed.get("claude_bias", "neutral"),
                "entry": {
                    "price": parsed["claude_entry_price"],
                    "direction": parsed["claude_direction"],
                },
                "stopLoss": {"price": parsed["claude_sl_price"]},
                "takeProfits": [{"price": p} for p in parsed.get("claude_tp_prices", [])],
                "orderBlocks": [],
                "fvgs": [],
                "liquidity": [],
                "confluences": ["c"] * parsed.get("claude_confluence_count", 0),
                "killzone": parsed.get("claude_killzone", ""),
            }
            db = TradeLogger()
            dm = TrainingDatasetManager()
            result = predict(analysis, candles, "1h", db=db, dataset_manager=dm)
            if result:
                win_prob = result.get("win_probability", result.get("confidence", 0.5))
                classification = result.get("classification", {})

                return {
                    "win_probability": win_prob,
                    "active_model_type": "binary",
                    "classification": classification,
                    "predicted_p95_drawdown_atr": result.get("suggested_sl"),
                    "predicted_favorable_atr": result.get("suggested_tp1"),
                }
        except Exception:
            pass
        return None

    # Candidate partial splits for EV optimisation (tp1, runner)
    CANDIDATE_SPLITS = [
        (1.0, 0.0),      # full TP1 scalp
        (0.7, 0.3),      # mostly TP1
        (0.5, 0.5),      # half-half TP1/runner
        (0.4, 0.6),      # runner-leaning
        (0.3, 0.7),      # runner-weighted
        (0.2, 0.8),      # heavy runner
    ]

    @staticmethod
    def compute_setup_ev(class_probs: dict, entry: float, sl: float,
                         tps: list, partial_split: tuple) -> dict:
        """Compute expected value using cumulative partial accounting.

        3-class model: stopped_out / tp1 / runner.
        A runner trade also passed TP1.
        """
        p_sl = class_probs.get("stopped_out", 0)
        p_tp1 = class_probs.get("tp1", 0)
        p_runner = class_probs.get("runner", 0)

        sl_dist = abs(entry - sl) if entry and sl else 1
        rewards = [abs(tp - entry) for tp in tps[:2]]
        while len(rewards) < 2:
            rewards.append(rewards[-1] * 2.0 if rewards else sl_dist)

        # Cumulative probabilities
        p_reaches_tp1 = p_tp1 + p_runner
        p_reaches_runner = p_runner

        s1, s2 = partial_split

        ev = (p_reaches_tp1 * s1 * rewards[0]
              + p_reaches_runner * s2 * rewards[1]
              - p_sl * sl_dist)

        return {
            "ev_per_unit": round(ev / sl_dist, 4) if sl_dist > 0 else 0,
            "ev_dollars": round(ev, 2),
            "p_win": round(1 - p_sl, 4),
            "p_reaches_tp1": round(p_reaches_tp1, 4),
            "p_reaches_runner": round(p_reaches_runner, 4),
        }

    def _find_optimal_split(self, class_probs: dict, entry: float,
                             sl: float, tps: list) -> dict:
        """Find the partial split that maximises EV across candidate splits."""
        best_ev = -float("inf")
        best_split = (0.5, 0.5)

        for split in self.CANDIDATE_SPLITS:
            result = self.compute_setup_ev(class_probs, entry, sl, tps, split)
            if result["ev_per_unit"] > best_ev:
                best_ev = result["ev_per_unit"]
                best_split = split

        if best_split[0] >= 0.7:
            style = "scalp"
        elif best_split[1] >= 0.5:
            style = "runner"
        else:
            style = "swing"

        return {
            "split": best_split,
            "style": style,
            "ev_per_unit": round(best_ev, 4),
            "advice": (f"Take {int(best_split[0]*100)}% at TP1, "
                       f"{int(best_split[1]*100)}% as runner"),
        }

    @staticmethod
    def compute_kelly_size(ev_per_unit: float, win_rate: float,
                           avg_win_rr: float, max_mult: float = 2.0) -> float:
        """Half-Kelly position sizing from predicted edge.

        Returns a multiplier on the base lot size (1.0 = standard).
        """
        if ev_per_unit <= 0 or win_rate <= 0 or avg_win_rr <= 0:
            return 1.0

        b = avg_win_rr
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b
        if kelly <= 0:
            return 1.0

        half_kelly = kelly * 0.5
        multiplier = 1.0 + (half_kelly * 5.0)
        return round(min(max(multiplier, 0.5), max_mult), 2)

    def _compute_multi3_enhancements(self, classification: dict,
                                       parsed: dict) -> dict:
        """Compute optimal partials, cumulative EV, and half-Kelly sizing.

        Only called when multi3 model is active and passed OOS gate.
        """
        entry = parsed.get("claude_entry_price", 0)
        sl = parsed.get("claude_sl_price", 0)
        tps = parsed.get("claude_tp_prices", [])

        # Find optimal split
        optimal = self._find_optimal_split(classification, entry, sl, tps)

        # Compute EV with optimal split
        ev = self.compute_setup_ev(
            classification, entry, sl, tps, optimal["split"])

        # Half-Kelly position sizing
        p_win = ev["p_win"]
        avg_win_rr = optimal["ev_per_unit"] / p_win if p_win > 0 else 1.0
        max_mult = self.cfg.get("max_position_multiplier", 2.0)
        kelly_mult = self.compute_kelly_size(
            ev["ev_per_unit"], p_win, avg_win_rr, max_mult)

        # EV filter: cap position at 1.0x if EV too low
        spread_cost = self.cfg.get("spread_cost_rr", 0.05)
        adjusted_ev = ev["ev_per_unit"] - spread_cost
        min_ev = self.cfg.get("min_ev_ratio", 0.15)
        low_ev_warning = None
        if adjusted_ev < min_ev:
            kelly_mult = min(kelly_mult, 1.0)
            low_ev_warning = (f"Low EV ({adjusted_ev:.2f}R after costs) — "
                              f"position capped at 1.0×")

        return {
            "class_probabilities": {k: round(v, 3) for k, v in classification.items()},
            "optimal_split": optimal,
            "ev": ev,
            "adjusted_ev_per_unit": round(adjusted_ev, 4),
            "position_multiplier": kelly_mult,
            "low_ev_warning": low_ev_warning,
        }

    def _find_historical_matches(self, parsed: dict, candles: list,
                                 session: str) -> dict:
        """Find similar historical setups by volatility profile."""
        try:
            from ml.dataset import TrainingDatasetManager
            dm = TrainingDatasetManager(config=self.cfg)
            df = dm.get_blended_dataset()

            if df.empty or len(df) < 5:
                return {"match_count": 0}

            # Current volatility features
            atr = compute_atr(candles, 14)
            idx = parsed.get("entry_candle_idx", len(candles) - 1)
            if idx < 0:
                idx = len(candles) - 1

            features = parsed.get("features", {})
            current_atr = features.get("atr_14", atr)
            current_vol_ratio = features.get("recent_volatility_ratio", 1.0)
            current_range = features.get("last_candle_body_atr", 0.5)

            direction = parsed.get("claude_direction", "long")

            # Filter: same direction + session, OR same direction
            mask = pd.Series([True] * len(df))
            if "direction" in df.columns:
                mask &= df["direction"] == direction

            # Volatility proximity (within 1.5 std)
            for feat_name, current_val in [
                ("atr_14", current_atr),
                ("recent_volatility_ratio", current_vol_ratio),
                ("last_candle_body_atr", current_range),
            ]:
                if feat_name in df.columns:
                    col = df[feat_name].astype(float)
                    std = col.std()
                    if std > 0:
                        mask &= (col - current_val).abs() <= 1.5 * std

            matches = df[mask]
            n = len(matches)

            if n == 0:
                return {"match_count": 0}

            win_outcomes = {"tp1_hit", "tp2_hit", "tp3_hit"}
            winners = matches[matches["outcome"].isin(win_outcomes)] if "outcome" in matches.columns else pd.DataFrame()

            return {
                "match_count": n,
                "win_rate": len(winners) / n if n > 0 else 0,
                "median_drawdown_atr": float(
                    matches["max_drawdown_atr"].median()
                ) if "max_drawdown_atr" in matches.columns and n > 0 else 0,
                "median_favorable_atr": float(
                    matches["max_favorable_atr"].median()
                ) if "max_favorable_atr" in matches.columns and n > 0 else 0,
            }
        except Exception:
            return {"match_count": 0}

    def _estimate_stop_pct(self, sl_atr: float, v1_stats: dict) -> float:
        """Estimate what % of winning trades would have been stopped at given SL."""
        median_dd = v1_stats.get("median_drawdown", 0)
        p95_dd = v1_stats.get("p95_drawdown", 0)

        if p95_dd <= 0 or sl_atr >= p95_dd:
            return 0.0

        # Linear interpolation between median (50%) and p95 (5%)
        if sl_atr <= median_dd:
            return 0.5 + 0.5 * (median_dd - sl_atr) / median_dd if median_dd > 0 else 0.5
        else:
            range_ = p95_dd - median_dd
            if range_ <= 0:
                return 0.05
            progress = (sl_atr - median_dd) / range_
            return 0.5 - 0.45 * progress

    def _map_session(self, session: str) -> str:
        """Map volatility.py session names to V1 session names."""
        mapping = {
            "london": "london",
            "overlap_london_ny": "ny_am",
            "new_york": "ny_pm",
            "asian": "asia",
            "off_hours": "off",
        }
        return mapping.get(session, session)

    def _get_setup_type(self, parsed: dict) -> str:
        """Get setup type string from parsed analysis."""
        from ml.claude_bridge import ClaudeAnalysisBridge
        bridge = ClaudeAnalysisBridge(config=self.cfg)
        return bridge.classify_setup_type(parsed)

    def _build_recommendation(self, **kwargs) -> str:
        """Build human-readable recommendation string."""
        parts = []

        if kwargs["sl_widened"]:
            parts.append(
                f"SL widened from {kwargs['sl_price']:.2f} to "
                f"{kwargs['final_sl_price']:.2f} "
                f"(+${kwargs['sl_widened_by']:.2f})."
            )

            source = kwargs["final_sl_source"]
            if source == "v1_session" and kwargs["v1_p95_dd"] > 0:
                parts.append(
                    f"During {kwargs['mapped_session']} session, V1 data shows "
                    f"95th percentile drawdown of {kwargs['v1_p95_dd']:.1f} ATR "
                    f"on winning trades — Claude's SL at {kwargs['claude_sl_atr']:.1f} "
                    f"ATR would have been stopped out frequently."
                )
            elif source == "volatility":
                parts.append(
                    "Volatility calibration requires wider stop for current conditions."
                )

            if kwargs["bayesian_trades"] > 0:
                parts.append(
                    f"Bayesian win rate for {kwargs['setup_type']}: "
                    f"{kwargs['bayesian_wr']:.1%} ({kwargs['bayesian_trades']} trades)."
                )
        else:
            parts.append(
                "Claude's levels look reasonable — SL is already wider than "
                "all model recommendations."
            )

        if kwargs["hist_matches"] > 0:
            parts.append(
                f"{kwargs['hist_matches']} historical matches with similar "
                f"volatility found"
                + (f", {kwargs['hist_wr']:.1%} win rate." if kwargs["hist_wr"] else ".")
            )

        if kwargs["ag_win_prob"] is not None:
            parts.append(
                f"AutoGluon predicts {kwargs['ag_win_prob']:.0%} win probability."
            )

        parts.append(f"Confidence {kwargs['grade']}.")

        if kwargs["warnings"]:
            for w in kwargs["warnings"][:2]:
                parts.append(f"⚠ {w}")

        return " ".join(parts)

    def _to_candle_list(self, candles_df) -> list:
        """Convert candles input to list of dicts."""
        if isinstance(candles_df, list):
            return candles_df
        if hasattr(candles_df, "to_dict"):
            return candles_df.to_dict("records")
        return []

    def _empty_result(self, parsed: dict) -> dict:
        """Return empty calibration result when data is insufficient."""
        return {
            "claude_original": {
                "entry": parsed.get("claude_entry_price", 0),
                "sl": parsed.get("claude_sl_price", 0),
                "tps": parsed.get("claude_tp_prices", []),
                "direction": parsed.get("claude_direction", "long"),
                "rr_ratios": [],
            },
            "calibrated": {
                "entry": parsed.get("claude_entry_price", 0),
                "sl": parsed.get("claude_sl_price", 0),
                "sl_source": "claude",
                "sl_distance_atr": 0,
                "tps": parsed.get("claude_tp_prices", []),
                "tp_distances_atr": [],
                "rr_ratios": [],
                "risk_amount": 0,
            },
            "adjustments": {
                "sl_widened": False,
                "sl_widened_by": 0,
                "sl_widened_by_atr": 0,
                "sl_widened_reason": "",
                "tp_adjusted": False,
                "tp_adjustment_direction": "unchanged",
            },
            "confidence": {
                "score": 0.1,
                "grade": "F",
                "claude_signal_strength": 0,
                "bayesian_win_rate": 0.5,
                "bayesian_trades_for_type": 0,
                "autogluon_win_prob": None,
                "historical_match_count": 0,
                "historical_match_win_rate": None,
                "session_win_rate": 0,
            },
            "session_context": {
                "session": "unknown",
                "v1_median_drawdown": 0,
                "v1_p95_drawdown": 0,
                "v1_median_favorable": 0,
                "v1_session_win_rate": 0,
                "v1_session_trades": 0,
            },
            "volatility_context": {
                "atr_14": 0,
                "effective_atr": 0,
                "regime": "unknown",
                "session_scale": 1.0,
                "regime_scale": 1.0,
            },
            "warnings": ["Insufficient data for calibration"],
            "recommendation": "Insufficient data — use Claude's levels with caution",
        }
