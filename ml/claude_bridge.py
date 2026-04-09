"""Claude Analysis Bridge — converts Claude's ICT analysis into structured ML input.

Parses Claude's narrative ICT analysis JSON and extracts:
  - Entry/SL/TP prices and ATR-normalized distances
  - ICT element presence (OB, FVG, liquidity sweeps)
  - HTF context (premium/discount, Power of 3)
  - Setup quality classification
  - Engineered features at the entry candle

Also handles trade lifecycle: logging completed trades, tracking Claude's
accuracy, and measuring calibration value-add.
"""
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone, timedelta

from ml.config import get_config
from ml.features import compute_atr, engineer_features_from_candles, _extract_hour
from ml.features import (
    detect_order_blocks, detect_fvgs, detect_liquidity,
    compute_market_structure,
)


class ClaudeAnalysisBridge:
    """Parse Claude's ICT analysis into structured data for ML calibration."""

    NARRATIVE_FIELDS = (
        "directional_bias", "p3_phase", "premium_discount",
        "confidence_calibration", "intermarket_synthesis", "key_levels",
    )
    KILLZONE_KEYS = ("Asian", "London", "NY_AM", "NY_PM", "Off")

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        model_dir = self.cfg.get("model_dir", os.path.join(os.path.dirname(__file__), "models"))
        self._accuracy_path = os.path.join(model_dir, "claude_accuracy.json")
        self._narrative_weights_path = os.path.join(model_dir, "narrative_weights.json")
        self._accuracy = self._load_accuracy()
        self._narrative_weights = self._load_narrative_weights()

    def parse_analysis(self, analysis_json: dict,
                       candles_df=None) -> dict:
        """Convert Claude's analysis JSON into structured calibration input.

        Args:
            analysis_json: Claude's raw ICT analysis response
            candles_df: DataFrame or list of candle dicts (optional — needed for features)

        Returns:
            Structured dict with all calibration-relevant fields
        """
        entry_data = analysis_json.get("entry") or {}
        sl_data = analysis_json.get("stopLoss") or {}
        tps_data = analysis_json.get("takeProfits") or []
        obs_data = analysis_json.get("orderBlocks") or []
        fvgs_data = analysis_json.get("fvgs") or []
        liqs_data = analysis_json.get("liquidity") or []
        confluences = analysis_json.get("confluences") or []
        htf_context = analysis_json.get("htf_context") or {}
        structure = analysis_json.get("structure") or {}

        entry_price = entry_data.get("price", 0) if entry_data else 0
        sl_price = sl_data.get("price", 0) if sl_data else 0
        direction = (entry_data.get("direction", "long") if entry_data else "long").lower()
        tp_prices = [tp.get("price", 0) for tp in tps_data]
        killzone = analysis_json.get("killzone", "")

        # Compute ATR if candles available
        candles = self._to_candle_list(candles_df) if candles_df is not None else []
        atr = compute_atr(candles, 14) if len(candles) >= 15 else 1.0
        if atr <= 0:
            atr = 1.0

        # Distance calculations
        sl_distance = abs(entry_price - sl_price) if entry_price and sl_price else 0
        sl_distance_atr = sl_distance / atr

        tp_distances = [abs(entry_price - tp) for tp in tp_prices] if entry_price else []
        tp_distances_atr = [d / atr for d in tp_distances]

        risk = sl_distance if sl_distance > 0 else 1.0
        rr_ratios = [d / risk for d in tp_distances]

        # ICT element detection
        has_ob = len(obs_data) > 0
        has_fvg = len(fvgs_data) > 0
        bsl_present = any(l.get("type") == "buyside" for l in liqs_data)
        ssl_present = any(l.get("type") == "sellside" for l in liqs_data)
        liq_swept = any(l.get("swept", False) for l in liqs_data)

        # OB sub-detail (from Claude's JSON — used by _build_minimal_features)
        ob_bullish = [ob for ob in obs_data if ob.get("type") == "bullish"]
        ob_bearish = [ob for ob in obs_data if ob.get("type") == "bearish"]
        ob_sizes = [(ob["high"] - ob["low"]) for ob in obs_data
                     if "high" in ob and "low" in ob]
        ob_distances = []
        if entry_price:
            for ob in obs_data:
                mid = (ob.get("high", 0) + ob.get("low", 0)) / 2
                if mid > 0:
                    ob_distances.append(abs(entry_price - mid))

        # FVG sub-detail
        unfilled_fvgs = [f for f in fvgs_data if not f.get("filled", True)]
        fvg_distances = []
        if entry_price:
            for f in unfilled_fvgs:
                mid = (f.get("high", 0) + f.get("low", 0)) / 2
                if mid > 0:
                    fvg_distances.append(abs(entry_price - mid))

        # Find entry candle index
        entry_candle_idx = -1
        features = {}
        if candles and entry_price:
            entry_candle_idx = self.find_entry_candle(candles, entry_price, direction)

            # Engineer features at entry candle
            if entry_candle_idx >= 0 and entry_candle_idx < len(candles):
                try:
                    obs = detect_order_blocks(candles, atr, 1.5)
                    fvgs_all = detect_fvgs(candles)
                    liqs = detect_liquidity(candles, window=20)
                    ms_score = compute_market_structure(
                        candles[:entry_candle_idx + 1], lookback=20
                    )
                    features = engineer_features_from_candles(
                        candles, entry_candle_idx, direction, atr,
                        obs, fvgs_all, liqs, ms_score,
                    )
                except Exception:
                    features = {}

        return {
            # Claude's raw suggestions
            "claude_entry_price": entry_price,
            "claude_sl_price": sl_price,
            "claude_tp_prices": tp_prices,
            "claude_direction": direction,
            "claude_bias": analysis_json.get("bias", "neutral"),
            "claude_killzone": killzone,
            "claude_confluence_count": len(confluences),

            # Derived distances
            "claude_sl_distance_atr": round(sl_distance_atr, 4),
            "claude_tp_distances_atr": [round(d, 4) for d in tp_distances_atr],
            "claude_rr_ratios": [round(r, 4) for r in rr_ratios],

            # ICT elements
            "has_ob": has_ob,
            "ob_count": len(obs_data),
            "ob_bullish_count": len(ob_bullish),
            "ob_bearish_count": len(ob_bearish),
            "ob_nearest_distance": min(ob_distances) if ob_distances else 0,
            "ob_avg_size": (sum(ob_sizes) / len(ob_sizes)) if ob_sizes else 0,
            "has_fvg": has_fvg,
            "fvg_count": len(fvgs_data),
            "fvg_unfilled_count": len(unfilled_fvgs),
            "fvg_nearest_distance": min(fvg_distances) if fvg_distances else 0,
            "has_bsl": bsl_present,
            "has_ssl": ssl_present,
            "liq_swept": liq_swept,

            # HTF context
            "htf_premium_discount": htf_context.get("premium_discount"),
            "htf_power_of_3": htf_context.get("power_of_3_phase"),
            "htf_bias": htf_context.get("htf_bias"),

            # Setup quality
            "claude_setup_grade": analysis_json.get("setup_quality"),

            # Features at entry
            "entry_candle_idx": entry_candle_idx,
            "features": features,
        }

    def find_entry_candle(self, candles: list, entry_price: float,
                          direction: str) -> int:
        """Find the candle closest to Claude's entry price in the last 20 candles.

        For longs: find candle whose low is nearest to entry_price.
        For shorts: find candle whose high is nearest to entry_price.
        """
        if not candles or not entry_price:
            return len(candles) - 1 if candles else -1

        search_start = max(0, len(candles) - 20)
        best_idx = len(candles) - 1
        best_dist = float("inf")

        for i in range(search_start, len(candles)):
            if direction == "long":
                dist = abs(candles[i]["low"] - entry_price)
            else:
                dist = abs(candles[i]["high"] - entry_price)

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def classify_setup_type(self, parsed: dict) -> str:
        """Build Bayesian segmentation key from parsed analysis.

        Format: {bull|bear}_{ob_}{fvg_}{sweep_}{session}
        Example: "bull_ob_fvg_sweep_london"
        """
        direction = parsed.get("claude_direction", "long")
        prefix = "bull" if direction == "long" else "bear"

        parts = [prefix]
        if parsed.get("has_ob"):
            parts.append("ob")
        if parsed.get("has_fvg"):
            parts.append("fvg")
        if parsed.get("liq_swept"):
            parts.append("sweep")

        # Map killzone to session
        session = self._map_killzone_to_session(
            parsed.get("claude_killzone", "")
        )
        parts.append(session)

        return "_".join(parts)

    def log_completed_trade(self, original_analysis: dict,
                            calibrated_result: dict,
                            actual_outcome: str,
                            actual_pnl_atr: float,
                            used_calibrated_sl: bool,
                            notes: str = "",
                            source: str = "live") -> dict:
        """Log a completed trade and update accuracy tracking.

        Args:
            original_analysis: Claude's parsed analysis
            calibrated_result: MLCalibrator output
            actual_outcome: tp1/tp2/tp3/stopped_out/breakeven/manual_close
            actual_pnl_atr: actual PnL in ATR units
            used_calibrated_sl: whether trader used the calibrated SL
            notes: optional trade notes

        Returns:
            Summary dict with updated stats
        """
        win_outcomes = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}
        is_win = actual_outcome in win_outcomes

        # Claude's original levels
        claude_sl = calibrated_result.get("claude_original", {}).get("sl", 0)
        calibrated_sl = calibrated_result.get("calibrated", {}).get("sl", 0)
        entry = calibrated_result.get("claude_original", {}).get("entry", 0)
        direction = calibrated_result.get("claude_original", {}).get("direction", "long")

        claude_sl_dist = abs(entry - claude_sl) if entry and claude_sl else 0
        calibrated_sl_dist = abs(entry - calibrated_sl) if entry and calibrated_sl else 0

        # Determine if SLs would have been hit
        # We approximate: if actual PnL > 0, neither SL was hit
        # If stopped out, check which SLs would have survived
        claude_survived = actual_outcome not in {"stopped_out"}
        calibrated_survived = claude_survived

        # If stopped out with calibrated SL that was wider, it might have survived
        trade_saved = False
        if actual_outcome == "stopped_out" and calibrated_sl_dist > claude_sl_dist:
            # Calibrated SL was wider — might have saved the trade
            # This is approximate; real tracking requires candle-by-candle sim
            trade_saved = True
            calibrated_survived = True

        # Update accuracy tracker
        a = self._accuracy
        a["total_trades"] += 1

        if is_win:
            a["claude_direction_correct"] += 1
        if claude_survived:
            a["claude_sl_would_survive"] += 1
        if calibrated_survived:
            a["calibrated_sl_survived"] += 1
        if trade_saved:
            a["trades_saved_by_calibration"] += 1

        atr = calibrated_result.get("volatility_context", {}).get("atr_14", 1.0)
        claude_tp1 = calibrated_result.get("claude_original", {}).get("tps", [None])[0]
        cal_tp1 = calibrated_result.get("calibrated", {}).get("tps", [None])[0]

        if claude_tp1 and actual_pnl_atr >= 0 and is_win:
            a["claude_tp1_reached"] += 1
        if cal_tp1 and actual_pnl_atr >= 0 and is_win:
            a["calibrated_tp1_reached"] += 1

        # Running averages
        n = a["total_trades"]
        a["avg_claude_sl_distance_atr"] = (
            (a["avg_claude_sl_distance_atr"] * (n - 1) + (claude_sl_dist / atr if atr > 0 else 0)) / n
        )
        a["avg_calibrated_sl_distance_atr"] = (
            (a["avg_calibrated_sl_distance_atr"] * (n - 1) + (calibrated_sl_dist / atr if atr > 0 else 0)) / n
        )

        widening = calibrated_sl_dist - claude_sl_dist
        widening_atr = widening / atr if atr > 0 else 0
        a["avg_sl_widening_atr"] = (
            (a["avg_sl_widening_atr"] * (n - 1) + widening_atr) / n
        )

        # Per-session tracking
        session = calibrated_result.get("session_context", {}).get("session", "off")
        if session not in a["by_session"]:
            a["by_session"][session] = {"trades": 0, "claude_survived": 0, "calibrated_survived": 0}
        a["by_session"][session]["trades"] += 1
        if claude_survived:
            a["by_session"][session]["claude_survived"] += 1
        if calibrated_survived:
            a["by_session"][session]["calibrated_survived"] += 1

        # Per-session TP distribution (for multi3 session profiling)
        tp_dist = a["by_session"][session].setdefault(
            "tp_distribution", {"tp1": 0, "tp2": 0, "tp3": 0, "sl": 0})
        outcome_key = actual_outcome.replace("_hit", "")
        if outcome_key in tp_dist:
            tp_dist[outcome_key] += 1
        elif actual_outcome == "stopped_out":
            tp_dist["sl"] += 1

        # Per-setup-type tracking
        setup_type = original_analysis.get("setup_type", "unknown")
        if isinstance(original_analysis, dict) and "claude_direction" in original_analysis:
            setup_type = self.classify_setup_type(original_analysis)
        if setup_type not in a["by_setup_type"]:
            a["by_setup_type"][setup_type] = {"trades": 0, "wins": 0}
        a["by_setup_type"][setup_type]["trades"] += 1
        if is_win:
            a["by_setup_type"][setup_type]["wins"] += 1

        self._save_accuracy()

        # Update Bayesian beliefs from live trades and scanner auto-resolves.
        # Only backtest/WFO synthetic data is excluded (inflated win rates).
        if source in ("live", "scanner"):
            try:
                from ml.bayesian import update_beliefs
                from ml.database import TradeLogger
                db = TradeLogger()
                state = db.get_bayesian_state()
                if not state:
                    # Auto-initialize with uninformative priors if never seeded
                    state = {
                        "alpha": 1, "beta_param": 1,
                        "consecutive_losses": 0, "max_consecutive_losses": 0,
                        "current_drawdown": 0.0, "max_drawdown": 0.0,
                        "total_trades": 0, "total_wins": 0, "total_losses": 0,
                        "cumulative_pnl": 0.0, "peak_pnl": 0.0,
                    }
                pnl = actual_pnl_atr * atr  # convert to price units
                outcome_map = {
                    "tp1": "tp1_hit", "tp2": "tp2_hit", "tp3": "tp3_hit",
                    "tp1_hit": "tp1_hit", "tp2_hit": "tp2_hit", "tp3_hit": "tp3_hit",
                    "stopped_out": "stopped_out",
                }
                mapped = outcome_map.get(actual_outcome, "stopped_out")
                posterior = update_beliefs(state, mapped, pnl)
                db.save_bayesian_state(posterior)
            except Exception:
                pass

        # Ingest into training dataset
        try:
            from ml.dataset import TrainingDatasetManager
            dm = TrainingDatasetManager()
            features = original_analysis.get("features", {})
            # If parsed analysis has no engineered features, build minimal
            # feature dict from the calibration metadata so the trade still
            # gets ingested (critical for AutoGluon retraining).
            if not features:
                features = self._build_minimal_features(
                    original_analysis, calibrated_result
                )
            if features:
                dm.ingest_live_trade(
                    features, actual_outcome,
                    mfe=max(0, actual_pnl_atr) if is_win else 0,
                    mae=abs(actual_pnl_atr) if not is_win else 0,
                    pnl=actual_pnl_atr,
                )
        except Exception:
            pass

        return {
            "trade_logged": True,
            "claude_survived": claude_survived,
            "calibrated_survived": calibrated_survived,
            "trade_saved_by_calibration": trade_saved,
            "total_trades": a["total_trades"],
            "calibration_survival_rate": (
                a["calibrated_sl_survived"] / a["total_trades"]
                if a["total_trades"] > 0 else 0
            ),
        }

    def get_calibration_value(self) -> dict:
        """Calculate how much value the ML calibration is adding."""
        a = self._accuracy
        total = a["total_trades"]

        if total == 0:
            return {
                "total_trades": 0,
                "claude_alone_survival_rate": 0,
                "calibrated_survival_rate": 0,
                "trades_saved": 0,
                "survival_improvement": "0%",
                "avg_sl_widening": "N/A",
                "best_session": "N/A",
                "worst_session": "N/A",
                "recommendation": "No trades logged yet — start trading to measure calibration value",
            }

        claude_rate = a["claude_sl_would_survive"] / total
        cal_rate = a["calibrated_sl_survived"] / total
        improvement = cal_rate - claude_rate

        atr_widen = a["avg_sl_widening_atr"]

        # Find best/worst sessions
        best_session = "N/A"
        worst_session = "N/A"
        best_improvement = -1
        worst_improvement = 1

        for session, data in a["by_session"].items():
            if data["trades"] < 2:
                continue
            s_claude = data["claude_survived"] / data["trades"]
            s_cal = data["calibrated_survived"] / data["trades"]
            imp = s_cal - s_claude
            if imp > best_improvement:
                best_improvement = imp
                best_session = session
            if imp < worst_improvement:
                worst_improvement = imp
                worst_session = session

        if improvement > 0.1:
            rec = "Calibration is adding significant value — continue using calibrated SL levels"
        elif improvement > 0.05:
            rec = "Calibration is adding moderate value — calibrated SLs help in volatile sessions"
        elif improvement > 0:
            rec = "Calibration adds marginal value — consider tightening calibration parameters"
        else:
            rec = "Calibration is not improving survival rate — review calibration logic"

        return {
            "total_trades": total,
            "claude_alone_survival_rate": round(claude_rate, 3),
            "calibrated_survival_rate": round(cal_rate, 3),
            "trades_saved": a["trades_saved_by_calibration"],
            "survival_improvement": f"+{improvement:.1%}" if improvement >= 0 else f"{improvement:.1%}",
            "avg_sl_widening": f"{atr_widen:.1f} ATR",
            "best_session": best_session,
            "worst_session": worst_session,
            "recommendation": rec,
        }

    def _map_killzone_to_session(self, killzone: str) -> str:
        """Map various killzone string formats to standard session names."""
        kz = killzone.lower().strip()

        if any(x in kz for x in ["london am", "london open", "london"]):
            return "london"
        if any(x in kz for x in ["new york am", "ny am", "ny open", "new york"]):
            return "ny_am"
        if any(x in kz for x in ["new york pm", "ny pm", "ny afternoon"]):
            return "ny_pm"
        if any(x in kz for x in ["asian", "asia", "tokyo", "sydney"]):
            return "asia"

        return "off"

    def _to_candle_list(self, candles_df) -> list:
        """Convert candles input to list of dicts."""
        if isinstance(candles_df, list):
            return candles_df
        if hasattr(candles_df, "to_dict"):
            return candles_df.to_dict("records")
        return []

    def _load_accuracy(self) -> dict:
        """Load Claude accuracy tracker from disk."""
        if os.path.exists(self._accuracy_path):
            try:
                with open(self._accuracy_path) as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "total_trades": 0,
            "claude_direction_correct": 0,
            "claude_sl_would_survive": 0,
            "calibrated_sl_survived": 0,
            "trades_saved_by_calibration": 0,
            "claude_tp1_reached": 0,
            "calibrated_tp1_reached": 0,
            "avg_claude_sl_distance_atr": 0.0,
            "avg_calibrated_sl_distance_atr": 0.0,
            "avg_sl_widening_atr": 0.0,
            "by_session": {},
            "by_setup_type": {},
        }

    def _make_initial_bucket(self) -> dict:
        """Create a fresh weight bucket with initial values for all narrative fields."""
        initial = self.cfg.get("narrative_ema_initial", 0.5)
        return {f: {"weight": initial, "total": 0} for f in self.NARRATIVE_FIELDS}

    def _load_narrative_weights(self) -> dict:
        """Load per-field EMA weights from disk (segmented by killzone).

        Handles migration from flat format to per-killzone format.
        Flat format: {field: {weight, total}, ...}
        Segmented format: {"_global": {field: {weight, total}}, "Asian": {...}, ...}
        """
        data = None
        if os.path.exists(self._narrative_weights_path):
            try:
                with open(self._narrative_weights_path) as f:
                    data = json.load(f)
            except Exception:
                pass

        if not data:
            result = {"_global": self._make_initial_bucket()}
            for kz in self.KILLZONE_KEYS:
                result[kz] = self._make_initial_bucket()
            return result

        # Detect flat format: first key is a narrative field name, not _global/killzone
        first_key = next(iter(data), "")
        if first_key in self.NARRATIVE_FIELDS:
            # Migrate: wrap existing data as _global
            result = {"_global": data}
            for kz in self.KILLZONE_KEYS:
                result[kz] = self._make_initial_bucket()
            self._narrative_weights = result
            self._save_narrative_weights()
            return result

        # Already segmented — ensure all killzones exist
        if "_global" not in data:
            data["_global"] = self._make_initial_bucket()
        for kz in self.KILLZONE_KEYS:
            if kz not in data:
                data[kz] = self._make_initial_bucket()
        return data

    def _save_narrative_weights(self):
        """Persist narrative weights to disk."""
        os.makedirs(os.path.dirname(self._narrative_weights_path), exist_ok=True)
        with open(self._narrative_weights_path, "w") as f:
            json.dump(self._narrative_weights, f, indent=2)

    def get_narrative_weights(self, killzone: str = None) -> dict:
        """Return EMA weights per narrative field, optionally for a specific killzone.

        Falls back to _global if the killzone bucket has insufficient data.
        """
        min_kz = self.cfg.get("narrative_min_kz_trades", 10)

        if killzone and killzone in self._narrative_weights:
            bucket = self._narrative_weights[killzone]
            # Check if this killzone has enough data
            totals = [v.get("total", 0) if isinstance(v, dict) else 0
                      for v in bucket.values()]
            if totals and min(totals) >= min_kz:
                return {k: v.get("weight", 0.5) if isinstance(v, dict) else v
                        for k, v in bucket.items()}

        # Fall back to global
        bucket = self._narrative_weights.get("_global", {})
        return {k: v.get("weight", 0.5) if isinstance(v, dict) else v
                for k, v in bucket.items()}

    def update_narrative_field_weights(self, narrative_json: dict,
                                        entry_direction: str, is_win: bool,
                                        outcome: str, setup: dict,
                                        mfe_atr: float | None = None,
                                        killzone: str | None = None):
        """Update EMA weight per narrative field based on trade outcome.

        Each field has its own correctness test. EMA update:
        weight = alpha * was_correct + (1 - alpha) * previous_weight

        Updates both _global and killzone-specific buckets when killzone is provided.

        MFE-aware scoring: when a loss has high MFE (≥1.0 ATR), the narrative
        was directionally correct but execution failed (SL too tight, bad
        entry timing, etc.). Aligned fields get boosted partial credit (0.6)
        instead of the flat 0.3, so the system doesn't wrongly penalise
        accurate narratives for execution-related losses.
        """
        alpha = self.cfg.get("narrative_ema_alpha", 0.15)

        # MFE-based loss classification:
        # Type 2 loss = narrative was right, execution failed (MFE ≥ 1.0 ATR)
        # Type 1 loss = wrong narrative (MFE < 0.5 ATR)
        # In-between = ambiguous, use default scoring
        _mfe = mfe_atr or 0.0
        is_type2_loss = not is_win and _mfe >= 1.0  # right direction, bad execution
        is_type1_loss = not is_win and _mfe < 0.5   # wrong narrative

        # Partial credit for aligned-but-lost fields:
        # Type 2 (execution fail): 0.6 — narrative was right, don't punish it
        # Type 1 (wrong narrative): 0.15 — narrative was wrong, penalise harder
        # Default/ambiguous: 0.3 — existing behaviour
        aligned_loss_score = 0.6 if is_type2_loss else (0.15 if is_type1_loss else 0.3)

        # Per-field correctness evaluation
        bias = narrative_json.get("directional_bias", "")
        phase = narrative_json.get("power_of_3_phase", "")
        pd_zone = narrative_json.get("premium_discount", "")
        conf = narrative_json.get("phase_confidence", "")
        intermarket = narrative_json.get("intermarket_synthesis")
        key_levels = narrative_json.get("key_levels", [])
        entry_price = setup.get("entry_price", 0)

        field_scores = {}  # float 0.0-1.0 (partial credit, not binary)

        # directional_bias: aligned with entry AND won = full credit
        bias_aligned = (
            (bias == "bullish" and entry_direction == "long") or
            (bias == "bearish" and entry_direction == "short")
        )
        if bias_aligned and is_win:
            field_scores["directional_bias"] = 1.0
        elif bias_aligned and not is_win:
            field_scores["directional_bias"] = aligned_loss_score
        elif not bias_aligned and is_win:
            field_scores["directional_bias"] = 0.2  # won despite wrong bias
        else:
            field_scores["directional_bias"] = 0.0

        # p3_phase: phase aligns with direction
        phase_aligned = (
            (phase == "distribution" and entry_direction == "short") or
            (phase == "accumulation" and entry_direction == "long") or
            (phase == "manipulation")  # transitional = neutral
        )
        if phase_aligned and is_win:
            field_scores["p3_phase"] = 1.0
        elif phase_aligned and not is_win:
            field_scores["p3_phase"] = aligned_loss_score
        elif phase == "manipulation":
            field_scores["p3_phase"] = 0.5  # neutral phase
        elif not phase_aligned and is_win:
            field_scores["p3_phase"] = 0.2
        else:
            field_scores["p3_phase"] = 0.0

        # premium_discount: zone aligns with ICT direction
        pd_aligned = (
            (pd_zone == "premium" and entry_direction == "short") or
            (pd_zone == "discount" and entry_direction == "long")
        )
        if pd_zone == "equilibrium":
            field_scores["premium_discount"] = 0.5  # neutral, don't punish
        elif pd_aligned and is_win:
            field_scores["premium_discount"] = 1.0
        elif pd_aligned and not is_win:
            field_scores["premium_discount"] = aligned_loss_score
        elif not pd_aligned and is_win:
            field_scores["premium_discount"] = 0.2  # won despite wrong zone
        else:
            field_scores["premium_discount"] = 0.0

        # confidence_calibration: high conf → win, low conf → loss
        if (conf == "high" and is_win) or (conf == "low" and not is_win):
            field_scores["confidence_calibration"] = 1.0
        elif conf == "medium":
            field_scores["confidence_calibration"] = 0.5  # neutral
        elif conf == "high" and not is_win:
            field_scores["confidence_calibration"] = 0.1  # overconfident
        elif conf == "low" and is_win:
            field_scores["confidence_calibration"] = 0.3  # underconfident
        else:
            field_scores["confidence_calibration"] = 0.5

        # intermarket_synthesis: score based on whether signals were correct,
        # not just whether intermarket data existed
        if intermarket:
            from ml.intermarket_validator import IntermarketValidator
            im_data = setup.get("calibration_json", {})
            if isinstance(im_data, str):
                try:
                    im_data = json.loads(im_data)
                except Exception:
                    im_data = {}
            im_block = im_data.get("intermarket", {})
            field_scores["intermarket_synthesis"] = IntermarketValidator.score_intermarket_signal(
                diverging=im_block.get("gold_dxy_diverging", 0),
                is_win=is_win,
                corr=im_block.get("gold_dxy_corr_20", 0),
                yield_dir=im_block.get("yield_direction", 0),
                direction=entry_direction,
            )
        elif not intermarket:
            field_scores["intermarket_synthesis"] = 0.5  # no data, neutral

        # key_levels: proximity-based partial credit
        best_proximity = float('inf')
        if entry_price and key_levels:
            for level in key_levels:
                price = level.get("price", 0) if isinstance(level, dict) else 0
                if price and entry_price:
                    dist_pct = abs(price - entry_price) / entry_price
                    best_proximity = min(best_proximity, dist_pct)

        if best_proximity <= 0.003 and is_win:
            field_scores["key_levels"] = 1.0
        elif best_proximity <= 0.003 and not is_win:
            field_scores["key_levels"] = 0.4  # level was relevant
        elif best_proximity <= 0.01:
            field_scores["key_levels"] = 0.3  # nearby but not precise
        else:
            field_scores["key_levels"] = 0.2  # don't zero out entirely

        # Apply EMA update with floor of 0.05 — dual-bucket (global + killzone)
        weight_floor = 0.05
        initial = self.cfg.get("narrative_ema_initial", 0.5)
        g_bucket = self._narrative_weights.setdefault("_global", {})

        for field, score in field_scores.items():
            # Update global
            g = g_bucket.setdefault(field, {"weight": initial, "total": 0})
            g["weight"] = max(weight_floor,
                              alpha * score + (1 - alpha) * g["weight"])
            g["total"] = g.get("total", 0) + 1

            # Update killzone-specific
            if killzone and killzone in self.KILLZONE_KEYS:
                kz_bucket = self._narrative_weights.setdefault(killzone, {})
                k = kz_bucket.setdefault(field, {"weight": initial, "total": 0})
                k["weight"] = max(weight_floor,
                                  alpha * score + (1 - alpha) * k["weight"])
                k["total"] = k.get("total", 0) + 1

        self._save_narrative_weights()

    @staticmethod
    def _encode_killzone_from_parsed(parsed: dict) -> int:
        """Encode killzone from parsed analysis to match features.py encoding."""
        kz = (parsed.get("killzone") or parsed.get("session") or "").lower()
        if "london" in kz:
            return 1
        if "new york" in kz or "ny" in kz:
            return 2
        if "asian" in kz or "asia" in kz:
            return 3
        return 0  # Off/unknown

    @staticmethod
    def _encode_htf_bias(parsed: dict) -> int:
        """Encode HTF bias from parsed analysis. bullish=1, bearish=-1, neutral=0."""
        htf_bias = parsed.get("htf_bias", "")
        return {"bullish": 1, "bearish": -1, "neutral": 0}.get(
            htf_bias.lower() if isinstance(htf_bias, str) else "", 0)

    @staticmethod
    def _compute_htf_alignment(parsed: dict, direction: str) -> int:
        """Compute HTF structure alignment. 1=agree, -1=conflict, 0=neutral."""
        htf_bias = parsed.get("htf_bias", "")
        if htf_bias and isinstance(htf_bias, str) and htf_bias.lower() != "neutral":
            htf_dir = "long" if htf_bias.lower() == "bullish" else "short"
            return 1 if htf_dir == direction else -1
        return 0

    @staticmethod
    def _compute_opus_agreement(calibrated: dict, direction: str) -> int:
        """Check Opus-Sonnet directional agreement. 1=agree, 0=disagree/absent."""
        opus = (calibrated or {}).get("opus_narrative") or {}
        opus_bias = opus.get("directional_bias", "")
        if opus_bias and isinstance(opus_bias, str):
            opus_dir = "long" if opus_bias.lower() == "bullish" else (
                "short" if opus_bias.lower() == "bearish" else "")
            return 1 if opus_dir == direction else 0
        return 0

    def _build_minimal_features(self, parsed: dict, calibrated: dict) -> dict:
        """Build a feature dict from parsed analysis + calibration.

        Uses real OB/FVG sub-features from parse_analysis() when available,
        falling back to zeros only when Claude's JSON didn't include them.
        Column names match ml/feature_schema.py exactly.
        """
        vol = calibrated.get("volatility_context", {})
        atr = vol.get("atr_14", 1.0) or 1.0
        conf = calibrated.get("confidence", {})
        direction = parsed.get("claude_direction", "short")
        im = calibrated.get("intermarket", {})

        # ICT context encodings
        pd_raw = parsed.get("premium_discount", "")
        pd_enc = {"premium": 1, "discount": -1, "equilibrium": 0}.get(
            pd_raw.lower() if isinstance(pd_raw, str) else "", 0)

        p3_raw = parsed.get("power_of_3_phase", "")
        p3_enc = {"accumulation": 1, "manipulation": 2, "distribution": 3}.get(
            p3_raw.lower() if isinstance(p3_raw, str) else "", 0)

        sq_raw = parsed.get("setup_quality", "")
        sq_enc = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}.get(
            sq_raw.upper() if isinstance(sq_raw, str) else "", 0)

        # OB/FVG sub-features from parse_analysis() (real data, not zeros)
        _safe_atr = atr if atr > 0 else 1.0
        ob_nearest = parsed.get("ob_nearest_distance", 0)
        ob_avg = parsed.get("ob_avg_size", 0)
        fvg_nearest = parsed.get("fvg_nearest_distance", 0)

        return {
            "ob_count": parsed.get("ob_count", 0),
            "ob_bullish_count": parsed.get("ob_bullish_count", 0),
            "ob_bearish_count": parsed.get("ob_bearish_count", 0),
            "ob_strongest_strength": 0.0,
            "ob_nearest_distance_atr": round(ob_nearest / _safe_atr, 4),
            "ob_avg_size_atr": round(ob_avg / _safe_atr, 4),
            "ob_alignment": 1 if parsed.get("has_ob") else 0,
            "fvg_count": parsed.get("fvg_count", 0),
            "fvg_unfilled_count": parsed.get("fvg_unfilled_count", 0),
            "fvg_nearest_distance_atr": round(fvg_nearest / _safe_atr, 4),
            "fvg_avg_size_atr": 0.0,
            "fvg_alignment": 1 if parsed.get("has_fvg") else 0,
            "liq_buyside_count": 1 if parsed.get("has_bsl") else 0,
            "liq_sellside_count": 1 if parsed.get("has_ssl") else 0,
            "liq_nearest_target_distance_atr": 0.0,
            "liq_nearest_threat_distance_atr": 0.0,
            "risk_reward_tp1": (parsed.get("claude_rr_ratios") or [0])[0],
            "risk_reward_tp2": (parsed.get("claude_rr_ratios") or [0, 0])[1]
                if len(parsed.get("claude_rr_ratios", [])) > 1 else 0,
            "sl_distance_atr": parsed.get("claude_sl_distance_atr", 0),
            "tp1_distance_atr": (parsed.get("claude_tp_distances_atr") or [0])[0],
            "entry_direction": 1 if direction == "long" else -1,
            "bias_direction_match": 1.0,
            "num_confluences": parsed.get("claude_confluence_count", 0),
            "has_ob_fvg_overlap": 0.0,
            "killzone_encoded": self._encode_killzone_from_parsed(parsed),
            "timeframe_encoded": 2.0,
            "atr_14": atr,
            "price_vs_20sma": 0.0,
            "recent_volatility_ratio": 1.0,
            "last_candle_body_atr": 0.0,
            "trend_strength": 0.0,
            "session_hour": 0,
            # ICT context (populated from parsed analysis)
            "premium_discount_encoded": pd_enc,
            "p3_phase_encoded": p3_enc,
            "setup_quality_encoded": sq_enc,
            "claude_direction_encoded": 1 if direction == "long" else -1,
            # Intermarket (from calibration if available, NaN if absent)
            "gold_dxy_corr_20": im.get("gold_dxy_corr_20") if im else float('nan'),
            "gold_dxy_diverging": 1 if im.get("gold_dxy_diverging") else 0,
            "dxy_range_position": im.get("dxy_range_position") if im else float('nan'),
            "yield_direction": im.get("yield_direction") if im else float('nan'),
            # Regime — from calibration's structural regime, NaN if unavailable
            "volatility_regime": vol.get("structural_regime", float('nan')),
            # Entry zone — NaN in minimal path (no OB data available)
            "entry_zone_position": float('nan'),
            "entry_zone_size_atr": float('nan'),
            # HTF context (5) — from parsed htf fields + calibration opus_narrative
            "htf_bias_encoded": self._encode_htf_bias(parsed),
            "htf_sweep_encoded": 0,  # Not available in minimal path
            "dealing_range_position": float('nan'),  # No dealing range in minimal
            "htf_structure_alignment": self._compute_htf_alignment(parsed, direction),
            "htf_displacement_quality": 0,  # Not computable in minimal path
            # Narrative state (4) — NaN/0 in minimal path (no narrative_state)
            "thesis_confidence": float('nan'),
            "p3_progress_encoded": 0,
            "thesis_scan_count": 0,
            "opus_sonnet_agreement": self._compute_opus_agreement(calibrated, direction),
        }

    def update_narrative_tracker(self, narrative_bias: str | None,
                                  entry_direction: str, is_win: bool,
                                  killzone: str | None = None,
                                  phase: str | None = None):
        """Track whether Opus HTF narratives align with trade outcomes.

        Args:
            narrative_bias: Opus's directional_bias ("bullish"/"bearish"/"neutral"/None)
            entry_direction: The actual trade direction ("long"/"short")
            is_win: Whether the trade won
            killzone: Optional killzone name for segment tracking
            phase: Optional Power of 3 phase for segment tracking
        """
        tracker = self._accuracy.setdefault("narrative_tracker", {
            "total_with_narrative": 0,
            "total_without_narrative": 0,
            "aligned_trades": 0, "misaligned_trades": 0,
            "aligned_wins": 0, "aligned_losses": 0,
            "misaligned_wins": 0, "misaligned_losses": 0,
            "no_narrative_wins": 0, "no_narrative_losses": 0,
        })

        if not narrative_bias or narrative_bias == "neutral":
            tracker["total_without_narrative"] += 1
            if is_win:
                tracker["no_narrative_wins"] = tracker.get("no_narrative_wins", 0) + 1
            else:
                tracker["no_narrative_losses"] = tracker.get("no_narrative_losses", 0) + 1
        else:
            tracker["total_with_narrative"] += 1
            aligned = (
                (narrative_bias == "bullish" and entry_direction == "long") or
                (narrative_bias == "bearish" and entry_direction == "short")
            )
            if aligned:
                tracker["aligned_trades"] += 1
                if is_win:
                    tracker["aligned_wins"] = tracker.get("aligned_wins", 0) + 1
                else:
                    tracker["aligned_losses"] = tracker.get("aligned_losses", 0) + 1
            else:
                tracker["misaligned_trades"] += 1
                if is_win:
                    tracker["misaligned_wins"] = tracker.get("misaligned_wins", 0) + 1
                else:
                    tracker["misaligned_losses"] = tracker.get("misaligned_losses", 0) + 1

        # Derived win rates
        a_total = tracker.get("aligned_wins", 0) + tracker.get("aligned_losses", 0)
        m_total = tracker.get("misaligned_wins", 0) + tracker.get("misaligned_losses", 0)
        n_total = tracker.get("no_narrative_wins", 0) + tracker.get("no_narrative_losses", 0)
        tracker["aligned_win_rate"] = round(
            tracker.get("aligned_wins", 0) / a_total, 3) if a_total > 0 else 0
        tracker["misaligned_win_rate"] = round(
            tracker.get("misaligned_wins", 0) / m_total, 3) if m_total > 0 else 0
        tracker["no_narrative_win_rate"] = round(
            tracker.get("no_narrative_wins", 0) / n_total, 3) if n_total > 0 else 0

        # Killzone×phase cross-tab tracking
        if killzone and phase:
            seg_key = f"{killzone}_{phase}"
            kp = tracker.setdefault("by_killzone_phase", {})
            seg = kp.setdefault(seg_key, {"total": 0, "wins": 0, "losses": 0})
            seg["total"] += 1
            if is_win:
                seg["wins"] += 1
            else:
                seg["losses"] += 1
            seg["win_rate"] = round(seg["wins"] / seg["total"], 3)

        self._save_accuracy()

    def get_narrative_trust_by_segment(self, min_trades: int = 10) -> dict:
        """Return per-killzone×phase accuracy stats for segments with enough data.

        Args:
            min_trades: Minimum trades in segment to include.

        Returns:
            Dict of segment_key → {total, wins, losses, win_rate}
        """
        tracker = self._accuracy.get("narrative_tracker", {})
        kp = tracker.get("by_killzone_phase", {})
        return {k: v for k, v in kp.items() if v.get("total", 0) >= min_trades}

    def backfill_killzone_weights(self, db) -> dict:
        """Replay resolved trades to warm up per-killzone EMA weight buckets.

        Returns:
            Summary dict with per-killzone trade counts.
        """
        import logging
        _log = logging.getLogger(__name__)

        setups = db.get_resolved_setups() if hasattr(db, "get_resolved_setups") else []
        if not setups:
            # Fallback: try get_training_data
            try:
                df = db.get_training_data()
                _log.info("Backfill: got %d rows from training data", len(df))
            except Exception:
                return {"status": "no_data"}
            return {"status": "no_resolved_setups"}

        # Reset killzone buckets (keep _global intact)
        saved_global = deepcopy(self._narrative_weights.get("_global", {}))
        for kz in self.KILLZONE_KEYS:
            self._narrative_weights[kz] = self._make_initial_bucket()

        counts = {kz: 0 for kz in self.KILLZONE_KEYS}
        WIN_OUTCOMES = {"tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "tp3_hit"}

        for setup in setups:
            cal_json = setup.get("calibration_json") or {}
            if isinstance(cal_json, str):
                try:
                    cal_json = json.loads(cal_json)
                except Exception:
                    continue
            narrative_json = cal_json.get("opus_narrative", {})
            if not narrative_json:
                continue

            kz = setup.get("killzone", "")
            if kz not in self.KILLZONE_KEYS:
                continue

            result = setup.get("actual_result", "")
            is_win = result in WIN_OUTCOMES
            entry_dir = setup.get("direction", "")
            mfe_atr = setup.get("mfe_atr")

            # Replay EMA update with killzone (also touches _global, restored below)
            self.update_narrative_field_weights(
                narrative_json, entry_dir, is_win, result, setup,
                mfe_atr=mfe_atr, killzone=kz)
            counts[kz] = counts.get(kz, 0) + 1

        # Restore _global to pre-backfill state (backfill only warms killzone buckets)
        self._narrative_weights["_global"] = saved_global
        self._save_narrative_weights()

        _log.info("Backfill killzone weights: %s", counts)
        return {"status": "ok", "counts": counts}

    def update_prospect_tracker(self, zone_type: str, was_reached: bool,
                                  was_triggered: bool, is_win: bool | None = None):
        """Track prospect zone accuracy.

        Args:
            zone_type: 'ob', 'fvg', 'liquidity', or 'conditional'
            was_reached: Did price reach the identified zone?
            was_triggered: Did the conditional setup actually trigger?
            is_win: If triggered, did the resulting trade win?
        """
        tracker = self._accuracy.setdefault("prospect_tracker", {
            "total_prospects": 0,
            "zones_reached": 0,
            "triggers_fired": 0,
            "trigger_wins": 0,
            "trigger_losses": 0,
            "by_zone_type": {},
        })

        tracker["total_prospects"] += 1
        if was_reached:
            tracker["zones_reached"] += 1
        if was_triggered:
            tracker["triggers_fired"] += 1
            if is_win is True:
                tracker["trigger_wins"] += 1
            elif is_win is False:
                tracker["trigger_losses"] += 1

        # Per zone type
        zt = tracker.setdefault("by_zone_type", {}).setdefault(zone_type, {
            "total": 0, "reached": 0, "triggered": 0, "wins": 0})
        zt["total"] += 1
        if was_reached:
            zt["reached"] += 1
        if was_triggered:
            zt["triggered"] += 1
        if is_win is True:
            zt["wins"] += 1

        self._save_accuracy()

    def update_opus_tracker(self, verdict: str, is_win: bool,
                            killzone: str = None, timeframe: str = None,
                            confidence: float = 0.5, direction: str = None,
                            pnl_rr: float = 0.0):
        """Track Opus validation outcomes for value measurement.

        Called for EVERY resolved trade that had Opus validation (including shadow).

        Args:
            verdict: "validated", "downgraded", or "rejected"
            is_win: True if trade would have / did hit TP
            killzone: Session killzone (e.g. "london", "ny_am")
            timeframe: Timeframe string (e.g. "1h", "4h")
            confidence: Opus confidence score (0.0–1.0)
            direction: "long" or "short"
            pnl_rr: Actual P&L in R multiples
        """
        tracker = self._accuracy.setdefault("opus_tracker", {})
        # Ensure all aggregate keys exist (handles partial / pre-events trackers)
        for _key, _default in [
            ("total_validations", 0), ("validated", 0), ("downgraded", 0),
            ("rejected", 0), ("validated_wins", 0), ("validated_losses", 0),
            ("rejected_would_have_won", 0), ("rejected_would_have_lost", 0),
            ("downgraded_wins", 0), ("downgraded_losses", 0),
        ]:
            tracker.setdefault(_key, _default)
        tracker["total_validations"] += 1
        tracker[verdict] = tracker.get(verdict, 0) + 1

        if verdict == "validated":
            if is_win:
                tracker["validated_wins"] = tracker.get("validated_wins", 0) + 1
            else:
                tracker["validated_losses"] = tracker.get("validated_losses", 0) + 1
        elif verdict == "rejected":
            if is_win:
                tracker["rejected_would_have_won"] = tracker.get("rejected_would_have_won", 0) + 1
            else:
                tracker["rejected_would_have_lost"] = tracker.get("rejected_would_have_lost", 0) + 1
        elif verdict == "downgraded":
            if is_win:
                tracker["downgraded_wins"] = tracker.get("downgraded_wins", 0) + 1
            else:
                tracker["downgraded_losses"] = tracker.get("downgraded_losses", 0) + 1

        # Compute derived stats
        v_total = tracker.get("validated_wins", 0) + tracker.get("validated_losses", 0)
        tracker["validated_win_rate"] = round(
            tracker.get("validated_wins", 0) / v_total, 3) if v_total > 0 else 0

        # ── Segmented event log (rolling 30-day window) ──
        event = {
            "verdict": verdict,
            "is_win": is_win,
            "killzone": killzone or "Off",
            "timeframe": timeframe or "?",
            "confidence": confidence,
            "direction": direction or "?",
            "pnl_rr": pnl_rr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        events = tracker.setdefault("events", [])
        events.append(event)

        # Prune events older than 30 days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        tracker["events"] = [e for e in events if e["timestamp"] >= cutoff]

        self._save_accuracy()

    def get_opus_rejection_policy(self, min_resolved: int = 10,
                                   killzone: str = None,
                                   timeframe: str = None) -> dict:
        """Determine whether Opus rejections should be rejected, downgraded, or allowed.

        Three-tier policy based on false negative rate among resolved rejections:
          - fn_rate < 0.35  → action: "reject"    (Opus is accurate here)
          - fn_rate < 0.60  → action: "downgrade"  (convert reject → C grade)
          - fn_rate >= 0.60 → action: "allow"      (ignore rejection, treat as validated)

        When killzone (and optionally timeframe) is provided, segment-specific
        event data is used if ≥ min_resolved rejection events exist for that segment.
        Falls back to global aggregate stats otherwise.

        Args:
            min_resolved: Minimum rejection events before segment-specific policy applies.
            killzone: Filter events to this killzone (e.g. "london", "ny_am").
            timeframe: Optional further filter (e.g. "1h").

        Returns:
            dict with keys: action, false_negative_rate, weighted_fn_rate,
            high_confidence_errors, segment, sample_size, total_resolved_rejections.
        """
        tracker = self._accuracy.get("opus_tracker", {})

        # ── Attempt segment-specific policy from events ──────────────────
        if killzone:
            events = tracker.get("events", [])
            seg_events = [e for e in events if e.get("killzone") == killzone]
            if timeframe:
                seg_events = [e for e in seg_events if e.get("timeframe") == timeframe]

            seg_rejections = [e for e in seg_events if e.get("verdict") == "rejected"]

            if len(seg_rejections) >= min_resolved:
                return self._policy_from_events(
                    seg_rejections,
                    segment=f"{killzone}_{timeframe}" if timeframe else killzone,
                )

        # ── Fallback: global aggregate stats (backward compatible) ────────
        won = tracker.get("rejected_would_have_won", 0)
        lost = tracker.get("rejected_would_have_lost", 0)
        total_resolved = won + lost

        if total_resolved < min_resolved:
            return {
                "action": "reject",
                "false_negative_rate": 0.0,
                "weighted_fn_rate": 0.0,
                "high_confidence_errors": False,
                "segment": "global",
                "sample_size": total_resolved,
                "total_resolved_rejections": total_resolved,
            }

        fn_rate = round(won / total_resolved, 3)
        action = self._fn_rate_to_action(fn_rate)

        result = {
            "action": action,
            "false_negative_rate": fn_rate,
            "weighted_fn_rate": fn_rate,   # no confidence data in global agg
            "high_confidence_errors": False,
            "segment": "global",
            "sample_size": total_resolved,
            "total_resolved_rejections": total_resolved,
        }
        if action == "downgrade":
            result["downgrade_to"] = "C"
        return result

    def _fn_rate_to_action(self, fn_rate: float) -> str:
        """Convert a false-negative rate to a policy action."""
        if fn_rate >= 0.60:
            return "allow"
        if fn_rate >= 0.35:
            return "downgrade"
        return "reject"

    def _policy_from_events(self, rejections: list, segment: str) -> dict:
        """Compute policy dict from a list of rejection event dicts."""
        wins = [e for e in rejections if e.get("is_win")]
        fn_rate = round(len(wins) / len(rejections), 3) if rejections else 0.0

        # Confidence-weighted FN rate
        total_conf = sum(e.get("confidence", 0.5) for e in rejections)
        if total_conf > 0:
            weighted_fn = round(
                sum(e.get("confidence", 0.5) * e.get("is_win", False)
                    for e in rejections) / total_conf, 3)
        else:
            weighted_fn = fn_rate

        high_conf_errors = (weighted_fn - fn_rate) > 0.15

        action = self._fn_rate_to_action(fn_rate)
        result = {
            "action": action,
            "false_negative_rate": fn_rate,
            "weighted_fn_rate": weighted_fn,
            "high_confidence_errors": high_conf_errors,
            "segment": segment,
            "sample_size": len(rejections),
            "total_resolved_rejections": len(rejections),
        }
        if action == "downgrade":
            result["downgrade_to"] = "C"
        return result

    def build_opus_rejection_context(self) -> str:
        """Build a human-readable rejection track record for the Opus prompt.

        Shows per-killzone false negative rates and R lost, so Opus can
        calibrate how strict its rejection threshold should be in each session.

        Returns:
            Formatted string (non-empty when ≥5 rejections exist in any killzone),
            or empty string when insufficient data.
        """
        tracker = self._accuracy.get("opus_tracker", {})
        events = tracker.get("events", [])
        rejections = [e for e in events if e.get("verdict") == "rejected"]

        if len(rejections) < 5:
            return ""

        # Group by killzone
        kz_map: dict[str, list] = {}
        for e in rejections:
            kz = e.get("killzone", "Off")
            kz_map.setdefault(kz, []).append(e)

        # Filter to killzones with ≥5 rejections
        qualifying = {kz: evts for kz, evts in kz_map.items() if len(evts) >= 5}
        if not qualifying:
            return ""

        lines = ["YOUR REJECTION ACCURACY BY SESSION (last 30 days):"]
        for kz, evts in sorted(qualifying.items()):
            wins = [e for e in evts if e.get("is_win")]
            fn_rate = len(wins) / len(evts)
            net_r = sum(e.get("pnl_rr", 0) for e in wins)

            fn_pct = f"{fn_rate:.0%}"
            net_r_str = f"+{net_r:.2f}R" if net_r >= 0 else f"{net_r:.2f}R"

            if fn_rate >= 0.50:
                note = "You are over-filtering here — only reject if structurally certain."
                line = (f"- {kz}: {len(evts)} rejections, {fn_pct} would have won "
                        f"({net_r_str} lost). {note}")
            else:
                line = (f"- {kz}: {len(evts)} rejections, {fn_pct} would have won "
                        f"({net_r_str} lost). Your rejections here are accurate.")

            lines.append(line)

            # ── Layer 4: directional bias note ──────────────────────────
            direction_note = self._build_directional_bias_note(kz, evts)
            if direction_note:
                lines.append(direction_note)

        lines.append(
            "Use this to calibrate how strict your rejection threshold should be per session."
        )
        return "\n".join(lines)

    def _build_directional_bias_note(self, kz: str, evts: list) -> str:
        """Return a directional bias note if long/short FN rates diverge ≥20pp.

        Requires both directions to have ≥5 samples.
        """
        long_evts = [e for e in evts if e.get("direction") == "long"]
        short_evts = [e for e in evts if e.get("direction") == "short"]

        if len(long_evts) < 5 or len(short_evts) < 5:
            return ""

        long_fn = sum(1 for e in long_evts if e.get("is_win")) / len(long_evts)
        short_fn = sum(1 for e in short_evts if e.get("is_win")) / len(short_evts)

        gap = abs(long_fn - short_fn)
        if gap < 0.20:
            return ""

        if long_fn > short_fn:
            return (f"  ↳ Your LONG rejections in {kz} have {long_fn:.0%} FN rate — "
                    f"be particularly cautious rejecting bullish {kz} setups.")
        else:
            return (f"  ↳ Your SHORT rejections in {kz} have {short_fn:.0%} FN rate — "
                    f"be particularly cautious rejecting bearish {kz} setups.")

    def _save_accuracy(self):
        """Persist accuracy tracker to disk."""
        os.makedirs(os.path.dirname(self._accuracy_path), exist_ok=True)
        with open(self._accuracy_path, "w") as f:
            json.dump(self._accuracy, f, indent=2)
