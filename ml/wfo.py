"""Walk-Forward Optimization engine for ICT setups.

Rolling-window cross-validation that detects ICT trade setups from raw candles,
trains per-fold models, evaluates out-of-sample, and produces an aggregate
report with calibrated SL/TP recommendations.

All core logic is pure — side effects (file I/O, DB) are isolated to
save_report(), load_report(), and update_bayesian_from_wfo().
"""
import json
import math
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.config import get_config
from ml.features import (
    compute_atr,
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity,
    compute_market_structure,
    create_trade_labels,
    engineer_features_from_candles,
    detect_swing_points,
    compute_ob_freshness,
    compute_fvg_fill_percentage,
    engineer_htf_features,
    _extract_hour,
    _safe_divide,
)


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class WFOConfig:
    train_window: int = 500
    test_window: int = 100
    step_size: int = 50
    sl_atr_mult: float = 1.5
    tp_atr_mults: list = field(default_factory=lambda: [1.0, 2.0, 3.5])
    max_bars_in_trade: int = 20
    ag_time_limit: int = 120
    ag_presets: str = "good_quality"
    max_folds: int = 20
    displacement_threshold: float = 1.5
    min_confluence_score: int = 2
    min_setups_per_fold: int = 5
    use_retracement_entry: bool = False
    use_mtf: bool = False
    filter_counter_trend: bool = False
    # V2 quality-weighted detection system
    min_quality_score: float = 5.0
    use_quality_scoring: bool = False
    use_rejection_entry: bool = False
    use_narrative_filter: bool = False
    max_bars_to_rejection: int = 15


# Recommended v2 config overrides from diagnostic results
V2_CONFIG_OVERRIDES = {
    "sl_atr_mult": 2.5,
    "use_quality_scoring": True,
    "use_rejection_entry": True,
    "use_mtf": True,
    "use_narrative_filter": True,
    "min_quality_score": 5.0,
    "max_bars_to_rejection": 15,
}


@dataclass
class FoldResult:
    fold_num: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    total_trades: int
    wins: int
    losses: int
    expired: int
    win_rate: float
    avg_rr: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    regime: str
    test_date_start: str = ""
    test_date_end: str = ""
    setup_types: dict = field(default_factory=dict)
    winning_drawdowns: list = field(default_factory=list)
    winning_excursions: list = field(default_factory=list)


@dataclass
class WFOReport:
    total_oos_trades: int
    oos_win_rate: float
    oos_avg_rr: float
    oos_profit_factor: float
    oos_sharpe: float
    oos_max_drawdown: float
    regime_stability: float
    recommended_sl_atr: float
    recommended_tp_atr: list
    grade: str
    folds: list
    fold_count: int
    skipped_folds: int
    setup_type_breakdown: dict
    timestamp: str
    setup_type_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "total_oos_trades": self.total_oos_trades,
            "oos_win_rate": self.oos_win_rate,
            "oos_avg_rr": self.oos_avg_rr,
            "oos_profit_factor": self.oos_profit_factor,
            "oos_sharpe": self.oos_sharpe,
            "oos_max_drawdown": self.oos_max_drawdown,
            "regime_stability": self.regime_stability,
            "recommended_sl_atr": self.recommended_sl_atr,
            "recommended_tp_atr": self.recommended_tp_atr,
            "grade": self.grade,
            "fold_count": self.fold_count,
            "skipped_folds": self.skipped_folds,
            "setup_type_breakdown": self.setup_type_breakdown,
            "timestamp": self.timestamp,
            "setup_type_stats": self.setup_type_stats,
            "folds": [asdict(f) if isinstance(f, FoldResult) else f for f in self.folds],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WFOReport":
        """Deserialize from saved JSON."""
        folds = []
        for fd in d.get("folds", []):
            if isinstance(fd, dict):
                folds.append(FoldResult(**fd))
            else:
                folds.append(fd)
        return cls(
            total_oos_trades=d["total_oos_trades"],
            oos_win_rate=d["oos_win_rate"],
            oos_avg_rr=d["oos_avg_rr"],
            oos_profit_factor=d["oos_profit_factor"],
            oos_sharpe=d["oos_sharpe"],
            oos_max_drawdown=d["oos_max_drawdown"],
            regime_stability=d["regime_stability"],
            recommended_sl_atr=d["recommended_sl_atr"],
            recommended_tp_atr=d["recommended_tp_atr"],
            grade=d["grade"],
            folds=folds,
            fold_count=d["fold_count"],
            skipped_folds=d["skipped_folds"],
            setup_type_breakdown=d["setup_type_breakdown"],
            timestamp=d["timestamp"],
            setup_type_stats=d.get("setup_type_stats", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# Regime detection (for WFO — different from volatility.py ATR-based)
# ═══════════════════════════════════════════════════════════════════════


def detect_regime(candles: list[dict], idx: int) -> str:
    """Classify market regime at a given candle index.

    Uses ratio of 5-bar rolling volatility to 30-bar rolling volatility:
    - ratio > 1.5 → "high_volatility"
    - ratio < 0.6 → "low_volatility"
    - else check 20-bar price direction → trending_up/down or ranging
    """
    if idx < 5 or len(candles) <= idx:
        return "ranging"

    # 5-bar volatility (std of close-to-close returns)
    returns_5 = [abs(candles[j]["close"] - candles[j - 1]["close"])
                 for j in range(max(1, idx - 4), idx + 1)]
    if len(returns_5) < 2:
        return "ranging"
    mean_5 = sum(returns_5) / len(returns_5)
    std_5 = math.sqrt(sum((r - mean_5) ** 2 for r in returns_5) / len(returns_5))

    # 30-bar volatility
    start_30 = max(1, idx - 29)
    returns_30 = [abs(candles[j]["close"] - candles[j - 1]["close"])
                  for j in range(start_30, idx + 1)]
    if len(returns_30) < 2:
        return "ranging"
    mean_30 = sum(returns_30) / len(returns_30)
    std_30 = math.sqrt(sum((r - mean_30) ** 2 for r in returns_30) / len(returns_30))

    if std_30 == 0:
        return "ranging"

    ratio = std_5 / std_30

    if ratio > 1.5:
        return "high_volatility"
    if ratio < 0.6:
        return "low_volatility"

    # Check price direction over 20 bars
    look_start = max(0, idx - 20)
    delta = candles[idx]["close"] - candles[look_start]["close"]
    atr = compute_atr(candles[max(0, idx - 15):idx + 1], 14)
    threshold = atr * 0.5 if atr > 0 else 0

    if delta > threshold:
        return "trending_up"
    if delta < -threshold:
        return "trending_down"
    return "ranging"


# ═══════════════════════════════════════════════════════════════════════
# ICT Setup Detector
# ═══════════════════════════════════════════════════════════════════════


class ICTSetupDetector:
    """Scan raw candle data and detect ICT trade setups with confluence scoring."""

    def __init__(self, wfo_config: WFOConfig = None):
        self.cfg = wfo_config or WFOConfig()

    def detect_setups(self, candles: list[dict],
                      timeframe: str = "1h") -> pd.DataFrame:
        """Scan candles and return DataFrame of detected setups with features + labels.

        Supports two scoring modes controlled by cfg.use_quality_scoring:
        - V1 (False): Original integer confluence scoring
        - V2 (True):  Quality-weighted float scoring with rejection entry
                       and narrative HTF filter

        Each row has engineered features plus outcome labels and setup taxonomy.
        """
        if len(candles) < 60:
            return pd.DataFrame()

        atr = compute_atr(candles, 14)
        if atr <= 0:
            return pd.DataFrame()

        # Detect all structures once upfront
        obs = detect_order_blocks(candles, atr, self.cfg.displacement_threshold)
        fvgs_all = detect_fvgs(candles)
        liqs = detect_liquidity(candles, window=20)

        rows = []
        start_idx = 50
        min_forward = min(self.cfg.max_bars_in_trade, max(5, len(candles) // 5))
        end_idx = len(candles) - min_forward

        # V2: pre-compute swing points for quality scoring
        swings = None
        if self.cfg.use_quality_scoring:
            swings = detect_swing_points(candles, lookback=5)

        # V1: pre-compute 4H structure for MTF alignment
        mtf_scores = {}
        if not self.cfg.use_quality_scoring and (self.cfg.use_mtf or self.cfg.filter_counter_trend):
            for idx in range(start_idx, max(start_idx, end_idx)):
                mtf_scores[idx] = compute_market_structure(
                    candles[:idx + 1], lookback=80
                )

        for idx in range(start_idx, max(start_idx, end_idx)):
            ms_score = compute_market_structure(candles[:idx + 1], lookback=20)

            for direction in ("long", "short"):
                # ── Scoring ──────────────────────────────
                score_breakdown = {}
                best_ob_for_entry = None
                best_fvg_for_entry = None

                if self.cfg.use_quality_scoring:
                    # V2: Quality-weighted scoring
                    score, tags, score_breakdown = self._score_quality(
                        candles, idx, direction, atr, obs, fvgs_all,
                        liqs, ms_score, swings,
                    )
                    if score < self.cfg.min_quality_score:
                        continue
                    # Extract best OB/FVG for rejection entry zone
                    # (re-find the best matching ones for this setup)
                    ob_type = "bullish" if direction == "long" else "bearish"
                    for ob in obs:
                        if ob["index"] < idx and ob["type"] == ob_type:
                            price = candles[idx]["close"]
                            if abs(price - (ob["high"] + ob["low"]) / 2) <= 2.0 * atr:
                                best_ob_for_entry = ob
                                break
                    fvg_type = "bullish" if direction == "long" else "bearish"
                    for fvg in fvgs_all:
                        if fvg["index"] < idx and fvg["type"] == fvg_type:
                            price = candles[idx]["close"]
                            if abs(price - (fvg["high"] + fvg["low"]) / 2) <= 2.0 * atr:
                                best_fvg_for_entry = fvg
                                break
                else:
                    # V1: Original confluence scoring
                    score, tags = self._score_confluence(
                        candles, idx, direction, atr, obs, fvgs_all, liqs, ms_score
                    )
                    if score < self.cfg.min_confluence_score:
                        continue

                # ── V1 MTF alignment filter ──────────────
                if not self.cfg.use_quality_scoring and (self.cfg.use_mtf or self.cfg.filter_counter_trend):
                    htf_score = mtf_scores.get(idx, 0)
                    is_long = direction == "long"
                    if self.cfg.filter_counter_trend:
                        if is_long and htf_score < -0.2:
                            continue
                        if not is_long and htf_score > 0.2:
                            continue
                    elif self.cfg.use_mtf:
                        if is_long and htf_score < -0.3:
                            continue
                        if not is_long and htf_score > 0.3:
                            continue

                # ── V2 Narrative HTF filter ──────────────
                htf_feats = {}
                narrative_reason = ""
                if self.cfg.use_quality_scoring and self.cfg.use_narrative_filter:
                    htf_feats = engineer_htf_features(candles, idx, direction, atr)
                    passes, narrative_reason = self.check_narrative_alignment(
                        direction, htf_feats
                    )
                    if not passes:
                        continue
                elif self.cfg.use_quality_scoring and self.cfg.use_mtf:
                    htf_feats = engineer_htf_features(candles, idx, direction, atr)

                # ── Entry detection ──────────────────────
                entry_idx = idx
                rejection_data = {}
                effective_sl_atr = self.cfg.sl_atr_mult

                if self.cfg.use_quality_scoring and self.cfg.use_rejection_entry:
                    # V2: rejection candle entry
                    rej = self._find_rejection_entry(
                        candles, idx, direction, atr,
                        best_ob=best_ob_for_entry,
                        best_fvg=best_fvg_for_entry,
                        max_forward=min(self.cfg.max_bars_to_rejection, min_forward),
                    )
                    if rej is None:
                        continue  # No rejection = not tradeable
                    entry_idx = rej["idx"]
                    rejection_data = rej
                    # Use wider of structural SL or ATR SL
                    structural_sl_dist = abs(rej["entry_price"] - rej["structural_sl"])
                    structural_sl_atr = structural_sl_dist / atr if atr > 0 else self.cfg.sl_atr_mult
                    effective_sl_atr = max(self.cfg.sl_atr_mult, structural_sl_atr)
                elif self.cfg.use_retracement_entry:
                    # V1: retracement entry
                    entry_idx = self._find_retracement_entry(
                        candles, idx, direction, atr, min_forward
                    )
                    if entry_idx is None:
                        continue

                # ── Label with forward outcome ───────────
                labels = create_trade_labels(
                    candles, entry_idx, direction, atr,
                    sl_atr_mult=effective_sl_atr,
                    tp_atr_mults=self.cfg.tp_atr_mults,
                    max_bars=min_forward,
                )

                # ── Engineer features ────────────────────
                features = engineer_features_from_candles(
                    candles, idx, direction, atr, obs, fvgs_all, liqs, ms_score
                )

                # Append HTF features if available
                if htf_feats:
                    features.update(htf_feats)

                # Build taxonomy
                dir_prefix = "bull" if direction == "long" else "bear"
                taxonomy = (f"{dir_prefix}_{'_'.join(sorted(tags))}"
                            if tags else dir_prefix)

                # Build row
                row = {**features}
                row["outcome"] = labels["outcome"]
                row["max_drawdown_atr"] = labels["max_drawdown_atr"]
                row["max_favorable_atr"] = labels["max_favorable_atr"]
                row["bars_held"] = labels["bars_held"]
                row["won"] = labels["won"]
                row["setup_type"] = taxonomy
                row["direction"] = direction
                row["confluence_score"] = score
                row["candle_index"] = idx

                # V2 quality metadata
                if self.cfg.use_quality_scoring:
                    row["total_quality_score"] = score_breakdown.get("total_quality_score", score)
                    row["ob_score"] = score_breakdown.get("ob_score", 0)
                    row["fvg_score"] = score_breakdown.get("fvg_score", 0)
                    row["liq_score"] = score_breakdown.get("liq_score", 0)
                    row["structure_score_detail"] = score_breakdown.get("structure_score", 0)
                    row["session_score"] = score_breakdown.get("session_score", 0)
                    row["displacement_score"] = score_breakdown.get("displacement_score", 0)
                    if rejection_data:
                        row["entry_type"] = rejection_data.get("entry_type", "direct")
                        row["rejection_quality"] = rejection_data.get("rejection_quality", 0)
                        row["structural_sl"] = rejection_data.get("structural_sl", 0)
                        row["candles_to_rejection"] = rejection_data.get("candles_to_rejection", 0)
                    else:
                        row["entry_type"] = "direct"
                    if narrative_reason:
                        row["narrative_reason"] = narrative_reason

                rows.append(row)

        return pd.DataFrame(rows)

    def _find_retracement_entry(self, candles: list[dict], signal_idx: int,
                                direction: str, atr: float,
                                max_wait: int) -> int | None:
        """Look for a retracement entry within max_wait bars of the signal.

        Scans forward from signal_idx looking for price to pull back to the
        62-79% retracement zone of the last impulse move. Returns the candle
        index of the retracement entry, or None if no pullback occurs.
        """
        is_long = direction == "long"

        # Find the impulse: look back up to 10 bars for the swing extremes
        lookback = min(10, signal_idx)
        impulse_candles = candles[signal_idx - lookback:signal_idx + 1]
        if not impulse_candles:
            return None

        if is_long:
            swing_low = min(c["low"] for c in impulse_candles)
            swing_high = max(c["high"] for c in impulse_candles)
        else:
            swing_high = max(c["high"] for c in impulse_candles)
            swing_low = min(c["low"] for c in impulse_candles)

        impulse_range = swing_high - swing_low
        if impulse_range < 0.5 * atr:
            return None  # No significant impulse

        # OTE zone: 62-79% retracement
        if is_long:
            ote_high = swing_high - 0.62 * impulse_range
            ote_low = swing_high - 0.79 * impulse_range
        else:
            ote_low = swing_low + 0.62 * impulse_range
            ote_high = swing_low + 0.79 * impulse_range

        # Scan forward for pullback into OTE zone (max 5 bars wait)
        wait_bars = min(5, max_wait - 1)
        end = min(signal_idx + wait_bars + 1, len(candles) - max_wait)
        for i in range(signal_idx + 1, end):
            c = candles[i]
            # Check if candle's range passes through OTE zone
            if c["low"] <= ote_high and c["high"] >= ote_low:
                return i

        return None

    # ── V1: Original confluence scoring (kept for A/B comparison) ──

    def _score_confluence(self, candles: list[dict], idx: int,
                          direction: str, atr: float,
                          obs: list[dict], fvgs_all: list[dict],
                          liqs: list[dict], ms_score: float) -> tuple:
        """Score ICT confluence for a candidate setup. Returns (score, tags)."""
        score = 0
        tags = []
        price = candles[idx]["close"]
        is_long = direction == "long"

        # 1. Order Block within 1 ATR (tightened) + mitigation filter
        ob_type = "bullish" if is_long else "bearish"
        for ob in obs:
            if ob["index"] >= idx:
                continue
            if ob["type"] != ob_type:
                continue
            ob_mid = (ob["high"] + ob["low"]) / 2
            if abs(price - ob_mid) <= 1.0 * atr:
                # Check mitigation: skip if any candle between OB and now
                # traded through the full OB zone (low-to-high)
                mitigated = False
                for k in range(ob["index"] + 1, idx):
                    if candles[k]["low"] <= ob["low"] and candles[k]["high"] >= ob["high"]:
                        mitigated = True
                        break
                if mitigated:
                    continue
                score += 1
                tags.append("ob")
                break

        # 2. Unfilled FVG within 1.5 ATR
        fvg_type = "bullish" if is_long else "bearish"
        for fvg in fvgs_all:
            if fvg["index"] >= idx:
                continue
            if fvg["type"] != fvg_type:
                continue
            fvg_mid = (fvg["high"] + fvg["low"]) / 2
            if abs(price - fvg_mid) <= 1.5 * atr:
                score += 1
                tags.append("fvg")
                break

        # 3. Market structure alignment (double weight — strongest edge)
        if is_long and ms_score > 0.3:
            score += 2
            tags.append("structure")
        elif not is_long and ms_score < -0.3:
            score += 2
            tags.append("structure")

        # 4. Displacement candle in last 3 bars
        for j in range(max(0, idx - 2), idx + 1):
            body = abs(candles[j]["close"] - candles[j]["open"])
            if body > 1.5 * atr:
                displacement_bullish = candles[j]["close"] > candles[j]["open"]
                if (is_long and displacement_bullish) or (not is_long and not displacement_bullish):
                    score += 1
                    tags.append("displacement")
                    break

        # 5. Liquidity sweep proximity
        for lq in liqs:
            if lq["index"] >= idx:
                continue
            dist = abs(price - lq["price"])
            if dist <= 0.5 * atr:
                if is_long and lq["type"] == "sellside":
                    score += 1
                    tags.append("sweep")
                    break
                elif not is_long and lq["type"] == "buyside":
                    score += 1
                    tags.append("sweep")
                    break

        # 6. Killzone timing
        hour = _extract_hour(candles[idx].get("datetime", ""))
        if 7 <= hour < 10:
            score += 1
            tags.append("london")
        elif 13 <= hour < 16:
            score += 1
            tags.append("ny_am")

        return score, tags

    # ── V2: Quality-weighted scoring system ─────────────────

    def _score_quality(self, candles: list[dict], idx: int,
                       direction: str, atr: float,
                       obs: list[dict], fvgs_all: list[dict],
                       liqs: list[dict], ms_score: float,
                       swings: list[dict] = None) -> tuple:
        """Quality-weighted confluence scoring (V2).

        Returns (total_score: float, tags: list[str], breakdown: dict).
        Total possible ~16.5 points.  Threshold: min_quality_score (default 5.0).

        Scoring categories:
            OB Quality:      0 to 4.0
            FVG Quality:     0 to 3.0
            Liquidity:       0 to 3.0
            Structure:       0 to 2.5  (can be -1.0 penalty)
            Session Timing:  0 to 2.0  (can be -0.5 penalty)
            Displacement:    0 to 2.0
        """
        price = candles[idx]["close"]
        is_long = direction == "long"
        safe_atr = atr if atr > 0 else 1.0

        breakdown = {}
        tags = []

        # ── 1. Order Block Quality (0 to 4.0) ──────────────
        ob_score = 0.0
        ob_type = "bullish" if is_long else "bearish"
        best_ob = None

        for ob in obs:
            if ob["index"] >= idx or ob["type"] != ob_type:
                continue
            ob_mid = (ob["high"] + ob["low"]) / 2
            if abs(price - ob_mid) > 2.0 * safe_atr:
                continue

            ob_range = ob["high"] - ob["low"]
            # Mitigation: >50% of OB range closed through → dead
            mitigated = False
            for k in range(ob["index"] + 1, idx):
                if is_long and candles[k]["close"] < ob["low"] + ob_range * 0.5:
                    mitigated = True
                    break
                if not is_long and candles[k]["close"] > ob["high"] - ob_range * 0.5:
                    mitigated = True
                    break
            if mitigated:
                continue

            this_score = 0.0

            # Displacement strength
            disp_idx = ob["index"] + 1
            if disp_idx < len(candles):
                disp_body = abs(candles[disp_idx]["close"] - candles[disp_idx]["open"])
                this_score += min(2.0, disp_body / safe_atr)

            # Freshness
            retests = compute_ob_freshness(candles, ob, idx)
            if retests == 0:
                this_score += 1.0
            elif retests == 1:
                this_score += 0.5

            # Size relative to ATR
            if ob_range > 1.0 * safe_atr:
                this_score += 1.0
            elif ob_range > 0.5 * safe_atr:
                this_score += 0.5

            if best_ob is None or this_score > ob_score:
                ob_score = this_score
                best_ob = ob

        if best_ob is not None:
            ob_score = min(4.0, ob_score)
            tags.append("ob")
        breakdown["ob_score"] = round(ob_score, 2)

        # ── 2. FVG Quality (0 to 3.0) ──────────────────────
        fvg_score = 0.0
        fvg_type = "bullish" if is_long else "bearish"
        best_fvg = None

        for fvg in fvgs_all:
            if fvg["index"] >= idx or fvg["type"] != fvg_type:
                continue
            fvg_mid = (fvg["high"] + fvg["low"]) / 2
            if abs(price - fvg_mid) > 2.0 * safe_atr:
                continue

            this_score = 0.0

            # FVG size
            gap_atr = fvg["size"] / safe_atr
            this_score += min(1.5, gap_atr * 2)

            # Fill percentage
            fill_pct = compute_fvg_fill_percentage(candles, fvg, idx)
            if fill_pct < 0.01:
                this_score += 1.0  # unfilled
            elif fill_pct < 0.5:
                this_score += 0.5  # partially filled

            # FVG + OB overlap bonus
            if best_ob is not None:
                if fvg["low"] <= best_ob["high"] and fvg["high"] >= best_ob["low"]:
                    this_score += 0.5

            if this_score > fvg_score:
                fvg_score = this_score
                best_fvg = fvg

        if best_fvg is not None:
            fvg_score = min(3.0, fvg_score)
            tags.append("fvg")
        breakdown["fvg_score"] = round(fvg_score, 2)

        # ── 3. Liquidity Context (0 to 3.0) ────────────────
        liq_score = 0.0

        if swings:
            # Recent sweep detection
            recent_swings = [s for s in swings
                             if s["index"] < idx and idx - s["index"] <= 10]
            sweep_found = False

            if is_long:
                for sw in [s for s in recent_swings if s["type"] == "low"]:
                    # Did price take out this swing low then reverse?
                    swept = any(
                        candles[k]["low"] < sw["price"]
                        for k in range(sw["index"], min(idx + 1, sw["index"] + 4))
                        if k < len(candles)
                    )
                    if swept:
                        reversed_up = any(
                            candles[k]["close"] > sw["price"]
                            for k in range(sw["index"] + 1, min(idx + 1, sw["index"] + 4))
                            if k < len(candles)
                        )
                        if reversed_up:
                            liq_score += 2.0
                            tags.append("sweep")
                            sweep_found = True
                            break
            else:
                for sw in [s for s in recent_swings if s["type"] == "high"]:
                    swept = any(
                        candles[k]["high"] > sw["price"]
                        for k in range(sw["index"], min(idx + 1, sw["index"] + 4))
                        if k < len(candles)
                    )
                    if swept:
                        reversed_down = any(
                            candles[k]["close"] < sw["price"]
                            for k in range(sw["index"] + 1, min(idx + 1, sw["index"] + 4))
                            if k < len(candles)
                        )
                        if reversed_down:
                            liq_score += 2.0
                            tags.append("sweep")
                            sweep_found = True
                            break

            # Liquidity target ahead
            if is_long:
                targets = [s for s in swings if s["type"] == "high"
                           and s["price"] > price and s["index"] < idx]
            else:
                targets = [s for s in swings if s["type"] == "low"
                           and s["price"] < price and s["index"] < idx]

            if targets:
                nearest_dist = min(abs(t["price"] - price) for t in targets)
                if nearest_dist <= 3.0 * safe_atr:
                    liq_score += 1.0
                    # Equal highs/lows (3+ at similar price)
                    best_price = min(targets, key=lambda t: abs(t["price"] - price))["price"]
                    eq_count = sum(1 for t in targets
                                   if abs(t["price"] - best_price) < 0.1 * safe_atr)
                    if eq_count >= 3:
                        liq_score += 0.5

        liq_score = min(3.0, liq_score)
        breakdown["liq_score"] = round(liq_score, 2)

        # ── 4. Market Structure (0 to 2.5, can be -1.0) ────
        structure_score = 0.0

        if swings:
            recent_highs = sorted(
                [s for s in swings if s["type"] == "high" and s["index"] < idx],
                key=lambda s: s["index"], reverse=True,
            )[:5]
            recent_lows = sorted(
                [s for s in swings if s["type"] == "low" and s["index"] < idx],
                key=lambda s: s["index"], reverse=True,
            )[:5]

            has_bos = False
            is_choch = False

            if is_long and recent_highs:
                most_recent = recent_highs[0]
                for k in range(max(0, idx - 15), idx + 1):
                    if candles[k]["close"] > most_recent["price"]:
                        has_bos = True
                        # ChoCH: was making lower highs before this break?
                        if len(recent_highs) >= 3:
                            lh_count = sum(
                                1 for j in range(len(recent_highs) - 1)
                                if recent_highs[j]["price"] < recent_highs[j + 1]["price"]
                            )
                            if lh_count >= 2:
                                is_choch = True
                        break
            elif not is_long and recent_lows:
                most_recent = recent_lows[0]
                for k in range(max(0, idx - 15), idx + 1):
                    if candles[k]["close"] < most_recent["price"]:
                        has_bos = True
                        if len(recent_lows) >= 3:
                            hl_count = sum(
                                1 for j in range(len(recent_lows) - 1)
                                if recent_lows[j]["price"] > recent_lows[j + 1]["price"]
                            )
                            if hl_count >= 2:
                                is_choch = True
                        break

            if is_choch:
                structure_score = 2.5
                tags.append("structure")
            elif has_bos:
                structure_score = 1.5
                tags.append("structure")
            elif (is_long and ms_score > 0.3) or (not is_long and ms_score < -0.3):
                structure_score = 0.5
                tags.append("structure")
            elif (is_long and ms_score < -0.3) or (not is_long and ms_score > 0.3):
                structure_score = -1.0  # Penalty for opposing structure
        else:
            if (is_long and ms_score > 0.3) or (not is_long and ms_score < -0.3):
                structure_score = 0.5
                tags.append("structure")
            elif (is_long and ms_score < -0.3) or (not is_long and ms_score > 0.3):
                structure_score = -1.0

        breakdown["structure_score"] = round(structure_score, 2)

        # ── 5. Session Timing (0 to 2.0, can be -0.5) ──────
        hour = _extract_hour(candles[idx].get("datetime", ""))
        session_score = 0.0

        if 7 <= hour < 9:       # London open
            session_score = 1.5
            tags.append("london")
        elif 13 <= hour < 15:   # NY AM
            session_score = 2.0
            tags.append("ny_am")
        elif 15 <= hour < 17:   # NY PM
            session_score = 1.0
        elif 0 <= hour < 4:     # Asia
            session_score = 0.0
        else:
            session_score = -0.5  # Off hours penalty

        breakdown["session_score"] = round(session_score, 2)

        # ── 6. Displacement Confirmation (0 to 2.0) ────────
        displacement_score = 0.0

        for j in range(max(0, idx - 4), idx + 1):
            body = abs(candles[j]["close"] - candles[j]["open"])
            body_atr = body / safe_atr
            bullish_disp = candles[j]["close"] > candles[j]["open"]
            aligned = (is_long and bullish_disp) or (not is_long and not bullish_disp)
            if aligned:
                if body_atr > 2.0:
                    displacement_score = 2.0
                    tags.append("displacement")
                    break
                elif body_atr > 1.5:
                    displacement_score = 1.0
                    tags.append("displacement")
                    break

        breakdown["displacement_score"] = round(displacement_score, 2)

        # ── Total ───────────────────────────────────────────
        total = (ob_score + fvg_score + liq_score +
                 structure_score + session_score + displacement_score)
        breakdown["total_quality_score"] = round(total, 2)

        return total, tags, breakdown

    # ── V2: Rejection-based entry detection ─────────────────

    def _find_rejection_entry(self, candles: list[dict], signal_idx: int,
                               direction: str, atr: float,
                               best_ob: dict = None, best_fvg: dict = None,
                               max_forward: int = None) -> dict | None:
        """Scan forward for a rejection candle in the OB/FVG zone.

        A valid rejection must:
        1. Retrace into the OB or FVG zone
        2. Close in the upper/lower 33% of its range (strong rejection wick)
        3. Have a directionally-aligned body (bullish for longs)
        4. Body smaller than the displacement candle
        5. Range >= 0.5 ATR (not indecision)

        Returns dict with entry info or None if no valid rejection.
        """
        is_long = direction == "long"
        safe_atr = atr if atr > 0 else 1.0
        max_wait = max_forward or self.cfg.max_bars_to_rejection

        # Determine the zone
        zone_high = zone_low = None
        if best_ob:
            zone_high, zone_low = best_ob["high"], best_ob["low"]
        if best_fvg:
            fh, fl = best_fvg["high"], best_fvg["low"]
            if zone_high is None:
                zone_high, zone_low = fh, fl
            else:
                zone_high = max(zone_high, fh)
                zone_low = min(zone_low, fl)
        if zone_high is None:
            # Fallback: use signal candle range
            zone_high = candles[signal_idx]["high"]
            zone_low = candles[signal_idx]["low"]

        # Find displacement body for comparison
        disp_body = 0.0
        for j in range(max(0, signal_idx - 4), signal_idx + 1):
            b = abs(candles[j]["close"] - candles[j]["open"])
            if b > disp_body:
                disp_body = b

        end = min(signal_idx + max_wait + 1, len(candles) - 5)
        for i in range(signal_idx + 1, end):
            c = candles[i]
            candle_range = c["high"] - c["low"]
            if candle_range <= 0:
                continue
            body = abs(c["close"] - c["open"])

            if is_long:
                # Must retrace into zone
                if c["low"] > zone_high or c["high"] < zone_low:
                    continue
                # Close in upper 33%
                close_pos = (c["close"] - c["low"]) / candle_range
                if close_pos < 0.67:
                    continue
                # Bullish body
                if c["close"] <= c["open"]:
                    continue
                # Body < displacement
                if disp_body > 0 and body >= disp_body:
                    continue
                # Range >= 0.5 ATR
                if candle_range < 0.5 * safe_atr:
                    continue

                structural_sl = c["low"] - 0.2 * safe_atr
                return {
                    "idx": i,
                    "entry_price": c["close"],
                    "structural_sl": structural_sl,
                    "rejection_quality": round(close_pos, 4),
                    "candles_to_rejection": i - signal_idx,
                    "entry_type": "rejection",
                }
            else:
                # Must retrace into zone
                if c["high"] < zone_low or c["low"] > zone_high:
                    continue
                # Close in lower 33%
                close_pos = (c["high"] - c["close"]) / candle_range
                if close_pos < 0.67:
                    continue
                # Bearish body
                if c["close"] >= c["open"]:
                    continue
                # Body < displacement
                if disp_body > 0 and body >= disp_body:
                    continue
                # Range >= 0.5 ATR
                if candle_range < 0.5 * safe_atr:
                    continue

                structural_sl = c["high"] + 0.2 * safe_atr
                return {
                    "idx": i,
                    "entry_price": c["close"],
                    "structural_sl": structural_sl,
                    "rejection_quality": round(close_pos, 4),
                    "candles_to_rejection": i - signal_idx,
                    "entry_type": "rejection",
                }

        return None

    # ── V2: Narrative-based HTF directional filter ──────────

    def check_narrative_alignment(self, direction: str,
                                   htf_features: dict) -> tuple:
        """Check if the 4H narrative supports the trade direction.

        Returns (passes_filter: bool, reason: str).

        Longs: must be in discount (< 0.5) OR near 4H demand OR SSL swept.
        Shorts: must be in premium (> 0.5) OR near 4H supply OR BSL swept.
        """
        is_long = direction == "long"
        pd_val = htf_features.get("htf_premium_discount", 0.5)
        ssl_swept = htf_features.get("htf_ssl_swept", 0)
        bsl_swept = htf_features.get("htf_bsl_swept", 0)
        liq_narr = htf_features.get("htf_liq_narrative", 0)
        ob_below = htf_features.get("htf_ob_below_dist", 10.0)
        ob_above = htf_features.get("htf_ob_above_dist", 10.0)

        if is_long:
            in_discount = pd_val < 0.5
            ssl_taken = ssl_swept == 1 or liq_narr == 1
            has_demand = ob_below < 1.5

            if in_discount and (ssl_taken or has_demand):
                return True, "Long in discount, SSL swept or 4H demand nearby"
            if has_demand:
                return True, "Long with 4H demand zone nearby"
            if in_discount:
                return True, "Long in discount array"
            return False, (f"Long rejected: pd={pd_val:.2f}, "
                           f"no SSL sweep, no nearby demand")
        else:
            in_premium = pd_val > 0.5
            bsl_taken = bsl_swept == 1 or liq_narr == -1
            has_supply = ob_above < 1.5

            if in_premium and (bsl_taken or has_supply):
                return True, "Short in premium, BSL swept or 4H supply nearby"
            if has_supply:
                return True, "Short with 4H supply zone nearby"
            if in_premium:
                return True, "Short in premium array"
            return False, (f"Short rejected: pd={pd_val:.2f}, "
                           f"no BSL sweep, no nearby supply")


# ═══════════════════════════════════════════════════════════════════════
# Walk-Forward Engine
# ═══════════════════════════════════════════════════════════════════════


class WalkForwardEngine:
    """Rolling-window walk-forward optimization."""

    def __init__(self, wfo_config: WFOConfig = None, use_autogluon: bool = True):
        self.cfg = wfo_config or WFOConfig()
        self.use_autogluon = use_autogluon
        self.detector = ICTSetupDetector(wfo_config=self.cfg)

    def run(self, candles: list[dict], timeframe: str = "1h") -> WFOReport:
        """Execute walk-forward optimization over candle data.

        Returns WFOReport with aggregated out-of-sample metrics.
        """
        min_candles = self.cfg.train_window + self.cfg.test_window
        if len(candles) < min_candles:
            raise ValueError(
                f"Need at least {min_candles} candles, got {len(candles)}"
            )

        # Calculate fold boundaries
        max_possible = (len(candles) - self.cfg.train_window - self.cfg.test_window) // self.cfg.step_size + 1
        n_folds = min(self.cfg.max_folds, max(1, max_possible))

        fold_results = []
        skipped = 0
        temp_dirs = []
        self.oos_trades = []  # Retain OOS trade records for dataset ingestion

        for f in range(n_folds):
            train_start = f * self.cfg.step_size
            train_end = train_start + self.cfg.train_window
            test_start = train_end
            test_end = min(test_start + self.cfg.test_window, len(candles))

            if test_end <= test_start:
                skipped += 1
                continue

            train_candles = candles[train_start:train_end]
            test_candles = candles[test_start:test_end]

            # Detect setups
            train_df = self.detector.detect_setups(train_candles, timeframe)
            test_df = self.detector.detect_setups(test_candles, timeframe)

            if len(train_df) < self.cfg.min_setups_per_fold:
                skipped += 1
                continue
            if len(test_df) < 1:
                skipped += 1
                continue

            # Retain OOS trade records with regime and fold metadata
            regime = detect_regime(candles, test_start)
            for _, row in test_df.iterrows():
                trade = row.to_dict()
                trade["regime"] = regime
                trade["fold"] = f
                self.oos_trades.append(trade)

            # Train and predict
            if self.use_autogluon:
                try:
                    predictor, temp_dir = self._train_fold_ag(train_df, f)
                    temp_dirs.append(temp_dir)
                    predictions = self._predict_ag(predictor, test_df)
                except Exception:
                    predictions = self._predict_heuristic(test_df)
            else:
                predictions = self._predict_heuristic(test_df)

            # Compute fold metrics
            test_date_start = candles[test_start].get("datetime", "") if test_start < len(candles) else ""
            test_date_end = candles[min(test_end, len(candles)) - 1].get("datetime", "") if test_end > 0 else ""
            fold = self._compute_fold_metrics(
                test_df, predictions, f, train_start, train_end,
                test_start, test_end, regime,
                test_date_start=test_date_start,
                test_date_end=test_date_end,
            )
            fold_results.append(fold)

        # Cleanup temp dirs
        for td in temp_dirs:
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass

        # Aggregate
        report = self._aggregate_report(fold_results, skipped)
        return report

    def _train_fold_ag(self, train_df: pd.DataFrame,
                       fold_num: int) -> tuple:
        """Train AutoGluon classifier for one fold in a temp directory."""
        from autogluon.tabular import TabularPredictor

        temp_dir = tempfile.mkdtemp(prefix=f"wfo_fold_{fold_num}_")

        # Prepare training data — drop all non-feature metadata columns
        label = "outcome"
        drop_cols = [c for c in [
            "candle_index", "won", "setup_type", "direction",
            "confluence_score", "max_drawdown_atr", "max_favorable_atr",
            "bars_held",
            # V2 metadata columns (not features)
            "total_quality_score", "ob_score", "fvg_score", "liq_score",
            "structure_score_detail", "session_score", "displacement_score",
            "entry_type", "rejection_quality", "structural_sl",
            "candles_to_rejection", "narrative_reason",
        ] if c in train_df.columns]
        fit_df = train_df.drop(columns=drop_cols)

        predictor = TabularPredictor(
            label=label,
            path=temp_dir,
            problem_type="multiclass",
        ).fit(
            fit_df,
            time_limit=self.cfg.ag_time_limit,
            presets=self.cfg.ag_presets,
            verbosity=0,
        )

        return predictor, temp_dir

    def _predict_ag(self, predictor, test_df: pd.DataFrame) -> pd.DataFrame:
        """Predict with trained AutoGluon model. Returns DataFrame with predictions."""
        drop_cols = [c for c in [
            "outcome", "candle_index", "won", "setup_type", "direction",
            "confluence_score", "max_drawdown_atr", "max_favorable_atr",
            "bars_held",
            # V2 metadata columns
            "total_quality_score", "ob_score", "fvg_score", "liq_score",
            "structure_score_detail", "session_score", "displacement_score",
            "entry_type", "rejection_quality", "structural_sl",
            "candles_to_rejection", "narrative_reason",
        ] if c in test_df.columns]
        feat_df = test_df.drop(columns=drop_cols)

        preds = predictor.predict(feat_df)
        probs = predictor.predict_proba(feat_df)

        result = pd.DataFrame(index=test_df.index)
        result["predicted_outcome"] = preds
        # Win probability = sum of tp* columns
        tp_cols = [c for c in probs.columns if c.startswith("tp")]
        result["win_prob"] = probs[tp_cols].sum(axis=1) if tp_cols else 0.0

        return result

    def _predict_heuristic(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Heuristic model: win_prob = 0.3 + confluence_score * 0.1, capped 0.8."""
        result = pd.DataFrame(index=test_df.index)

        if "confluence_score" in test_df.columns:
            result["win_prob"] = test_df["confluence_score"].apply(
                lambda c: min(0.8, 0.3 + c * 0.1)
            )
        else:
            result["win_prob"] = 0.5

        # Simple prediction based on win probability
        result["predicted_outcome"] = result["win_prob"].apply(
            lambda p: "tp1_hit" if p > 0.5 else "stopped_out"
        )

        return result

    def _compute_fold_metrics(self, test_df: pd.DataFrame,
                              predictions: pd.DataFrame,
                              fold_num: int, train_start: int,
                              train_end: int, test_start: int,
                              test_end: int, regime: str,
                              test_date_start: str = "",
                              test_date_end: str = "") -> FoldResult:
        """Compute metrics for one fold from test results."""
        win_outcomes = {"tp1_hit", "tp2_hit", "tp3_hit"}
        actuals = test_df["outcome"]

        wins = int((actuals.isin(win_outcomes)).sum())
        losses = int((actuals == "stopped_out").sum())
        expired = int((actuals == "expired").sum())
        total = len(test_df)

        win_rate = wins / total if total > 0 else 0.0

        # Average R:R (MFE / MAE for winners)
        winners = test_df[test_df["outcome"].isin(win_outcomes)]
        losers = test_df[test_df["outcome"] == "stopped_out"]

        if len(winners) > 0 and "max_favorable_atr" in winners.columns:
            winning_mfe = winners["max_favorable_atr"].values
            winning_mae = winners["max_drawdown_atr"].values
            avg_rr_vals = [_safe_divide(mfe, mae) for mfe, mae in zip(winning_mfe, winning_mae)]
            avg_rr = sum(avg_rr_vals) / len(avg_rr_vals) if avg_rr_vals else 0
        else:
            avg_rr = 0

        # Profit factor
        avg_win = winners["max_favorable_atr"].mean() if len(winners) > 0 else 0
        avg_loss = losers["max_drawdown_atr"].mean() if len(losers) > 0 else 0
        winning_pnl = avg_win * len(winners) if len(winners) > 0 else 0
        losing_pnl = avg_loss * len(losers) if len(losers) > 0 else 0
        profit_factor = _safe_divide(winning_pnl, losing_pnl)
        profit_factor = min(profit_factor, 99.9)

        # Sharpe estimate
        trade_returns = []
        for _, row in test_df.iterrows():
            if row["outcome"] in win_outcomes:
                trade_returns.append(row.get("max_favorable_atr", 0))
            elif row["outcome"] == "stopped_out":
                trade_returns.append(-row.get("max_drawdown_atr", 0))
            else:
                trade_returns.append(0)

        if len(trade_returns) >= 2:
            mean_ret = sum(trade_returns) / len(trade_returns)
            std_ret = math.sqrt(
                sum((r - mean_ret) ** 2 for r in trade_returns) / len(trade_returns)
            )
            sharpe = _safe_divide(mean_ret, std_ret)
        else:
            sharpe = 0

        # Max drawdown (peak-to-trough of cumulative PnL)
        cum_pnl = 0
        peak = 0
        max_dd = 0
        for ret in trade_returns:
            cum_pnl += ret
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            max_dd = max(max_dd, dd)

        # Setup type counts
        setup_counts = test_df["setup_type"].value_counts().to_dict() if "setup_type" in test_df.columns else {}

        # Winning trade stats for SL/TP calibration
        w_drawdowns = list(winners["max_drawdown_atr"].values) if len(winners) > 0 else []
        w_excursions = list(winners["max_favorable_atr"].values) if len(winners) > 0 else []

        return FoldResult(
            fold_num=fold_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            total_trades=total,
            wins=wins,
            losses=losses,
            expired=expired,
            win_rate=round(win_rate, 4),
            avg_rr=round(avg_rr, 4),
            profit_factor=round(profit_factor, 4),
            sharpe=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            regime=regime,
            test_date_start=test_date_start,
            test_date_end=test_date_end,
            setup_types=setup_counts,
            winning_drawdowns=w_drawdowns,
            winning_excursions=w_excursions,
        )

    def _aggregate_report(self, folds: list, skipped: int) -> WFOReport:
        """Aggregate fold results into a WFO report."""
        now = datetime.now(timezone.utc).isoformat()

        if not folds:
            return WFOReport(
                total_oos_trades=0, oos_win_rate=0, oos_avg_rr=0,
                oos_profit_factor=0, oos_sharpe=0, oos_max_drawdown=0,
                regime_stability=0, recommended_sl_atr=self.cfg.sl_atr_mult,
                recommended_tp_atr=list(self.cfg.tp_atr_mults),
                grade="D", folds=[], fold_count=0,
                skipped_folds=skipped, setup_type_breakdown={},
                timestamp=now,
            )

        # Aggregate OOS totals
        total_trades = sum(f.total_trades for f in folds)
        total_wins = sum(f.wins for f in folds)
        total_losses = sum(f.losses for f in folds)

        oos_win_rate = _safe_divide(total_wins, total_trades)

        # Aggregate R:R
        rr_values = [f.avg_rr for f in folds if f.avg_rr > 0]
        oos_avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0

        # Aggregate profit factor
        all_winning_pnl = sum(
            sum(f.winning_excursions) for f in folds
        )
        all_losing_dd = sum(
            f.max_drawdown * f.losses for f in folds
        )
        oos_profit_factor = min(_safe_divide(all_winning_pnl, all_losing_dd), 99.9)

        # Aggregate Sharpe
        sharpe_vals = [f.sharpe for f in folds if f.total_trades > 0]
        oos_sharpe = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0

        # Max drawdown across folds
        oos_max_dd = max(f.max_drawdown for f in folds) if folds else 0

        # Regime stability
        fold_wrs = [f.win_rate for f in folds]
        if len(fold_wrs) >= 2:
            wr_std = float(np.std(fold_wrs))
        else:
            wr_std = 0
        regime_stability = round(1.0 - min(wr_std, 0.5) * 2, 4)

        # Calibrated SL/TP from winning OOS trades
        all_w_drawdowns = []
        all_w_excursions = []
        for f in folds:
            all_w_drawdowns.extend(f.winning_drawdowns)
            all_w_excursions.extend(f.winning_excursions)

        if len(all_w_drawdowns) >= 5:
            recommended_sl = round(float(np.percentile(all_w_drawdowns, 95)) * 1.1, 4)
        else:
            recommended_sl = self.cfg.sl_atr_mult

        if len(all_w_excursions) >= 5:
            recommended_tp = [
                round(float(np.percentile(all_w_excursions, 40)), 4),
                round(float(np.percentile(all_w_excursions, 65)), 4),
                round(float(np.percentile(all_w_excursions, 85)), 4),
            ]
        else:
            recommended_tp = list(self.cfg.tp_atr_mults)

        # Grade
        grade = _compute_wfo_grade(oos_win_rate, oos_profit_factor)

        # Setup type breakdown
        type_counts = {}
        for f in folds:
            for st, count in f.setup_types.items():
                type_counts[st] = type_counts.get(st, 0) + count

        # Per-setup-type win rate stats from OOS trades
        setup_type_stats = {}
        if hasattr(self, "oos_trades") and self.oos_trades:
            from collections import defaultdict
            st_counts = defaultdict(lambda: {"wins": 0, "total": 0})
            for t in self.oos_trades:
                st = t.get("setup_type", "unknown")
                st_counts[st]["total"] += 1
                if t.get("won"):
                    st_counts[st]["wins"] += 1
            for st, d in st_counts.items():
                d["win_rate"] = round(d["wins"] / d["total"], 4) if d["total"] > 0 else 0
            setup_type_stats = dict(st_counts)

        return WFOReport(
            total_oos_trades=total_trades,
            oos_win_rate=round(oos_win_rate, 4),
            oos_avg_rr=round(oos_avg_rr, 4),
            oos_profit_factor=round(oos_profit_factor, 4),
            oos_sharpe=round(oos_sharpe, 4),
            oos_max_drawdown=round(oos_max_dd, 4),
            regime_stability=regime_stability,
            recommended_sl_atr=recommended_sl,
            recommended_tp_atr=recommended_tp,
            grade=grade,
            folds=folds,
            fold_count=len(folds),
            skipped_folds=skipped,
            setup_type_breakdown=type_counts,
            timestamp=now,
            setup_type_stats=setup_type_stats,
        )


def _compute_wfo_grade(win_rate: float, profit_factor: float) -> str:
    """Assign system grade based on OOS metrics."""
    if profit_factor > 1.5 and win_rate > 0.50:
        return "A"
    if profit_factor > 1.2 and win_rate > 0.40:
        return "B"
    if profit_factor > 1.0:
        return "C"
    return "D"


def build_setup_filter(report: WFOReport, min_win_rate: float = 0.40,
                       min_trades: int = 3) -> dict:
    """Classify setup types as profitable/unprofitable/insufficient from WFO stats.

    Args:
        report: WFOReport with setup_type_stats populated.
        min_win_rate: Minimum OOS win rate to consider a setup type profitable.
        min_trades: Minimum trade count; below this → insufficient data.

    Returns dict with keys: profitable, unprofitable, insufficient, stats,
    min_win_rate, min_trades.
    """
    profitable = []
    unprofitable = []
    insufficient = []
    for st, data in report.setup_type_stats.items():
        if data["total"] < min_trades:
            insufficient.append(st)
        elif data["win_rate"] >= min_win_rate:
            profitable.append(st)
        else:
            unprofitable.append(st)
    return {
        "profitable": profitable,
        "unprofitable": unprofitable,
        "insufficient": insufficient,
        "stats": report.setup_type_stats,
        "min_win_rate": min_win_rate,
        "min_trades": min_trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════


def save_report(report: WFOReport, path: str = None):
    """Save WFO report to JSON file."""
    cfg = get_config()
    path = path or cfg.get("wfo_report_path", "ml/models/wfo_report.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_report(path: str = None) -> WFOReport | None:
    """Load saved WFO report, or None if not found."""
    cfg = get_config()
    path = path or cfg.get("wfo_report_path", "ml/models/wfo_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return WFOReport.from_dict(json.load(f))


def update_bayesian_from_wfo(report: WFOReport, db) -> None:
    """Update Bayesian priors with WFO-calibrated win/loss counts.

    Seeds the Beta distribution with OOS results so the Bayesian updater
    starts from data-backed priors instead of uninformative Beta(1,1).
    """
    from ml.bayesian import get_default_prior

    if report.total_oos_trades < 20:
        return

    prior = db.get_bayesian_state() or get_default_prior()

    oos_wins = int(report.oos_win_rate * report.total_oos_trades)
    oos_losses = report.total_oos_trades - oos_wins

    # Set alpha/beta to reflect OOS results (at minimum)
    prior["alpha"] = max(prior["alpha"], 1 + oos_wins)
    prior["beta_param"] = max(prior["beta_param"], 1 + oos_losses)

    # Cap kappa to prevent prior from becoming too rigid
    from ml.dataset import DriftAlarm
    alarm = DriftAlarm(config=get_config())
    prior = alarm.cap_kappa(prior)

    db.save_bayesian_state(prior)


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def _fetch_candles_twelve_data(api_key: str, count: int,
                               interval: str = "1h") -> list[dict]:
    """Fetch XAU/USD candles from Twelve Data API."""
    import requests

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": count,
        "apikey": api_key,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")

    candles = []
    for v in reversed(data["values"]):
        candles.append({
            "datetime": v["datetime"],
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
        })

    return candles


def _print_report(report: WFOReport):
    """Print formatted WFO report to terminal."""
    print("\n" + "=" * 60)
    print("  WALK-FORWARD OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"  Grade:            {report.grade}")
    print(f"  OOS Trades:       {report.total_oos_trades}")
    print(f"  OOS Win Rate:     {report.oos_win_rate:.1%}")
    print(f"  OOS Profit Factor:{report.oos_profit_factor:.2f}")
    print(f"  OOS Sharpe:       {report.oos_sharpe:.2f}")
    print(f"  OOS Max Drawdown: {report.oos_max_drawdown:.2f} ATR")
    print(f"  Regime Stability: {report.regime_stability:.2f}")
    print(f"  Folds Completed:  {report.fold_count}")
    print(f"  Folds Skipped:    {report.skipped_folds}")
    print()
    print("  CALIBRATED LEVELS:")
    print(f"  Recommended SL:   {report.recommended_sl_atr:.2f} ATR")
    for i, tp in enumerate(report.recommended_tp_atr):
        label = ["Conservative", "Moderate", "Aggressive"][i] if i < 3 else f"TP{i+1}"
        print(f"  TP{i+1} ({label:12s}): {tp:.2f} ATR")
    print()
    print("  FOLD BREAKDOWN:")
    print(f"  {'Fold':>4} {'Trades':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Regime':<16}")
    print("  " + "-" * 50)
    for f in report.folds:
        fr = f if isinstance(f, FoldResult) else FoldResult(**f)
        print(f"  {fr.fold_num:>4} {fr.total_trades:>7} {fr.win_rate:>5.1%} "
              f"{fr.profit_factor:>6.2f} {fr.sharpe:>7.2f} {fr.regime:<16}")
    print()
    if report.setup_type_breakdown:
        print("  SETUP TYPES:")
        for st, count in sorted(report.setup_type_breakdown.items(), key=lambda x: -x[1])[:10]:
            print(f"    {st}: {count}")
    print("=" * 60)


def main():
    """CLI for running Walk-Forward Optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization for ICT trade setups (XAU/USD)"
    )
    parser.add_argument("--td-key", help="Twelve Data API key")
    parser.add_argument("--candles", type=int, default=2000,
                        help="Number of candles to fetch (default: 2000)")
    parser.add_argument("--candles-file", help="JSON file with candle data (alternative to --td-key)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--interval", default="1h", help="Twelve Data interval (default: 1h)")
    parser.add_argument("--train-window", type=int, default=500)
    parser.add_argument("--test-window", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--time-limit", type=int, default=120,
                        help="AutoGluon time limit per fold in seconds")
    parser.add_argument("--no-autogluon", action="store_true",
                        help="Use heuristic model instead of AutoGluon")
    parser.add_argument("--output", help="Output JSON path")

    args = parser.parse_args()

    # Load candle data
    if args.candles_file:
        print(f"Loading candles from {args.candles_file}...")
        with open(args.candles_file) as f:
            candles = json.load(f)
        print(f"Loaded {len(candles)} candles")
    elif args.td_key:
        print(f"Fetching {args.candles} XAU/USD candles from Twelve Data...")
        candles = _fetch_candles_twelve_data(args.td_key, args.candles, args.interval)
        print(f"Fetched {len(candles)} candles")
    else:
        parser.error("Provide either --td-key or --candles-file")

    # Configure WFO
    cfg = WFOConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
        ag_time_limit=args.time_limit,
    )

    # Run
    engine = WalkForwardEngine(wfo_config=cfg, use_autogluon=not args.no_autogluon)
    print(f"\nRunning WFO ({'AutoGluon' if not args.no_autogluon else 'heuristic'} mode)...")
    print(f"  Train window: {cfg.train_window}, Test window: {cfg.test_window}, "
          f"Step: {cfg.step_size}")
    report = engine.run(candles, args.timeframe)

    # Save report
    out_path = args.output or get_config().get("wfo_report_path", "ml/models/wfo_report.json")
    save_report(report, out_path)
    print(f"\nReport saved to {out_path}")

    # Print summary
    _print_report(report)


if __name__ == "__main__":
    main()
