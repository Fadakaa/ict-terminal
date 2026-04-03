"""Execution cost simulator for WFO trade setups.

Applies realistic spread, slippage, and entry delay to WFO-detected trades
so that backtested edge isn't fictional. Session-specific XAU/USD spreads.
"""
import random

from ml.config import get_config

# Typical XAU/USD spreads by session (in pips / price points)
SESSION_SPREADS = {
    "asian": 3.5,
    "london": 2.0,
    "new_york": 2.5,
    "overlap_london_ny": 1.8,
    "off_hours": 4.0,
}


def _hour_to_session(hour: int) -> str:
    """Map UTC hour to trading session."""
    if 12 <= hour < 16:
        return "overlap_london_ny"
    elif 7 <= hour < 12:
        return "london"
    elif 16 <= hour < 21:
        return "new_york"
    elif 0 <= hour < 7:
        return "asian"
    return "off_hours"


def _extract_hour(dt_str: str) -> int:
    """Extract hour from datetime string."""
    try:
        time_part = dt_str.split("T")[-1] if "T" in dt_str else dt_str.split(" ")[-1]
        return int(time_part.split(":")[0])
    except (ValueError, IndexError):
        return 0


class ExecutionSimulator:
    """Apply execution costs to WFO trades before dataset ingestion."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self.slippage_pips = self.cfg.get("default_slippage_pips", 0.5)

    def simulate(self, trades: list[dict], candles: list[dict]) -> list[dict]:
        """Apply spread + slippage + entry delay to each trade.

        Args:
            trades: list of trade dicts from WFO detector (with candle_index, direction, etc.)
            candles: full candle list used during WFO

        Returns:
            Filtered list of trades that survive cost adjustment (outcome may change).
        """
        result = []
        for trade in trades:
            adjusted = self._apply_costs(trade, candles)
            if adjusted is not None:
                result.append(adjusted)
        return result

    def _apply_costs(self, trade: dict, candles: list[dict]) -> dict | None:
        """Apply execution costs to a single trade. Returns None if trade becomes invalid."""
        trade = dict(trade)  # shallow copy
        idx = trade.get("candle_index", 0)

        # Determine session and spread
        if idx < len(candles):
            hour = _extract_hour(candles[idx].get("datetime", ""))
        else:
            hour = 0
        session = _hour_to_session(hour)
        spread = SESSION_SPREADS.get(session, 3.0)

        # Entry delay: use next candle's open instead of current close
        if idx + 1 < len(candles):
            delayed_entry = candles[idx + 1]["open"]
        else:
            delayed_entry = candles[idx]["close"] if idx < len(candles) else 0

        # Apply slippage (random within bounds, deterministic via trade index for reproducibility)
        rng = random.Random(idx)
        slippage = rng.uniform(0, self.slippage_pips)

        direction = trade.get("direction", "long")
        is_long = direction == "long"

        # Total cost in price points
        total_cost = spread + slippage
        if is_long:
            adjusted_entry = delayed_entry + total_cost
        else:
            adjusted_entry = delayed_entry - total_cost

        # Recalculate distances — adjust MFE/MAE by cost
        original_mfe = trade.get("max_favorable_atr", 0)
        original_mae = trade.get("max_drawdown_atr", 0)

        # Cost as fraction of ATR (rough approximation)
        # ATR is typically embedded in the features; use a proxy
        atr_proxy = abs(trade.get("price_action_range_atr", 1.0))
        if atr_proxy <= 0:
            atr_proxy = 1.0

        cost_atr = total_cost / (atr_proxy * 100)  # normalize
        # Cap cost impact to something reasonable
        cost_atr = min(cost_atr, 0.5)

        adjusted_mfe = max(0, original_mfe - cost_atr)
        adjusted_mae = original_mae + cost_atr

        trade["max_favorable_atr"] = adjusted_mfe
        trade["max_drawdown_atr"] = adjusted_mae
        trade["execution_cost_atr"] = cost_atr
        trade["session_spread"] = spread
        trade["adjusted_entry"] = adjusted_entry

        # Relabel if SL would now be hit (MFE too small for any TP)
        outcome = trade.get("outcome", "stopped_out")
        if outcome in ("tp1_hit", "tp2_hit", "tp3_hit"):
            if adjusted_mfe < cost_atr * 2:
                trade["outcome"] = "stopped_out"
                trade["won"] = False

        return trade
