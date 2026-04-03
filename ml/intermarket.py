"""Intermarket correlation analysis: DXY + US10Y alongside XAU/USD.

Gold is inversely correlated with the US Dollar Index (DXY) and typically
moves opposite to real yields.  This module computes rolling correlations,
divergence signals, and a session-aware narrative for Claude's prompt.
"""
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Session-aware correlation strength (empirical ranges)
# London/NY overlap has the tightest gold-DXY inverse correlation.
SESSION_WEIGHT = {
    "Asian":  {"strength": "weak",     "note": "Gold-DXY correlation is weaker during Asian hours — divergence signals are less reliable."},
    "London": {"strength": "moderate", "note": "Gold-DXY inverse correlation is moderate during London session."},
    "NY_AM":  {"strength": "strong",   "note": "Gold-DXY inverse correlation is strongest during NY AM (London/NY overlap 12-16 UTC)."},
    "NY_PM":  {"strength": "moderate", "note": "Gold-DXY correlation is moderate during NY PM."},
    "Off":    {"strength": "weak",     "note": "Intermarket correlations are unreliable during off-hours."},
}


def _pct_change_series(candles: list[dict], field: str = "close") -> list[float]:
    """Compute percentage returns from candle close prices."""
    prices = [float(c[field]) for c in candles]
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] != 0 else 0
            for i in range(1, len(prices))]


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two return series."""
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _range_position(candles: list[dict], lookback: int = 20) -> float:
    """Where is current price within the recent N-bar range? 0=low, 1=high."""
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    if not recent:
        return 0.5
    highs = [float(c["high"]) for c in recent]
    lows = [float(c["low"]) for c in recent]
    hi = max(highs)
    lo = min(lows)
    if hi == lo:
        return 0.5
    current = float(recent[-1]["close"])
    return round((current - lo) / (hi - lo), 4)


def _parse_dt(dt_str: str):
    """Parse datetime string to datetime object. Handles common OANDA formats."""
    from datetime import datetime as _dt
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.strptime(dt_str[:19], fmt)
        except (ValueError, TypeError):
            continue
    return None


def align_candles(gold: list[dict], other: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align two candle lists by nearest timestamp within tolerance.

    Different instruments have different trading hours (e.g. USB10Y_USD
    only trades during US hours while XAU_USD trades nearly 24h). Exact
    datetime matching drops 60-70% of candles.

    Strategy: for each gold candle, find the nearest other candle within
    a 30-minute tolerance window. If no match within tolerance, carry
    forward the last known other candle (markets that are closed hold
    their last price).

    Returns (gold_aligned, other_aligned) same length.
    """
    from datetime import timedelta

    if not other:
        return [], []

    # Build sorted list of (datetime_obj, candle) for other instrument
    other_parsed = []
    for c in other:
        dt = _parse_dt(c["datetime"])
        if dt:
            other_parsed.append((dt, c))
    other_parsed.sort(key=lambda x: x[0])

    if not other_parsed:
        return [], []

    g_out, o_out = [], []
    tolerance = timedelta(minutes=30)
    last_matched = other_parsed[0][1]  # carry-forward fallback
    other_idx = 0

    for gc in gold:
        g_dt = _parse_dt(gc["datetime"])
        if not g_dt:
            continue

        # Advance other_idx to closest timestamp
        best_candle = None
        best_delta = None
        while other_idx < len(other_parsed) - 1:
            delta = abs(other_parsed[other_idx][0] - g_dt)
            next_delta = abs(other_parsed[other_idx + 1][0] - g_dt)
            if next_delta < delta:
                other_idx += 1
            else:
                break

        delta = abs(other_parsed[other_idx][0] - g_dt)
        if delta <= tolerance:
            best_candle = other_parsed[other_idx][1]
            last_matched = best_candle
        else:
            # Carry forward last known value (instrument was closed)
            best_candle = last_matched

        g_out.append(gc)
        o_out.append(best_candle)

    total = len(gold)
    exact_matches = sum(
        1 for gc, oc in zip(g_out, o_out)
        if gc["datetime"] == oc["datetime"]
    )
    if total > 0 and exact_matches < total * 0.5:
        logger.info(
            "Intermarket alignment: %d/%d exact matches, %d carry-forward "
            "(different trading hours — normal for bonds/FX vs gold).",
            exact_matches, total, total - exact_matches,
        )
    return g_out, o_out


def compute_intermarket_context(
    gold_candles: list[dict],
    dxy_candles: list[dict] | None,
    us10y_candles: list[dict] | None,
    session: str = "Off",
    lookback: int = 20,
) -> dict:
    """Compute intermarket context for Claude's prompt.

    Args:
        gold_candles:  XAU/USD OHLC candles (chronological)
        dxy_candles:   DXY (US Dollar Index) candles, or None
        us10y_candles: US 10-year yield candles, or None
        session:       Current killzone (Asian/London/NY_AM/NY_PM/Off)
        lookback:      Rolling window for correlation + %change

    Returns:
        Dict with metrics + narrative string for prompt injection.
    """
    result = {
        "gold_pct_20": 0.0,
        "dxy_pct_20": None,
        "us10y_pct_20": None,
        "gold_dxy_corr_20": 0.0,
        "gold_dxy_diverging": 0,
        "dxy_range_position": 0.5,
        "yield_direction": 0,
        "narrative": "",
        "session_strength": SESSION_WEIGHT.get(session, SESSION_WEIGHT["Off"])["strength"],
    }

    if not gold_candles or len(gold_candles) < 5:
        result["narrative"] = "Insufficient gold candle data for intermarket analysis."
        return result

    # Gold returns
    gold_returns = _pct_change_series(gold_candles[-lookback - 1:])
    gold_pct = sum(gold_returns[-lookback:]) * 100 if gold_returns else 0
    result["gold_pct_20"] = round(gold_pct, 3)

    narratives = []

    # ── DXY analysis ─────────────────────────────────────────────
    if dxy_candles and len(dxy_candles) >= 5:
        gold_aligned, dxy_aligned = align_candles(gold_candles, dxy_candles)

        if len(gold_aligned) >= 5:
            gold_ret = _pct_change_series(gold_aligned[-lookback - 1:])
            dxy_ret = _pct_change_series(dxy_aligned[-lookback - 1:])

            # 20-bar correlation
            corr = _pearson_corr(gold_ret, dxy_ret)
            result["gold_dxy_corr_20"] = round(corr, 4)

            # DXY pct change
            dxy_pct = sum(dxy_ret[-lookback:]) * 100 if dxy_ret else 0
            result["dxy_pct_20"] = round(dxy_pct, 3)

            # DXY range position
            result["dxy_range_position"] = _range_position(dxy_aligned, lookback)

            # Divergence: gold and DXY moving in same direction is unusual
            # (normal = inverse correlation, same direction = divergence)
            same_dir = (gold_pct > 0 and dxy_pct > 0) or (gold_pct < 0 and dxy_pct < 0)
            result["gold_dxy_diverging"] = 1 if same_dir and abs(gold_pct) > 0.05 and abs(dxy_pct) > 0.05 else 0

            # Build DXY narrative
            corr_desc = "strongly inversely" if corr < -0.5 else "inversely" if corr < -0.2 else "weakly" if abs(corr) < 0.2 else "positively"
            narratives.append(
                f"DXY: {dxy_pct:+.2f}% over {lookback} bars (range position: {result['dxy_range_position']:.0%}). "
                f"Gold-DXY correlation: {corr:.2f} ({corr_desc} correlated)."
            )

            if result["gold_dxy_diverging"]:
                sw = SESSION_WEIGHT.get(session, SESSION_WEIGHT["Off"])
                if sw["strength"] in ("strong", "moderate"):
                    narratives.append(
                        f"WARNING: Gold ({gold_pct:+.2f}%) and DXY ({dxy_pct:+.2f}%) moving SAME direction — "
                        f"divergence during {session} session is a {sw['strength']} warning signal. "
                        "Validate the setup carefully."
                    )
                else:
                    narratives.append(
                        f"Note: Gold and DXY moving same direction, but during {session} session "
                        "this divergence is less significant."
                    )
        else:
            narratives.append("DXY: insufficient aligned candles for correlation.")
    else:
        narratives.append("DXY data unavailable.")

    # ── US10Y analysis ───────────────────────────────────────────
    if us10y_candles and len(us10y_candles) >= 5:
        gold_aligned_y, yield_aligned = align_candles(gold_candles, us10y_candles)

        if len(yield_aligned) >= 5:
            yield_ret = _pct_change_series(yield_aligned[-lookback - 1:])
            yield_pct = sum(yield_ret[-lookback:]) * 100 if yield_ret else 0
            result["us10y_pct_20"] = round(yield_pct, 3)

            # Yield direction: 1 = rising yields, -1 = falling
            result["yield_direction"] = 1 if yield_pct > 0.05 else (-1 if yield_pct < -0.05 else 0)

            yield_dir = "rising" if result["yield_direction"] == 1 else "falling" if result["yield_direction"] == -1 else "flat"
            narratives.append(
                f"US10Y: {yield_pct:+.2f}% ({yield_dir} yields)."
            )

            # Gold dropping while yields falling = likely liquidity grab
            if gold_pct < -0.1 and yield_pct < -0.05:
                narratives.append(
                    "Gold falling with yields — suggests a liquidity grab, not a fundamental sell-off. "
                    "Bullish setups may still be valid."
                )
            # Gold rising while yields rising = unusual, check for momentum
            elif gold_pct > 0.1 and yield_pct > 0.1:
                narratives.append(
                    "Gold rising despite rising yields — strong demand signal. "
                    "Momentum may override the usual inverse relationship."
                )
        else:
            narratives.append("US10Y: insufficient aligned candles.")
    else:
        narratives.append("US10Y data unavailable.")

    # Session context
    sw = SESSION_WEIGHT.get(session, SESSION_WEIGHT["Off"])
    narratives.append(sw["note"])

    result["narrative"] = " ".join(narratives)
    return result
