"""Pure helpers to snap Claude's analysis JSON onto actual candle wicks.

Mirror of src/analysisSnap.js. Both modules implement the same algorithm:
- OBs: snap (high, low) to candles[candleIndex].(high, low) when off by > tolerance
- FVGs: snap (high, low) to gap range across (startIndex, startIndex+2)
- Liquidity: snap price to candle wick (high for buyside, low for sellside)

Items with missing / negative / out-of-bounds indices are dropped.
Degenerate FVGs (expected_low >= expected_high) are also dropped.
"""
from typing import Any

DEFAULT_TOLERANCE = 0.50


def _make_diagnostics() -> dict:
    return {
        "snapped_obs": 0, "dropped_obs": 0,
        "snapped_fvgs": 0, "dropped_fvgs": 0,
        "snapped_liquidity": 0, "dropped_liquidity": 0,
        "unresolved_anchor": 0,
        "wrong_color_obs": 0,
        "deltas": [],
    }


def _resolve_anchor_index(item: dict, candles: list[dict], legacy_key: str) -> int:
    """Resolve item's anchor to a numeric candle index.

    Priority: anchor_dt (datetime string match) > legacy numeric key > -1 (drop).
    """
    anchor_dt = item.get("anchor_dt")
    if anchor_dt:
        for i, c in enumerate(candles):
            if c.get("datetime") == anchor_dt:
                return i
        return -1
    legacy = item.get(legacy_key)
    if isinstance(legacy, int) and not isinstance(legacy, bool) and 0 <= legacy < len(candles):
        return legacy
    return -1


def _snap_obs(obs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    for ob in obs:
        ci = _resolve_anchor_index(ob, candles, "candleIndex")
        if ci < 0:
            diag["dropped_obs"] += 1
            if ob.get("anchor_dt"):
                diag["unresolved_anchor"] += 1
            continue
        c = candles[ci]
        c_high = float(c["high"])
        c_low = float(c["low"])

        c_open = float(c["open"])
        c_close = float(c["close"])

        # ICT color validation: bullish OB must anchor on a down-closed (red) candle,
        # bearish OB must anchor on an up-closed (green) candle. Dojis (close === open)
        # are ambiguous — accept them rather than drop.
        candle_is_bullish = c_close > c_open
        candle_is_bearish = c_close < c_open
        ob_type = ob.get("type")
        if ob_type == "bullish" and candle_is_bullish:
            diag["dropped_obs"] += 1
            diag["wrong_color_obs"] += 1
            continue
        if ob_type == "bearish" and candle_is_bearish:
            diag["dropped_obs"] += 1
            diag["wrong_color_obs"] += 1
            continue

        high_off = abs(float(ob.get("high", 0)) - c_high)
        low_off = abs(float(ob.get("low", 0)) - c_low)
        if high_off > tolerance or low_off > tolerance:
            diag["snapped_obs"] += 1
            diag["deltas"].append({
                "kind": "ob", "candleIndex": ci, "anchor_dt": ob.get("anchor_dt"),
                "claimed": {"high": ob.get("high"), "low": ob.get("low")},
                "snapped": {"high": c_high, "low": c_low},
            })
            out.append({**ob, "high": c_high, "low": c_low, "candleIndex": ci, "snapped": True})
        else:
            out.append({**ob, "candleIndex": ci})
    return out


def _snap_fvgs(fvgs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for fvg in fvgs:
        si = _resolve_anchor_index(fvg, candles, "startIndex")
        if si < 0 or si + 2 >= n:
            diag["dropped_fvgs"] += 1
            if si < 0 and fvg.get("anchor_dt"):
                diag["unresolved_anchor"] += 1
            continue
        c0 = candles[si]
        c2 = candles[si + 2]
        if fvg.get("type") == "bullish":
            expected_low = float(c0["high"])
            expected_high = float(c2["low"])
        else:
            expected_high = float(c0["low"])
            expected_low = float(c2["high"])
        if expected_low >= expected_high:
            diag["dropped_fvgs"] += 1
            continue
        high_off = abs(float(fvg.get("high", 0)) - expected_high)
        low_off = abs(float(fvg.get("low", 0)) - expected_low)
        if high_off > tolerance or low_off > tolerance:
            diag["snapped_fvgs"] += 1
            diag["deltas"].append({
                "kind": "fvg", "startIndex": si, "anchor_dt": fvg.get("anchor_dt"),
                "claimed": {"high": fvg.get("high"), "low": fvg.get("low")},
                "snapped": {"high": expected_high, "low": expected_low},
            })
            out.append({**fvg, "high": expected_high, "low": expected_low, "startIndex": si, "snapped": True})
        else:
            out.append({**fvg, "startIndex": si})
    return out


def _snap_liquidity(liqs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    for liq in liqs:
        ci = _resolve_anchor_index(liq, candles, "candleIndex")
        if ci < 0:
            diag["dropped_liquidity"] += 1
            if liq.get("anchor_dt"):
                diag["unresolved_anchor"] += 1
            continue
        c = candles[ci]
        expected = float(c["high"]) if liq.get("type") == "buyside" else float(c["low"])
        off = abs(float(liq.get("price", 0)) - expected)
        if off > tolerance:
            diag["snapped_liquidity"] += 1
            diag["deltas"].append({
                "kind": "liquidity", "candleIndex": ci, "anchor_dt": liq.get("anchor_dt"),
                "claimed": {"price": liq.get("price")},
                "snapped": {"price": expected},
            })
            out.append({**liq, "price": expected, "candleIndex": ci, "snapped": True})
        else:
            out.append({**liq, "candleIndex": ci})
    return out


def snap_analysis_to_candles(
    analysis: dict[str, Any],
    candles: list[dict],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[dict, dict]:
    """Return (snapped_analysis, diagnostics).

    Pure: no I/O, no logging. Caller decides whether/how to log.
    """
    diag = _make_diagnostics()
    obs = analysis.get("orderBlocks") or []
    fvgs = analysis.get("fvgs") or []
    liqs = analysis.get("liquidity") or []
    snapped = {
        **analysis,
        "orderBlocks": _snap_obs(obs, candles, tolerance, diag),
        "fvgs": _snap_fvgs(fvgs, candles, tolerance, diag),
        "liquidity": _snap_liquidity(liqs, candles, tolerance, diag),
    }
    return snapped, diag
