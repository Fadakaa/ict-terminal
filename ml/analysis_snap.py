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
        "deltas": [],
    }


def _snap_obs(obs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for ob in obs:
        ci = ob.get("candleIndex")
        if ci is None or ci < 0 or ci >= n:
            diag["dropped_obs"] += 1
            continue
        c = candles[ci]
        c_high = float(c["high"])
        c_low = float(c["low"])
        high_off = abs(float(ob.get("high", 0)) - c_high)
        low_off = abs(float(ob.get("low", 0)) - c_low)
        if high_off > tolerance or low_off > tolerance:
            diag["snapped_obs"] += 1
            diag["deltas"].append({
                "kind": "ob", "candleIndex": ci,
                "claimed": {"high": ob.get("high"), "low": ob.get("low")},
                "snapped": {"high": c_high, "low": c_low},
            })
            out.append({**ob, "high": c_high, "low": c_low, "snapped": True})
        else:
            out.append(ob)
    return out


def _snap_fvgs(fvgs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for fvg in fvgs:
        si = fvg.get("startIndex")
        if si is None or si < 0 or si + 2 >= n:
            diag["dropped_fvgs"] += 1
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
                "kind": "fvg", "startIndex": si,
                "claimed": {"high": fvg.get("high"), "low": fvg.get("low")},
                "snapped": {"high": expected_high, "low": expected_low},
            })
            out.append({**fvg, "high": expected_high, "low": expected_low, "snapped": True})
        else:
            out.append(fvg)
    return out


def _snap_liquidity(liqs: list[dict], candles: list[dict], tolerance: float, diag: dict) -> list[dict]:
    out = []
    n = len(candles)
    for liq in liqs:
        ci = liq.get("candleIndex")
        if ci is None or ci < 0 or ci >= n:
            diag["dropped_liquidity"] += 1
            continue
        c = candles[ci]
        expected = float(c["high"]) if liq.get("type") == "buyside" else float(c["low"])
        off = abs(float(liq.get("price", 0)) - expected)
        if off > tolerance:
            diag["snapped_liquidity"] += 1
            diag["deltas"].append({
                "kind": "liquidity", "candleIndex": ci,
                "claimed": {"price": liq.get("price")},
                "snapped": {"price": expected},
            })
            out.append({**liq, "price": expected, "snapped": True})
        else:
            out.append(liq)
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
