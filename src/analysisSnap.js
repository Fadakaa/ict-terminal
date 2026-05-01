const DEFAULT_TOLERANCE = 0.50;

function makeDiagnostics() {
  return {
    snapped_obs: 0, dropped_obs: 0,
    snapped_fvgs: 0, dropped_fvgs: 0,
    snapped_liquidity: 0, dropped_liquidity: 0,
    deltas: [],
  };
}

function snapOrderBlocks(obs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const ob of obs) {
    const ci = ob.candleIndex;
    if (ci === undefined || ci === null || ci < 0 || ci >= n) {
      diag.dropped_obs += 1;
      continue;
    }
    const c = candles[ci];
    const highOff = Math.abs((ob.high ?? 0) - c.high);
    const lowOff = Math.abs((ob.low ?? 0) - c.low);
    if (highOff > tolerance || lowOff > tolerance) {
      diag.snapped_obs += 1;
      diag.deltas.push({
        kind: "ob", candleIndex: ci,
        claimed: { high: ob.high, low: ob.low },
        snapped: { high: c.high, low: c.low },
      });
      out.push({ ...ob, high: c.high, low: c.low, snapped: true });
    } else {
      out.push(ob);
    }
  }
  return out;
}

export function snapAnalysisToCandles(analysis, candles, options = {}) {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const diag = makeDiagnostics();
  const obs = analysis.orderBlocks ?? [];
  return {
    analysis: { ...analysis, orderBlocks: snapOrderBlocks(obs, candles, tolerance, diag) },
    diagnostics: diag,
  };
}
