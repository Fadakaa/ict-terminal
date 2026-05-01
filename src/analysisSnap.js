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

function snapFvgs(fvgs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const fvg of fvgs) {
    const si = fvg.startIndex;
    if (si === undefined || si === null || si < 0 || si + 2 >= n) {
      diag.dropped_fvgs += 1;
      continue;
    }
    const c0 = candles[si];
    const c2 = candles[si + 2];
    let expectedHigh, expectedLow;
    if (fvg.type === "bullish") {
      expectedLow = c0.high;
      expectedHigh = c2.low;
    } else {
      expectedHigh = c0.low;
      expectedLow = c2.high;
    }
    if (expectedLow >= expectedHigh) {
      diag.dropped_fvgs += 1;
      continue;
    }
    const highOff = Math.abs((fvg.high ?? 0) - expectedHigh);
    const lowOff = Math.abs((fvg.low ?? 0) - expectedLow);
    if (highOff > tolerance || lowOff > tolerance) {
      diag.snapped_fvgs += 1;
      diag.deltas.push({
        kind: "fvg", startIndex: si,
        claimed: { high: fvg.high, low: fvg.low },
        snapped: { high: expectedHigh, low: expectedLow },
      });
      out.push({ ...fvg, high: expectedHigh, low: expectedLow, snapped: true });
    } else {
      out.push(fvg);
    }
  }
  return out;
}

function snapLiquidity(liqs, candles, tolerance, diag) {
  const out = [];
  const n = candles.length;
  for (const liq of liqs) {
    const ci = liq.candleIndex;
    if (ci === undefined || ci === null || ci < 0 || ci >= n) {
      diag.dropped_liquidity += 1;
      continue;
    }
    const c = candles[ci];
    const expected = liq.type === "buyside" ? c.high : c.low;
    const off = Math.abs((liq.price ?? 0) - expected);
    if (off > tolerance) {
      diag.snapped_liquidity += 1;
      diag.deltas.push({
        kind: "liquidity", candleIndex: ci,
        claimed: { price: liq.price },
        snapped: { price: expected },
      });
      out.push({ ...liq, price: expected, snapped: true });
    } else {
      out.push(liq);
    }
  }
  return out;
}

export function snapAnalysisToCandles(analysis, candles, options = {}) {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const diag = makeDiagnostics();
  const obs = analysis.orderBlocks ?? [];
  const fvgs = analysis.fvgs ?? [];
  const liqs = analysis.liquidity ?? [];
  return {
    analysis: {
      ...analysis,
      orderBlocks: snapOrderBlocks(obs, candles, tolerance, diag),
      fvgs: snapFvgs(fvgs, candles, tolerance, diag),
      liquidity: snapLiquidity(liqs, candles, tolerance, diag),
    },
    diagnostics: diag,
  };
}

export function groupLiquidityByLevel(liquidity, tolerance = DEFAULT_TOLERANCE) {
  // Single-link clustering: sort by price, then merge an item into an existing
  // group (same type+tf) when any item already in the group is within tolerance.
  // This preserves the user's contract "items within $0.50 of each other collapse"
  // — bucket-based binning would split items that straddle bucket boundaries.
  const sorted = [...liquidity].sort((a, b) => (a.price ?? 0) - (b.price ?? 0));
  const groups = [];
  for (const liq of sorted) {
    const tf = liq.tf ?? "";
    const price = liq.price ?? 0;
    const matching = groups.find((g) =>
      g.type === liq.type &&
      g.tf === tf &&
      g.items.some((other) => Math.abs((other.price ?? 0) - price) <= tolerance)
    );
    if (matching) {
      matching.items.push(liq);
    } else {
      groups.push({ key: `${liq.type}|${price}|${tf}`, type: liq.type, tf, items: [liq] });
    }
  }
  // Sort each group's items by ascending candleIndex so the leftmost is first
  for (const g of groups) {
    g.items.sort((a, b) => (a.candleIndex ?? 0) - (b.candleIndex ?? 0));
  }
  return groups;
}
