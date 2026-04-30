export function scaleYDomain({ startDomain, anchorPrice, deltaY, chartHeight }) {
  const [startMin, startMax] = startDomain;
  const startSpan = startMax - startMin;
  const scaleFactor = Math.exp((deltaY / chartHeight) * 2);

  let newMin = anchorPrice - (anchorPrice - startMin) * scaleFactor;
  let newMax = anchorPrice + (startMax - anchorPrice) * scaleFactor;

  const newSpan = newMax - newMin;
  const minSpan = 0.1;
  const maxSpan = startSpan * 10;

  if (newSpan < minSpan) {
    const center = (newMin + newMax) / 2;
    newMin = center - minSpan / 2;
    newMax = center + minSpan / 2;
  } else if (newSpan > maxSpan) {
    const center = (newMin + newMax) / 2;
    newMin = center - maxSpan / 2;
    newMax = center + maxSpan / 2;
  }

  return [newMin, newMax];
}

export function scaleXRange({ startRange, anchorIndex, deltaX, chartWidth, allCandleIndices }) {
  const [startStart, startEnd] = startRange;
  const startSpan = startEnd - startStart + 1;
  const scaleFactor = Math.exp((-deltaX / chartWidth) * 2);

  const totalCandles = allCandleIndices.length;
  const minSpan = Math.min(5, totalCandles);
  const maxSpan = totalCandles;

  let newSpan = Math.round(startSpan * scaleFactor);
  newSpan = Math.max(minSpan, Math.min(maxSpan, newSpan));

  const fractionFromStart = startSpan === 1 ? 0 : (anchorIndex - startStart) / (startSpan - 1);
  let newStart = Math.round(anchorIndex - fractionFromStart * (newSpan - 1));

  const firstIdx = allCandleIndices[0];
  const lastIdx = allCandleIndices[allCandleIndices.length - 1];
  newStart = Math.max(firstIdx, Math.min(lastIdx - newSpan + 1, newStart));
  const newEnd = newStart + newSpan - 1;

  return [newStart, newEnd];
}

export function panXRange({ startRange, deltaX, bandWidth, allCandleIndices }) {
  const [startStart, startEnd] = startRange;
  const span = startEnd - startStart + 1;
  const candlesShift = Math.round(deltaX / bandWidth);

  const firstIdx = allCandleIndices[0];
  const lastIdx = allCandleIndices[allCandleIndices.length - 1];
  let newStart = startStart - candlesShift;
  newStart = Math.max(firstIdx, Math.min(lastIdx - span + 1, newStart));

  return [newStart, newStart + span - 1];
}

export function wheelZoom({
  startYDomain,
  startXRange,
  anchorPrice,
  anchorIndex,
  deltaY,
  modifiers,
  chartHeight,
  chartWidth,
  allCandleIndices,
}) {
  if (deltaY === 0) {
    return { yDomain: startYDomain, xRange: startXRange };
  }

  const zoomDirection = deltaY > 0 ? 1 : -1;
  const equivalentDeltaY = zoomDirection * chartHeight * 0.05;
  const equivalentDeltaX = -zoomDirection * chartWidth * 0.05;

  const updateY = !modifiers?.shift;
  const updateX = !modifiers?.ctrl;

  const yDomain = updateY
    ? scaleYDomain({ startDomain: startYDomain, anchorPrice, deltaY: equivalentDeltaY, chartHeight })
    : startYDomain;

  const xRange = updateX
    ? scaleXRange({ startRange: startXRange, anchorIndex, deltaX: equivalentDeltaX, chartWidth, allCandleIndices })
    : startXRange;

  return { yDomain, xRange };
}
