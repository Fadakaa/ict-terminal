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
