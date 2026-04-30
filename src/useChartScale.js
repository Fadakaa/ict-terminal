import { useState, useEffect, useCallback } from "react";

export function useChartScale(timeframe) {
  const [yManualDomain, setYManualDomain] = useState(null);
  const [xManualRange, setXManualRange] = useState(null);

  useEffect(() => {
    setYManualDomain(null);
    setXManualRange(null);
  }, [timeframe]);

  const reset = useCallback(() => {
    setYManualDomain(null);
    setXManualRange(null);
  }, []);

  const isManual = yManualDomain !== null || xManualRange !== null;

  return { yManualDomain, setYManualDomain, xManualRange, setXManualRange, reset, isManual };
}
