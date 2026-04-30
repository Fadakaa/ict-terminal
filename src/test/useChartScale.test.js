import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChartScale } from "../useChartScale.js";

describe("useChartScale", () => {
  it("starts with null manual values (auto mode)", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
    expect(result.current.isManual).toBe(false);
  });

  it("isManual is true when yManualDomain is set", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => result.current.setYManualDomain([100, 200]));
    expect(result.current.isManual).toBe(true);
  });

  it("isManual is true when xManualRange is set", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => result.current.setXManualRange([10, 30]));
    expect(result.current.isManual).toBe(true);
  });

  it("reset() clears both manual values", () => {
    const { result } = renderHook(() => useChartScale("1h"));
    act(() => {
      result.current.setYManualDomain([100, 200]);
      result.current.setXManualRange([10, 30]);
    });
    act(() => result.current.reset());
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
    expect(result.current.isManual).toBe(false);
  });

  it("changing timeframe clears both manual values", () => {
    const { result, rerender } = renderHook(({ tf }) => useChartScale(tf), {
      initialProps: { tf: "1h" },
    });
    act(() => {
      result.current.setYManualDomain([100, 200]);
      result.current.setXManualRange([10, 30]);
    });
    rerender({ tf: "4h" });
    expect(result.current.yManualDomain).toBeNull();
    expect(result.current.xManualRange).toBeNull();
  });
});
