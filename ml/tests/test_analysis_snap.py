"""Tests for ml/analysis_snap.py — overlay snap-to-candle helpers."""
from ml.analysis_snap import snap_analysis_to_candles


def _candle(h, l, dt="2026-04-30 12:00"):
    return {"datetime": dt, "open": l, "high": h, "low": l, "close": h}


class TestOrderBlockSnap:
    def test_snaps_high_low_when_diverged(self):
        candles = [_candle(100, 90), _candle(110, 95), _candle(120, 105)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 80, "low": 70, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"][0]["high"] == 110
        assert out["orderBlocks"][0]["low"] == 95
        assert out["orderBlocks"][0]["snapped"] is True
        assert diag["snapped_obs"] == 1
        assert diag["dropped_obs"] == 0

    def test_within_tolerance_unchanged(self):
        candles = [_candle(100, 90), _candle(110.3, 95.2)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["orderBlocks"][0]
        assert diag["snapped_obs"] == 0

    def test_drops_missing_index(self):
        candles = [_candle(100, 90)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_drops_negative_index(self):
        candles = [_candle(100, 90)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91, "candleIndex": -1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_drops_oob_index(self):
        candles = [_candle(100, 90), _candle(110, 95)]
        analysis = {"orderBlocks": [{"type": "bullish", "high": 99, "low": 91, "candleIndex": 5}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1

    def test_preserves_other_fields(self):
        candles = [_candle(100, 90), _candle(110, 95)]
        analysis = {"orderBlocks": [{
            "type": "bearish", "high": 80, "low": 70, "candleIndex": 1,
            "tf": "1H", "strength": "strong", "note": "key zone",
        }]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        ob = out["orderBlocks"][0]
        assert ob["type"] == "bearish"
        assert ob["tf"] == "1H"
        assert ob["strength"] == "strong"
        assert ob["note"] == "key zone"
        assert ob["snapped"] is True


class TestFvgSnap:
    def test_snaps_bullish_to_gap_range(self):
        candles = [_candle(110, 100), _candle(118, 112), _candle(125, 120), _candle(130, 122)]
        analysis = {"fvgs": [{"type": "bullish", "high": 130, "low": 90, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"][0]["low"] == 110
        assert out["fvgs"][0]["high"] == 120
        assert out["fvgs"][0]["snapped"] is True
        assert diag["snapped_fvgs"] == 1

    def test_snaps_bearish_to_gap_range(self):
        candles = [_candle(110, 90), _candle(95, 85), _candle(85, 80)]
        analysis = {"fvgs": [{"type": "bearish", "high": 80, "low": 70, "startIndex": 0}]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"][0]["high"] == 90
        assert out["fvgs"][0]["low"] == 85

    def test_drops_degenerate_gap(self):
        # bullish: c0.high=120, c2.low=110 → expected_low (120) >= expected_high (110) → drop
        candles = [_candle(120, 100), _candle(115, 105), _candle(115, 110)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"] == []
        assert diag["dropped_fvgs"] == 1

    def test_drops_oob_startindex(self):
        candles = [_candle(110, 100), _candle(118, 112)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"] == []
        assert diag["dropped_fvgs"] == 1

    def test_within_tolerance_unchanged(self):
        candles = [_candle(110, 100), _candle(118, 112), _candle(125, 120.3)]
        analysis = {"fvgs": [{"type": "bullish", "high": 120, "low": 110, "startIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["fvgs"][0]
        assert diag["snapped_fvgs"] == 0


class TestLiquiditySnap:
    def test_snaps_buyside_to_high(self):
        candles = [_candle(100, 90), _candle(112.5, 100)]
        analysis = {"liquidity": [{"type": "buyside", "price": 105, "candleIndex": 1}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"][0]["price"] == 112.5
        assert out["liquidity"][0]["snapped"] is True
        assert diag["snapped_liquidity"] == 1

    def test_snaps_sellside_to_low(self):
        candles = [_candle(100, 90), _candle(110, 88.7)]
        analysis = {"liquidity": [{"type": "sellside", "price": 95, "candleIndex": 1}]}
        out, _ = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"][0]["price"] == 88.7

    def test_within_tolerance_unchanged(self):
        candles = [_candle(100.3, 90)]
        analysis = {"liquidity": [{"type": "buyside", "price": 100, "candleIndex": 0}]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert "snapped" not in out["liquidity"][0]
        assert diag["snapped_liquidity"] == 0

    def test_drops_missing_or_oob_index(self):
        candles = [_candle(100, 90)]
        analysis = {"liquidity": [
            {"type": "buyside", "price": 100},
            {"type": "sellside", "price": 90, "candleIndex": -1},
            {"type": "buyside", "price": 100, "candleIndex": 99},
        ]}
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"] == []
        assert diag["dropped_liquidity"] == 3


class TestIdempotency:
    def test_round_trip_no_snaps(self):
        candles = [_candle(100, 90), _candle(110, 95), _candle(120, 105), _candle(130, 115)]
        analysis = {
            "bias": "bullish",
            "orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "candleIndex": 1}],
            "fvgs": [{"type": "bullish", "high": 105, "low": 100, "startIndex": 0}],
            "liquidity": [{"type": "buyside", "price": 130, "candleIndex": 3}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert diag["snapped_obs"] == 0
        assert diag["snapped_fvgs"] == 0
        assert diag["snapped_liquidity"] == 0
        # Second pass on snapped output is also a no-op
        out2, diag2 = snap_analysis_to_candles(out, candles)
        assert diag2["snapped_obs"] == 0
        assert diag2["snapped_fvgs"] == 0
        assert diag2["snapped_liquidity"] == 0
        assert out2 == out  # structural equality


class TestCalibrateEndpointIntegration:
    def test_calibrate_snaps_analysis_and_returns_diagnostics(self):
        """Hitting /calibrate with diverged OBs returns snap_diagnostics in the response."""
        from starlette.testclient import TestClient
        from ml.server import app

        # Build candles where candle[1] has high=4598.5, low=4584.0
        candles = [
            {"datetime": f"2026-04-30 {i:02d}:00:00",
             "open": 4580 + i, "high": 4598.5 if i == 1 else 4595 + i,
             "low": 4584.0 if i == 1 else 4580 + i, "close": 4590 + i}
            for i in range(60)
        ]
        # Claude returns OB with high/low diverged by ~$60 from candle[1] wick
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bullish", "high": 4540.0, "low": 4520.0, "candleIndex": 1, "tf": "1H"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 4590.0, "direction": "long", "rationale": "test"},
            "stopLoss": {"price": 4580.0, "rationale": "test"},
            "takeProfits": [{"price": 4610.0, "rationale": "test", "rr": 2.0}],
            "killzone": "London",
            "confluences": ["test"],
        }

        with TestClient(app) as client:
            r = client.post("/calibrate", json={"analysis": analysis, "candles": candles})

        assert r.status_code == 200
        body = r.json()
        assert "snap_diagnostics" in body
        assert body["snap_diagnostics"]["snapped_obs"] == 1
        assert body["snap_diagnostics"]["dropped_obs"] == 0
        # The first delta should record the OB snap
        deltas = body["snap_diagnostics"]["deltas"]
        assert any(d["kind"] == "ob" and d["candleIndex"] == 1 for d in deltas)

    def test_calibrate_no_snaps_for_aligned_analysis(self):
        """When analysis already matches candles, /calibrate returns zero-count diagnostics."""
        from starlette.testclient import TestClient
        from ml.server import app

        candles = [
            {"datetime": f"2026-04-30 {i:02d}:00:00",
             "open": 4580 + i, "high": 4595 + i, "low": 4580 + i, "close": 4590 + i}
            for i in range(60)
        ]
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bullish", "high": 4596, "low": 4581, "candleIndex": 1, "tf": "1H"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 4590.0, "direction": "long", "rationale": "test"},
            "stopLoss": {"price": 4580.0, "rationale": "test"},
            "takeProfits": [{"price": 4610.0, "rationale": "test", "rr": 2.0}],
            "killzone": "London",
            "confluences": ["test"],
        }

        with TestClient(app) as client:
            r = client.post("/calibrate", json={"analysis": analysis, "candles": candles})

        assert r.status_code == 200
        body = r.json()
        assert body["snap_diagnostics"]["snapped_obs"] == 0
        assert body["snap_diagnostics"]["dropped_obs"] == 0


class TestScannerSnapIntegration:
    """Verify Scanner._analyze_and_store snaps Claude's analysis before storing/calibrating."""

    def test_snap_helper_called_with_trimmed_candles_frame(self):
        """Direct unit test on the snap call site logic.

        We don't spin up the full Scanner (heavy: DB, scheduler, OANDA fetcher).
        Instead, we verify the snap-helper contract that scanner.py relies on:
        passing the trimmed candle view (the same one sent to Claude) to the snap
        helper produces the right snap target.
        """
        from ml.analysis_snap import snap_analysis_to_candles

        # Trimmed view (60 candles) — the slice Claude was given
        trimmed = [
            {"datetime": f"2026-04-30 {i:02d}:00:00",
             "open": 4580 + i,
             "high": 4598.5 if i == 1 else 4595 + i,
             "low": 4584.0 if i == 1 else 4580 + i,
             "close": 4590 + i}
            for i in range(60)
        ]
        # Claude returns OB at candleIndex=1 with diverged values
        analysis = {
            "orderBlocks": [
                {"type": "bullish", "high": 4540.0, "low": 4520.0, "candleIndex": 1, "tf": "1H"},
            ],
            "fvgs": [], "liquidity": [],
        }

        snapped, diag = snap_analysis_to_candles(analysis, trimmed)

        # OB high/low replaced with the actual wick of trimmed[1]
        assert snapped["orderBlocks"][0]["high"] == 4598.5
        assert snapped["orderBlocks"][0]["low"] == 4584.0
        assert snapped["orderBlocks"][0]["snapped"] is True
        assert diag["snapped_obs"] == 1

    def test_scanner_snap_hook_present(self):
        """Smoke test: the scanner module imports snap_analysis_to_candles and uses
        it in _analyze_and_store. Catches regressions where someone removes the
        snap call."""
        import inspect
        from ml import scanner

        source = inspect.getsource(scanner.ScannerEngine._analyze_and_store)
        assert "snap_analysis_to_candles" in source, (
            "ScannerEngine._analyze_and_store must call snap_analysis_to_candles. "
            "If you renamed/moved the snap, update this test.")
        assert "snap_diagnostics" in source, (
            "ScannerEngine._analyze_and_store should bind snap_diagnostics for logging")


class TestAnchorDtResolution:
    def _candle_at(self, h, l, dt):
        return {"datetime": dt, "open": l, "high": h, "low": l, "close": h}

    def _build_candles(self):
        return [
            self._candle_at(100, 90, "2026-04-30 08:00"),
            self._candle_at(110, 95, "2026-04-30 09:00"),
            self._candle_at(120, 105, "2026-04-30 10:00"),
            self._candle_at(130, 115, "2026-04-30 11:00"),
        ]

    def test_resolves_ob_anchor_dt_to_candleindex(self):
        candles = self._build_candles()
        analysis = {
            "orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "anchor_dt": "2026-04-30 09:00"}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert len(out["orderBlocks"]) == 1
        assert out["orderBlocks"][0]["candleIndex"] == 1
        assert out["orderBlocks"][0]["anchor_dt"] == "2026-04-30 09:00"
        assert diag["snapped_obs"] == 0

    def test_drops_ob_and_increments_unresolved_anchor_on_no_match(self):
        candles = self._build_candles()
        analysis = {
            "orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "anchor_dt": "2099-01-01 00:00"}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"] == []
        assert diag["dropped_obs"] == 1
        assert diag["unresolved_anchor"] == 1

    def test_anchor_dt_wins_over_legacy_candleindex(self):
        candles = self._build_candles()
        analysis = {
            "orderBlocks": [{
                "type": "bullish",
                "high": 130, "low": 115,   # matches index 3 (11:00)
                "candleIndex": 0,          # wrong index
                "anchor_dt": "2026-04-30 11:00",  # correct datetime
            }],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"][0]["candleIndex"] == 3
        assert diag["snapped_obs"] == 0  # values already match candle[3]

    def test_falls_back_to_legacy_candleindex_when_no_anchor_dt(self):
        candles = self._build_candles()
        analysis = {
            "orderBlocks": [{"type": "bullish", "high": 110, "low": 95, "candleIndex": 1}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert len(out["orderBlocks"]) == 1
        assert out["orderBlocks"][0]["candleIndex"] == 1
        assert diag["unresolved_anchor"] == 0

    def test_snaps_high_low_when_anchor_dt_resolves_but_values_diverge(self):
        candles = self._build_candles()
        analysis = {
            "orderBlocks": [{"type": "bullish", "high": 200, "low": 50, "anchor_dt": "2026-04-30 09:00"}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["orderBlocks"][0]["high"] == 110
        assert out["orderBlocks"][0]["low"] == 95
        assert out["orderBlocks"][0]["snapped"] is True
        assert diag["snapped_obs"] == 1

    def test_resolves_fvg_anchor_dt_to_startindex(self):
        candles = self._build_candles()
        # bullish FVG: anchor=09:00 (idx 1), c0=idx1(h=110), c2=idx3(l=115)
        # expectedLow=110, expectedHigh=115
        analysis = {
            "fvgs": [{"type": "bullish", "high": 115, "low": 110, "anchor_dt": "2026-04-30 09:00"}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"][0]["startIndex"] == 1
        assert diag["snapped_fvgs"] == 0

    def test_drops_fvg_when_anchor_dt_plus_2_is_oob(self):
        candles = self._build_candles()
        # Only 4 candles. anchor_dt at idx 3 means startIndex+2=5 → OOB.
        analysis = {
            "fvgs": [{"type": "bullish", "high": 130, "low": 115, "anchor_dt": "2026-04-30 11:00"}],
        }
        out, diag = snap_analysis_to_candles(analysis, candles)
        assert out["fvgs"] == []
        assert diag["dropped_fvgs"] == 1

    def test_resolves_liquidity_anchor_dt(self):
        candles = self._build_candles()
        analysis = {
            "liquidity": [{"type": "buyside", "price": 130, "anchor_dt": "2026-04-30 11:00"}],
        }
        out, _ = snap_analysis_to_candles(analysis, candles)
        assert out["liquidity"][0]["candleIndex"] == 3
        assert out["liquidity"][0]["anchor_dt"] == "2026-04-30 11:00"
