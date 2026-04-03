"""Tests for ml/entry_placement.py — Entry Placement Refinement (V3 Priority 2)."""
import json
import os
import pytest


# ── Test compute_entry_position ──────────────────────────────────────

class TestComputeEntryPosition:
    """Entry position normalized 0.0 (shallow/worst) to 1.0 (deep/best)."""

    def test_long_at_zone_low_returns_1(self):
        from ml.entry_placement import compute_entry_position
        # For longs, zone_low = deepest (best fill in discount)
        pos = compute_entry_position(2645.0, 2650.0, 2645.0, "long")
        assert pos == 1.0

    def test_long_at_zone_high_returns_0(self):
        from ml.entry_placement import compute_entry_position
        # For longs, zone_high = shallowest (worst)
        pos = compute_entry_position(2650.0, 2650.0, 2645.0, "long")
        assert pos == 0.0

    def test_short_at_zone_high_returns_1(self):
        from ml.entry_placement import compute_entry_position
        # For shorts, zone_high = deepest (best fill in premium)
        pos = compute_entry_position(2680.0, 2680.0, 2675.0, "short")
        assert pos == 1.0

    def test_short_at_zone_low_returns_0(self):
        from ml.entry_placement import compute_entry_position
        pos = compute_entry_position(2675.0, 2680.0, 2675.0, "short")
        assert pos == 0.0

    def test_midpoint_returns_0_5(self):
        from ml.entry_placement import compute_entry_position
        pos = compute_entry_position(2647.5, 2650.0, 2645.0, "long")
        assert pos == pytest.approx(0.5)

    def test_degenerate_zone_returns_0_5(self):
        from ml.entry_placement import compute_entry_position
        # zone_high == zone_low → degenerate
        pos = compute_entry_position(2650.0, 2650.0, 2650.0, "long")
        assert pos == 0.5

    def test_clamps_outside_zone(self):
        from ml.entry_placement import compute_entry_position
        # Entry above zone high
        pos = compute_entry_position(2655.0, 2650.0, 2645.0, "long")
        assert 0.0 <= pos <= 1.0
        # Entry below zone low
        pos2 = compute_entry_position(2640.0, 2650.0, 2645.0, "long")
        assert 0.0 <= pos2 <= 1.0


# ── Test identify_entry_zone ─────────────────────────────────────────

class TestIdentifyEntryZone:
    """Zone identification with OB > FVG priority."""

    def _make_analysis(self):
        return {
            "orderBlocks": [
                {"type": "bullish", "high": 2650.0, "low": 2645.0},
                {"type": "bearish", "high": 2680.0, "low": 2675.0},
            ],
            "fvgs": [
                {"type": "bullish", "high": 2660.0, "low": 2655.0},
            ],
        }

    def test_inside_ob_returns_ob(self):
        from ml.entry_placement import identify_entry_zone
        zone = identify_entry_zone(2647.0, self._make_analysis(), atr=5.0)
        assert zone is not None
        assert zone["zone_type"] == "ob"
        assert zone["zone_subtype"] == "bullish"
        assert zone["contains_entry"] is True

    def test_inside_fvg_returns_fvg(self):
        from ml.entry_placement import identify_entry_zone
        zone = identify_entry_zone(2657.0, self._make_analysis(), atr=5.0)
        assert zone is not None
        assert zone["zone_type"] == "fvg"
        assert zone["contains_entry"] is True

    def test_ob_priority_over_fvg(self):
        from ml.entry_placement import identify_entry_zone
        # Create overlapping OB and FVG
        analysis = {
            "orderBlocks": [{"type": "bullish", "high": 2660.0, "low": 2655.0}],
            "fvgs": [{"type": "bullish", "high": 2660.0, "low": 2655.0}],
        }
        zone = identify_entry_zone(2657.0, analysis, atr=5.0)
        assert zone["zone_type"] == "ob"  # OB wins

    def test_nearby_within_half_atr(self):
        from ml.entry_placement import identify_entry_zone
        # Entry at 2651, OB is 2645-2650, mid=2647.5, dist=3.5
        # Use ATR=8 so threshold = 0.5*8 = 4.0 > 3.5
        zone = identify_entry_zone(2651.0, {"orderBlocks": [{"type": "bullish", "high": 2650.0, "low": 2645.0}], "fvgs": []}, atr=8.0)
        assert zone is not None
        assert zone["contains_entry"] is False

    def test_far_from_zones_returns_none(self):
        from ml.entry_placement import identify_entry_zone
        # Entry at 2700 — far from all zones
        zone = identify_entry_zone(2700.0, self._make_analysis(), atr=5.0)
        assert zone is None

    def test_handles_none_fields(self):
        from ml.entry_placement import identify_entry_zone
        # analysis with None instead of lists
        analysis = {"orderBlocks": None, "fvgs": None}
        zone = identify_entry_zone(2650.0, analysis, atr=5.0)
        assert zone is None

    def test_handles_empty_analysis(self):
        from ml.entry_placement import identify_entry_zone
        zone = identify_entry_zone(2650.0, {}, atr=5.0)
        assert zone is None


# ── Test EntryPlacementAnalyzer ──────────────────────────────────────

class TestEntryPlacementAnalyzer:
    """Placement statistics aggregation and guidance generation."""

    def _make_metrics(self, n=50, seed_wr=0.5):
        """Generate synthetic entry zone metrics."""
        import random
        rng = random.Random(42)
        metrics = []
        for i in range(n):
            position = rng.random()  # 0.0 to 1.0
            # Higher position = better outcome (for testing)
            is_win = rng.random() < (seed_wr + 0.2 * position)
            direction = "long" if i % 2 == 0 else "short"
            metrics.append({
                "setup_id": f"test-{i:04d}",
                "entry_price": 2650.0 + rng.uniform(-10, 10),
                "zone_type": "ob" if i % 3 != 0 else "fvg",
                "zone_subtype": "bullish" if direction == "long" else "bearish",
                "zone_high": 2655.0,
                "zone_low": 2645.0,
                "zone_size": 10.0,
                "zone_size_atr": 2.0,
                "entry_position": round(position, 4),
                "entry_depth_atr": round(position * 2, 4),
                "contains_entry": rng.random() > 0.2,
                "outcome": "tp1" if is_win else "stopped_out",
                "mfe_atr": round(rng.uniform(0.5, 4.0) if is_win else rng.uniform(0.1, 1.5), 4),
                "mae_atr": round(rng.uniform(0.2, 1.5), 4),
                "pnl_rr": round(rng.uniform(1.0, 3.0) if is_win else -1.0, 4),
                "killzone": rng.choice(["London", "NY_AM", "NY_PM"]),
                "timeframe": "1h",
                "direction": direction,
            })
        return metrics

    def test_bin_by_position_tertiles(self, tmp_path):
        """With < 200 metrics, uses 3 bins (tertiles)."""
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)

        for m in self._make_metrics(n=50):
            analyzer.ingest_metric(m)

        summary = analyzer.compute_summary(min_trades=10)
        bins = summary.get("overall_bins", [])
        assert len(bins) == 3  # Tertiles for < 200

    def test_bin_by_position_quintiles(self, tmp_path):
        """With >= 200 metrics, uses 5 bins (quintiles)."""
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)

        for m in self._make_metrics(n=220):
            analyzer.ingest_metric(m)

        summary = analyzer.compute_summary(min_trades=10)
        bins = summary.get("overall_bins", [])
        assert len(bins) == 5  # Quintiles for >= 200

    def test_find_optimal_bin(self, tmp_path):
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)

        for m in self._make_metrics(n=60):
            analyzer.ingest_metric(m)

        summary = analyzer.compute_summary(min_trades=5)
        optimal = summary.get("optimal_position", {})
        assert "range" in optimal
        assert optimal.get("count", 0) >= 5

    def test_guidance_insufficient_data(self, tmp_path):
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)
        # Only 5 metrics — below threshold
        for m in self._make_metrics(n=5):
            analyzer.ingest_metric(m)
        analyzer.compute_summary(min_trades=15)
        guidance = analyzer.get_placement_guidance()
        assert guidance["status"] == "insufficient_data"
        assert guidance["rules"] == []

    def test_guidance_generates_rules(self, tmp_path):
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)

        for m in self._make_metrics(n=80):
            analyzer.ingest_metric(m)

        analyzer.compute_summary(min_trades=10)
        guidance = analyzer.get_placement_guidance()
        assert guidance["status"] == "active"
        assert len(guidance["rules"]) >= 1
        assert guidance["total_trades"] == 80

    def test_direction_split_stats(self, tmp_path):
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}
        analyzer = EntryPlacementAnalyzer(config=cfg)

        for m in self._make_metrics(n=60):
            analyzer.ingest_metric(m)

        summary = analyzer.compute_summary(min_trades=5)
        by_dir = summary.get("by_direction", {})
        assert "long" in by_dir or "short" in by_dir

    def test_persistence(self, tmp_path):
        """Stats survive save/reload."""
        from ml.entry_placement import EntryPlacementAnalyzer
        cfg = {"model_dir": str(tmp_path)}

        analyzer1 = EntryPlacementAnalyzer(config=cfg)
        for m in self._make_metrics(n=30):
            analyzer1.ingest_metric(m)
        analyzer1.compute_summary(min_trades=5)

        # Reload
        analyzer2 = EntryPlacementAnalyzer(config=cfg)
        assert len(analyzer2._stats.get("metrics", [])) == 30


# ── Test extract_entry_zone_metrics ──────────────────────────────────

class TestExtractEntryZoneMetrics:

    def test_full_extraction(self):
        from ml.entry_placement import extract_entry_zone_metrics
        setup_row = {
            "id": "test-001",
            "entry_price": 2647.0,
            "direction": "long",
            "sl_price": 2643.0,
            "outcome": "tp1",
            "mfe_atr": 2.5,
            "mae_atr": 0.8,
            "pnl_rr": 2.0,
            "killzone": "London",
            "timeframe": "1h",
            "analysis_json": json.dumps({
                "orderBlocks": [
                    {"type": "bullish", "high": 2650.0, "low": 2645.0},
                ],
                "fvgs": [],
            }),
            "calibration_json": json.dumps({
                "volatility_context": {"atr": 4.0},
            }),
        }
        metric = extract_entry_zone_metrics(setup_row)
        assert metric is not None
        assert metric["zone_type"] == "ob"
        assert metric["setup_id"] == "test-001"
        assert 0.0 <= metric["entry_position"] <= 1.0
        # Entry at 2647 in zone 2645-2650, long → position = 1 - (2647-2645)/5 = 0.6
        assert metric["entry_position"] == pytest.approx(0.6, abs=0.01)

    def test_missing_analysis_returns_none(self):
        from ml.entry_placement import extract_entry_zone_metrics
        setup_row = {"id": "x", "entry_price": 2650.0, "analysis_json": None}
        assert extract_entry_zone_metrics(setup_row) is None

    def test_no_zone_found_returns_none(self):
        from ml.entry_placement import extract_entry_zone_metrics
        setup_row = {
            "id": "x",
            "entry_price": 2700.0,  # Far from any zone
            "direction": "long",
            "sl_price": 2695.0,
            "analysis_json": json.dumps({
                "orderBlocks": [{"type": "bullish", "high": 2650.0, "low": 2645.0}],
                "fvgs": [],
            }),
            "calibration_json": "{}",
        }
        assert extract_entry_zone_metrics(setup_row) is None

    def test_uses_calibration_atr_over_sl_estimate(self):
        from ml.entry_placement import extract_entry_zone_metrics
        setup_row = {
            "id": "test-atr",
            "entry_price": 2647.0,
            "direction": "long",
            "sl_price": 2640.0,  # SL dist = 7, estimate = 3.5
            "analysis_json": json.dumps({
                "orderBlocks": [{"type": "bullish", "high": 2650.0, "low": 2645.0}],
                "fvgs": [],
            }),
            "calibration_json": json.dumps({
                "volatility_context": {"atr": 5.0},  # Explicit ATR
            }),
        }
        metric = extract_entry_zone_metrics(setup_row)
        assert metric is not None
        # zone_size_atr should use calibration ATR (5.0), not SL estimate (3.5)
        assert metric["zone_size_atr"] == pytest.approx(1.0, abs=0.01)  # 5/5 = 1.0


# ── Test compute_live_mfe_mae ────────────────────────────────────────

class TestComputeLiveMfeMae:

    def test_long_mfe_mae(self):
        from ml.entry_placement import compute_live_mfe_mae
        candles = [
            {"high": 2655.0, "low": 2648.0},  # +5 fav, -2 adv
            {"high": 2660.0, "low": 2645.0},  # +10 fav, -5 adv
            {"high": 2652.0, "low": 2649.0},  # +2 fav, -1 adv
        ]
        result = compute_live_mfe_mae(candles, 2650.0, "long", atr=5.0)
        assert result["mfe"] == 10.0   # max high - entry
        assert result["mae"] == 5.0    # max entry - low
        assert result["mfe_atr"] == pytest.approx(2.0)
        assert result["mae_atr"] == pytest.approx(1.0)

    def test_short_mfe_mae(self):
        from ml.entry_placement import compute_live_mfe_mae
        candles = [
            {"high": 2652.0, "low": 2645.0},  # +5 fav (short), -2 adv
            {"high": 2655.0, "low": 2640.0},  # +10 fav, -5 adv
        ]
        result = compute_live_mfe_mae(candles, 2650.0, "short", atr=5.0)
        assert result["mfe"] == 10.0   # entry - min low
        assert result["mae"] == 5.0    # max high - entry
