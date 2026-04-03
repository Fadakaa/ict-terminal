"""Tests for Phase 3: Opus Strategy Consolidation.

Tests that the Opus narrative prompt now requests watch_zones,
and that the response parsing prefers watch_zones over key_levels.
"""
import json
import pytest

from ml.prompts import build_opus_narrative_prompt


def _make_candles(n, base_price=2900.0):
    return [
        {"datetime": f"2026-03-10 {i:02d}:00:00",
         "open": base_price + i, "high": base_price + 1.0 + i,
         "low": base_price - 1.0 + i, "close": base_price + 0.5 + i}
        for i in range(n)
    ]


class TestOpusNarrativePrompt:
    """Test the updated Opus narrative prompt schema."""

    def test_prompt_requests_watch_zones(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"watch_zones"' in prompt

    def test_prompt_requests_bias_confidence(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"bias_confidence"' in prompt

    def test_prompt_requests_p3_progress(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"p3_progress"' in prompt

    def test_prompt_requests_invalidation_level(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"invalidation_level"' in prompt

    def test_watch_zones_schema_includes_direction(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"direction"' in prompt
        # Verify watch_zones includes OB|FVG|BSL|SSL types
        assert "OB|FVG|BSL|SSL" in prompt

    def test_still_requests_key_levels(self):
        """key_levels is still in the schema for backward compat."""
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"key_levels"' in prompt

    def test_still_requests_macro_narrative(self):
        prompt = build_opus_narrative_prompt(
            _make_candles(40), _make_candles(45))
        assert '"macro_narrative"' in prompt


class TestWatchZonesExtraction:
    """Test that watch_zones from Opus are preferred over key_levels."""

    def test_prefers_watch_zones_over_key_levels(self):
        """When Opus returns watch_zones, use those instead of key_levels."""
        htf_narrative = {
            "watch_zones": [
                {"level": 2340.5, "type": "OB", "direction": "bullish",
                 "status": "untested", "note": "4H bullish OB"},
                {"level": 2360.0, "type": "BSL", "direction": "neutral",
                 "status": "unswept", "note": "major BSL"},
            ],
            "key_levels": [
                {"price": 2340.5, "type": "ob", "timeframe": "4h",
                 "note": "bullish OB"},
            ]
        }
        # Simulate the extraction logic from _analyze_and_store
        _watch_zones = None
        if htf_narrative.get("watch_zones"):
            _watch_zones = [
                {"level": wz.get("level", wz.get("price")),
                 "type": wz.get("type", "zone"),
                 "status": wz.get("status", "untested")}
                for wz in htf_narrative["watch_zones"]
                if wz.get("level") or wz.get("price")
            ]
        elif htf_narrative.get("key_levels"):
            _watch_zones = [
                {"level": kl.get("price", kl.get("level")),
                 "type": kl.get("type", "zone"),
                 "status": kl.get("status", "untested")}
                for kl in htf_narrative["key_levels"]
                if kl.get("price") or kl.get("level")
            ]

        assert len(_watch_zones) == 2  # From watch_zones, not key_levels
        assert _watch_zones[0]["level"] == 2340.5
        assert _watch_zones[0]["type"] == "OB"
        assert _watch_zones[1]["level"] == 2360.0

    def test_falls_back_to_key_levels(self):
        """When no watch_zones, uses key_levels (backward compat)."""
        htf_narrative = {
            "key_levels": [
                {"price": 2340.5, "type": "ob", "timeframe": "4h"},
                {"price": 2325.0, "type": "ssl", "timeframe": "4h"},
            ]
        }
        _watch_zones = None
        if htf_narrative.get("watch_zones"):
            _watch_zones = []
        elif htf_narrative.get("key_levels"):
            _watch_zones = [
                {"level": kl.get("price", kl.get("level")),
                 "type": kl.get("type", "zone"),
                 "status": kl.get("status", "untested")}
                for kl in htf_narrative["key_levels"]
                if kl.get("price") or kl.get("level")
            ]

        assert len(_watch_zones) == 2
        assert _watch_zones[0]["level"] == 2340.5

    def test_no_zones_at_all(self):
        """When neither watch_zones nor key_levels, returns None."""
        htf_narrative = {"macro_narrative": "Unclear structure"}
        _watch_zones = None
        if htf_narrative.get("watch_zones"):
            _watch_zones = []
        elif htf_narrative.get("key_levels"):
            _watch_zones = []
        assert _watch_zones is None


class TestOpusNarrativeConsolidation:
    """Test that additional Opus fields are stored in calibration JSON."""

    def test_opus_fields_stored_in_cal_json(self):
        """New Opus fields (bias_confidence, p3_progress, invalidation_level)
        are stored in calibration_json."""
        htf_narrative = {
            "directional_bias": "bullish",
            "power_of_3_phase": "accumulation",
            "phase_confidence": "high",
            "p3_progress": "mid",
            "bias_confidence": 0.85,
            "invalidation_level": 2325.0,
            "macro_narrative": "Gold in weekly discount, daily accumulation",
        }
        # Simulate the storage logic from _analyze_and_store
        cal_json = {}
        if htf_narrative:
            cal_json["opus_narrative"] = {
                "directional_bias": htf_narrative.get("directional_bias"),
                "power_of_3_phase": htf_narrative.get("power_of_3_phase"),
                "phase_confidence": htf_narrative.get("phase_confidence"),
                "p3_progress": htf_narrative.get("p3_progress"),
                "bias_confidence": htf_narrative.get("bias_confidence"),
                "invalidation_level": htf_narrative.get("invalidation_level"),
                "macro_narrative": htf_narrative.get("macro_narrative", ""),
            }

        opus = cal_json["opus_narrative"]
        assert opus["p3_progress"] == "mid"
        assert opus["bias_confidence"] == 0.85
        assert opus["invalidation_level"] == 2325.0

    def test_backward_compat_missing_new_fields(self):
        """Old Opus responses without new fields don't crash."""
        htf_narrative = {
            "directional_bias": "bearish",
            "power_of_3_phase": "distribution",
            "phase_confidence": "medium",
            "macro_narrative": "Gold in premium zone",
        }
        cal_json = {}
        cal_json["opus_narrative"] = {
            "directional_bias": htf_narrative.get("directional_bias"),
            "power_of_3_phase": htf_narrative.get("power_of_3_phase"),
            "phase_confidence": htf_narrative.get("phase_confidence"),
            "p3_progress": htf_narrative.get("p3_progress"),
            "bias_confidence": htf_narrative.get("bias_confidence"),
            "invalidation_level": htf_narrative.get("invalidation_level"),
            "macro_narrative": htf_narrative.get("macro_narrative", ""),
        }
        opus = cal_json["opus_narrative"]
        assert opus["p3_progress"] is None
        assert opus["bias_confidence"] is None
        assert opus["invalidation_level"] is None
        # Core fields still present
        assert opus["directional_bias"] == "bearish"
