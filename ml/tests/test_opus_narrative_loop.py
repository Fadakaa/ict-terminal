"""Tests for V3 Priority 4 — Opus Narrative Accuracy Loop optimizations.

Three optimizations:
1. Opus rejection override: downgrade instead of reject when false negative rate > 50%
2. Per-field trust levels injected into Sonnet's narrative prompt block
3. Killzone×phase cross-tab tracking for granular narrative accuracy
"""
import json
import os
import pytest
from unittest.mock import patch

from ml.claude_bridge import ClaudeAnalysisBridge
from ml.prompts import build_enhanced_ict_prompt


# ── Helpers ──────────────────────────────────────────────────────────

def _make_bridge(tmp_path):
    bridge = ClaudeAnalysisBridge()
    bridge._accuracy_path = str(tmp_path / "accuracy.json")
    bridge._narrative_weights_path = str(tmp_path / "weights.json")
    bridge._accuracy = bridge._load_accuracy()
    bridge._narrative_weights = bridge._load_narrative_weights()
    return bridge


def _make_candles(n=60):
    return [{"datetime": f"2026-03-20 {i % 24:02d}:00:00",
             "open": 3050, "high": 3055, "low": 3045, "close": 3051}
            for i in range(n)]


def _make_narrative():
    return {
        "macro_narrative": "Distribution phase complete, expect bearish continuation",
        "directional_bias": "bearish",
        "power_of_3_phase": "distribution",
        "premium_discount": "premium",
        "phase_confidence": "high",
        "dealing_range": {"high": 3060, "low": 3020},
        "key_levels": [{"price": 3050, "type": "OB"}],
        "intermarket_synthesis": "DXY rising supports gold short",
        "session_outlook": "Bearish continuation expected",
    }


# ── Opt 1: Opus Rejection Override ───────────────────────────────────

class TestOpusRejectionPolicy:

    def test_get_rejection_accuracy_empty(self, tmp_path):
        """No data → default to reject (conservative)."""
        bridge = _make_bridge(tmp_path)
        policy = bridge.get_opus_rejection_policy()
        assert policy["action"] == "reject"
        assert policy["false_negative_rate"] == 0

    def test_high_false_negative_triggers_downgrade(self, tmp_path):
        """When 35-59% of rejections would have won, switch to downgrade."""
        bridge = _make_bridge(tmp_path)
        # Simulate: 10 rejected winners, 14 rejected losers (41.7% FN rate)
        bridge._accuracy["opus_tracker"] = {
            "total_validations": 24, "rejected": 24,
            "validated": 0, "downgraded": 0,
            "rejected_would_have_won": 10,
            "rejected_would_have_lost": 14,
            "validated_wins": 0, "validated_losses": 0,
            "downgraded_wins": 0, "downgraded_losses": 0,
        }
        policy = bridge.get_opus_rejection_policy()
        assert policy["action"] == "downgrade"
        assert policy["false_negative_rate"] == pytest.approx(0.417, abs=0.01)

    def test_very_high_false_negative_triggers_allow(self, tmp_path):
        """When >=60% of rejections would have won, allow through (ignore rejection)."""
        bridge = _make_bridge(tmp_path)
        # Simulate: 18 rejected winners, 9 rejected losers (67% FN rate)
        bridge._accuracy["opus_tracker"] = {
            "total_validations": 27, "rejected": 27,
            "validated": 0, "downgraded": 0,
            "rejected_would_have_won": 18,
            "rejected_would_have_lost": 9,
            "validated_wins": 0, "validated_losses": 0,
            "downgraded_wins": 0, "downgraded_losses": 0,
        }
        policy = bridge.get_opus_rejection_policy()
        assert policy["action"] == "allow"
        assert policy["false_negative_rate"] == pytest.approx(0.667, abs=0.01)

    def test_low_false_negative_keeps_reject(self, tmp_path):
        """When <50% of rejections would have won, keep rejecting."""
        bridge = _make_bridge(tmp_path)
        bridge._accuracy["opus_tracker"] = {
            "total_validations": 20, "rejected": 20,
            "validated": 0, "downgraded": 0,
            "rejected_would_have_won": 5,
            "rejected_would_have_lost": 15,
            "validated_wins": 0, "validated_losses": 0,
            "downgraded_wins": 0, "downgraded_losses": 0,
        }
        policy = bridge.get_opus_rejection_policy()
        assert policy["action"] == "reject"
        assert policy["false_negative_rate"] == pytest.approx(0.25, abs=0.01)

    def test_insufficient_data_defaults_to_reject(self, tmp_path):
        """Fewer than 10 resolved rejections → conservative default."""
        bridge = _make_bridge(tmp_path)
        bridge._accuracy["opus_tracker"] = {
            "total_validations": 5, "rejected": 5,
            "validated": 0, "downgraded": 0,
            "rejected_would_have_won": 4,
            "rejected_would_have_lost": 1,
            "validated_wins": 0, "validated_losses": 0,
            "downgraded_wins": 0, "downgraded_losses": 0,
        }
        policy = bridge.get_opus_rejection_policy()
        # Only 5 resolved rejections — not enough data to override
        assert policy["action"] == "reject"

    def test_policy_includes_recommendation(self, tmp_path):
        """Policy should include the adjusted quality for downgrade (35-59% FN)."""
        bridge = _make_bridge(tmp_path)
        # 40% FN rate — in the downgrade zone (35–59%)
        bridge._accuracy["opus_tracker"] = {
            "total_validations": 20, "rejected": 20,
            "validated": 0, "downgraded": 0,
            "rejected_would_have_won": 8,
            "rejected_would_have_lost": 12,
            "validated_wins": 0, "validated_losses": 0,
            "downgraded_wins": 0, "downgraded_losses": 0,
        }
        policy = bridge.get_opus_rejection_policy()
        assert policy["action"] == "downgrade"
        assert policy["downgrade_to"] == "C"


# ── Opt 2: Per-Field Trust Levels in Prompt ──────────────────────────

class TestNarrativeTrustInjection:

    def test_trust_levels_in_narrative_block(self):
        """Narrative block should include per-field trust levels when weights provided."""
        narrative = _make_narrative()
        weights = {
            "directional_bias": 0.877,
            "p3_phase": 0.878,
            "premium_discount": 0.165,
            "confidence_calibration": 0.586,
            "intermarket_synthesis": 0.500,
            "key_levels": 0.200,
        }
        prompt = build_enhanced_ict_prompt(
            _make_candles(), [], htf_narrative=narrative,
            narrative_weights=weights)
        assert "FIELD ACCURACY" in prompt or "TRUST LEVEL" in prompt

    def test_high_trust_field_labeled(self):
        """Fields with weight >0.7 should be labeled HIGH trust."""
        narrative = _make_narrative()
        weights = {"directional_bias": 0.877, "p3_phase": 0.878,
                    "premium_discount": 0.165, "confidence_calibration": 0.586,
                    "intermarket_synthesis": 0.500, "key_levels": 0.200}
        prompt = build_enhanced_ict_prompt(
            _make_candles(), [], htf_narrative=narrative,
            narrative_weights=weights)
        # Directional bias should be labeled as high trust
        assert "HIGH" in prompt

    def test_low_trust_field_labeled(self):
        """Fields with weight <0.3 should be labeled LOW trust."""
        narrative = _make_narrative()
        weights = {"directional_bias": 0.877, "p3_phase": 0.878,
                    "premium_discount": 0.165, "confidence_calibration": 0.586,
                    "intermarket_synthesis": 0.500, "key_levels": 0.200}
        prompt = build_enhanced_ict_prompt(
            _make_candles(), [], htf_narrative=narrative,
            narrative_weights=weights)
        assert "LOW" in prompt

    def test_no_weights_no_trust_section(self):
        """Without weights, narrative block should NOT have trust levels."""
        narrative = _make_narrative()
        prompt = build_enhanced_ict_prompt(
            _make_candles(), [], htf_narrative=narrative)
        assert "FIELD ACCURACY" not in prompt

    def test_defers_conditionally_with_trust_data(self):
        """When trust data present, the defer instruction should be nuanced."""
        narrative = _make_narrative()
        weights = {"directional_bias": 0.877, "p3_phase": 0.878,
                    "premium_discount": 0.165, "confidence_calibration": 0.586,
                    "intermarket_synthesis": 0.500, "key_levels": 0.200}
        prompt = build_enhanced_ict_prompt(
            _make_candles(), [], htf_narrative=narrative,
            narrative_weights=weights)
        # Should still tell Sonnet to trust directional bias
        assert "directional bias" in prompt.lower() or "Directional Bias" in prompt


# ── Opt 3: Killzone×Phase Cross-Tab ──────────────────────────────────

class TestKillzonePhaseTracking:

    def test_update_tracks_killzone_phase(self, tmp_path):
        """update_narrative_tracker should accept killzone and phase."""
        bridge = _make_bridge(tmp_path)
        bridge.update_narrative_tracker(
            "bullish", "long", True,
            killzone="London", phase="distribution")
        tracker = bridge._accuracy["narrative_tracker"]
        kp = tracker.get("by_killzone_phase", {})
        assert "London_distribution" in kp
        assert kp["London_distribution"]["wins"] == 1

    def test_multiple_killzone_phase_entries(self, tmp_path):
        """Multiple trades in different segments tracked separately."""
        bridge = _make_bridge(tmp_path)
        bridge.update_narrative_tracker("bullish", "long", True,
                                         killzone="London", phase="distribution")
        bridge.update_narrative_tracker("bearish", "short", False,
                                         killzone="Asian", phase="accumulation")
        bridge.update_narrative_tracker("bullish", "long", True,
                                         killzone="London", phase="distribution")
        kp = bridge._accuracy["narrative_tracker"]["by_killzone_phase"]
        assert kp["London_distribution"]["total"] == 2
        assert kp["London_distribution"]["wins"] == 2
        assert kp["Asian_accumulation"]["total"] == 1
        assert kp["Asian_accumulation"]["wins"] == 0

    def test_win_rate_per_segment(self, tmp_path):
        """Each segment should compute its own win rate."""
        bridge = _make_bridge(tmp_path)
        for _ in range(8):
            bridge.update_narrative_tracker("bullish", "long", True,
                                             killzone="London", phase="distribution")
        for _ in range(2):
            bridge.update_narrative_tracker("bullish", "long", False,
                                             killzone="London", phase="distribution")
        kp = bridge._accuracy["narrative_tracker"]["by_killzone_phase"]
        assert kp["London_distribution"]["win_rate"] == pytest.approx(0.8, abs=0.01)

    def test_no_killzone_phase_still_works(self, tmp_path):
        """Backward compat: no killzone/phase args → no error."""
        bridge = _make_bridge(tmp_path)
        bridge.update_narrative_tracker("bullish", "long", True)
        tracker = bridge._accuracy["narrative_tracker"]
        assert tracker["aligned_wins"] == 1
        # by_killzone_phase should exist but be empty
        assert "by_killzone_phase" not in tracker or len(tracker.get("by_killzone_phase", {})) == 0

    def test_get_narrative_trust_by_segment(self, tmp_path):
        """get_narrative_trust_by_segment returns per-killzone×phase stats."""
        bridge = _make_bridge(tmp_path)
        for _ in range(15):
            bridge.update_narrative_tracker("bullish", "long", True,
                                             killzone="London", phase="distribution")
        for _ in range(5):
            bridge.update_narrative_tracker("bullish", "long", False,
                                             killzone="London", phase="distribution")
        result = bridge.get_narrative_trust_by_segment(min_trades=10)
        assert "London_distribution" in result
        assert result["London_distribution"]["win_rate"] == pytest.approx(0.75, abs=0.01)
        assert result["London_distribution"]["total"] == 20

    def test_get_narrative_trust_filters_low_count(self, tmp_path):
        """Segments with fewer than min_trades are excluded."""
        bridge = _make_bridge(tmp_path)
        for _ in range(3):
            bridge.update_narrative_tracker("bullish", "long", True,
                                             killzone="Asian", phase="accumulation")
        result = bridge.get_narrative_trust_by_segment(min_trades=10)
        assert "Asian_accumulation" not in result
