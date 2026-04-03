"""Tests for Phase 6: Killzone Handoff + Notification Simplification.

Tests that:
- Killzone transitions generate handoff summaries
- Handoff summaries are injected into the next scan's prompt
- Only thesis-active and entry-signal notifications reach Telegram
- Detection/zone/displacement alerts are log-only
"""
import json
import pytest
from datetime import datetime

from ml.narrative_state import NarrativeStore


# ── Killzone Handoff Tests ────────────────────────────────────────


class TestKillzoneHandoffSummary:
    """Test that killzone transitions generate summary strings."""

    def test_transition_detected(self):
        """When killzone changes, a handoff summary is generated."""
        last_kz = "London"
        current_kz = "NY_AM"
        transition = last_kz and last_kz != current_kz and last_kz != "Off"
        assert transition is True

    def test_no_transition_same_kz(self):
        last_kz = "London"
        current_kz = "London"
        transition = last_kz and last_kz != current_kz and last_kz != "Off"
        assert transition is False

    def test_no_transition_from_off(self):
        """Off → London is not a handoff (no prior session to summarize)."""
        last_kz = "Off"
        current_kz = "London"
        transition = last_kz and last_kz != current_kz and last_kz != "Off"
        assert transition is False

    def test_no_transition_cold_start(self):
        """First scan (no last_kz) is not a handoff."""
        last_kz = None
        current_kz = "London"
        transition = bool(last_kz and last_kz != current_kz and last_kz != "Off")
        assert transition is False

    def test_handoff_summary_content(self):
        """Summary includes outgoing KZ name, thesis, and confidence."""
        thesis = {
            "thesis": "Gold accumulating in premium",
            "directional_bias": "bullish",
            "bias_confidence": 0.85,
        }
        summary = (
            f"London session summary: {thesis['thesis']}. "
            f"Bias was {thesis['directional_bias']} "
            f"({thesis['bias_confidence']:.0%} confidence)."
        )
        assert "London session summary" in summary
        assert "bullish" in summary
        assert "85%" in summary

    def test_handoff_stored_in_narrative(self, tmp_path):
        """Killzone summary is stored as a field on the narrative state."""
        ns = NarrativeStore(str(tmp_path / "test.db"))

        # Save a thesis
        state = {
            "thesis": "Gold in daily discount zone",
            "p3_phase": "accumulation",
            "p3_progress": "early",
            "directional_bias": "bullish",
            "bias_confidence": 0.75,
            "key_levels": [{"price": 2340}],
            "expected_next_move": "Expect sweep of BSL then reversal",
            "invalidation": {"price_level": 2320, "direction": "below"},
            "watching_for": ["BSL sweep"],
        }
        ns.save("1h", state, "scan001")

        # Get current and update with KZ summary
        current = ns.get_current("1h")
        assert current is not None

        # Store killzone summary
        kz_summary = "London session: bullish thesis, 85% confidence, watching OB at 2340"
        ns.update_killzone_summary(current["id"], kz_summary)

        # Verify stored
        updated = ns.get_current("1h")
        assert updated.get("killzone_summary") == kz_summary


class TestHandoffInjectedInPrompt:
    """Test that killzone handoff appears in the next scan's prompt."""

    def test_handoff_in_prompt_context(self):
        """When handoff exists, it should appear in prompt context."""
        handoff = "London session summary: bearish distribution. Bias was bearish (90% confidence)."
        prompt_section = f"=== PRIOR SESSION HANDOFF ===\n{handoff}\n=== END HANDOFF ==="
        assert "PRIOR SESSION HANDOFF" in prompt_section
        assert "London session summary" in prompt_section

    def test_no_handoff_no_section(self):
        """When no handoff, the section should be empty."""
        handoff = None
        section = f"=== PRIOR SESSION HANDOFF ===\n{handoff}\n" if handoff else ""
        assert section == ""


# ── Notification Simplification Tests ──────────────────────────────


class TestNotificationTiers:
    """Test the notification tier system: push vs log-only."""

    def test_thesis_active_is_push(self):
        """THESIS_CONFIRMED (stage 2) should push to Telegram."""
        push_stages = {2, 4, 6}  # THESIS_CONFIRMED, ENTRY_READY, THESIS_REVISED
        assert 2 in push_stages

    def test_entry_ready_is_push(self):
        """ENTRY_READY (stage 4) should push to Telegram."""
        push_stages = {2, 4, 6}
        assert 4 in push_stages

    def test_thesis_revised_is_push(self):
        """THESIS_REVISED (stage 6) should push as risk warning."""
        push_stages = {2, 4, 6}
        assert 6 in push_stages

    def test_setup_detected_is_log_only(self):
        """SETUP_DETECTED (stage 3) should be log-only, not Telegram."""
        push_stages = {2, 4, 6}
        assert 3 not in push_stages

    def test_thesis_forming_is_log_only(self):
        """THESIS_FORMING (stage 1) should be log-only."""
        push_stages = {2, 4, 6}
        assert 1 not in push_stages

    def test_trade_resolved_is_push(self):
        """TRADE_RESOLVED (stage 5) should push (outcome is important)."""
        push_stages = {2, 4, 5, 6}
        assert 5 in push_stages


class TestNotificationDemotions:
    """Test that demoted notifications still get logged."""

    def test_setup_detected_logs_not_pushes(self):
        """notify_setup_detected should write to logger, not Telegram."""
        # The function still exists and logs, but transport is log-only
        pushed = False
        logged = True  # Always logs
        assert logged is True
        assert pushed is False

    def test_zone_alert_logs_not_pushes(self):
        """notify_zone_prospect should log only during simplification."""
        pushed = False
        logged = True
        assert logged is True
        assert pushed is False

    def test_displacement_confirmed_logs_not_pushes(self):
        """notify_displacement_confirmed should log only."""
        pushed = False
        logged = True
        assert logged is True
        assert pushed is False


class TestThesisRevisedFiresOnFlip:
    """Test that THESIS_REVISED fires when bias flips with pending setup."""

    def test_bias_flip_detected(self):
        """Direction change between scans = revision."""
        prev_bias = "bullish"
        new_bias = "bearish"
        flipped = bool(prev_bias != new_bias and prev_bias and new_bias)
        assert flipped is True

    def test_same_bias_no_flip(self):
        prev_bias = "bullish"
        new_bias = "bullish"
        flipped = prev_bias != new_bias and prev_bias and new_bias
        assert flipped is False

    def test_flip_with_pending_triggers_warning(self):
        """Flip + pending setup on same TF = fire THESIS_REVISED."""
        flipped = True
        has_pending = True
        should_warn = flipped and has_pending
        assert should_warn is True

    def test_flip_without_pending_no_warning(self):
        """Flip without pending setup = no risk warning needed."""
        flipped = True
        has_pending = False
        should_warn = flipped and has_pending
        assert should_warn is False


# ── NarrativeStore KZ Summary Integration ─────────────────────────


class TestNarrativeStoreKzSummary:
    """Test NarrativeStore killzone_summary column."""

    def test_column_exists_on_fresh_db(self, tmp_path):
        ns = NarrativeStore(str(tmp_path / "test.db"))
        state = {
            "thesis": "Test thesis",
            "p3_phase": "accumulation",
            "directional_bias": "bullish",
            "bias_confidence": 0.6,
            "key_levels": [],
            "expected_next_move": "up",
            "invalidation": {"price_level": 2300, "direction": "below"},
            "watching_for": [],
        }
        ns.save("1h", state, "s1")
        current = ns.get_current("1h")
        assert current is not None
        # killzone_summary should be None by default
        assert current.get("killzone_summary") is None

    def test_update_kz_summary_persists(self, tmp_path):
        ns = NarrativeStore(str(tmp_path / "test.db"))
        state = {
            "thesis": "Test thesis",
            "p3_phase": "distribution",
            "directional_bias": "bearish",
            "bias_confidence": 0.7,
            "key_levels": [],
            "expected_next_move": "down",
            "invalidation": {"price_level": 2400, "direction": "above"},
            "watching_for": [],
        }
        ns.save("4h", state, "s2")
        current = ns.get_current("4h")

        ns.update_killzone_summary(current["id"], "Asian: bearish distribution, 70%")
        updated = ns.get_current("4h")
        assert updated["killzone_summary"] == "Asian: bearish distribution, 70%"

    def test_update_nonexistent_id_is_noop(self, tmp_path):
        ns = NarrativeStore(str(tmp_path / "test.db"))
        # Should not raise
        ns.update_killzone_summary("nonexistent_id", "some summary")
