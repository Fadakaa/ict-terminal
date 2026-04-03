"""Tests for segmented Opus rejection policy (V3 Priority 4 extension).

Layers tested:
  Layer 1 — update_opus_tracker() stores timestamped events with killzone/timeframe/confidence
  Layer 2 — get_opus_rejection_policy() uses per-segment data with 3-tier action
  Layer 3 — build_opus_rejection_context() returns human-readable rejection track record
  Layer 4 — directional bias gap detection within build_opus_rejection_context()
"""
import json
import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from ml.claude_bridge import ClaudeAnalysisBridge


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bridge(tmp_path):
    bridge = ClaudeAnalysisBridge()
    bridge._accuracy_path = str(tmp_path / "accuracy.json")
    bridge._narrative_weights_path = str(tmp_path / "weights.json")
    bridge._accuracy = bridge._load_accuracy()
    bridge._narrative_weights = bridge._load_narrative_weights()
    return bridge


def _inject_events(bridge, events: list[dict]):
    """Directly inject events into the opus_tracker for setup."""
    tracker = bridge._accuracy.setdefault("opus_tracker", {})
    tracker["events"] = events


def _make_event(verdict="rejected", is_win=True, killzone="london",
                timeframe="1h", confidence=0.8, direction="long",
                pnl_rr=2.0, days_ago=0):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "verdict": verdict,
        "is_win": is_win,
        "killzone": killzone,
        "timeframe": timeframe,
        "confidence": confidence,
        "direction": direction,
        "pnl_rr": pnl_rr,
        "timestamp": ts,
    }


# ── Layer 1: update_opus_tracker() events ─────────────────────────────────────

class TestUpdateOpusTrackerEvents:

    def test_event_stored_with_killzone(self, tmp_path):
        """update_opus_tracker should store killzone in events list."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True, killzone="london")
        events = bridge._accuracy["opus_tracker"]["events"]
        assert len(events) == 1
        assert events[0]["killzone"] == "london"

    def test_event_stored_with_timeframe(self, tmp_path):
        """update_opus_tracker should store timeframe in events list."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", True, timeframe="4h")
        events = bridge._accuracy["opus_tracker"]["events"]
        assert events[0]["timeframe"] == "4h"

    def test_event_stored_with_confidence(self, tmp_path):
        """update_opus_tracker should store confidence in events list."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", False, confidence=0.9)
        events = bridge._accuracy["opus_tracker"]["events"]
        assert events[0]["confidence"] == pytest.approx(0.9)

    def test_event_stored_with_direction(self, tmp_path):
        """update_opus_tracker should store direction in events list."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", True, direction="short")
        events = bridge._accuracy["opus_tracker"]["events"]
        assert events[0]["direction"] == "short"

    def test_event_stored_with_pnl_rr(self, tmp_path):
        """update_opus_tracker should store pnl_rr in events list."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True, pnl_rr=3.5)
        events = bridge._accuracy["opus_tracker"]["events"]
        assert events[0]["pnl_rr"] == pytest.approx(3.5)

    def test_event_has_timestamp(self, tmp_path):
        """Each event should have an ISO timestamp."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True)
        events = bridge._accuracy["opus_tracker"]["events"]
        ts = events[0]["timestamp"]
        # Should parse as valid ISO datetime
        datetime.fromisoformat(ts)

    def test_events_accumulate_across_calls(self, tmp_path):
        """Multiple calls should append events."""
        bridge = _make_bridge(tmp_path)
        for _ in range(5):
            bridge.update_opus_tracker("rejected", True, killzone="london")
        events = bridge._accuracy["opus_tracker"]["events"]
        assert len(events) == 5

    def test_old_events_pruned_after_30_days(self, tmp_path):
        """Events older than 30 days should be pruned on the next update."""
        bridge = _make_bridge(tmp_path)
        # Inject 3 old events + 1 recent
        old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        tracker = bridge._accuracy.setdefault("opus_tracker", {})
        tracker["events"] = [
            {"verdict": "rejected", "is_win": True, "killzone": "london",
             "timeframe": "1h", "confidence": 0.8, "direction": "long",
             "pnl_rr": 1.0, "timestamp": old_ts},
        ] * 3
        # This call should trigger pruning
        bridge.update_opus_tracker("rejected", True, killzone="london")
        events = bridge._accuracy["opus_tracker"]["events"]
        # Only the fresh event survives
        assert len(events) == 1
        assert events[0]["killzone"] == "london"

    def test_global_aggregate_stats_preserved(self, tmp_path):
        """Existing global aggregate counters must still be updated (backward compat)."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("rejected", True)
        bridge.update_opus_tracker("rejected", False)
        tracker = bridge._accuracy["opus_tracker"]
        assert tracker["rejected_would_have_won"] == 1
        assert tracker["rejected_would_have_lost"] == 1

    def test_defaults_used_when_kwargs_omitted(self, tmp_path):
        """When called with just verdict + is_win, defaults should be used."""
        bridge = _make_bridge(tmp_path)
        bridge.update_opus_tracker("validated", True)
        event = bridge._accuracy["opus_tracker"]["events"][0]
        assert event["killzone"] == "Off"
        assert event["timeframe"] == "?"
        assert event["direction"] == "?"
        assert event["confidence"] == 0.5
        assert event["pnl_rr"] == 0.0


# ── Layer 2: get_opus_rejection_policy() segmented ────────────────────────────

class TestSegmentedRejectionPolicy:

    def test_falls_back_to_global_when_segment_too_small(self, tmp_path):
        """With <10 segment events, fall back to global stats."""
        bridge = _make_bridge(tmp_path)
        # 5 London rejections (not enough) but plenty globally via legacy stats
        bridge._accuracy["opus_tracker"] = {
            "rejected_would_have_won": 20,
            "rejected_would_have_lost": 5,
            "events": [_make_event(killzone="london") for _ in range(5)],
        }
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert policy["segment"] in ("global", "london_None")

    def test_uses_segment_when_enough_data(self, tmp_path):
        """With ≥10 segment rejections, use segment-specific stats."""
        bridge = _make_bridge(tmp_path)
        # 12 London rejections, 9 would have won (75% FN rate → allow)
        events = [_make_event(killzone="london", is_win=True) for _ in range(9)]
        events += [_make_event(killzone="london", is_win=False) for _ in range(3)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert "london" in policy["segment"]

    def test_fn_rate_below_35_gives_reject(self, tmp_path):
        """FN rate < 35% → action: reject."""
        bridge = _make_bridge(tmp_path)
        # 3 wins, 7 losses = 30% FN rate
        events = [_make_event(killzone="ny_pm", is_win=True) for _ in range(3)]
        events += [_make_event(killzone="ny_pm", is_win=False) for _ in range(7)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="ny_pm")
        assert policy["action"] == "reject"
        assert policy["false_negative_rate"] == pytest.approx(0.3, abs=0.01)

    def test_fn_rate_35_to_60_gives_downgrade(self, tmp_path):
        """FN rate 35–60% → action: downgrade."""
        bridge = _make_bridge(tmp_path)
        # 5 wins, 7 losses = 41.7% FN rate
        events = [_make_event(killzone="ny_am", is_win=True) for _ in range(5)]
        events += [_make_event(killzone="ny_am", is_win=False) for _ in range(7)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="ny_am")
        assert policy["action"] == "downgrade"

    def test_fn_rate_above_60_gives_allow(self, tmp_path):
        """FN rate ≥ 60% → action: allow."""
        bridge = _make_bridge(tmp_path)
        # 9 wins, 3 losses = 75% FN rate
        events = [_make_event(killzone="london", is_win=True) for _ in range(9)]
        events += [_make_event(killzone="london", is_win=False) for _ in range(3)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert policy["action"] == "allow"

    def test_fn_rate_exactly_60_gives_allow(self, tmp_path):
        """FN rate exactly 60% is on the boundary — should be 'allow'."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(killzone="london", is_win=True) for _ in range(6)]
        events += [_make_event(killzone="london", is_win=False) for _ in range(4)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert policy["action"] == "allow"

    def test_fn_rate_exactly_35_gives_downgrade(self, tmp_path):
        """FN rate exactly 35% is on the boundary — should be 'downgrade'."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(killzone="london", is_win=True) for _ in range(7)]
        events += [_make_event(killzone="london", is_win=False) for _ in range(13)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert policy["action"] == "downgrade"

    def test_policy_includes_sample_size(self, tmp_path):
        """Policy dict should include sample_size."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(killzone="london", is_win=True) for _ in range(10)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert "sample_size" in policy
        assert policy["sample_size"] == 10

    def test_policy_includes_segment_label(self, tmp_path):
        """Policy dict should include segment label."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(killzone="london", is_win=True) for _ in range(10)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert "london" in policy["segment"]

    def test_confidence_weighted_fn_rate_computed(self, tmp_path):
        """Policy should include weighted_fn_rate."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(killzone="london", is_win=True, confidence=0.9)
                  for _ in range(8)]
        events += [_make_event(killzone="london", is_win=False, confidence=0.5)
                   for _ in range(4)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert "weighted_fn_rate" in policy
        assert 0.0 <= policy["weighted_fn_rate"] <= 1.0

    def test_high_confidence_errors_flagged(self, tmp_path):
        """When weighted_fn > raw_fn by 15pp, flag high_confidence_errors."""
        bridge = _make_bridge(tmp_path)
        # High-confidence wins, low-confidence losses → weighted > raw by >15pp
        events = [_make_event(killzone="london", is_win=True, confidence=0.95)
                  for _ in range(8)]
        events += [_make_event(killzone="london", is_win=False, confidence=0.1)
                   for _ in range(4)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert "high_confidence_errors" in policy
        assert policy["high_confidence_errors"] is True

    def test_high_confidence_errors_not_flagged_when_gap_small(self, tmp_path):
        """When weighted_fn is close to raw_fn, high_confidence_errors should be False."""
        bridge = _make_bridge(tmp_path)
        # Uniform confidence — weighted ≈ raw
        events = [_make_event(killzone="ny_pm", is_win=True, confidence=0.7)
                  for _ in range(3)]
        events += [_make_event(killzone="ny_pm", is_win=False, confidence=0.7)
                   for _ in range(7)]
        _inject_events(bridge, events)
        policy = bridge.get_opus_rejection_policy(killzone="ny_pm")
        assert policy.get("high_confidence_errors") is False

    def test_no_events_falls_back_to_global_conservatively(self, tmp_path):
        """No events at all → conservative reject."""
        bridge = _make_bridge(tmp_path)
        policy = bridge.get_opus_rejection_policy(killzone="london")
        assert policy["action"] == "reject"

    def test_no_killzone_uses_global(self, tmp_path):
        """Calling without killzone uses global stats (backward compatible)."""
        bridge = _make_bridge(tmp_path)
        bridge._accuracy["opus_tracker"] = {
            "rejected_would_have_won": 5,
            "rejected_would_have_lost": 5,
            "events": [],
        }
        policy = bridge.get_opus_rejection_policy()
        # 5+5=10 samples, 50% FN — just over downgrade threshold?
        # backward compat: global policy still uses aggregate stats
        assert "action" in policy


# ── Layer 3: build_opus_rejection_context() ───────────────────────────────────

class TestBuildOpusRejectionContext:

    def test_returns_empty_string_with_no_rejections(self, tmp_path):
        """No rejection events → empty string returned."""
        bridge = _make_bridge(tmp_path)
        _inject_events(bridge, [])
        result = bridge.build_opus_rejection_context()
        assert result == ""

    def test_returns_empty_string_with_fewer_than_5_rejections(self, tmp_path):
        """Fewer than 5 rejections total → empty string."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london") for _ in range(4)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert result == ""

    def test_returns_string_with_enough_data(self, tmp_path):
        """5+ rejections in a killzone → returns non-empty string."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london") for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_line(self, tmp_path):
        """Output should include a header about rejection accuracy."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london") for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "REJECTION" in result.upper()

    def test_includes_london_when_enough_data(self, tmp_path):
        """London with ≥5 rejections should appear in the output."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london") for _ in range(7)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "london" in result.lower() or "London" in result

    def test_includes_fn_rate(self, tmp_path):
        """Context should include win rate / false negative percentage."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london", is_win=True)
                  for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        # Should contain a percentage
        assert "%" in result

    def test_includes_r_lost(self, tmp_path):
        """Context should include net R lost from false negatives."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="london",
                              is_win=True, pnl_rr=2.5)
                  for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "R" in result

    def test_excludes_sessions_with_fewer_than_5_rejections(self, tmp_path):
        """Sessions with <5 rejections should not appear in output."""
        bridge = _make_bridge(tmp_path)
        # ny_pm has only 3 rejections → should not appear
        events = [_make_event(verdict="rejected", killzone="ny_pm") for _ in range(3)]
        events += [_make_event(verdict="rejected", killzone="london") for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "ny_pm" not in result.lower()

    def test_over_filtering_note_for_high_fn_rate(self, tmp_path):
        """Killzones with FN rate >50% should get an 'over-filtering' warning."""
        bridge = _make_bridge(tmp_path)
        # 6 wins, 0 losses = 100% FN rate in London
        events = [_make_event(verdict="rejected", killzone="london", is_win=True)
                  for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "over-filter" in result.lower() or "over filter" in result.lower()

    def test_accurate_note_for_low_fn_rate(self, tmp_path):
        """Killzones with low FN rate should get a positive note."""
        bridge = _make_bridge(tmp_path)
        events = [_make_event(verdict="rejected", killzone="ny_pm", is_win=False)
                  for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "accurate" in result.lower()

    def test_empty_when_no_tracker_data(self, tmp_path):
        """No tracker at all → empty string."""
        bridge = _make_bridge(tmp_path)
        result = bridge.build_opus_rejection_context()
        assert result == ""


# ── Layer 4: Directional bias tracking ────────────────────────────────────────

class TestDirectionalBiasTracking:

    def test_direction_note_when_20pp_gap_both_sides_have_5_samples(self, tmp_path):
        """When longs have 80% FN and shorts have 30% FN (both ≥5 samples),
        context should include a directional note."""
        bridge = _make_bridge(tmp_path)
        # London: 8 long wins, 2 long losses = 80% long FN
        events = [_make_event(verdict="rejected", killzone="london",
                              is_win=True, direction="long") for _ in range(8)]
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=False, direction="long") for _ in range(2)]
        # London: 2 short wins, 8 short losses = 20% short FN
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=True, direction="short") for _ in range(2)]
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=False, direction="short") for _ in range(8)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        # Should mention longs/direction imbalance
        assert "long" in result.lower() or "LONG" in result

    def test_no_direction_note_when_gap_under_20pp(self, tmp_path):
        """When long/short FN rates are within 20pp, no directional note."""
        bridge = _make_bridge(tmp_path)
        # London: 5 long wins, 5 long losses = 50% FN
        events = [_make_event(verdict="rejected", killzone="london",
                              is_win=True, direction="long") for _ in range(5)]
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=False, direction="long") for _ in range(5)]
        # London: 4 short wins, 6 short losses = 40% FN (10pp gap — below threshold)
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=True, direction="short") for _ in range(4)]
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=False, direction="short") for _ in range(6)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        # No directional note expected
        assert "↳" not in result or "cautious" not in result.lower()

    def test_no_direction_note_when_one_direction_under_5_samples(self, tmp_path):
        """If only one direction has <5 samples, skip directional note."""
        bridge = _make_bridge(tmp_path)
        # 8 long wins (high FN), only 2 short samples (insufficient)
        events = [_make_event(verdict="rejected", killzone="london",
                              is_win=True, direction="long") for _ in range(8)]
        events += [_make_event(verdict="rejected", killzone="london",
                               is_win=False, direction="short") for _ in range(2)]
        _inject_events(bridge, events)
        result = bridge.build_opus_rejection_context()
        assert "↳" not in result
