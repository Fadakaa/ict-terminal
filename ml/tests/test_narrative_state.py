"""Tests for the Narrative State Engine — persistent per-timeframe thesis tracking."""
import json
import os
import pytest
from datetime import datetime, timedelta

from ml.narrative_state import NarrativeStore, check_invalidation, THESIS_MAX_AGE, CONFIDENCE_DECAY_RATE


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def ns_store(tmp_path):
    """NarrativeStore backed by a temp database."""
    db = str(tmp_path / "test_scanner.db")
    return NarrativeStore(db_path=db)


@pytest.fixture
def sample_state():
    """A valid narrative state dict as Claude would return."""
    return {
        "thesis": "Smart money accumulated longs during Asia. Manipulation sweeping SSL at 2338.",
        "p3_phase": "manipulation",
        "p3_progress": "late",
        "directional_bias": "bullish",
        "bias_confidence": 0.75,
        "key_levels": [
            {"price": 2338, "label": "SSL sweep target", "role": "target"},
            {"price": 2348, "label": "Distribution target", "role": "target"},
        ],
        "expected_next_move": "Bearish sweep of SSL at 2338 followed by bullish reversal",
        "invalidation": {
            "condition": "Price closes below 2335 (below accumulation range)",
            "price_level": 2335,
            "direction": "below",
        },
        "watching_for": ["displacement below 2338", "new bullish OB formation"],
        "last_revision": None,
    }


@pytest.fixture
def sample_candles():
    """Candles for invalidation testing."""
    base = 2340.0
    candles = []
    for i in range(20):
        o = base + i * 0.2
        candles.append({
            "datetime": f"2026-03-31 {10 + i // 4:02d}:{(i % 4) * 15:02d}:00",
            "open": round(o, 2),
            "high": round(o + 2.0, 2),
            "low": round(o - 1.5, 2),
            "close": round(o + 0.5, 2),
        })
    return candles


# ── Component 2: NarrativeStore tests ────────────────────────────────

class TestNarrativeStore:
    """Tests for NarrativeStore persistence and logic."""

    def test_save_and_get_current(self, ns_store, sample_state):
        """First save creates active state, retrievable by timeframe."""
        state_id = ns_store.save("1h", sample_state)
        assert state_id is not None

        current = ns_store.get_current("1h")
        assert current is not None
        assert current["thesis"] == sample_state["thesis"]
        assert current["p3_phase"] == "manipulation"
        assert current["directional_bias"] == "bullish"
        assert current["scan_count"] == 1
        assert current["status"] == "active"

    def test_get_current_returns_none_when_empty(self, ns_store):
        """No state yet → None."""
        assert ns_store.get_current("15min") is None

    def test_get_current_respects_timeframe(self, ns_store, sample_state):
        """States are isolated per timeframe."""
        ns_store.save("1h", sample_state)
        assert ns_store.get_current("1h") is not None
        assert ns_store.get_current("4h") is None

    def test_continuation_increments_scan_count(self, ns_store, sample_state):
        """Same bias + phase = continuation → scan_count increments."""
        ns_store.save("1h", sample_state)

        # Second scan with same bias and phase
        state2 = dict(sample_state)
        state2["thesis"] = "Manipulation continues, sweep progressing."
        state2["p3_progress"] = "late"  # same phase
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 2
        assert current["is_revision"] == 0

    def test_revision_resets_scan_count(self, ns_store, sample_state):
        """Different bias = revision → scan_count resets to 1."""
        ns_store.save("1h", sample_state)

        # Second scan with different bias
        state2 = dict(sample_state)
        state2["directional_bias"] = "bearish"
        state2["thesis"] = "Thesis reversed to bearish."
        state2["last_revision"] = "Invalidation triggered, switching bearish."
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 1
        assert current["is_revision"] == 1
        assert current["directional_bias"] == "bearish"

    def test_phase_change_triggers_revision(self, ns_store, sample_state):
        """Different p3_phase = revision even with same bias."""
        ns_store.save("1h", sample_state)

        state2 = dict(sample_state)
        state2["p3_phase"] = "distribution"  # was manipulation
        state2["last_revision"] = "Moved to distribution phase."
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 1
        assert current["is_revision"] == 1

    def test_neutral_to_directional_is_revision(self, ns_store, sample_state):
        """neutral → bullish = revision (new conviction forming)."""
        neutral = dict(sample_state)
        neutral["directional_bias"] = "neutral"
        ns_store.save("1h", neutral)

        # Now form conviction
        state2 = dict(sample_state)
        state2["directional_bias"] = "bullish"
        state2["last_revision"] = "Conviction formed: bullish."
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["is_revision"] == 1
        assert current["scan_count"] == 1

    def test_p3_progress_change_is_continuation(self, ns_store, sample_state):
        """Same phase + bias but different progress = continuation (thesis evolving)."""
        ns_store.save("1h", sample_state)  # manipulation, late

        state2 = dict(sample_state)
        state2["p3_progress"] = "mid"  # progress changed, phase same
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 2  # continuation
        assert current["is_revision"] == 0

    def test_previous_state_superseded(self, ns_store, sample_state):
        """Saving a new state marks the previous as 'superseded'."""
        ns_store.save("1h", sample_state)
        first = ns_store.get_current("1h")
        first_id = first["id"]

        state2 = dict(sample_state)
        state2["thesis"] = "Updated thesis."
        ns_store.save("1h", state2)

        # First state should now be superseded
        history = ns_store.get_history("1h", limit=10)
        old = [h for h in history if h["id"] == first_id]
        assert len(old) == 1
        assert old[0]["status"] == "superseded"

    def test_mandatory_invalidation_safeguard(self, ns_store, sample_state):
        """Narrative without concrete invalidation is NOT persisted."""
        no_inv = dict(sample_state)
        no_inv["invalidation"] = {}  # Missing price_level and direction
        result = ns_store.save("1h", no_inv)
        assert result is None
        assert ns_store.get_current("1h") is None

    def test_mandatory_invalidation_no_level(self, ns_store, sample_state):
        """Invalidation with direction but no price_level → not persisted."""
        bad_inv = dict(sample_state)
        bad_inv["invalidation"] = {"direction": "below"}
        result = ns_store.save("1h", bad_inv)
        assert result is None

    def test_get_history(self, ns_store, sample_state):
        """History returns states in reverse chronological order."""
        for i in range(5):
            s = dict(sample_state)
            s["thesis"] = f"Thesis version {i}"
            # Force revision for different ids
            if i % 2:
                s["directional_bias"] = "bearish"
            else:
                s["directional_bias"] = "bullish"
            ns_store.save("1h", s)

        history = ns_store.get_history("1h", limit=3)
        assert len(history) == 3
        assert history[0]["thesis"] == "Thesis version 4"

    def test_expire_stale(self, ns_store, sample_state):
        """Stale theses get expired."""
        ns_store.save("15min", sample_state)

        # Manually backdate the created_at to be stale
        with ns_store._conn() as conn:
            old_time = (datetime.utcnow() - timedelta(hours=3)).isoformat()
            conn.execute(
                "UPDATE narrative_states SET created_at = ? WHERE status = 'active'",
                (old_time,))

        ns_store.expire_stale("15min")
        current = ns_store.get_current("15min")
        assert current is None  # expired

    def test_expire_stale_respects_timeframe_thresholds(self, ns_store, sample_state):
        """4H thesis with 12h age is NOT stale (threshold is 24h)."""
        ns_store.save("4h", sample_state)

        with ns_store._conn() as conn:
            old_time = (datetime.utcnow() - timedelta(hours=12)).isoformat()
            conn.execute(
                "UPDATE narrative_states SET created_at = ? WHERE status = 'active'",
                (old_time,))

        ns_store.expire_stale("4h")
        current = ns_store.get_current("4h")
        assert current is not None  # still active

    def test_revision_rate(self, ns_store, sample_state):
        """Revision rate is calculated correctly."""
        # 3 continuations then 2 revisions
        for i in range(5):
            s = dict(sample_state)
            if i >= 3:
                s["directional_bias"] = "bearish"
                s["last_revision"] = "Changed"
            ns_store.save("1h", s)

        rate = ns_store.get_revision_rate("1h", window=5)
        # 2 revisions out of 5 (note: first save is never a revision)
        assert 0.2 <= rate <= 0.6  # at least some revisions

    def test_confidence_decay(self, ns_store, sample_state):
        """Confidence decays by CONFIDENCE_DECAY_RATE."""
        ns_store.save("1h", sample_state)
        original_conf = sample_state["bias_confidence"]

        new_conf = ns_store.apply_confidence_decay("1h")
        assert new_conf is not None
        assert abs(new_conf - (original_conf - CONFIDENCE_DECAY_RATE)) < 0.01

    def test_confidence_decay_floors_at_zero(self, ns_store, sample_state):
        """Confidence cannot go below 0."""
        low_conf = dict(sample_state)
        low_conf["bias_confidence"] = 0.05
        ns_store.save("1h", low_conf)

        new_conf = ns_store.apply_confidence_decay("1h")
        assert new_conf == 0.0

    def test_confidence_decay_no_active_thesis(self, ns_store):
        """Decay returns None when no active thesis."""
        assert ns_store.apply_confidence_decay("1h") is None


# ── Component 3: Prediction scoring ─────────────────────────────────

class TestPredictionScoring:
    """Tests for prediction hit/miss scoring."""

    def test_revision_scores_previous_as_miss(self, ns_store, sample_state):
        """When thesis is revised, previous predictions are scored as miss."""
        ns_store.save("1h", sample_state)
        first = ns_store.get_current("1h")
        first_id = first["id"]

        # Revision
        state2 = dict(sample_state)
        state2["directional_bias"] = "bearish"
        state2["last_revision"] = "Market reversed."
        ns_store.save("1h", state2)

        history = ns_store.get_history("1h")
        old = [h for h in history if h["id"] == first_id][0]
        assert old["prediction_hit"] == 0

    def test_continuation_with_matching_bias_scores_hit(self, ns_store, sample_state):
        """Continuation where bias matches expected direction → hit."""
        # Expected move is bullish, continuation with bullish bias
        ns_store.save("1h", sample_state)
        first = ns_store.get_current("1h")
        first_id = first["id"]

        state2 = dict(sample_state)
        state2["thesis"] = "Bullish continuation confirmed."
        ns_store.save("1h", state2)

        history = ns_store.get_history("1h")
        old = [h for h in history if h["id"] == first_id][0]
        assert old["prediction_hit"] == 1

    def test_accuracy_metrics(self, ns_store, sample_state):
        """Aggregate accuracy metrics are calculated."""
        # Save 3 states
        for i in range(3):
            s = dict(sample_state)
            s["thesis"] = f"Thesis {i}"
            ns_store.save("1h", s)

        metrics = ns_store.get_accuracy_metrics("1h")
        assert "prediction_accuracy" in metrics
        assert "thesis_stability" in metrics
        assert "revision_rate" in metrics
        assert metrics["total_states"] == 3

    def test_accuracy_metrics_all_timeframes(self, ns_store, sample_state):
        """Metrics without timeframe filter span all TFs."""
        ns_store.save("1h", sample_state)
        ns_store.save("4h", sample_state)

        metrics = ns_store.get_accuracy_metrics()
        assert metrics["total_states"] == 2


# ── Component 4: Invalidation detector ──────────────────────────────

class TestCheckInvalidation:
    """Tests for the pre-prompt invalidation price check."""

    def test_clear_when_price_far_from_level(self, sample_state, sample_candles):
        """Price well above invalidation level → CLEAR."""
        result = check_invalidation(sample_state, 2350.0, sample_candles)
        assert result == "CLEAR"

    def test_triggered_when_price_below_level(self, sample_state):
        """Price closed below 'below' invalidation → TRIGGERED."""
        # Invalidation: below 2335
        candles = [{"close": 2334.0}] * 3
        result = check_invalidation(sample_state, 2334.0, candles)
        assert result == "TRIGGERED"

    def test_triggered_above_direction(self):
        """Bearish thesis with 'above' invalidation — price closed above → TRIGGERED."""
        state = {
            "invalidation": {
                "condition": "Price reclaims above 2355",
                "price_level": 2355,
                "direction": "above",
            }
        }
        candles = [{"close": 2356.0}] * 3
        result = check_invalidation(state, 2356.0, candles)
        assert result == "TRIGGERED"

    def test_approaching_within_half_atr(self, sample_state, sample_candles):
        """Price within 0.5 ATR of invalidation → APPROACHING."""
        # Invalidation at 2335. Set price close to it but not below.
        # With synthetic candles, ATR is ~3.5, so 0.5*ATR ~ 1.75
        result = check_invalidation(sample_state, 2335.8, sample_candles)
        assert result == "APPROACHING"

    def test_clear_when_no_invalidation(self, sample_candles):
        """State with no invalidation → CLEAR."""
        state = {"invalidation": {}}
        result = check_invalidation(state, 2340.0, sample_candles)
        assert result == "CLEAR"

    def test_triggered_needs_candle_close(self, sample_state):
        """Only one candle close below → still TRIGGERED (threshold is 1)."""
        candles = [{"close": 2340.0}, {"close": 2340.0}, {"close": 2334.0}]
        result = check_invalidation(sample_state, 2334.0, candles)
        assert result == "TRIGGERED"

    def test_not_triggered_by_wick_only(self, sample_state):
        """Wicks below invalidation don't count — only closes."""
        candles = [
            {"close": 2340.0, "low": 2330.0},  # wick below but close above
            {"close": 2338.0},
            {"close": 2336.0},  # all closes above 2335
        ]
        result = check_invalidation(sample_state, 2336.0, candles)
        assert result != "TRIGGERED"


# ── Component 5: Prompt injection ────────────────────────────────────

class TestPromptInjection:
    """Tests for the narrative state prompt section builder."""

    def test_no_narrative_returns_empty(self):
        """No previous thesis → empty string."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(None, None)
        assert result == ""

    def test_basic_section_includes_thesis(self, sample_state):
        """Section includes the thesis text."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "CLEAR")
        assert "YOUR PREVIOUS THESIS" in result
        assert sample_state["thesis"] in result
        assert "manipulation" in result
        assert "bullish" in result

    def test_section_includes_watching_for(self, sample_state):
        """Watching_for items appear in the section."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "CLEAR")
        assert "displacement below 2338" in result

    def test_section_includes_expected(self, sample_state):
        """Expected next move appears."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "CLEAR")
        assert "Bearish sweep of SSL" in result

    def test_invalidation_triggered_warning(self, sample_state):
        """TRIGGERED status produces strong warning."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "TRIGGERED")
        assert "INVALIDATION TRIGGERED" in result
        assert "MUST re-assess" in result

    def test_approaching_warning(self, sample_state):
        """APPROACHING status produces note."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "APPROACHING")
        assert "approaching your invalidation" in result

    def test_stale_thesis_warning(self, sample_state):
        """Expired thesis shows stale warning."""
        from ml.prompts import _build_narrative_state_section
        stale = dict(sample_state)
        stale["status"] = "expired"
        stale["thesis_age_minutes"] = 360
        result = _build_narrative_state_section(stale, "CLEAR")
        assert "STALE" in result

    def test_low_confidence_warning(self, sample_state):
        """Confidence below 30% shows conviction warning."""
        from ml.prompts import _build_narrative_state_section
        low = dict(sample_state)
        low["bias_confidence"] = 0.25
        result = _build_narrative_state_section(low, "CLEAR")
        assert "losing conviction" in result

    def test_key_levels_in_section(self, sample_state):
        """Key levels appear in the prompt section."""
        from ml.prompts import _build_narrative_state_section
        result = _build_narrative_state_section(sample_state, "CLEAR")
        assert "2338" in result
        assert "SSL sweep target" in result

    def test_prompt_signature_accepts_new_params(self, sample_state):
        """build_enhanced_ict_prompt accepts prev_narrative + invalidation_status."""
        from ml.prompts import build_enhanced_ict_prompt
        candles = [{"datetime": "2026-03-31 10:00", "open": 2340, "high": 2342,
                    "low": 2338, "close": 2341}] * 10
        h4 = candles[:5]

        result = build_enhanced_ict_prompt(
            candles, h4,
            prev_narrative=sample_state,
            invalidation_status="CLEAR")
        assert "YOUR PREVIOUS THESIS" in result
        assert "narrative_state" in result  # In JSON schema

    def test_narrative_state_in_json_schema(self):
        """The JSON response schema includes narrative_state fields."""
        from ml.prompts import build_enhanced_ict_prompt
        candles = [{"datetime": "2026-03-31 10:00", "open": 2340, "high": 2342,
                    "low": 2338, "close": 2341}] * 10
        result = build_enhanced_ict_prompt(candles, candles[:5])
        assert '"narrative_state"' in result
        assert '"thesis"' in result
        assert '"p3_phase"' in result
        assert '"invalidation"' in result
        assert '"watching_for"' in result


# ── Component 6: Anti-anchoring safeguards ───────────────────────────

class TestAntiAnchoringSafeguards:
    """Tests for safeguards that prevent thesis anchoring."""

    def test_safeguard_1_mandatory_invalidation(self, ns_store, sample_state):
        """State without invalidation is not persisted."""
        bad = dict(sample_state)
        bad["invalidation"] = {"condition": "Some condition"}  # no price_level
        result = ns_store.save("1h", bad)
        assert result is None

    def test_safeguard_2_confidence_decay(self, ns_store, sample_state):
        """Confidence decays when thesis stalls."""
        ns_store.save("1h", sample_state)
        # Apply 3 rounds of decay
        for _ in range(3):
            ns_store.apply_confidence_decay("1h")
        current = ns_store.get_current("1h")
        expected = max(0.0, 0.75 - 3 * CONFIDENCE_DECAY_RATE)
        assert abs(current["bias_confidence"] - expected) < 0.01

    def test_safeguard_4_max_thesis_age_15min(self, ns_store, sample_state):
        """15min thesis expires after 2 hours."""
        ns_store.save("15min", sample_state)

        # Backdate to 2.5 hours ago
        with ns_store._conn() as conn:
            old = (datetime.utcnow() - timedelta(minutes=150)).isoformat()
            conn.execute(
                "UPDATE narrative_states SET created_at = ? WHERE status = 'active'",
                (old,))

        ns_store.expire_stale("15min")
        assert ns_store.get_current("15min") is None

    def test_safeguard_4_max_thesis_age_1day(self, ns_store, sample_state):
        """1day thesis persists beyond 48h but expires after 72h."""
        ns_store.save("1day", sample_state)

        # 50 hours ago — still valid
        with ns_store._conn() as conn:
            old = (datetime.utcnow() - timedelta(hours=50)).isoformat()
            conn.execute(
                "UPDATE narrative_states SET created_at = ? WHERE status = 'active'",
                (old,))
        ns_store.expire_stale("1day")
        assert ns_store.get_current("1day") is not None

        # 73 hours ago — expired
        with ns_store._conn() as conn:
            old = (datetime.utcnow() - timedelta(hours=73)).isoformat()
            conn.execute(
                "UPDATE narrative_states SET created_at = ? WHERE status = 'active'",
                (old,))
        ns_store.expire_stale("1day")
        assert ns_store.get_current("1day") is None


# ── Integration ──────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end narrative state flow tests."""

    def test_three_scan_evolution(self, ns_store, sample_state):
        """Simulates the 3-scan example from the spec."""
        # Scan 1: Fresh thesis
        ns_store.save("1h", sample_state)
        current = ns_store.get_current("1h")
        assert current["scan_count"] == 1

        # Scan 2: Continuation — same bias, phase progresses
        state2 = dict(sample_state)
        state2["thesis"] = "Manipulation sweep complete. Bullish displacement confirmed."
        state2["p3_phase"] = "manipulation"  # same phase
        state2["p3_progress"] = "late"
        state2["bias_confidence"] = 0.85
        ns_store.save("1h", state2)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 2
        assert current["bias_confidence"] == 0.85

        # Scan 3: Revision — invalidation triggered, bias flips
        state3 = dict(sample_state)
        state3["thesis"] = "Previous bullish thesis invalidated. Bearish continuation."
        state3["p3_phase"] = "distribution"
        state3["directional_bias"] = "bearish"
        state3["bias_confidence"] = 0.60
        state3["last_revision"] = "Invalidation triggered: price closed below 2335."
        ns_store.save("1h", state3)

        current = ns_store.get_current("1h")
        assert current["scan_count"] == 1
        assert current["is_revision"] == 1
        assert current["directional_bias"] == "bearish"

        # Check history
        history = ns_store.get_history("1h")
        assert len(history) == 3

    def test_full_cycle_with_invalidation_check(self, ns_store, sample_state, sample_candles):
        """Full cycle: save → check invalidation → inject into prompt."""
        from ml.prompts import _build_narrative_state_section

        ns_store.save("1h", sample_state)
        current = ns_store.get_current("1h")

        # Price is safe
        inv_status = check_invalidation(current, 2340.0, sample_candles)
        assert inv_status == "CLEAR"

        # Build prompt section
        section = _build_narrative_state_section(current, inv_status)
        assert "YOUR PREVIOUS THESIS" in section
        assert "INVALIDATION TRIGGERED" not in section

        # Now price breaches invalidation
        bad_candles = [{"close": 2334.0}] * 3
        inv_status = check_invalidation(current, 2334.0, bad_candles)
        assert inv_status == "TRIGGERED"

        section = _build_narrative_state_section(current, inv_status)
        assert "INVALIDATION TRIGGERED" in section
        assert "MUST re-assess" in section
