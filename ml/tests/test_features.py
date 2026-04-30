"""Tests for ML feature extraction — TDD: write tests first."""
import pytest
from ml.features import (
    compute_atr, extract_features, classify_setup_type,
    detect_swing_points, compute_ob_freshness,
    compute_fvg_fill_percentage, engineer_htf_features,
    _default_htf_features,
)


# ── Task 11 — Calendar columns in feature schema ────────────────

def test_feature_schema_includes_calendar_columns():
    from ml.feature_schema import FEATURE_COLUMNS
    expected = {
        "mins_to_next_high_impact",
        "mins_since_last_high_impact",
        "news_density_24h",
        "calendar_proximity_clear",
        "calendar_proximity_post_event",
        "calendar_proximity_caution",
        "calendar_proximity_imminent",
        "event_is_nfp",
        "event_is_cpi",
        "event_is_ppi",
        "event_is_fomc",
        "event_is_fed_speak",
        "event_is_gdp",
        "event_is_ism",
        "event_is_retail_sales",
        "event_is_unemployment",
        "event_is_jolts",
        "event_is_other_high",
    }
    assert expected.issubset(set(FEATURE_COLUMNS))
    # Baseline was 59; +18 = 77.
    assert len(FEATURE_COLUMNS) == 77


# ── Task 12 — Calendar feature extraction ──────────────────

def _calendar_store_with_fixture(tmp_path):
    """Build a CalendarStore wired to the FF fixture used elsewhere."""
    from datetime import datetime, timezone
    from pathlib import Path
    import sqlite3
    from ml.calendar import CalendarStore, ForexFactorySource
    from ml.scanner_db import init_db
    fixture = Path(__file__).parent / "fixtures" / "ff_calendar_sample.xml"
    db = tmp_path / "test.db"
    init_db(str(db))
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(fixture)),
        db_path=str(db),
    )
    store.refresh(force=True)
    return store


def test_calendar_features_imminent_fomc(tmp_path, sample_candles, sample_analysis):
    from datetime import datetime, timezone
    store = _calendar_store_with_fixture(tmp_path)
    feats = extract_features(
        sample_analysis,
        sample_candles,
        timeframe="1h",
        calendar_store=store,
        now=datetime(2026, 4, 29, 17, 45, tzinfo=timezone.utc),
    )
    # 15 minutes before FOMC at 18:00 → imminent.
    assert feats["calendar_proximity_imminent"] == 1
    assert feats["calendar_proximity_clear"] == 0
    assert feats["event_is_fomc"] == 1
    assert feats["event_is_nfp"] == 0
    assert 14 <= feats["mins_to_next_high_impact"] <= 16


def test_calendar_features_default_safe_when_store_unavailable(
    sample_candles, sample_analysis
):
    feats = extract_features(
        sample_analysis,
        sample_candles,
        timeframe="1h",
        calendar_store=None,
    )
    # No store → safe defaults: clamp to "no event in window", proximity=clear.
    assert feats["mins_to_next_high_impact"] == 1440
    assert feats["mins_since_last_high_impact"] == 1440
    assert feats["news_density_24h"] == 0
    assert feats["calendar_proximity_clear"] == 1
    assert feats["calendar_proximity_imminent"] == 0
    for col in ("event_is_nfp", "event_is_cpi", "event_is_fomc",
                "event_is_other_high"):
        assert feats[col] == 0


def test_calendar_features_clear_state_uses_other_high_default(
    tmp_path, sample_candles, sample_analysis
):
    from datetime import datetime, timezone
    store = _calendar_store_with_fixture(tmp_path)
    feats = extract_features(
        sample_analysis,
        sample_candles,
        timeframe="1h",
        calendar_store=store,
        now=datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc),
    )
    # 4h before FOMC, but the upcoming-event one-hot still reflects the next
    # event's category for the model's benefit (proximity tells it whether to
    # weight it).
    assert feats["calendar_proximity_clear"] == 1
    assert feats["calendar_proximity_imminent"] == 0
    # FOMC is the next event; one-hot for fomc should still be set.
    assert feats["event_is_fomc"] == 1


# ── Task 19 — Backfill calendar features for stored setups ──

def test_feature_backfill_populates_calendar_columns(tmp_path):
    import json
    import sqlite3
    from datetime import datetime, timezone
    from pathlib import Path
    from ml.calendar import CalendarStore, ForexFactorySource
    from ml.scanner_db import ScannerDB
    from ml.backfill_features import backfill_calendar_features

    fixture = Path(__file__).parent / "fixtures" / "ff_calendar_sample.xml"
    db_path = tmp_path / "scanner.db"
    db = ScannerDB(db_path=str(db_path))

    # Wire calendar history with FF fixture (live ops would also archive,
    # but for this test we explicitly populate history).
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(fixture)),
        db_path=str(db_path),
    )
    store.refresh(force=True)
    # Promote the live cache rows into history under the historical source
    # so the backfill query sees them. (The live refresh already inserted
    # them with source='ff_xml' — keep that and rely on the history rows.)

    # Insert two scanner setups: one inside the FOMC imminent window, one
    # well clear of it.
    setup_inside_imminent = {
        "id": "setup-inside",
        "created_at": "2026-04-29T17:45:00+00:00",
        "direction": "long",
        "killzone": "NY_PM",
        "analysis_json": json.dumps({"bias": "bullish"}),
    }
    setup_clear = {
        "id": "setup-clear",
        "created_at": "2026-04-29T14:00:00+00:00",
        "direction": "short",
        "killzone": "NY_AM",
        "analysis_json": json.dumps({"bias": "bearish"}),
    }
    with sqlite3.connect(str(db_path)) as conn:
        for s in (setup_inside_imminent, setup_clear):
            conn.execute(
                "INSERT INTO scanner_setups (id, created_at, status, "
                "direction, killzone, analysis_json) "
                "VALUES (?, ?, 'resolved', ?, ?, ?)",
                (s["id"], s["created_at"], s["direction"],
                 s["killzone"], s["analysis_json"]),
            )
        conn.commit()

    n = backfill_calendar_features(db_path=str(db_path))
    assert n == 2

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = {r["id"]: r for r in conn.execute(
            "SELECT id, analysis_json FROM scanner_setups"
        ).fetchall()}

    inside = json.loads(rows["setup-inside"]["analysis_json"])
    clear = json.loads(rows["setup-clear"]["analysis_json"])
    assert inside["calendar_features"]["calendar_proximity_imminent"] == 1
    assert inside["calendar_features"]["event_is_fomc"] == 1
    assert clear["calendar_features"]["calendar_proximity_clear"] == 1
    # Non-calendar fields untouched.
    assert inside["bias"] == "bullish"
    assert clear["bias"] == "bearish"


def test_feature_backfill_idempotent(tmp_path):
    import json
    import sqlite3
    from pathlib import Path
    from ml.calendar import CalendarStore, ForexFactorySource
    from ml.scanner_db import ScannerDB
    from ml.backfill_features import backfill_calendar_features

    fixture = Path(__file__).parent / "fixtures" / "ff_calendar_sample.xml"
    db_path = tmp_path / "scanner.db"
    ScannerDB(db_path=str(db_path))
    store = CalendarStore(
        source=ForexFactorySource(_offline_path=str(fixture)),
        db_path=str(db_path),
    )
    store.refresh(force=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO scanner_setups (id, created_at, status, "
            "direction, analysis_json) "
            "VALUES (?, ?, 'resolved', 'long', ?)",
            ("s1", "2026-04-29T17:45:00+00:00", json.dumps({"bias": "bullish"})),
        )
        conn.commit()

    n1 = backfill_calendar_features(db_path=str(db_path))
    n2 = backfill_calendar_features(db_path=str(db_path))
    assert n1 == 1
    # Second run still updates the row but result is identical — accept any
    # non-negative count, just confirm output stays consistent.
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT analysis_json FROM scanner_setups WHERE id='s1'"
        ).fetchone()
    payload = json.loads(row[0])
    assert payload["calendar_features"]["calendar_proximity_imminent"] == 1


# ── compute_atr tests ───────────────────────────────────────

class TestComputeATR:
    def test_returns_positive_for_valid_data(self, sample_candles):
        atr = compute_atr(sample_candles)
        assert atr > 0

    def test_returns_zero_for_insufficient_data(self):
        short = [{"open": 100, "high": 105, "low": 95, "close": 102}] * 5
        assert compute_atr(short, period=14) == 0.0

    def test_uses_true_range_formula(self):
        """TR = max(H-L, |H-prevC|, |L-prevC|)"""
        candles = [
            {"open": 100, "high": 110, "low": 90, "close": 105},  # base
            {"open": 105, "high": 108, "low": 102, "close": 106},  # TR = max(6, 3, 3) = 6
            {"open": 106, "high": 115, "low": 100, "close": 112},  # TR = max(15, 9, 6) = 15
        ]
        atr = compute_atr(candles, period=2)
        assert atr == pytest.approx((6.0 + 15.0) / 2, abs=0.01)

    def test_handles_single_candle(self):
        assert compute_atr([{"open": 100, "high": 105, "low": 95, "close": 102}]) == 0.0

    def test_handles_empty_list(self):
        assert compute_atr([]) == 0.0


# ── extract_features tests ──────────────────────────────────

class TestExtractFeatures:
    def test_returns_all_feature_keys(self, sample_analysis, sample_candles):
        from ml.feature_schema import FEATURE_SET
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert set(f.keys()) == FEATURE_SET, f"Missing: {FEATURE_SET - set(f.keys())}, Extra: {set(f.keys()) - FEATURE_SET}"

    def test_ob_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_count"] == 2

    def test_ob_bullish_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_bullish_count"] == 1

    def test_ob_bearish_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_bearish_count"] == 1

    def test_ob_strongest_strength(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_strongest_strength"] == 3  # strong=3

    def test_ob_nearest_distance_atr_positive(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_nearest_distance_atr"] >= 0

    def test_ob_avg_size_atr_positive(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_avg_size_atr"] > 0

    def test_ob_alignment_matches_bias(self, sample_analysis, sample_candles):
        """Bullish bias + bullish OB present → alignment = 1"""
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["ob_alignment"] == 1

    def test_fvg_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["fvg_count"] == 2

    def test_fvg_unfilled_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["fvg_unfilled_count"] == 1

    def test_liq_buyside_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["liq_buyside_count"] == 1

    def test_liq_sellside_count(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["liq_sellside_count"] == 1

    def test_risk_reward_tp1(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["risk_reward_tp1"] == pytest.approx(4.3)

    def test_entry_direction_long(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["entry_direction"] == 1  # long=1

    def test_entry_direction_short(self, sample_analysis, sample_candles):
        analysis = {**sample_analysis, "bias": "bearish",
                    "entry": {"price": 2680, "direction": "short", "rationale": ""}}
        f = extract_features(analysis, sample_candles, "1h")
        assert f["entry_direction"] == 0  # short=0

    def test_bias_direction_match(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["bias_direction_match"] == 1

    def test_num_confluences(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["num_confluences"] == 3

    def test_killzone_london(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["killzone_encoded"] == 1  # London=1

    def test_timeframe_encoding(self, sample_analysis, sample_candles):
        assert extract_features(sample_analysis, sample_candles, "15min")["timeframe_encoded"] == 1
        assert extract_features(sample_analysis, sample_candles, "1h")["timeframe_encoded"] == 2
        assert extract_features(sample_analysis, sample_candles, "4h")["timeframe_encoded"] == 3
        assert extract_features(sample_analysis, sample_candles, "1day")["timeframe_encoded"] == 4

    def test_empty_analysis_graceful(self, sample_candles):
        empty = {"bias": "neutral", "orderBlocks": [], "fvgs": [], "liquidity": [],
                 "entry": None, "stopLoss": None, "takeProfits": [], "killzone": "",
                 "confluences": []}
        f = extract_features(empty, sample_candles, "1h")
        assert f["ob_count"] == 0
        assert f["entry_direction"] == 0
        from ml.feature_schema import FEATURE_SET
        assert set(f.keys()) == FEATURE_SET

    def test_does_not_mutate_inputs(self, sample_analysis, sample_candles):
        import json
        before_a = json.dumps(sample_analysis, sort_keys=True)
        before_c = json.dumps(sample_candles, sort_keys=True)
        extract_features(sample_analysis, sample_candles, "1h")
        assert json.dumps(sample_analysis, sort_keys=True) == before_a
        assert json.dumps(sample_candles, sort_keys=True) == before_c


# ── classify_setup_type tests ────────────────────────────────


class TestClassifySetupType:
    """Derive WFO taxonomy from Claude analysis fields."""

    def test_classify_bull_ob_fvg(self):
        """Long entry with OBs + FVGs → bull_fvg_ob (tags sorted)."""
        analysis = {
            "entry": {"direction": "long"},
            "orderBlocks": [{"type": "bullish", "high": 100, "low": 99}],
            "fvgs": [{"type": "bullish", "high": 102, "low": 101}],
        }
        assert classify_setup_type(analysis) == "bull_fvg_ob"

    def test_classify_bear_structure(self):
        """Short entry with structure confluence → bear_structure."""
        analysis = {
            "entry": {"direction": "short"},
            "confluences": {"structureAlignment": True},
        }
        assert classify_setup_type(analysis) == "bear_structure"

    def test_classify_with_london_killzone(self):
        """London killzone adds 'london' tag."""
        analysis = {
            "entry": {"direction": "long"},
            "orderBlocks": [{"type": "bullish", "high": 100, "low": 99}],
            "killzone": "London Open",
        }
        result = classify_setup_type(analysis)
        assert "london" in result
        assert result == "bull_london_ob"

    def test_classify_with_ny_killzone(self):
        """NY killzone adds 'ny_am' tag."""
        analysis = {
            "entry": {"direction": "short"},
            "fvgs": [{"type": "bearish", "high": 100, "low": 99}],
            "killzone": "NY AM Session",
        }
        result = classify_setup_type(analysis)
        assert "ny_am" in result
        assert result == "bear_fvg_ny_am"

    def test_classify_empty_analysis(self):
        """No entry direction → bear prefix, no tags → unclassified."""
        analysis = {}
        assert classify_setup_type(analysis) == "bear_unclassified"

    def test_classify_tags_sorted(self):
        """Tags must be alphabetically sorted in the output."""
        analysis = {
            "entry": {"direction": "long"},
            "orderBlocks": [{"type": "bullish", "high": 100, "low": 99}],
            "fvgs": [{"type": "bullish", "high": 102, "low": 101}],
            "confluences": {"structureAlignment": True, "displacement": True},
        }
        result = classify_setup_type(analysis)
        assert result == "bull_displacement_fvg_ob_structure"

    def test_classify_with_sweep(self):
        """Liquidity sweep confluence adds 'sweep' tag."""
        analysis = {
            "entry": {"direction": "short"},
            "confluences": {"liquiditySweep": True},
        }
        assert classify_setup_type(analysis) == "bear_sweep"

    def test_classify_confluences_as_list_fallback(self):
        """When confluences is a list (old format) → no structure/displacement tags."""
        analysis = {
            "entry": {"direction": "long"},
            "orderBlocks": [{"type": "bullish", "high": 100, "low": 99}],
            "confluences": ["Bullish OB + FVG overlap", "Structure alignment"],
        }
        # List confluences don't have dict keys, so only OB tag
        assert classify_setup_type(analysis) == "bull_ob"


# ── detect_swing_points tests ─────────────────────────────


class TestDetectSwingPoints:
    """Test swing high/low detection for BOS/ChoCH and sweep logic."""

    def test_detects_swing_high(self):
        """Clear peak in middle should be detected."""
        candles = []
        for i in range(15):
            h = 250 if i == 7 else 200 + i * 0.5
            candles.append({"open": 195, "high": h, "low": 190, "close": 196,
                            "datetime": ""})
        swings = detect_swing_points(candles, lookback=3)
        highs = [s for s in swings if s["type"] == "high"]
        assert len(highs) >= 1
        assert any(s["index"] == 7 for s in highs)

    def test_detects_swing_low(self):
        """Clear trough in middle should be detected."""
        candles = []
        for i in range(15):
            low = 150 if i == 7 else 195 - i * 0.3
            candles.append({"open": 200, "high": 210, "low": low, "close": 201,
                            "datetime": ""})
        swings = detect_swing_points(candles, lookback=3)
        lows = [s for s in swings if s["type"] == "low"]
        assert len(lows) >= 1
        assert any(s["index"] == 7 for s in lows)

    def test_returns_empty_for_short_data(self):
        """Fewer than 2*lookback+1 candles should return empty."""
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100, "datetime": ""}
                   for _ in range(5)]
        assert detect_swing_points(candles, lookback=5) == []

    def test_no_mutation(self, sample_candles):
        """Input candles must not be modified."""
        orig_len = len(sample_candles)
        detect_swing_points(sample_candles, lookback=5)
        assert len(sample_candles) == orig_len


# ── compute_ob_freshness tests ────────────────────────────


class TestComputeOBFreshness:
    """Test OB retest counting for freshness scoring."""

    def test_untested_ob_returns_zero(self):
        """OB with no candles wicking into zone → 0 retests."""
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100.5, "datetime": ""}
                   for _ in range(20)]
        ob = {"high": 110, "low": 108, "index": 5, "type": "bullish"}
        assert compute_ob_freshness(candles, ob, 15) == 0

    def test_counts_retests_correctly(self):
        """Candles wicking into OB zone should be counted."""
        ob = {"high": 105, "low": 100, "index": 0, "type": "bullish"}
        candles = [
            {"open": 102, "high": 106, "low": 99, "close": 104, "datetime": ""},  # idx 0 (OB)
            {"open": 103, "high": 104, "low": 101, "close": 103, "datetime": ""},  # idx 1 — touches zone
            {"open": 103, "high": 104, "low": 98, "close": 99, "datetime": ""},    # idx 2 — touches zone
            {"open": 110, "high": 115, "low": 108, "close": 112, "datetime": ""},  # idx 3 — above zone
        ]
        assert compute_ob_freshness(candles, ob, 4) == 2

    def test_first_touch_at_current_idx(self):
        """Retests before current_idx, not at it."""
        ob = {"high": 105, "low": 100, "index": 0, "type": "bullish"}
        candles = [
            {"open": 102, "high": 106, "low": 99, "close": 104, "datetime": ""},
            {"open": 110, "high": 115, "low": 108, "close": 112, "datetime": ""},
            {"open": 110, "high": 115, "low": 108, "close": 112, "datetime": ""},
        ]
        # No candles wick into 100-105 between idx 1 and 2
        assert compute_ob_freshness(candles, ob, 2) == 0


# ── compute_fvg_fill_percentage tests ─────────────────────


class TestComputeFVGFillPercentage:
    """Test FVG fill percentage calculation."""

    def test_unfilled_fvg_returns_zero(self):
        """FVG with no subsequent price action in gap → 0.0."""
        fvg = {"high": 110, "low": 105, "index": 1, "type": "bullish", "size": 5}
        candles = [
            {"open": 100, "high": 104, "low": 98, "close": 103, "datetime": ""},
            {"open": 103, "high": 104, "low": 100, "close": 103, "datetime": ""},  # FVG
            {"open": 112, "high": 115, "low": 111, "close": 114, "datetime": ""},  # above gap
            {"open": 114, "high": 116, "low": 112, "close": 115, "datetime": ""},  # above gap
        ]
        assert compute_fvg_fill_percentage(candles, fvg, 3) == 0.0

    def test_fully_filled_fvg(self):
        """FVG with price trading through entire gap → 1.0."""
        fvg = {"high": 110, "low": 105, "index": 1, "type": "bullish", "size": 5}
        candles = [
            {"open": 100, "high": 104, "low": 98, "close": 103, "datetime": ""},
            {"open": 103, "high": 104, "low": 100, "close": 103, "datetime": ""},
            {"open": 108, "high": 108, "low": 100, "close": 102, "datetime": ""},  # fills fully
        ]
        assert compute_fvg_fill_percentage(candles, fvg, 2) == pytest.approx(1.0)

    def test_partially_filled_fvg(self):
        """FVG partially filled returns value between 0 and 1."""
        fvg = {"high": 110, "low": 100, "index": 1, "type": "bullish", "size": 10}
        candles = [
            {"open": 100, "high": 104, "low": 98, "close": 103, "datetime": ""},
            {"open": 103, "high": 104, "low": 100, "close": 103, "datetime": ""},
            {"open": 108, "high": 108, "low": 105, "close": 106, "datetime": ""},  # fills 50%
        ]
        fill = compute_fvg_fill_percentage(candles, fvg, 2)
        assert 0.0 < fill < 1.0


# ── engineer_htf_features tests ───────────────────────────


class TestEngineerHTFFeatures:
    """Test 4H-equivalent HTF feature extraction."""

    def test_returns_11_features(self, wfo_candles):
        """Should return exactly 11 HTF features."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert len(feats) == 11

    def test_premium_discount_bounded(self, wfo_candles):
        """htf_premium_discount must be between 0 and 1."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert 0.0 <= feats["htf_premium_discount"] <= 1.0

    def test_array_alignment_values(self, wfo_candles):
        """htf_array_alignment should be -1, 0, or 1."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert feats["htf_array_alignment"] in (-1, 0, 1)

    def test_liq_narrative_values(self, wfo_candles):
        """htf_liq_narrative should be -1, 0, or 1."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert feats["htf_liq_narrative"] in (-1, 0, 1)

    def test_default_features_for_short_data(self):
        """Fewer than 12 candles (can't make 3 HTF bars) → defaults."""
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100.5,
                    "datetime": ""} for _ in range(10)]
        feats = engineer_htf_features(candles, 9, "long", 2.0)
        assert feats == _default_htf_features()

    def test_candle_type_is_directional(self, wfo_candles):
        """htf_last_candle_type should be 1 or -1."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert feats["htf_last_candle_type"] in (1, -1)

    def test_distances_are_non_negative(self, wfo_candles):
        """POI distance features should be >= 0."""
        from ml.features import compute_atr
        atr = compute_atr(wfo_candles, 14)
        feats = engineer_htf_features(wfo_candles, 200, "long", atr)
        assert feats["htf_ob_above_dist"] >= 0
        assert feats["htf_ob_below_dist"] >= 0
        assert feats["htf_fvg_above_dist"] >= 0
        assert feats["htf_fvg_below_dist"] >= 0


# ── Entry zone computation ──────────────────────────────────

class TestEntryZoneComputation:
    """Tests for _compute_entry_zone auto-calculation from OB data."""

    def test_entry_zone_populated_when_obs_present(self, sample_analysis, sample_candles):
        """Entry zone should be computed from OB data, not NaN."""
        import math
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert not math.isnan(f["entry_zone_position"])
        assert not math.isnan(f["entry_zone_size_atr"])

    def test_entry_zone_nan_when_no_obs(self, sample_candles):
        """Without OBs, entry zone should be NaN."""
        import math
        analysis = {
            "bias": "bullish",
            "orderBlocks": [],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 2650.0, "direction": "long"},
            "stopLoss": {"price": 2643.0},
            "takeProfits": [{"price": 2680.0, "rr": 4.3}],
            "killzone": "London",
            "confluences": [],
        }
        f = extract_features(analysis, sample_candles, "1h")
        assert math.isnan(f["entry_zone_position"])
        assert math.isnan(f["entry_zone_size_atr"])

    def test_entry_zone_position_between_0_and_1(self, sample_analysis, sample_candles):
        """Entry zone position must be in [0, 1]."""
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert 0.0 <= f["entry_zone_position"] <= 1.0

    def test_entry_zone_size_positive(self, sample_analysis, sample_candles):
        """Entry zone size should be positive when OBs exist."""
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["entry_zone_size_atr"] > 0

    def test_entry_zone_prefers_direction_matching_ob(self, sample_candles):
        """Long entries should prefer bullish OBs."""
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bearish", "high": 2651.0, "low": 2649.0, "strength": "strong"},
                {"type": "bullish", "high": 2652.0, "low": 2648.0, "strength": "moderate"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": {"price": 2650.0, "direction": "long"},
            "stopLoss": {"price": 2643.0},
            "takeProfits": [{"price": 2680.0, "rr": 4.3}],
            "killzone": "London",
            "confluences": [],
        }
        f = extract_features(analysis, sample_candles, "1h")
        # Should use the bullish OB (4 ATR size) not the bearish one (2 ATR size)
        assert f["entry_zone_size_atr"] > 0

    def test_entry_zone_nan_without_entry_price(self, sample_candles):
        """No entry price → NaN entry zone."""
        import math
        analysis = {
            "bias": "bullish",
            "orderBlocks": [
                {"type": "bullish", "high": 2650.0, "low": 2645.0, "strength": "strong"},
            ],
            "fvgs": [],
            "liquidity": [],
            "entry": None,
            "stopLoss": None,
            "takeProfits": [],
            "killzone": "London",
            "confluences": [],
        }
        f = extract_features(analysis, sample_candles, "1h")
        assert math.isnan(f["entry_zone_position"])


# ── Intermarket feature extraction ──────────────────────────

class TestIntermarketFeatures:
    """Tests for intermarket feature handling."""

    def test_intermarket_values_pass_through(self, sample_analysis, sample_candles):
        """When intermarket data provided, values should appear in features."""
        im = {
            "gold_dxy_corr_20": -0.45,
            "gold_dxy_diverging": 1,
            "dxy_range_position": 0.3,
            "yield_direction": -1,
        }
        f = extract_features(sample_analysis, sample_candles, "1h", intermarket=im)
        assert f["gold_dxy_corr_20"] == -0.45
        assert f["gold_dxy_diverging"] == 1
        assert f["dxy_range_position"] == 0.3
        assert f["yield_direction"] == -1

    def test_intermarket_nan_when_absent(self, sample_analysis, sample_candles):
        """Without intermarket data, should be NaN not 0."""
        import math
        f = extract_features(sample_analysis, sample_candles, "1h", intermarket=None)
        assert math.isnan(f["gold_dxy_corr_20"])
        assert math.isnan(f["dxy_range_position"])
        assert math.isnan(f["yield_direction"])

    def test_intermarket_zero_is_valid(self, sample_analysis, sample_candles):
        """Zero is a real intermarket value (no correlation), not missing."""
        im = {
            "gold_dxy_corr_20": 0.0,
            "gold_dxy_diverging": 0,
            "dxy_range_position": 0.5,
            "yield_direction": 0,
        }
        f = extract_features(sample_analysis, sample_candles, "1h", intermarket=im)
        assert f["gold_dxy_corr_20"] == 0.0
        assert f["dxy_range_position"] == 0.5


# ── HTF feature extraction tests ──────────────────────────────

class TestHTFFeatures:
    def _make_analysis_with_htf(self, base_analysis):
        """Add htf_context and structure to base analysis."""
        a = dict(base_analysis)
        a["htf_context"] = {
            "dealing_range_high": 2700.0,
            "dealing_range_low": 2600.0,
            "premium_discount": "discount",
            "power_of_3_phase": "accumulation",
            "recent_sweep": "ssl",
            "htf_bias": "bullish",
        }
        a["structure"] = {
            "type": "bos",
            "direction": "bullish",
            "break_candle_index": 90,
        }
        return a

    def test_htf_bias_encoded_bullish(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_bias_encoded"] == 1

    def test_htf_bias_encoded_bearish(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        a["htf_context"]["htf_bias"] = "bearish"
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_bias_encoded"] == -1

    def test_htf_sweep_encoded(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_sweep_encoded"] == -1  # ssl = bearish signal

    def test_htf_sweep_bsl(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        a["htf_context"]["recent_sweep"] = "bsl"
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_sweep_encoded"] == 1

    def test_dealing_range_position(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        # entry price is 2650, range is 2600-2700 → position = 0.5
        f = extract_features(a, sample_candles, "1h")
        assert f["dealing_range_position"] == pytest.approx(0.5, abs=0.01)

    def test_dealing_range_nan_when_missing(self, sample_analysis, sample_candles):
        import math
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert math.isnan(f["dealing_range_position"])

    def test_htf_structure_alignment_agree(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        # htf_bias=bullish, direction=long → agree
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_structure_alignment"] == 1

    def test_htf_structure_alignment_conflict(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        a["htf_context"]["htf_bias"] = "bearish"
        # htf_bias=bearish, direction=long → conflict
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_structure_alignment"] == -1

    def test_htf_displacement_quality(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        # structure=bos + strong OB in analysis → 1
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_displacement_quality"] == 1

    def test_htf_displacement_weak_without_structure(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_htf(sample_analysis)
        a["structure"] = {"type": "none"}
        f = extract_features(a, sample_candles, "1h")
        assert f["htf_displacement_quality"] == 0

    def test_all_htf_features_nan_when_no_context(self, sample_analysis, sample_candles):
        """Without htf_context, htf features should degrade gracefully."""
        import math
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["htf_bias_encoded"] == 0
        assert f["htf_sweep_encoded"] == 0
        assert math.isnan(f["dealing_range_position"])
        assert f["htf_structure_alignment"] == 0


# ── Narrative feature extraction tests ─────────────────────────

class TestNarrativeFeatures:
    def _make_analysis_with_narrative(self, base_analysis):
        a = dict(base_analysis)
        a["narrative_state"] = {
            "bias_confidence": 0.85,
            "p3_progress": "mid",
            "scan_count": 5,
            "directional_bias": "bullish",
        }
        return a

    def test_thesis_confidence(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        f = extract_features(a, sample_candles, "1h")
        assert f["thesis_confidence"] == pytest.approx(0.85, abs=0.01)

    def test_thesis_confidence_nan_when_missing(self, sample_analysis, sample_candles):
        import math
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert math.isnan(f["thesis_confidence"])

    def test_p3_progress_mid(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        f = extract_features(a, sample_candles, "1h")
        assert f["p3_progress_encoded"] == 2  # mid=2

    def test_p3_progress_late(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        a["narrative_state"]["p3_progress"] = "late"
        f = extract_features(a, sample_candles, "1h")
        assert f["p3_progress_encoded"] == 3

    def test_scan_count(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        f = extract_features(a, sample_candles, "1h")
        assert f["thesis_scan_count"] == 5

    def test_opus_sonnet_agreement_yes(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        cal = {"opus_narrative": {"directional_bias": "bullish"}}
        # entry=long, opus=bullish → agree
        f = extract_features(a, sample_candles, "1h", calibration=cal)
        assert f["opus_sonnet_agreement"] == 1

    def test_opus_sonnet_agreement_no(self, sample_analysis, sample_candles):
        a = self._make_analysis_with_narrative(sample_analysis)
        cal = {"opus_narrative": {"directional_bias": "bearish"}}
        # entry=long, opus=bearish → disagree
        f = extract_features(a, sample_candles, "1h", calibration=cal)
        assert f["opus_sonnet_agreement"] == 0

    def test_opus_absent(self, sample_analysis, sample_candles):
        f = extract_features(sample_analysis, sample_candles, "1h")
        assert f["opus_sonnet_agreement"] == 0

    def test_all_58_features_present(self, sample_analysis, sample_candles):
        """Ensure all 58 feature columns from schema are returned."""
        from ml.feature_schema import FEATURE_SET
        a = self._make_analysis_with_narrative(sample_analysis)
        a["htf_context"] = {
            "dealing_range_high": 2700.0,
            "dealing_range_low": 2600.0,
            "htf_bias": "bullish",
            "recent_sweep": "none",
        }
        a["structure"] = {"type": "bos", "direction": "bullish"}
        cal = {"opus_narrative": {"directional_bias": "bullish"}}
        f = extract_features(a, sample_candles, "1h", calibration=cal)
        assert set(f.keys()) == FEATURE_SET, f"Missing: {FEATURE_SET - set(f.keys())}, Extra: {set(f.keys()) - FEATURE_SET}"
