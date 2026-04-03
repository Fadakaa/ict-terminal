"""Tests for setup DNA encoder, similarity matching, and profile store."""
import pytest
from ml.setup_dna import encode_setup_dna, compute_similarity
from ml.setup_profiles import SetupProfileStore


# ═══════════════════════════════════════════════════════════════
# DNA Encoder
# ═══════════════════════════════════════════════════════════════

class TestEncodeDNA:
    """Step 1: encode_setup_dna extracts 21 features from analysis JSON."""

    EXPECTED_KEYS = {
        "killzone", "timeframe", "has_ob", "ob_strength", "ob_times_tested",
        "has_fvg", "fvg_overlaps_ob", "fvg_fill_pct", "has_sweep", "sweep_type",
        "structure_type", "structure_direction", "premium_discount", "p3_phase",
        "confluence_count", "opus_validated", "opus_sonnet_agree", "direction",
        "entry_type", "rr_ratio_tp1", "volatility_regime",
    }

    def test_returns_dict_with_expected_keys(self, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert set(dna.keys()) == self.EXPECTED_KEYS

    def test_handles_missing_structure(self, sample_analysis):
        """sample_analysis fixture lacks 'structure' key."""
        assert "structure" not in sample_analysis
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["structure_type"] == "none"
        assert dna["structure_direction"] == "unknown"

    def test_handles_missing_htf_context(self, sample_analysis):
        """sample_analysis fixture lacks 'htf_context' key."""
        assert "htf_context" not in sample_analysis
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["premium_discount"] == "unknown"
        assert dna["p3_phase"] == "unknown"

    def test_extracts_ob_features(self, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["has_ob"] is True
        assert dna["ob_strength"] == "strong"

    def test_extracts_fvg_features(self, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["has_fvg"] is True

    def test_extracts_direction(self, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["direction"] == "long"

    def test_rr_bucket_high(self, sample_analysis):
        """sample_analysis has rr=4.3 for TP1 → 'high' bucket."""
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert dna["rr_ratio_tp1"] == "high"

    def test_rr_bucket_mid(self):
        analysis = {"takeProfits": [{"price": 100, "rr": 2.5}]}
        dna = encode_setup_dna(analysis, {}, "1h", "Off")
        assert dna["rr_ratio_tp1"] == "mid"

    def test_rr_bucket_low(self):
        analysis = {"takeProfits": [{"price": 100, "rr": 1.2}]}
        dna = encode_setup_dna(analysis, {}, "1h", "Off")
        assert dna["rr_ratio_tp1"] == "low"

    def test_empty_analysis(self):
        dna = encode_setup_dna({}, {}, "1h", "Off")
        assert dna["has_ob"] is False
        assert dna["has_fvg"] is False
        assert dna["has_sweep"] is False
        assert dna["ob_strength"] == "none"
        assert dna["direction"] == "unknown"
        assert dna["confluence_count"] == 0

    def test_volatility_regime_from_calibration(self):
        cal = {"volatility_context": {"regime": "trending"}}
        dna = encode_setup_dna({}, cal, "1h", "Off")
        assert dna["volatility_regime"] == "trending"

    def test_opus_sonnet_agree(self):
        analysis = {"entry": {"direction": "long"}, "opus_validated": True}
        cal = {"opus_narrative": {"directional_bias": "bullish"}}
        dna = encode_setup_dna(analysis, cal, "1h", "London")
        assert dna["opus_sonnet_agree"] is True

    def test_opus_sonnet_disagree(self):
        analysis = {"entry": {"direction": "long"}}
        cal = {"opus_narrative": {"directional_bias": "bearish"}}
        dna = encode_setup_dna(analysis, cal, "1h", "London")
        assert dna["opus_sonnet_agree"] is False

    def test_sweep_extraction(self):
        analysis = {
            "liquidity": [
                {"type": "sellside", "price": 2630, "swept": True},
                {"type": "buyside", "price": 2690, "swept": False},
            ]
        }
        dna = encode_setup_dna(analysis, {}, "1h", "London")
        assert dna["has_sweep"] is True
        assert dna["sweep_type"] == "ssl"

    def test_full_analysis_with_structure(self, sample_analysis):
        """Add structure and htf_context to sample_analysis."""
        analysis = {
            **sample_analysis,
            "structure": {"type": "bos", "direction": "bullish"},
            "htf_context": {
                "premium_discount": "discount",
                "power_of_3_phase": "accumulation",
            },
        }
        dna = encode_setup_dna(analysis, {}, "1h", "London")
        assert dna["structure_type"] == "bos"
        assert dna["structure_direction"] == "bullish"
        assert dna["premium_discount"] == "discount"
        assert dna["p3_phase"] == "accumulation"


# ═══════════════════════════════════════════════════════════════
# Similarity
# ═══════════════════════════════════════════════════════════════

class TestComputeSimilarity:
    """Step 1: compute_similarity returns weighted match score."""

    def test_identical_dna_returns_1(self, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        assert compute_similarity(dna, dna) == pytest.approx(1.0)

    def test_completely_different_returns_low(self, sample_analysis):
        dna_a = encode_setup_dna(sample_analysis, {}, "1h", "London")
        dna_b = encode_setup_dna({}, {}, "4h", "Asian")
        # Some default values match (e.g. fvg_fill_pct="none", entry_type="unknown")
        # but core ICT features diverge — score should be well below 0.5
        assert compute_similarity(dna_a, dna_b) < 0.5

    def test_partial_match_different_killzone(self):
        """Same analysis, different killzone — score drops by exactly 0.15."""
        analysis = {
            "orderBlocks": [{"strength": "strong"}],
            "fvgs": [{"filled": False, "overlaps_ob": True}],
            "liquidity": [{"type": "sellside", "swept": True}],
            "structure": {"type": "bos", "direction": "bullish"},
            "entry": {"direction": "long", "entry_type": "retracement"},
            "takeProfits": [{"rr": 3.0}],
            "confluences": ["a", "b"],
            "killzone": "London",
        }
        dna_a = encode_setup_dna(analysis, {}, "1h", "London")
        dna_b = encode_setup_dna({**analysis, "killzone": "NY_AM"}, {}, "1h", "NY_AM")
        sim = compute_similarity(dna_a, dna_b)
        # Everything matches except killzone (-0.15)
        assert 0.8 <= sim <= 0.9

    def test_symmetry(self, sample_analysis):
        dna_a = encode_setup_dna(sample_analysis, {}, "1h", "London")
        dna_b = encode_setup_dna({}, {}, "4h", "Asian")
        assert compute_similarity(dna_a, dna_b) == compute_similarity(dna_b, dna_a)

    def test_returns_float_between_0_and_1(self, sample_analysis):
        dna_a = encode_setup_dna(sample_analysis, {}, "1h", "London")
        dna_b = encode_setup_dna({}, {}, "4h", "Asian")
        sim = compute_similarity(dna_a, dna_b)
        assert 0.0 <= sim <= 1.0


# ═══════════════════════════════════════════════════════════════
# Profile Store
# ═══════════════════════════════════════════════════════════════

class TestSetupProfileStore:
    """Step 2: SetupProfileStore manages historical profiles."""

    def test_add_and_count(self, tmp_path):
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        store.add_profile("id1", {"killzone": "London"}, "tp1", 2.0)
        assert store.profile_count() == 1

    def test_dedup_by_setup_id(self, tmp_path):
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        store.add_profile("id1", {"killzone": "London"}, "tp1", 2.0)
        store.add_profile("id1", {"killzone": "London"}, "tp1", 2.0)
        assert store.profile_count() == 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "profiles.json")
        store1 = SetupProfileStore(path)
        store1.add_profile("id1", {"killzone": "London"}, "tp1", 2.0)
        # Reload from disk
        store2 = SetupProfileStore(path)
        assert store2.profile_count() == 1

    def test_find_similar(self, tmp_path, sample_analysis):
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        store.add_profile("id1", dna, "tp1", 2.0)
        store.add_profile("id2", dna, "sl", -1.0)
        matches = store.find_similar(dna, top_k=5, min_similarity=0.5)
        assert len(matches) == 2
        assert matches[0][1] == pytest.approx(1.0)

    def test_find_similar_respects_min_similarity(self, tmp_path, sample_analysis):
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        dna_good = encode_setup_dna(sample_analysis, {}, "1h", "London")
        dna_bad = encode_setup_dna({}, {}, "4h", "Asian")
        store.add_profile("id1", dna_good, "tp1", 2.0)
        store.add_profile("id2", dna_bad, "sl", -1.0)
        matches = store.find_similar(dna_good, top_k=5, min_similarity=0.8)
        assert len(matches) == 1  # Only the good match

    def test_get_conditional_stats(self, tmp_path, sample_analysis):
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        for i in range(10):
            outcome = "tp1" if i < 7 else "sl"
            pnl = 2.0 if i < 7 else -1.0
            store.add_profile(f"id{i}", dna, outcome, pnl)
        stats = store.get_conditional_stats(dna)
        assert stats["match_count"] == 10
        assert stats["win_rate"] == pytest.approx(0.7)
        assert stats["avg_rr"] > 0
        assert stats["best_outcome"] == "tp1"

    def test_conditional_stats_few_matches(self, tmp_path):
        """Fewer than 3 matches returns empty stats."""
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        store.add_profile("id1", {"killzone": "London"}, "tp1", 2.0)
        stats = store.get_conditional_stats({"killzone": "London"})
        assert stats["match_count"] <= 1
        assert stats["win_rate"] == 0.0

    def test_get_learned_rules_empty(self, tmp_path):
        """No rules with insufficient data."""
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        rules = store.get_learned_rules(min_samples=20)
        assert rules == []

    def test_get_learned_rules_generates_rules(self, tmp_path):
        """Generate rules when enough data with significant WR difference."""
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        # Add 30 London+sweep wins
        for i in range(30):
            store.add_profile(f"lon_win_{i}",
                              {"killzone": "London", "has_sweep": True,
                               "timeframe": "1h", "ob_strength": "strong",
                               "has_fvg": True},
                              "tp1", 2.0)
        # Add 30 Asian+no_sweep losses to bring baseline down
        for i in range(30):
            store.add_profile(f"asi_loss_{i}",
                              {"killzone": "Asian", "has_sweep": False,
                               "timeframe": "4h", "ob_strength": "weak",
                               "has_fvg": False},
                              "sl", -1.0)
        rules = store.get_learned_rules(min_samples=20)
        assert len(rules) > 0
        # At least one rule should mention London or Asian
        rule_text = " ".join(rules)
        assert "London" in rule_text or "Asian" in rule_text


# ═══════════════════════════════════════════════════════════════
# Quality Adjustment (integration logic from scanner.py)
# ═══════════════════════════════════════════════════════════════

class TestQualityAdjustment:
    """Step 5b: DNA-based quality upgrades and downgrades."""

    def _make_store_with_profiles(self, tmp_path, dna, win_rate, count=20):
        """Helper: create a store with `count` profiles at given win_rate."""
        store = SetupProfileStore(str(tmp_path / "profiles.json"))
        wins = int(count * win_rate)
        for i in range(count):
            outcome = "tp1" if i < wins else "sl"
            pnl = 2.0 if i < wins else -1.0
            store.add_profile(f"id{i}", dna, outcome, pnl)
        return store

    def test_upgrade_c_to_b_above_70pct(self, tmp_path, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        store = self._make_store_with_profiles(tmp_path, dna, 0.75, count=20)
        stats = store.get_conditional_stats(dna)
        assert stats["match_count"] >= 15
        assert stats["win_rate"] > 0.70

        # Simulate scanner logic
        quality = "C"
        if stats["win_rate"] > 0.70 and quality in ("C", "D"):
            quality = {"C": "B", "D": "C"}[quality]
        assert quality == "B"

    def test_downgrade_b_to_c_below_35pct(self, tmp_path, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        store = self._make_store_with_profiles(tmp_path, dna, 0.25, count=20)
        stats = store.get_conditional_stats(dna)
        assert stats["match_count"] >= 15
        assert stats["win_rate"] < 0.35

        # Simulate scanner logic
        quality = "B"
        if stats["win_rate"] < 0.35 and quality in ("A", "B"):
            quality = {"A": "B", "B": "C"}[quality]
        assert quality == "C"

    def test_no_adjustment_moderate_wr(self, tmp_path, sample_analysis):
        dna = encode_setup_dna(sample_analysis, {}, "1h", "London")
        store = self._make_store_with_profiles(tmp_path, dna, 0.55, count=20)
        stats = store.get_conditional_stats(dna)
        # 55% WR — no adjustment
        quality = "B"
        if stats["win_rate"] < 0.35 and quality in ("A", "B"):
            quality = {"A": "B", "B": "C"}[quality]
        elif stats["win_rate"] > 0.70 and quality in ("C", "D"):
            quality = {"C": "B", "D": "C"}[quality]
        assert quality == "B"  # Unchanged
