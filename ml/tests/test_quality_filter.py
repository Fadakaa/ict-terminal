"""Tests for setup quality filter."""
import os
import pytest
from ml.quality_filter import SetupQualityFilter
from ml.config import make_test_config


class TestSetupQualityFilter:
    @pytest.fixture
    def filt(self):
        cfg = make_test_config(wfo_min_confluence_score=2)
        return SetupQualityFilter(config=cfg)

    @pytest.fixture
    def sample_trades(self):
        return [
            {"candle_index": 50, "direction": "long", "confluence_score": 3,
             "ob_count": 2, "fvg_count": 1, "outcome": "tp1_hit"},
            {"candle_index": 55, "direction": "short", "confluence_score": 1,
             "ob_count": 0, "fvg_count": 0, "outcome": "stopped_out"},
            {"candle_index": 60, "direction": "long", "confluence_score": 2,
             "ob_count": 1, "fvg_count": 1, "outcome": "expired"},
            {"candle_index": 65, "direction": "long", "confluence_score": 4,
             "ob_count": 3, "fvg_count": 2, "outcome": "tp2_hit"},
            # Duplicate of first
            {"candle_index": 50, "direction": "long", "confluence_score": 3,
             "ob_count": 2, "fvg_count": 1, "outcome": "tp1_hit"},
        ]

    def test_removes_low_confluence(self, filt, sample_trades):
        result = filt.filter_basic(sample_trades)
        scores = [t["confluence_score"] for t in result]
        assert all(s >= 2 for s in scores)

    def test_removes_expired(self, filt, sample_trades):
        result = filt.filter_basic(sample_trades)
        assert not any(t["outcome"] == "expired" for t in result)

    def test_removes_duplicates(self, filt, sample_trades):
        result = filt.filter_basic(sample_trades)
        keys = [(t["candle_index"], t["direction"]) for t in result]
        assert len(keys) == len(set(keys))

    def test_keeps_valid_trades(self, filt, sample_trades):
        result = filt.filter_basic(sample_trades)
        # Should keep index 50 (first occurrence) and 65
        assert len(result) == 2

    def test_empty_input(self, filt):
        assert filt.filter_basic([]) == []

    def test_export_for_review(self, filt, sample_trades, tmp_path):
        path = str(tmp_path / "review.csv")
        filt.export_for_review(sample_trades, path)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 6  # header + 5 trades

    def test_export_empty(self, filt, tmp_path):
        path = str(tmp_path / "empty.csv")
        filt.export_for_review([], path)
        assert not os.path.exists(path)

    def test_missing_features_filtered(self, filt):
        trades = [
            {"candle_index": 50, "direction": "long", "confluence_score": 3,
             "outcome": "tp1_hit"},  # Missing ob_count, fvg_count
        ]
        result = filt.filter_basic(trades)
        assert len(result) == 0
