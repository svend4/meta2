"""Extra tests for puzzle_reconstruction/utils/score_norm_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.score_norm_utils import (
    ScoreNormConfig,
    ScoreNormEntry,
    ScoreNormSummary,
    make_norm_entry,
    entries_from_scores,
    summarise_norm,
    filter_by_normalized_range,
    filter_by_original_range,
    top_k_norm_entries,
    norm_entry_stats,
    compare_norm_summaries,
    batch_summarise_norm,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(idx=0, orig=0.5, norm=0.7, method="minmax") -> ScoreNormEntry:
    return ScoreNormEntry(idx=idx, original_score=orig,
                           normalized_score=norm, method=method)


# ─── ScoreNormConfig ──────────────────────────────────────────────────────────

class TestScoreNormConfigExtra:
    def test_defaults(self):
        cfg = ScoreNormConfig()
        assert cfg.method == "minmax" and cfg.clip is True

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ScoreNormConfig(method="invalid")

    def test_bad_feature_range_raises(self):
        with pytest.raises(ValueError):
            ScoreNormConfig(feature_range=(1.0, 0.0))

    def test_valid_methods(self):
        for m in ("minmax", "zscore", "rank", "calibrated"):
            ScoreNormConfig(method=m)  # no exception


# ─── ScoreNormEntry ───────────────────────────────────────────────────────────

class TestScoreNormEntryExtra:
    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            _entry(idx=-1)

    def test_delta_property(self):
        e = _entry(orig=0.3, norm=0.8)
        assert e.delta == pytest.approx(0.5)

    def test_meta_default_empty(self):
        assert _entry().meta == {}


# ─── make_norm_entry ──────────────────────────────────────────────────────────

class TestMakeNormEntryExtra:
    def test_returns_entry(self):
        e = make_norm_entry(0, 0.5, 0.7)
        assert isinstance(e, ScoreNormEntry)

    def test_meta_stored(self):
        e = make_norm_entry(0, 0.5, 0.7, meta={"k": 1})
        assert e.meta["k"] == 1


# ─── entries_from_scores ──────────────────────────────────────────────────────

class TestEntriesFromScoresExtra:
    def test_returns_list(self):
        result = entries_from_scores([0.1, 0.5], [0.3, 0.9])
        assert len(result) == 2

    def test_ids_assigned(self):
        result = entries_from_scores([0.1, 0.5], [0.3, 0.9])
        assert result[0].idx == 0 and result[1].idx == 1

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            entries_from_scores([0.1], [0.3, 0.9])


# ─── summarise_norm ───────────────────────────────────────────────────────────

class TestSummariseNormExtra:
    def test_empty(self):
        s = summarise_norm([])
        assert s.n_total == 0

    def test_min_max_computed(self):
        entries = [_entry(orig=0.2, norm=0.4), _entry(orig=0.8, norm=0.9)]
        s = summarise_norm(entries)
        assert s.original_min == pytest.approx(0.2)
        assert s.original_max == pytest.approx(0.8)
        assert s.normalized_min == pytest.approx(0.4)
        assert s.normalized_max == pytest.approx(0.9)

    def test_repr_contains_method(self):
        s = summarise_norm([_entry()])
        assert "minmax" in repr(s)


# ─── filters ─────────────────────────────────────────────────────────────────

class TestFilterNormExtra:
    def test_by_normalized_range(self):
        entries = [_entry(norm=0.2), _entry(norm=0.8)]
        assert len(filter_by_normalized_range(entries, 0.5, 1.0)) == 1

    def test_by_original_range(self):
        entries = [_entry(orig=0.1), _entry(orig=0.9)]
        assert len(filter_by_original_range(entries, 0.5, 1.0)) == 1


# ─── top_k_norm_entries ───────────────────────────────────────────────────────

class TestTopKNormEntriesExtra:
    def test_returns_k(self):
        entries = [_entry(norm=0.3), _entry(norm=0.9), _entry(norm=0.5)]
        top = top_k_norm_entries(entries, 2)
        assert len(top) == 2 and top[0].normalized_score == pytest.approx(0.9)

    def test_k_less_than_one_raises(self):
        with pytest.raises(ValueError):
            top_k_norm_entries([], 0)


# ─── norm_entry_stats ─────────────────────────────────────────────────────────

class TestNormEntryStatsExtra:
    def test_empty(self):
        s = norm_entry_stats([])
        assert s["n"] == 0

    def test_mean_computed(self):
        entries = [_entry(orig=0.4, norm=0.6), _entry(orig=0.8, norm=1.0)]
        s = norm_entry_stats(entries)
        assert s["mean_original"] == pytest.approx(0.6)
        assert s["mean_normalized"] == pytest.approx(0.8)


# ─── compare / batch ─────────────────────────────────────────────────────────

class TestCompareNormExtra:
    def test_returns_dict(self):
        s = summarise_norm([_entry()])
        d = compare_norm_summaries(s, s)
        assert isinstance(d, dict)

    def test_zero_deltas_identical(self):
        s = summarise_norm([_entry()])
        d = compare_norm_summaries(s, s)
        assert d["n_total_delta"] == 0

    def test_batch_length(self):
        result = batch_summarise_norm([([0.1], [0.5]), ([0.2, 0.4], [0.6, 0.8])])
        assert len(result) == 2
