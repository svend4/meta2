"""Extra tests for puzzle_reconstruction/utils/shape_match_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.shape_match_utils import (
    ShapeMatchConfig,
    ShapeMatchEntry,
    ShapeMatchSummary,
    make_match_entry,
    entries_from_results,
    summarise_matches,
    filter_good_matches,
    filter_poor_matches,
    filter_by_hu_dist,
    filter_match_by_score_range,
    top_k_match_entries,
    match_entry_stats,
    compare_match_summaries,
    batch_summarise_matches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(idx1=0, idx2=1, score=0.7, hu=1.0, iou=0.5) -> ShapeMatchEntry:
    return ShapeMatchEntry(idx1=idx1, idx2=idx2, score=score,
                            hu_dist=hu, iou=iou)


# ─── ShapeMatchConfig ─────────────────────────────────────────────────────────

class TestShapeMatchConfigExtra:
    def test_defaults(self):
        cfg = ShapeMatchConfig()
        assert cfg.min_score == pytest.approx(0.0) and cfg.method == "hu"

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(min_score=-1.0)

    def test_zero_max_pairs_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(method="invalid")

    def test_valid_methods(self):
        for m in ("hu", "zernike", "combined"):
            ShapeMatchConfig(method=m)


# ─── ShapeMatchEntry ──────────────────────────────────────────────────────────

class TestShapeMatchEntryExtra:
    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            _entry(idx1=-1)

    def test_is_good_true(self):
        assert _entry(score=0.7).is_good is True

    def test_is_good_false(self):
        assert _entry(score=0.3).is_good is False

    def test_meta_default_empty(self):
        assert _entry().meta == {}


# ─── make_match_entry / entries_from_results ──────────────────────────────────

class TestMakeMatchEntryExtra:
    def test_returns_entry(self):
        e = make_match_entry(0, 1, 0.7)
        assert isinstance(e, ShapeMatchEntry)

    def test_entries_from_results_length(self):
        results = [(0, 1, 0.5), (2, 3, 0.8)]
        entries = entries_from_results(results)
        assert len(entries) == 2

    def test_entries_from_results_rank_assigned(self):
        results = [(0, 1, 0.5), (2, 3, 0.8)]
        entries = entries_from_results(results)
        assert entries[0].rank == 0 and entries[1].rank == 1


# ─── summarise_matches ────────────────────────────────────────────────────────

class TestSummariseMatchesExtra:
    def test_empty(self):
        s = summarise_matches([])
        assert s.n_total == 0

    def test_n_good_counted(self):
        entries = [_entry(score=0.8), _entry(score=0.3)]
        s = summarise_matches(entries)
        assert s.n_good == 1 and s.n_poor == 1

    def test_mean_score(self):
        entries = [_entry(score=0.4), _entry(score=0.8)]
        s = summarise_matches(entries)
        assert s.mean_score == pytest.approx(0.6)

    def test_repr_contains_n(self):
        s = summarise_matches([_entry()])
        assert "n=1" in repr(s)


# ─── filters ─────────────────────────────────────────────────────────────────

class TestFilterMatchExtra:
    def test_good(self):
        entries = [_entry(score=0.8), _entry(score=0.3)]
        assert len(filter_good_matches(entries)) == 1

    def test_poor(self):
        entries = [_entry(score=0.8), _entry(score=0.3)]
        assert len(filter_poor_matches(entries)) == 1

    def test_by_hu_dist(self):
        entries = [_entry(hu=2.0), _entry(hu=15.0)]
        assert len(filter_by_hu_dist(entries, max_hu=10.0)) == 1

    def test_by_score_range(self):
        entries = [_entry(score=0.2), _entry(score=0.6), _entry(score=0.9)]
        assert len(filter_match_by_score_range(entries, 0.3, 0.7)) == 1


# ─── top_k / stats ───────────────────────────────────────────────────────────

class TestRankMatchExtra:
    def test_top_k(self):
        entries = [_entry(score=0.3), _entry(score=0.9)]
        top = top_k_match_entries(entries, 1)
        assert top[0].score == pytest.approx(0.9)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_match_entries([], 0)

    def test_stats_empty(self):
        s = match_entry_stats([])
        assert s["n"] == 0

    def test_stats_means(self):
        entries = [_entry(score=0.4, hu=2.0, iou=0.3),
                   _entry(score=0.8, hu=4.0, iou=0.7)]
        s = match_entry_stats(entries)
        assert s["mean_score"] == pytest.approx(0.6)
        assert s["mean_hu_dist"] == pytest.approx(3.0)


# ─── compare / batch ─────────────────────────────────────────────────────────

class TestCompareMatchExtra:
    def test_returns_dict(self):
        s = summarise_matches([_entry()])
        d = compare_match_summaries(s, s)
        assert d["mean_score_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_matches([[_entry()], []])
        assert len(result) == 2
