"""Extra tests for puzzle_reconstruction/utils/patch_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.patch_score_utils import (
    PatchScoreConfig,
    PatchScoreEntry,
    PatchScoreSummary,
    make_patch_entry,
    entries_from_patch_matches,
    summarise_patch_scores,
    filter_good_patch_scores,
    filter_poor_patch_scores,
    filter_patch_by_score_range,
    filter_by_side_pair,
    filter_by_ncc_range,
    top_k_patch_entries,
    best_patch_entry,
    patch_score_stats,
    compare_patch_summaries,
    batch_summarise_patch_scores,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(pair_id=0, idx1=0, idx2=1, side1=0, side2=1,
           ncc=0.5, ssd=0.3, ssim=0.7, total=0.6, rank=1) -> PatchScoreEntry:
    return PatchScoreEntry(pair_id=pair_id, idx1=idx1, idx2=idx2,
                           side1=side1, side2=side2, ncc=ncc, ssd=ssd,
                           ssim=ssim, total_score=total, rank=rank)


# ─── PatchScoreConfig ─────────────────────────────────────────────────────────

class TestPatchScoreConfigExtra:
    def test_default_min_score(self):
        assert PatchScoreConfig().min_score == pytest.approx(0.0)

    def test_default_max_pairs(self):
        assert PatchScoreConfig().max_pairs == 1000

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(min_score=-0.1)

    def test_zero_max_pairs_raises(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(max_pairs=0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(method="unknown")

    def test_valid_methods(self):
        for m in ("total", "ncc", "ssd", "ssim"):
            cfg = PatchScoreConfig(method=m)
            assert cfg.method == m


# ─── PatchScoreEntry ──────────────────────────────────────────────────────────

class TestPatchScoreEntryExtra:
    def test_pair_property(self):
        e = _entry(idx1=2, idx2=3)
        assert e.pair == (2, 3)

    def test_is_good_true(self):
        e = _entry(total=0.8)
        assert e.is_good is True

    def test_is_good_false(self):
        e = _entry(total=0.3)
        assert e.is_good is False

    def test_negative_pair_id_raises(self):
        with pytest.raises(ValueError):
            PatchScoreEntry(pair_id=-1, idx1=0, idx2=1, side1=0, side2=1,
                            ncc=0.5, ssd=0.3, ssim=0.7, total_score=0.6)

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            PatchScoreEntry(pair_id=0, idx1=-1, idx2=1, side1=0, side2=1,
                            ncc=0.5, ssd=0.3, ssim=0.7, total_score=0.6)

    def test_ssd_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScoreEntry(pair_id=0, idx1=0, idx2=1, side1=0, side2=1,
                            ncc=0.5, ssd=1.5, ssim=0.7, total_score=0.6)

    def test_total_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScoreEntry(pair_id=0, idx1=0, idx2=1, side1=0, side2=1,
                            ncc=0.5, ssd=0.3, ssim=0.7, total_score=1.5)


# ─── make_patch_entry ─────────────────────────────────────────────────────────

class TestMakePatchEntryExtra:
    def test_returns_entry(self):
        e = make_patch_entry(0, 0, 1, 0, 1, 0.5, 0.3, 0.7, 0.6)
        assert isinstance(e, PatchScoreEntry)

    def test_values_stored(self):
        e = make_patch_entry(1, 2, 3, 0, 1, 0.8, 0.2, 0.9, 0.85)
        assert e.total_score == pytest.approx(0.85)
        assert e.pair_id == 1

    def test_meta_stored(self):
        e = make_patch_entry(0, 0, 1, 0, 1, 0.5, 0.3, 0.7, 0.6, meta={"k": 1})
        assert e.meta["k"] == 1


# ─── entries_from_patch_matches ───────────────────────────────────────────────

class TestEntriesFromPatchMatchesExtra:
    def test_returns_list(self):
        class FakePM:
            idx1=0; idx2=1; side1=0; side2=1
            ncc=0.5; ssd=0.3; ssim=0.7; total_score=0.6
        result = entries_from_patch_matches([FakePM()])
        assert isinstance(result, list) and len(result) == 1

    def test_rank_assigned(self):
        class FakePM:
            idx1=0; idx2=1; side1=0; side2=1
            ncc=0.5; ssd=0.3; ssim=0.7; total_score=0.6
        result = entries_from_patch_matches([FakePM()])
        assert result[0].rank == 1

    def test_empty_input(self):
        assert entries_from_patch_matches([]) == []


# ─── summarise_patch_scores ───────────────────────────────────────────────────

class TestSummarisePatchScoresExtra:
    def test_empty_returns_summary(self):
        s = summarise_patch_scores([])
        assert s.n_total == 0

    def test_n_good_n_poor(self):
        entries = [_entry(total=0.8), _entry(total=0.3)]
        s = summarise_patch_scores(entries)
        assert s.n_good == 1 and s.n_poor == 1

    def test_mean_total(self):
        entries = [_entry(total=0.4), _entry(total=0.6)]
        s = summarise_patch_scores(entries)
        assert s.mean_total == pytest.approx(0.5)

    def test_max_min_total(self):
        entries = [_entry(total=0.2), _entry(total=0.9)]
        s = summarise_patch_scores(entries)
        assert s.max_total == pytest.approx(0.9)
        assert s.min_total == pytest.approx(0.2)

    def test_repr_contains_n_total(self):
        s = summarise_patch_scores([_entry()])
        assert "n_total=1" in repr(s)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterPatchScoreExtra:
    def test_filter_good(self):
        entries = [_entry(total=0.8), _entry(total=0.3)]
        assert len(filter_good_patch_scores(entries)) == 1

    def test_filter_poor(self):
        entries = [_entry(total=0.8), _entry(total=0.3)]
        assert len(filter_poor_patch_scores(entries)) == 1

    def test_filter_score_range(self):
        entries = [_entry(total=0.2), _entry(total=0.5), _entry(total=0.9)]
        result = filter_patch_by_score_range(entries, 0.4, 0.7)
        assert len(result) == 1

    def test_filter_by_side_pair(self):
        entries = [_entry(side1=0, side2=1), _entry(side1=1, side2=2)]
        result = filter_by_side_pair(entries, 0, 1)
        assert len(result) == 1

    def test_filter_by_ncc_range(self):
        entries = [_entry(ncc=0.2), _entry(ncc=0.8)]
        result = filter_by_ncc_range(entries, 0.5, 1.0)
        assert len(result) == 1


# ─── top_k / best ─────────────────────────────────────────────────────────────

class TestRankPatchScoreExtra:
    def test_top_k(self):
        entries = [_entry(total=0.3), _entry(total=0.9), _entry(total=0.6)]
        top = top_k_patch_entries(entries, 2)
        assert top[0].total_score == pytest.approx(0.9)
        assert len(top) == 2

    def test_top_k_zero_returns_empty(self):
        entries = [_entry(total=0.8)]
        assert top_k_patch_entries(entries, 0) == []

    def test_best_patch_entry_none(self):
        assert best_patch_entry([]) is None

    def test_best_patch_entry(self):
        entries = [_entry(total=0.3), _entry(total=0.9)]
        best = best_patch_entry(entries)
        assert best.total_score == pytest.approx(0.9)


# ─── patch_score_stats ────────────────────────────────────────────────────────

class TestPatchScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = patch_score_stats([])
        assert s["n"] == 0

    def test_count_correct(self):
        s = patch_score_stats([_entry(), _entry()])
        assert s["n"] == 2

    def test_min_max(self):
        entries = [_entry(total=0.2), _entry(total=0.8)]
        s = patch_score_stats(entries)
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)


# ─── compare_patch_summaries / batch ─────────────────────────────────────────

class TestComparePatchSummariesExtra:
    def test_returns_dict(self):
        s = summarise_patch_scores([_entry()])
        d = compare_patch_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_patch_scores([_entry()])
        d = compare_patch_summaries(s, s)
        assert d["delta_mean_total"] == pytest.approx(0.0)

    def test_a_better_same(self):
        s = summarise_patch_scores([_entry()])
        d = compare_patch_summaries(s, s)
        assert d["a_better"] is True

    def test_batch_length(self):
        result = batch_summarise_patch_scores([[_entry()], [_entry(), _entry()]])
        assert len(result) == 2
