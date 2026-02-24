"""Extra tests for puzzle_reconstruction/utils/color_hist_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.color_hist_utils import (
    ColorHistConfig,
    ColorHistEntry,
    ColorHistSummary,
    make_color_hist_entry,
    entries_from_comparisons,
    summarise_color_hist,
    filter_good_hist_entries,
    filter_poor_hist_entries,
    filter_by_intersection_range,
    filter_by_chi2_range,
    filter_by_space,
    top_k_hist_entries,
    best_hist_entry,
    color_hist_stats,
    compare_hist_summaries,
    batch_summarise_color_hist,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(frag_i=0, frag_j=1, inter=0.7, chi2=0.6, space="hsv") -> ColorHistEntry:
    return ColorHistEntry(
        frag_i=frag_i, frag_j=frag_j,
        intersection=inter, chi2=chi2, space=space,
    )


def _entries(n=5) -> list:
    return [_entry(frag_i=i, frag_j=i+1,
                   inter=float(i+1)/n, chi2=float(i+1)/n)
            for i in range(n)]


def _summary(entries=None) -> ColorHistSummary:
    return summarise_color_hist(entries or _entries())


# ─── ColorHistConfig ──────────────────────────────────────────────────────────

class TestColorHistConfigExtra:
    def test_default_min_score(self):
        assert ColorHistConfig().min_score == pytest.approx(0.0)

    def test_default_max_score(self):
        assert ColorHistConfig().max_score == pytest.approx(1.0)

    def test_default_good_threshold(self):
        assert ColorHistConfig().good_threshold == pytest.approx(0.7)

    def test_default_poor_threshold(self):
        assert ColorHistConfig().poor_threshold == pytest.approx(0.3)

    def test_default_space(self):
        assert ColorHistConfig().space == "hsv"

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=-0.1)

    def test_max_score_lt_min_raises(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=0.8, max_score=0.3)

    def test_good_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ColorHistConfig(good_threshold=1.5)

    def test_poor_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            ColorHistConfig(poor_threshold=-0.1)

    def test_custom_values(self):
        cfg = ColorHistConfig(min_score=0.1, max_score=0.9, space="lab")
        assert cfg.min_score == pytest.approx(0.1)
        assert cfg.space == "lab"


# ─── ColorHistEntry ───────────────────────────────────────────────────────────

class TestColorHistEntryExtra:
    def test_stores_frag_i(self):
        assert _entry(frag_i=3).frag_i == 3

    def test_stores_frag_j(self):
        assert _entry(frag_j=7).frag_j == 7

    def test_stores_intersection(self):
        assert _entry(inter=0.65).intersection == pytest.approx(0.65)

    def test_stores_chi2(self):
        assert _entry(chi2=0.45).chi2 == pytest.approx(0.45)

    def test_score_is_average(self):
        e = _entry(inter=0.8, chi2=0.6)
        assert e.score == pytest.approx(0.7)

    def test_default_n_bins(self):
        assert _entry().n_bins == 32

    def test_default_params_empty(self):
        assert _entry().params == {}

    def test_default_space_hsv(self):
        e = ColorHistEntry(frag_i=0, frag_j=1, intersection=0.5, chi2=0.5)
        assert e.space == "hsv"


# ─── make_color_hist_entry ────────────────────────────────────────────────────

class TestMakeColorHistEntryExtra:
    def test_returns_entry(self):
        e = make_color_hist_entry(0, 1, 0.7, 0.6)
        assert isinstance(e, ColorHistEntry)

    def test_values_stored(self):
        e = make_color_hist_entry(2, 5, 0.8, 0.4, space="lab", n_bins=64)
        assert e.frag_i == 2 and e.frag_j == 5
        assert e.intersection == pytest.approx(0.8)
        assert e.space == "lab"
        assert e.n_bins == 64

    def test_none_params_empty(self):
        e = make_color_hist_entry(0, 1, 0.5, 0.5, params=None)
        assert e.params == {}

    def test_score_computed(self):
        e = make_color_hist_entry(0, 1, 0.6, 0.4)
        assert e.score == pytest.approx(0.5)


# ─── entries_from_comparisons ─────────────────────────────────────────────────

class TestEntriesFromComparisonsExtra:
    def test_returns_list(self):
        result = entries_from_comparisons([(0, 1)], [0.5], [0.5])
        assert isinstance(result, list)

    def test_length_matches(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = entries_from_comparisons(pairs, [0.5]*3, [0.4]*3)
        assert len(result) == 3

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            entries_from_comparisons([(0, 1), (1, 2)], [0.5], [0.5, 0.4])

    def test_empty_returns_empty(self):
        assert entries_from_comparisons([], [], []) == []

    def test_all_are_entries(self):
        for e in entries_from_comparisons([(0, 1)], [0.6], [0.7]):
            assert isinstance(e, ColorHistEntry)


# ─── summarise_color_hist ─────────────────────────────────────────────────────

class TestSummariseColorHistExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_color_hist(_entries()), ColorHistSummary)

    def test_n_entries_correct(self):
        assert summarise_color_hist(_entries(6)).n_entries == 6

    def test_empty_entries(self):
        s = summarise_color_hist([])
        assert s.n_entries == 0

    def test_mean_in_range(self):
        s = summarise_color_hist(_entries(5))
        assert s.min_score <= s.mean_score <= s.max_score

    def test_space_stored(self):
        s = summarise_color_hist(_entries(), space="lab")
        assert s.space == "lab"


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterHistEntriesExtra:
    def test_filter_good(self):
        entries = [_entry(inter=0.8, chi2=0.8), _entry(inter=0.2, chi2=0.2)]
        result = filter_good_hist_entries(entries, threshold=0.7)
        assert all(e.score >= 0.7 for e in result)

    def test_filter_poor(self):
        entries = [_entry(inter=0.8, chi2=0.8), _entry(inter=0.2, chi2=0.2)]
        result = filter_poor_hist_entries(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in result)

    def test_filter_by_intersection_range(self):
        entries = [_entry(inter=0.2), _entry(inter=0.6), _entry(inter=0.9)]
        result = filter_by_intersection_range(entries, lo=0.4, hi=0.8)
        assert all(0.4 <= e.intersection <= 0.8 for e in result)

    def test_filter_by_chi2_range(self):
        entries = [_entry(chi2=0.1), _entry(chi2=0.5)]
        result = filter_by_chi2_range(entries, lo=0.3, hi=0.7)
        assert all(0.3 <= e.chi2 <= 0.7 for e in result)

    def test_filter_by_space(self):
        entries = [_entry(space="hsv"), _entry(space="lab")]
        result = filter_by_space(entries, "lab")
        assert all(e.space == "lab" for e in result)

    def test_empty_returns_empty(self):
        assert filter_good_hist_entries([], 0.5) == []


# ─── top_k and best ───────────────────────────────────────────────────────────

class TestTopKAndBestHistExtra:
    def test_top_k_length(self):
        result = top_k_hist_entries(_entries(5), 3)
        assert len(result) == 3

    def test_top_k_descending(self):
        result = top_k_hist_entries(_entries(5), 5)
        scores = [e.score for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_best_is_highest(self):
        entries = [_entry(inter=0.2, chi2=0.2), _entry(inter=0.9, chi2=0.9)]
        best = best_hist_entry(entries)
        assert best.score == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_hist_entry([]) is None


# ─── color_hist_stats ─────────────────────────────────────────────────────────

class TestColorHistStatsExtra:
    def test_returns_dict(self):
        assert isinstance(color_hist_stats(_entries()), dict)

    def test_keys_present(self):
        stats = color_hist_stats(_entries(3))
        for k in ("count", "mean", "std", "min", "max",
                  "mean_intersection", "mean_chi2"):
            assert k in stats

    def test_count_correct(self):
        assert color_hist_stats(_entries(7))["count"] == 7

    def test_empty_entries(self):
        assert color_hist_stats([])["count"] == 0


# ─── compare_hist_summaries ───────────────────────────────────────────────────

class TestCompareHistSummariesExtra:
    def test_returns_dict(self):
        s = _summary()
        assert isinstance(compare_hist_summaries(s, s), dict)

    def test_keys_present(self):
        s = _summary()
        d = compare_hist_summaries(s, s)
        for k in ("d_mean_intersection", "d_mean_chi2", "d_mean_score",
                  "d_n_entries"):
            assert k in d

    def test_identical_zero_delta(self):
        s = _summary()
        d = compare_hist_summaries(s, s)
        assert d["d_mean_score"] == pytest.approx(0.0)


# ─── batch_summarise_color_hist ───────────────────────────────────────────────

class TestBatchSummariseColorHistExtra:
    def test_returns_list(self):
        assert isinstance(batch_summarise_color_hist([_entries(2)]), list)

    def test_length_matches(self):
        result = batch_summarise_color_hist([_entries(2), _entries(3)])
        assert len(result) == 2

    def test_each_is_summary(self):
        for s in batch_summarise_color_hist([_entries(2)]):
            assert isinstance(s, ColorHistSummary)

    def test_empty_groups(self):
        assert batch_summarise_color_hist([]) == []
