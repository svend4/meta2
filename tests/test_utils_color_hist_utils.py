"""Tests for puzzle_reconstruction.utils.color_hist_utils."""
import pytest
import numpy as np
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

np.random.seed(7)


def _make_entries(n=10, space="hsv"):
    entries = []
    for i in range(n):
        inter = float(np.random.uniform(0.0, 1.0))
        chi2 = float(np.random.uniform(0.0, 1.0))
        entries.append(make_color_hist_entry(i, i+1, inter, chi2, space=space))
    return entries


# ── 1. ColorHistConfig defaults ──────────────────────────────────────────────
def test_config_defaults():
    cfg = ColorHistConfig()
    assert cfg.min_score == 0.0
    assert cfg.max_score == 1.0
    assert cfg.good_threshold == 0.7
    assert cfg.poor_threshold == 0.3
    assert cfg.space == "hsv"


# ── 2. ColorHistConfig validation ────────────────────────────────────────────
def test_config_invalid_min_score():
    with pytest.raises(ValueError):
        ColorHistConfig(min_score=-0.1)


def test_config_invalid_max_score():
    with pytest.raises(ValueError):
        ColorHistConfig(min_score=0.5, max_score=0.3)


def test_config_invalid_good_threshold():
    with pytest.raises(ValueError):
        ColorHistConfig(good_threshold=1.5)


# ── 5. ColorHistEntry score property ─────────────────────────────────────────
def test_entry_score():
    e = ColorHistEntry(0, 1, intersection=0.8, chi2=0.6)
    assert abs(e.score - 0.7) < 1e-9


# ── 6. make_color_hist_entry ─────────────────────────────────────────────────
def test_make_entry_basic():
    e = make_color_hist_entry(2, 3, 0.7, 0.5, space="lab", n_bins=64)
    assert e.frag_i == 2
    assert e.frag_j == 3
    assert e.intersection == 0.7
    assert e.chi2 == 0.5
    assert e.space == "lab"
    assert e.n_bins == 64


def test_make_entry_default_params():
    e = make_color_hist_entry(0, 1, 0.4, 0.3)
    assert e.space == "hsv"
    assert e.n_bins == 32
    assert e.params == {}


# ── 8. entries_from_comparisons ──────────────────────────────────────────────
def test_entries_from_comparisons():
    pairs = [(0,1), (1,2), (2,3)]
    inters = [0.9, 0.6, 0.3]
    chi2s = [0.8, 0.7, 0.4]
    entries = entries_from_comparisons(pairs, inters, chi2s)
    assert len(entries) == 3
    assert entries[0].frag_i == 0
    assert entries[0].intersection == 0.9


def test_entries_from_comparisons_length_mismatch():
    with pytest.raises(ValueError):
        entries_from_comparisons([(0,1)], [0.5], [0.4, 0.3])


# ── 10. summarise_color_hist empty ───────────────────────────────────────────
def test_summarise_empty():
    s = summarise_color_hist([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0


# ── 11. summarise_color_hist nonempty ────────────────────────────────────────
def test_summarise_nonempty():
    entries = _make_entries(10)
    s = summarise_color_hist(entries)
    assert s.n_entries == 10
    assert s.min_score <= s.mean_score <= s.max_score
    assert s.std_score >= 0.0


# ── 12. filter_good_hist_entries ─────────────────────────────────────────────
def test_filter_good():
    entries = _make_entries(20)
    good = filter_good_hist_entries(entries, 0.6)
    assert all(e.score >= 0.6 for e in good)


# ── 13. filter_poor_hist_entries ─────────────────────────────────────────────
def test_filter_poor():
    entries = _make_entries(20)
    poor = filter_poor_hist_entries(entries, 0.4)
    assert all(e.score < 0.4 for e in poor)


# ── 14. filter_by_intersection_range ─────────────────────────────────────────
def test_filter_intersection_range():
    entries = _make_entries(20)
    filtered = filter_by_intersection_range(entries, 0.3, 0.7)
    assert all(0.3 <= e.intersection <= 0.7 for e in filtered)


# ── 15. filter_by_chi2_range ─────────────────────────────────────────────────
def test_filter_chi2_range():
    entries = _make_entries(20)
    filtered = filter_by_chi2_range(entries, 0.2, 0.8)
    assert all(0.2 <= e.chi2 <= 0.8 for e in filtered)


# ── 16. filter_by_space ──────────────────────────────────────────────────────
def test_filter_by_space():
    e1 = _make_entries(5, space="hsv")
    e2 = _make_entries(5, space="lab")
    all_entries = e1 + e2
    hsv = filter_by_space(all_entries, "hsv")
    assert len(hsv) == 5
    assert all(e.space == "hsv" for e in hsv)


# ── 17. top_k_hist_entries ───────────────────────────────────────────────────
def test_top_k():
    entries = _make_entries(10)
    top3 = top_k_hist_entries(entries, 3)
    assert len(top3) == 3
    scores = [e.score for e in top3]
    assert scores == sorted(scores, reverse=True)


# ── 18. best_hist_entry ──────────────────────────────────────────────────────
def test_best_entry():
    entries = _make_entries(10)
    best = best_hist_entry(entries)
    assert best is not None
    assert best.score == max(e.score for e in entries)


def test_best_entry_empty():
    assert best_hist_entry([]) is None


# ── 19. color_hist_stats ─────────────────────────────────────────────────────
def test_hist_stats():
    entries = _make_entries(10)
    stats = color_hist_stats(entries)
    assert stats["count"] == 10
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert "mean_intersection" in stats
    assert "mean_chi2" in stats


def test_hist_stats_empty():
    stats = color_hist_stats([])
    assert stats["count"] == 0


# ── 20. compare_hist_summaries ───────────────────────────────────────────────
def test_compare_summaries():
    ea = _make_entries(8)
    eb = _make_entries(6)
    sa = summarise_color_hist(ea)
    sb = summarise_color_hist(eb)
    delta = compare_hist_summaries(sa, sb)
    assert "d_mean_score" in delta
    assert "d_n_entries" in delta
    assert delta["d_n_entries"] == 2


# ── 21. batch_summarise_color_hist ───────────────────────────────────────────
def test_batch_summarise():
    groups = [_make_entries(5), _make_entries(3)]
    summaries = batch_summarise_color_hist(groups)
    assert len(summaries) == 2
    assert summaries[0].n_entries == 5


# ── 22. single entry std ─────────────────────────────────────────────────────
def test_single_entry_std():
    entries = [make_color_hist_entry(0, 1, 0.5, 0.5)]
    s = summarise_color_hist(entries)
    assert s.std_score == 0.0
