"""Tests for puzzle_reconstruction.utils.color_edge_export_utils."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.color_edge_export_utils import (
    ColorMatchAnalysisConfig,
    ColorMatchAnalysisEntry,
    ColorMatchAnalysisSummary,
    make_color_match_analysis_entry,
    summarise_color_match_analysis,
    filter_strong_color_matches,
    filter_weak_color_matches,
    filter_color_by_method,
    top_k_color_match_entries,
    best_color_match_entry,
    color_match_analysis_stats,
    compare_color_match_summaries,
    batch_summarise_color_match_analysis,
    EdgeDetectionAnalysisConfig,
    EdgeDetectionAnalysisEntry,
    EdgeDetectionAnalysisSummary,
    make_edge_detection_entry,
    summarise_edge_detection_entries,
    filter_edge_by_min_density,
    filter_edge_by_method,
    top_k_edge_density_entries,
    best_edge_density_entry,
    edge_detection_stats,
    compare_edge_detection_summaries,
    batch_summarise_edge_detection_entries,
)

np.random.seed(0)


def _make_color_entries(n=8):
    entries = []
    for i in range(n):
        s = float(np.random.uniform(0.1, 1.0))
        entries.append(ColorMatchAnalysisEntry(
            idx1=i, idx2=i+1, score=s,
            hist_score=s * 0.9, moment_score=s * 0.8,
            profile_score=s * 0.95, method="hsv" if i % 2 == 0 else "rgb",
        ))
    return entries


def _make_edge_entries(n=8):
    entries = []
    for i in range(n):
        d = float(np.random.uniform(0.0, 1.0))
        entries.append(EdgeDetectionAnalysisEntry(
            fragment_id=i, density=d,
            n_contours=int(d * 10),
            method="canny" if i % 2 == 0 else "sobel",
        ))
    return entries


# ── 1. ColorMatchAnalysisConfig defaults ─────────────────────────────────────
def test_color_config_defaults():
    cfg = ColorMatchAnalysisConfig()
    assert cfg.min_score == 0.0
    assert cfg.colorspace == "hsv"
    assert cfg.metric == "bhatt"


# ── 2. make_color_match_analysis_entry ───────────────────────────────────────
def test_make_color_entry():
    e = make_color_match_analysis_entry(0, 1, 0.7, 0.6, 0.5, 0.8, "rgb")
    assert e.idx1 == 0
    assert e.idx2 == 1
    assert e.score == 0.7
    assert e.method == "rgb"


# ── 3. summarise_color_match_analysis empty ──────────────────────────────────
def test_summarise_color_empty():
    s = summarise_color_match_analysis([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0


# ── 4. summarise_color_match_analysis nonempty ───────────────────────────────
def test_summarise_color_nonempty():
    entries = _make_color_entries(10)
    s = summarise_color_match_analysis(entries)
    assert s.n_entries == 10
    assert s.min_score <= s.mean_score <= s.max_score


# ── 5. filter_strong_color_matches ───────────────────────────────────────────
def test_filter_strong():
    entries = _make_color_entries(10)
    strong = filter_strong_color_matches(entries, 0.6)
    assert all(e.score >= 0.6 for e in strong)


# ── 6. filter_weak_color_matches ─────────────────────────────────────────────
def test_filter_weak():
    entries = _make_color_entries(10)
    weak = filter_weak_color_matches(entries, 0.6)
    assert all(e.score < 0.6 for e in weak)


# ── 7. filter_color_by_method ────────────────────────────────────────────────
def test_filter_by_method():
    entries = _make_color_entries(8)
    hsv = filter_color_by_method(entries, "hsv")
    assert all(e.method == "hsv" for e in hsv)


# ── 8. top_k_color_match_entries ─────────────────────────────────────────────
def test_top_k_color():
    entries = _make_color_entries(10)
    top3 = top_k_color_match_entries(entries, 3)
    assert len(top3) == 3
    scores = [e.score for e in top3]
    assert scores == sorted(scores, reverse=True)


# ── 9. best_color_match_entry ────────────────────────────────────────────────
def test_best_color_entry():
    entries = _make_color_entries(10)
    best = best_color_match_entry(entries)
    assert best is not None
    assert best.score == max(e.score for e in entries)


def test_best_color_entry_empty():
    assert best_color_match_entry([]) is None


# ── 10. color_match_analysis_stats ───────────────────────────────────────────
def test_color_stats():
    entries = _make_color_entries(10)
    stats = color_match_analysis_stats(entries)
    assert stats["count"] == 10.0
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_color_stats_empty():
    stats = color_match_analysis_stats([])
    assert stats["count"] == 0


# ── 11. compare_color_match_summaries ────────────────────────────────────────
def test_compare_color_summaries():
    ea = _make_color_entries(8)
    eb = _make_color_entries(6)
    sa = summarise_color_match_analysis(ea)
    sb = summarise_color_match_analysis(eb)
    delta = compare_color_match_summaries(sa, sb)
    assert "mean_score_delta" in delta
    assert "mean_hist_delta" in delta


# ── 12. batch_summarise_color_match_analysis ─────────────────────────────────
def test_batch_color():
    groups = [_make_color_entries(5), _make_color_entries(3)]
    summaries = batch_summarise_color_match_analysis(groups)
    assert len(summaries) == 2
    assert summaries[0].n_entries == 5
    assert summaries[1].n_entries == 3


# ── 13. EdgeDetectionAnalysisConfig ──────────────────────────────────────────
def test_edge_config_defaults():
    cfg = EdgeDetectionAnalysisConfig()
    assert cfg.min_density == 0.0
    assert cfg.method == "canny"


# ── 14. make_edge_detection_entry ────────────────────────────────────────────
def test_make_edge_entry():
    e = make_edge_detection_entry(5, 0.4, 10, "sobel")
    assert e.fragment_id == 5
    assert e.density == 0.4
    assert e.n_contours == 10
    assert e.method == "sobel"


# ── 15. summarise_edge_detection_entries ─────────────────────────────────────
def test_summarise_edge_empty():
    s = summarise_edge_detection_entries([])
    assert s.n_entries == 0
    assert s.methods == []


def test_summarise_edge_nonempty():
    entries = _make_edge_entries(10)
    s = summarise_edge_detection_entries(entries)
    assert s.n_entries == 10
    assert s.min_density <= s.mean_density <= s.max_density


# ── 16. filter_edge_by_min_density ───────────────────────────────────────────
def test_filter_edge_density():
    entries = _make_edge_entries(10)
    filtered = filter_edge_by_min_density(entries, 0.5)
    assert all(e.density >= 0.5 for e in filtered)


# ── 17. filter_edge_by_method ────────────────────────────────────────────────
def test_filter_edge_method():
    entries = _make_edge_entries(8)
    canny = filter_edge_by_method(entries, "canny")
    assert all(e.method == "canny" for e in canny)


# ── 18. top_k_edge_density_entries ───────────────────────────────────────────
def test_top_k_edge():
    entries = _make_edge_entries(10)
    top5 = top_k_edge_density_entries(entries, 5)
    assert len(top5) == 5
    densities = [e.density for e in top5]
    assert densities == sorted(densities, reverse=True)


# ── 19. best_edge_density_entry ──────────────────────────────────────────────
def test_best_edge_entry():
    entries = _make_edge_entries(8)
    best = best_edge_density_entry(entries)
    assert best is not None
    assert best.density == max(e.density for e in entries)


# ── 20. edge_detection_stats ─────────────────────────────────────────────────
def test_edge_stats():
    entries = _make_edge_entries(8)
    stats = edge_detection_stats(entries)
    assert stats["count"] == 8.0
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_edge_stats_empty():
    stats = edge_detection_stats([])
    assert stats["count"] == 0


# ── 21. compare_edge_detection_summaries ─────────────────────────────────────
def test_compare_edge_summaries():
    ea = _make_edge_entries(6)
    eb = _make_edge_entries(4)
    sa = summarise_edge_detection_entries(ea)
    sb = summarise_edge_detection_entries(eb)
    delta = compare_edge_detection_summaries(sa, sb)
    assert "mean_density_delta" in delta
    assert "mean_contours_delta" in delta


# ── 22. batch_summarise_edge_detection_entries ───────────────────────────────
def test_batch_edge():
    groups = [_make_edge_entries(5), _make_edge_entries(3)]
    summaries = batch_summarise_edge_detection_entries(groups)
    assert len(summaries) == 2
