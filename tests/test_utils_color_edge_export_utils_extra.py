"""Extra tests for puzzle_reconstruction/utils/color_edge_export_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cm_entry(idx1=0, idx2=1, score=0.7, method="hsv") -> ColorMatchAnalysisEntry:
    return ColorMatchAnalysisEntry(
        idx1=idx1, idx2=idx2, score=score,
        hist_score=score, moment_score=score, profile_score=score,
        method=method,
    )


def _cm_entries(n=4) -> list:
    return [_cm_entry(idx1=i, score=float(i + 1) / n) for i in range(n)]


def _ed_entry(fid=0, density=0.5, n_contours=3, method="canny") -> EdgeDetectionAnalysisEntry:
    return EdgeDetectionAnalysisEntry(
        fragment_id=fid, density=density, n_contours=n_contours, method=method,
    )


def _ed_entries(n=4) -> list:
    return [_ed_entry(fid=i, density=float(i + 1) / n) for i in range(n)]


# ─── ColorMatchAnalysisConfig ─────────────────────────────────────────────────

class TestColorMatchAnalysisConfigExtra:
    def test_default_min_score(self):
        assert ColorMatchAnalysisConfig().min_score == pytest.approx(0.0)

    def test_default_colorspace(self):
        assert ColorMatchAnalysisConfig().colorspace == "hsv"

    def test_default_metric(self):
        assert ColorMatchAnalysisConfig().metric == "bhatt"

    def test_custom_values(self):
        cfg = ColorMatchAnalysisConfig(min_score=0.4, colorspace="lab", metric="chi")
        assert cfg.min_score == pytest.approx(0.4)
        assert cfg.colorspace == "lab"


# ─── ColorMatchAnalysisEntry ──────────────────────────────────────────────────

class TestColorMatchAnalysisEntryExtra:
    def test_stores_idx1_idx2(self):
        e = _cm_entry(idx1=3, idx2=7)
        assert e.idx1 == 3 and e.idx2 == 7

    def test_stores_score(self):
        assert _cm_entry(score=0.85).score == pytest.approx(0.85)

    def test_stores_method(self):
        assert _cm_entry(method="lab").method == "lab"

    def test_default_method_hsv(self):
        e = ColorMatchAnalysisEntry(idx1=0, idx2=1, score=0.5,
                                     hist_score=0.5, moment_score=0.5,
                                     profile_score=0.5)
        assert e.method == "hsv"


# ─── make_color_match_analysis_entry ──────────────────────────────────────────

class TestMakeColorMatchAnalysisEntryExtra:
    def test_returns_entry(self):
        e = make_color_match_analysis_entry(0, 1, 0.6, 0.7, 0.5, 0.6)
        assert isinstance(e, ColorMatchAnalysisEntry)

    def test_values_stored(self):
        e = make_color_match_analysis_entry(2, 5, 0.8, 0.9, 0.7, 0.8, method="lab")
        assert e.idx1 == 2 and e.idx2 == 5
        assert e.score == pytest.approx(0.8)
        assert e.method == "lab"


# ─── summarise_color_match_analysis ───────────────────────────────────────────

class TestSummariseColorMatchAnalysisExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_color_match_analysis(_cm_entries()), ColorMatchAnalysisSummary)

    def test_n_entries_correct(self):
        s = summarise_color_match_analysis(_cm_entries(5))
        assert s.n_entries == 5

    def test_empty_entries(self):
        s = summarise_color_match_analysis([])
        assert s.n_entries == 0 and s.mean_score == pytest.approx(0.0)

    def test_mean_in_range(self):
        s = summarise_color_match_analysis(_cm_entries(4))
        assert s.min_score <= s.mean_score <= s.max_score


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterColorMatchExtra:
    def test_filter_strong_keeps_above(self):
        entries = [_cm_entry(score=0.3), _cm_entry(score=0.7)]
        result = filter_strong_color_matches(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in result)

    def test_filter_weak_keeps_below(self):
        entries = [_cm_entry(score=0.3), _cm_entry(score=0.7)]
        result = filter_weak_color_matches(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in result)

    def test_filter_by_method(self):
        entries = [_cm_entry(method="hsv"), _cm_entry(method="lab")]
        result = filter_color_by_method(entries, "lab")
        assert all(e.method == "lab" for e in result)

    def test_empty_input(self):
        assert filter_strong_color_matches([], 0.5) == []

    def test_top_k_returns_k(self):
        result = top_k_color_match_entries(_cm_entries(5), 3)
        assert len(result) == 3

    def test_best_returns_highest(self):
        entries = [_cm_entry(score=0.2), _cm_entry(score=0.9), _cm_entry(score=0.5)]
        best = best_color_match_entry(entries)
        assert best.score == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_color_match_entry([]) is None


# ─── color_match_analysis_stats ───────────────────────────────────────────────

class TestColorMatchAnalysisStatsExtra:
    def test_returns_dict(self):
        assert isinstance(color_match_analysis_stats(_cm_entries()), dict)

    def test_keys_present(self):
        stats = color_match_analysis_stats(_cm_entries(3))
        for k in ("count", "mean", "std", "min", "max"):
            assert k in stats

    def test_empty_entries(self):
        assert color_match_analysis_stats([])["count"] == 0

    def test_count_correct(self):
        assert color_match_analysis_stats(_cm_entries(6))["count"] == pytest.approx(6)


# ─── compare_color_match_summaries ────────────────────────────────────────────

class TestCompareColorMatchSummariesExtra:
    def test_returns_dict(self):
        s = summarise_color_match_analysis(_cm_entries(3))
        assert isinstance(compare_color_match_summaries(s, s), dict)

    def test_identical_zero_delta(self):
        s = summarise_color_match_analysis(_cm_entries(3))
        d = compare_color_match_summaries(s, s)
        assert d["mean_score_delta"] == pytest.approx(0.0)


# ─── batch_summarise_color_match_analysis ─────────────────────────────────────

class TestBatchSummariseColorMatchAnalysisExtra:
    def test_returns_list(self):
        result = batch_summarise_color_match_analysis([_cm_entries(2)])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_summarise_color_match_analysis([_cm_entries(2), _cm_entries(3)])
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_color_match_analysis([]) == []


# ─── EdgeDetectionAnalysisConfig ─────────────────────────────────────────────

class TestEdgeDetectionAnalysisConfigExtra:
    def test_default_min_density(self):
        assert EdgeDetectionAnalysisConfig().min_density == pytest.approx(0.0)

    def test_default_method(self):
        assert EdgeDetectionAnalysisConfig().method == "canny"


# ─── EdgeDetectionAnalysisEntry ───────────────────────────────────────────────

class TestEdgeDetectionAnalysisEntryExtra:
    def test_stores_fragment_id(self):
        assert _ed_entry(fid=4).fragment_id == 4

    def test_stores_density(self):
        assert _ed_entry(density=0.3).density == pytest.approx(0.3)

    def test_stores_n_contours(self):
        assert _ed_entry(n_contours=7).n_contours == 7

    def test_default_method(self):
        e = EdgeDetectionAnalysisEntry(fragment_id=0, density=0.5, n_contours=2)
        assert e.method == "canny"


# ─── make_edge_detection_entry ────────────────────────────────────────────────

class TestMakeEdgeDetectionEntryExtra:
    def test_returns_entry(self):
        e = make_edge_detection_entry(0, 0.5, 3)
        assert isinstance(e, EdgeDetectionAnalysisEntry)

    def test_values_stored(self):
        e = make_edge_detection_entry(2, 0.4, 5, method="sobel")
        assert e.fragment_id == 2
        assert e.density == pytest.approx(0.4)
        assert e.n_contours == 5
        assert e.method == "sobel"


# ─── summarise_edge_detection_entries ─────────────────────────────────────────

class TestSummariseEdgeDetectionEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_edge_detection_entries(_ed_entries()), EdgeDetectionAnalysisSummary)

    def test_n_entries_correct(self):
        assert summarise_edge_detection_entries(_ed_entries(5)).n_entries == 5

    def test_empty_entries(self):
        s = summarise_edge_detection_entries([])
        assert s.n_entries == 0

    def test_mean_in_range(self):
        s = summarise_edge_detection_entries(_ed_entries(4))
        assert s.min_density <= s.mean_density <= s.max_density


# ─── edge filter functions ────────────────────────────────────────────────────

class TestFilterEdgeDetectionExtra:
    def test_filter_by_min_density(self):
        entries = [_ed_entry(density=0.2), _ed_entry(density=0.8)]
        result = filter_edge_by_min_density(entries, 0.5)
        assert all(e.density >= 0.5 for e in result)

    def test_filter_by_method(self):
        entries = [_ed_entry(method="canny"), _ed_entry(method="sobel")]
        result = filter_edge_by_method(entries, "sobel")
        assert all(e.method == "sobel" for e in result)

    def test_top_k_density(self):
        result = top_k_edge_density_entries(_ed_entries(5), 2)
        assert len(result) == 2

    def test_best_density(self):
        entries = [_ed_entry(density=0.1), _ed_entry(density=0.9)]
        best = best_edge_density_entry(entries)
        assert best.density == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_edge_density_entry([]) is None


# ─── edge_detection_stats ─────────────────────────────────────────────────────

class TestEdgeDetectionStatsExtra:
    def test_returns_dict(self):
        assert isinstance(edge_detection_stats(_ed_entries()), dict)

    def test_empty_entries(self):
        assert edge_detection_stats([])["count"] == 0

    def test_keys_present(self):
        for k in ("count", "mean", "std", "min", "max"):
            assert k in edge_detection_stats(_ed_entries(3))


# ─── compare_edge_detection_summaries ─────────────────────────────────────────

class TestCompareEdgeDetectionSummariesExtra:
    def test_returns_dict(self):
        s = summarise_edge_detection_entries(_ed_entries(3))
        assert isinstance(compare_edge_detection_summaries(s, s), dict)

    def test_identical_zero_delta(self):
        s = summarise_edge_detection_entries(_ed_entries(3))
        d = compare_edge_detection_summaries(s, s)
        assert d["mean_density_delta"] == pytest.approx(0.0)


# ─── batch_summarise_edge_detection_entries ────────────────────────────────────

class TestBatchSummariseEdgeDetectionEntriesExtra:
    def test_returns_list(self):
        assert isinstance(batch_summarise_edge_detection_entries([_ed_entries(2)]), list)

    def test_length_matches(self):
        result = batch_summarise_edge_detection_entries([_ed_entries(2), _ed_entries(3)])
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_edge_detection_entries([]) == []
