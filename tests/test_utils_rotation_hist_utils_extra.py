"""Extra tests for puzzle_reconstruction/utils/rotation_hist_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.rotation_hist_utils import (
    RotationAnalysisConfig,
    RotationAnalysisEntry,
    RotationAnalysisSummary,
    HistogramDistanceConfig,
    HistogramDistanceEntry,
    HistogramDistanceSummary,
    make_rotation_analysis_entry,
    summarise_rotation_analysis,
    filter_rotation_by_confidence,
    filter_rotation_by_method,
    filter_rotation_by_angle_range,
    top_k_rotation_entries,
    best_rotation_entry,
    rotation_angle_stats,
    compare_rotation_summaries,
    batch_summarise_rotation_analysis,
    make_histogram_distance_entry,
    summarise_histogram_distance_entries,
    filter_histogram_by_max_distance,
    filter_histogram_by_metric,
    filter_histogram_by_fragment,
    top_k_closest_histogram_pairs,
    best_histogram_distance_entry,
    histogram_distance_stats,
    compare_histogram_distance_summaries,
    batch_summarise_histogram_distance_entries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rot(frag_id=0, angle=45.0, confidence=0.8,
         method="procrustes") -> RotationAnalysisEntry:
    return RotationAnalysisEntry(fragment_id=frag_id, angle_deg=angle,
                                  confidence=confidence, method=method)


def _hist(a=0, b=1, dist=0.3, metric="emd") -> HistogramDistanceEntry:
    return HistogramDistanceEntry(frag_a=a, frag_b=b,
                                   distance=dist, metric=metric)


# ─── RotationAnalysisConfig ───────────────────────────────────────────────────

class TestRotationAnalysisConfigExtra:
    def test_defaults(self):
        cfg = RotationAnalysisConfig()
        assert cfg.min_confidence == pytest.approx(0.0)
        assert cfg.discrete_steps == 4

    def test_custom(self):
        cfg = RotationAnalysisConfig(min_confidence=0.5, discrete_steps=8)
        assert cfg.discrete_steps == 8


# ─── RotationAnalysisEntry ────────────────────────────────────────────────────

class TestRotationAnalysisEntryExtra:
    def test_fields_stored(self):
        e = _rot(frag_id=3, angle=90.0)
        assert e.fragment_id == 3 and e.angle_deg == pytest.approx(90.0)

    def test_params_default_empty(self):
        assert _rot().params == {}


# ─── make_rotation_analysis_entry ────────────────────────────────────────────

class TestMakeRotationAnalysisEntryExtra:
    def test_returns_entry(self):
        e = make_rotation_analysis_entry(0, 45.0, 0.9)
        assert isinstance(e, RotationAnalysisEntry)

    def test_params_stored(self):
        e = make_rotation_analysis_entry(0, 0.0, 0.5, method="phase", k=3)
        assert e.params["k"] == 3


# ─── summarise_rotation_analysis ─────────────────────────────────────────────

class TestSummariseRotationAnalysisExtra:
    def test_empty(self):
        s = summarise_rotation_analysis([])
        assert s.n_entries == 0 and s.methods_used == []

    def test_mean_angle(self):
        entries = [_rot(angle=30.0), _rot(angle=90.0)]
        s = summarise_rotation_analysis(entries)
        assert s.mean_angle_deg == pytest.approx(60.0)

    def test_methods_used(self):
        entries = [_rot(method="procrustes"), _rot(method="phase")]
        s = summarise_rotation_analysis(entries)
        assert "procrustes" in s.methods_used and "phase" in s.methods_used

    def test_dominant_angle_computed(self):
        entries = [_rot(angle=0.0), _rot(angle=0.0), _rot(angle=90.0)]
        s = summarise_rotation_analysis(entries)
        assert s.dominant_angle == pytest.approx(0.0)


# ─── filters ─────────────────────────────────────────────────────────────────

class TestFilterRotationExtra:
    def test_by_confidence(self):
        entries = [_rot(confidence=0.3), _rot(confidence=0.8)]
        assert len(filter_rotation_by_confidence(entries, 0.5)) == 1

    def test_by_method(self):
        entries = [_rot(method="a"), _rot(method="b")]
        assert len(filter_rotation_by_method(entries, "a")) == 1

    def test_by_angle_range(self):
        entries = [_rot(angle=10.0), _rot(angle=50.0), _rot(angle=100.0)]
        assert len(filter_rotation_by_angle_range(entries, 20.0, 80.0)) == 1


# ─── top_k / best / stats ────────────────────────────────────────────────────

class TestRankRotationExtra:
    def test_top_k(self):
        entries = [_rot(confidence=0.3), _rot(confidence=0.9), _rot(confidence=0.6)]
        top = top_k_rotation_entries(entries, 2)
        assert top[0].confidence == pytest.approx(0.9) and len(top) == 2

    def test_best_entry(self):
        entries = [_rot(confidence=0.4), _rot(confidence=0.95)]
        assert best_rotation_entry(entries).confidence == pytest.approx(0.95)

    def test_best_empty(self):
        assert best_rotation_entry([]) is None

    def test_angle_stats_empty(self):
        s = rotation_angle_stats([])
        assert s["count"] == 0

    def test_angle_stats(self):
        entries = [_rot(angle=30.0), _rot(angle=90.0)]
        s = rotation_angle_stats(entries)
        assert s["min"] == pytest.approx(30.0)
        assert s["max"] == pytest.approx(90.0)


# ─── compare / batch rotation ────────────────────────────────────────────────

class TestCompareRotationExtra:
    def test_same_summaries(self):
        s = summarise_rotation_analysis([_rot()])
        d = compare_rotation_summaries(s, s)
        assert d["same_dominant_angle"] is True

    def test_batch_length(self):
        result = batch_summarise_rotation_analysis([[_rot()], []])
        assert len(result) == 2


# ─── HistogramDistanceConfig ──────────────────────────────────────────────────

class TestHistogramDistanceConfigExtra:
    def test_defaults(self):
        cfg = HistogramDistanceConfig()
        assert cfg.max_distance == pytest.approx(1.0) and cfg.metric == "emd"


# ─── HistogramDistanceEntry ──────────────────────────────────────────────────

class TestHistogramDistanceEntryExtra:
    def test_fields_stored(self):
        e = _hist(a=2, b=5, dist=0.7)
        assert e.frag_a == 2 and e.distance == pytest.approx(0.7)

    def test_params_default_empty(self):
        assert _hist().params == {}


# ─── make / summarise histogram distance ─────────────────────────────────────

class TestHistogramDistanceExtra:
    def test_make_returns_entry(self):
        e = make_histogram_distance_entry(0, 1, 0.5)
        assert isinstance(e, HistogramDistanceEntry)

    def test_summarise_empty(self):
        s = summarise_histogram_distance_entries([])
        assert s.n_pairs == 0

    def test_summarise_stats(self):
        entries = [_hist(dist=0.2), _hist(dist=0.8)]
        s = summarise_histogram_distance_entries(entries)
        assert s.mean_distance == pytest.approx(0.5)
        assert s.min_distance == pytest.approx(0.2)


# ─── filters histogram ───────────────────────────────────────────────────────

class TestFilterHistogramExtra:
    def test_by_max_distance(self):
        entries = [_hist(dist=0.2), _hist(dist=0.8)]
        assert len(filter_histogram_by_max_distance(entries, 0.5)) == 1

    def test_by_metric(self):
        entries = [_hist(metric="emd"), _hist(metric="chi2")]
        assert len(filter_histogram_by_metric(entries, "emd")) == 1

    def test_by_fragment(self):
        entries = [_hist(a=0, b=1), _hist(a=2, b=3)]
        assert len(filter_histogram_by_fragment(entries, 0)) == 1


# ─── top_k / best / stats histogram ──────────────────────────────────────────

class TestRankHistogramExtra:
    def test_top_k_closest(self):
        entries = [_hist(dist=0.8), _hist(dist=0.1), _hist(dist=0.5)]
        top = top_k_closest_histogram_pairs(entries, 2)
        assert top[0].distance == pytest.approx(0.1)

    def test_best_entry(self):
        entries = [_hist(dist=0.5), _hist(dist=0.1)]
        assert best_histogram_distance_entry(entries).distance == pytest.approx(0.1)

    def test_best_empty(self):
        assert best_histogram_distance_entry([]) is None

    def test_stats_empty(self):
        s = histogram_distance_stats([])
        assert s["count"] == 0


# ─── compare / batch histogram ────────────────────────────────────────────────

class TestCompareHistogramExtra:
    def test_same_metric(self):
        s = summarise_histogram_distance_entries([_hist()])
        d = compare_histogram_distance_summaries(s, s)
        assert d["same_metric"] is True

    def test_batch(self):
        result = batch_summarise_histogram_distance_entries([[_hist()], []])
        assert len(result) == 2
