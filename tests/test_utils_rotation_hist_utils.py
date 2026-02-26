"""Tests for puzzle_reconstruction.utils.rotation_hist_utils"""
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_rotation_entries():
    return [
        make_rotation_analysis_entry(0, 45.0, 0.9, "procrustes"),
        make_rotation_analysis_entry(1, 90.0, 0.7, "procrustes"),
        make_rotation_analysis_entry(2, 45.0, 0.85, "phase"),
    ]


def make_hist_entries():
    return [
        make_histogram_distance_entry(0, 1, 0.2, "emd"),
        make_histogram_distance_entry(1, 2, 0.5, "emd"),
        make_histogram_distance_entry(2, 3, 0.1, "chi2"),
    ]


# ─── RotationAnalysisConfig ───────────────────────────────────────────────────

def test_rotation_analysis_config_defaults():
    cfg = RotationAnalysisConfig()
    assert cfg.min_confidence == 0.0
    assert cfg.discrete_steps == 4
    assert cfg.angle_tolerance_deg == 5.0


# ─── make_rotation_analysis_entry ─────────────────────────────────────────────

def test_make_rotation_analysis_entry():
    e = make_rotation_analysis_entry(5, 30.0, 0.8, "phase", key="v")
    assert e.fragment_id == 5
    assert isinstance(e.angle_deg, float)
    assert e.params == {"key": "v"}


def test_make_rotation_entry_default_method():
    e = make_rotation_analysis_entry(0, 0.0, 1.0)
    assert e.method == "procrustes"


# ─── summarise_rotation_analysis ─────────────────────────────────────────────

def test_summarise_rotation_analysis_empty():
    s = summarise_rotation_analysis([])
    assert s.n_entries == 0
    assert s.methods_used == []
    assert s.dominant_angle == 0.0


def test_summarise_rotation_analysis_basic():
    entries = make_rotation_entries()
    s = summarise_rotation_analysis(entries)
    assert s.n_entries == 3
    assert abs(s.mean_angle_deg - (45 + 90 + 45) / 3) < 1e-9
    assert s.std_angle_deg >= 0


def test_summarise_rotation_analysis_methods_used():
    entries = make_rotation_entries()
    s = summarise_rotation_analysis(entries)
    assert set(s.methods_used) == {"procrustes", "phase"}


def test_summarise_rotation_dominant_angle():
    # Use angles that clearly cluster in the 90-degree bin (80, 90, 100)
    entries_90 = [
        make_rotation_analysis_entry(0, 80.0, 0.9, "procrustes"),
        make_rotation_analysis_entry(1, 90.0, 0.7, "procrustes"),
        make_rotation_analysis_entry(2, 100.0, 0.85, "phase"),
    ]
    s = summarise_rotation_analysis(entries_90)
    assert s.dominant_angle == pytest.approx(90.0, abs=1.0)


# ─── filter_rotation_by_confidence ───────────────────────────────────────────

def test_filter_rotation_by_confidence():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_confidence(entries, 0.85)
    assert len(filtered) == 2
    assert all(e.confidence >= 0.85 for e in filtered)


def test_filter_rotation_by_confidence_zero():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_confidence(entries, 0.0)
    assert len(filtered) == 3


# ─── filter_rotation_by_method ───────────────────────────────────────────────

def test_filter_rotation_by_method():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_method(entries, "procrustes")
    assert len(filtered) == 2


def test_filter_rotation_by_method_nonexistent():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_method(entries, "unknown")
    assert len(filtered) == 0


# ─── filter_rotation_by_angle_range ──────────────────────────────────────────

def test_filter_rotation_by_angle_range():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_angle_range(entries, 40.0, 50.0)
    assert len(filtered) == 2


def test_filter_rotation_by_angle_range_none():
    entries = make_rotation_entries()
    filtered = filter_rotation_by_angle_range(entries, 200.0, 300.0)
    assert len(filtered) == 0


# ─── top_k_rotation_entries ───────────────────────────────────────────────────

def test_top_k_rotation_entries():
    entries = make_rotation_entries()
    top = top_k_rotation_entries(entries, 2)
    assert len(top) == 2
    assert top[0].confidence >= top[1].confidence


def test_top_k_rotation_entries_all():
    entries = make_rotation_entries()
    top = top_k_rotation_entries(entries, 10)
    assert len(top) == 3


# ─── best_rotation_entry ──────────────────────────────────────────────────────

def test_best_rotation_entry():
    entries = make_rotation_entries()
    best = best_rotation_entry(entries)
    assert best.fragment_id == 0  # confidence=0.9


def test_best_rotation_entry_empty():
    assert best_rotation_entry([]) is None


# ─── rotation_angle_stats ─────────────────────────────────────────────────────

def test_rotation_angle_stats_empty():
    d = rotation_angle_stats([])
    assert d["count"] == 0


def test_rotation_angle_stats_basic():
    entries = make_rotation_entries()
    d = rotation_angle_stats(entries)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(45.0)
    assert d["max"] == pytest.approx(90.0)
    assert d["std"] >= 0


# ─── compare_rotation_summaries ───────────────────────────────────────────────

def test_compare_rotation_summaries():
    entries = make_rotation_entries()
    a = summarise_rotation_analysis(entries[:2])
    b = summarise_rotation_analysis(entries)
    diff = compare_rotation_summaries(a, b)
    assert "delta_mean_angle_deg" in diff
    assert "same_dominant_angle" in diff


# ─── batch_summarise_rotation_analysis ───────────────────────────────────────

def test_batch_summarise_rotation():
    entries = make_rotation_entries()
    result = batch_summarise_rotation_analysis([entries[:2], entries[2:]])
    assert len(result) == 2
    assert result[0].n_entries == 2


# ─── make_histogram_distance_entry ────────────────────────────────────────────

def test_make_histogram_distance_entry():
    e = make_histogram_distance_entry(0, 1, 0.3, "emd", channel="gray")
    assert e.frag_a == 0
    assert e.frag_b == 1
    assert isinstance(e.distance, float)
    assert e.params == {"channel": "gray"}


# ─── summarise_histogram_distance_entries ─────────────────────────────────────

def test_summarise_histogram_distance_empty():
    cfg = HistogramDistanceConfig()
    s = summarise_histogram_distance_entries([], cfg)
    assert s.n_pairs == 0


def test_summarise_histogram_distance_basic():
    entries = make_hist_entries()
    s = summarise_histogram_distance_entries(entries)
    assert s.n_pairs == 3
    assert s.min_distance == pytest.approx(0.1)
    assert s.max_distance == pytest.approx(0.5)
    assert abs(s.mean_distance - (0.2 + 0.5 + 0.1) / 3) < 1e-9


# ─── filter_histogram_by_max_distance ─────────────────────────────────────────

def test_filter_histogram_by_max_distance():
    entries = make_hist_entries()
    filtered = filter_histogram_by_max_distance(entries, 0.25)
    assert len(filtered) == 2


# ─── filter_histogram_by_metric ───────────────────────────────────────────────

def test_filter_histogram_by_metric():
    entries = make_hist_entries()
    filtered = filter_histogram_by_metric(entries, "emd")
    assert len(filtered) == 2


def test_filter_histogram_by_metric_none():
    entries = make_hist_entries()
    filtered = filter_histogram_by_metric(entries, "intersection")
    assert len(filtered) == 0


# ─── filter_histogram_by_fragment ────────────────────────────────────────────

def test_filter_histogram_by_fragment():
    entries = make_hist_entries()
    filtered = filter_histogram_by_fragment(entries, 1)
    # Entry (0,1) and (1,2) both contain frag 1
    assert len(filtered) == 2


# ─── top_k_closest_histogram_pairs ────────────────────────────────────────────

def test_top_k_closest_histogram_pairs():
    entries = make_hist_entries()
    top = top_k_closest_histogram_pairs(entries, 2)
    assert len(top) == 2
    assert top[0].distance <= top[1].distance


# ─── best_histogram_distance_entry ────────────────────────────────────────────

def test_best_histogram_distance_entry():
    entries = make_hist_entries()
    best = best_histogram_distance_entry(entries)
    assert best.distance == pytest.approx(0.1)


def test_best_histogram_distance_entry_empty():
    assert best_histogram_distance_entry([]) is None


# ─── histogram_distance_stats ─────────────────────────────────────────────────

def test_histogram_distance_stats_empty():
    d = histogram_distance_stats([])
    assert d["count"] == 0


def test_histogram_distance_stats_basic():
    entries = make_hist_entries()
    d = histogram_distance_stats(entries)
    assert d["count"] == 3
    assert d["std"] >= 0


# ─── compare_histogram_distance_summaries ─────────────────────────────────────

def test_compare_histogram_distance_summaries():
    entries = make_hist_entries()
    a = summarise_histogram_distance_entries(entries[:2])
    b = summarise_histogram_distance_entries(entries)
    diff = compare_histogram_distance_summaries(a, b)
    assert "delta_mean_distance" in diff
    assert "same_metric" in diff


# ─── batch_summarise_histogram_distance_entries ───────────────────────────────

def test_batch_summarise_histogram_distance():
    entries = make_hist_entries()
    result = batch_summarise_histogram_distance_entries([entries[:2], entries[2:]])
    assert len(result) == 2
