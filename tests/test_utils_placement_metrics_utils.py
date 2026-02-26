"""Tests for puzzle_reconstruction.utils.placement_metrics_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.placement_metrics_utils import (
    PlacementMetricsConfig,
    PlacementMetrics,
    placement_density,
    bbox_of_contour,
    bbox_area,
    bbox_intersection_area,
    compute_coverage,
    compute_pairwise_overlap,
    quality_score,
    assess_placement,
    compare_metrics,
    best_of,
    normalize_metrics,
    batch_quality_scores,
)

np.random.seed(42)


# ─── PlacementMetricsConfig ───────────────────────────────────────────────────

def test_config_defaults():
    cfg = PlacementMetricsConfig()
    assert cfg.w_density == pytest.approx(0.4)
    assert cfg.w_coverage == pytest.approx(0.4)
    assert cfg.w_overlap == pytest.approx(0.2)
    assert cfg.canvas_size == (512, 512)


def test_config_custom():
    cfg = PlacementMetricsConfig(w_density=0.5, w_coverage=0.3, w_overlap=0.2,
                                  canvas_size=(256, 256))
    assert cfg.canvas_size == (256, 256)
    assert cfg.w_density == pytest.approx(0.5)


# ─── placement_density ───────────────────────────────────────────────────────

def test_placement_density_basic():
    assert placement_density(5, 10) == pytest.approx(0.5)


def test_placement_density_all_placed():
    assert placement_density(10, 10) == pytest.approx(1.0)


def test_placement_density_none_placed():
    assert placement_density(0, 10) == pytest.approx(0.0)


def test_placement_density_zero_total():
    assert placement_density(0, 0) == pytest.approx(0.0)


def test_placement_density_exceeds_total():
    # clamp to 1.0
    assert placement_density(15, 10) == pytest.approx(1.0)


def test_placement_density_negative_raises():
    with pytest.raises(ValueError):
        placement_density(-1, 10)


def test_placement_density_negative_total_raises():
    with pytest.raises(ValueError):
        placement_density(5, -1)


# ─── bbox_of_contour ─────────────────────────────────────────────────────────

def test_bbox_of_contour_basic():
    contour = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64)
    x_min, y_min, x_max, y_max = bbox_of_contour(contour)
    assert x_min == pytest.approx(0.0)
    assert y_min == pytest.approx(0.0)
    assert x_max == pytest.approx(10.0)
    assert y_max == pytest.approx(20.0)


def test_bbox_of_contour_with_offset():
    contour = np.array([[0, 0], [10, 10]], dtype=np.float64)
    x_min, y_min, x_max, y_max = bbox_of_contour(contour, position=(5.0, 3.0))
    assert x_min == pytest.approx(5.0)
    assert y_min == pytest.approx(3.0)
    assert x_max == pytest.approx(15.0)
    assert y_max == pytest.approx(13.0)


def test_bbox_of_contour_empty():
    result = bbox_of_contour(np.array([]).reshape(0, 2), position=(7.0, 8.0))
    assert result == (7.0, 8.0, 7.0, 8.0)


# ─── bbox_area ───────────────────────────────────────────────────────────────

def test_bbox_area_basic():
    assert bbox_area((0.0, 0.0, 10.0, 20.0)) == pytest.approx(200.0)


def test_bbox_area_zero_width():
    assert bbox_area((5.0, 0.0, 5.0, 10.0)) == pytest.approx(0.0)


def test_bbox_area_inverted_coords():
    # x_max < x_min → area = 0
    assert bbox_area((10.0, 0.0, 5.0, 10.0)) == pytest.approx(0.0)


# ─── bbox_intersection_area ──────────────────────────────────────────────────

def test_bbox_intersection_area_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert bbox_intersection_area(a, b) == pytest.approx(25.0)


def test_bbox_intersection_area_no_overlap():
    a = (0.0, 0.0, 5.0, 5.0)
    b = (10.0, 10.0, 20.0, 20.0)
    assert bbox_intersection_area(a, b) == pytest.approx(0.0)


def test_bbox_intersection_area_touching():
    a = (0.0, 0.0, 5.0, 5.0)
    b = (5.0, 0.0, 10.0, 5.0)
    assert bbox_intersection_area(a, b) == pytest.approx(0.0)


# ─── compute_coverage ────────────────────────────────────────────────────────

def test_compute_coverage_no_fragments():
    assert compute_coverage([], [], canvas_size=(100, 100)) == pytest.approx(0.0)


def test_compute_coverage_single_fragment():
    contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    cov = compute_coverage([(0.0, 0.0)], [contour], canvas_size=(100, 100))
    assert 0.0 < cov <= 1.0


def test_compute_coverage_returns_float():
    contour = np.array([[0, 0], [10, 10]], dtype=np.float64)
    result = compute_coverage([(0.0, 0.0)], [contour], canvas_size=(100, 100))
    assert isinstance(result, float)


def test_compute_coverage_range():
    contour = np.array([[0, 0], [100, 100]], dtype=np.float64)
    cov = compute_coverage([(0.0, 0.0)], [contour], canvas_size=(100, 100))
    assert 0.0 <= cov <= 1.0


# ─── compute_pairwise_overlap ────────────────────────────────────────────────

def test_compute_pairwise_overlap_no_overlap():
    c1 = np.array([[0, 0], [10, 10]], dtype=np.float64)
    c2 = np.array([[0, 0], [10, 10]], dtype=np.float64)
    ovlp = compute_pairwise_overlap(
        [(0.0, 0.0), (100.0, 100.0)], [c1, c2]
    )
    assert ovlp == pytest.approx(0.0)


def test_compute_pairwise_overlap_single_fragment():
    c = np.array([[0, 0], [10, 10]], dtype=np.float64)
    assert compute_pairwise_overlap([(0.0, 0.0)], [c]) == pytest.approx(0.0)


def test_compute_pairwise_overlap_returns_nonneg():
    c1 = np.array([[0, 0], [20, 20]], dtype=np.float64)
    c2 = np.array([[0, 0], [20, 20]], dtype=np.float64)
    ovlp = compute_pairwise_overlap([(0.0, 0.0), (5.0, 5.0)], [c1, c2])
    assert ovlp >= 0.0


# ─── quality_score ───────────────────────────────────────────────────────────

def test_quality_score_perfect():
    qs = quality_score(1.0, 1.0, 0.0)
    assert qs == pytest.approx(1.0)


def test_quality_score_zero():
    qs = quality_score(0.0, 0.0, 1e10)
    assert qs == pytest.approx(0.0)


def test_quality_score_range():
    qs = quality_score(0.5, 0.5, 100.0)
    assert 0.0 <= qs <= 1.0


def test_quality_score_zero_weights():
    qs = quality_score(0.5, 0.5, 0.0, w_density=0.0, w_coverage=0.0, w_overlap=0.0)
    assert qs == pytest.approx(0.0)


# ─── assess_placement ────────────────────────────────────────────────────────

def test_assess_placement_returns_metrics():
    contour = np.array([[0, 0], [50, 50]], dtype=np.float64)
    m = assess_placement([(0.0, 0.0)], [contour], n_total=5)
    assert isinstance(m, PlacementMetrics)


def test_assess_placement_density():
    contour = np.array([[0, 0], [10, 10]], dtype=np.float64)
    m = assess_placement([(0.0, 0.0), (100.0, 0.0)], [contour, contour], n_total=4)
    assert m.density == pytest.approx(0.5)


def test_assess_placement_empty():
    m = assess_placement([], [], n_total=10)
    assert m.n_placed == 0
    assert m.density == pytest.approx(0.0)


# ─── compare_metrics ─────────────────────────────────────────────────────────

def test_compare_metrics_keys():
    m1 = PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.6)
    m2 = PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.9)
    result = compare_metrics(m1, m2)
    assert "density_diff" in result
    assert "coverage_diff" in result
    assert "overlap_diff" in result
    assert "quality_diff" in result
    assert "better" in result


def test_compare_metrics_better():
    m1 = PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.6)
    m2 = PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.9)
    result = compare_metrics(m1, m2)
    assert result["better"] == "b"


def test_compare_metrics_equal_quality():
    m = PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.7)
    result = compare_metrics(m, m)
    assert result["better"] == "a"


# ─── best_of ─────────────────────────────────────────────────────────────────

def test_best_of_basic():
    metrics = [
        PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.5),
        PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.9),
        PlacementMetrics(3, 10, 0.3, 0.2, 0.0, 0.3),
    ]
    assert best_of(metrics) == 1


def test_best_of_empty_raises():
    with pytest.raises(ValueError):
        best_of([])


def test_best_of_single():
    m = PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.7)
    assert best_of([m]) == 0


# ─── normalize_metrics ───────────────────────────────────────────────────────

def test_normalize_metrics_range():
    metrics = [
        PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.2),
        PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.8),
    ]
    result = normalize_metrics(metrics)
    scores = [m.quality_score for m in result]
    assert min(scores) == pytest.approx(0.0)
    assert max(scores) == pytest.approx(1.0)


def test_normalize_metrics_constant():
    metrics = [
        PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.7),
        PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.7),
    ]
    result = normalize_metrics(metrics)
    for m in result:
        assert m.quality_score == pytest.approx(1.0)


def test_normalize_metrics_empty():
    assert normalize_metrics([]) == []


def test_normalize_metrics_length_preserved():
    metrics = [PlacementMetrics(i, 10, i/10, i/10, 0.0, i/10) for i in range(5)]
    result = normalize_metrics(metrics)
    assert len(result) == 5


# ─── batch_quality_scores ────────────────────────────────────────────────────

def test_batch_quality_scores_length():
    metrics = [
        PlacementMetrics(5, 10, 0.5, 0.3, 0.0, 0.5),
        PlacementMetrics(8, 10, 0.8, 0.6, 0.0, 0.9),
    ]
    scores = batch_quality_scores(metrics)
    assert len(scores) == 2


def test_batch_quality_scores_range():
    metrics = [PlacementMetrics(5, 10, 0.5, 0.3, 100.0, 0.5)]
    scores = batch_quality_scores(metrics)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_batch_quality_scores_empty():
    assert batch_quality_scores([]) == []
