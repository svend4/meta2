"""Extra tests for puzzle_reconstruction/utils/placement_metrics_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _metrics(n_placed=5, n_total=10, density=0.5, coverage=0.4,
             pairwise_overlap=0.0, quality=0.5) -> PlacementMetrics:
    return PlacementMetrics(n_placed=n_placed, n_total=n_total,
                             density=density, coverage=coverage,
                             pairwise_overlap=pairwise_overlap,
                             quality_score=quality)


def _square_contour(size=10.0):
    return np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)


# ─── PlacementMetricsConfig ───────────────────────────────────────────────────

class TestPlacementMetricsConfigExtra:
    def test_default_weights(self):
        cfg = PlacementMetricsConfig()
        assert cfg.w_density == pytest.approx(0.4)
        assert cfg.w_coverage == pytest.approx(0.4)
        assert cfg.w_overlap == pytest.approx(0.2)

    def test_default_canvas_size(self):
        assert PlacementMetricsConfig().canvas_size == (512, 512)

    def test_custom_canvas_size(self):
        cfg = PlacementMetricsConfig(canvas_size=(256, 256))
        assert cfg.canvas_size == (256, 256)


# ─── PlacementMetrics ─────────────────────────────────────────────────────────

class TestPlacementMetricsExtra:
    def test_repr_contains_quality(self):
        m = _metrics()
        assert "quality=" in repr(m)

    def test_fields_stored(self):
        m = _metrics(n_placed=3, n_total=5)
        assert m.n_placed == 3 and m.n_total == 5


# ─── placement_density ────────────────────────────────────────────────────────

class TestPlacementDensityExtra:
    def test_zero_total_returns_zero(self):
        assert placement_density(0, 0) == pytest.approx(0.0)

    def test_full_placement(self):
        assert placement_density(10, 10) == pytest.approx(1.0)

    def test_partial_placement(self):
        assert placement_density(3, 10) == pytest.approx(0.3)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            placement_density(-1, 10)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            placement_density(5, -1)

    def test_placed_exceeds_total_clamped(self):
        assert placement_density(15, 10) == pytest.approx(1.0)


# ─── bbox_of_contour ─────────────────────────────────────────────────────────

class TestBboxOfContourExtra:
    def test_unit_square(self):
        c = _square_contour(10.0)
        x0, y0, x1, y1 = bbox_of_contour(c)
        assert x0 == pytest.approx(0.0) and y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(10.0) and y1 == pytest.approx(10.0)

    def test_with_position_offset(self):
        c = _square_contour(10.0)
        x0, y0, x1, y1 = bbox_of_contour(c, position=(5.0, 3.0))
        assert x0 == pytest.approx(5.0) and y0 == pytest.approx(3.0)

    def test_empty_contour(self):
        x0, y0, x1, y1 = bbox_of_contour(np.zeros((0, 2)), position=(2.0, 3.0))
        assert x0 == pytest.approx(2.0) and y0 == pytest.approx(3.0)


# ─── bbox_area ────────────────────────────────────────────────────────────────

class TestBboxAreaExtra:
    def test_unit_square(self):
        assert bbox_area((0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)

    def test_rectangle(self):
        assert bbox_area((0.0, 0.0, 3.0, 5.0)) == pytest.approx(15.0)

    def test_degenerate_returns_zero(self):
        assert bbox_area((5.0, 5.0, 3.0, 5.0)) == pytest.approx(0.0)


# ─── bbox_intersection_area ───────────────────────────────────────────────────

class TestBboxIntersectionAreaExtra:
    def test_identical_boxes(self):
        b = (0.0, 0.0, 4.0, 4.0)
        assert bbox_intersection_area(b, b) == pytest.approx(16.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 2.0, 2.0)
        b = (5.0, 5.0, 8.0, 8.0)
        assert bbox_intersection_area(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 3.0, 3.0)
        b = (1.0, 1.0, 4.0, 4.0)
        assert bbox_intersection_area(a, b) == pytest.approx(4.0)


# ─── compute_coverage ────────────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_empty_positions_returns_zero(self):
        assert compute_coverage([], [], canvas_size=(100, 100)) == pytest.approx(0.0)

    def test_full_coverage(self):
        c = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        cov = compute_coverage([(0.0, 0.0)], [c], canvas_size=(100, 100))
        assert cov == pytest.approx(1.0)

    def test_partial_coverage(self):
        c = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
        cov = compute_coverage([(0.0, 0.0)], [c], canvas_size=(100, 100))
        assert 0.0 < cov <= 1.0


# ─── compute_pairwise_overlap ────────────────────────────────────────────────

class TestComputePairwiseOverlapExtra:
    def test_single_fragment_zero_overlap(self):
        c = _square_contour()
        assert compute_pairwise_overlap([(0, 0)], [c]) == pytest.approx(0.0)

    def test_no_overlap(self):
        c = _square_contour(10.0)
        assert compute_pairwise_overlap([(0, 0), (100, 100)], [c, c]) == pytest.approx(0.0)

    def test_full_overlap(self):
        c = _square_contour(10.0)
        overlap = compute_pairwise_overlap([(0, 0), (0, 0)], [c, c])
        assert overlap == pytest.approx(100.0)


# ─── quality_score ────────────────────────────────────────────────────────────

class TestQualityScoreExtra:
    def test_no_overlap_perfect(self):
        qs = quality_score(1.0, 1.0, 0.0)
        assert qs == pytest.approx(1.0)

    def test_zero_density_and_coverage(self):
        # zero density/coverage but zero overlap → overlap_score=1.0 contributes
        qs = quality_score(0.0, 0.0, 0.0)
        assert 0.0 <= qs <= 1.0

    def test_result_in_unit_range(self):
        qs = quality_score(0.5, 0.5, 500.0)
        assert 0.0 <= qs <= 1.0

    def test_zero_weights_returns_zero(self):
        qs = quality_score(0.8, 0.8, 0.0, w_density=0.0, w_coverage=0.0, w_overlap=0.0)
        assert qs == pytest.approx(0.0)


# ─── assess_placement ────────────────────────────────────────────────────────

class TestAssessPlacementExtra:
    def test_returns_placement_metrics(self):
        c = _square_contour(50.0)
        m = assess_placement([(0.0, 0.0)], [c], n_total=5)
        assert isinstance(m, PlacementMetrics)

    def test_density_correct(self):
        c = _square_contour(10.0)
        m = assess_placement([(0.0, 0.0)], [c], n_total=10)
        assert m.density == pytest.approx(0.1)

    def test_no_fragments(self):
        m = assess_placement([], [], n_total=5)
        assert m.density == pytest.approx(0.0)


# ─── compare_metrics ─────────────────────────────────────────────────────────

class TestCompareMetricsExtra:
    def test_returns_dict(self):
        m = _metrics()
        d = compare_metrics(m, m)
        assert isinstance(d, dict)

    def test_identical_better_a(self):
        m = _metrics(quality=0.5)
        d = compare_metrics(m, m)
        assert d["better"] == "a"

    def test_b_better(self):
        a = _metrics(quality=0.3)
        b = _metrics(quality=0.8)
        d = compare_metrics(a, b)
        assert d["better"] == "b"


# ─── best_of ─────────────────────────────────────────────────────────────────

class TestBestOfExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            best_of([])

    def test_returns_best_index(self):
        ml = [_metrics(quality=0.3), _metrics(quality=0.9), _metrics(quality=0.1)]
        assert best_of(ml) == 1


# ─── normalize_metrics ───────────────────────────────────────────────────────

class TestNormalizeMetricsExtra:
    def test_empty_returns_empty(self):
        assert normalize_metrics([]) == []

    def test_constant_scores_normalized_to_one(self):
        ml = [_metrics(quality=0.5), _metrics(quality=0.5)]
        norm = normalize_metrics(ml)
        assert all(m.quality_score == pytest.approx(1.0) for m in norm)

    def test_min_max_normalized(self):
        ml = [_metrics(quality=0.0), _metrics(quality=1.0)]
        norm = normalize_metrics(ml)
        scores = [m.quality_score for m in norm]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)


# ─── batch_quality_scores ────────────────────────────────────────────────────

class TestBatchQualityScoresExtra:
    def test_length_preserved(self):
        ml = [_metrics(), _metrics(), _metrics()]
        scores = batch_quality_scores(ml)
        assert len(scores) == 3

    def test_scores_in_unit_range(self):
        ml = [_metrics(density=0.5, coverage=0.5, pairwise_overlap=0.0)]
        scores = batch_quality_scores(ml)
        assert all(0.0 <= s <= 1.0 for s in scores)
