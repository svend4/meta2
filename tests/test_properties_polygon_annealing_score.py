"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.polygon_ops_utils
  - puzzle_reconstruction.utils.annealing_schedule
  - puzzle_reconstruction.utils.score_aggregator

polygon_ops_utils:
    signed_area:      CCW polygon > 0; CW < 0; area(reversed) = -area(original)
    polygon_area:     >= 0; area(reversed) = area(original)
    polygon_perimeter: >= 0; unchanged by orientation reversal
    polygon_centroid:  within bounding box
    polygon_bounding_box: x0 <= x1; y0 <= y1; all points within box
    polygon_stats:    area >= 0; perimeter >= 0; compactness ∈ [0, 1]
    ensure_ccw:       signed_area >= 0
    ensure_cw:        signed_area <= 0
    polygon_overlap:  iou ∈ [0, 1]; self-iou = 1.0; commutative
    polygon_similarity: ∈ [0, 1]; self = 1.0
    batch_polygon_stats: length preserved
    batch_polygon_overlap: length preserved

annealing_schedule:
    linear_schedule:       len = n_steps; T[0] = t_start; T[-1] ≥ t_end; decreasing
    geometric_schedule:    len = n_steps; T ≥ t_end; T ≤ t_start; decreasing
    exponential_schedule:  len = n_steps; T[0] ≈ t_start; T ≥ t_end
    cosine_schedule:       len = n_steps; T ≥ t_end; T ≤ t_start
    stepped_schedule:      len = n_steps; T ≥ t_end; T ≤ t_start; piecewise constant
    get_temperature:       T ∈ [t_end, t_start]
    estimate_steps:        result >= 1
    batch_temperatures:    length = len(steps); values ≥ t_end

score_aggregator:
    weighted_sum:      ∈ [0, 1]; uniform-weight = arithmetic mean
    harmonic_mean:     ∈ [0, 1]; <= arithmetic mean
    geometric_mean:    ∈ [0, 1]; <= arithmetic mean
    aggregate_pair:    ∈ [0, 1] for all AggregationMethod values
    aggregate_matrix:  shape = (n, n); symmetric; values ∈ [0, 1]
    batch_aggregate:   n_pairs = len(unique keys); mean ∈ [0, 1]
    ScoreVector.mean_score: == mean of channels
    ScoreVector.max_score:  >= mean_score
    ScoreVector.min_score:  <= mean_score
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from puzzle_reconstruction.utils.polygon_ops_utils import (
    PolygonOpsConfig,
    PolygonOverlapResult,
    PolygonStats,
    signed_area,
    polygon_area,
    polygon_perimeter,
    polygon_centroid,
    polygon_bounding_box,
    polygon_stats,
    point_in_polygon,
    polygon_overlap,
    remove_collinear,
    ensure_ccw,
    ensure_cw,
    polygon_similarity,
    batch_polygon_stats,
    batch_polygon_overlap,
)
from puzzle_reconstruction.utils.annealing_schedule import (
    ScheduleConfig,
    TemperatureRecord,
    linear_schedule,
    geometric_schedule,
    exponential_schedule,
    cosine_schedule,
    stepped_schedule,
    get_temperature,
    estimate_steps,
    batch_temperatures,
)
from puzzle_reconstruction.utils.score_aggregator import (
    AggregationMethod,
    ScoreVector,
    AggregationResult,
    weighted_sum,
    harmonic_mean,
    geometric_mean,
    aggregate_pair,
    aggregate_matrix,
    top_k_pairs,
    batch_aggregate,
)

RNG = np.random.default_rng(2033)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_polygon(n: int = 5, scale: float = 10.0, seed: int = 0) -> np.ndarray:
    """Random convex-ish polygon (random points on/inside circle)."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * math.pi, n))
    r = rng.uniform(0.5, 1.0, n) * scale
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def _rect_polygon(x0: float = 0.0, y0: float = 0.0,
                  w: float = 10.0, h: float = 5.0) -> np.ndarray:
    return np.array([
        [x0,     y0],
        [x0 + w, y0],
        [x0 + w, y0 + h],
        [x0,     y0 + h],
    ], dtype=np.float64)


def _make_score_vector(idx_a: int = 0, idx_b: int = 1,
                       channels: Dict[str, float] | None = None,
                       seed: int = 0) -> ScoreVector:
    if channels is None:
        rng = np.random.default_rng(seed)
        channels = {
            "color":    float(rng.uniform(0, 1)),
            "texture":  float(rng.uniform(0, 1)),
            "geometry": float(rng.uniform(0, 1)),
        }
    return ScoreVector(idx_a=idx_a, idx_b=idx_b, channels=channels)


def _make_schedule(kind: str, n_steps: int = 100) -> ScheduleConfig:
    return ScheduleConfig(t_start=100.0, t_end=0.01, n_steps=n_steps, kind=kind)


# ═══════════════════════════════════════════════════════════════════════════════
# polygon_ops_utils — signed_area, polygon_area
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolygonArea:
    """signed_area, polygon_area: orientation and sign invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_polygon_area_nonnegative(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        assert polygon_area(poly) >= 0.0

    def test_signed_area_rectangle_ccw(self) -> None:
        rect = _rect_polygon()
        # CCW rect → positive signed area
        assert signed_area(rect) > 0.0

    def test_signed_area_reversal_negates(self) -> None:
        poly = _rand_polygon()
        sa = signed_area(poly)
        sa_rev = signed_area(poly[::-1])
        assert sa == pytest.approx(-sa_rev, abs=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_area_reversal_unchanged(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        assert polygon_area(poly) == pytest.approx(polygon_area(poly[::-1]))

    def test_less_than_3_points_area_zero(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert signed_area(pts) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# polygon_ops_utils — polygon_perimeter, polygon_centroid
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerimeterCentroid:
    """polygon_perimeter and polygon_centroid invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_perimeter_nonnegative(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        assert polygon_perimeter(poly) >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_perimeter_reversal_unchanged(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        p1 = polygon_perimeter(poly)
        p2 = polygon_perimeter(poly[::-1])
        assert p1 == pytest.approx(p2, abs=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_centroid_within_bbox(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        c = polygon_centroid(poly)
        bbox = polygon_bounding_box(poly)
        assert bbox[0] - 1e-9 <= c[0] <= bbox[2] + 1e-9
        assert bbox[1] - 1e-9 <= c[1] <= bbox[3] + 1e-9

    def test_centroid_shape(self) -> None:
        poly = _rand_polygon()
        c = polygon_centroid(poly)
        assert c.shape == (2,)

    def test_rect_centroid_is_center(self) -> None:
        rect = _rect_polygon(0.0, 0.0, 10.0, 6.0)
        c = polygon_centroid(rect)
        assert c[0] == pytest.approx(5.0, abs=0.1)
        assert c[1] == pytest.approx(3.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# polygon_ops_utils — polygon_bounding_box, polygon_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxAndStats:
    """polygon_bounding_box and polygon_stats invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_bbox_x0_leq_x1(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        x0, y0, x1, y1 = polygon_bounding_box(poly)
        assert x0 <= x1
        assert y0 <= y1

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_all_points_within_bbox(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        x0, y0, x1, y1 = polygon_bounding_box(poly)
        assert np.all(poly[:, 0] >= x0 - 1e-9)
        assert np.all(poly[:, 0] <= x1 + 1e-9)
        assert np.all(poly[:, 1] >= y0 - 1e-9)
        assert np.all(poly[:, 1] <= y1 + 1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_stats_area_nonneg(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        stats = polygon_stats(poly)
        assert stats.area >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_stats_perimeter_nonneg(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        stats = polygon_stats(poly)
        assert stats.perimeter >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_stats_compactness_in_range(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        stats = polygon_stats(poly)
        assert 0.0 <= stats.compactness <= 1.0 + 1e-9

    def test_batch_polygon_stats_length(self) -> None:
        polys = [_rand_polygon(seed=i) for i in range(4)]
        results = batch_polygon_stats(polys)
        assert len(results) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# polygon_ops_utils — ensure_ccw, ensure_cw, polygon_overlap, polygon_similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrientationAndOverlap:
    """ensure_ccw, ensure_cw, polygon_overlap, polygon_similarity."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_ensure_ccw_positive_area(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        ccw = ensure_ccw(poly)
        assert signed_area(ccw) >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_ensure_cw_negative_area(self, seed: int) -> None:
        poly = _rand_polygon(seed=seed)
        cw = ensure_cw(poly)
        assert signed_area(cw) <= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_overlap_iou_in_range(self, seed: int) -> None:
        a = _rand_polygon(seed=seed)
        b = _rand_polygon(seed=seed + 100)
        result = polygon_overlap(a, b)
        assert 0.0 <= result.iou <= 1.0 + 1e-9

    def test_polygon_self_overlap_iou_is_one(self) -> None:
        poly = _rect_polygon()
        result = polygon_overlap(poly, poly)
        assert result.iou == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_polygon_overlap_commutative(self, seed: int) -> None:
        a = _rand_polygon(seed=seed)
        b = _rand_polygon(seed=seed + 200)
        r1 = polygon_overlap(a, b)
        r2 = polygon_overlap(b, a)
        assert r1.iou == pytest.approx(r2.iou)

    def test_non_overlapping_iou_zero(self) -> None:
        a = _rect_polygon(0, 0, 5, 5)
        b = _rect_polygon(100, 100, 5, 5)
        result = polygon_overlap(a, b)
        assert result.iou == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_polygon_similarity_in_range(self, seed: int) -> None:
        a = _rand_polygon(seed=seed)
        b = _rand_polygon(seed=seed + 300)
        sim = polygon_similarity(a, b)
        assert 0.0 <= sim <= 1.0 + 1e-9

    def test_polygon_self_similarity_is_one(self) -> None:
        poly = _rect_polygon()
        sim = polygon_similarity(poly, poly)
        assert sim == pytest.approx(1.0)

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_batch_polygon_overlap_length(self, n_pairs: int) -> None:
        polys_a = [_rand_polygon(seed=i) for i in range(n_pairs)]
        polys_b = [_rand_polygon(seed=i + 50) for i in range(n_pairs)]
        results = batch_polygon_overlap(polys_a, polys_b)
        assert len(results) == n_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# annealing_schedule — schedule length and monotonicity
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnealingSchedules:
    """linear/geometric/exponential/cosine/stepped schedule invariants."""

    @pytest.mark.parametrize("kind,n", [
        ("linear", 50), ("geometric", 100), ("exponential", 80),
        ("cosine", 60), ("stepped", 100),
    ])
    def test_schedule_length(self, kind: str, n: int) -> None:
        cfg = _make_schedule(kind, n_steps=n)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule, "cosine": cosine_schedule,
              "stepped": stepped_schedule}[kind]
        records = fn(cfg)
        assert len(records) == n

    @pytest.mark.parametrize("kind", ["linear", "geometric", "exponential",
                                       "cosine", "stepped"])
    def test_first_temp_approx_t_start(self, kind: str) -> None:
        cfg = _make_schedule(kind, n_steps=100)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule, "cosine": cosine_schedule,
              "stepped": stepped_schedule}[kind]
        records = fn(cfg)
        assert records[0].temperature == pytest.approx(cfg.t_start, rel=0.05)

    @pytest.mark.parametrize("kind", ["linear", "geometric", "exponential",
                                       "cosine", "stepped"])
    def test_all_temps_geq_t_end(self, kind: str) -> None:
        cfg = _make_schedule(kind, n_steps=100)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule, "cosine": cosine_schedule,
              "stepped": stepped_schedule}[kind]
        records = fn(cfg)
        for r in records:
            assert r.temperature >= cfg.t_end - 1e-9

    @pytest.mark.parametrize("kind", ["linear", "geometric", "exponential",
                                       "cosine", "stepped"])
    def test_all_temps_leq_t_start(self, kind: str) -> None:
        cfg = _make_schedule(kind, n_steps=100)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule, "cosine": cosine_schedule,
              "stepped": stepped_schedule}[kind]
        records = fn(cfg)
        for r in records:
            assert r.temperature <= cfg.t_start + 1e-9

    @pytest.mark.parametrize("kind", ["linear", "geometric", "exponential"])
    def test_monotone_decreasing(self, kind: str) -> None:
        cfg = _make_schedule(kind, n_steps=50)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule}[kind]
        records = fn(cfg)
        temps = [r.temperature for r in records]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1] - 1e-9

    @pytest.mark.parametrize("kind", ["linear", "geometric", "exponential",
                                       "cosine", "stepped"])
    def test_progress_in_range(self, kind: str) -> None:
        cfg = _make_schedule(kind, n_steps=20)
        fn = {"linear": linear_schedule, "geometric": geometric_schedule,
              "exponential": exponential_schedule, "cosine": cosine_schedule,
              "stepped": stepped_schedule}[kind]
        records = fn(cfg)
        for r in records:
            assert 0.0 <= r.progress <= 1.0 + 1e-9

    @pytest.mark.parametrize("kind,step", [
        ("linear", 0), ("linear", 49), ("geometric", 25), ("exponential", 10)
    ])
    def test_get_temperature_in_range(self, kind: str, step: int) -> None:
        cfg = _make_schedule(kind, n_steps=50)
        t = get_temperature(step, cfg)
        assert cfg.t_end - 1e-9 <= t <= cfg.t_start + 1e-9

    @pytest.mark.parametrize("t_start,t_target,alpha", [
        (100.0, 0.01, 0.9), (10.0, 0.1, 0.95), (50.0, 1.0, 0.8)
    ])
    def test_estimate_steps_geq_one(self, t_start: float, t_target: float,
                                    alpha: float) -> None:
        result = estimate_steps(t_start, t_target, alpha)
        assert result >= 1

    def test_batch_temperatures_length(self) -> None:
        cfg = _make_schedule("geometric", n_steps=100)
        steps = np.array([0, 25, 50, 75, 99])
        temps = batch_temperatures(steps, cfg)
        assert len(temps) == 5

    def test_batch_temperatures_geq_t_end(self) -> None:
        cfg = _make_schedule("linear", n_steps=100)
        steps = np.arange(10)
        temps = batch_temperatures(steps, cfg)
        assert np.all(temps >= cfg.t_end - 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# score_aggregator — weighted_sum, harmonic_mean, geometric_mean
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregationFunctions:
    """weighted_sum, harmonic_mean, geometric_mean invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_weighted_sum_in_range(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        result = weighted_sum(sv.channels)
        assert 0.0 <= result <= 1.0

    def test_weighted_sum_equal_weights_is_mean(self) -> None:
        channels = {"a": 0.2, "b": 0.8, "c": 0.5}
        result = weighted_sum(channels)
        expected = sum(channels.values()) / len(channels)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_harmonic_mean_in_range(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        result = harmonic_mean(sv.channels)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_harmonic_leq_arithmetic(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        h = harmonic_mean(sv.channels)
        arith = sum(sv.channels.values()) / len(sv.channels)
        assert h <= arith + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_geometric_mean_in_range(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        result = geometric_mean(sv.channels)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_geometric_leq_arithmetic(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        g = geometric_mean(sv.channels)
        arith = sum(sv.channels.values()) / len(sv.channels)
        assert g <= arith + 1e-9

    def test_equal_channels_all_means_equal(self) -> None:
        channels = {"a": 0.5, "b": 0.5, "c": 0.5}
        w = weighted_sum(channels)
        h = harmonic_mean(channels)
        g = geometric_mean(channels)
        assert w == pytest.approx(0.5, abs=1e-6)
        assert h == pytest.approx(0.5, abs=1e-6)
        assert g == pytest.approx(0.5, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# score_aggregator — aggregate_pair
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregatePair:
    """aggregate_pair: output ∈ [0, 1] for all methods."""

    @pytest.mark.parametrize("method", list(AggregationMethod))
    def test_aggregate_pair_in_range(self, method: AggregationMethod) -> None:
        sv = _make_score_vector(seed=42)
        result = aggregate_pair(sv, method)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_all_methods_return_scalar(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        for method in AggregationMethod:
            result = aggregate_pair(sv, method)
            assert isinstance(result, float)

    def test_max_method_geq_weighted(self) -> None:
        sv = _make_score_vector(seed=7)
        max_score = aggregate_pair(sv, AggregationMethod.MAX)
        wt_score = aggregate_pair(sv, AggregationMethod.WEIGHTED)
        assert max_score >= wt_score - 1e-9

    def test_min_method_leq_weighted(self) -> None:
        sv = _make_score_vector(seed=7)
        min_score = aggregate_pair(sv, AggregationMethod.MIN)
        wt_score = aggregate_pair(sv, AggregationMethod.WEIGHTED)
        assert min_score <= wt_score + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# score_aggregator — aggregate_matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregateMatrix:
    """aggregate_matrix: shape, symmetry, value bounds."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_shape(self, n: int) -> None:
        vectors = [_make_score_vector(i, j, seed=i * 10 + j)
                   for i in range(n) for j in range(i + 1, n)]
        mat = aggregate_matrix(vectors, n_fragments=n)
        assert mat.shape == (n, n)

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_symmetric(self, n: int) -> None:
        vectors = [_make_score_vector(i, j, seed=i * 10 + j)
                   for i in range(n) for j in range(i + 1, n)]
        mat = aggregate_matrix(vectors, n_fragments=n)
        np.testing.assert_allclose(mat, mat.T, atol=1e-7)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_values_in_range(self, n: int) -> None:
        vectors = [_make_score_vector(i, j, seed=i * 5 + j)
                   for i in range(n) for j in range(i + 1, n)]
        mat = aggregate_matrix(vectors, n_fragments=n)
        assert float(mat.min()) >= 0.0 - 1e-7
        assert float(mat.max()) <= 1.0 + 1e-7


# ═══════════════════════════════════════════════════════════════════════════════
# score_aggregator — batch_aggregate, ScoreVector properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchAggregateAndScoreVector:
    """batch_aggregate and ScoreVector property invariants."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_batch_aggregate_mean_in_range(self, n: int) -> None:
        vectors = [_make_score_vector(i, i + 1, seed=i) for i in range(n)]
        result = batch_aggregate(vectors)
        assert 0.0 <= result.mean <= 1.0

    @pytest.mark.parametrize("n", [3, 5])
    def test_batch_aggregate_n_pairs(self, n: int) -> None:
        vectors = [_make_score_vector(i, i + 1, seed=i) for i in range(n)]
        result = batch_aggregate(vectors)
        assert result.n_pairs == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_vector_mean_score(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        expected = float(np.mean(list(sv.channels.values())))
        assert sv.mean_score == pytest.approx(expected)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_vector_max_geq_mean(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        assert sv.max_score >= sv.mean_score - 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_vector_min_leq_mean(self, seed: int) -> None:
        sv = _make_score_vector(seed=seed)
        assert sv.min_score <= sv.mean_score + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_vector_pair_key_canonical(self, seed: int) -> None:
        sv = ScoreVector(idx_a=5, idx_b=3,
                         channels={"c": 0.5})
        key = sv.pair_key
        assert key == (3, 5)  # (min, max)

    def test_batch_aggregate_top_pair_has_max_score(self) -> None:
        vectors = [_make_score_vector(i, i + 1, seed=i) for i in range(6)]
        result = batch_aggregate(vectors, method=AggregationMethod.WEIGHTED)
        if result.top_pair is not None:
            top_score = result.scores[result.top_pair]
            for score in result.scores.values():
                assert top_score >= score - 1e-9
