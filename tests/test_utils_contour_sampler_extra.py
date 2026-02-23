"""Extra tests for puzzle_reconstruction/utils/contour_sampler.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.contour_sampler import (
    SamplerConfig,
    SampledContour,
    sample_uniform,
    sample_curvature,
    sample_random,
    sample_corners,
    sample_contour,
    normalize_contour,
    batch_sample,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _line(n=20):
    x = np.linspace(0.0, 100.0, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def _square(n=40):
    t = np.linspace(0, 1, n, endpoint=False)
    pts = []
    for ti in t:
        if ti < 0.25:
            pts.append([ti * 4 * 100, 0])
        elif ti < 0.5:
            pts.append([100, (ti - 0.25) * 4 * 100])
        elif ti < 0.75:
            pts.append([100 - (ti - 0.5) * 4 * 100, 100])
        else:
            pts.append([0, 100 - (ti - 0.75) * 4 * 100])
    return np.array(pts, dtype=np.float64)


def _zigzag(n=20):
    pts = [[float(i) * 10.0, float((i % 2) * 50)] for i in range(n)]
    return np.array(pts, dtype=np.float64)


# ─── SamplerConfig (extra) ────────────────────────────────────────────────────

class TestSamplerConfigExtra:
    def test_large_n_points_ok(self):
        cfg = SamplerConfig(n_points=1000)
        assert cfg.n_points == 1000

    def test_n_points_2_minimum(self):
        cfg = SamplerConfig(n_points=2)
        assert cfg.n_points == 2

    def test_closed_true_ok(self):
        cfg = SamplerConfig(closed=True)
        assert cfg.closed is True

    def test_seed_stored(self):
        cfg = SamplerConfig(seed=42)
        assert cfg.seed == 42

    def test_corner_threshold_large_ok(self):
        cfg = SamplerConfig(corner_threshold=100.0)
        assert cfg.corner_threshold == pytest.approx(100.0)

    def test_uniform_strategy_default_like(self):
        cfg = SamplerConfig(strategy="uniform")
        assert cfg.strategy == "uniform"

    def test_random_strategy_ok(self):
        cfg = SamplerConfig(strategy="random")
        assert cfg.strategy == "random"

    def test_corners_strategy_ok(self):
        cfg = SamplerConfig(strategy="corners")
        assert cfg.strategy == "corners"

    def test_curvature_strategy_ok(self):
        cfg = SamplerConfig(strategy="curvature")
        assert cfg.strategy == "curvature"


# ─── SampledContour (extra) ───────────────────────────────────────────────────

class TestSampledContourExtra:
    def _make(self, n=5, strategy="uniform", n_source=10):
        return SampledContour(
            points=np.zeros((n, 2)),
            indices=np.arange(n, dtype=np.int64),
            arc_lengths=np.linspace(0, 1, n),
            strategy=strategy,
            n_source=n_source,
        )

    def test_n_points_property(self):
        sc = self._make(8)
        assert sc.n_points == 8

    def test_total_arc_length_last_value(self):
        sc = SampledContour(
            points=np.zeros((4, 2)),
            indices=np.zeros(4, dtype=np.int64),
            arc_lengths=np.array([0.0, 2.0, 5.0, 9.0]),
            strategy="uniform",
            n_source=10,
        )
        assert sc.total_arc_length == pytest.approx(9.0)

    def test_strategy_stored(self):
        sc = self._make(strategy="curvature")
        assert sc.strategy == "curvature"

    def test_n_source_stored(self):
        sc = self._make(n_source=50)
        assert sc.n_source == 50

    def test_indices_dtype_int64(self):
        sc = self._make()
        assert sc.indices.dtype == np.int64

    def test_points_shape_n_2(self):
        sc = self._make(6)
        assert sc.points.shape == (6, 2)

    def test_points_third_column_raises(self):
        with pytest.raises(ValueError):
            SampledContour(
                points=np.zeros((5, 3)),  # 3 columns, not 2
                indices=np.zeros(5, dtype=np.int64),
                arc_lengths=np.zeros(5),
                strategy="uniform",
                n_source=5,
            )


# ─── sample_uniform (extra) ───────────────────────────────────────────────────

class TestSampleUniformExtra:
    def test_square_contour(self):
        result = sample_uniform(_square(), n_points=16)
        assert result.n_points == 16

    def test_arc_lengths_start_at_zero(self):
        result = sample_uniform(_line(), n_points=10)
        assert result.arc_lengths[0] == pytest.approx(0.0)

    def test_n_source_equals_contour_length(self):
        c = _line(n=25)
        result = sample_uniform(c, n_points=10)
        assert result.n_source == 25

    def test_closed_contour(self):
        result = sample_uniform(_square(), n_points=12, closed=True)
        assert result.n_points == 12

    def test_points_within_contour_bbox(self):
        c = _line(n=30)  # x in [0,100], y=0
        result = sample_uniform(c, n_points=10)
        assert result.points[:, 0].min() >= 0.0 - 1e-9
        assert result.points[:, 0].max() <= 100.0 + 1e-9

    def test_large_n_points(self):
        c = _line(n=100)
        result = sample_uniform(c, n_points=50)
        assert result.n_points == 50

    def test_strategy_uniform(self):
        result = sample_uniform(_line(), n_points=8)
        assert result.strategy == "uniform"


# ─── sample_curvature (extra) ─────────────────────────────────────────────────

class TestSampleCurvatureExtra:
    def test_points_shape(self):
        result = sample_curvature(_zigzag(), n_points=8)
        assert result.points.shape == (8, 2)

    def test_indices_within_contour(self):
        c = _zigzag(n=30)
        result = sample_curvature(c, n_points=10)
        assert (result.indices >= 0).all()
        assert (result.indices < 30).all()

    def test_strategy_curvature(self):
        result = sample_curvature(_zigzag(), n_points=6)
        assert result.strategy == "curvature"

    def test_closed_flag(self):
        result = sample_curvature(_square(), n_points=8, closed=True)
        assert result.n_points == 8

    def test_returns_sampled_contour_type(self):
        result = sample_curvature(_zigzag(), n_points=5)
        assert isinstance(result, SampledContour)

    def test_n_source_equals_input_length(self):
        c = _zigzag(n=20)
        result = sample_curvature(c, n_points=10)
        assert result.n_source == 20


# ─── sample_random (extra) ────────────────────────────────────────────────────

class TestSampleRandomExtra:
    def test_same_seed_reproducible(self):
        c = _line(n=50)
        r1 = sample_random(c, n_points=10, seed=7)
        r2 = sample_random(c, n_points=10, seed=7)
        np.testing.assert_array_equal(r1.indices, r2.indices)

    def test_strategy_random(self):
        result = sample_random(_line(), n_points=6)
        assert result.strategy == "random"

    def test_points_shape(self):
        result = sample_random(_line(n=30), n_points=12)
        assert result.points.shape == (12, 2)

    def test_indices_in_bounds(self):
        c = _line(n=40)
        result = sample_random(c, n_points=15, seed=1)
        assert (result.indices >= 0).all()
        assert (result.indices < 40).all()

    def test_n_source_stored(self):
        c = _line(n=30)
        result = sample_random(c, n_points=10)
        assert result.n_source == 30

    def test_different_seeds_produce_different_results(self):
        c = _line(n=100)
        r1 = sample_random(c, n_points=20, seed=0)
        r2 = sample_random(c, n_points=20, seed=99)
        assert not np.array_equal(r1.indices, r2.indices)


# ─── sample_corners (extra) ───────────────────────────────────────────────────

class TestSampleCornersExtra:
    def test_strategy_corners(self):
        result = sample_corners(_zigzag(), n_points=6)
        assert result.strategy == "corners"

    def test_points_shape(self):
        result = sample_corners(_zigzag(), n_points=8)
        assert result.points.shape == (8, 2)

    def test_n_source_stored(self):
        c = _zigzag(n=25)
        result = sample_corners(c, n_points=8)
        assert result.n_source == 25

    def test_indices_within_contour(self):
        c = _square(n=40)
        result = sample_corners(c, n_points=10)
        assert (result.indices >= 0).all()
        assert (result.indices < 40).all()

    def test_square_corners_detected(self):
        # Square has 4 sharp corners; at low threshold should find them
        result = sample_corners(_square(n=40), n_points=4, corner_threshold=0.01)
        assert result.n_points == 4

    def test_closed_flag_ok(self):
        result = sample_corners(_square(), n_points=4, closed=True)
        assert result.n_points == 4


# ─── sample_contour (extra) ───────────────────────────────────────────────────

class TestSampleContourExtra:
    def test_all_strategies_return_correct_n_points(self):
        c = _line(n=30)
        for strategy in ("uniform", "curvature", "random", "corners"):
            cfg = SamplerConfig(strategy=strategy, n_points=10)
            result = sample_contour(c, cfg=cfg)
            assert result.n_points == 10

    def test_cfg_n_points_used(self):
        cfg = SamplerConfig(n_points=12, strategy="uniform")
        result = sample_contour(_line(), cfg=cfg)
        assert result.n_points == 12

    def test_closed_cfg_used(self):
        cfg = SamplerConfig(n_points=8, strategy="uniform", closed=True)
        result = sample_contour(_square(), cfg=cfg)
        assert result.n_points == 8

    def test_default_cfg(self):
        result = sample_contour(_line(), cfg=None)
        assert isinstance(result, SampledContour)

    def test_zigzag_curvature(self):
        cfg = SamplerConfig(n_points=6, strategy="curvature")
        result = sample_contour(_zigzag(), cfg=cfg)
        assert result.strategy == "curvature"


# ─── normalize_contour (extra) ────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_square_normalized(self):
        c = _square(n=40)
        result = normalize_contour(c)
        assert result.max() <= 1.0 + 1e-9
        assert result.min() >= -1.0 - 1e-9

    def test_centered_line(self):
        c = _line(n=20)
        result = normalize_contour(c)
        center = result.mean(axis=0)
        np.testing.assert_allclose(center, 0.0, atol=1e-9)

    def test_float64_output(self):
        result = normalize_contour(_line())
        assert result.dtype == np.float64

    def test_shape_preserved_square(self):
        c = _square(n=40)
        result = normalize_contour(c)
        assert result.shape == c.shape

    def test_zigzag_normalized(self):
        c = _zigzag(n=20)
        result = normalize_contour(c)
        assert result.max() <= 1.0 + 1e-9

    def test_translation_invariant(self):
        c = _line(n=20)
        c_shifted = c + 100.0
        r1 = normalize_contour(c)
        r2 = normalize_contour(c_shifted)
        np.testing.assert_allclose(r1, r2, atol=1e-9)

    def test_scale_invariant(self):
        c = _line(n=20)
        c_scaled = c * 10.0
        r1 = normalize_contour(c)
        r2 = normalize_contour(c_scaled)
        np.testing.assert_allclose(r1, r2, atol=1e-9)


# ─── batch_sample (extra) ─────────────────────────────────────────────────────

class TestBatchSampleExtra:
    def test_single_contour(self):
        result = batch_sample([_line()])
        assert len(result) == 1
        assert isinstance(result[0], SampledContour)

    def test_cfg_applied_to_all(self):
        cfg = SamplerConfig(n_points=12, strategy="uniform")
        contours = [_line(), _zigzag(), _square()]
        result = batch_sample(contours, cfg=cfg)
        for r in result:
            assert r.n_points == 12

    def test_different_contour_lengths(self):
        contours = [_line(n=15), _line(n=30), _line(n=50)]
        cfg = SamplerConfig(n_points=10, strategy="uniform")
        result = batch_sample(contours, cfg=cfg)
        assert all(r.n_points == 10 for r in result)

    def test_random_strategy_batch(self):
        cfg = SamplerConfig(n_points=8, strategy="random", seed=42)
        result = batch_sample([_line(), _zigzag()], cfg=cfg)
        assert all(r.strategy == "random" for r in result)

    def test_corners_strategy_batch(self):
        cfg = SamplerConfig(n_points=5, strategy="corners")
        result = batch_sample([_zigzag(), _square()], cfg=cfg)
        assert all(r.strategy == "corners" for r in result)

    def test_large_batch(self):
        contours = [_line(n=20) for _ in range(20)]
        result = batch_sample(contours)
        assert len(result) == 20
