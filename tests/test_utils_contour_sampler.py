"""Тесты для puzzle_reconstruction/utils/contour_sampler.py."""
import pytest
import numpy as np

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

def make_line(n=20) -> np.ndarray:
    """Прямой контур из n точек."""
    x = np.linspace(0.0, 100.0, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def make_square(n=40) -> np.ndarray:
    """Примерно квадратный замкнутый контур."""
    t = np.linspace(0, 1, n, endpoint=False)
    # four sides
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


def make_zigzag(n=20) -> np.ndarray:
    """Зигзаг-контур с острыми углами."""
    pts = []
    for i in range(n):
        x = float(i) * 10.0
        y = float((i % 2) * 50)
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)


# ─── SamplerConfig ────────────────────────────────────────────────────────────

class TestSamplerConfig:
    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.n_points == 32
        assert cfg.strategy == "uniform"
        assert cfg.closed is False
        assert cfg.seed == 0
        assert cfg.corner_threshold >= 0.0

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            SamplerConfig(n_points=1)

    def test_n_points_0_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            SamplerConfig(n_points=0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            SamplerConfig(strategy="fourier")

    def test_valid_strategies(self):
        for s in ("uniform", "curvature", "random", "corners"):
            cfg = SamplerConfig(strategy=s)
            assert cfg.strategy == s

    def test_negative_corner_threshold_raises(self):
        with pytest.raises(ValueError, match="corner_threshold"):
            SamplerConfig(corner_threshold=-0.1)

    def test_n_points_2_valid(self):
        cfg = SamplerConfig(n_points=2)
        assert cfg.n_points == 2


# ─── SampledContour ───────────────────────────────────────────────────────────

class TestSampledContour:
    def test_creation(self):
        pts = np.zeros((5, 2))
        sc = SampledContour(
            points=pts,
            indices=np.zeros(5, dtype=np.int64),
            arc_lengths=np.zeros(5),
            strategy="uniform",
            n_source=10,
        )
        assert sc.n_points == 5
        assert sc.strategy == "uniform"

    def test_negative_n_source_raises(self):
        with pytest.raises(ValueError, match="n_source"):
            SampledContour(
                points=np.zeros((2, 2)),
                indices=np.zeros(2, dtype=np.int64),
                arc_lengths=np.zeros(2),
                strategy="uniform",
                n_source=-1,
            )

    def test_wrong_points_shape_raises(self):
        with pytest.raises(ValueError):
            SampledContour(
                points=np.zeros(5),  # 1D, not (N,2)
                indices=np.zeros(5, dtype=np.int64),
                arc_lengths=np.zeros(5),
                strategy="uniform",
                n_source=5,
            )

    def test_n_points_property(self):
        sc = SampledContour(
            points=np.zeros((8, 2)),
            indices=np.zeros(8, dtype=np.int64),
            arc_lengths=np.zeros(8),
            strategy="random",
            n_source=20,
        )
        assert sc.n_points == 8

    def test_total_arc_length_empty(self):
        sc = SampledContour(
            points=np.zeros((2, 2)),
            indices=np.zeros(2, dtype=np.int64),
            arc_lengths=np.array([]),
            strategy="uniform",
            n_source=2,
        )
        assert sc.total_arc_length == pytest.approx(0.0)

    def test_total_arc_length_nonzero(self):
        sc = SampledContour(
            points=np.zeros((3, 2)),
            indices=np.zeros(3, dtype=np.int64),
            arc_lengths=np.array([0.0, 5.0, 12.0]),
            strategy="uniform",
            n_source=10,
        )
        assert sc.total_arc_length == pytest.approx(12.0)


# ─── sample_uniform ───────────────────────────────────────────────────────────

class TestSampleUniform:
    def test_returns_sampled_contour(self):
        result = sample_uniform(make_line(), n_points=10)
        assert isinstance(result, SampledContour)

    def test_n_points_correct(self):
        result = sample_uniform(make_line(), n_points=16)
        assert result.n_points == 16

    def test_strategy_is_uniform(self):
        result = sample_uniform(make_line(), n_points=8)
        assert result.strategy == "uniform"

    def test_points_shape(self):
        result = sample_uniform(make_line(), n_points=12)
        assert result.points.shape == (12, 2)

    def test_arc_lengths_ascending(self):
        result = sample_uniform(make_line(), n_points=10)
        diff = np.diff(result.arc_lengths)
        assert (diff >= 0).all()

    def test_n_source_stored(self):
        contour = make_line(n=20)
        result = sample_uniform(contour, n_points=10)
        assert result.n_source == 20

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            sample_uniform(make_line(), n_points=1)

    def test_bad_contour_1d_raises(self):
        with pytest.raises(ValueError):
            sample_uniform(np.zeros(10), n_points=5)

    def test_bad_contour_too_short_raises(self):
        with pytest.raises(ValueError):
            sample_uniform(np.zeros((1, 2)), n_points=5)

    def test_closed_flag(self):
        result = sample_uniform(make_line(), n_points=8, closed=True)
        assert result.n_points == 8

    def test_degenerate_contour(self):
        # All points are the same
        pts = np.ones((10, 2)) * 5.0
        result = sample_uniform(pts, n_points=6)
        assert result.n_points == 6

    def test_points_dtype_float64(self):
        result = sample_uniform(make_line(), n_points=8)
        assert result.points.dtype == np.float64


# ─── sample_curvature ─────────────────────────────────────────────────────────

class TestSampleCurvature:
    def test_returns_sampled_contour(self):
        result = sample_curvature(make_zigzag(), n_points=10)
        assert isinstance(result, SampledContour)

    def test_n_points_correct(self):
        result = sample_curvature(make_zigzag(), n_points=15)
        assert result.n_points == 15

    def test_strategy_is_curvature(self):
        result = sample_curvature(make_zigzag(), n_points=8)
        assert result.strategy == "curvature"

    def test_indices_dtype(self):
        result = sample_curvature(make_zigzag(), n_points=8)
        assert result.indices.dtype == np.int64

    def test_indices_in_bounds(self):
        contour = make_zigzag(n=30)
        result = sample_curvature(contour, n_points=10)
        assert (result.indices >= 0).all()
        assert (result.indices < 30).all()

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError):
            sample_curvature(make_zigzag(), n_points=1)

    def test_closed_flag(self):
        result = sample_curvature(make_zigzag(), n_points=5, closed=True)
        assert result.n_points == 5


# ─── sample_random ────────────────────────────────────────────────────────────

class TestSampleRandom:
    def test_returns_sampled_contour(self):
        result = sample_random(make_line(), n_points=10)
        assert isinstance(result, SampledContour)

    def test_n_points_correct(self):
        result = sample_random(make_line(n=50), n_points=20)
        assert result.n_points == 20

    def test_strategy_is_random(self):
        result = sample_random(make_line(), n_points=5)
        assert result.strategy == "random"

    def test_deterministic_with_seed(self):
        c = make_line(n=30)
        r1 = sample_random(c, n_points=10, seed=42)
        r2 = sample_random(c, n_points=10, seed=42)
        np.testing.assert_array_equal(r1.indices, r2.indices)

    def test_different_seeds_different(self):
        c = make_line(n=50)
        r1 = sample_random(c, n_points=15, seed=0)
        r2 = sample_random(c, n_points=15, seed=99)
        # Very unlikely to be equal
        assert not np.array_equal(r1.indices, r2.indices)

    def test_n_points_exceeds_contour_length(self):
        c = make_line(n=5)  # only 5 points
        result = sample_random(c, n_points=10)  # more than contour
        assert result.n_points == 10

    def test_indices_in_bounds(self):
        c = make_line(n=20)
        result = sample_random(c, n_points=8, seed=0)
        assert (result.indices >= 0).all()
        assert (result.indices < 20).all()

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError):
            sample_random(make_line(), n_points=1)


# ─── sample_corners ───────────────────────────────────────────────────────────

class TestSampleCorners:
    def test_returns_sampled_contour(self):
        result = sample_corners(make_zigzag(), n_points=10)
        assert isinstance(result, SampledContour)

    def test_n_points_correct(self):
        result = sample_corners(make_zigzag(), n_points=12)
        assert result.n_points == 12

    def test_strategy_is_corners(self):
        result = sample_corners(make_zigzag(), n_points=8)
        assert result.strategy == "corners"

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="corner_threshold"):
            sample_corners(make_zigzag(), n_points=8, corner_threshold=-0.1)

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError):
            sample_corners(make_zigzag(), n_points=1)

    def test_high_threshold_no_corners(self):
        # Very high threshold → no corners detected → uniform fallback
        result = sample_corners(make_line(), n_points=8, corner_threshold=100.0)
        assert result.n_points == 8

    def test_zero_threshold_many_corners(self):
        # threshold=0 → all points are corners
        result = sample_corners(make_zigzag(n=10), n_points=5, corner_threshold=0.0)
        assert result.n_points == 5

    def test_indices_in_bounds(self):
        contour = make_square(n=40)
        result = sample_corners(contour, n_points=10)
        assert (result.indices >= 0).all()
        assert (result.indices < 40).all()

    def test_closed_flag(self):
        result = sample_corners(make_zigzag(), n_points=6, closed=True)
        assert result.n_points == 6


# ─── sample_contour ───────────────────────────────────────────────────────────

class TestSampleContour:
    def test_uniform_strategy(self):
        cfg = SamplerConfig(strategy="uniform", n_points=10)
        result = sample_contour(make_line(), cfg=cfg)
        assert result.strategy == "uniform"
        assert result.n_points == 10

    def test_curvature_strategy(self):
        cfg = SamplerConfig(strategy="curvature", n_points=8)
        result = sample_contour(make_zigzag(), cfg=cfg)
        assert result.strategy == "curvature"

    def test_random_strategy(self):
        cfg = SamplerConfig(strategy="random", n_points=8)
        result = sample_contour(make_line(), cfg=cfg)
        assert result.strategy == "random"

    def test_corners_strategy(self):
        cfg = SamplerConfig(strategy="corners", n_points=8)
        result = sample_contour(make_zigzag(), cfg=cfg)
        assert result.strategy == "corners"

    def test_none_cfg_uses_defaults(self):
        result = sample_contour(make_line(), cfg=None)
        assert isinstance(result, SampledContour)


# ─── normalize_contour ────────────────────────────────────────────────────────

class TestNormalizeContour:
    def test_returns_ndarray(self):
        c = make_line()
        result = normalize_contour(c)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self):
        c = make_line(n=20)
        result = normalize_contour(c)
        assert result.shape == c.shape

    def test_values_in_neg1_1(self):
        c = make_square(n=40)
        result = normalize_contour(c)
        assert result.max() <= 1.0 + 1e-9
        assert result.min() >= -1.0 - 1e-9

    def test_centered(self):
        c = make_line()
        result = normalize_contour(c)
        center = result.mean(axis=0)
        np.testing.assert_allclose(center, 0.0, atol=1e-6)

    def test_degenerate_returns_zero(self):
        c = np.ones((5, 2)) * 42.0
        result = normalize_contour(c)
        assert result.max() <= 1e-9

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.zeros(10))

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.zeros((1, 2)))

    def test_dtype_float64(self):
        c = make_line()
        result = normalize_contour(c)
        assert result.dtype == np.float64


# ─── batch_sample ─────────────────────────────────────────────────────────────

class TestBatchSample:
    def test_empty_returns_empty(self):
        result = batch_sample([])
        assert result == []

    def test_length_matches(self):
        contours = [make_line(), make_zigzag(), make_square()]
        result = batch_sample(contours)
        assert len(result) == 3

    def test_all_sampled_contours(self):
        contours = [make_line(), make_zigzag()]
        result = batch_sample(contours)
        for r in result:
            assert isinstance(r, SampledContour)

    def test_cfg_applied(self):
        cfg = SamplerConfig(n_points=16, strategy="uniform")
        contours = [make_line()]
        result = batch_sample(contours, cfg=cfg)
        assert result[0].n_points == 16
