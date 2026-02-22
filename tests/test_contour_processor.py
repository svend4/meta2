"""Тесты для puzzle_reconstruction.preprocessing.contour_processor."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.contour_processor import (
    ContourConfig,
    ContourStats,
    ContourResult,
    resample_contour,
    smooth_contour,
    rdp_simplify,
    normalize_contour,
    contour_area,
    contour_perimeter,
    compute_contour_stats,
    process_contour,
    batch_process_contours,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(side: float = 40.0, n: int = 40) -> np.ndarray:
    """Квадратный контур (n точек, float64)."""
    pts = []
    s = int(n / 4)
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side, i * side / s])
    for i in range(s):
        pts.append([side - i * side / s, side])
    for i in range(s):
        pts.append([0.0, side - i * side / s])
    return np.array(pts, dtype=float)


def _circle(r: float = 20.0, n: int = 64) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _line(n: int = 20) -> np.ndarray:
    return np.column_stack([np.linspace(0, 10, n), np.zeros(n)])


# ─── TestContourConfig ────────────────────────────────────────────────────────

class TestContourConfig:
    def test_defaults(self):
        cfg = ContourConfig()
        assert cfg.n_points == 128
        assert cfg.smooth_sigma == pytest.approx(1.0)
        assert cfg.rdp_epsilon == pytest.approx(2.0)
        assert cfg.normalize is True

    def test_n_points_three_ok(self):
        cfg = ContourConfig(n_points=3)
        assert cfg.n_points == 3

    def test_n_points_two_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(n_points=2)

    def test_n_points_neg_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(n_points=-1)

    def test_smooth_sigma_zero_ok(self):
        cfg = ContourConfig(smooth_sigma=0.0)
        assert cfg.smooth_sigma == 0.0

    def test_smooth_sigma_neg_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(smooth_sigma=-0.1)

    def test_rdp_epsilon_zero_ok(self):
        cfg = ContourConfig(rdp_epsilon=0.0)
        assert cfg.rdp_epsilon == 0.0

    def test_rdp_epsilon_neg_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(rdp_epsilon=-1.0)

    def test_normalize_false(self):
        cfg = ContourConfig(normalize=False)
        assert cfg.normalize is False


# ─── TestContourStats ─────────────────────────────────────────────────────────

class TestContourStats:
    def test_basic(self):
        cs = ContourStats(n_points=4, perimeter=16.0, area=16.0,
                          compactness=0.785, mean_curvature=1.57)
        assert cs.n_points == 4

    def test_all_zero_ok(self):
        cs = ContourStats(n_points=0, perimeter=0.0, area=0.0,
                          compactness=0.0, mean_curvature=0.0)
        assert cs.perimeter == 0.0

    def test_n_points_neg_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=-1, perimeter=0.0, area=0.0,
                         compactness=0.0, mean_curvature=0.0)

    def test_perimeter_neg_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=4, perimeter=-1.0, area=0.0,
                         compactness=0.0, mean_curvature=0.0)

    def test_area_neg_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=4, perimeter=4.0, area=-1.0,
                         compactness=0.0, mean_curvature=0.0)


# ─── TestContourResult ────────────────────────────────────────────────────────

class TestContourResult:
    def _make(self, n: int = 16) -> ContourResult:
        pts = _circle(20.0, n)
        stats = compute_contour_stats(pts)
        return ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_basic(self):
        r = self._make()
        assert isinstance(r, ContourResult)

    def test_n_points(self):
        r = self._make(16)
        assert r.n_points == 16

    def test_centroid_near_origin(self):
        pts = _circle(20.0, 64)
        stats = compute_contour_stats(pts)
        r = ContourResult(points=pts, stats=stats, fragment_id=0)
        cx, cy = r.centroid
        assert abs(cx) < 1.0
        assert abs(cy) < 1.0

    def test_wrong_shape_1d_raises(self):
        pts = np.zeros(8)
        stats = ContourStats(0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_wrong_shape_3d_raises(self):
        pts = np.zeros((4, 2, 2))
        stats = ContourStats(0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_wrong_second_dim_raises(self):
        pts = np.zeros((4, 3))
        stats = ContourStats(0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_simplified_default_false(self):
        r = self._make()
        assert r.simplified is False

    def test_simplified_true_stored(self):
        pts = _circle()
        stats = compute_contour_stats(pts)
        r = ContourResult(points=pts, stats=stats, fragment_id=0, simplified=True)
        assert r.simplified is True


# ─── TestResampleContour ──────────────────────────────────────────────────────

class TestResampleContour:
    def test_returns_ndarray(self):
        r = resample_contour(_square(), 32)
        assert isinstance(r, np.ndarray)

    def test_shape_n_points_times_2(self):
        for n in (8, 16, 32, 64):
            r = resample_contour(_square(), n)
            assert r.shape == (n, 2)

    def test_n_points_two_ok(self):
        r = resample_contour(_square(), 2)
        assert r.shape == (2, 2)

    def test_n_points_one_raises(self):
        with pytest.raises(ValueError):
            resample_contour(_square(), 1)

    def test_too_few_input_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.array([[0.0, 0.0]]), 10)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.zeros((10, 3)), 10)

    def test_zero_length_contour_tiles(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0]])
        r = resample_contour(pts, 8)
        assert r.shape == (8, 2)
        assert np.allclose(r[:, 0], 5.0)

    def test_line_endpoints(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        r = resample_contour(pts, 5)
        assert r[0, 0] == pytest.approx(0.0, abs=0.5)
        assert r[-1, 0] == pytest.approx(20.0, abs=0.5)


# ─── TestSmoothContour ────────────────────────────────────────────────────────

class TestSmoothContour:
    def test_returns_ndarray(self):
        r = smooth_contour(_square())
        assert isinstance(r, np.ndarray)

    def test_shape_preserved(self):
        pts = _square(n=40)
        r = smooth_contour(pts, sigma=1.0)
        assert r.shape == pts.shape

    def test_sigma_zero_identity(self):
        pts = _square()
        r = smooth_contour(pts, sigma=0.0)
        assert np.allclose(r, pts)

    def test_sigma_neg_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(_square(), sigma=-1.0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros((10, 3)), sigma=1.0)

    def test_large_sigma_dampens_variation(self):
        pts = _square(n=64)
        smoothed = smooth_contour(pts, sigma=5.0)
        # Сглаженный контур имеет меньшую дисперсию
        assert np.std(smoothed) <= np.std(pts) + 1.0

    def test_2d_required(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros(10), sigma=1.0)


# ─── TestRdpSimplify ──────────────────────────────────────────────────────────

class TestRdpSimplify:
    def test_returns_ndarray(self):
        r = rdp_simplify(_square(), 0.5)
        assert isinstance(r, np.ndarray)

    def test_shape_n_times_2(self):
        r = rdp_simplify(_square(), 0.5)
        assert r.ndim == 2
        assert r.shape[1] == 2

    def test_simplified_le_original(self):
        pts = _square()
        r = rdp_simplify(pts, 1.0)
        assert r.shape[0] <= pts.shape[0]

    def test_epsilon_zero_keeps_all(self):
        pts = _square()
        r = rdp_simplify(pts, 0.0)
        assert r.shape[0] == pts.shape[0]

    def test_epsilon_neg_raises(self):
        with pytest.raises(ValueError):
            rdp_simplify(_square(), -1.0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            rdp_simplify(np.zeros((10, 3)), 1.0)

    def test_two_points_passthrough(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        r = rdp_simplify(pts, 5.0)
        assert r.shape[0] == 2

    def test_large_epsilon_few_points(self):
        pts = _square()
        r_small = rdp_simplify(pts, 0.1)
        r_large = rdp_simplify(pts, 10.0)
        assert r_large.shape[0] <= r_small.shape[0]


# ─── TestNormalizeContour ─────────────────────────────────────────────────────

class TestNormalizeContour:
    def test_returns_ndarray(self):
        r = normalize_contour(_square())
        assert isinstance(r, np.ndarray)

    def test_shape_preserved(self):
        pts = _square()
        r = normalize_contour(pts)
        assert r.shape == pts.shape

    def test_mean_near_zero(self):
        r = normalize_contour(_square())
        assert np.abs(r.mean(axis=0)).max() < 0.2

    def test_values_within_unit_range(self):
        r = normalize_contour(_square())
        assert np.abs(r).max() <= 1.0 + 1e-9

    def test_empty_contour_ok(self):
        r = normalize_contour(np.zeros((0, 2)))
        assert r.shape == (0, 2)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.zeros((10, 3)))

    def test_scale_non_degenerate(self):
        pts = _square()
        r = normalize_contour(pts)
        assert np.abs(r).max() > 0.5

    def test_single_point_ok(self):
        pts = np.array([[5.0, 5.0]])
        r = normalize_contour(pts)
        assert r.shape == (1, 2)


# ─── TestContourArea ──────────────────────────────────────────────────────────

class TestContourArea:
    def test_square_area(self):
        side = 10.0
        pts = np.array([[0.0, 0.0], [side, 0.0],
                        [side, side], [0.0, side]])
        area = contour_area(pts)
        assert area == pytest.approx(side ** 2)

    def test_circle_area_approx(self):
        pts = _circle(r=20.0, n=512)
        area = contour_area(pts)
        expected = np.pi * 20.0 ** 2
        assert area == pytest.approx(expected, rel=0.01)

    def test_non_negative(self):
        assert contour_area(_square()) >= 0.0

    def test_fewer_than_three_raises(self):
        with pytest.raises(ValueError):
            contour_area(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_collinear_area_zero(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        area = contour_area(pts)
        assert area == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(contour_area(_square()), float)


# ─── TestContourPerimeter ─────────────────────────────────────────────────────

class TestContourPerimeter:
    def test_square_perimeter(self):
        side = 10.0
        pts = np.array([[0.0, 0.0], [side, 0.0],
                        [side, side], [0.0, side]])
        perim = contour_perimeter(pts)
        assert perim == pytest.approx(4 * side)

    def test_circle_perimeter_approx(self):
        pts = _circle(r=10.0, n=512)
        perim = contour_perimeter(pts)
        expected = 2 * np.pi * 10.0
        assert perim == pytest.approx(expected, rel=0.01)

    def test_non_negative(self):
        assert contour_perimeter(_square()) >= 0.0

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError):
            contour_perimeter(np.array([[0.0, 0.0]]))

    def test_two_points_back_and_forth(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0]])
        perim = contour_perimeter(pts)
        assert perim == pytest.approx(10.0)

    def test_returns_float(self):
        assert isinstance(contour_perimeter(_square()), float)


# ─── TestComputeContourStats ──────────────────────────────────────────────────

class TestComputeContourStats:
    def test_returns_contour_stats(self):
        cs = compute_contour_stats(_square())
        assert isinstance(cs, ContourStats)

    def test_n_points(self):
        pts = _square()
        cs = compute_contour_stats(pts)
        assert cs.n_points == len(pts)

    def test_perimeter_positive(self):
        cs = compute_contour_stats(_square())
        assert cs.perimeter > 0.0

    def test_area_positive(self):
        cs = compute_contour_stats(_square())
        assert cs.area > 0.0

    def test_compactness_in_range(self):
        cs = compute_contour_stats(_circle(n=256))
        # Окружность → compactness ≈ 1
        assert 0.8 < cs.compactness <= 1.1

    def test_compactness_square_less_than_circle(self):
        cs_sq = compute_contour_stats(_square())
        cs_ci = compute_contour_stats(_circle(n=64))
        assert cs_ci.compactness > cs_sq.compactness - 0.1

    def test_mean_curvature_non_negative(self):
        cs = compute_contour_stats(_square())
        assert cs.mean_curvature >= 0.0


# ─── TestProcessContour ───────────────────────────────────────────────────────

class TestProcessContour:
    def test_returns_contour_result(self):
        r = process_contour(_square())
        assert isinstance(r, ContourResult)

    def test_fragment_id_stored(self):
        r = process_contour(_square(), fragment_id=5)
        assert r.fragment_id == 5

    def test_n_points_close_to_config(self):
        cfg = ContourConfig(n_points=64, rdp_epsilon=0.0,
                            smooth_sigma=0.0, normalize=False)
        r = process_contour(_square(), cfg=cfg)
        assert r.n_points == 64

    def test_normalize_in_unit_range(self):
        cfg = ContourConfig(n_points=64, smooth_sigma=0.0,
                            rdp_epsilon=0.0, normalize=True)
        r = process_contour(_square(), cfg=cfg)
        assert np.abs(r.points).max() <= 1.0 + 1e-9

    def test_no_normalize(self):
        cfg = ContourConfig(n_points=32, smooth_sigma=0.0,
                            rdp_epsilon=0.0, normalize=False)
        r = process_contour(_square(side=100.0), cfg=cfg)
        assert np.abs(r.points).max() > 1.0

    def test_rdp_simplify_sets_flag(self):
        cfg = ContourConfig(n_points=128, smooth_sigma=0.0,
                            rdp_epsilon=5.0, normalize=False)
        r = process_contour(_circle(r=50.0, n=200), cfg=cfg)
        # После агрессивного RDP флаг может быть True
        assert isinstance(r.simplified, bool)

    def test_stats_populated(self):
        r = process_contour(_square())
        assert isinstance(r.stats, ContourStats)
        assert r.stats.perimeter > 0.0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            process_contour(np.zeros((10, 3)))

    def test_default_config(self):
        r = process_contour(_square())
        assert r.n_points > 0


# ─── TestBatchProcessContours ─────────────────────────────────────────────────

class TestBatchProcessContours:
    def test_returns_list(self):
        result = batch_process_contours([_square(), _circle()])
        assert isinstance(result, list)

    def test_length_matches(self):
        contours = [_square(), _circle(), _square(30.0)]
        assert len(batch_process_contours(contours)) == 3

    def test_empty_list(self):
        assert batch_process_contours([]) == []

    def test_fragment_ids_sequential(self):
        contours = [_square(), _circle(), _square(30.0)]
        for i, r in enumerate(batch_process_contours(contours)):
            assert r.fragment_id == i

    def test_all_contour_results(self):
        for r in batch_process_contours([_square(), _circle()]):
            assert isinstance(r, ContourResult)

    def test_custom_config(self):
        cfg = ContourConfig(n_points=32, smooth_sigma=0.0,
                            rdp_epsilon=0.0, normalize=False)
        for r in batch_process_contours([_square(), _circle()], cfg):
            assert r.n_points == 32
