"""Extra tests for puzzle_reconstruction/preprocessing/contour_processor.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _square(side=40.0, n=40):
    pts, s = [], int(n / 4)
    for i in range(s): pts.append([i * side / s, 0.0])
    for i in range(s): pts.append([side, i * side / s])
    for i in range(s): pts.append([side - i * side / s, side])
    for i in range(s): pts.append([0.0, side - i * side / s])
    return np.array(pts, dtype=float)


def _circle(r=20.0, n=64):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


# ─── ContourConfig (extra) ───────────────────────────────────────────────────

class TestContourConfigExtra:
    def test_defaults_n_points(self):
        assert ContourConfig().n_points == 128

    def test_defaults_smooth_sigma(self):
        assert ContourConfig().smooth_sigma == pytest.approx(1.0)

    def test_defaults_rdp_epsilon(self):
        assert ContourConfig().rdp_epsilon == pytest.approx(2.0)

    def test_defaults_normalize_true(self):
        assert ContourConfig().normalize is True

    def test_n_points_large_ok(self):
        cfg = ContourConfig(n_points=1024)
        assert cfg.n_points == 1024

    def test_n_points_negative_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(n_points=-10)

    def test_smooth_sigma_zero_ok(self):
        cfg = ContourConfig(smooth_sigma=0.0)
        assert cfg.smooth_sigma == pytest.approx(0.0)

    def test_smooth_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(smooth_sigma=-0.5)

    def test_rdp_epsilon_zero_ok(self):
        cfg = ContourConfig(rdp_epsilon=0.0)
        assert cfg.rdp_epsilon == pytest.approx(0.0)

    def test_rdp_epsilon_negative_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(rdp_epsilon=-1.0)

    def test_normalize_false(self):
        cfg = ContourConfig(normalize=False)
        assert cfg.normalize is False


# ─── ContourStats (extra) ────────────────────────────────────────────────────

class TestContourStatsExtra:
    def test_all_fields_stored(self):
        cs = ContourStats(n_points=10, perimeter=40.0, area=100.0,
                          compactness=0.8, mean_curvature=0.5)
        assert cs.n_points == 10
        assert cs.perimeter == pytest.approx(40.0)
        assert cs.area == pytest.approx(100.0)
        assert cs.compactness == pytest.approx(0.8)
        assert cs.mean_curvature == pytest.approx(0.5)

    def test_negative_n_points_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=-1, perimeter=0.0, area=0.0,
                         compactness=0.0, mean_curvature=0.0)

    def test_negative_perimeter_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=4, perimeter=-1.0, area=0.0,
                         compactness=0.0, mean_curvature=0.0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=4, perimeter=4.0, area=-1.0,
                         compactness=0.0, mean_curvature=0.0)

    def test_all_zero_ok(self):
        cs = ContourStats(n_points=0, perimeter=0.0, area=0.0,
                          compactness=0.0, mean_curvature=0.0)
        assert cs.perimeter == pytest.approx(0.0)


# ─── ContourResult (extra) ───────────────────────────────────────────────────

class TestContourResultExtra:
    def _make(self, n=16):
        pts = _circle(20.0, n)
        stats = compute_contour_stats(pts)
        return ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_n_points_correct(self):
        r = self._make(32)
        assert r.n_points == 32

    def test_fragment_id_stored(self):
        pts = _circle()
        stats = compute_contour_stats(pts)
        r = ContourResult(points=pts, stats=stats, fragment_id=7)
        assert r.fragment_id == 7

    def test_centroid_near_origin(self):
        pts = _circle(20.0, 64)
        stats = compute_contour_stats(pts)
        r = ContourResult(points=pts, stats=stats, fragment_id=0)
        cx, cy = r.centroid
        assert abs(cx) < 1.0
        assert abs(cy) < 1.0

    def test_simplified_default_false(self):
        r = self._make()
        assert r.simplified is False

    def test_wrong_1d_raises(self):
        stats = ContourStats(0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            ContourResult(points=np.zeros(8), stats=stats, fragment_id=0)

    def test_wrong_second_dim_raises(self):
        stats = ContourStats(0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            ContourResult(points=np.zeros((4, 3)), stats=stats, fragment_id=0)


# ─── resample_contour (extra) ────────────────────────────────────────────────

class TestResampleContourExtra:
    def test_shape_correct(self):
        for n in (8, 16, 32):
            r = resample_contour(_square(), n)
            assert r.shape == (n, 2)

    def test_returns_ndarray(self):
        assert isinstance(resample_contour(_square(), 16), np.ndarray)

    def test_n_points_one_raises(self):
        with pytest.raises(ValueError):
            resample_contour(_square(), 1)

    def test_too_few_input_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.array([[0.0, 0.0]]), 10)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.zeros((10, 3)), 10)

    def test_zero_length_contour(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0]])
        r = resample_contour(pts, 8)
        assert r.shape == (8, 2)

    def test_circle_stays_on_circle(self):
        r_val = 20.0
        pts = _circle(r_val, 64)
        resampled = resample_contour(pts, 32)
        radii = np.sqrt(resampled[:, 0] ** 2 + resampled[:, 1] ** 2)
        np.testing.assert_allclose(radii, r_val, atol=0.5)


# ─── smooth_contour (extra) ──────────────────────────────────────────────────

class TestSmoothContourExtra:
    def test_shape_preserved(self):
        pts = _square(n=40)
        r = smooth_contour(pts, sigma=1.0)
        assert r.shape == pts.shape

    def test_sigma_zero_identity(self):
        pts = _square()
        r = smooth_contour(pts, sigma=0.0)
        np.testing.assert_allclose(r, pts)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(_square(), sigma=-1.0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros((10, 3)), sigma=1.0)

    def test_returns_ndarray(self):
        assert isinstance(smooth_contour(_square()), np.ndarray)

    def test_large_sigma_reduces_std(self):
        pts = _square(n=64)
        smoothed = smooth_contour(pts, sigma=5.0)
        assert np.std(smoothed) <= np.std(pts) + 1.0


# ─── rdp_simplify (extra) ────────────────────────────────────────────────────

class TestRdpSimplifyExtra:
    def test_returns_ndarray(self):
        assert isinstance(rdp_simplify(_square(), 0.5), np.ndarray)

    def test_shape_n_times_2(self):
        r = rdp_simplify(_square(), 0.5)
        assert r.ndim == 2 and r.shape[1] == 2

    def test_epsilon_zero_keeps_most(self):
        pts = _square()
        r = rdp_simplify(pts, 0.0)
        assert r.shape[0] > 0

    def test_simplified_le_original(self):
        pts = _square()
        r = rdp_simplify(pts, 1.0)
        assert r.shape[0] <= pts.shape[0]

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            rdp_simplify(_square(), -1.0)

    def test_large_epsilon_fewer_points(self):
        pts = _square()
        r_small = rdp_simplify(pts, 0.1)
        r_large = rdp_simplify(pts, 10.0)
        assert r_large.shape[0] <= r_small.shape[0]

    def test_two_points_passthrough(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        r = rdp_simplify(pts, 5.0)
        assert r.shape[0] == 2


# ─── normalize_contour (extra) ───────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_shape_preserved(self):
        pts = _square()
        assert normalize_contour(pts).shape == pts.shape

    def test_mean_near_zero(self):
        r = normalize_contour(_square())
        assert np.abs(r.mean(axis=0)).max() < 0.2

    def test_values_in_unit_range(self):
        r = normalize_contour(_square())
        assert np.abs(r).max() <= 1.0 + 1e-9

    def test_empty_ok(self):
        r = normalize_contour(np.zeros((0, 2)))
        assert r.shape == (0, 2)

    def test_single_point_ok(self):
        r = normalize_contour(np.array([[5.0, 3.0]]))
        assert r.shape == (1, 2)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.zeros((10, 3)))


# ─── contour_area (extra) ────────────────────────────────────────────────────

class TestContourAreaExtra:
    def test_square_area(self):
        side = 10.0
        pts = np.array([[0., 0.], [side, 0.], [side, side], [0., side]])
        assert contour_area(pts) == pytest.approx(side ** 2)

    def test_circle_area_approx(self):
        pts = _circle(r=20.0, n=512)
        assert contour_area(pts) == pytest.approx(np.pi * 400, rel=0.01)

    def test_nonneg(self):
        assert contour_area(_square()) >= 0.0

    def test_collinear_zero(self):
        pts = np.array([[0., 0.], [1., 0.], [2., 0.]])
        assert contour_area(pts) == pytest.approx(0.0, abs=1e-9)

    def test_fewer_than_three_raises(self):
        with pytest.raises(ValueError):
            contour_area(np.array([[0., 0.], [1., 1.]]))

    def test_returns_float(self):
        assert isinstance(contour_area(_square()), float)


# ─── contour_perimeter (extra) ───────────────────────────────────────────────

class TestContourPerimeterExtra:
    def test_square_perimeter(self):
        side = 10.0
        pts = np.array([[0., 0.], [side, 0.], [side, side], [0., side]])
        assert contour_perimeter(pts) == pytest.approx(4 * side)

    def test_circle_perimeter_approx(self):
        pts = _circle(r=10.0, n=512)
        assert contour_perimeter(pts) == pytest.approx(2 * np.pi * 10, rel=0.01)

    def test_nonneg(self):
        assert contour_perimeter(_square()) >= 0.0

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError):
            contour_perimeter(np.array([[0., 0.]]))

    def test_returns_float(self):
        assert isinstance(contour_perimeter(_square()), float)


# ─── compute_contour_stats (extra) ───────────────────────────────────────────

class TestComputeContourStatsExtra:
    def test_returns_stats(self):
        cs = compute_contour_stats(_square())
        assert isinstance(cs, ContourStats)

    def test_n_points(self):
        pts = _square()
        cs = compute_contour_stats(pts)
        assert cs.n_points == len(pts)

    def test_perimeter_positive(self):
        assert compute_contour_stats(_square()).perimeter > 0.0

    def test_area_positive(self):
        assert compute_contour_stats(_square()).area > 0.0

    def test_circle_compactness_near_one(self):
        cs = compute_contour_stats(_circle(n=256))
        assert 0.8 < cs.compactness <= 1.1

    def test_mean_curvature_nonneg(self):
        assert compute_contour_stats(_square()).mean_curvature >= 0.0


# ─── process_contour (extra) ─────────────────────────────────────────────────

class TestProcessContourExtra:
    def test_returns_contour_result(self):
        r = process_contour(_square())
        assert isinstance(r, ContourResult)

    def test_fragment_id_stored(self):
        r = process_contour(_square(), fragment_id=3)
        assert r.fragment_id == 3

    def test_n_points_matches_config(self):
        cfg = ContourConfig(n_points=32, rdp_epsilon=0.0,
                            smooth_sigma=0.0, normalize=False)
        r = process_contour(_square(), cfg=cfg)
        assert r.n_points == 32

    def test_normalize_in_unit_range(self):
        cfg = ContourConfig(n_points=32, smooth_sigma=0.0,
                            rdp_epsilon=0.0, normalize=True)
        r = process_contour(_square(), cfg=cfg)
        assert np.abs(r.points).max() <= 1.0 + 1e-9

    def test_stats_populated(self):
        r = process_contour(_square())
        assert isinstance(r.stats, ContourStats)
        assert r.stats.perimeter > 0.0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            process_contour(np.zeros((10, 3)))


# ─── batch_process_contours (extra) ──────────────────────────────────────────

class TestBatchProcessContoursExtra:
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
        cfg = ContourConfig(n_points=16, smooth_sigma=0.0,
                            rdp_epsilon=0.0, normalize=False)
        for r in batch_process_contours([_square(), _circle()], cfg):
            assert r.n_points == 16
