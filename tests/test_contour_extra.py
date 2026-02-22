"""Extra tests for puzzle_reconstruction.preprocessing.contour."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.contour import (
    extract_contour,
    rdp_simplify,
    split_contour_to_edges,
    resample_curve,
    normalize_contour,
)
from puzzle_reconstruction.models import EdgeSide


def _square_mask(h=64, w=64, margin=10):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


def _circle_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx, r = h // 2, w // 2, min(h, w) // 2 - 4
    yy, xx = np.mgrid[0:h, 0:w]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 255
    return mask


def _rect_contour(h=40, w=60):
    pts = []
    for x in range(0, w): pts.append([x, 0])
    for y in range(0, h): pts.append([w - 1, y])
    for x in range(w - 1, -1, -1): pts.append([x, h - 1])
    for y in range(h - 1, -1, -1): pts.append([0, y])
    return np.array(pts, dtype=np.float32)


def _line_contour(n=50):
    return np.column_stack([
        np.linspace(0, 100, n), np.zeros(n)
    ]).astype(np.float32)


# ─── extract_contour extras ───────────────────────────────────────────────────

class TestExtractContourExtra:
    def test_non_square_mask(self):
        mask = _square_mask(h=32, w=96)
        c = extract_contour(mask)
        assert c.ndim == 2
        assert c.shape[1] == 2

    def test_small_mask(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        c = extract_contour(mask)
        assert c.shape[0] > 0

    def test_thin_rectangle(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:22, 10:54] = 255
        c = extract_contour(mask)
        assert c.shape[0] > 0

    def test_large_mask(self):
        mask = _square_mask(h=256, w=256, margin=20)
        c = extract_contour(mask)
        assert c.shape[0] > 10

    def test_dtype_float32(self):
        c = extract_contour(_square_mask())
        assert c.dtype == np.float32

    def test_circle_contour_many_points(self):
        c = extract_contour(_circle_mask())
        assert c.shape[0] > 20

    def test_x_within_width(self):
        h, w = 64, 128
        mask = _square_mask(h, w)
        c = extract_contour(mask)
        assert c[:, 0].max() < w

    def test_y_within_height(self):
        h, w = 64, 128
        mask = _square_mask(h, w)
        c = extract_contour(mask)
        assert c[:, 1].max() < h


# ─── rdp_simplify extras ──────────────────────────────────────────────────────

class TestRdpSimplifyExtra:
    def test_epsilon_ratio_0_result_ndarray(self):
        c = _rect_contour()
        s = rdp_simplify(c, epsilon_ratio=0.0)
        assert isinstance(s, np.ndarray)
        assert s.ndim == 2

    def test_epsilon_ratio_1_result_at_least_1(self):
        c = _rect_contour()
        s = rdp_simplify(c, epsilon_ratio=1.0)
        assert isinstance(s, np.ndarray)
        assert s.shape[0] >= 1

    def test_line_contour_simplified_to_2(self):
        c = _line_contour(100)
        s = rdp_simplify(c, epsilon_ratio=0.1)
        assert s.shape[0] <= c.shape[0]

    def test_output_float32(self):
        c = _rect_contour()
        s = rdp_simplify(c)
        assert s.dtype == np.float32

    def test_output_ndim_2(self):
        s = rdp_simplify(_rect_contour())
        assert s.ndim == 2

    def test_output_shape_1_2(self):
        s = rdp_simplify(_rect_contour())
        assert s.shape[1] == 2

    def test_large_contour(self):
        c = _rect_contour(h=200, w=300)
        s = rdp_simplify(c, epsilon_ratio=0.01)
        assert s.shape[0] <= c.shape[0]

    def test_very_small_contour(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        s = rdp_simplify(c, epsilon_ratio=0.01)
        assert s.shape[0] >= 2


# ─── split_contour_to_edges extras ───────────────────────────────────────────

class TestSplitContourToEdgesExtra:
    def test_two_sides(self):
        c = _rect_contour()
        edges = split_contour_to_edges(c, n_sides=2)
        assert len(edges) == 2

    def test_six_sides(self):
        c = _rect_contour(h=100, w=100)
        edges = split_contour_to_edges(c, n_sides=6)
        assert len(edges) == 6

    def test_all_edge_sides_valid(self):
        c = _rect_contour()
        valid = {EdgeSide.TOP, EdgeSide.BOTTOM, EdgeSide.LEFT, EdgeSide.RIGHT, EdgeSide.UNKNOWN}
        for _, side in split_contour_to_edges(c, n_sides=4):
            assert side in valid

    def test_points_ndim_2(self):
        c = _rect_contour()
        for pts, _ in split_contour_to_edges(c, n_sides=4):
            assert pts.ndim == 2

    def test_points_second_dim_2(self):
        c = _rect_contour()
        for pts, _ in split_contour_to_edges(c, n_sides=4):
            assert pts.shape[1] == 2

    def test_circle_contour_4_sides(self):
        c = extract_contour(_circle_mask())
        edges = split_contour_to_edges(c, n_sides=4)
        assert len(edges) == 4

    def test_circle_contour_3_sides(self):
        c = extract_contour(_circle_mask())
        edges = split_contour_to_edges(c, n_sides=3)
        assert len(edges) == 3

    def test_non_empty_points(self):
        c = _rect_contour()
        for pts, _ in split_contour_to_edges(c, n_sides=4):
            assert len(pts) > 0


# ─── resample_curve extras ────────────────────────────────────────────────────

class TestResampleCurveExtra:
    def test_n_points_16(self):
        r = resample_curve(_rect_contour(), n_points=16)
        assert r.shape == (16, 2)

    def test_n_points_256(self):
        r = resample_curve(_rect_contour(), n_points=256)
        assert r.shape == (256, 2)

    def test_dtype_numeric(self):
        r = resample_curve(_rect_contour(), n_points=64)
        assert r.dtype in (np.float32, np.float64)

    def test_line_start_x(self):
        c = _line_contour(100)
        r = resample_curve(c, n_points=10)
        assert r[0, 0] == pytest.approx(0.0, abs=2.0)

    def test_line_end_x(self):
        c = _line_contour(100)
        r = resample_curve(c, n_points=10)
        assert r[-1, 0] == pytest.approx(100.0, abs=2.0)

    def test_circle_contour_n_points(self):
        c = extract_contour(_circle_mask())
        r = resample_curve(c, n_points=128)
        assert r.shape == (128, 2)

    def test_two_point_contour(self):
        c = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        r = resample_curve(c, n_points=5)
        assert r.shape == (5, 2)

    def test_n_points_2(self):
        r = resample_curve(_rect_contour(), n_points=2)
        assert r.shape == (2, 2)


# ─── normalize_contour extras ─────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_centroid_close_to_zero_after_normalize(self):
        c = _rect_contour()
        norm, _, _ = normalize_contour(c)
        mean = norm.mean(axis=0)
        assert abs(mean[0]) < 1.0
        assert abs(mean[1]) < 1.0

    def test_scale_type_numeric(self):
        _, _, scale = normalize_contour(_rect_contour())
        assert isinstance(scale, (float, np.floating))

    def test_centroid_type_ndarray(self):
        _, centroid, _ = normalize_contour(_rect_contour())
        assert isinstance(centroid, np.ndarray)

    def test_centroid_shape_2(self):
        _, centroid, _ = normalize_contour(_rect_contour())
        assert centroid.shape == (2,)

    def test_reconstruction_line(self):
        c = _line_contour(50)
        norm, centroid, scale = normalize_contour(c)
        restored = norm * scale + centroid
        assert np.allclose(restored, c, atol=1e-3)

    def test_reconstruction_circle(self):
        c = extract_contour(_circle_mask()).astype(np.float32)
        norm, centroid, scale = normalize_contour(c)
        restored = norm * scale + centroid
        assert np.allclose(restored, c, atol=1e-3)

    def test_two_point_contour(self):
        c = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        norm, centroid, scale = normalize_contour(c)
        assert scale > 0.0

    def test_normalized_ndim_2(self):
        norm, _, _ = normalize_contour(_rect_contour())
        assert norm.ndim == 2

    def test_normalized_shape_matches_input(self):
        c = _rect_contour()
        norm, _, _ = normalize_contour(c)
        assert norm.shape == c.shape
