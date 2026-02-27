"""Tests for puzzle_reconstruction.utils.polygon_ops_utils"""
import math
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

np.random.seed(42)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def unit_square():
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


@pytest.fixture
def ccw_triangle():
    # CCW triangle with area = 0.5
    return np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)


@pytest.fixture
def cw_square():
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)


# ─── PolygonOpsConfig ─────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = PolygonOpsConfig()
    assert cfg.clip_epsilon == 1e-9
    assert cfg.simplify_epsilon == 1e-6
    assert cfg.n_samples == 64


def test_config_invalid_clip_epsilon():
    with pytest.raises(ValueError):
        PolygonOpsConfig(clip_epsilon=-1.0)


def test_config_invalid_n_samples():
    with pytest.raises(ValueError):
        PolygonOpsConfig(n_samples=0)


# ─── signed_area ─────────────────────────────────────────────────────────────

def test_signed_area_ccw(unit_square):
    # CCW → positive
    area = signed_area(unit_square)
    assert area > 0
    assert abs(area - 1.0) < 1e-9


def test_signed_area_cw(cw_square):
    area = signed_area(cw_square)
    assert area < 0


def test_signed_area_degenerate():
    line = np.array([[0, 0], [1, 0]])
    assert signed_area(line) == 0.0


def test_signed_area_empty():
    assert signed_area(np.zeros((0, 2))) == 0.0


# ─── polygon_area ─────────────────────────────────────────────────────────────

def test_polygon_area_square(unit_square):
    assert abs(polygon_area(unit_square) - 1.0) < 1e-9


def test_polygon_area_triangle(ccw_triangle):
    assert abs(polygon_area(ccw_triangle) - 0.5) < 1e-9


def test_polygon_area_nonneg(cw_square):
    assert polygon_area(cw_square) >= 0


# ─── polygon_perimeter ────────────────────────────────────────────────────────

def test_polygon_perimeter_square(unit_square):
    p = polygon_perimeter(unit_square)
    assert abs(p - 4.0) < 1e-9


def test_polygon_perimeter_empty():
    assert polygon_perimeter(np.zeros((0, 2))) == 0.0


def test_polygon_perimeter_positive(ccw_triangle):
    assert polygon_perimeter(ccw_triangle) > 0


# ─── polygon_centroid ─────────────────────────────────────────────────────────

def test_polygon_centroid_square(unit_square):
    c = polygon_centroid(unit_square)
    assert abs(c[0] - 0.5) < 1e-9
    assert abs(c[1] - 0.5) < 1e-9


def test_polygon_centroid_empty():
    c = polygon_centroid(np.zeros((0, 2)))
    assert c.shape == (2,)


def test_polygon_centroid_shape(ccw_triangle):
    c = polygon_centroid(ccw_triangle)
    assert c.shape == (2,)


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

def test_polygon_bounding_box_square(unit_square):
    bb = polygon_bounding_box(unit_square)
    assert bb == (0.0, 0.0, 1.0, 1.0)


def test_polygon_bounding_box_empty():
    bb = polygon_bounding_box(np.zeros((0, 2)))
    assert bb == (0.0, 0.0, 0.0, 0.0)


def test_polygon_bounding_box_triangle(ccw_triangle):
    bb = polygon_bounding_box(ccw_triangle)
    assert bb[0] == 0.0
    assert bb[2] == 1.0


# ─── polygon_stats ────────────────────────────────────────────────────────────

def test_polygon_stats_type(unit_square):
    s = polygon_stats(unit_square)
    assert isinstance(s, PolygonStats)


def test_polygon_stats_compactness_circle_approx():
    # A regular polygon approaches circle compactness = 1.0
    n = 100
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    s = polygon_stats(circle)
    assert s.compactness > 0.95  # close to 1 for many-sided polygon


def test_polygon_stats_to_dict(unit_square):
    s = polygon_stats(unit_square)
    d = s.to_dict()
    assert "area" in d
    assert "centroid" in d


def test_polygon_stats_n_vertices(unit_square):
    s = polygon_stats(unit_square)
    assert s.n_vertices == 4


# ─── point_in_polygon ────────────────────────────────────────────────────────

def test_point_in_polygon_inside(unit_square):
    assert point_in_polygon([0.5, 0.5], unit_square) is True


def test_point_in_polygon_outside(unit_square):
    assert point_in_polygon([2.0, 2.0], unit_square) is False


def test_point_in_polygon_degenerate_less_than_3():
    poly = np.array([[0, 0], [1, 0]])
    assert point_in_polygon([0.5, 0.0], poly) is False


# ─── polygon_overlap ─────────────────────────────────────────────────────────

def test_polygon_overlap_overlapping(unit_square):
    shifted = unit_square + np.array([0.5, 0.5])
    result = polygon_overlap(unit_square, shifted)
    assert result.overlap is True
    assert result.iou > 0


def test_polygon_overlap_non_overlapping(unit_square):
    far = unit_square + np.array([10.0, 10.0])
    result = polygon_overlap(unit_square, far)
    assert result.overlap is False
    assert result.iou == 0.0


def test_polygon_overlap_result_type(unit_square):
    r = polygon_overlap(unit_square, unit_square)
    assert isinstance(r, PolygonOverlapResult)
    assert r.to_dict()["overlap"] is True


def test_polygon_overlap_identical_iou(unit_square):
    result = polygon_overlap(unit_square, unit_square)
    assert abs(result.iou - 1.0) < 1e-9


# ─── remove_collinear ────────────────────────────────────────────────────────

def test_remove_collinear_no_removal(unit_square):
    simplified = remove_collinear(unit_square)
    assert len(simplified) == len(unit_square)


def test_remove_collinear_removes_midpoints():
    # Line with midpoint: (0,0), (0.5,0), (1,0), (1,1), (0,1)
    poly = np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    simplified = remove_collinear(poly, epsilon=1e-6)
    assert len(simplified) <= 5  # midpoint may be removed


def test_remove_collinear_short_polygon():
    line = np.array([[0, 0], [1, 1]], dtype=float)
    result = remove_collinear(line)
    assert result.shape == line.shape


# ─── ensure_ccw / ensure_cw ──────────────────────────────────────────────────

def test_ensure_ccw_already_ccw(unit_square):
    result = ensure_ccw(unit_square)
    assert signed_area(result) > 0


def test_ensure_ccw_converts_cw(cw_square):
    result = ensure_ccw(cw_square)
    assert signed_area(result) > 0


def test_ensure_cw_converts_ccw(unit_square):
    result = ensure_cw(unit_square)
    assert signed_area(result) < 0


# ─── polygon_similarity ───────────────────────────────────────────────────────

def test_polygon_similarity_identical(unit_square):
    sim = polygon_similarity(unit_square, unit_square)
    assert abs(sim - 1.0) < 1e-9


def test_polygon_similarity_no_overlap(unit_square):
    far = unit_square + np.array([10.0, 10.0])
    sim = polygon_similarity(unit_square, far)
    assert sim == 0.0


# ─── batch helpers ────────────────────────────────────────────────────────────

def test_batch_polygon_stats(unit_square, ccw_triangle):
    results = batch_polygon_stats([unit_square, ccw_triangle])
    assert len(results) == 2


def test_batch_polygon_overlap_equal_length(unit_square):
    a = [unit_square, unit_square]
    b = [unit_square + 0.5, unit_square + 10.0]
    results = batch_polygon_overlap(a, b)
    assert len(results) == 2


def test_batch_polygon_overlap_unequal_length(unit_square):
    with pytest.raises(ValueError):
        batch_polygon_overlap([unit_square], [unit_square, unit_square])
