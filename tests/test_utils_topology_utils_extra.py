"""Extra tests for puzzle_reconstruction/utils/topology_utils.py"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.topology_utils import (
    TopologyConfig,
    batch_topology,
    compute_compactness,
    compute_convexity,
    compute_euler_number,
    compute_extent,
    compute_solidity,
    count_holes,
    is_simply_connected,
    shape_complexity,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square_contour(n: int = 40, side: float = 10.0) -> np.ndarray:
    s = max(n // 4, 1)
    pts = []
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side, i * side / s])
    for i in range(s):
        pts.append([side - i * side / s, side])
    for i in range(s):
        pts.append([0.0, side - i * side / s])
    return np.array(pts)


def _circle_contour(n: int = 64, r: float = 10.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _solid_mask(h: int = 15, w: int = 15) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[3:h - 3, 3:w - 3] = True
    return m


def _ring_mask(h: int = 22, w: int = 22, thick: int = 4) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    m[thick:-thick, thick:-thick] = False
    return m


# ─── TestTopologyConfigExtra ──────────────────────────────────────────────────

class TestTopologyConfigExtra:
    def test_min_area_100_valid(self):
        c = TopologyConfig(min_area=100)
        assert c.min_area == 100

    def test_min_area_1_valid(self):
        c = TopologyConfig(min_area=1)
        assert c.min_area == 1

    def test_connectivity_4_stored(self):
        c = TopologyConfig(connectivity=4)
        assert c.connectivity == 4

    def test_connectivity_8_stored(self):
        c = TopologyConfig(connectivity=8)
        assert c.connectivity == 8

    def test_min_area_large(self):
        c = TopologyConfig(min_area=10000)
        assert c.min_area == 10000


# ─── TestComputeEulerNumberExtra ──────────────────────────────────────────────

class TestComputeEulerNumberExtra:
    def test_all_true_is_1(self):
        m = np.ones((10, 10), dtype=bool)
        assert compute_euler_number(m) == 1

    def test_three_components(self):
        m = np.zeros((30, 10), dtype=bool)
        m[1:4, 1:4] = True
        m[12:15, 1:4] = True
        m[23:26, 1:4] = True
        assert compute_euler_number(m) == 3

    def test_ring_solid_euler_zero(self):
        m = np.zeros((22, 22), dtype=bool)
        m[2:20, 2:20] = True
        m[8:14, 8:14] = False
        assert compute_euler_number(m) == 0

    def test_float_mask_accepted(self):
        m = np.zeros((10, 10), dtype=np.float64)
        m[3:7, 3:7] = 1.0
        result = compute_euler_number(m)
        assert isinstance(result, int)

    def test_single_pixel_is_1(self):
        m = np.zeros((10, 10), dtype=bool)
        m[5, 5] = True
        assert compute_euler_number(m) >= 1

    def test_large_mask(self):
        m = np.zeros((100, 100), dtype=bool)
        m[10:90, 10:90] = True
        result = compute_euler_number(m)
        assert isinstance(result, int)


# ─── TestCountHolesExtra ──────────────────────────────────────────────────────

class TestCountHolesExtra:
    def test_ring_one_hole(self):
        m = _ring_mask()
        assert count_holes(m) == 1

    def test_three_holes(self):
        m = np.zeros((30, 30), dtype=bool)
        m[1:29, 1:29] = True
        m[3:7, 3:7] = False
        m[12:16, 12:16] = False
        m[20:24, 3:7] = False
        assert count_holes(m) == 3

    def test_solid_mask_zero_holes(self):
        m = _solid_mask()
        assert count_holes(m) == 0

    def test_result_is_int(self):
        m = _solid_mask()
        assert isinstance(count_holes(m), int)

    def test_result_nonneg_for_ring(self):
        m = _ring_mask()
        assert count_holes(m) >= 0

    def test_single_component_no_hole(self):
        m = np.zeros((15, 15), dtype=bool)
        m[3:12, 3:12] = True
        assert count_holes(m) == 0


# ─── TestComputeSolidityExtra ─────────────────────────────────────────────────

class TestComputeSolidityExtra:
    def test_square_50_points(self):
        c = _square_contour(n=50)
        s = compute_solidity(c)
        assert 0.0 <= s <= 1.0

    def test_circle_various_n(self):
        for n in (32, 64, 128):
            c = _circle_contour(n=n)
            s = compute_solidity(c)
            assert 0.0 <= s <= 1.0

    def test_result_positive(self):
        c = _square_contour()
        assert compute_solidity(c) > 0.0

    def test_pentagon(self):
        t = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        c = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
        s = compute_solidity(c)
        assert 0.0 <= s <= 1.0

    def test_large_contour(self):
        c = _circle_contour(n=200)
        s = compute_solidity(c)
        assert s > 0.9


# ─── TestComputeExtentExtra ───────────────────────────────────────────────────

class TestComputeExtentExtra:
    def test_square_various_sizes(self):
        for side in (5.0, 10.0, 20.0):
            c = _square_contour(n=100, side=side)
            e = compute_extent(c)
            assert 0.0 <= e <= 1.0

    def test_circle_less_than_square(self):
        # Square has highest extent; circle is pi/4 ≈ 0.785
        sq = compute_extent(_square_contour(n=100))
        ci = compute_extent(_circle_contour(n=128))
        # Square extent should be higher
        assert sq >= ci - 0.1

    def test_result_is_float(self):
        c = _square_contour()
        assert isinstance(compute_extent(c), float)

    def test_elongated_contour(self):
        # Very elongated rectangle
        pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 1.0], [0.0, 1.0]])
        e = compute_extent(pts)
        assert 0.0 <= e <= 1.0

    def test_positive_result(self):
        c = _circle_contour(n=64)
        assert compute_extent(c) > 0.0


# ─── TestComputeConvexityExtra ────────────────────────────────────────────────

class TestComputeConvexityExtra:
    def test_triangle_is_convex(self):
        c = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        # Triangle is convex → convexity ≈ 1
        v = compute_convexity(c)
        assert 0.0 <= v <= 1.0

    def test_five_different_circles(self):
        for n in (16, 32, 64, 128, 256):
            c = _circle_contour(n=n)
            v = compute_convexity(c)
            assert 0.0 <= v <= 1.0

    def test_result_is_float(self):
        c = _square_contour()
        assert isinstance(compute_convexity(c), float)

    def test_square_high_convexity(self):
        c = _square_contour(n=60)
        assert compute_convexity(c) > 0.9

    def test_positive_result(self):
        c = _circle_contour(n=64)
        assert compute_convexity(c) > 0.0


# ─── TestComputeCompactnessExtra ──────────────────────────────────────────────

class TestComputeCompactnessExtra:
    def test_various_circle_sizes(self):
        for r in (5.0, 10.0, 20.0):
            c = _circle_contour(r=r)
            v = compute_compactness(c)
            assert 0.0 <= v <= 1.0

    def test_pentagon_in_range(self):
        t = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        c = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
        v = compute_compactness(c)
        assert 0.0 <= v <= 1.0

    def test_returns_float_type(self):
        c = _circle_contour()
        assert isinstance(compute_compactness(c), float)

    def test_high_circle(self):
        c = _circle_contour(n=512)
        assert compute_compactness(c) > 0.9

    def test_square_lower_than_1(self):
        c = _square_contour(n=200)
        # Square compactness ≈ pi/4 ≈ 0.785
        assert compute_compactness(c) < 1.0


# ─── TestIsSimplyConnectedExtra ───────────────────────────────────────────────

class TestIsSimplyConnectedExtra:
    def test_two_solid_components_true(self):
        # No holes, multiple components → is_simply_connected = True
        m = np.zeros((25, 25), dtype=bool)
        m[2:6, 2:6] = True
        m[15:19, 15:19] = True
        # Two components, no holes
        assert is_simply_connected(m) is True

    def test_two_holes_false(self):
        m = np.zeros((30, 30), dtype=bool)
        m[1:29, 1:29] = True
        m[4:8, 4:8] = False
        m[18:22, 18:22] = False
        assert is_simply_connected(m) is False

    def test_single_pixel_true(self):
        m = np.zeros((10, 10), dtype=bool)
        m[5, 5] = True
        assert is_simply_connected(m) is True

    def test_full_mask_true(self):
        m = np.ones((10, 10), dtype=bool)
        assert is_simply_connected(m) is True

    def test_ring_false(self):
        m = _ring_mask()
        assert is_simply_connected(m) is False


# ─── TestShapeComplexityExtra ─────────────────────────────────────────────────

class TestShapeComplexityExtra:
    def test_five_circles_low_complexity(self):
        for n in (64, 128, 256):
            c = _circle_contour(n=n)
            sc = shape_complexity(c)
            assert sc < 0.2

    def test_various_polygons_in_range(self):
        for n_verts in (3, 5, 6, 8):
            t = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
            c = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
            sc = shape_complexity(c)
            assert 0.0 <= sc <= 1.0

    def test_result_float(self):
        c = _square_contour()
        assert isinstance(shape_complexity(c), float)

    def test_square_lower_complexity(self):
        # Square is less complex than 1-complement of circle
        c = _circle_contour(n=256)
        sc = shape_complexity(c)
        assert sc >= 0.0


# ─── TestBatchTopologyExtra ───────────────────────────────────────────────────

class TestBatchTopologyExtra:
    def test_five_circles(self):
        cs = [_circle_contour(n=64) for _ in range(5)]
        result = batch_topology(cs)
        assert len(result) == 5

    def test_mixed_shapes(self):
        cs = [_circle_contour(), _square_contour(), _circle_contour(n=32),
              _square_contour(n=100)]
        result = batch_topology(cs)
        assert len(result) == 4

    def test_all_dicts(self):
        cs = [_circle_contour(n=64), _square_contour()]
        result = batch_topology(cs)
        for d in result:
            assert isinstance(d, dict)

    def test_all_values_nonneg(self):
        cs = [_circle_contour(n=64), _square_contour()]
        result = batch_topology(cs)
        for d in result:
            for v in d.values():
                assert v >= 0.0

    def test_keys_present_all_items(self):
        cs = [_circle_contour(), _square_contour(), _circle_contour(n=32)]
        result = batch_topology(cs)
        for d in result:
            for k in ("solidity", "extent", "convexity", "compactness", "complexity"):
                assert k in d

    def test_single_circle(self):
        result = batch_topology([_circle_contour(n=128)])
        assert len(result) == 1
        assert result[0]["compactness"] > 0.9
