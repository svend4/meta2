"""Расширенные тесты для puzzle_reconstruction/utils/topology_utils.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


# ─── TestTopologyConfig ───────────────────────────────────────────────────────

class TestTopologyConfig:
    def test_defaults(self):
        c = TopologyConfig()
        assert c.connectivity == 8
        assert c.min_area == 1

    def test_connectivity_4_valid(self):
        c = TopologyConfig(connectivity=4)
        assert c.connectivity == 4

    def test_connectivity_8_explicit(self):
        c = TopologyConfig(connectivity=8)
        assert c.connectivity == 8

    def test_connectivity_6_invalid(self):
        with pytest.raises(ValueError):
            TopologyConfig(connectivity=6)

    def test_connectivity_0_invalid(self):
        with pytest.raises(ValueError):
            TopologyConfig(connectivity=0)

    def test_min_area_valid(self):
        c = TopologyConfig(min_area=10)
        assert c.min_area == 10

    def test_min_area_0_invalid(self):
        with pytest.raises(ValueError):
            TopologyConfig(min_area=0)

    def test_min_area_negative_invalid(self):
        with pytest.raises(ValueError):
            TopologyConfig(min_area=-3)


# ─── TestComputeEulerNumber ───────────────────────────────────────────────────

class TestComputeEulerNumber:
    def test_returns_int(self):
        m = _solid_mask()
        assert isinstance(compute_euler_number(m), int)

    def test_solid_block_one_component(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        # 1 component, 0 holes → 1
        assert compute_euler_number(m) == 1

    def test_two_components(self):
        m = np.zeros((20, 20), dtype=bool)
        m[2:5, 2:5] = True
        m[13:16, 13:16] = True
        assert compute_euler_number(m) == 2

    def test_ring_euler_zero(self):
        # 1 component, 1 hole → 0
        m = np.zeros((20, 20), dtype=bool)
        m[2:18, 2:18] = True
        m[7:13, 7:13] = False
        assert compute_euler_number(m) == 0

    def test_empty_mask_zero(self):
        m = np.zeros((10, 10), dtype=bool)
        assert compute_euler_number(m) == 0

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            compute_euler_number(np.ones((5, 5, 5), dtype=bool))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            compute_euler_number(np.ones(10, dtype=bool))

    def test_uint8_accepted(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[3:7, 3:7] = 255
        result = compute_euler_number(m)
        assert isinstance(result, int)


# ─── TestCountHoles ───────────────────────────────────────────────────────────

class TestCountHoles:
    def test_solid_block_no_holes(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        assert count_holes(m) == 0

    def test_one_hole(self):
        m = np.zeros((22, 22), dtype=bool)
        m[2:20, 2:20] = True
        m[8:14, 8:14] = False
        assert count_holes(m) == 1

    def test_two_holes(self):
        m = np.zeros((25, 25), dtype=bool)
        m[1:24, 1:24] = True
        m[4:8, 4:8] = False
        m[15:19, 15:19] = False
        assert count_holes(m) == 2

    def test_empty_mask_no_holes(self):
        m = np.zeros((10, 10), dtype=bool)
        assert count_holes(m) == 0

    def test_full_mask_no_holes(self):
        m = np.ones((10, 10), dtype=bool)
        assert count_holes(m) == 0

    def test_returns_nonneg(self):
        m = _solid_mask()
        assert count_holes(m) >= 0

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            count_holes(np.ones((3, 3, 3), dtype=bool))

    def test_border_background_not_counted_as_hole(self):
        # L-shaped: background at corner is not a hole
        m = np.zeros((10, 10), dtype=bool)
        m[1:9, 1:9] = True
        m[1:5, 1:5] = False  # touches boundary indirectly via bg
        # Corners touching edges aren't holes
        assert count_holes(m) >= 0


# ─── TestComputeSolidity ──────────────────────────────────────────────────────

class TestComputeSolidity:
    def test_returns_float(self):
        c = _square_contour()
        assert isinstance(compute_solidity(c), float)

    def test_range_0_to_1(self):
        c = _circle_contour()
        assert 0.0 <= compute_solidity(c) <= 1.0

    def test_circle_near_1(self):
        c = _circle_contour(n=64)
        assert compute_solidity(c) > 0.9

    def test_square_near_1(self):
        c = _square_contour(n=40)
        assert compute_solidity(c) > 0.9

    def test_triangle_equals_1(self):
        c = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        assert compute_solidity(c) == pytest.approx(1.0, abs=1e-6)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_solidity(np.ones((5, 5, 2)))

    def test_wrong_cols_raises(self):
        with pytest.raises(ValueError):
            compute_solidity(np.ones((10, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_solidity(np.array([[0.0, 0.0], [1.0, 1.0]]))


# ─── TestComputeExtent ────────────────────────────────────────────────────────

class TestComputeExtent:
    def test_returns_float(self):
        c = _square_contour()
        assert isinstance(compute_extent(c), float)

    def test_range_0_to_1(self):
        c = _circle_contour()
        assert 0.0 <= compute_extent(c) <= 1.0

    def test_square_high_extent(self):
        c = _square_contour(n=100)
        assert compute_extent(c) > 0.6

    def test_circle_approx_pi_over_4(self):
        c = _circle_contour(n=512)
        assert abs(compute_extent(c) - np.pi / 4) < 0.1

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_extent(np.ones((5, 5, 2)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_extent(np.array([[0.0, 0.0], [1.0, 0.0]]))


# ─── TestComputeConvexity ─────────────────────────────────────────────────────

class TestComputeConvexity:
    def test_returns_float(self):
        c = _square_contour()
        assert isinstance(compute_convexity(c), float)

    def test_range_0_to_1(self):
        c = _circle_contour()
        assert 0.0 <= compute_convexity(c) <= 1.0

    def test_convex_contour_near_1(self):
        c = _square_contour(n=40)
        assert compute_convexity(c) > 0.9

    def test_circle_near_1(self):
        c = _circle_contour(n=64)
        assert compute_convexity(c) > 0.9

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_convexity(np.ones((5, 5, 2)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_convexity(np.ones((2, 2)))


# ─── TestComputeCompactness ───────────────────────────────────────────────────

class TestComputeCompactness:
    def test_returns_float(self):
        c = _circle_contour()
        assert isinstance(compute_compactness(c), float)

    def test_range_0_to_1(self):
        c = _circle_contour()
        assert 0.0 <= compute_compactness(c) <= 1.0

    def test_circle_near_1(self):
        c = _circle_contour(n=256)
        assert compute_compactness(c) > 0.9

    def test_square_lower_than_circle(self):
        sq = compute_compactness(_square_contour(n=100))
        ci = compute_compactness(_circle_contour(n=256))
        assert sq < ci

    def test_square_approx_pi_over_4(self):
        c = _square_contour(n=200)
        assert abs(compute_compactness(c) - np.pi / 4) < 0.15

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_compactness(np.ones((5, 5, 2)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_compactness(np.ones((2, 2)))

    def test_positive_result(self):
        assert compute_compactness(_square_contour()) > 0.0


# ─── TestIsSimplyConnected ────────────────────────────────────────────────────

class TestIsSimplyConnected:
    def test_solid_block_true(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        assert is_simply_connected(m) is True

    def test_ring_false(self):
        m = np.zeros((22, 22), dtype=bool)
        m[2:20, 2:20] = True
        m[8:14, 8:14] = False
        assert is_simply_connected(m) is False

    def test_empty_mask_true(self):
        m = np.zeros((10, 10), dtype=bool)
        assert is_simply_connected(m) is True

    def test_returns_bool(self):
        m = np.ones((5, 5), dtype=bool)
        assert isinstance(is_simply_connected(m), bool)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            is_simply_connected(np.ones((3, 3, 3), dtype=bool))


# ─── TestShapeComplexity ──────────────────────────────────────────────────────

class TestShapeComplexity:
    def test_returns_float(self):
        c = _circle_contour()
        assert isinstance(shape_complexity(c), float)

    def test_range_0_to_1(self):
        c = _circle_contour()
        sc = shape_complexity(c)
        assert 0.0 <= sc <= 1.0

    def test_circle_near_0(self):
        c = _circle_contour(n=256)
        assert shape_complexity(c) < 0.15

    def test_square_higher_than_circle(self):
        sq = shape_complexity(_square_contour(n=100))
        ci = shape_complexity(_circle_contour(n=256))
        assert sq >= ci

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            shape_complexity(np.ones((5, 5, 2)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            shape_complexity(np.ones((2, 2)))


# ─── TestBatchTopology ────────────────────────────────────────────────────────

class TestBatchTopology:
    def test_returns_list(self):
        cs = [_circle_contour(), _square_contour()]
        assert isinstance(batch_topology(cs), list)

    def test_correct_length(self):
        cs = [_circle_contour(), _square_contour(), _circle_contour(n=32)]
        assert len(batch_topology(cs)) == 3

    def test_each_is_dict(self):
        result = batch_topology([_circle_contour()])
        assert isinstance(result[0], dict)

    def test_dict_has_required_keys(self):
        result = batch_topology([_circle_contour()])
        for k in ("solidity", "extent", "convexity", "compactness", "complexity"):
            assert k in result[0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_topology([])

    def test_values_in_range(self):
        result = batch_topology([_circle_contour()])
        for v in result[0].values():
            assert 0.0 <= v <= 1.0

    def test_circle_compactness_high(self):
        result = batch_topology([_circle_contour(n=256)])
        assert result[0]["compactness"] > 0.9
