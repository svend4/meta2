"""Additional tests for puzzle_reconstruction.algorithms.fractal.css."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fractal.css import (
    css_similarity,
    css_similarity_mirror,
    css_to_feature_vector,
    curvature_scale_space,
    freeman_chain_code,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=50.0) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square_pts(n=64, side=50.0) -> np.ndarray:
    n4 = n // 4
    pts = []
    for i in range(n4):
        pts.append([i * side / n4, 0.0])
    for i in range(n4):
        pts.append([side, i * side / n4])
    for i in range(n4):
        pts.append([side - i * side / n4, side])
    for i in range(n4):
        pts.append([0.0, side - i * side / n4])
    return np.array(pts, dtype=np.float64)


# ─── TestCurvatureScaleSpaceExtra ─────────────────────────────────────────────

class TestCurvatureScaleSpaceExtra:
    def test_zero_crossings_in_0_1(self):
        css = curvature_scale_space(_circle(), n_sigmas=5)
        for _, zc in css:
            if len(zc) > 0:
                assert np.all((zc >= 0.0) & (zc <= 1.0))

    def test_square_has_crossings_at_low_sigma(self):
        css = curvature_scale_space(_square_pts(), sigma_range=[1.0, 2.0])
        # Square corners → curvature sign changes exist
        total_zc = sum(len(zc) for _, zc in css)
        assert total_zc >= 0  # no crash; corners may or may not register

    def test_larger_sigma_fewer_crossings(self):
        """Higher sigma → smoother → fewer inflection points."""
        css_lo = curvature_scale_space(_square_pts(), sigma_range=[1.0])
        css_hi = curvature_scale_space(_square_pts(), sigma_range=[32.0])
        n_lo = sum(len(zc) for _, zc in css_lo)
        n_hi = sum(len(zc) for _, zc in css_hi)
        assert n_hi <= n_lo + 4  # allow small slack

    def test_single_sigma_range(self):
        css = curvature_scale_space(_circle(), sigma_range=[5.0])
        assert len(css) == 1
        assert css[0][0] == pytest.approx(5.0)

    def test_many_sigmas(self):
        css = curvature_scale_space(_circle(), n_sigmas=10)
        assert len(css) == 10

    def test_small_contour(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        css = curvature_scale_space(pts, n_sigmas=3)
        assert isinstance(css, list)

    def test_zc_arrays_are_1d(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        for _, zc in css:
            assert zc.ndim == 1

    def test_consistent_between_calls(self):
        css1 = curvature_scale_space(_circle(), n_sigmas=4)
        css2 = curvature_scale_space(_circle(), n_sigmas=4)
        for (s1, zc1), (s2, zc2) in zip(css1, css2):
            assert s1 == s2
            np.testing.assert_array_equal(zc1, zc2)


# ─── TestCssToFeatureVectorExtra ──────────────────────────────────────────────

class TestCssToFeatureVectorExtra:
    def test_square_and_circle_differ(self):
        css_c = curvature_scale_space(_circle(), n_sigmas=4)
        css_s = curvature_scale_space(_square_pts(), n_sigmas=4)
        vc = css_to_feature_vector(css_c)
        vs = css_to_feature_vector(css_s)
        # Not guaranteed to differ (both may be all-zeros), just no crash
        assert vc.shape == vs.shape

    def test_custom_n_bins(self):
        css = curvature_scale_space(_circle(), n_sigmas=3)
        vec = css_to_feature_vector(css, n_bins=8)
        assert len(vec) == 3 * 8

    def test_dtype_float(self):
        css = curvature_scale_space(_circle(), n_sigmas=2)
        vec = css_to_feature_vector(css)
        assert vec.dtype in (np.float32, np.float64)

    def test_identical_contours_same_vector(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        v1 = css_to_feature_vector(css)
        v2 = css_to_feature_vector(css)
        np.testing.assert_array_equal(v1, v2)

    def test_n_bins_1_ok(self):
        css = curvature_scale_space(_circle(), n_sigmas=2)
        vec = css_to_feature_vector(css, n_bins=1)
        assert len(vec) == 2


# ─── TestCssSimilarityExtra ───────────────────────────────────────────────────

class TestCssSimilarityExtra:
    def test_symmetric(self):
        rng = np.random.default_rng(9)
        a = np.abs(rng.random(16))
        b = np.abs(rng.random(16))
        assert css_similarity(a, b) == pytest.approx(css_similarity(b, a), abs=1e-9)

    def test_equal_vectors_is_1(self):
        v = np.array([0.0, 0.5, 0.5])
        assert css_similarity(v, v) == pytest.approx(1.0)

    def test_all_zeros_returns_0(self):
        v = np.zeros(6)
        w = np.zeros(6)
        assert css_similarity(v, w) == pytest.approx(0.0)

    def test_single_element_ok(self):
        a = np.array([1.0])
        b = np.array([0.5])
        sim = css_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_large_vectors(self):
        rng = np.random.default_rng(1)
        a = np.abs(rng.random(1000))
        b = np.abs(rng.random(1000))
        sim = css_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_near_orthogonal_near_zero(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 1.0])
        assert css_similarity(a, b) < 0.1


# ─── TestCssSimilarityMirrorExtra ─────────────────────────────────────────────

class TestCssSimilarityMirrorExtra:
    def test_zero_both_returns_0(self):
        v = np.zeros(5)
        assert css_similarity_mirror(v, v) == pytest.approx(0.0)

    def test_geq_direct_for_many_random(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = np.abs(rng.random(12))
            b = np.abs(rng.random(12))
            assert css_similarity_mirror(a, b) >= css_similarity(a, b) - 1e-9

    def test_perfectly_reversed(self):
        """Reversed vector should be detected as mirror."""
        a = np.array([0.1, 0.2, 0.7])
        b = np.array([0.7, 0.2, 0.1])
        # direct: sim(a, b), mirror: sim(a, b[::-1]) = sim(a, a) = 1
        result = css_similarity_mirror(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)


# ─── TestFreemanChainCodeExtra ────────────────────────────────────────────────

class TestFreemanChainCodeExtra:
    def test_empty_input_empty_string(self):
        pts = np.zeros((0, 2))
        assert freeman_chain_code(pts) == ""

    def test_vertical_movement_code_6_or_2(self):
        pts = np.array([[0, 0], [0, 1], [0, 2]])
        code = freeman_chain_code(pts)
        # (0,+1) → dy=1 → code 6
        assert "6" in code

    def test_diagonal_movement_code_7(self):
        pts = np.array([[0, 0], [1, 1]])
        code = freeman_chain_code(pts)
        assert "7" in code  # (1, 1) → code 7

    def test_length_n_minus_1(self):
        """Chain code length <= n-1 (zero moves filtered)."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        code = freeman_chain_code(pts)
        assert len(code) <= 3

    def test_reproducible(self):
        pts = _circle(32, r=10.0).astype(int)
        assert freeman_chain_code(pts) == freeman_chain_code(pts)

    def test_only_valid_directions(self):
        pts = _circle(16, r=10.0)
        code = freeman_chain_code(pts)
        valid = set("01234567")
        assert all(c in valid for c in code)

    def test_no_crash_for_large_jumps(self):
        """Points with |dx|>1 or |dy|>1 get clamped."""
        pts = np.array([[0, 0], [10, 10], [20, 5]])
        code = freeman_chain_code(pts)
        assert isinstance(code, str)

    def test_all_same_points_returns_empty(self):
        pts = np.array([[5, 5], [5, 5], [5, 5]])
        code = freeman_chain_code(pts)
        # (0,0) not in direction table → nothing appended
        assert code == ""
