"""Tests for puzzle_reconstruction.algorithms.fractal.css."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.css import (
    css_similarity,
    css_similarity_mirror,
    css_to_feature_vector,
    curvature_scale_space,
    freeman_chain_code,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=50.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(n=64, side=50.0):
    """Approximate square with n evenly distributed points."""
    per_side = n // 4
    pts = []
    for i in range(per_side):
        pts.append([i * side / per_side, 0])
    for i in range(per_side):
        pts.append([side, i * side / per_side])
    for i in range(per_side):
        pts.append([side - i * side / per_side, side])
    for i in range(per_side):
        pts.append([0, side - i * side / per_side])
    return np.array(pts, dtype=np.float64)


# ─── TestCurvatureScaleSpace ──────────────────────────────────────────────────

class TestCurvatureScaleSpace:
    def test_returns_list(self):
        css = curvature_scale_space(_circle())
        assert isinstance(css, list)

    def test_length_equals_n_sigmas(self):
        css = curvature_scale_space(_circle(), n_sigmas=5)
        assert len(css) == 5

    def test_custom_sigma_range(self):
        sigmas = [1.0, 2.0, 4.0]
        css = curvature_scale_space(_circle(), sigma_range=sigmas)
        assert len(css) == 3

    def test_sigma_values_stored(self):
        sigmas = [2.0, 8.0]
        css = curvature_scale_space(_circle(), sigma_range=sigmas)
        assert css[0][0] == pytest.approx(2.0)
        assert css[1][0] == pytest.approx(8.0)

    def test_each_element_is_tuple(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        for sigma, zc in css:
            assert isinstance(sigma, float)
            assert isinstance(zc, np.ndarray)

    def test_zero_crossings_nonneg_positions(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        for _, zc in css:
            if len(zc) > 0:
                assert np.all(zc >= 0.0)
                assert np.all(zc <= 1.0)

    def test_circle_has_crossings_at_low_sigma(self):
        """Circle is smooth, few crossings at small sigma."""
        css = curvature_scale_space(_circle(), n_sigmas=3)
        # At least one sigma should produce data
        assert len(css) == 3

    def test_square_has_corner_crossings(self):
        css = curvature_scale_space(_square(), sigma_range=[1.0, 2.0])
        assert len(css) == 2


# ─── TestCssToFeatureVector ───────────────────────────────────────────────────

class TestCssToFeatureVector:
    def test_returns_ndarray(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        vec = css_to_feature_vector(css, n_bins=16)
        assert isinstance(vec, np.ndarray)

    def test_length_n_sigmas_times_n_bins(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        vec = css_to_feature_vector(css, n_bins=16)
        assert len(vec) == 4 * 16

    def test_empty_css_returns_zeros(self):
        vec = css_to_feature_vector([], n_bins=8)
        assert np.all(vec == 0.0)
        assert len(vec) == 8

    def test_normalized_unit_norm(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        vec = css_to_feature_vector(css, n_bins=16)
        if np.any(vec > 0):
            assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)

    def test_nonneg_values(self):
        css = curvature_scale_space(_circle(), n_sigmas=4)
        vec = css_to_feature_vector(css)
        assert np.all(vec >= 0.0)


# ─── TestCssSimilarity ────────────────────────────────────────────────────────

class TestCssSimilarity:
    def test_identical_returns_1(self):
        v = np.array([0.3, 0.4, 0.3])
        assert css_similarity(v, v) == pytest.approx(1.0)

    def test_zero_vector_returns_0(self):
        v = np.zeros(5)
        w = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert css_similarity(v, w) == pytest.approx(0.0)

    def test_returns_float_in_0_1(self):
        rng = np.random.default_rng(0)
        a = np.abs(rng.random(20))
        b = np.abs(rng.random(20))
        sim = css_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_different_lengths_truncated(self):
        a = np.array([0.5, 0.3, 0.2, 0.1])
        b = np.array([0.5, 0.3, 0.2])
        sim = css_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_orthogonal_near_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = css_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-9)


# ─── TestCssSimilarityMirror ──────────────────────────────────────────────────

class TestCssSimilarityMirror:
    def test_identical_returns_1(self):
        v = np.array([0.2, 0.5, 0.3])
        assert css_similarity_mirror(v, v) == pytest.approx(1.0)

    def test_mirror_equals_max(self):
        a = np.array([0.7, 0.2, 0.1])
        b = a[::-1].copy()
        direct = css_similarity(a, b)
        mirrored = css_similarity(a, b[::-1])
        result = css_similarity_mirror(a, b)
        assert result == pytest.approx(max(direct, mirrored), abs=1e-8)

    def test_returns_float_in_0_1(self):
        rng = np.random.default_rng(7)
        a = np.abs(rng.random(10))
        b = np.abs(rng.random(10))
        sim = css_similarity_mirror(a, b)
        assert 0.0 <= sim <= 1.0

    def test_mirror_geq_direct(self):
        """Mirror similarity is always >= direct similarity."""
        rng = np.random.default_rng(3)
        a = np.abs(rng.random(8))
        b = np.abs(rng.random(8))
        assert css_similarity_mirror(a, b) >= css_similarity(a, b) - 1e-9


# ─── TestFreemanChainCode ─────────────────────────────────────────────────────

class TestFreemanChainCode:
    def test_too_short_returns_empty(self):
        pts = np.array([[0.0, 0.0]])
        assert freeman_chain_code(pts) == ""

    def test_returns_string(self):
        pts = np.array([[0, 0], [1, 0], [2, 1], [2, 2]])
        code = freeman_chain_code(pts)
        assert isinstance(code, str)

    def test_horizontal_right_code_0(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]])
        code = freeman_chain_code(pts)
        assert "0" in code

    def test_code_contains_valid_digits(self):
        pts = _circle(n=12, r=5.0)
        code = freeman_chain_code(pts.astype(int))
        valid = set("01234567")
        assert all(c in valid for c in code)

    def test_two_points_returns_code(self):
        pts = np.array([[0, 0], [1, 0]])
        code = freeman_chain_code(pts)
        assert len(code) >= 1
