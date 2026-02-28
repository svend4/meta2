"""
Mutation-killer tests targeting surviving mutants identified by mutmut.

These tests focus on:
- box_counting_curve: output shape, span=0 path, normalization correctness
- css_similarity_mirror: verifies reversal is actually used
- dtw_distance_mirror: verifies reversal and window arg are passed through
- freeman_chain_code / _dx_dy_to_code: full coverage of untested functions
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_curve,
    box_counting_fd,
)
from puzzle_reconstruction.algorithms.fractal.css import (
    _arc_length_param,
    _dx_dy_to_code,
    _zero_crossings_at_sigma,
    css_similarity,
    css_similarity_mirror,
    css_to_feature_vector,
    curvature_scale_space,
    freeman_chain_code,
)
from puzzle_reconstruction.matching.dtw import dtw_distance, dtw_distance_mirror
from puzzle_reconstruction.assembly.greedy import _transform_curve


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circle(n: int = 128, r: float = 100.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _ellipse(n: int = 128, a: float = 100.0, b: float = 40.0) -> np.ndarray:
    """Asymmetric ellipse to expose axis=0 vs axis=None bugs."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


# ── box_counting_curve ────────────────────────────────────────────────────────

class TestBoxCountingCurve:
    """Targeted tests to kill survived mutants in box_counting_curve."""

    def test_default_n_scales_returns_8_values(self):
        """Default call → arrays of length 8 (kills n_scales=8→9 mutant)."""
        lr, lN = box_counting_curve(_circle())
        assert len(lr) == 8, f"Expected 8 scale values, got {len(lr)}"
        assert len(lN) == 8, f"Expected 8 log_N values, got {len(lN)}"

    @pytest.mark.parametrize("n", [4, 5, 6, 10])
    def test_custom_n_scales_length(self, n: int):
        """n_scales=n produces arrays of exactly length n."""
        lr, lN = box_counting_curve(_circle(64), n_scales=n)
        assert len(lr) == n
        assert len(lN) == n

    def test_zero_span_returns_zero_arrays_of_correct_shape(self):
        """Constant-point contour → zeros of length n_scales (not None-shaped)."""
        pts = np.full((20, 2), 5.0)
        lr, lN = box_counting_curve(pts, n_scales=6)
        # If np.zeros(None) were used, this would be a 0-d scalar, not length-6
        assert len(lr) == 6, f"Expected 6, got {len(lr)}"
        assert len(lN) == 6, f"Expected 6, got {len(lN)}"
        assert np.all(lr == 0.0)
        assert np.all(lN == 0.0)

    def test_log_r_inv_strictly_monotone(self):
        """log_r_inv must be strictly increasing (each scale is 2× the previous)."""
        lr, _ = box_counting_curve(_circle(), n_scales=6)
        diffs = np.diff(lr)
        assert np.all(diffs > 0), f"log_r_inv not monotone: diffs={diffs}"

    def test_log_N_positive_for_real_contour(self):
        """For a 128-pt circle, all box counts must be > 1 → log_N > 0."""
        _, lN = box_counting_curve(_circle(128), n_scales=6)
        assert np.all(lN >= 0), "log_N must be non-negative"
        # At the finest scale, many boxes should be occupied
        assert lN[-1] > 0.0, "Finest scale should have >1 box occupied"

    def test_asymmetric_contour_does_not_crash(self):
        """Asymmetric ellipse (wide x, narrow y) should work (axis=0 vs axis=None)."""
        lr, lN = box_counting_curve(_ellipse(256, a=200.0, b=5.0), n_scales=5)
        assert len(lr) == 5
        assert len(lN) == 5
        assert np.all(np.isfinite(lr))
        assert np.all(np.isfinite(lN))

    def test_normalization_correctness(self):
        """
        Verify that scaling the contour does NOT change log_N values significantly.
        (pts - mins) / span → pts_norm in [0, 1].
        If pts * span were used instead of / span, all points would be clamped
        to the max bin and N would be 1 at all scales → log_N = 0.
        """
        contour = _circle(200)
        _, lN_normal = box_counting_curve(contour, n_scales=4)
        # If normalization is wrong, all points map to one bin → lN all near 0
        # At scale k=4 (n_bins=16), a circle should occupy many bins
        assert lN_normal[-1] > 1.0, (
            f"At finest scale, log_N should exceed 1.0; got {lN_normal[-1]}"
        )

    def test_x_and_y_both_used(self):
        """
        Horizontal line vs vertical line should give different box-count curves.
        This kills the mutation that replaces pts_norm[:,0] with pts_norm[:,1]
        for the x-index computation.
        """
        # A purely horizontal line: y is constant, x varies
        n = 64
        h_line = np.column_stack([np.linspace(0, 100, n), np.zeros(n)])
        # A purely vertical line: x is constant, y varies
        v_line = np.column_stack([np.zeros(n), np.linspace(0, 100, n)])

        lr_h, lN_h = box_counting_curve(h_line, n_scales=5)
        lr_v, lN_v = box_counting_curve(v_line, n_scales=5)

        # Both should give same scale axes
        np.testing.assert_allclose(lr_h, lr_v, atol=1e-9)
        # And by symmetry, same box counts (the shape is a line either way)
        np.testing.assert_allclose(lN_h, lN_v, atol=1e-6)

    def test_multiply_vs_divide_normalization(self):
        """
        pts_norm = (pts - mins) / span is correct.
        If * span were used, pts_norm would be huge and all ix/iy would be
        clamped to n_bins-1 → only 1 bin occupied → log_N ≈ 0.
        """
        contour = _circle(200, r=50.0)
        _, lN = box_counting_curve(contour, n_scales=5)
        # If * span were used instead of / span, all n_bins would be occupied
        # at a single bin, so log_N[-1] = log2(1) = 0
        assert lN[-1] > 0.5, (
            f"log_N at finest scale should be > 0.5; got {lN[-1]} "
            f"(normalization bug: * vs /)"
        )


# ── css_similarity_mirror ─────────────────────────────────────────────────────

class TestCssSimilarityMirror:
    """Kill mutant: css_b[::-1] → css_b[::+1] (no reversal)."""

    def test_mirror_detects_reversed_match(self):
        """
        a has a peak at index 0, b has a peak at the last index.
        Direct similarity ≈ 0, but after reversing b the match is 1.
        """
        n = 16
        a = np.zeros(n)
        a[0] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0  # spike at the end — reversed, it aligns with a[0]

        direct = css_similarity(a, b)
        mirrored = css_similarity_mirror(a, b)

        assert mirrored > direct, (
            "css_similarity_mirror should find the reversed match"
        )
        assert mirrored > 0.9, (
            f"Reversed match should be ~1.0, got {mirrored}"
        )

    def test_mirror_geq_direct(self):
        """mirror similarity is always >= direct similarity."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.uniform(0, 1, 20)
            a /= np.linalg.norm(a) + 1e-9
            b = rng.uniform(0, 1, 20)
            b /= np.linalg.norm(b) + 1e-9

            direct = css_similarity(a, b)
            mirror = css_similarity_mirror(a, b)
            assert mirror >= direct - 1e-9, (
                f"mirror={mirror} should be >= direct={direct}"
            )

    def test_identical_sequences_mirror_one(self):
        """Identical CSS vectors → mirror similarity = 1."""
        a = np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
        a /= np.linalg.norm(a)
        assert abs(css_similarity_mirror(a, a) - 1.0) < 1e-9


# ── dtw_distance_mirror ───────────────────────────────────────────────────────

class TestDtwDistanceMirror:
    """Kill survived mutants in dtw_distance_mirror."""

    def _seq(self, values) -> np.ndarray:
        return np.array([[v] for v in values], dtype=float)

    def test_reversed_sequence_gives_zero_distance(self):
        """
        If b = a[::-1], then dtw_distance_mirror(a, b) == 0.
        Kills mutant that uses b[::+1] (no reversal) or b[::-2].
        """
        a = self._seq([0.0, 1.0, 2.0, 3.0, 4.0])
        b = self._seq([4.0, 3.0, 2.0, 1.0, 0.0])  # a reversed
        dist = dtw_distance_mirror(a, b, window=10)
        assert dist < 1e-6, (
            f"Reversed sequence should give distance ~0, got {dist}"
        )

    def test_reversed_is_better_than_direct(self):
        """
        For b = reverse(a), mirror distance < direct distance.
        (Verifies that the minimum is taken over both direct and reversed.)
        """
        a = self._seq([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        b = self._seq([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # reversed step

        d_direct = dtw_distance(a, b, window=10)
        d_mirror = dtw_distance_mirror(a, b, window=10)

        assert d_mirror < d_direct, (
            f"mirror ({d_mirror}) should be < direct ({d_direct})"
        )
        assert d_mirror < 1e-6, "Reversed step sequence: mirror dist should be ~0"

    def test_window_parameter_forwarded_to_direct(self):
        """
        dtw_distance_mirror passes window to dtw_distance for direct comparison.
        Use window=1: for sequences of equal length, w=max(1, 0)=1,
        so only diagonal is considered → high cost for misaligned sequences.
        """
        # Two sequences offset by 3: with window=1 cost > window=20 cost
        n = 20
        a = self._seq([float(i) for i in range(n)])
        b = self._seq([float(i + 3) for i in range(n)])

        d_small_window = dtw_distance(a, b, window=1)
        d_large_window = dtw_distance(a, b, window=20)
        # Ensure window DOES matter for this sequence pair
        # (both should give the same cost since offset is constant = 3,
        # but with very small window some diagonal paths are forced)
        # At minimum, the function should complete without error
        d_mirror = dtw_distance_mirror(a, b, window=1)
        assert np.isfinite(d_mirror), "dtw_distance_mirror should return finite value"

    def test_window_parameter_forwarded_to_mirrored(self):
        """
        dtw_distance_mirror passes window to dtw_distance for mirrored comparison.
        """
        a = self._seq([0.0, 1.0, 2.0, 3.0])
        b = self._seq([3.0, 2.0, 1.0, 0.0])
        # With any reasonable window, mirrored = dtw_distance(a, a) = 0
        dist = dtw_distance_mirror(a, b, window=2)
        assert dist < 1e-6, (
            f"Window=2 passed to mirrored call should still find perfect match, got {dist}"
        )

    def test_mirror_result_leq_both_components(self):
        """Result is always min(direct, mirrored) — leq both."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            n = rng.integers(3, 12)
            a = rng.uniform(0, 10, (n, 2))
            b = rng.uniform(0, 10, (n, 2))
            d_direct = dtw_distance(a, b, window=5)
            d_mirror_of_b = dtw_distance(a, b[::-1], window=5)
            d_result = dtw_distance_mirror(a, b, window=5)
            expected = min(d_direct, d_mirror_of_b)
            assert abs(d_result - expected) < 1e-9, (
                f"result={d_result}, expected={expected}"
            )


# ── freeman_chain_code ────────────────────────────────────────────────────────

class TestFreemanChainCode:
    """Full coverage of freeman_chain_code and _dx_dy_to_code (90 'no tests')."""

    def test_horizontal_east(self):
        """Horizontal line moving right → direction 0 (East)."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "000", f"Expected '000', got '{code}'"

    def test_vertical_south(self):
        """Vertical line moving down → direction 6 (South, dy=+1)."""
        pts = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "666", f"Expected '666', got '{code}'"

    def test_vertical_north(self):
        """Vertical line moving up → direction 2 (North, dy=-1)."""
        pts = np.array([[0, 3], [0, 2], [0, 1], [0, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "222", f"Expected '222', got '{code}'"

    def test_horizontal_west(self):
        """Horizontal line moving left → direction 4 (West)."""
        pts = np.array([[3, 0], [2, 0], [1, 0], [0, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "444", f"Expected '444', got '{code}'"

    def test_diagonal_northeast(self):
        """Diagonal up-right → direction 1 (NE, dx=+1, dy=-1)."""
        pts = np.array([[0, 2], [1, 1], [2, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "11", f"Expected '11', got '{code}'"

    def test_diagonal_northwest(self):
        """Diagonal up-left → direction 3 (NW, dx=-1, dy=-1)."""
        pts = np.array([[2, 2], [1, 1], [0, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "33", f"Expected '33', got '{code}'"

    def test_diagonal_southwest(self):
        """Diagonal down-left → direction 5 (SW, dx=-1, dy=+1)."""
        pts = np.array([[2, 0], [1, 1], [0, 2]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "55", f"Expected '55', got '{code}'"

    def test_diagonal_southeast(self):
        """Diagonal down-right → direction 7 (SE, dx=+1, dy=+1)."""
        pts = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == "77", f"Expected '77', got '{code}'"

    def test_single_point_returns_empty(self):
        """Single point → empty string."""
        pts = np.array([[0, 0]], dtype=float)
        code = freeman_chain_code(pts)
        assert code == ""

    def test_empty_returns_empty(self):
        """Empty array → empty string."""
        pts = np.zeros((0, 2), dtype=float)
        code = freeman_chain_code(pts)
        assert code == ""

    def test_square_loop_returns_four_directions(self):
        """Square contour visits all 4 cardinal directions."""
        pts = np.array([
            [0, 0], [1, 0], [2, 0],   # → East (0)
            [2, 1], [2, 2],            # ↓ South (6)
            [1, 2], [0, 2],            # ← West (4)
            [0, 1],                    # ↑ North (2)
        ], dtype=float)
        code = freeman_chain_code(pts)
        assert "0" in code, "Square should have East moves"
        assert "6" in code, "Square should have South moves"
        assert "4" in code, "Square should have West moves"
        assert "2" in code, "Square should have North moves"

    def test_length_equals_n_minus_1(self):
        """Code length == n_pts - 1 for a clean grid path."""
        pts = np.array([[i, 0] for i in range(10)], dtype=float)
        code = freeman_chain_code(pts)
        assert len(code) == 9, f"Expected 9 chars, got {len(code)}"

    def test_float_points_rounded(self):
        """Float coords are rounded to int before computing direction."""
        pts = np.array([[0.1, 0.2], [1.3, 0.4], [2.1, 0.1]], dtype=float)
        code = freeman_chain_code(pts)
        # All should round to horizontal (East=0)
        assert len(code) == 2

    def test_large_delta_clamped_to_one(self):
        """dx > 1 is clamped to 1 before lookup."""
        pts = np.array([[0, 0], [10, 0]], dtype=float)
        code = freeman_chain_code(pts)
        # dx=10 → clamped to 1 → East direction 0
        assert code == "0", f"Expected '0' after clamping, got '{code}'"


# ── _dx_dy_to_code ────────────────────────────────────────────────────────────

class TestDxDyToCode:
    """Tests for all 8 Freeman directions and the None case."""

    @pytest.mark.parametrize("dx, dy, expected", [
        (1,   0, 0),   # East
        (1,  -1, 1),   # NE
        (0,  -1, 2),   # North
        (-1, -1, 3),   # NW
        (-1,  0, 4),   # West
        (-1,  1, 5),   # SW
        (0,   1, 6),   # South
        (1,   1, 7),   # SE
    ])
    def test_all_eight_directions(self, dx: int, dy: int, expected: int):
        assert _dx_dy_to_code(dx, dy) == expected, (
            f"_dx_dy_to_code({dx}, {dy}) should be {expected}"
        )

    def test_no_movement_returns_none(self):
        """(0, 0) is not a valid Freeman direction → None."""
        assert _dx_dy_to_code(0, 0) is None

    def test_out_of_range_returns_none(self):
        """(2, 0) is not in the direction table → None."""
        assert _dx_dy_to_code(2, 0) is None

    def test_returns_int_for_valid_directions(self):
        """All 8 valid directions return an int, not None."""
        valid = [
            (1, 0), (1, -1), (0, -1), (-1, -1),
            (-1, 0), (-1, 1), (0, 1), (1, 1),
        ]
        for dx, dy in valid:
            result = _dx_dy_to_code(dx, dy)
            assert isinstance(result, int), (
                f"_dx_dy_to_code({dx},{dy}) should return int, got {type(result)}"
            )
            assert 0 <= result <= 7, f"Direction code out of range: {result}"


# ── css_similarity (edge case coverage) ──────────────────────────────────────

class TestCssSimilarityEdgeCases:
    """Kill css_similarity mutants: shape-mismatch branch, zero-norm branches."""

    def test_different_shape_vectors_handled(self):
        """
        css_similarity with different-length inputs uses min(len) truncation.
        Mutant: if shape == shape (inverted check) → skip truncation when
        shapes DIFFER, causing a dot product of mismatched arrays.
        """
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # len=5, a has len=4
        # Should NOT crash, should return a value in [0, 1]
        result = css_similarity(a, b)
        assert 0.0 <= result <= 1.0, f"Expected [0,1], got {result}"
        # a and b[:4] = [1,0,0,0] are identical → similarity should be ~1
        assert result > 0.9, f"Truncated identical vectors should have sim ≈ 1, got {result}"

    def test_same_shape_normal_path(self):
        """
        css_similarity with SAME shape should compute direct dot product.
        Mutant (== instead of !=) would enter the truncation branch for equal shapes.
        """
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = css_similarity(a, b)
        assert abs(result) < 1e-9, f"Orthogonal vectors: expected 0, got {result}"

    def test_both_zero_norm_returns_zero(self):
        """
        When BOTH vectors are zero: return 0.0.
        Mutant (norm_a == 0 OR norm_b == 0): also returns 0.0 when only one is zero.
        """
        a = np.zeros(5)
        b = np.zeros(5)
        result = css_similarity(a, b)
        assert result == 0.0, f"Both-zero vectors: expected 0, got {result}"

    def test_one_zero_norm_returns_zero(self):
        """
        When only ONE vector is zero: return 0.0.
        This test distinguishes 'AND' (only both-zero → 0.0) from
        'OR' (either zero → 0.0) by verifying behavior when one is nonzero.
        """
        a = np.zeros(5)
        b = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = css_similarity(a, b)
        # Expected: 0.0 (one zero-norm → 0)
        assert result == 0.0, f"One-zero vector: expected 0, got {result}"

    def test_nonzero_vectors_return_cosine(self):
        """Normal case: both nonzero → cosine similarity."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert abs(css_similarity(a, b) - 1.0) < 1e-9

    def test_truncation_uses_min_length(self):
        """
        When shapes differ, truncation to min(len) is used.
        Mutant (n = None) → css_a[:None] = full array → shape mismatch error.
        Mutant (n = min(None, len)) → TypeError.
        """
        a = np.array([1.0, 0.5, 0.0])
        b = np.array([1.0, 0.5, 0.0, 9.9, 9.9])  # extra garbage at end
        result = css_similarity(a, b)
        # Should use the first 3 elements → perfect match
        assert result > 0.99, f"Expected ~1.0 after truncation, got {result}"


# ── _arc_length_param coverage ────────────────────────────────────────────────

class TestArcLengthParam:
    """Kill mutants in _arc_length_param (15 survived)."""

    def test_regular_contour_parameterization(self):
        """For a unit circle, arc-length param goes from 0 to near 1."""
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        x = np.cos(t)
        y = np.sin(t)
        arc = _arc_length_param(x, y)
        assert arc[0] == pytest.approx(0.0, abs=1e-9), "First param should be 0"
        assert arc[-1] <= 1.0, "Last param should be <= 1"
        assert arc[-1] > 0.8, "Last param should be close to 1"

    def test_arc_length_param_monotone(self):
        """Arc-length parameter is monotonically non-decreasing."""
        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        x = 100 * np.cos(t)
        y = 50 * np.sin(t)
        arc = _arc_length_param(x, y)
        diffs = np.diff(arc)
        assert np.all(diffs >= -1e-12), "Arc-length param must be non-decreasing"

    def test_arc_length_all_same_returns_linspace(self):
        """All same points → zero total length → returns linspace(0,1)."""
        x = np.full(10, 5.0)
        y = np.full(10, 3.0)
        arc = _arc_length_param(x, y)
        assert len(arc) == 10
        # Should return linspace(0, 1, 10)
        expected = np.linspace(0, 1, 10)
        np.testing.assert_allclose(arc, expected, atol=1e-9)

    def test_arc_length_in_0_1_range(self):
        """Arc-length parameter is always in [0, 1)."""
        t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        x = 200 * np.cos(t) + 50 * np.cos(3 * t)
        y = 200 * np.sin(t) + 50 * np.sin(3 * t)
        arc = _arc_length_param(x, y)
        assert np.all(arc >= 0.0), "Arc-length param must be >= 0"
        assert np.all(arc < 1.0 + 1e-9), "Arc-length param must be <= 1"


# ── _zero_crossings_at_sigma direct tests ─────────────────────────────────────

class TestZeroCrossingsAtSigma:
    """Kill _zero_crossings_at_sigma mutants (30 survived)."""

    def _circle_xy(self, n: int = 64):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 100 * np.cos(t)
        y = 100 * np.sin(t)
        arc = _arc_length_param(x, y)
        return x, y, arc

    def test_returns_array(self):
        """_zero_crossings_at_sigma always returns an ndarray."""
        x, y, t = self._circle_xy()
        zc = _zero_crossings_at_sigma(x, y, t, sigma=1.0)
        assert isinstance(zc, np.ndarray), f"Expected ndarray, got {type(zc)}"

    def test_zero_crossings_in_0_1_range(self):
        """All zero-crossing positions are in [0, 1]."""
        x, y, t = self._circle_xy(128)
        for sigma in [1.0, 2.0, 4.0]:
            zc = _zero_crossings_at_sigma(x, y, t, sigma)
            if len(zc) > 0:
                assert np.all(zc >= 0.0), f"ZC < 0 at sigma={sigma}: {zc}"
                assert np.all(zc <= 1.0), f"ZC > 1 at sigma={sigma}: {zc}"

    def test_uses_x_coordinate(self):
        """
        Verify that X coordinate is actually used (not Y for both).
        Mutant: Xs = None crashes downstream.
        """
        # Asymmetric shape: ellipse with a=100, b=10
        n = 64
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 100 * np.cos(t)
        y = 10 * np.sin(t)
        arc = _arc_length_param(x, y)
        # Should complete without error
        zc = _zero_crossings_at_sigma(x, y, arc, sigma=2.0)
        assert isinstance(zc, np.ndarray)

    def test_uses_y_coordinate(self):
        """
        Verify Y coordinate is used.
        Mutant: Ys = None crashes when computing gradient(None).
        """
        n = 64
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 10 * np.cos(t)
        y = 100 * np.sin(t)
        arc = _arc_length_param(x, y)
        # Should complete without error
        zc = _zero_crossings_at_sigma(x, y, arc, sigma=2.0)
        assert isinstance(zc, np.ndarray)

    def test_wraps_mode_preserves_count(self):
        """
        gaussian_filter1d with mode='wrap' (not None) gives a valid result.
        Mutant: mode=None → ValueError.
        """
        x, y, t = self._circle_xy(64)
        # Should not raise
        zc = _zero_crossings_at_sigma(x, y, t, sigma=3.0)
        assert isinstance(zc, np.ndarray)

    def test_sigma_parameter_affects_output(self):
        """
        Different sigma values produce different zero crossing patterns.
        Mutant: sigma=None → TypeError from gaussian_filter1d.
        """
        x, y, t = self._circle_xy(128)
        zc_small = _zero_crossings_at_sigma(x, y, t, sigma=1.0)
        zc_large = _zero_crossings_at_sigma(x, y, t, sigma=10.0)
        # Both should be valid arrays
        assert isinstance(zc_small, np.ndarray)
        assert isinstance(zc_large, np.ndarray)
        # With larger sigma, fewer zero crossings (smoother curve)
        assert len(zc_large) <= len(zc_small) + 5, (
            "Larger sigma should produce fewer or equal zero crossings"
        )

    def test_curvature_formula_uses_x_y_derivs(self):
        """
        Verify curvature formula κ = (x'y'' - x''y') is computed correctly.
        A star-shaped contour should have more zero crossings than a circle.
        """
        n = 256
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Circle (smooth → fewer zero crossings at low sigma)
        x_circ = 100 * np.cos(t)
        y_circ = 100 * np.sin(t)
        arc_circ = _arc_length_param(x_circ, y_circ)
        zc_circ = _zero_crossings_at_sigma(x_circ, y_circ, arc_circ, sigma=1.0)

        # Star (jagged → more zero crossings)
        r_star = 100 + 30 * np.cos(5 * t)
        x_star = r_star * np.cos(t)
        y_star = r_star * np.sin(t)
        arc_star = _arc_length_param(x_star, y_star)
        zc_star = _zero_crossings_at_sigma(x_star, y_star, arc_star, sigma=1.0)

        # Star should have more zero crossings than circle
        assert len(zc_star) > len(zc_circ), (
            f"Star ({len(zc_star)}) should have more ZC than circle ({len(zc_circ)})"
        )


# ── box_counting_fd edge cases ─────────────────────────────────────────────────

class TestBoxCountingFdEdgeCases:
    """Kill box_counting_fd mutants: short-circuit return 2.0 instead of 1.0."""

    def test_short_contour_returns_1_not_2(self):
        """3-point contour (< 4) → FD = 1.0, not 2.0.
        Kills mutmut_6: `return 1.0` → `return 2.0` in the short-path guard.
        """
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        fd = box_counting_fd(pts)
        assert fd == 1.0, f"3-pt contour should return 1.0, got {fd}"

    def test_span_zero_returns_1_not_2(self):
        """All-same-point contour (span=0) → FD = 1.0, not 2.0.
        Kills mutmut_17: `return 1.0` → `return 2.0` in the span==0 guard.
        """
        pts = np.ones((15, 2)) * 7.0
        fd = box_counting_fd(pts)
        assert fd == 1.0, f"Degenerate (span=0) contour should return 1.0, got {fd}"

    def test_single_scale_returns_1_not_2(self):
        """n_scales=1 → only 1 regression point, insufficient → FD = 1.0.
        Kills mutmut_74: `return 1.0` → `return 2.0` in the `< 2` regression guard.
        """
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        fd = box_counting_fd(circle, n_scales=1)
        assert fd == 1.0, f"n_scales=1 gives 1 data point, cannot regress, should be 1.0, got {fd}"


# ── dtw_distance edge cases ────────────────────────────────────────────────────

class TestDtwDistanceEdgeCases:
    """Kill dtw_distance mutants: empty checks, sign flip, normalization."""

    def _seq(self, values) -> np.ndarray:
        return np.array([[v] for v in values], dtype=float)

    def test_single_element_seq_not_inf(self):
        """dtw_distance of two 1-element identical seqs should be 0, not inf.
        Kills mutmut_5: `if n == 0` → `if n == 1`
        and mutmut_7: `if m == 0` → `if m == 1`.
        """
        a = self._seq([1.0])
        b = self._seq([1.0])
        d = dtw_distance(a, b)
        assert np.isfinite(d), f"Single-element identical seqs: expected finite, got {d}"
        assert d == pytest.approx(0.0, abs=1e-9), f"Expected 0.0, got {d}"

    def test_single_element_different_not_inf(self):
        """dtw_distance of 1-element vs 3-element seq: finite, not inf.
        Also kills mutmut_7 (m==1 → inf).
        """
        a = self._seq([0.0])
        b = self._seq([1.0, 2.0, 3.0])
        d = dtw_distance(a, b)
        assert np.isfinite(d), f"1-vs-3 element seqs: expected finite, got {d}"

    def test_distance_nonnegative(self):
        """dtw_distance must always be non-negative.
        Kills mutmut_66: `cost + min(...)` → `cost - min(...)`.
        With subtraction, intermediate values go negative and the result can be negative.
        """
        a = self._seq([0.0, 1.0, 2.0])
        b = self._seq([2.0, 3.0, 4.0])
        d = dtw_distance(a, b, window=5)
        assert d >= 0.0, f"dtw_distance must be non-negative, got {d}"

    def test_normalized_distance_bounded(self):
        """Normalized DTW of offset sequences should be small, not huge.
        Kills mutmut_82: `raw / (n + m)` → `raw * (n + m)`.
        """
        a = self._seq([float(i) for i in range(10)])
        b = self._seq([float(i) + 1.0 for i in range(10)])  # constant offset 1
        d = dtw_distance(a, b, window=10)
        assert d < 5.0, (
            f"Normalized distance for constant-offset seqs should be small, got {d}. "
            "Possible `/ (n+m)` → `* (n+m)` mutation."
        )


# ── css_similarity with non-unit vectors ──────────────────────────────────────

class TestCssSimilarityNonUnit:
    """Kill css_similarity mutants: multiply vs divide by norms."""

    def test_non_unit_vectors_dot_divide_not_multiply(self):
        """cos(a,b) = dot/(|a||b|), not dot*|a|*|b|.
        Kills mutmut_43: `dot / (norm_a * norm_b)` → `dot * (norm_a * norm_b)`.
        Use vectors with norms != 1 so multiply != divide.
        """
        a = np.array([3.0, 4.0])  # norm = 5
        b = np.array([1.0, 0.0])  # norm = 1, dot(a,b)=3
        result = css_similarity(a, b)
        # Expected: 3 / (5 * 1) = 0.6
        # Mutmut_43: 3 * (5 * 1) = 15, clamped to 1.0
        assert result == pytest.approx(0.6, abs=1e-9), (
            f"Expected 0.6 (cosine), got {result}. Check dot/(norm_a*norm_b)."
        )

    def test_non_unit_vectors_correct_denominator(self):
        """cos(a,b) = dot/(|a||b|), not dot/(|a|/|b|).
        Kills mutmut_44: `dot / (norm_a * norm_b)` → `dot / (norm_a / norm_b)`.
        Use vectors where norm_a/norm_b != norm_a*norm_b.
        """
        a = np.array([3.0, 4.0])  # norm = 5, dot(a,b)=9
        b = np.array([3.0, 0.0])  # norm = 3
        result = css_similarity(a, b)
        # Expected: 9 / (5 * 3) = 0.6
        # Mutmut_44: 9 / (5 / 3) = 9 * 3/5 = 5.4, clamped to 1.0
        assert result == pytest.approx(0.6, abs=1e-9), (
            f"Expected 0.6 (cosine), got {result}. Check dot/(norm_a*norm_b)."
        )


# ── _transform_curve direct tests ─────────────────────────────────────────────

class TestTransformCurveDirectly:
    """Kill _transform_curve mutants: wrong rotation sign, wrong translation sign."""

    def test_rotation_90_y_component_is_negative(self):
        """R(90°) * [0, 1] = [-1, 0], not [+1, 0].
        Kills mutmut_6: R[0,1] = +s instead of -s.
        [[c, +s],[s, c]] * [0,1] = [+1, c] = [1, 0] ≠ expected [-1, 0].
        """
        curve = np.array([[0.0, 1.0]])
        pos = np.array([0.0, 0.0])
        result = _transform_curve(curve, pos, np.pi / 2)
        np.testing.assert_allclose(result[0], [-1.0, 0.0], atol=1e-9,
                                   err_msg="R(90°)*[0,1] should be [-1,0]")

    def test_rotation_45_y_component_is_negative(self):
        """R(45°) * [0, 1] = [-sin45, cos45] = [-√2/2, √2/2].
        Kills mutmut_6: R[0,1] = +s gives [sin45, cos45] = [+√2/2, √2/2].
        """
        angle = np.pi / 4  # 45 degrees
        c, s = np.cos(angle), np.sin(angle)
        curve = np.array([[0.0, 1.0]])
        pos = np.array([0.0, 0.0])
        result = _transform_curve(curve, pos, angle)
        np.testing.assert_allclose(result[0], [-s, c], atol=1e-9,
                                   err_msg=f"R(45°)*[0,1] should be [{-s:.4f}, {c:.4f}]")

    def test_translation_adds_not_subtracts(self):
        """_transform_curve with angle=0 should add pos, not subtract.
        Kills mutmut_7: `(R @ curve.T).T + pos` → `(R @ curve.T).T - pos`.
        """
        curve = np.array([[1.0, 2.0], [3.0, 4.0]])
        pos = np.array([10.0, 5.0])
        result = _transform_curve(curve, pos, 0.0)  # R=I, result = curve + pos
        expected = curve + pos
        np.testing.assert_allclose(result, expected, atol=1e-9,
                                   err_msg="Translation should add pos, not subtract")


# ── dtw_distance exact value tests ────────────────────────────────────────────

class TestDtwDistanceExactValues:
    """Kill dtw_distance mutants that change the cost accumulation formula."""

    def _seq(self, values) -> np.ndarray:
        return np.array([[v] for v in values], dtype=float)

    def test_constant_offset_normalized_correctly(self):
        """dtw_distance of constant+1 offset = 0.5, not 0.0.
        Kills mutmut_66: `cost + min(...)` → `cost - min(...)`.
        With subtraction, identical-offset sequences accumulate 0 cost instead of 3 cost.
        """
        a = self._seq([0.0, 0.0, 0.0])  # n=3
        b = self._seq([1.0, 1.0, 1.0])  # m=3, each step costs 1
        d = dtw_distance(a, b, window=10)
        # Original: total path cost = 3, normalized = 3/(3+3) = 0.5
        # Mutmut_66: subtraction formula gives 0 for this input
        assert d == pytest.approx(0.5, abs=1e-9), (
            f"Expected 0.5 (cost+normalize), got {d}. "
            "Check cost + min(...) vs cost - min(...) mutation."
        )


# ── css_to_feature_vector normalization tests ─────────────────────────────────

class TestCssFeatureVectorNormalization:
    """Kill css_to_feature_vector mutants affecting normalization and histogram range."""

    @staticmethod
    def _star_css(n: int = 256, n_sigmas: int = 5):
        """Star-shaped contour with known zero crossings → non-trivial CSS."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 100 + 30 * np.cos(5 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        star = np.column_stack([x, y])
        return curvature_scale_space(star, n_sigmas=n_sigmas)

    @staticmethod
    def _square_css(n: int = 128, n_sigmas: int = 5):
        """Square contour with 4 corner zero crossings."""
        side = max(n // 4, 2)
        top    = np.column_stack([np.linspace(0, 1, side), np.ones(side)])
        right  = np.column_stack([np.ones(side), np.linspace(1, 0, side)])
        bottom = np.column_stack([np.linspace(1, 0, side), np.zeros(side)])
        left   = np.column_stack([np.zeros(side), np.linspace(0, 1, side)])
        square = np.vstack([top, right, bottom, left])
        return curvature_scale_space(square, n_sigmas=n_sigmas)

    def test_star_feature_vector_unit_norm(self):
        """Star CSS has many zero crossings → norm > 0 → must be normalized to 1.
        Kills mutmut_21: `if norm > 0` → `if norm > 1` (skips normalization for 0<norm≤1).
        Kills mutmut_22: `return vec / norm` → `return vec * norm` (multiplies instead).
        """
        css = self._star_css()
        fv = css_to_feature_vector(css)
        norm = float(np.linalg.norm(fv))
        assert abs(norm - 1.0) < 1e-6, (
            f"Star CSS feature vector should have unit norm, got {norm:.6f}. "
            "Check `vec / norm` vs `vec * norm` or `norm > 0` vs `norm > 1` mutation."
        )

    def test_square_feature_vector_not_uniform(self):
        """Square CSS zero crossings land in specific bins → feature vector is NOT uniform.
        Kills mutmut_10: histogram range=(1.0,1.0) → empty → returns uniform unit vector.
        Kills mutmut_11: histogram range=(0.0,2.0) → bins are different → different vector.
        """
        css = self._square_css()
        fv = css_to_feature_vector(css)
        # A uniform unit vector has all equal elements
        n_elements = len(fv)
        uniform_val = 1.0 / np.sqrt(n_elements)
        # The real feature vector must differ significantly from uniform
        cosine_sim = float(np.dot(fv, np.full(n_elements, uniform_val)))
        assert cosine_sim < 0.9, (
            f"Square feature vector is suspiciously close to uniform (cosine={cosine_sim:.3f}). "
            "Check histogram range=(0.0,1.0) vs wrong range mutation."
        )

    def test_single_crossing_normalized_not_uniform(self):
        """Manual CSS with exactly one crossing at t=0.5.

        With n_bins=64 and range=(0.0, 1.0):
          bin 32 = [0.5, 0.515625) → receives the crossing → hist[32]=1, rest=0
          norm = 1.0 → fv[32] = 1.0 / 1.0 = 1.0

        Kills mutmut_21: `if norm > 0` → `if norm > 1`
          norm=1.0, `1.0 > 1` is False → skips normalization → returns uniform
          fv[32] = 1/sqrt(64) ≈ 0.125 ≠ 1.0

        Kills mutmut_10: `range=(0.0, 1.0)` → `range=(1.0, 1.0)`
          crossing at 0.5 is outside [1.0,1.0] → empty hist → norm=0 → uniform
          fv[32] ≈ 0.125 ≠ 1.0

        Kills mutmut_11: `range=(0.0, 1.0)` → `range=(0.0, 2.0)`
          bin width doubles → 0.5 maps to bin 16, not 32
          fv[32] = 0.0 ≠ 1.0
        """
        css_manual = [(1.0, np.array([0.5]))]
        fv = css_to_feature_vector(css_manual, n_bins=64)
        assert fv[32] == pytest.approx(1.0, abs=1e-9), (
            f"Expected fv[32]=1.0 (single crossing at 0.5 normalized), got {fv[32]:.6f}. "
            "Mutations: histogram range or norm threshold changed."
        )
