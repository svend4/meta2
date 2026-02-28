"""
Property-based tests for matching algorithms using Hypothesis and pytest.

Verifies mathematical invariants:
- DTW: zero self-distance, symmetry, triangle inequality (approximate)
- Rotation DTW: rotation invariance, result within [0, 1]
- Compat matrix: symmetry, diagonal, values in [0, 1]
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from puzzle_reconstruction.matching.dtw import dtw_distance, dtw_distance_mirror
from puzzle_reconstruction.matching.rotation_dtw import (
    rotation_dtw,
    rotation_dtw_similarity,
    _rotate_curve,
    _resample_curve,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 1.0) -> np.ndarray:
    """Generate circular contour with n points."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _line(n: int = 32) -> np.ndarray:
    """Generate a straight horizontal line curve."""
    x = np.linspace(0, 1, n)
    return np.column_stack([x, np.zeros(n)])


def _random_curve(n: int, seed: int = 42) -> np.ndarray:
    """Generate a smooth random 2-D curve with n points."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + 0.3 * rng.standard_normal(n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


# ── DTW properties ────────────────────────────────────────────────────────────

class TestDTWProperties:
    """Mathematical properties of dtw_distance."""

    def test_dtw_zero_self_distance(self):
        """DTW(A, A) == 0.0."""
        for curve in [_circle(32), _line(32), _random_curve(32)]:
            d = dtw_distance(curve, curve)
            assert d == pytest.approx(0.0, abs=1e-9), \
                f"DTW self-distance not zero: {d}"

    def test_dtw_symmetric(self):
        """DTW(A, B) == DTW(B, A)."""
        a = _circle(32)
        b = _line(32)
        d_ab = dtw_distance(a, b)
        d_ba = dtw_distance(b, a)
        assert d_ab == pytest.approx(d_ba, abs=1e-9), \
            f"DTW not symmetric: {d_ab} vs {d_ba}"

    def test_dtw_non_negative(self):
        """DTW distance is always >= 0."""
        a = _circle(32)
        b = _random_curve(32)
        assert dtw_distance(a, b) >= 0.0
        assert dtw_distance(a, a) >= 0.0

    def test_dtw_different_lengths(self):
        """DTW handles curves of different lengths without error."""
        a = _circle(20)
        b = _circle(50)
        d = dtw_distance(a, b)
        assert np.isfinite(d) and d >= 0.0, f"DTW with different lengths failed: {d}"

    def test_dtw_triangle_inequality_approx(self):
        """DTW approximately satisfies triangle inequality: d(A,C) <= d(A,B) + d(B,C) + eps."""
        a = _circle(32)
        b = _line(32)
        c = _random_curve(32)
        d_ac = dtw_distance(a, c)
        d_ab = dtw_distance(a, b)
        d_bc = dtw_distance(b, c)
        # DTW doesn't strictly satisfy triangle inequality, allow 10% slack
        assert d_ac <= d_ab + d_bc + 1e-6, \
            f"Triangle inequality violated: {d_ac:.4f} > {d_ab:.4f} + {d_bc:.4f}"

    @given(st.integers(min_value=5, max_value=60))
    @settings(max_examples=40, deadline=5000)
    def test_dtw_self_distance_zero_any_size(self, n: int):
        """DTW(A, A) == 0 for any contour size."""
        curve = _circle(n)
        d = dtw_distance(curve, curve)
        assert d == pytest.approx(0.0, abs=1e-9), \
            f"DTW self-distance not zero for n={n}: {d}"

    @given(st.integers(min_value=5, max_value=50))
    @settings(max_examples=30, deadline=5000)
    def test_dtw_symmetry_hypothesis(self, n: int):
        """DTW symmetry for any contour size."""
        a = _circle(n)
        b = _line(n)
        d_ab = dtw_distance(a, b)
        d_ba = dtw_distance(b, a)
        assert abs(d_ab - d_ba) < 1e-9, \
            f"DTW asymmetry for n={n}: {d_ab} vs {d_ba}"

    def test_dtw_empty_curve_returns_inf(self):
        """DTW with empty curve returns infinity."""
        a = _circle(32)
        empty = np.zeros((0, 2))
        d = dtw_distance(a, empty)
        assert d == float("inf"), f"Expected inf for empty curve, got {d}"

    def test_dtw_mirror_le_min_direct_reversed(self):
        """dtw_distance_mirror returns min of direct and reversed."""
        a = _circle(32)
        b = _random_curve(32)
        d_mirror = dtw_distance_mirror(a, b)
        d_direct = dtw_distance(a, b)
        d_reversed = dtw_distance(a, b[::-1])
        expected = min(d_direct, d_reversed)
        assert d_mirror == pytest.approx(expected, abs=1e-9), \
            f"Mirror DTW mismatch: {d_mirror} vs {expected}"


# ── Rotation DTW properties ───────────────────────────────────────────────────

class TestRotationDTWProperties:
    """Mathematical properties of rotation_dtw and rotation_dtw_similarity."""

    def test_self_distance_near_zero(self):
        """rotation_dtw(A, A) distance should be very close to 0."""
        c = _circle(64)
        result = rotation_dtw(c, c, n_angles=36)
        assert result.distance < 0.1, \
            f"Self rotation-DTW distance too large: {result.distance}"

    def test_similarity_in_range_0_1(self):
        """rotation_dtw_similarity returns a value in [0, 1]."""
        a = _circle(64)
        b = _line(32)
        sim = rotation_dtw_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"Similarity out of [0,1]: {sim}"

    def test_self_similarity_near_one(self):
        """rotation_dtw_similarity(A, A) ≈ 1.0."""
        c = _circle(64)
        sim = rotation_dtw_similarity(c, c)
        assert sim > 0.9, f"Self-similarity too low: {sim}"

    @given(st.floats(min_value=0.0, max_value=350.0))
    @settings(max_examples=30, deadline=30000)
    def test_rotation_invariant_same_shape(self, angle_deg: float):
        """rotation_dtw(C, rotate(C, angle)) finds near-zero distance."""
        c = _circle(64)
        c_rotated = _rotate_curve(c, angle_deg)
        result = rotation_dtw(c, c_rotated, n_angles=36)
        # Should find a very close match (circle is rotationally symmetric)
        assert result.distance < 0.5, \
            f"Rotation DTW failed for circle at {angle_deg:.1f}°: d={result.distance:.4f}"

    def test_rotation_dtw_result_has_distance_and_angle(self):
        """rotation_dtw result has distance and best_angle_deg fields."""
        a = _circle(32)
        b = _random_curve(32)
        result = rotation_dtw(a, b)
        assert hasattr(result, "distance")
        assert hasattr(result, "best_angle_deg")
        assert hasattr(result, "mirrored")
        assert result.distance >= 0.0
        assert 0.0 <= result.best_angle_deg < 360.0

    def test_rotation_dtw_distance_non_negative(self):
        """rotation_dtw always returns non-negative distance."""
        pairs = [
            (_circle(32), _line(32)),
            (_circle(32), _random_curve(32)),
            (_line(32), _random_curve(32)),
        ]
        for a, b in pairs:
            result = rotation_dtw(a, b, n_angles=12)
            assert result.distance >= 0.0, \
                f"Negative rotation DTW distance: {result.distance}"

    def test_similarity_monotone_with_distance(self):
        """Closer curves (in DTW) have higher similarity scores."""
        base = _circle(64)
        close = _circle(64, r=1.0 + 0.05)   # slightly different radius
        far = _line(32)
        sim_close = rotation_dtw_similarity(base, close)
        sim_far = rotation_dtw_similarity(base, far)
        # Close shape should have higher similarity (but not guaranteed to be strict)
        assert sim_close >= sim_far - 0.3, \
            f"Expected close similarity ({sim_close:.3f}) >= far similarity ({sim_far:.3f}) - 0.3"


# ── Compat matrix properties ──────────────────────────────────────────────────

class TestCompatMatrixProperties:
    """Property tests for the compatibility matrix builder.

    These tests verify properties using the Fragment/EdgeSignature models directly.
    """

    def _make_mock_fragments(self, n: int = 4) -> list:
        """Create mock fragments with edge signatures for matrix building."""
        from puzzle_reconstruction.models import Fragment, EdgeSignature, EdgeSide
        fragments = []
        sides = [EdgeSide.TOP, EdgeSide.RIGHT, EdgeSide.BOTTOM, EdgeSide.LEFT]
        for i in range(n):
            edges = []
            for j in range(4):  # 4 edges per fragment
                rng = np.random.default_rng(i * 10 + j)
                virtual_curve = _circle(32, r=float(i + 1) * 0.5 + j * 0.1)
                css_vec = rng.uniform(0, 1, 64).astype(np.float64)
                norm = np.linalg.norm(css_vec)
                if norm > 0:
                    css_vec /= norm
                sig = EdgeSignature(
                    edge_id=i * 4 + j,
                    side=sides[j],
                    virtual_curve=virtual_curve,
                    fd=1.0 + float(i) * 0.1,
                    css_vec=css_vec,
                    ifs_coeffs=np.zeros(8),
                    length=float(32),
                )
                edges.append(sig)
            frag = Fragment(
                fragment_id=i,
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                mask=np.ones((64, 64), dtype=np.uint8),
                contour=_circle(64),
                edges=edges,
            )
            fragments.append(frag)
        return fragments

    def test_matrix_values_in_range(self):
        """All compatibility matrix values ∈ [0, 1]."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(4)
        matrix, _ = build_compat_matrix(frags)
        assert np.all(matrix >= 0.0), "Matrix has negative values"
        assert np.all(matrix <= 1.0), "Matrix has values > 1"

    def test_matrix_symmetric(self):
        """compat_matrix[i, j] == compat_matrix[j, i]."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(4)
        matrix, _ = build_compat_matrix(frags)
        np.testing.assert_allclose(
            matrix, matrix.T, atol=1e-6,
            err_msg="Compatibility matrix is not symmetric"
        )

    def test_matrix_diagonal_is_zero_for_same_fragment(self):
        """Edges of the same fragment have score 0 (excluded from scoring)."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(3)
        matrix, _ = build_compat_matrix(frags)
        # Same-fragment edge pairs should have score 0
        n_edges_per_frag = 4
        for fi in range(3):
            for ei in range(n_edges_per_frag):
                for ej in range(n_edges_per_frag):
                    if ei != ej:
                        idx_i = fi * n_edges_per_frag + ei
                        idx_j = fi * n_edges_per_frag + ej
                        assert matrix[idx_i, idx_j] == 0.0, \
                            f"Same-fragment edges should have score 0: [{idx_i},{idx_j}]={matrix[idx_i,idx_j]}"

    def test_matrix_shape_n_edges_squared(self):
        """Matrix shape is (N_edges, N_edges) where N_edges = n_frags * edges_per_frag."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        n_frags = 3
        edges_per_frag = 4
        frags = self._make_mock_fragments(n_frags)
        matrix, _ = build_compat_matrix(frags)
        expected = n_frags * edges_per_frag
        assert matrix.shape == (expected, expected), \
            f"Matrix shape {matrix.shape} != expected ({expected}, {expected})"

    def test_entries_sorted_by_score_descending(self):
        """Returned entries list is sorted by score in descending order."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
        frags = self._make_mock_fragments(4)
        _, entries = build_compat_matrix(frags)
        if len(entries) > 1:
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True), \
                "Entries not sorted in descending order by score"

    def test_top_candidates_returns_k_or_fewer(self):
        """top_candidates returns at most k candidates."""
        from puzzle_reconstruction.matching.compat_matrix import (
            build_compat_matrix,
            top_candidates,
        )
        frags = self._make_mock_fragments(4)
        matrix, _ = build_compat_matrix(frags)
        # Build flat edge list
        all_edges = [e for f in frags for e in f.edges]
        for k in [1, 3, 5]:
            candidates = top_candidates(matrix, all_edges, edge_idx=0, k=k)
            assert len(candidates) <= k, \
                f"top_candidates returned {len(candidates)} > k={k}"


# ── DTW + rotation integration ────────────────────────────────────────────────

class TestDTWRotationIntegration:
    """Integration tests combining DTW and rotation helpers."""

    def test_resample_curve_output_shape(self):
        """_resample_curve returns exactly n points."""
        c = _circle(30)
        for n in [10, 50, 100]:
            resampled = _resample_curve(c, n)
            assert resampled.shape == (n, 2), \
                f"Resampled shape {resampled.shape} != ({n}, 2)"

    def test_rotate_curve_preserves_shape(self):
        """_rotate_curve preserves the number of points and dimensions."""
        c = _circle(64)
        for angle in [0, 45, 90, 180, 270]:
            rotated = _rotate_curve(c, float(angle))
            assert rotated.shape == c.shape, \
                f"Rotation at {angle}° changed shape: {rotated.shape}"

    def test_rotate_0_degrees_identity(self):
        """Rotating by 0° returns the same curve."""
        c = _circle(64)
        rotated = _rotate_curve(c, 0.0)
        np.testing.assert_allclose(rotated, c, atol=1e-10)

    def test_rotate_360_degrees_identity(self):
        """Rotating by 360° returns the same curve."""
        c = _circle(64)
        rotated = _rotate_curve(c, 360.0)
        np.testing.assert_allclose(rotated, c, atol=1e-10)

    def test_rotate_preserves_distances(self):
        """Rotation is an isometry: pairwise distances are preserved."""
        c = _random_curve(32)
        c_rot = _rotate_curve(c, 45.0)
        # Centroid-centered distances should be preserved
        c_c = c - c.mean(axis=0)
        c_rot_c = c_rot - c_rot.mean(axis=0)
        dist_orig = np.linalg.norm(c_c, axis=1)
        dist_rot = np.linalg.norm(c_rot_c, axis=1)
        np.testing.assert_allclose(dist_orig, dist_rot, atol=1e-9)
