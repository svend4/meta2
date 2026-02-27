"""Extra tests for puzzle_reconstruction/algorithms/wavelet_descriptor.py"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.wavelet_descriptor import (
    WaveletDescriptor,
    _haar_dwt_1d,
    _next_pow2,
    _normalise_contour,
    _resample_contour,
    batch_wavelet_similarity,
    compute_wavelet_descriptor,
    wavelet_similarity,
    wavelet_similarity_mirror,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n: int = 64) -> np.ndarray:
    k = n // 4
    top    = np.column_stack([np.linspace(0, 1, k), np.ones(k)])
    right  = np.column_stack([np.ones(k), np.linspace(1, 0, k)])
    bottom = np.column_stack([np.linspace(1, 0, k), np.zeros(k)])
    left   = np.column_stack([np.zeros(k), np.linspace(0, 1, k)])
    return np.vstack([top, right, bottom, left])


def _line(n: int = 64) -> np.ndarray:
    return np.column_stack([np.linspace(0, 1, n), np.zeros(n)])


def _zigzag(n: int = 64) -> np.ndarray:
    x = np.linspace(0, 1, n)
    y = np.where(np.arange(n) % 2 == 0, 0.0, 0.5)
    return np.column_stack([x, y])


# ── _next_pow2 edge cases ─────────────────────────────────────────────────────

class TestNextPow2Extra:

    @pytest.mark.parametrize("n, expected", [
        (1, 1), (2, 2), (3, 4), (7, 8), (8, 8), (9, 16),
        (100, 128), (127, 128), (128, 128), (129, 256), (255, 256), (256, 256),
    ])
    def test_boundary_values(self, n, expected):
        assert _next_pow2(n) == expected

    def test_result_is_power_of_two(self):
        for n in range(1, 200):
            p = _next_pow2(n)
            # p is a power of 2 iff p & (p-1) == 0
            assert p & (p - 1) == 0

    def test_result_ge_input(self):
        for n in range(1, 300):
            assert _next_pow2(n) >= n

    def test_result_lt_double_input(self):
        for n in range(1, 300):
            # For n > 1, 2^k is within [n, 2n)
            assert _next_pow2(n) < 2 * n + 1


# ── _haar_dwt_1d additional ───────────────────────────────────────────────────

class TestHaarDWTExtra:

    def test_level_count_for_length_8(self):
        x = np.ones(8)
        details = _haar_dwt_1d(x)
        # length 8 → 3 levels (8→4, 4→2, 2→1)
        assert len(details) == 3

    def test_level_count_for_length_16(self):
        x = np.ones(16)
        details = _haar_dwt_1d(x)
        assert len(details) == 4

    def test_linear_ramp_details_consistent(self):
        x = np.linspace(0, 1, 8)
        details = _haar_dwt_1d(x)
        assert all(isinstance(d, np.ndarray) for d in details)

    def test_negated_signal_negated_details(self):
        x = np.random.default_rng(7).standard_normal(16)
        d_pos = _haar_dwt_1d(x)
        d_neg = _haar_dwt_1d(-x)
        for a, b in zip(d_pos, d_neg):
            np.testing.assert_allclose(a, -b, atol=1e-12)

    def test_scaled_signal_scaled_details(self):
        x = np.random.default_rng(8).standard_normal(8)
        scale = 3.0
        d_orig = _haar_dwt_1d(x)
        d_scaled = _haar_dwt_1d(x * scale)
        for a, b in zip(d_orig, d_scaled):
            np.testing.assert_allclose(a * scale, b, atol=1e-12)

    def test_length_4_two_levels(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        details = _haar_dwt_1d(x)
        assert len(details) == 2


# ── _resample_contour extra ──────────────────────────────────────────────────

class TestResampleContourExtra:

    def test_larger_target(self):
        c = _circle(16)
        r = _resample_contour(c, 128)
        assert r.shape == (128, 2)
        assert np.all(np.isfinite(r))

    def test_same_size_preserves_endpoints(self):
        c = _line(8)
        r = _resample_contour(c, 8)
        assert r.shape == (8, 2)

    def test_two_points_simple_line(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        r = _resample_contour(c, 4)
        assert r.shape == (4, 2)
        assert np.all(np.isfinite(r))

    def test_output_dtype_float(self):
        c = np.array([[0, 0], [1, 0]], dtype=int)
        r = _resample_contour(c.astype(float), 4)
        assert r.dtype.kind == "f"

    def test_all_finite(self):
        rng = np.random.default_rng(42)
        c = rng.standard_normal((20, 2))
        r = _resample_contour(c, 32)
        assert np.all(np.isfinite(r))


# ── _normalise_contour extra ─────────────────────────────────────────────────

class TestNormaliseContourExtra:

    def test_result_shape_unchanged(self):
        c = _circle(64)
        n = _normalise_contour(c)
        assert n.shape == c.shape

    def test_already_centered_stays_same(self):
        c = _circle(64)  # already centered at origin
        n = _normalise_contour(c)
        np.testing.assert_allclose(n.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_large_translation_removed(self):
        c = _circle(64) + 1000.0
        n = _normalise_contour(c)
        np.testing.assert_allclose(n.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_all_identical_points_returns_zeros(self):
        c = np.tile([5.0, 3.0], (10, 1))
        n = _normalise_contour(c)
        assert np.all(n == 0.0)

    def test_output_all_finite(self):
        c = _square(64)
        n = _normalise_contour(c)
        assert np.all(np.isfinite(n))

    def test_diagonal_of_bounding_box_is_one(self):
        c = _square(64)
        n = _normalise_contour(c)
        diag = np.linalg.norm(n.max(axis=0) - n.min(axis=0))
        assert abs(diag - 1.0) < 1e-9


# ── compute_wavelet_descriptor extra ─────────────────────────────────────────

class TestComputeWaveletDescriptorExtra:

    def test_very_small_n_levels_1(self):
        d = compute_wavelet_descriptor(_circle(64), n_levels=1)
        assert d.n_levels == 1

    def test_n_levels_cap_at_available(self):
        # 4 points → only 2 DWT levels possible
        c = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        d = compute_wavelet_descriptor(c, n_points=4, n_levels=10)
        assert d.n_levels <= 10
        assert d.n_levels >= 1

    def test_energy_length_equals_n_levels(self):
        d = compute_wavelet_descriptor(_circle(64), n_levels=3)
        assert len(d.energy_per_level) == d.n_levels

    def test_all_energy_values_between_0_and_1(self):
        d = compute_wavelet_descriptor(_circle(64), n_levels=4)
        assert np.all(d.energy_per_level >= 0.0)
        assert np.all(d.energy_per_level <= 1.0)

    def test_zigzag_descriptor_valid(self):
        d = compute_wavelet_descriptor(_zigzag(64))
        assert isinstance(d, WaveletDescriptor)
        assert np.all(np.isfinite(d.coeffs))

    def test_large_contour_no_crash(self):
        c = _circle(1000)
        d = compute_wavelet_descriptor(c, n_points=256, n_levels=5)
        assert isinstance(d, WaveletDescriptor)

    def test_n_points_forced_to_power_of_2(self):
        # n_points=10 should be rounded up to 16
        d = compute_wavelet_descriptor(_circle(64), n_points=10)
        assert d.coeffs.ndim == 1
        assert np.all(np.isfinite(d.coeffs))

    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4, 5])
    def test_parametrised_n_levels(self, n_levels):
        d = compute_wavelet_descriptor(_circle(128), n_levels=n_levels)
        assert d.n_levels <= n_levels
        assert len(d.energy_per_level) == d.n_levels

    def test_circle_and_square_give_different_descriptors(self):
        dc = compute_wavelet_descriptor(_circle(64), n_points=64)
        ds = compute_wavelet_descriptor(_square(64), n_points=64)
        # They should not be identical
        assert not np.allclose(dc.coeffs, ds.coeffs)

    def test_line_descriptor_is_valid(self):
        d = compute_wavelet_descriptor(_line(64), n_points=64, n_levels=3)
        assert np.all(np.isfinite(d.coeffs))
        assert d.n_levels >= 1


# ── wavelet_similarity extra ─────────────────────────────────────────────────

class TestWaveletSimilarityExtra:

    def test_nonnegative_result(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_square(64))
        assert wavelet_similarity(da, db) >= 0.0

    def test_leq_1(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_line(64))
        assert wavelet_similarity(da, db) <= 1.0

    def test_unequal_length_handled(self):
        da = WaveletDescriptor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               np.ones(3) / 3.0, 3)
        db = WaveletDescriptor(np.array([1.0, 0.0]),
                               np.ones(1), 1)
        sim = wavelet_similarity(da, db)
        assert 0.0 <= sim <= 1.0

    def test_both_zero_returns_zero(self):
        dz1 = WaveletDescriptor(np.zeros(8), np.zeros(3), 3)
        dz2 = WaveletDescriptor(np.zeros(8), np.zeros(3), 3)
        assert wavelet_similarity(dz1, dz2) == pytest.approx(0.0)

    def test_orthogonal_descriptors_zero_or_low(self):
        # Build two descriptors with orthogonal coefficients
        da = WaveletDescriptor(np.array([1.0, 0.0, 0.0, 0.0]),
                               np.ones(2) / 2.0, 2)
        db = WaveletDescriptor(np.array([0.0, 1.0, 0.0, 0.0]),
                               np.ones(2) / 2.0, 2)
        assert wavelet_similarity(da, db) == pytest.approx(0.0, abs=1e-9)

    def test_parallel_descriptors_one(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        v = v / np.linalg.norm(v)
        da = WaveletDescriptor(v, np.ones(2) / 2.0, 2)
        db = WaveletDescriptor(v, np.ones(2) / 2.0, 2)
        assert wavelet_similarity(da, db) == pytest.approx(1.0, abs=1e-9)


# ── wavelet_similarity_mirror extra ─────────────────────────────────────────

class TestWaveletSimilarityMirrorExtra:

    def test_range_0_1(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_zigzag(64))
        sim = wavelet_similarity_mirror(da, db)
        assert 0.0 <= sim <= 1.0

    def test_symmetric_property(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_square(64))
        assert wavelet_similarity_mirror(da, db) == pytest.approx(
            wavelet_similarity_mirror(db, da), abs=1e-6
        )

    def test_mirror_with_zero_descriptor_zero(self):
        dz = WaveletDescriptor(np.zeros(8), np.zeros(3), 3)
        da = compute_wavelet_descriptor(_circle(64))
        assert wavelet_similarity_mirror(dz, da) == pytest.approx(0.0)

    def test_reversed_descriptor_symmetric_result(self):
        d = compute_wavelet_descriptor(_circle(64))
        d_rev = WaveletDescriptor(
            d.coeffs[::-1].copy(),
            d.energy_per_level[::-1].copy(),
            d.n_levels,
        )
        sim_direct = wavelet_similarity(d, d_rev)
        sim_mirror = wavelet_similarity_mirror(d, d_rev)
        assert sim_mirror >= sim_direct


# ── batch_wavelet_similarity extra ───────────────────────────────────────────

class TestBatchWaveletSimilarityExtra:

    def test_single_candidate(self):
        q = compute_wavelet_descriptor(_circle(64))
        c1 = compute_wavelet_descriptor(_square(64))
        result = batch_wavelet_similarity(q, [c1])
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_all_same_descriptor_all_one(self):
        q = compute_wavelet_descriptor(_circle(64))
        result = batch_wavelet_similarity(q, [q, q, q])
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-9)

    def test_all_zero_descriptors_all_zero(self):
        q = compute_wavelet_descriptor(_circle(64))
        dz = WaveletDescriptor(np.zeros(len(q.coeffs)), np.zeros(q.n_levels), q.n_levels)
        result = batch_wavelet_similarity(q, [dz, dz])
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-9)

    def test_with_mirror_all_ge_without_mirror(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [
            compute_wavelet_descriptor(_circle(64)),
            compute_wavelet_descriptor(_square(64)),
            compute_wavelet_descriptor(_line(64)),
        ]
        direct = batch_wavelet_similarity(q, cands, use_mirror=False)
        mirror = batch_wavelet_similarity(q, cands, use_mirror=True)
        assert np.all(mirror >= direct - 1e-12)

    def test_larger_batch_shape(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [compute_wavelet_descriptor(_circle(64, r=float(i + 1))) for i in range(10)]
        result = batch_wavelet_similarity(q, cands)
        assert result.shape == (10,)

    def test_result_dtype_float(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [compute_wavelet_descriptor(_square(64))]
        result = batch_wavelet_similarity(q, cands)
        assert result.dtype.kind == "f"
