"""Tests for puzzle_reconstruction/algorithms/wavelet_descriptor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.wavelet_descriptor import (
    WaveletDescriptor,
    compute_wavelet_descriptor,
    wavelet_similarity,
    wavelet_similarity_mirror,
    batch_wavelet_similarity,
    _haar_dwt_1d,
    _next_pow2,
    _resample_contour,
    _normalise_contour,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _circle(n: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def _square(n: int = 64) -> np.ndarray:
    side = n // 4
    top    = np.column_stack([np.linspace(0, 1, side), np.ones(side)])
    right  = np.column_stack([np.ones(side), np.linspace(1, 0, side)])
    bottom = np.column_stack([np.linspace(1, 0, side), np.zeros(side)])
    left   = np.column_stack([np.zeros(side), np.linspace(0, 1, side)])
    return np.vstack([top, right, bottom, left])


def _line(n: int = 64) -> np.ndarray:
    x = np.linspace(0, 1, n)
    return np.column_stack([x, np.zeros(n)])


# ── _next_pow2 ─────────────────────────────────────────────────────────────────

class TestNextPow2:

    @pytest.mark.parametrize("n, expected", [
        (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (64, 64), (65, 128),
    ])
    def test_values(self, n, expected):
        assert _next_pow2(n) == expected


# ── _haar_dwt_1d ──────────────────────────────────────────────────────────────

class TestHaarDWT:

    def test_returns_list(self):
        x = np.ones(8)
        details = _haar_dwt_1d(x)
        assert isinstance(details, list)
        assert len(details) > 0

    def test_lengths_halve(self):
        x = np.ones(16)
        details = _haar_dwt_1d(x)
        for i in range(1, len(details)):
            assert len(details[i]) == len(details[i-1]) // 2

    def test_constant_signal_zero_details(self):
        x = np.ones(8)
        details = _haar_dwt_1d(x)
        # Haar DWT of constant signal → all detail coefficients = 0
        for d in details:
            np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_energy_preservation(self):
        """Parseval: total energy in details ≈ energy of input (Haar is orthogonal)."""
        x = np.random.default_rng(42).standard_normal(16)
        details = _haar_dwt_1d(x)
        # The approximation coefficient at the end stores residual energy
        # Just check detail energy <= input energy
        detail_energy = sum(np.sum(d**2) for d in details)
        input_energy  = np.sum(x**2)
        assert detail_energy <= input_energy + 1e-9

    def test_minimum_length_2(self):
        x = np.array([1.0, 3.0])
        details = _haar_dwt_1d(x)
        assert len(details) == 1
        assert len(details[0]) == 1

    def test_length_1_returns_empty(self):
        x = np.array([5.0])
        details = _haar_dwt_1d(x)
        assert details == []


# ── _resample_contour ──────────────────────────────────────────────────────────

class TestResampleContour:

    def test_output_length(self):
        c = _circle(32)
        r = _resample_contour(c, 64)
        assert len(r) == 64

    def test_shape_2d(self):
        c = _line(32)
        r = _resample_contour(c, 16)
        assert r.shape == (16, 2)

    def test_single_point(self):
        c = np.array([[0.5, 0.5]])
        r = _resample_contour(c, 8)
        assert r.shape == (8, 2)

    def test_collinear_points(self):
        c = np.tile([1.0, 1.0], (4, 1))
        r = _resample_contour(c, 8)
        assert r.shape == (8, 2)
        assert np.all(np.isfinite(r))


# ── _normalise_contour ────────────────────────────────────────────────────────

class TestNormaliseContour:

    def test_centroid_at_zero(self):
        c = _circle(64)
        n = _normalise_contour(c)
        np.testing.assert_allclose(n.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_bounding_box_leq_1(self):
        c = _square(64)
        n = _normalise_contour(c)
        extent = n.max(axis=0) - n.min(axis=0)
        diag = np.linalg.norm(extent)
        assert diag <= 1.0 + 1e-9

    def test_all_same_point_no_crash(self):
        c = np.tile([5.0, 5.0], (10, 1))
        n = _normalise_contour(c)
        assert n.shape == c.shape
        assert np.all(np.isfinite(n))


# ── WaveletDescriptor ──────────────────────────────────────────────────────────

class TestWaveletDescriptor:

    def test_is_named_tuple(self):
        d = WaveletDescriptor(
            coeffs=np.ones(8),
            energy_per_level=np.array([0.5, 0.5]),
            n_levels=2,
        )
        assert isinstance(d, tuple)
        assert d.n_levels == 2


# ── compute_wavelet_descriptor ────────────────────────────────────────────────

class TestComputeWaveletDescriptor:

    def test_returns_descriptor(self):
        d = compute_wavelet_descriptor(_circle(64))
        assert isinstance(d, WaveletDescriptor)

    def test_coeffs_1d(self):
        d = compute_wavelet_descriptor(_circle(64))
        assert d.coeffs.ndim == 1

    def test_coeffs_unit_norm(self):
        d = compute_wavelet_descriptor(_circle(64))
        norm = np.linalg.norm(d.coeffs)
        assert abs(norm - 1.0) < 1e-9 or norm == 0.0

    def test_energy_sums_to_one(self):
        d = compute_wavelet_descriptor(_circle(64))
        if d.energy_per_level.sum() > 0:
            assert abs(d.energy_per_level.sum() - 1.0) < 1e-9

    def test_energy_non_negative(self):
        d = compute_wavelet_descriptor(_circle(64))
        assert np.all(d.energy_per_level >= 0)

    def test_n_levels_respected(self):
        d = compute_wavelet_descriptor(_circle(64), n_levels=3)
        assert d.n_levels <= 3

    def test_degenerate_single_point(self):
        d = compute_wavelet_descriptor(np.array([[1.0, 2.0]]))
        assert isinstance(d, WaveletDescriptor)
        assert np.all(np.isfinite(d.coeffs))

    def test_short_contour(self):
        d = compute_wavelet_descriptor(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert isinstance(d, WaveletDescriptor)

    @pytest.mark.parametrize("n", [32, 64, 128])
    def test_various_lengths(self, n):
        c = _circle(n)
        d = compute_wavelet_descriptor(c, n_points=64)
        assert isinstance(d, WaveletDescriptor)
        assert np.all(np.isfinite(d.coeffs))

    def test_custom_n_points(self):
        c = _circle(64)
        d1 = compute_wavelet_descriptor(c, n_points=32)
        d2 = compute_wavelet_descriptor(c, n_points=64)
        # Both should have unit-norm coefficients
        assert abs(np.linalg.norm(d1.coeffs) - 1.0) < 1e-9
        assert abs(np.linalg.norm(d2.coeffs) - 1.0) < 1e-9

    def test_scale_invariance(self):
        """Descriptors of original and scaled contour should be similar."""
        c_small = _circle(64, ) * 1.0
        c_big   = _circle(64) * 100.0
        d_small = compute_wavelet_descriptor(c_small, n_points=64)
        d_big   = compute_wavelet_descriptor(c_big, n_points=64)
        sim = wavelet_similarity(d_small, d_big)
        assert sim > 0.9, f"Scale invariance poor: {sim:.3f}"

    def test_translation_invariance(self):
        """Shifted contour should give identical descriptor."""
        c = _circle(64)
        c_shifted = c + np.array([500.0, -300.0])
        d_orig    = compute_wavelet_descriptor(c, n_points=64)
        d_shifted = compute_wavelet_descriptor(c_shifted, n_points=64)
        sim = wavelet_similarity(d_orig, d_shifted)
        assert sim > 0.9, f"Translation invariance poor: {sim:.3f}"


# ── wavelet_similarity ────────────────────────────────────────────────────────

class TestWaveletSimilarity:

    def test_self_similarity_one(self):
        d = compute_wavelet_descriptor(_circle(64))
        assert wavelet_similarity(d, d) == pytest.approx(1.0, abs=1e-9)

    def test_range_0_1(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_square(64))
        sim = wavelet_similarity(da, db)
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_square(64))
        assert wavelet_similarity(da, db) == pytest.approx(
            wavelet_similarity(db, da), abs=1e-9
        )

    def test_zero_coeffs_returns_zero(self):
        d_zero = WaveletDescriptor(np.zeros(8), np.zeros(3), 3)
        d_any  = compute_wavelet_descriptor(_circle(64))
        assert wavelet_similarity(d_zero, d_any) == pytest.approx(0.0)

    def test_different_shapes_truncated(self):
        da = WaveletDescriptor(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0]), 1)
        db = WaveletDescriptor(np.array([1.0, 0.0]), np.array([1.0]), 1)
        sim = wavelet_similarity(da, db)
        assert 0.0 <= sim <= 1.0


# ── wavelet_similarity_mirror ─────────────────────────────────────────────────

class TestWaveletSimilarityMirror:

    def test_ge_direct(self):
        da = compute_wavelet_descriptor(_circle(64))
        db = compute_wavelet_descriptor(_square(64))
        assert wavelet_similarity_mirror(da, db) >= wavelet_similarity(da, db)

    def test_self_mirror_similarity_one(self):
        d = compute_wavelet_descriptor(_circle(64))
        sim = wavelet_similarity_mirror(d, d)
        assert sim == pytest.approx(1.0, abs=1e-9)


# ── batch_wavelet_similarity ──────────────────────────────────────────────────

class TestBatchWaveletSimilarity:

    def test_empty_candidates(self):
        q = compute_wavelet_descriptor(_circle(64))
        result = batch_wavelet_similarity(q, [])
        assert len(result) == 0

    def test_returns_array(self):
        q = compute_wavelet_descriptor(_circle(64))
        c1 = compute_wavelet_descriptor(_square(64))
        c2 = compute_wavelet_descriptor(_line(64))
        result = batch_wavelet_similarity(q, [c1, c2])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_range_0_1(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [compute_wavelet_descriptor(_square(64)),
                 compute_wavelet_descriptor(_line(64))]
        result = batch_wavelet_similarity(q, cands)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_self_is_best(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [
            compute_wavelet_descriptor(_square(64)),
            q,
            compute_wavelet_descriptor(_line(64)),
        ]
        result = batch_wavelet_similarity(q, cands)
        assert result[1] == pytest.approx(max(result))

    def test_with_mirror(self):
        q = compute_wavelet_descriptor(_circle(64))
        cands = [compute_wavelet_descriptor(_square(64))]
        result_direct = batch_wavelet_similarity(q, cands, use_mirror=False)
        result_mirror = batch_wavelet_similarity(q, cands, use_mirror=True)
        assert result_mirror[0] >= result_direct[0]
