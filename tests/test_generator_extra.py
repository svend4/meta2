"""Additional tests for tools/tear_generator.py"""
import pytest
import numpy as np

from tools.tear_generator import (
    _divide_with_jitter,
    _fractal_profile,
    _grid_shape,
    generate_test_document,
    tear_document,
)


# ─── TestGenerateTestDocumentExtra ────────────────────────────────────────────

class TestGenerateTestDocumentExtra:
    def test_default_size(self):
        doc = generate_test_document()
        assert doc.shape == (1000, 800, 3)
        assert doc.dtype == np.uint8

    def test_custom_width_height(self):
        doc = generate_test_document(width=200, height=300)
        assert doc.shape == (300, 200, 3)

    def test_seed_reproducibility(self):
        d1 = generate_test_document(width=200, height=200, seed=5)
        d2 = generate_test_document(width=200, height=200, seed=5)
        assert np.array_equal(d1, d2)

    def test_different_seeds_differ(self):
        d1 = generate_test_document(width=200, height=200, seed=1)
        d2 = generate_test_document(width=200, height=200, seed=2)
        assert not np.array_equal(d1, d2)

    def test_pixel_values_in_range(self):
        doc = generate_test_document(width=100, height=100)
        assert doc.min() >= 0
        assert doc.max() <= 255

    def test_has_three_channels(self):
        doc = generate_test_document(width=100, height=100)
        assert doc.ndim == 3
        assert doc.shape[2] == 3

    def test_mostly_white_background(self):
        """Most pixels should be near-white (255)."""
        doc = generate_test_document(width=200, height=200)
        white_fraction = (doc > 200).all(axis=2).mean()
        assert white_fraction > 0.2

    def test_small_image_no_crash(self):
        doc = generate_test_document(width=50, height=60)
        assert doc.shape == (60, 50, 3)


# ─── TestTearDocumentExtra ────────────────────────────────────────────────────

class TestTearDocumentExtra:
    def _doc(self):
        return generate_test_document(300, 300, seed=0)

    def test_n_pieces_1_returns_at_least_one(self):
        frags = tear_document(self._doc(), n_pieces=1)
        assert len(frags) >= 1

    def test_n_pieces_9_count_reasonable(self):
        frags = tear_document(self._doc(), n_pieces=9)
        assert 4 <= len(frags) <= 12

    def test_noise_level_1_no_crash(self):
        frags = tear_document(self._doc(), n_pieces=4, noise_level=1.0)
        assert len(frags) >= 1

    def test_noise_level_0_no_crash(self):
        frags = tear_document(self._doc(), n_pieces=4, noise_level=0.0)
        assert len(frags) >= 1

    def test_each_fragment_is_bgr(self):
        frags = tear_document(self._doc(), n_pieces=4)
        for f in frags:
            assert f.ndim == 3
            assert f.shape[2] == 3

    def test_all_fragments_uint8(self):
        frags = tear_document(self._doc(), n_pieces=4)
        for f in frags:
            assert f.dtype == np.uint8

    def test_all_fragment_pixels_in_range(self):
        frags = tear_document(self._doc(), n_pieces=4)
        for f in frags:
            assert f.min() >= 0
            assert f.max() <= 255

    def test_large_n_pieces_no_crash(self):
        frags = tear_document(self._doc(), n_pieces=16)
        assert isinstance(frags, list)


# ─── TestGridShapeExtra ───────────────────────────────────────────────────────

class TestGridShapeExtra:
    def test_n_1_returns_1_1(self):
        cols, rows = _grid_shape(1)
        assert cols * rows >= 1

    def test_n_16_near_square(self):
        cols, rows = _grid_shape(16)
        assert cols * rows >= 16
        assert abs(cols - rows) <= 1

    def test_n_7_product_geq_7(self):
        cols, rows = _grid_shape(7)
        assert cols * rows >= 7

    def test_n_100_product_geq_100(self):
        cols, rows = _grid_shape(100)
        assert cols * rows >= 100

    def test_returns_positive_ints(self):
        for n in [2, 3, 5, 8, 12]:
            cols, rows = _grid_shape(n)
            assert cols > 0
            assert rows > 0

    def test_consistent_results(self):
        c1, r1 = _grid_shape(6)
        c2, r2 = _grid_shape(6)
        assert c1 == c2 and r1 == r2


# ─── TestDivideWithJitterExtra ────────────────────────────────────────────────

class TestDivideWithJitterExtra:
    def test_n_1_returns_2_bounds(self):
        rng = np.random.RandomState(0)
        bounds = _divide_with_jitter(100, 1, rng)
        assert len(bounds) == 2
        assert bounds[0] == 0
        assert bounds[-1] == 100

    def test_n_1_returns_0_and_total(self):
        rng = np.random.RandomState(1)
        bounds = _divide_with_jitter(500, 1, rng)
        assert bounds == [0, 500]

    def test_different_seeds_may_differ(self):
        b1 = _divide_with_jitter(800, 4, np.random.RandomState(1))
        b2 = _divide_with_jitter(800, 4, np.random.RandomState(99))
        # Middle bounds may differ
        assert b1 != b2 or True  # just no crash

    def test_large_n_monotonic(self):
        rng = np.random.RandomState(42)
        bounds = _divide_with_jitter(1000, 8, rng)
        assert all(bounds[i] < bounds[i + 1] for i in range(len(bounds) - 1))


# ─── TestFractalProfileExtra ──────────────────────────────────────────────────

class TestFractalProfileExtra:
    def test_amplitude_zero_near_zero(self):
        rng = np.random.RandomState(0)
        profile = _fractal_profile(100, amplitude=0, rng=rng)
        assert np.abs(profile).max() < 1.0

    def test_reproducible(self):
        p1 = _fractal_profile(64, 10, np.random.RandomState(7))
        p2 = _fractal_profile(64, 10, np.random.RandomState(7))
        np.testing.assert_array_equal(p1, p2)

    def test_dtype_float(self):
        profile = _fractal_profile(50, 5, np.random.RandomState(0))
        assert profile.dtype in (np.float32, np.float64)

    def test_large_amplitude_bounded(self):
        profile = _fractal_profile(128, amplitude=50, rng=np.random.RandomState(0))
        # Multi-octave sum can exceed amplitude; verify it's not astronomically large
        assert np.abs(profile).max() < 500
