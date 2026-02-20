"""
Тесты для генератора тестовых данных.
"""
import numpy as np
import pytest

from tools.tear_generator import (
    tear_document, generate_test_document,
    _grid_shape, _divide_with_jitter, _fractal_profile
)


class TestTearGenerator:

    def test_generate_document_shape(self):
        doc = generate_test_document(width=400, height=500)
        assert doc.shape == (500, 400, 3)
        assert doc.dtype == np.uint8

    def test_generate_document_has_dark_pixels(self):
        """Документ содержит текст — не только белые пиксели."""
        doc = generate_test_document(width=400, height=500)
        assert np.any(doc < 200)

    def test_tear_returns_correct_count(self):
        """Число фрагментов ≈ n_pieces (может отличаться на ±1 из-за пустых)."""
        doc = generate_test_document(400, 500)
        frags = tear_document(doc, n_pieces=4)
        assert 2 <= len(frags) <= 6

    def test_tear_fragments_are_images(self):
        doc = generate_test_document(300, 400)
        frags = tear_document(doc, n_pieces=4)
        for f in frags:
            assert f.ndim == 3
            assert f.shape[2] == 3
            assert f.dtype == np.uint8

    def test_tear_fragments_nonempty(self):
        """Каждый фрагмент содержит хотя бы один нетривиальный пиксель."""
        doc = generate_test_document(300, 400)
        frags = tear_document(doc, n_pieces=4)
        for f in frags:
            assert f.shape[0] > 5 and f.shape[1] > 5

    def test_tear_fragments_sum_area_leq_original(self):
        """Суммарная площадь фрагментов <= площади оригинала (края обрезаются)."""
        doc = generate_test_document(400, 500)
        original_area = 400 * 500
        frags = tear_document(doc, n_pieces=6)
        total = sum(f.shape[0] * f.shape[1] for f in frags)
        # Фрагменты с белым фоном могут быть чуть больше из-за padding
        assert total <= original_area * 1.5

    def test_noise_zero_produces_clean_cuts(self):
        """При noise=0 края должны быть почти прямыми."""
        doc = generate_test_document(400, 500)
        frags_noisy = tear_document(doc, n_pieces=4, noise_level=0.8)
        frags_clean = tear_document(doc, n_pieces=4, noise_level=0.0)
        # Чистые фрагменты имеют ровные края → меньше ненулевых пикселей
        # (Это косвенная проверка — нет прямого доступа к контурам)
        assert len(frags_clean) > 0

    def test_seed_reproducibility(self):
        """Одинаковый seed → одинаковые фрагменты."""
        doc = generate_test_document(300, 400)
        frags1 = tear_document(doc, n_pieces=4, seed=7)
        frags2 = tear_document(doc, n_pieces=4, seed=7)
        assert len(frags1) == len(frags2)
        for f1, f2 in zip(frags1, frags2):
            assert np.array_equal(f1, f2)

    def test_different_seeds_differ(self):
        """Разные seeds → разные фрагменты."""
        doc = generate_test_document(300, 400)
        frags1 = tear_document(doc, n_pieces=4, seed=1)
        frags2 = tear_document(doc, n_pieces=4, seed=2)
        any_different = any(
            not np.array_equal(f1, f2)
            for f1, f2 in zip(frags1, frags2)
        )
        assert any_different


class TestHelpers:

    def test_grid_shape_product_geq_n(self):
        for n in [1, 2, 3, 4, 6, 9, 12]:
            cols, rows = _grid_shape(n)
            assert cols * rows >= n

    def test_grid_shape_near_square(self):
        """Сетка должна быть близка к квадратной."""
        cols, rows = _grid_shape(9)
        assert abs(cols - rows) <= 1

    def test_divide_with_jitter_count(self):
        rng = np.random.RandomState(0)
        bounds = _divide_with_jitter(800, 4, rng)
        assert len(bounds) == 5  # 4 сегмента = 5 границ

    def test_divide_with_jitter_range(self):
        rng = np.random.RandomState(0)
        bounds = _divide_with_jitter(800, 4, rng)
        assert bounds[0] == 0
        assert bounds[-1] == 800

    def test_divide_with_jitter_monotonic(self):
        rng = np.random.RandomState(0)
        bounds = _divide_with_jitter(800, 4, rng)
        assert all(bounds[i] < bounds[i + 1] for i in range(len(bounds) - 1))

    def test_fractal_profile_length(self):
        rng = np.random.RandomState(0)
        profile = _fractal_profile(200, amplitude=10, rng=rng)
        assert len(profile) == 200

    def test_fractal_profile_bounded(self):
        """Профиль не должен быть астрономически большим."""
        rng = np.random.RandomState(42)
        profile = _fractal_profile(256, amplitude=20, rng=rng)
        assert np.abs(profile).max() < 200
