"""
Тесты для puzzle_reconstruction/algorithms/shape_context.py

Покрытие:
    log_polar_histogram     — форма, неотрицательность, все точки в диапазоне r
    compute_shape_context   — ValueError для неправильного input,
                              форма дескрипторов, L1-нормализация,
                              вырожденный N=1, разные контуры → разные SC,
                              одинаковые точки → одинаковые SC
    normalize_shape_context — нулевой вектор, единичный вектор, L1=1 после нормализации
    shape_context_distance  — SC = SC → 0, разные формы → > 0,
                              несовпадающие формы → ValueError
    match_shape_contexts    — возвращает (float, ndarray), cost≥0,
                              correspondence имеет правильную форму, нет дубликатов
    contour_similarity      — пустой/одноточечный → 0.0, один контур с собой → ≈1.0,
                              разные контуры → < 1.0, значение ∈ [0,1]
    _preprocess_contour     — 3D контур (N,1,2), различное n_sample
    ShapeContextResult      — repr, descriptor_dim
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.shape_context import (
    ShapeContextResult,
    _preprocess_contour,
    compute_shape_context,
    contour_similarity,
    log_polar_histogram,
    match_shape_contexts,
    normalize_shape_context,
    shape_context_distance,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _square_contour(n: int = 40, size: float = 100.0) -> np.ndarray:
    """Равномерно распределённые точки на периметре квадрата."""
    side  = n // 4
    pts   = []
    for i in range(side):
        pts.append([i * size / side, 0.0])
    for i in range(side):
        pts.append([size, i * size / side])
    for i in range(side):
        pts.append([size - i * size / side, size])
    for i in range(side):
        pts.append([0.0, size - i * size / side])
    return np.array(pts, dtype=np.float64)


def _circle_contour(n: int = 40, r: float = 50.0) -> np.ndarray:
    """Точки на окружности."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)


@pytest.fixture
def square():
    return _square_contour(40, 100.0)


@pytest.fixture
def circle():
    return _circle_contour(40, 50.0)


# ─── log_polar_histogram ──────────────────────────────────────────────────────

class TestLogPolarHistogram:
    def _make_bins(self, n_r=5, n_t=12):
        r_bins     = np.logspace(-1, 2, n_r + 1)
        theta_bins = np.linspace(-np.pi, np.pi, n_t + 1)
        return r_bins, theta_bins

    def test_output_shape(self):
        n_r, n_t   = 5, 12
        r_b, t_b   = self._make_bins(n_r, n_t)
        dists  = np.array([1.0, 2.0, 5.0, 10.0])
        angles = np.array([0.0, 1.0, -1.0, 2.0])
        h = log_polar_histogram(dists, angles, r_b, t_b, n_r, n_t)
        assert h.shape == (n_r * n_t,)

    def test_nonnegative(self):
        r_b, t_b = self._make_bins()
        dists  = np.random.rand(20) * 100
        angles = np.random.uniform(-np.pi, np.pi, 20)
        h = log_polar_histogram(dists, angles, r_b, t_b, 5, 12)
        assert (h >= 0).all()

    def test_empty_points(self):
        r_b, t_b = self._make_bins()
        h = log_polar_histogram(np.array([]), np.array([]), r_b, t_b, 5, 12)
        assert h.shape == (60,)
        assert (h == 0).all()

    def test_out_of_range_excluded(self):
        """Точки вне [r_min, r_max) не попадают в гистограмму."""
        r_b    = np.array([10.0, 50.0, 100.0])  # 2 кольца
        t_b    = np.linspace(-np.pi, np.pi, 5)   # 4 сектора
        dists  = np.array([0.5, 1000.0])          # Оба вне диапазона
        angles = np.zeros(2)
        h = log_polar_histogram(dists, angles, r_b, t_b, 2, 4)
        assert h.sum() == 0


# ─── compute_shape_context ────────────────────────────────────────────────────

class TestComputeShapeContext:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            compute_shape_context(np.zeros((10, 3)))

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            compute_shape_context(np.zeros(10))

    def test_output_shape(self, square):
        r = compute_shape_context(square, n_bins_r=5, n_bins_theta=12)
        assert r.descriptors.shape == (len(square), 5 * 12)

    def test_descriptor_dim_property(self, square):
        r = compute_shape_context(square, n_bins_r=4, n_bins_theta=8)
        assert r.descriptor_dim == 32

    def test_single_point(self):
        """N=1 → вырожденный случай, нет ошибки."""
        pts = np.array([[0.0, 0.0]])
        r   = compute_shape_context(pts)
        assert r.descriptors.shape[0] == 1

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        r   = compute_shape_context(pts)
        assert r.descriptors.shape == (2, r.descriptor_dim)

    def test_l1_normalized(self, square):
        r = compute_shape_context(square, normalize=True)
        for desc in r.descriptors:
            total = desc.sum()
            assert total == pytest.approx(1.0) or total == pytest.approx(0.0)

    def test_not_normalized(self, square):
        r = compute_shape_context(square, normalize=False)
        # Суммы не должны быть все ≈1
        totals = r.descriptors.sum(axis=1)
        assert not np.allclose(totals, 1.0)  # Хотя бы некоторые отличаются

    def test_mean_dist_positive(self, square):
        r = compute_shape_context(square)
        assert r.mean_dist > 0.0

    def test_different_shapes_different_descriptors(self, square, circle):
        r_sq = compute_shape_context(square[:10], n_bins_r=5, n_bins_theta=12)
        r_ci = compute_shape_context(circle[:10], n_bins_r=5, n_bins_theta=12)
        # Дескрипторы квадрата и круга должны отличаться
        assert not np.allclose(r_sq.descriptors, r_ci.descriptors)

    def test_same_points_identical(self, square):
        r1 = compute_shape_context(square)
        r2 = compute_shape_context(square)
        assert np.allclose(r1.descriptors, r2.descriptors)

    def test_repr(self, square):
        r = compute_shape_context(square)
        assert "ShapeContextResult" in repr(r)

    def test_n_bins_r_and_theta_stored(self, square):
        r = compute_shape_context(square, n_bins_r=3, n_bins_theta=6)
        assert r.n_bins_r == 3
        assert r.n_bins_theta == 6


# ─── normalize_shape_context ──────────────────────────────────────────────────

class TestNormalizeShapeContext:
    def test_zero_vector_unchanged(self):
        sc = np.zeros(60)
        n  = normalize_shape_context(sc)
        assert (n == 0).all()

    def test_l1_equals_one(self):
        sc = np.array([1.0, 2.0, 3.0, 4.0])
        n  = normalize_shape_context(sc)
        assert math.isclose(n.sum(), 1.0)

    def test_already_normalized_unchanged(self):
        sc = np.array([0.25, 0.25, 0.25, 0.25])
        n  = normalize_shape_context(sc)
        assert np.allclose(n, sc)

    def test_positive_scaling(self):
        sc = np.array([0.0, 0.0, 3.0, 0.0])
        n  = normalize_shape_context(sc)
        assert math.isclose(n[2], 1.0)

    def test_returns_copy(self):
        sc = np.array([1.0, 2.0, 3.0])
        n  = normalize_shape_context(sc)
        n[0] = 999.0
        assert sc[0] == 1.0  # оригинал не изменился


# ─── shape_context_distance ───────────────────────────────────────────────────

class TestShapeContextDistance:
    def test_identical_descriptors_zero(self):
        sc = np.array([0.25, 0.25, 0.25, 0.25])
        assert math.isclose(shape_context_distance(sc, sc), 0.0)

    def test_different_descriptors_positive(self):
        sc1 = np.array([1.0, 0.0, 0.0, 0.0])
        sc2 = np.array([0.0, 0.0, 0.0, 1.0])
        d   = shape_context_distance(sc1, sc2)
        assert d > 0.0

    def test_shape_mismatch_raises(self):
        sc1 = np.zeros(10)
        sc2 = np.zeros(12)
        with pytest.raises(ValueError):
            shape_context_distance(sc1, sc2)

    def test_symmetric(self):
        sc1 = np.array([0.5, 0.2, 0.2, 0.1])
        sc2 = np.array([0.1, 0.3, 0.4, 0.2])
        d12 = shape_context_distance(sc1, sc2)
        d21 = shape_context_distance(sc2, sc1)
        assert math.isclose(d12, d21, rel_tol=1e-9)

    def test_max_value_at_most_half(self):
        """χ²-расстояние ∈ [0, 0.5] для нормированных дескрипторов."""
        sc1 = np.array([1.0, 0.0, 0.0, 0.0])
        sc2 = np.array([0.0, 1.0, 0.0, 0.0])
        d   = shape_context_distance(sc1, sc2)
        assert d <= 0.5 + 1e-9


# ─── match_shape_contexts ─────────────────────────────────────────────────────

class TestMatchShapeContexts:
    def test_returns_tuple(self, square, circle):
        r_sq = compute_shape_context(square[:8])
        r_ci = compute_shape_context(circle[:8])
        result = match_shape_contexts(r_sq, r_ci)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cost_nonneg(self, square, circle):
        r_sq = compute_shape_context(square[:8])
        r_ci = compute_shape_context(circle[:8])
        cost, _ = match_shape_contexts(r_sq, r_ci)
        assert cost >= 0.0

    def test_correspondence_shape(self, square):
        r = compute_shape_context(square[:6])
        cost, corr = match_shape_contexts(r, r)
        assert corr.ndim == 2
        assert corr.shape[1] == 2
        assert len(corr) == 6  # min(6, 6)

    def test_no_duplicate_indices(self, square):
        r = compute_shape_context(square[:8])
        _, corr = match_shape_contexts(r, r)
        assert len(set(corr[:, 0])) == len(corr)
        assert len(set(corr[:, 1])) == len(corr)

    def test_self_match_low_cost(self, square):
        """Сопоставление контура с самим собой → минимальные χ²-расстояния."""
        r = compute_shape_context(square[:10], normalize=True)
        cost_self, _ = match_shape_contexts(r, r)
        assert cost_self == pytest.approx(0.0, abs=1e-9)

    def test_asymmetric_sizes(self, square, circle):
        """N_a ≠ N_b → корректное назначение min(N_a, N_b) пар."""
        r_sq = compute_shape_context(square[:6])
        r_ci = compute_shape_context(circle[:10])
        cost, corr = match_shape_contexts(r_sq, r_ci)
        assert len(corr) == 6  # min(6, 10)


# ─── contour_similarity ───────────────────────────────────────────────────────

class TestContourSimilarity:
    def test_empty_contour_returns_zero(self):
        c = np.zeros((0, 2), dtype=np.float64)
        assert contour_similarity(c, c) == 0.0

    def test_single_point_returns_zero(self):
        c = np.array([[0.0, 0.0]])
        assert contour_similarity(c, c) == 0.0

    def test_self_similarity_near_one(self, square):
        s = contour_similarity(square, square)
        assert s >= 0.9, f"Self-similarity должна быть высокой, получено {s}"

    def test_range(self, square, circle):
        s = contour_similarity(square, circle)
        assert 0.0 <= s <= 1.0

    def test_different_shapes_less_than_self(self, square, circle):
        s_self  = contour_similarity(square, square)
        s_cross = contour_similarity(square, circle)
        assert s_self >= s_cross - 0.01  # мягкое неравенство

    def test_3d_contour_input(self, square):
        """Контур формы (N, 1, 2) — формат OpenCV."""
        c3d = square[:, np.newaxis, :]
        s   = contour_similarity(c3d, square)
        assert 0.0 <= s <= 1.0

    def test_custom_n_sample(self, square, circle):
        s = contour_similarity(square, circle, n_sample=20)
        assert 0.0 <= s <= 1.0

    def test_float_output(self, square):
        s = contour_similarity(square, square)
        assert isinstance(s, float)


# ─── _preprocess_contour ──────────────────────────────────────────────────────

class TestPreprocessContour:
    def test_2d_input_unchanged_shape(self, square):
        pts = _preprocess_contour(square, n_sample=len(square))
        assert pts.shape == (len(square), 2)

    def test_3d_input_squeezed(self, square):
        c3d = square[:, np.newaxis, :]
        pts = _preprocess_contour(c3d, n_sample=len(square))
        assert pts.ndim == 2

    def test_empty_returns_empty(self):
        c = np.zeros((0, 2))
        pts = _preprocess_contour(c, n_sample=10)
        assert len(pts) == 0

    def test_downsampling(self, square):
        pts = _preprocess_contour(square, n_sample=10)
        assert len(pts) == 10

    def test_upsampling(self, square):
        short = square[:5]
        pts   = _preprocess_contour(short, n_sample=20)
        assert len(pts) == 20


# ─── ShapeContextResult ───────────────────────────────────────────────────────

class TestShapeContextResult:
    def test_descriptor_dim(self):
        desc = np.zeros((10, 60))
        r    = ShapeContextResult(
            descriptors=desc,
            points=np.zeros((10, 2)),
            mean_dist=5.0,
            n_bins_r=5,
            n_bins_theta=12,
        )
        assert r.descriptor_dim == 60

    def test_repr_contains_N(self, square):
        r = compute_shape_context(square)
        assert str(len(square)) in repr(r)

    def test_repr_contains_mean_dist(self, square):
        r = compute_shape_context(square)
        assert "mean_dist" in repr(r)
