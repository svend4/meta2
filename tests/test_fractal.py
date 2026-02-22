"""
Тесты для фрактальных алгоритмов.
"""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_fd, box_counting_curve
)
from puzzle_reconstruction.algorithms.fractal.divider import (
    divider_fd, divider_curve
)
from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients, reconstruct_from_ifs, ifs_distance
)
from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space, css_to_feature_vector,
    css_similarity, css_similarity_mirror, freeman_chain_code
)


# ─── Вспомогательные фигуры ───────────────────────────────────────────────

def make_circle(n=256, r=100):
    """Идеальная окружность — FD ≈ 1.0."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def make_square(n=256):
    """Квадрат — FD ≈ 1.0."""
    side = np.linspace(0, 1, n // 4)
    top    = np.column_stack([side,       np.zeros_like(side)])
    right  = np.column_stack([np.ones_like(side), side])
    bottom = np.column_stack([1 - side,   np.ones_like(side)])
    left   = np.column_stack([np.zeros_like(side), 1 - side])
    return np.vstack([top, right, bottom, left]) * 200


def make_noisy_line(n=256, noise=5.0, seed=0):
    """Прямая с добавленным шумом — FD между 1.0 и 1.5."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 200, n)
    y = rng.randn(n) * noise
    return np.column_stack([x, y])


def make_fractal_coastline(n=512, seed=1):
    """Псевдо-фрактальная береговая линия через fBm."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 200, n)
    y = np.zeros(n)
    amp, freq = 20.0, 1.0
    for _ in range(8):
        knots = rng.randn(max(2, int(freq * 4)))
        xs = np.linspace(0, n - 1, len(knots))
        y += np.interp(np.arange(n), xs, knots) * amp
        amp *= 0.55; freq *= 2.0
    return np.column_stack([x, y])


# ─── Box-counting ─────────────────────────────────────────────────────────

class TestBoxCounting:

    def test_circle_fd_near_one(self):
        """Окружность должна иметь FD близкую к 1.0."""
        circle = make_circle()
        fd = box_counting_fd(circle)
        assert 0.9 <= fd <= 1.3, f"FD окружности = {fd}"

    def test_fractal_fd_greater_than_circle(self):
        """Фрактальная кривая должна иметь FD > FD окружности."""
        circle   = make_circle()
        fractal  = make_fractal_coastline()
        fd_circ  = box_counting_fd(circle)
        fd_frac  = box_counting_fd(fractal)
        assert fd_frac > fd_circ, (
            f"Ожидалось FD_fractal({fd_frac:.3f}) > FD_circle({fd_circ:.3f})"
        )

    def test_output_range(self):
        """FD всегда в диапазоне [1.0, 2.0]."""
        for contour in [make_circle(), make_square(), make_noisy_line()]:
            fd = box_counting_fd(contour)
            assert 1.0 <= fd <= 2.0, f"FD вне диапазона: {fd}"

    def test_curve_returns_arrays(self):
        log_r, log_N = box_counting_curve(make_circle())
        assert len(log_r) == len(log_N)
        assert len(log_r) > 0

    def test_empty_contour(self):
        """Пустой контур не вызывает исключение."""
        fd = box_counting_fd(np.zeros((2, 2)))
        assert fd == 1.0

    def test_reproducibility(self):
        """Одинаковые входные данные → одинаковый результат."""
        c = make_fractal_coastline()
        assert box_counting_fd(c) == box_counting_fd(c)


# ─── Divider method ───────────────────────────────────────────────────────

class TestDivider:

    def test_circle_fd_near_one(self):
        fd = divider_fd(make_circle())
        assert 0.9 <= fd <= 1.3, f"FD окружности (divider) = {fd}"

    def test_fractal_greater_than_smooth(self):
        fd_smooth  = divider_fd(make_circle())
        fd_fractal = divider_fd(make_fractal_coastline())
        assert fd_fractal >= fd_smooth - 0.05  # Небольшой допуск

    def test_output_range(self):
        for contour in [make_circle(), make_noisy_line()]:
            fd = divider_fd(contour)
            assert 1.0 <= fd <= 2.0

    def test_curve_returns_matching_lengths(self):
        log_s, log_L = divider_curve(make_circle())
        assert len(log_s) == len(log_L)


# ─── IFS ──────────────────────────────────────────────────────────────────

class TestIFS:

    def test_coefficients_bounded(self):
        """Коэффициенты IFS должны быть < 1 для сходимости."""
        curve = make_fractal_coastline()
        coeffs = fit_ifs_coefficients(curve, n_transforms=6)
        assert np.all(np.abs(coeffs) < 1.0), "Некоторые |d_k| >= 1"

    def test_coefficients_shape(self):
        curve = make_noisy_line()
        n = 8
        coeffs = fit_ifs_coefficients(curve, n_transforms=n)
        assert coeffs.shape == (n,)

    def test_reconstruct_returns_profile(self):
        curve  = make_fractal_coastline()
        coeffs = fit_ifs_coefficients(curve, n_transforms=4)
        profile = reconstruct_from_ifs(coeffs, n_points=128)
        assert profile.shape == (128,)

    def test_ifs_distance_same(self):
        """Расстояние между одинаковыми коэффициентами = 0."""
        coeffs = np.array([0.3, -0.5, 0.2, 0.1])
        assert ifs_distance(coeffs, coeffs) < 1e-10

    def test_ifs_distance_different(self):
        a = np.array([0.5,  0.3,  0.1])
        b = np.array([-0.5, -0.3, -0.1])
        assert ifs_distance(a, b) > 0.5

    def test_ifs_distance_different_lengths(self):
        """Разные длины массивов не вызывают исключение."""
        a = np.array([0.1, 0.2, 0.3, 0.4])
        b = np.array([0.1, 0.2])
        dist = ifs_distance(a, b)
        assert dist >= 0.0


# ─── CSS ──────────────────────────────────────────────────────────────────

class TestCSS:

    def test_css_returns_list(self):
        circle = make_circle(n=128)
        css = curvature_scale_space(circle, n_sigmas=4)
        assert isinstance(css, list)
        assert len(css) == 4

    def test_feature_vector_normalized(self):
        """CSS-вектор должен быть нормализован (норма ≤ 1 + ε)."""
        circle = make_circle(n=128)
        css = curvature_scale_space(circle, n_sigmas=4)
        vec = css_to_feature_vector(css, n_bins=16)
        norm = np.linalg.norm(vec)
        assert norm <= 1.0 + 1e-6, f"Норма CSS-вектора = {norm}"

    def test_similarity_identical(self):
        """Одинаковые контуры: сходство = 1.0."""
        circle = make_circle(n=128)
        css = curvature_scale_space(circle, n_sigmas=4)
        vec = css_to_feature_vector(css)
        sim = css_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_similarity_different(self):
        """Разные формы: сходство < 1.0."""
        circle = make_circle(n=128)
        noisy  = make_fractal_coastline(n=128)
        css_c = css_to_feature_vector(curvature_scale_space(circle, n_sigmas=4))
        css_n = css_to_feature_vector(curvature_scale_space(noisy,  n_sigmas=4))
        sim = css_similarity(css_c, css_n)
        assert sim < 0.999

    def test_similarity_mirror_geq_direct(self):
        """Зеркальное сходство ≥ прямого."""
        vec = css_to_feature_vector(curvature_scale_space(make_noisy_line()))
        vec_rev = vec[::-1]
        sim_direct   = css_similarity(vec, vec_rev)
        sim_mirror   = css_similarity_mirror(vec, vec_rev)
        assert sim_mirror >= sim_direct - 1e-9

    def test_chain_code_nonempty(self):
        square = make_square(n=64)
        code = freeman_chain_code(square)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_chain_code_valid_chars(self):
        """Цепной код содержит только цифры 0–7."""
        code = freeman_chain_code(make_circle(n=64))
        assert all(c in "01234567" for c in code), f"Неверные символы: {code[:20]}"
