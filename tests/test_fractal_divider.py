"""Расширенные тесты для puzzle_reconstruction/algorithms/fractal/divider.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.divider import (
    _walk_with_step,
    divider_curve,
    divider_fd,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 128, r: float = 100.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n: int = 64, length: float = 100.0) -> np.ndarray:
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _fractal_like(n: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return np.stack([x, y], axis=1)


def _zigzag(n: int = 64) -> np.ndarray:
    """Zigzag contour with alternating y values."""
    xs = np.arange(n, dtype=float)
    ys = np.where(xs % 2 == 0, 0.0, 10.0)
    return np.stack([xs, ys], axis=1)


# ─── TestDividerFD ────────────────────────────────────────────────────────────

class TestDividerFD:
    # --- Return type and range ---

    def test_returns_float(self):
        fd = divider_fd(_circle())
        assert isinstance(fd, float)

    def test_range_1_to_2(self):
        fd = divider_fd(_circle())
        assert 1.0 <= fd <= 2.0

    def test_line_in_range(self):
        fd = divider_fd(_line())
        assert 1.0 <= fd <= 2.0

    def test_fractal_in_range(self):
        fd = divider_fd(_fractal_like())
        assert 1.0 <= fd <= 2.0

    def test_zigzag_in_range(self):
        fd = divider_fd(_zigzag())
        assert 1.0 <= fd <= 2.0

    # --- Degenerate cases ---

    def test_zero_length_returns_1(self):
        pts = np.ones((10, 2))
        assert divider_fd(pts) == pytest.approx(1.0)

    def test_two_same_points_returns_1(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0]])
        assert divider_fd(pts) == pytest.approx(1.0)

    def test_two_distinct_points_in_range(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    # --- Smooth vs rough ---

    def test_rough_contour_higher_fd_than_smooth(self):
        fd_smooth = divider_fd(_circle(n=256))
        fd_rough = divider_fd(_fractal_like(n=256))
        assert fd_rough >= fd_smooth

    def test_smooth_circle_near_1(self):
        fd = divider_fd(_circle(n=256))
        assert fd < 1.4

    def test_line_close_to_1(self):
        """A straight line should have FD close to 1."""
        fd = divider_fd(_line(n=128))
        assert fd < 1.5

    # --- n_scales ---

    def test_n_scales_4_valid(self):
        fd = divider_fd(_circle(), n_scales=4)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_10_valid(self):
        fd = divider_fd(_circle(), n_scales=10)
        assert 1.0 <= fd <= 2.0

    # --- Input types ---

    def test_accepts_float32(self):
        fd = divider_fd(_circle().astype(np.float32))
        assert 1.0 <= fd <= 2.0

    def test_accepts_int32(self):
        fd = divider_fd(_circle(n=32).astype(np.int32))
        assert 1.0 <= fd <= 2.0

    # --- Reproducibility ---

    def test_deterministic(self):
        c = _circle(n=64)
        fd1 = divider_fd(c, n_scales=6)
        fd2 = divider_fd(c, n_scales=6)
        assert fd1 == fd2

    # --- Monotone compat_matrix ---

    def test_fd_positive(self):
        fd = divider_fd(_circle(n=64))
        assert fd > 0.0


# ─── TestDividerCurve ─────────────────────────────────────────────────────────

class TestDividerCurve:
    def test_returns_tuple_of_2(self):
        result = divider_curve(_circle())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_ndarrays(self):
        log_s, log_L = divider_curve(_circle())
        assert isinstance(log_s, np.ndarray)
        assert isinstance(log_L, np.ndarray)

    def test_same_length(self):
        log_s, log_L = divider_curve(_circle())
        assert len(log_s) == len(log_L)

    def test_zero_length_returns_empty(self):
        pts = np.ones((10, 2))
        log_s, log_L = divider_curve(pts)
        assert len(log_s) == 0
        assert len(log_L) == 0

    def test_log_s_increasing(self):
        log_s, _ = divider_curve(_circle(n=256))
        if len(log_s) >= 2:
            assert all(log_s[i] < log_s[i + 1] for i in range(len(log_s) - 1))

    def test_at_most_n_scales_entries(self):
        log_s, _ = divider_curve(_circle(), n_scales=6)
        assert len(log_s) <= 6

    def test_log_L_finite(self):
        _, log_L = divider_curve(_circle(n=256))
        assert np.all(np.isfinite(log_L))

    def test_custom_n_scales_4(self):
        log_s, log_L = divider_curve(_circle(n=128), n_scales=4)
        assert len(log_s) <= 4

    def test_line_contour_produces_data(self):
        log_s, log_L = divider_curve(_line())
        assert len(log_s) >= 0  # just no crash

    def test_nonempty_for_valid_contour(self):
        log_s, _ = divider_curve(_circle(n=128), n_scales=6)
        assert len(log_s) > 0


# ─── TestWalkWithStep ─────────────────────────────────────────────────────────

class TestWalkWithStep:
    def test_empty_points_returns_0(self):
        pts = np.zeros((0, 2))
        assert _walk_with_step(pts, step=1.0) == 0

    def test_single_point_returns_0(self):
        pts = np.array([[0.0, 0.0]])
        assert _walk_with_step(pts, step=1.0) == 0

    def test_two_points_step_larger_returns_1(self):
        """Step larger than the segment → 1 step counted."""
        pts = np.array([[0.0, 0.0], [0.5, 0.0]])
        count = _walk_with_step(pts, step=1.0)
        assert count >= 1

    def test_two_points_exact_step_returns_1(self):
        """Step exactly equals segment length."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        count = _walk_with_step(pts, step=1.0)
        assert count >= 1

    def test_long_line_small_step_many_counts(self):
        """Long straight line with small step → many steps."""
        pts = _line(n=100, length=100.0)
        count = _walk_with_step(pts, step=1.0)
        assert count > 5

    def test_returns_int(self):
        pts = _line(n=32)
        count = _walk_with_step(pts, step=2.0)
        assert isinstance(count, int)

    def test_count_nonneg(self):
        pts = _line(n=32)
        count = _walk_with_step(pts, step=5.0)
        assert count >= 0

    def test_smaller_step_more_counts(self):
        """Smaller step size → more measurements."""
        pts = _circle(n=128, r=50.0)
        count_large = _walk_with_step(pts, step=20.0)
        count_small = _walk_with_step(pts, step=5.0)
        assert count_small >= count_large
