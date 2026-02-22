"""Tests for puzzle_reconstruction.algorithms.fractal.box_counting."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_curve,
    box_counting_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=100.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n=64):
    xs = np.linspace(0.0, 100.0, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _fractal_like(n=256):
    """Rough random-walk contour — higher FD than smooth line."""
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return np.stack([x, y], axis=1)


# ─── TestBoxCountingFd ────────────────────────────────────────────────────────

class TestBoxCountingFd:
    def test_returns_float(self):
        fd = box_counting_fd(_circle())
        assert isinstance(fd, float)

    def test_range_1_to_2(self):
        fd = box_counting_fd(_circle())
        assert 1.0 <= fd <= 2.0

    def test_too_few_points_returns_1(self):
        contour = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        fd = box_counting_fd(contour)
        assert fd == pytest.approx(1.0)

    def test_span_zero_returns_1(self):
        contour = np.ones((10, 2)) * 5.0
        fd = box_counting_fd(contour)
        assert fd == pytest.approx(1.0)

    def test_smooth_circle_near_1(self):
        fd = box_counting_fd(_circle(n=128))
        assert fd < 1.3  # smooth circle should be low-FD

    def test_rougher_contour_higher_fd(self):
        fd_smooth = box_counting_fd(_circle(n=256))
        fd_rough = box_counting_fd(_fractal_like(n=256))
        assert fd_rough > fd_smooth

    def test_n_scales_parameter(self):
        c = _circle()
        fd4 = box_counting_fd(c, n_scales=4)
        fd12 = box_counting_fd(c, n_scales=12)
        # Both should be in valid range regardless of n_scales
        assert 1.0 <= fd4 <= 2.0
        assert 1.0 <= fd12 <= 2.0

    def test_accepts_integer_coordinates(self):
        pts = np.array([[0, 0], [1, 0], [2, 1], [3, 2]], dtype=np.int32)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0


# ─── TestBoxCountingCurve ─────────────────────────────────────────────────────

class TestBoxCountingCurve:
    def test_returns_two_arrays(self):
        log_r, log_n = box_counting_curve(_circle())
        assert isinstance(log_r, np.ndarray)
        assert isinstance(log_n, np.ndarray)

    def test_same_length(self):
        log_r, log_n = box_counting_curve(_circle())
        assert len(log_r) == len(log_n)

    def test_length_equals_n_scales(self):
        n = 6
        log_r, log_n = box_counting_curve(_circle(), n_scales=n)
        assert len(log_r) == n

    def test_span_zero_returns_zeros(self):
        contour = np.ones((10, 2))
        log_r, log_n = box_counting_curve(contour)
        assert np.all(log_r == 0.0)
        assert np.all(log_n == 0.0)

    def test_log_r_increasing(self):
        log_r, _ = box_counting_curve(_circle())
        # log(2^k) = k*log(2) — strictly increasing
        assert all(log_r[i] < log_r[i + 1] for i in range(len(log_r) - 1))

    def test_log_n_nonneg(self):
        _, log_n = box_counting_curve(_circle())
        assert np.all(log_n >= 0.0)

    def test_custom_n_scales(self):
        log_r, log_n = box_counting_curve(_circle(n=64), n_scales=4)
        assert len(log_r) == 4
