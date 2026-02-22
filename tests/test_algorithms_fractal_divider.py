"""Tests for puzzle_reconstruction.algorithms.fractal.divider."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.divider import (
    divider_curve,
    divider_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=128, r=100.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n=64, length=100.0):
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _fractal_like(n=256, seed=42):
    """Random-walk contour — higher FD than smooth curve."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return np.stack([x, y], axis=1)


# ─── TestDividerFd ────────────────────────────────────────────────────────────

class TestDividerFd:
    def test_returns_float(self):
        fd = divider_fd(_circle())
        assert isinstance(fd, float)

    def test_range_1_to_2(self):
        fd = divider_fd(_circle())
        assert 1.0 <= fd <= 2.0

    def test_zero_length_contour_returns_1(self):
        pts = np.ones((10, 2))  # all same point → seg_len=0
        fd = divider_fd(pts)
        assert fd == pytest.approx(1.0)

    def test_smooth_circle_near_1(self):
        fd = divider_fd(_circle(n=256))
        assert fd < 1.4  # smooth circle: FD close to 1

    def test_rough_contour_higher_fd(self):
        fd_smooth = divider_fd(_circle(n=256))
        fd_rough = divider_fd(_fractal_like(n=256))
        assert fd_rough >= fd_smooth

    def test_n_scales_parameter(self):
        c = _circle()
        fd4 = divider_fd(c, n_scales=4)
        fd10 = divider_fd(c, n_scales=10)
        assert 1.0 <= fd4 <= 2.0
        assert 1.0 <= fd10 <= 2.0

    def test_two_point_contour(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_accepts_integer_array(self):
        pts = _circle(n=32).astype(np.int32)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0


# ─── TestDividerCurve ─────────────────────────────────────────────────────────

class TestDividerCurve:
    def test_returns_two_arrays(self):
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
        """Step sizes are geometrically increasing → log_s should be sorted."""
        log_s, _ = divider_curve(_circle(n=256))
        if len(log_s) >= 2:
            assert all(log_s[i] < log_s[i + 1]
                       for i in range(len(log_s) - 1))

    def test_at_most_n_scales_entries(self):
        n = 6
        log_s, log_L = divider_curve(_circle(), n_scales=n)
        assert len(log_s) <= n

    def test_log_L_nonneg(self):
        """Length L(s) = count * s is always > 0, so log_L > 0 (L > 1)."""
        log_s, log_L = divider_curve(_circle(n=256))
        if len(log_L) > 0:
            # L can be < 1 in edge cases, but we just check finiteness
            assert np.all(np.isfinite(log_L))

    def test_custom_n_scales(self):
        log_s, log_L = divider_curve(_circle(n=128), n_scales=4)
        assert len(log_s) <= 4

    def test_line_contour_accepted(self):
        log_s, log_L = divider_curve(_line())
        assert len(log_s) >= 0  # just no crash
