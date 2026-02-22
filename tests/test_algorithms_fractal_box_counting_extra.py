"""Additional tests for puzzle_reconstruction.algorithms.fractal.box_counting."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_curve,
    box_counting_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(n=64, side=100.0) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    n4 = n // 4
    sides = [
        np.stack([t[:n4] * side, np.zeros(n4)], axis=1),
        np.stack([np.full(n4, side), t[:n4] * side], axis=1),
        np.stack([side - t[:n4] * side, np.full(n4, side)], axis=1),
        np.stack([np.zeros(n4), side - t[:n4] * side], axis=1),
    ]
    return np.vstack(sides).astype(np.float64)


def _line(n=64, length=100.0) -> np.ndarray:
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n)], axis=1).astype(np.float64)


def _noisy_circle(n=256, r=50.0, noise=5.0, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = r * np.cos(t) + rng.normal(0, noise, n)
    y = r * np.sin(t) + rng.normal(0, noise, n)
    return np.stack([x, y], axis=1).astype(np.float64)


# ─── TestBoxCountingFdExtra ───────────────────────────────────────────────────

class TestBoxCountingFdExtra:
    def test_exactly_4_points_not_trivial(self):
        """4 points is the minimum for non-trivial FD."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_square_contour_in_range(self):
        fd = box_counting_fd(_square())
        assert 1.0 <= fd <= 2.0

    def test_line_fd_near_1(self):
        fd = box_counting_fd(_line(n=128))
        assert fd < 1.5  # collinear points → FD close to 1

    def test_noisy_circle_higher_than_smooth(self):
        fd_smooth = box_counting_fd(_square(n=128))
        fd_noisy = box_counting_fd(_noisy_circle(n=256, noise=10.0))
        assert fd_noisy >= fd_smooth - 0.1  # noisy at least as complex

    def test_reproducible(self):
        pts = _square(n=128)
        assert box_counting_fd(pts) == box_counting_fd(pts)

    def test_scales_1_returns_valid(self):
        pts = _square(n=64)
        fd = box_counting_fd(pts, n_scales=1)
        assert 1.0 <= fd <= 2.0

    def test_scales_16_returns_valid(self):
        pts = _square(n=128)
        fd = box_counting_fd(pts, n_scales=16)
        assert 1.0 <= fd <= 2.0

    def test_float32_input_ok(self):
        pts = _square(n=64).astype(np.float32)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_large_coordinates_ok(self):
        pts = _square(n=64, side=1e6)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_tiny_coordinates_ok(self):
        pts = _square(n=64, side=1e-3)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_two_points_returns_1(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        fd = box_counting_fd(pts)
        assert fd == pytest.approx(1.0)

    def test_one_point_returns_1(self):
        pts = np.array([[5.0, 5.0]])
        fd = box_counting_fd(pts)
        assert fd == pytest.approx(1.0)

    def test_value_stable_across_n_scales(self):
        """FD estimates for neighboring n_scales should be similar."""
        pts = _square(n=128)
        fd_a = box_counting_fd(pts, n_scales=6)
        fd_b = box_counting_fd(pts, n_scales=8)
        assert abs(fd_a - fd_b) < 0.5  # rough stability

    def test_does_not_return_nan(self):
        fd = box_counting_fd(_square())
        assert not math.isnan(fd)

    def test_does_not_return_inf(self):
        fd = box_counting_fd(_square())
        assert not math.isinf(fd)


# ─── TestBoxCountingCurveExtra ────────────────────────────────────────────────

class TestBoxCountingCurveExtra:
    def test_log_n_increases_with_larger_r_inv(self):
        """More boxes at finer scales (higher log_r_inv → higher log_N)."""
        log_r, log_n = box_counting_curve(_square(n=128))
        # Each finer scale should have at least as many boxes
        for i in range(len(log_n) - 1):
            assert log_n[i] <= log_n[i + 1] + 1.0  # allow small slack

    def test_nonneg_values(self):
        log_r, log_n = box_counting_curve(_square())
        assert np.all(log_r >= 0.0)
        assert np.all(log_n >= 0.0)

    def test_float32_input(self):
        pts = _square(n=64).astype(np.float32)
        log_r, log_n = box_counting_curve(pts)
        assert len(log_r) > 0

    def test_large_n_scales(self):
        pts = _square(n=128)
        log_r, log_n = box_counting_curve(pts, n_scales=20)
        assert len(log_r) == 20

    def test_outputs_are_1d(self):
        log_r, log_n = box_counting_curve(_square())
        assert log_r.ndim == 1
        assert log_n.ndim == 1

    def test_slope_reflects_fd(self):
        """Slope of log_N vs log_r_inv should approximate FD."""
        pts = _square(n=256)
        log_r, log_n = box_counting_curve(pts, n_scales=8)
        fd_direct = box_counting_fd(pts, n_scales=8)
        # Use numpy polyfit to check slope consistency
        if np.any(log_r > 0):
            slope = np.polyfit(log_r, log_n, 1)[0]
            assert abs(slope - fd_direct) < 0.5
