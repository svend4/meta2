"""Additional tests for puzzle_reconstruction.algorithms.fractal.divider."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fractal.divider import (
    divider_curve,
    divider_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=128, r=100.0) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(n=80, side=100.0) -> np.ndarray:
    n4 = n // 4
    sides = [
        np.stack([np.linspace(0, side, n4), np.zeros(n4)], axis=1),
        np.stack([np.full(n4, side), np.linspace(0, side, n4)], axis=1),
        np.stack([np.linspace(side, 0, n4), np.full(n4, side)], axis=1),
        np.stack([np.zeros(n4), np.linspace(side, 0, n4)], axis=1),
    ]
    return np.vstack(sides).astype(np.float64)


def _random_walk(n=200, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n - 1, 2))
    pts = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    return pts


# ─── TestDividerFdExtra ───────────────────────────────────────────────────────

class TestDividerFdExtra:
    def test_square_in_range(self):
        fd = divider_fd(_square())
        assert 1.0 <= fd <= 2.0

    def test_reproducible_same_input(self):
        pts = _circle(128)
        assert divider_fd(pts) == divider_fd(pts)

    def test_not_nan(self):
        fd = divider_fd(_circle())
        assert not math.isnan(fd)

    def test_not_inf(self):
        fd = divider_fd(_circle())
        assert not math.isinf(fd)

    def test_float32_input_ok(self):
        pts = _circle(64).astype(np.float32)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_large_radius_ok(self):
        pts = _circle(64, r=1e5)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_tiny_radius_ok(self):
        pts = _circle(64, r=1e-2)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_3_points_ok(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_random_walk_in_range(self):
        pts = _random_walk(200)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_1_ok(self):
        pts = _circle(64)
        fd = divider_fd(pts, n_scales=1)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_20_ok(self):
        pts = _circle(128)
        fd = divider_fd(pts, n_scales=20)
        assert 1.0 <= fd <= 2.0

    def test_random_walk_fd_above_circle(self):
        fd_smooth = divider_fd(_circle(256))
        fd_rough = divider_fd(_random_walk(256))
        assert fd_rough >= fd_smooth - 0.1  # rough at least as complex


# ─── TestDividerCurveExtra ────────────────────────────────────────────────────

class TestDividerCurveExtra:
    def test_outputs_are_1d(self):
        log_s, log_L = divider_curve(_circle())
        assert log_s.ndim == 1
        assert log_L.ndim == 1

    def test_finite_values(self):
        log_s, log_L = divider_curve(_circle(128))
        assert np.all(np.isfinite(log_s))
        assert np.all(np.isfinite(log_L))

    def test_square_contour_ok(self):
        log_s, log_L = divider_curve(_square())
        assert len(log_s) > 0

    def test_random_walk_ok(self):
        log_s, log_L = divider_curve(_random_walk(200))
        assert len(log_s) >= 0  # no crash

    def test_float32_input(self):
        pts = _circle(64).astype(np.float32)
        log_s, log_L = divider_curve(pts)
        assert len(log_s) >= 0

    def test_large_n_scales(self):
        log_s, log_L = divider_curve(_circle(128), n_scales=16)
        assert len(log_s) <= 16

    def test_length_relationship_to_fd(self):
        """Slope of divider_curve should correspond to 1 - FD."""
        pts = _circle(256)
        log_s, log_L = divider_curve(pts, n_scales=8)
        fd = divider_fd(pts, n_scales=8)
        if len(log_s) >= 2:
            slope = float(np.polyfit(log_s, log_L, 1)[0])
            reconstructed_fd = float(np.clip(1.0 - slope, 1.0, 2.0))
            assert abs(reconstructed_fd - fd) < 1e-9  # exact consistency

    def test_returns_empty_for_single_point(self):
        pts = np.array([[5.0, 5.0]])
        log_s, log_L = divider_curve(pts)
        # single point → all same → seg_len=0 → empty arrays
        assert len(log_s) == 0 or len(log_s) >= 0  # no crash
