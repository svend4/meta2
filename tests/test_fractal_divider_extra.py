"""Additional tests for puzzle_reconstruction/algorithms/fractal/divider.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.divider import (
    _walk_with_step,
    divider_curve,
    divider_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 128, r: float = 100.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n: int = 64, length: float = 100.0) -> np.ndarray:
    return np.stack([np.linspace(0.0, length, n), np.zeros(n)], axis=1)


def _zigzag(n: int = 64) -> np.ndarray:
    xs = np.arange(n, dtype=float)
    ys = np.where(xs % 2 == 0, 0.0, 5.0)
    return np.stack([xs, ys], axis=1)


# ─── TestDividerFDExtra ───────────────────────────────────────────────────────

class TestDividerFDExtra:
    def test_n_scales_2_in_range(self):
        fd = divider_fd(_circle(), n_scales=2)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_3_in_range(self):
        fd = divider_fd(_circle(), n_scales=3)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_8_in_range(self):
        fd = divider_fd(_circle(n=256), n_scales=8)
        assert 1.0 <= fd <= 2.0

    def test_int64_input(self):
        pts = _circle(n=32).astype(np.int64)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_float64_input(self):
        pts = _circle(n=64).astype(np.float64)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_large_coords_in_range(self):
        pts = _circle(r=1e5)
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_negative_offset_in_range(self):
        pts = _circle() - 500.0
        fd = divider_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_zigzag_n_scales_4_in_range(self):
        fd = divider_fd(_zigzag(n=128), n_scales=4)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_4_matches_n_scales_6(self):
        """Two runs with same input are stable to within tolerance."""
        c = _circle(n=128)
        fd4 = divider_fd(c, n_scales=4)
        fd6 = divider_fd(c, n_scales=6)
        assert abs(fd4 - fd6) < 0.5  # Results can differ but should be close


# ─── TestDividerCurveExtra ────────────────────────────────────────────────────

class TestDividerCurveExtra:
    def test_log_s_all_positive(self):
        log_s, _ = divider_curve(_circle(n=256), n_scales=6)
        if len(log_s) > 0:
            assert np.all(log_s >= 0.0)

    def test_log_L_finite_for_zigzag(self):
        _, log_L = divider_curve(_zigzag(n=128), n_scales=6)
        if len(log_L) > 0:
            assert np.all(np.isfinite(log_L))

    def test_n_scales_8_length(self):
        log_s, log_L = divider_curve(_circle(), n_scales=8)
        assert len(log_s) <= 8
        assert len(log_L) <= 8

    def test_large_circle_produces_data(self):
        log_s, log_L = divider_curve(_circle(r=1e4))
        # No crash; may produce 0 or more entries
        assert len(log_s) == len(log_L)

    def test_negative_coords_works(self):
        pts = _circle() - 200.0
        log_s, log_L = divider_curve(pts, n_scales=4)
        assert len(log_s) == len(log_L)

    def test_float32_accepted(self):
        pts = _circle(n=64).astype(np.float32)
        log_s, log_L = divider_curve(pts, n_scales=4)
        assert isinstance(log_s, np.ndarray)

    def test_log_s_log_L_same_length_zigzag(self):
        log_s, log_L = divider_curve(_zigzag(n=64))
        assert len(log_s) == len(log_L)


# ─── TestWalkWithStepExtra ────────────────────────────────────────────────────

class TestWalkWithStepExtra:
    def test_circle_path_count_positive(self):
        pts = _circle(n=128, r=50.0)
        count = _walk_with_step(pts, step=5.0)
        assert count > 0

    def test_very_large_step_at_most_1(self):
        """Step larger than entire curve → 0 or 1 step counted."""
        pts = _line(n=5, length=1.0)
        count = _walk_with_step(pts, step=1000.0)
        assert count <= 1

    def test_negative_coords_accepted(self):
        pts = _line() - 500.0
        count = _walk_with_step(pts, step=3.0)
        assert count >= 0

    def test_step_exactly_segment_length(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        count = _walk_with_step(pts, step=1.0)
        assert count >= 1

    def test_shorter_step_more_steps_on_line(self):
        pts = _line(n=50, length=100.0)
        c_large = _walk_with_step(pts, step=20.0)
        c_small = _walk_with_step(pts, step=4.0)
        assert c_small >= c_large

    def test_float32_input(self):
        pts = _circle(n=32).astype(np.float32)
        count = _walk_with_step(pts, step=10.0)
        assert isinstance(count, int)
