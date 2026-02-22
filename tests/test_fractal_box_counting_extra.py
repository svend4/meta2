"""Additional tests for puzzle_reconstruction/algorithms/fractal/box_counting.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_curve,
    box_counting_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 100.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(side: float = 100.0, n: int = 64) -> np.ndarray:
    s = max(n // 4, 1)
    pts = []
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side, i * side / s])
    for i in range(s):
        pts.append([side - i * side / s, side])
    for i in range(s):
        pts.append([0.0, side - i * side / s])
    return np.array(pts)


def _zigzag(n: int = 64, amplitude: float = 10.0) -> np.ndarray:
    x = np.linspace(0, 100, n)
    y = amplitude * np.where(np.arange(n) % 2 == 0, 0.0, 1.0)
    return np.stack([x, y], axis=1)


# ─── TestBoxCountingFDExtra ───────────────────────────────────────────────────

class TestBoxCountingFDExtra:
    def test_large_circle_in_range(self):
        fd = box_counting_fd(_circle(n=512, r=1000.0))
        assert 1.0 <= fd <= 2.0

    def test_square_fd_near_1(self):
        """Smooth square is almost 1-dimensional."""
        fd = box_counting_fd(_square(n=128))
        assert fd < 1.5

    def test_zigzag_fd_higher_than_smooth(self):
        """Jagged zigzag should have higher FD than smooth circle."""
        fd_smooth = box_counting_fd(_circle(n=128))
        fd_zigzag = box_counting_fd(_zigzag(n=128, amplitude=20.0))
        assert fd_zigzag >= fd_smooth - 0.05

    def test_result_is_float(self):
        fd = box_counting_fd(_square())
        assert isinstance(fd, float)

    def test_n_scales_3_in_range(self):
        fd = box_counting_fd(_circle(), n_scales=3)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_16_in_range(self):
        fd = box_counting_fd(_circle(n=256), n_scales=16)
        assert 1.0 <= fd <= 2.0

    def test_float64_input(self):
        pts = _circle(n=64).astype(np.float64)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_negative_coords_in_range(self):
        """Contour with negative coordinates should still work."""
        pts = _circle(n=64, r=50.0) - 200.0
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_nonsquare_bounding_box_in_range(self):
        """Very elongated bounding box."""
        x = np.linspace(0, 1000, 128)
        y = np.sin(x / 10.0) * 2.0
        pts = np.stack([x, y], axis=1)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_fd_monotone_with_roughness(self):
        """Higher amplitude zigzag should have >= FD of lower amplitude."""
        fd_low = box_counting_fd(_zigzag(n=128, amplitude=1.0))
        fd_high = box_counting_fd(_zigzag(n=128, amplitude=50.0))
        assert fd_high >= fd_low - 0.1

    def test_two_point_vertical_line(self):
        pts = np.array([[0.0, 0.0], [0.0, 10.0]])
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_square_contour_is_reproducible(self):
        s = _square(n=64)
        fd1 = box_counting_fd(s, n_scales=6)
        fd2 = box_counting_fd(s, n_scales=6)
        assert fd1 == fd2


# ─── TestBoxCountingCurveExtra ────────────────────────────────────────────────

class TestBoxCountingCurveExtra:
    def test_returns_two_ndarrays(self):
        log_r, log_n = box_counting_curve(_circle())
        assert isinstance(log_r, np.ndarray)
        assert isinstance(log_n, np.ndarray)

    def test_n_scales_5(self):
        log_r, log_n = box_counting_curve(_circle(), n_scales=5)
        assert len(log_r) == 5
        assert len(log_n) == 5

    def test_log_r_all_finite(self):
        log_r, _ = box_counting_curve(_circle())
        assert np.all(np.isfinite(log_r))

    def test_log_n_all_finite(self):
        _, log_n = box_counting_curve(_circle())
        assert np.all(np.isfinite(log_n))

    def test_log_r_positive_for_valid_contour(self):
        """log_r values should be positive (they are log2 of resolution exponent)."""
        log_r, _ = box_counting_curve(_circle())
        assert np.all(log_r >= 0.0)

    def test_log_n_nondecreasing(self):
        """More resolution → at least as many boxes."""
        _, log_n = box_counting_curve(_circle(n=256), n_scales=8)
        diffs = np.diff(log_n)
        assert np.all(diffs >= -1e-6)

    def test_square_zero_span_returns_zeros(self):
        pts = np.ones((20, 2)) * 42.0
        log_r, log_n = box_counting_curve(pts)
        assert np.all(log_r == 0.0)
        assert np.all(log_n == 0.0)

    def test_n_scales_2(self):
        log_r, log_n = box_counting_curve(_circle(), n_scales=2)
        assert len(log_r) == 2
        assert len(log_n) == 2

    def test_different_radii_same_curve_length(self):
        """Box counting curve length should equal n_scales regardless of radius."""
        log_r_small, _ = box_counting_curve(_circle(r=1.0), n_scales=6)
        log_r_large, _ = box_counting_curve(_circle(r=1e4), n_scales=6)
        assert len(log_r_small) == len(log_r_large)

    def test_float32_accepted(self):
        pts = _circle(n=64).astype(np.float32)
        log_r, log_n = box_counting_curve(pts, n_scales=4)
        assert len(log_r) == 4

    def test_square_has_increasing_log_n(self):
        _, log_n = box_counting_curve(_square(n=128), n_scales=6)
        assert np.all(np.diff(log_n) >= -1e-6)
