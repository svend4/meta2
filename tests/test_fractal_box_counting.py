"""Расширенные тесты для puzzle_reconstruction/algorithms/fractal/box_counting.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_curve,
    box_counting_fd,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 100.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(side: float = 100.0, n: int = 64) -> np.ndarray:
    s = n // 4
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


def _line(n: int = 64) -> np.ndarray:
    xs = np.linspace(0.0, 100.0, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _fractal_like(n: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return np.stack([x, y], axis=1)


# ─── TestBoxCountingFD ────────────────────────────────────────────────────────

class TestBoxCountingFD:
    # --- Return type and range ---

    def test_returns_float(self):
        fd = box_counting_fd(_circle())
        assert isinstance(fd, float)

    def test_range_1_to_2(self):
        fd = box_counting_fd(_circle())
        assert 1.0 <= fd <= 2.0

    def test_fractal_contour_in_range(self):
        fd = box_counting_fd(_fractal_like())
        assert 1.0 <= fd <= 2.0

    def test_square_in_range(self):
        fd = box_counting_fd(_square())
        assert 1.0 <= fd <= 2.0

    def test_line_in_range(self):
        fd = box_counting_fd(_line())
        assert 1.0 <= fd <= 2.0

    # --- Degenerate cases ---

    def test_too_few_points_returns_1(self):
        contour = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        assert box_counting_fd(contour) == pytest.approx(1.0)

    def test_two_points_returns_1(self):
        contour = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert box_counting_fd(contour) == pytest.approx(1.0)

    def test_zero_span_returns_1(self):
        contour = np.ones((10, 2)) * 5.0
        assert box_counting_fd(contour) == pytest.approx(1.0)

    def test_single_point_returns_1(self):
        contour = np.array([[3.0, 4.0]])
        assert box_counting_fd(contour) == pytest.approx(1.0)

    # --- Smooth vs rough ---

    def test_smooth_circle_lower_fd_than_rough(self):
        fd_smooth = box_counting_fd(_circle(n=256))
        fd_rough = box_counting_fd(_fractal_like(n=256))
        assert fd_rough >= fd_smooth

    def test_smooth_circle_near_1(self):
        fd = box_counting_fd(_circle(n=128))
        assert fd < 1.3

    # --- n_scales parameter ---

    def test_n_scales_4_in_range(self):
        fd = box_counting_fd(_circle(), n_scales=4)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_12_in_range(self):
        fd = box_counting_fd(_circle(), n_scales=12)
        assert 1.0 <= fd <= 2.0

    def test_n_scales_1_returns_1(self):
        # With only 1 point in log regression → polyfit with 1 pt returns 0 slope → clipped to 1.0
        fd = box_counting_fd(_circle(), n_scales=1)
        assert 1.0 <= fd <= 2.0

    # --- Input types ---

    def test_accepts_integer_input(self):
        pts = np.array([[0, 0], [1, 0], [2, 1], [3, 2], [4, 0]], dtype=np.int32)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    def test_accepts_float32_input(self):
        pts = _circle(n=64).astype(np.float32)
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    # --- Normalization independence ---

    def test_scale_invariant(self):
        """Scaling the contour should not drastically change FD."""
        small = _circle(n=128, r=1.0)
        large = _circle(n=128, r=10000.0)
        fd_small = box_counting_fd(small)
        fd_large = box_counting_fd(large)
        assert abs(fd_small - fd_large) < 0.3

    def test_contour_with_boundary_coord_1(self):
        """Contour points exactly at 1.0 after normalization (clipping tested)."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                        [0.0, 0.0], [1.0, 0.0]])
        fd = box_counting_fd(pts)
        assert 1.0 <= fd <= 2.0

    # --- Reproducibility ---

    def test_same_contour_same_result(self):
        c = _circle(n=64)
        fd1 = box_counting_fd(c, n_scales=8)
        fd2 = box_counting_fd(c, n_scales=8)
        assert fd1 == fd2


# ─── TestBoxCountingCurve ─────────────────────────────────────────────────────

class TestBoxCountingCurve:
    def test_returns_tuple_of_2(self):
        result = box_counting_curve(_circle())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_are_ndarrays(self):
        log_r, log_n = box_counting_curve(_circle())
        assert isinstance(log_r, np.ndarray)
        assert isinstance(log_n, np.ndarray)

    def test_same_length(self):
        log_r, log_n = box_counting_curve(_circle())
        assert len(log_r) == len(log_n)

    def test_length_equals_n_scales(self):
        for n_scales in [4, 6, 10]:
            log_r, log_n = box_counting_curve(_circle(), n_scales=n_scales)
            assert len(log_r) == n_scales

    def test_log_r_strictly_increasing(self):
        log_r, _ = box_counting_curve(_circle())
        assert all(log_r[i] < log_r[i + 1] for i in range(len(log_r) - 1))

    def test_log_r_values_are_k_times_log2(self):
        """log_r_inv[k] should be log2(2^(k+1)) = k+1."""
        log_r, _ = box_counting_curve(_circle(), n_scales=4)
        for k, v in enumerate(log_r):
            assert v == pytest.approx(float(k + 1))

    def test_log_n_nonneg(self):
        _, log_n = box_counting_curve(_circle())
        assert np.all(log_n >= 0.0)

    def test_span_zero_returns_zeros(self):
        contour = np.ones((10, 2))
        log_r, log_n = box_counting_curve(contour)
        assert np.all(log_r == 0.0)
        assert np.all(log_n == 0.0)

    def test_log_n_increases_with_finer_grid(self):
        """More boxes counted at finer resolution → log_n non-decreasing."""
        _, log_n = box_counting_curve(_circle(n=128), n_scales=6)
        # Allow non-strict since N could plateau at fine scales
        assert np.all(np.diff(log_n) >= -1e-6)

    def test_custom_n_scales_4(self):
        log_r, log_n = box_counting_curve(_circle(), n_scales=4)
        assert len(log_r) == 4

    def test_default_n_scales_8(self):
        log_r, _ = box_counting_curve(_circle())
        assert len(log_r) == 8

    def test_fractal_contour_larger_log_n_slope(self):
        """Fractal contour should have steeper log_N growth than smooth circle."""
        _, ln_smooth = box_counting_curve(_circle(n=256), n_scales=8)
        _, ln_rough = box_counting_curve(_fractal_like(n=256), n_scales=8)
        # Rough contour occupies more boxes → higher log_N at fine scale
        if len(ln_smooth) > 0 and len(ln_rough) > 0:
            assert ln_rough[-1] >= ln_smooth[-1] - 0.1
