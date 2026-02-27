"""
Бенчмарки: дескрипторы формы (CSS, Box-counting, IFS, DTW, Freeman).

Запуск:
    python -m pytest benchmarks/bench_descriptors.py -v
    python -m pytest benchmarks/bench_descriptors.py -v --tb=short -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import make_contour, timeit_fn, BenchResult

from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space, css_to_feature_vector, css_similarity,
)
from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_fd, box_counting_curve,
)
from puzzle_reconstruction.algorithms.synthesis import compute_fractal_signature

pytestmark = pytest.mark.benchmark


class BenchCSS:
    """Curvature Scale Space benchmarks."""

    def test_css_compute_64pts(self, contour_64, benchmark_result):
        ms = timeit_fn(curvature_scale_space, 3, contour_64)
        benchmark_result.record_time("css_compute_64pts", ms)
        assert ms < 500  # should be under 500ms

    def test_css_compute_128pts(self, contour_128, benchmark_result):
        ms = timeit_fn(curvature_scale_space, 3, contour_128)
        benchmark_result.record_time("css_compute_128pts", ms)
        assert ms < 1000

    def test_css_compute_256pts(self, contour_256, benchmark_result):
        ms = timeit_fn(curvature_scale_space, 3, contour_256)
        benchmark_result.record_time("css_compute_256pts", ms)
        assert ms < 2000

    def test_css_feature_vector(self, contour_256, benchmark_result):
        css = curvature_scale_space(contour_256)
        ms = timeit_fn(css_to_feature_vector, 5, css)
        benchmark_result.record_time("css_feature_vector_256pts", ms)
        assert ms < 100

    def test_css_similarity(self, contour_256, benchmark_result):
        css = curvature_scale_space(contour_256)
        vec = css_to_feature_vector(css)
        ms = timeit_fn(css_similarity, 20, vec, vec)
        benchmark_result.record_time("css_similarity", ms)
        assert ms < 10

    @pytest.mark.parametrize("n_pts", [64, 128, 256, 512])
    def test_css_scaling(self, n_pts):
        contour = make_contour(n_pts)
        t0 = time.perf_counter()
        curvature_scale_space(contour)
        ms = (time.perf_counter() - t0) * 1000
        # Should scale roughly O(N) not O(N^2)
        assert ms < 5000, f"css on {n_pts} pts took {ms:.1f}ms (too slow)"


class BenchBoxCounting:
    """Box-counting fractal dimension benchmarks."""

    def test_fd_compute_64pts(self, contour_64, benchmark_result):
        ms = timeit_fn(box_counting_fd, 10, contour_64)
        benchmark_result.record_time("box_counting_fd_64pts", ms)
        assert ms < 100

    def test_fd_compute_256pts(self, contour_256, benchmark_result):
        ms = timeit_fn(box_counting_fd, 10, contour_256)
        benchmark_result.record_time("box_counting_fd_256pts", ms)
        assert ms < 200

    def test_fd_curve_256pts(self, contour_256, benchmark_result):
        ms = timeit_fn(box_counting_curve, 10, contour_256)
        benchmark_result.record_time("box_counting_curve_256pts", ms)
        assert ms < 200

    @pytest.mark.parametrize("n_pts", [64, 128, 256, 512])
    def test_fd_scaling(self, n_pts):
        contour = make_contour(n_pts)
        t0 = time.perf_counter()
        box_counting_fd(contour)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 1000, f"box_counting on {n_pts} pts took {ms:.1f}ms"


class BenchFullFractalSignature:
    """Full fractal signature computation."""

    def test_full_signature_256pts(self, contour_256, benchmark_result):
        ms = timeit_fn(compute_fractal_signature, 3, contour_256)
        benchmark_result.record_time("full_fractal_signature_256pts", ms)
        assert ms < 5000

    def test_full_signature_128pts(self, contour_128, benchmark_result):
        ms = timeit_fn(compute_fractal_signature, 3, contour_128)
        benchmark_result.record_time("full_fractal_signature_128pts", ms)
        assert ms < 3000


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Descriptors")
    yield result
    result.print_summary()
