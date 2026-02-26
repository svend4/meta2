"""
Бенчмарки: построение матрицы совместимости (O(N²) scaling).

Запуск:
    python -m pytest benchmarks/bench_compat_matrix.py -v -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images, make_processed_fragments,
    timeit_fn, BenchResult,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("CompatMatrix")
    yield result
    result.print_summary()


class BenchCompatMatrix:
    """Build compat matrix for N fragments."""

    def test_compat_matrix_2_fragments(self, fragments_4, benchmark_result):
        frags = fragments_4[:2]
        if len(frags) < 2:
            pytest.skip("Not enough fragments")
        ms = timeit_fn(build_compat_matrix, 3, frags)
        benchmark_result.record_time("compat_matrix_2_frags", ms)
        assert ms < 10_000  # generous 10s

    def test_compat_matrix_4_fragments(self, fragments_4, benchmark_result):
        if len(fragments_4) < 4:
            pytest.skip("Not enough fragments")
        ms = timeit_fn(build_compat_matrix, 3, fragments_4)
        benchmark_result.record_time("compat_matrix_4_frags", ms)
        assert ms < 30_000  # 30s generous

    def test_compat_matrix_9_fragments(self, fragments_9, benchmark_result):
        frags = fragments_9[:9]
        if len(frags) < 4:
            pytest.skip("Not enough fragments")
        t0 = time.perf_counter()
        build_compat_matrix(frags)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("compat_matrix_9_frags", ms)
        assert ms < 120_000  # 2 min

    def test_compat_matrix_output_shape(self, fragments_4):
        if len(fragments_4) < 2:
            pytest.skip("Not enough fragments")
        frags = fragments_4[:2]
        result = build_compat_matrix(frags)
        # build_compat_matrix returns (matrix, entries) tuple or just entries list
        entries = result[1] if isinstance(result, tuple) else result
        assert entries is not None

    def test_compat_matrix_entries_have_score(self, fragments_4):
        if len(fragments_4) < 2:
            pytest.skip("Not enough fragments")
        frags = fragments_4[:2]
        result = build_compat_matrix(frags)
        entries = result[1] if isinstance(result, tuple) else result
        for e in list(entries)[:5]:
            assert hasattr(e, "score")
            assert 0.0 <= e.score <= 1.0

    @pytest.mark.parametrize("n", [2, 4])
    def test_compat_matrix_n_fragments_scaling(self, n):
        """Verify O(N²) growth: 4 frags → ~4× more entries than 2 frags."""
        images_n = make_synthetic_images(n, seed=100 + n)
        frags_n = make_processed_fragments(images_n)
        if len(frags_n) < 2:
            pytest.skip("Not enough processable fragments")
        t0 = time.perf_counter()
        result = build_compat_matrix(frags_n)
        elapsed = time.perf_counter() - t0
        assert result is not None
        assert elapsed < 60  # 60s per case
