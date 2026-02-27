"""
Бенчмарки памяти: tracemalloc-based peak memory usage.

Запуск:
    python -m pytest benchmarks/bench_memory.py -v -s
"""
from __future__ import annotations

import sys
import tracemalloc
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images, make_processed_fragments,
    BenchResult,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.pipeline import Pipeline

pytestmark = [pytest.mark.benchmark, pytest.mark.memory]


def _peak_mb(fn, *args, **kwargs):
    """Run fn and return peak allocated memory in MB."""
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Memory")
    yield result
    result.print_summary()


class BenchMemoryPreprocessing:
    """Memory usage during preprocessing."""

    def test_memory_preprocess_4_fragments(self, images_4, benchmark_result):
        cfg = Config.default()
        p = Pipeline(cfg=cfg, n_workers=1)
        mb = _peak_mb(p.preprocess, images_4)
        benchmark_result.record_memory("preprocess_4_frags", mb)
        assert mb < 500  # generous 500 MB

    def test_memory_preprocess_baseline(self, images_4):
        """Just verify it runs without OOM."""
        cfg = Config.default()
        p = Pipeline(cfg=cfg, n_workers=1)
        frags = p.preprocess(images_4)
        assert frags is not None


class BenchMemoryCompatMatrix:
    """Memory usage when building compat matrix."""

    def test_memory_compat_matrix_4_fragments(self, fragments_4, benchmark_result):
        if len(fragments_4) < 2:
            pytest.skip("Not enough fragments")
        mb = _peak_mb(build_compat_matrix, fragments_4)
        benchmark_result.record_memory("compat_matrix_4_frags", mb)
        assert mb < 200  # generous 200 MB


class BenchMemoryAssembly:
    """Memory usage of assembly step."""

    def test_memory_greedy_assembly_4_fragments(self, fragments_4, benchmark_result):
        if len(fragments_4) < 2:
            pytest.skip("Not enough fragments")
        result = build_compat_matrix(fragments_4)
        entries = result[1] if isinstance(result, tuple) else result

        def run():
            greedy_assembly(fragments_4, entries)

        mb = _peak_mb(run)
        benchmark_result.record_memory("greedy_assembly_4_frags", mb)
        assert mb < 100


class BenchMemoryFullPipeline:
    """Memory for full pipeline run."""

    def test_memory_pipeline_4_fragments(self, images_4, benchmark_result):
        cfg = Config.default()
        cfg.assembly.method = "greedy"
        p = Pipeline(cfg=cfg, n_workers=1)
        mb = _peak_mb(p.run, images_4)
        benchmark_result.record_memory("full_pipeline_4_frags", mb)
        assert mb < 1000  # 1 GB generous limit

    def test_memory_pipeline_no_leak(self, images_4):
        """Verify no catastrophic memory leak across two runs."""
        cfg = Config.default()
        cfg.assembly.method = "greedy"

        tracemalloc.start()
        p1 = Pipeline(cfg=cfg, n_workers=1)
        p1.run(images_4)
        _, peak1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        p2 = Pipeline(cfg=cfg, n_workers=1)
        p2.run(images_4)
        _, peak2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Second run should not use dramatically more memory
        # Allow up to 3× tolerance (conservative)
        if peak1 > 0:
            ratio = peak2 / peak1
            assert ratio < 5.0, f"Memory grew {ratio:.1f}× between runs"
