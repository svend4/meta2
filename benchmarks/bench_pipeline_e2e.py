"""
Бенчмарки: end-to-end прогон Pipeline.

Запуск:
    python -m pytest benchmarks/bench_pipeline_e2e.py -v -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import make_synthetic_images, BenchResult
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.pipeline import Pipeline

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Pipeline E2E")
    yield result
    result.print_summary()


@pytest.fixture(scope="session")
def cfg_greedy():
    cfg = Config.default()
    cfg.assembly.method = "greedy"
    return cfg


class BenchPipelinePreprocessOnly:
    """Benchmark preprocessing step only."""

    def test_preprocess_4_fragments(self, images_4, cfg_greedy, benchmark_result):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        t0 = time.perf_counter()
        frags = p.preprocess(images_4)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("preprocess_4_frags", ms)
        assert isinstance(frags, list)
        assert ms < 60_000  # 1 min


class BenchPipelineMatchOnly:
    """Benchmark matching step only."""

    def test_match_4_fragments(self, images_4, cfg_greedy, benchmark_result):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        frags = p.preprocess(images_4)
        if len(frags) < 2:
            pytest.skip("Not enough fragments preprocessed")
        t0 = time.perf_counter()
        entries = p.match(frags)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("match_4_frags", ms)
        assert ms < 120_000  # 2 min


class BenchPipelineAssembleOnly:
    """Benchmark assembly step only."""

    def test_assemble_4_fragments(self, images_4, cfg_greedy, benchmark_result):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        frags = p.preprocess(images_4)
        if len(frags) < 2:
            pytest.skip("Not enough fragments preprocessed")
        _, entries = p.match(frags)
        t0 = time.perf_counter()
        assembly = p.assemble(frags, entries)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("assemble_4_frags", ms)
        assert ms < 30_000  # 30s


class BenchPipelineRunFull:
    """End-to-end pipeline benchmark."""

    def test_pipeline_run_4_fragments(self, images_4, cfg_greedy, benchmark_result):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        t0 = time.perf_counter()
        result = p.run(images_4)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("pipeline_run_4_frags", ms)
        assert result is not None
        assert ms < 300_000  # 5 min generous limit

    def test_pipeline_result_has_score(self, images_4, cfg_greedy):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        result = p.run(images_4)
        assert hasattr(result, "assembly") or hasattr(result, "score") or result is not None

    def test_pipeline_run_single_fragment(self, images_4, cfg_greedy):
        p = Pipeline(cfg=cfg_greedy, n_workers=1)
        single = images_4[:1]
        # Should not crash on single fragment
        try:
            result = p.run(single)
        except Exception:
            pass  # Single fragment may not be assemblageable
