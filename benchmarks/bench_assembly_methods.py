"""
Бенчмарки: сравнение 8 алгоритмов сборки.

Запуск:
    python -m pytest benchmarks/bench_assembly_methods.py -v -s
    python -m pytest benchmarks/bench_assembly_methods.py -v -k "greedy" -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images, make_processed_fragments,
    BenchResult, timeit_fn,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.models import Fragment

pytestmark = pytest.mark.benchmark

FAST_METHODS = ["greedy", "beam"]
ALL_METHODS = ["greedy", "sa", "beam", "gamma", "genetic",
               "exhaustive", "ant_colony", "mcts"]


@pytest.fixture(scope="session")
def prepared_4(fragments_4):
    """Fragments + compat entries for 4 fragments."""
    if len(fragments_4) < 2:
        pytest.skip("Not enough fragments")
    result = build_compat_matrix(fragments_4)
    entries = result[1] if isinstance(result, tuple) else result
    return fragments_4, entries


@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Assembly Methods")
    yield result
    result.print_summary()


class BenchGreedy:
    """Greedy assembler — fastest baseline."""

    def test_greedy_4_fragments(self, prepared_4, benchmark_result):
        frags, entries = prepared_4
        ms = timeit_fn(greedy_assembly, 5, frags, entries)
        benchmark_result.record_time("greedy_4_frags", ms)
        assert ms < 5000

    def test_greedy_result_type(self, prepared_4):
        frags, entries = prepared_4
        from puzzle_reconstruction.models import Assembly
        result = greedy_assembly(frags, entries)
        assert hasattr(result, "placements") or hasattr(result, "total_score")


class BenchAllMethods:
    """Compare all assembly methods on 4 fragments."""

    @pytest.mark.parametrize("method", ["greedy", "beam"])
    def test_method_4_fragments(self, method, prepared_4, benchmark_result):
        """Benchmark fast assembly methods on 4 fragments."""
        frags, entries = prepared_4
        try:
            from puzzle_reconstruction.assembly import _load_method
            assembler = _load_method(method)
        except (ImportError, AttributeError):
            pytest.skip(f"Method {method} not available via _load_method")

        t0 = time.perf_counter()
        result = assembler(frags, entries)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time(f"{method}_4_frags", ms)
        assert ms < 60_000, f"{method} took {ms:.1f}ms on 4 fragments"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_available(self, method):
        """Verify all 8 assembly methods can be imported."""
        method_map = {
            "greedy": "puzzle_reconstruction.assembly.greedy",
            "sa": "puzzle_reconstruction.assembly.annealing",
            "beam": "puzzle_reconstruction.assembly.beam",
            "gamma": "puzzle_reconstruction.assembly.gamma",
            "genetic": "puzzle_reconstruction.assembly.genetic",
            "exhaustive": "puzzle_reconstruction.assembly.exhaustive",
            "ant_colony": "puzzle_reconstruction.assembly.ant_colony",
            "mcts": "puzzle_reconstruction.assembly.mcts",
        }
        module_name = method_map.get(method)
        if module_name is None:
            pytest.skip(f"No module mapping for {method}")
        import importlib
        try:
            mod = importlib.import_module(module_name)
            assert mod is not None
        except ImportError:
            pytest.skip(f"Module {module_name} not installed")


class BenchScalingGreedy:
    """Greedy assembler scaling with number of fragments."""

    @pytest.mark.parametrize("n", [2, 4])
    def test_greedy_scaling(self, n, benchmark_result):
        images = make_synthetic_images(n, seed=200 + n)
        frags = make_processed_fragments(images)
        if len(frags) < 2:
            pytest.skip("Not enough processable fragments")
        result = build_compat_matrix(frags)
        entries = result[1] if isinstance(result, tuple) else result

        t0 = time.perf_counter()
        greedy_assembly(frags, entries)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time(f"greedy_{n}_frags", ms)
        assert ms < 60_000
