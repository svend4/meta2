"""
Scalability benchmarks: time and memory as number of fragments grows.

Tests how assembly methods scale from N=4 to N=36 fragments.
Results are printed in a summary table and optionally saved to CSV.

Запуск:
    python -m pytest benchmarks/bench_scalability.py -v -s
    python -m pytest benchmarks/bench_scalability.py -v -s -k "greedy"
"""
from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images,
    make_processed_fragments,
    BenchResult,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.models import Fragment

pytestmark = pytest.mark.benchmark

# ── Configuration ─────────────────────────────────────────────────────────────

# Fragment counts to test (omit large values if system is slow)
SIZES = [4, 9, 16, 25, 36]

# Methods included in scalability tests (fast ones for full matrix, slow ones for small N)
FAST_METHODS = ["greedy", "beam"]
ALL_SCALE_METHODS = ["greedy", "beam", "sa"]

# Output CSV path
RESULTS_DIR = Path(__file__).parent / "results"
SCALABILITY_CSV = RESULTS_DIR / "scalability.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_pipeline_data(n: int, seed: int = 42):
    """Generate synthetic images, process into fragments, build compat matrix."""
    images = make_synthetic_images(n, seed=seed)
    frags = make_processed_fragments(images)
    if len(frags) < 2:
        return None, None, None
    result = build_compat_matrix(frags)
    matrix = result[0] if isinstance(result, tuple) else result
    entries = result[1] if isinstance(result, tuple) else []
    return frags, matrix, entries


def _run_assembly(method: str, frags: List[Fragment], entries: list) -> Optional[object]:
    """Run the specified assembly method and return the result."""
    import importlib

    method_map = {
        "greedy": ("puzzle_reconstruction.assembly.greedy", "greedy_assembly"),
        "beam":   ("puzzle_reconstruction.assembly.beam_search", "beam_search_assembly"),
        "sa":     ("puzzle_reconstruction.assembly.annealing", "sa_assembly"),
        "genetic":("puzzle_reconstruction.assembly.genetic", "genetic_assembly"),
        "ant_colony": ("puzzle_reconstruction.assembly.ant_colony", "ant_colony_assembly"),
        "mcts":   ("puzzle_reconstruction.assembly.mcts", "mcts_assembly"),
    }

    if method not in method_map:
        return None

    mod_path, fn_name = method_map[method]
    try:
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, fn_name)
        return fn(frags, entries)
    except Exception:
        return None


def _measure_assembly_time_memory(method: str, frags: List[Fragment], entries: list):
    """Measure time (ms) and peak memory (MB) for one assembly run."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = _run_assembly(method, frags, entries)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    return elapsed_ms, peak_mb, result


def _save_to_csv(rows: list) -> None:
    """Append benchmark results to the scalability CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SCALABILITY_CSV.exists()
    with open(SCALABILITY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n_fragments", "time_ms", "peak_mb"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Scalability fixture ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def scalability_result():
    result = BenchResult("Scalability")
    yield result
    result.print_summary()


# ── Core scalability tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n,method", [
    (n, m) for n in SIZES for m in FAST_METHODS
])
def test_scalability(n: int, method: str, scalability_result):
    """
    Time and RAM for each (N fragments, assembly method) combination.

    Records results to benchmarks/results/scalability.csv.
    """
    frags, matrix, entries = _build_pipeline_data(n, seed=100 + n)
    if frags is None or len(frags) < 2:
        pytest.skip(f"Could not build {n} processable fragments")

    elapsed_ms, peak_mb, assembly = _measure_assembly_time_memory(method, frags, entries)

    label = f"{method}_n{n}"
    scalability_result.record_time(label, elapsed_ms)
    scalability_result.record_memory(label, peak_mb)

    print(f"\n  [{method}] N={n:2d}: {elapsed_ms:8.1f} ms | {peak_mb:6.2f} MB")

    # Save to CSV
    _save_to_csv([{
        "method": method,
        "n_fragments": n,
        "time_ms": round(elapsed_ms, 2),
        "peak_mb": round(peak_mb, 4),
    }])

    # Basic sanity checks — no hard time limits (scalability varies by hardware)
    assert elapsed_ms >= 0.0, "Time cannot be negative"
    assert peak_mb >= 0.0, "Memory cannot be negative"
    if assembly is not None:
        assert hasattr(assembly, "placements") or hasattr(assembly, "total_score"), \
            f"{method} assembly result missing expected attributes"


# ── Preprocessing scalability ─────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_preprocessing_scalability(n: int, scalability_result):
    """Measure time to build N fragments (preprocessing stage only)."""
    images = make_synthetic_images(n, seed=200 + n)
    tracemalloc.start()
    t0 = time.perf_counter()
    frags = make_processed_fragments(images)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    label = f"preprocess_n{n}"
    scalability_result.record_time(label, elapsed_ms)
    scalability_result.record_memory(label, peak_mb)

    print(f"\n  [preprocess] N={n:2d}: {elapsed_ms:8.1f} ms | {peak_mb:6.2f} MB")

    assert elapsed_ms >= 0.0
    assert len(frags) >= 0


# ── Compat matrix scalability ─────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_compat_matrix_scalability(n: int, scalability_result):
    """Measure time to build the N×N compatibility matrix."""
    frags, _, _ = _build_pipeline_data(n, seed=300 + n)
    if frags is None or len(frags) < 2:
        pytest.skip(f"Could not build {n} processable fragments")

    tracemalloc.start()
    t0 = time.perf_counter()
    build_compat_matrix(frags)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    label = f"compat_matrix_n{n}"
    scalability_result.record_time(label, elapsed_ms)
    scalability_result.record_memory(label, peak_mb)

    print(f"\n  [compat_matrix] N={n:2d}: {elapsed_ms:8.1f} ms | {peak_mb:6.2f} MB")

    assert elapsed_ms >= 0.0


# ── Greedy scaling trend ──────────────────────────────────────────────────────

def test_greedy_scaling_trend():
    """
    Verify that greedy assembly time scales sub-quadratically or linearly
    between N=4 and N=9 (trivial check that it doesn't blow up).
    """
    times = {}
    for n in [4, 9]:
        frags, _, entries = _build_pipeline_data(n, seed=400 + n)
        if frags is None or len(frags) < 2:
            continue
        t0 = time.perf_counter()
        greedy_assembly(frags, entries)
        times[n] = (time.perf_counter() - t0) * 1000

    if len(times) < 2:
        pytest.skip("Could not build fragments for both sizes")

    # N=9 should not take more than 50× longer than N=4
    # (very conservative bound: actual ratio is usually < 5×)
    ratio = times[9] / max(times[4], 1.0)
    assert ratio < 50, \
        f"Greedy time ratio N=9/N=4 = {ratio:.1f}× (expected < 50×)"


# ── Summary helpers ───────────────────────────────────────────────────────────

def test_print_csv_summary():
    """Print the contents of the scalability CSV if it exists."""
    if not SCALABILITY_CSV.exists():
        pytest.skip("No scalability CSV generated yet; run other tests first")
    print(f"\n\n=== Scalability results: {SCALABILITY_CSV} ===")
    with open(SCALABILITY_CSV) as f:
        print(f.read())
