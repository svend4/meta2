"""
Benchmarks: verification suite and reconstruction metrics.

Run:
    python -m pytest benchmarks/bench_verification.py -v
    python -m pytest benchmarks/bench_verification.py -v --tb=short -s
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images,
    timeit_fn,
    BenchResult,
)

from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.verification.suite import VerificationSuite
from puzzle_reconstruction.verification.metrics import evaluate_reconstruction

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helpers: build synthetic Assembly objects
# ---------------------------------------------------------------------------

def _make_fragment_image(width: int = 300, height: int = 200,
                         seed: int = 0) -> np.ndarray:
    """Return a simple synthetic BGR image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(180, 255, (height, width, 3), dtype=np.uint8)
    margin = 20
    img[margin:-margin, margin:-margin] = rng.integers(
        200, 240, (height - 2 * margin, width - 2 * margin, 3), dtype=np.uint8
    )
    return img


def _make_assembly(n: int, frag_w: int = 300, frag_h: int = 200) -> Assembly:
    """Build a synthetic Assembly with n fragments arranged in a grid."""
    fragments: List[Fragment] = []
    placements: List[Placement] = []

    cols = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        row, col = divmod(i, cols)
        img = _make_fragment_image(frag_w, frag_h, seed=i)
        mask = np.zeros((frag_h, frag_w), dtype=np.uint8)
        mask[20:-20, 20:-20] = 255

        frag = Fragment(
            fragment_id=i,
            image=img,
            mask=mask,
        )
        fragments.append(frag)

        placements.append(Placement(
            fragment_id=i,
            position=(float(col * (frag_w + 5)), float(row * (frag_h + 5))),
            rotation=0.0,
        ))

    return Assembly(
        placements=placements,
        fragments=fragments,
        total_score=0.75,
    )


def _make_reconstruction_dicts(
    n: int, frag_w: int = 300, frag_h: int = 200
) -> Tuple[Dict[int, Tuple[np.ndarray, float]],
           Dict[int, Tuple[np.ndarray, float]]]:
    """Build predicted and ground-truth position dicts for evaluate_reconstruction."""
    cols = int(np.ceil(np.sqrt(n)))
    gt: Dict[int, Tuple[np.ndarray, float]] = {}
    pred: Dict[int, Tuple[np.ndarray, float]] = {}
    rng = np.random.default_rng(42)
    for i in range(n):
        row, col = divmod(i, cols)
        gt_pos = np.array([col * (frag_w + 5), row * (frag_h + 5)], dtype=float)
        gt[i] = (gt_pos, 0.0)
        # Predicted: add small noise
        noise = rng.uniform(-15, 15, 2)
        pred[i] = (gt_pos + noise, rng.uniform(-5, 5))
    return pred, gt


# ---------------------------------------------------------------------------
# BenchVerificationSuite: run_all and individual validators
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class BenchVerificationSuite:
    """Benchmarks for VerificationSuite.run_all and individual validators."""

    def test_run_all_validators(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite()
        ms = timeit_fn(suite.run_all, 3, asm)
        benchmark_result.record_time("run_all_validators_4_frags", ms)
        assert ms < 30_000, f"run_all took {ms:.1f} ms"

    # ── Individual validators ─────────────────────────────────────────────────

    def test_individual_spatial(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite(validators=["spatial"])
        ms = timeit_fn(suite.run, 5, asm)
        benchmark_result.record_time("individual_spatial_4_frags", ms)
        assert ms < 5_000, f"spatial took {ms:.1f} ms"

    def test_individual_boundary(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite(validators=["boundary"])
        ms = timeit_fn(suite.run, 5, asm)
        benchmark_result.record_time("individual_boundary_4_frags", ms)
        assert ms < 5_000, f"boundary took {ms:.1f} ms"

    def test_individual_color(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite(validators=["seam"])
        ms = timeit_fn(suite.run, 5, asm)
        benchmark_result.record_time("individual_color_seam_4_frags", ms)
        assert ms < 5_000, f"color/seam took {ms:.1f} ms"

    def test_individual_layout(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite(validators=["layout"])
        ms = timeit_fn(suite.run, 5, asm)
        benchmark_result.record_time("individual_layout_4_frags", ms)
        assert ms < 5_000, f"layout took {ms:.1f} ms"

    def test_individual_seam(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite(validators=["edge_quality"])
        ms = timeit_fn(suite.run, 5, asm)
        benchmark_result.record_time("individual_seam_edge_quality_4_frags", ms)
        assert ms < 5_000, f"seam/edge_quality took {ms:.1f} ms"


# ---------------------------------------------------------------------------
# BenchEvaluateReconstruction: metrics.py
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class BenchEvaluateReconstruction:
    """Benchmarks for evaluate_reconstruction from metrics.py."""

    def test_evaluate_reconstruction_4(self, benchmark_result):
        pred, gt = _make_reconstruction_dicts(4)
        ms = timeit_fn(evaluate_reconstruction, 10, pred, gt)
        benchmark_result.record_time("evaluate_reconstruction_4_frags", ms)
        assert ms < 1_000, f"evaluate_reconstruction_4 took {ms:.1f} ms"

    def test_evaluate_reconstruction_9(self, benchmark_result):
        pred, gt = _make_reconstruction_dicts(9)
        ms = timeit_fn(evaluate_reconstruction, 10, pred, gt)
        benchmark_result.record_time("evaluate_reconstruction_9_frags", ms)
        assert ms < 1_000, f"evaluate_reconstruction_9 took {ms:.1f} ms"

    def test_evaluate_reconstruction_25(self, benchmark_result):
        pred, gt = _make_reconstruction_dicts(25)
        ms = timeit_fn(evaluate_reconstruction, 10, pred, gt)
        benchmark_result.record_time("evaluate_reconstruction_25_frags", ms)
        assert ms < 5_000, f"evaluate_reconstruction_25 took {ms:.1f} ms"


# ---------------------------------------------------------------------------
# BenchVerificationScaling: suite scaling with fragment count
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class BenchVerificationScaling:
    """Scaling benchmarks: verification suite vs fragment count."""

    def test_verification_suite_2_fragments(self, benchmark_result):
        asm = _make_assembly(2)
        suite = VerificationSuite()
        t0 = time.perf_counter()
        suite.run_all(asm)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("verification_suite_2_frags", ms)
        assert ms < 30_000, f"suite_2 took {ms:.1f} ms"

    def test_verification_suite_4_fragments(self, benchmark_result):
        asm = _make_assembly(4)
        suite = VerificationSuite()
        t0 = time.perf_counter()
        suite.run_all(asm)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("verification_suite_4_frags", ms)
        assert ms < 60_000, f"suite_4 took {ms:.1f} ms"

    def test_verification_suite_9_fragments(self, benchmark_result):
        asm = _make_assembly(9)
        suite = VerificationSuite()
        t0 = time.perf_counter()
        suite.run_all(asm)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("verification_suite_9_frags", ms)
        assert ms < 120_000, f"suite_9 took {ms:.1f} ms"

    @pytest.mark.parametrize("n_frags", [2, 4, 9])
    def test_suite_scaling(self, n_frags):
        asm = _make_assembly(n_frags)
        suite = VerificationSuite()
        t0 = time.perf_counter()
        suite.run_all(asm)
        ms = (time.perf_counter() - t0) * 1000
        # Generous bound: 15 s per fragment
        assert ms < n_frags * 15_000, (
            f"suite_{n_frags} took {ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Verification")
    yield result
    result.print_summary()
