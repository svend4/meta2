"""
Benchmarks: preprocessing pipeline (segmentation, contour, colour, orientation).

Run:
    python -m pytest benchmarks/bench_preprocessing.py -v
    python -m pytest benchmarks/bench_preprocessing.py -v --tb=short -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images,
    make_processed_fragments,
    timeit_fn,
    BenchResult,
)

from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.preprocessing.color_norm import normalize_color
from puzzle_reconstruction.preprocessing.orientation import estimate_orientation

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fragment(width: int = 500, height: int = 400, seed: int = 0) -> np.ndarray:
    """Return a single synthetic BGR fragment image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(200, 255, (height, width, 3), dtype=np.uint8)
    # Add a dark rectangular "paper" region with some texture
    margin = 30
    img[margin:-margin, margin:-margin] = rng.integers(
        180, 220, (height - 2 * margin, width - 2 * margin, 3), dtype=np.uint8
    )
    # Simulate text lines
    for row in range(margin + 20, height - margin, 25):
        img[row : row + 3, margin + 10 : width - margin - 10] = rng.integers(
            0, 80, (3, width - 2 * margin - 20, 3), dtype=np.uint8
        )
    return img


def _make_mask(width: int = 500, height: int = 400) -> np.ndarray:
    """Return a synthetic uint8 binary mask (255 = paper)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    margin = 30
    mask[margin:-margin, margin:-margin] = 255
    return mask


# ---------------------------------------------------------------------------
# BenchPreprocessing
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class BenchPreprocessing:
    """Benchmarks for individual preprocessing steps."""

    # ── Segmentation ─────────────────────────────────────────────────────────

    def test_segment_otsu(self, benchmark_result):
        img = _make_fragment(500, 400, seed=1)
        ms = timeit_fn(segment_fragment, 5, img, method="otsu")
        benchmark_result.record_time("segment_otsu_400x500", ms)
        assert ms < 500, f"segment_otsu took {ms:.1f} ms (expected < 500 ms)"

    def test_segment_adaptive(self, benchmark_result):
        img = _make_fragment(500, 400, seed=2)
        ms = timeit_fn(segment_fragment, 5, img, method="adaptive")
        benchmark_result.record_time("segment_adaptive_400x500", ms)
        assert ms < 1000, f"segment_adaptive took {ms:.1f} ms (expected < 1000 ms)"

    # ── Contour extraction ────────────────────────────────────────────────────

    def test_extract_contour(self, benchmark_result):
        mask = _make_mask(500, 400)
        ms = timeit_fn(extract_contour, 10, mask)
        benchmark_result.record_time("extract_contour_400x500", ms)
        assert ms < 200, f"extract_contour took {ms:.1f} ms (expected < 200 ms)"

    # ── Colour normalisation ──────────────────────────────────────────────────

    def test_color_norm(self, benchmark_result):
        img = _make_fragment(500, 400, seed=3)
        ms = timeit_fn(normalize_color, 5, img)
        benchmark_result.record_time("color_norm_400x500", ms)
        assert ms < 1000, f"color_norm took {ms:.1f} ms (expected < 1000 ms)"

    # ── Orientation estimation ────────────────────────────────────────────────

    def test_orientation_estimate(self, benchmark_result):
        img = _make_fragment(500, 400, seed=4)
        ms = timeit_fn(estimate_orientation, 5, img)
        benchmark_result.record_time("orientation_estimate_400x500", ms)
        assert ms < 1000, f"orientation_estimate took {ms:.1f} ms (expected < 1000 ms)"


# ---------------------------------------------------------------------------
# Full chain scaling benchmarks
# ---------------------------------------------------------------------------

def _run_full_chain(images):
    """Process each image through the full preprocessing chain."""
    results = []
    for img in images:
        mask = segment_fragment(img, method="otsu")
        contour = extract_contour(mask)
        img_norm = normalize_color(img)
        angle = estimate_orientation(img_norm, mask)
        results.append((mask, contour, img_norm, angle))
    return results


@pytest.mark.benchmark
class BenchFullChain:
    """End-to-end preprocessing chain scaling benchmarks."""

    def test_full_chain_1_fragment(self, benchmark_result):
        images = make_synthetic_images(1, width=500, height=400, seed=10)
        t0 = time.perf_counter()
        _run_full_chain(images)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("full_chain_1_fragment", ms)
        assert ms < 5000, f"full_chain_1 took {ms:.1f} ms"

    def test_full_chain_4_fragments(self, benchmark_result):
        images = make_synthetic_images(4, width=500, height=400, seed=11)
        t0 = time.perf_counter()
        _run_full_chain(images)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("full_chain_4_fragments", ms)
        assert ms < 50_000, f"full_chain_4 took {ms:.1f} ms (expected < 50000 ms)"

    def test_full_chain_9_fragments(self, benchmark_result):
        images = make_synthetic_images(9, width=500, height=400, seed=12)
        t0 = time.perf_counter()
        _run_full_chain(images)
        ms = (time.perf_counter() - t0) * 1000
        benchmark_result.record_time("full_chain_9_fragments", ms)
        assert ms < 200_000, f"full_chain_9 took {ms:.1f} ms (expected < 200000 ms)"

    @pytest.mark.parametrize("n_frags", [1, 4, 9])
    def test_full_chain_scaling(self, n_frags):
        images = make_synthetic_images(n_frags, width=500, height=400, seed=99)
        t0 = time.perf_counter()
        _run_full_chain(images)
        ms = (time.perf_counter() - t0) * 1000
        # Generous upper bound: 10 s per fragment
        assert ms < n_frags * 10_000, (
            f"full_chain_{n_frags} took {ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_result():
    result = BenchResult("Preprocessing")
    yield result
    result.print_summary()
