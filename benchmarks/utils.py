"""
Вспомогательные функции для бенчмарков.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.models import Fragment


# ─── Timing ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager: measures wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if label:
        print(f"  [{label}] {elapsed * 1000:.2f} ms")
    return elapsed


def timeit_fn(fn: Callable, n: int = 5, *args, **kwargs) -> float:
    """Time a function over n runs, return median ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000  # ms


# ─── Memory ───────────────────────────────────────────────────────────────────

@contextmanager
def mem_tracer():
    """Context manager: measures peak memory in bytes."""
    tracemalloc.start()
    try:
        yield
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def measure_peak_mb(fn: Callable, *args, **kwargs) -> float:
    """Run fn and return peak memory in MB."""
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# ─── Data generation ──────────────────────────────────────────────────────────

def make_synthetic_images(n_pieces: int, width: int = 300, height: int = 400,
                           seed: int = 42) -> List[np.ndarray]:
    """Generate n_pieces synthetic document fragment images."""
    doc = generate_test_document(width=width, height=height, seed=seed)
    return tear_document(doc, n_pieces=n_pieces, noise_level=0.3, seed=seed)


def make_processed_fragments(images: List[np.ndarray]) -> List[Fragment]:
    """Segment and build Fragment objects from raw images."""
    fragments = []
    for i, img in enumerate(images):
        try:
            mask = segment_fragment(img, method="otsu")
            contour = extract_contour(mask)
            if len(contour) < 4:
                continue
            tangram = fit_tangram(contour)
            fractal = compute_fractal_signature(contour)
            frag = Fragment(fragment_id=i, image=img, mask=mask, contour=contour)
            frag.tangram = tangram
            frag.fractal = fractal
            edges = build_edge_signatures(frag)
            frag.edges = edges
            fragments.append(frag)
        except Exception:
            continue
    return fragments


def make_contour(n_pts: int = 256, shape: str = "circle") -> np.ndarray:
    """Generate a synthetic contour with n_pts points."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    if shape == "circle":
        x = 100 * np.cos(t)
        y = 100 * np.sin(t)
    elif shape == "square":
        # Approximate square
        x = 100 * np.clip(np.cos(t), -0.7, 0.7)
        y = 100 * np.clip(np.sin(t), -0.7, 0.7)
    elif shape == "star":
        r = 100 + 40 * np.cos(5 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
    else:
        x = 100 * np.cos(t)
        y = 100 * np.sin(t)
    return np.column_stack([x, y]).astype(np.float32)


# ─── BenchResult ──────────────────────────────────────────────────────────────

class BenchResult:
    """Simple container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.timings: Dict[str, float] = {}  # label: ms
        self.memory: Dict[str, float] = {}   # label: MB

    def record_time(self, label: str, ms: float) -> None:
        self.timings[label] = ms

    def record_memory(self, label: str, mb: float) -> None:
        self.memory[label] = mb

    def summary(self) -> str:
        lines = [f"=== {self.name} ==="]
        for label, ms in sorted(self.timings.items()):
            lines.append(f"  {label:<40}: {ms:8.2f} ms")
        for label, mb in sorted(self.memory.items()):
            lines.append(f"  {label:<40}: {mb:8.2f} MB")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())
