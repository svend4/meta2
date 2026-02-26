"""
Shared fixtures for benchmarks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils import (
    make_synthetic_images, make_processed_fragments, make_contour,
)


# ── Contours ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def contour_64():
    return make_contour(64)


@pytest.fixture(scope="session")
def contour_128():
    return make_contour(128)


@pytest.fixture(scope="session")
def contour_256():
    return make_contour(256)


@pytest.fixture(scope="session")
def contour_512():
    return make_contour(512)


# ── Images ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def images_4():
    return make_synthetic_images(4, seed=1)


@pytest.fixture(scope="session")
def images_9():
    return make_synthetic_images(9, seed=2)


@pytest.fixture(scope="session")
def images_16():
    return make_synthetic_images(16, width=400, height=500, seed=3)


# ── Processed fragments ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fragments_4(images_4):
    return make_processed_fragments(images_4)


@pytest.fixture(scope="session")
def fragments_9(images_9):
    return make_processed_fragments(images_9)
