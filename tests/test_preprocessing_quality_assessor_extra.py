"""Extra tests for puzzle_reconstruction/preprocessing/quality_assessor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.quality_assessor import (
    QualityReport,
    estimate_blur,
    estimate_noise,
    estimate_contrast,
    estimate_completeness,
    assess_quality,
    filter_by_quality,
    batch_assess_quality,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _edge_image():
    """High contrast edge image."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[:, 25:] = 255
    return img


# ─── QualityReport ────────────────────────────────────────────────────────────

class TestQualityReportExtra:
    def test_fields(self):
        r = QualityReport(blur_score=0.8, noise_score=0.9,
                          contrast_score=0.7, completeness=0.95,
                          overall=0.85, is_acceptable=True)
        assert r.is_acceptable is True

    def test_repr(self):
        r = QualityReport(blur_score=0.5, noise_score=0.5,
                          contrast_score=0.5, completeness=0.5,
                          overall=0.5, is_acceptable=True)
        s = repr(r)
        assert "0.500" in s


# ─── estimate_blur ────────────────────────────────────────────────────────────

class TestEstimateBlurExtra:
    def test_range(self):
        score = estimate_blur(_gray())
        assert 0.0 <= score <= 1.0

    def test_edge_sharper(self):
        edge_score = estimate_blur(_edge_image())
        flat_score = estimate_blur(_gray())
        assert edge_score > flat_score

    def test_bgr_input(self):
        score = estimate_blur(_bgr())
        assert isinstance(score, float)


# ─── estimate_noise ───────────────────────────────────────────────────────────

class TestEstimateNoiseExtra:
    def test_range(self):
        score = estimate_noise(_gray())
        assert 0.0 <= score <= 1.0

    def test_uniform_clean(self):
        score = estimate_noise(_gray())
        assert score > 0.9

    def test_bgr_input(self):
        score = estimate_noise(_bgr())
        assert isinstance(score, float)


# ─── estimate_contrast ────────────────────────────────────────────────────────

class TestEstimateContrastExtra:
    def test_range(self):
        score = estimate_contrast(_gray())
        assert 0.0 <= score <= 1.0

    def test_uniform_low(self):
        score = estimate_contrast(_gray())
        assert score < 0.1

    def test_edge_higher(self):
        score = estimate_contrast(_edge_image())
        assert score > estimate_contrast(_gray())


# ─── estimate_completeness ────────────────────────────────────────────────────

class TestEstimateCompletenessExtra:
    def test_range(self):
        comp = estimate_completeness(_gray())
        assert 0.0 <= comp <= 1.0

    def test_all_white(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        assert estimate_completeness(img) < 0.01

    def test_all_dark(self):
        img = np.full((50, 50), 0, dtype=np.uint8)
        assert estimate_completeness(img) > 0.99


# ─── assess_quality ───────────────────────────────────────────────────────────

class TestAssessQualityExtra:
    def test_returns_report(self):
        r = assess_quality(_gray())
        assert isinstance(r, QualityReport)

    def test_params_populated(self):
        r = assess_quality(_gray())
        assert "min_score" in r.params

    def test_custom_weights(self):
        r = assess_quality(_gray(), weights=(1.0, 0.0, 0.0, 0.0))
        assert isinstance(r, QualityReport)

    def test_high_min_score_rejects(self):
        r = assess_quality(_gray(), min_score=1.0)
        assert r.is_acceptable is False

    def test_bgr_input(self):
        r = assess_quality(_bgr())
        assert isinstance(r, QualityReport)


# ─── filter_by_quality ────────────────────────────────────────────────────────

class TestFilterByQualityExtra:
    def test_empty(self):
        good, rejected = filter_by_quality([])
        assert good == [] and rejected == []

    def test_splits(self):
        imgs = [_gray(), _edge_image()]
        good, rejected = filter_by_quality(imgs, min_score=0.3)
        assert len(good) + len(rejected) == 2


# ─── batch_assess_quality ─────────────────────────────────────────────────────

class TestBatchAssessQualityExtra:
    def test_empty(self):
        assert batch_assess_quality([]) == []

    def test_length(self):
        results = batch_assess_quality([_gray(), _gray()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_assess_quality([_gray()])
        assert isinstance(results[0], QualityReport)
