"""Tests for puzzle_reconstruction.utils.contour_contrast_records."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.contour_contrast_records import (
    ContourProcessRecord,
    ContrastEnhanceRecord,
    CostMatrixRecord,
    ContourBatchRecord,
    make_contour_process_record,
    make_contrast_enhance_record,
)

np.random.seed(13)


# ── 1. ContourProcessRecord basic ────────────────────────────────────────────
def test_contour_process_basic():
    r = ContourProcessRecord(
        fragment_id=1, n_points_before=100, n_points_after=60,
        perimeter=250.0, area=5000.0, compactness=0.8,
    )
    assert r.fragment_id == 1
    assert r.n_points_before == 100
    assert r.n_points_after == 60
    assert abs(r.compression_ratio - 0.6) < 1e-9


# ── 2. ContourProcessRecord compression_ratio zero before ────────────────────
def test_compression_ratio_zero_before():
    r = ContourProcessRecord(0, 0, 0, 0.0, 0.0, 0.0)
    assert r.compression_ratio == 0.0


# ── 3. ContourProcessRecord defaults ─────────────────────────────────────────
def test_contour_process_defaults():
    r = ContourProcessRecord(5, 50, 40, 100.0, 800.0, 0.7)
    assert r.normalized is True
    assert r.simplified is False


# ── 4. make_contour_process_record ───────────────────────────────────────────
def test_make_contour_process_record():
    r = make_contour_process_record(
        fragment_id=10, n_points_before=200, n_points_after=100,
        perimeter=300.0, area=7000.0, compactness=0.9,
        normalized=False, simplified=True,
    )
    assert r.fragment_id == 10
    assert r.normalized is False
    assert r.simplified is True
    assert abs(r.compression_ratio - 0.5) < 1e-9


# ── 5. ContrastEnhanceRecord basic ───────────────────────────────────────────
def test_contrast_enhance_basic():
    r = ContrastEnhanceRecord(
        method="clahe", contrast_before=50.0, contrast_after=80.0,
        image_height=480, image_width=640, n_channels=3,
    )
    assert abs(r.improvement - 30.0) < 1e-9
    assert abs(r.improvement_ratio - 0.6) < 1e-9
    assert r.is_grayscale is False


# ── 6. ContrastEnhanceRecord grayscale ───────────────────────────────────────
def test_contrast_enhance_grayscale():
    r = ContrastEnhanceRecord("hist_eq", 30.0, 70.0, 256, 256, n_channels=1)
    assert r.is_grayscale is True


# ── 7. ContrastEnhanceRecord zero before ─────────────────────────────────────
def test_contrast_enhance_zero_before():
    r = ContrastEnhanceRecord("clahe", 0.0, 50.0, 256, 256)
    assert r.improvement_ratio == 0.0


# ── 8. make_contrast_enhance_record with 2D shape ─────────────────────────────
def test_make_contrast_enhance_2d():
    r = make_contrast_enhance_record("clahe", 40.0, 70.0, (480, 640))
    assert r.image_height == 480
    assert r.image_width == 640
    assert r.n_channels == 1


# ── 9. make_contrast_enhance_record with 3D shape ─────────────────────────────
def test_make_contrast_enhance_3d():
    r = make_contrast_enhance_record("clahe", 40.0, 70.0, (480, 640, 3))
    assert r.n_channels == 3


# ── 10. CostMatrixRecord basic ───────────────────────────────────────────────
def test_cost_matrix_basic():
    r = CostMatrixRecord(n_fragments=5, method="euclidean",
                          min_cost=0.1, max_cost=5.0, mean_cost=2.5)
    assert r.n_fragments == 5
    assert abs(r.cost_range - 4.9) < 1e-9


# ── 11. CostMatrixRecord normalized ──────────────────────────────────────────
def test_cost_matrix_normalized():
    r = CostMatrixRecord(5, "cosine", 0.0, 1.0, 0.5, normalized=True)
    assert r.normalized is True
    assert abs(r.cost_range - 1.0) < 1e-9


# ── 12. ContourBatchRecord success_rate ──────────────────────────────────────
def test_batch_record_success_rate():
    r = ContourBatchRecord(
        n_contours=10, n_points_config=50, smooth_sigma=1.0,
        rdp_epsilon=0.01, normalize=True, n_successful=8,
    )
    assert abs(r.success_rate - 0.8) < 1e-9


# ── 13. ContourBatchRecord zero contours ─────────────────────────────────────
def test_batch_record_zero_contours():
    r = ContourBatchRecord(0, 50, 1.0, 0.01, True)
    assert r.success_rate == 0.0


# ── 14. ContrastEnhanceRecord negative improvement ───────────────────────────
def test_contrast_negative_improvement():
    r = ContrastEnhanceRecord("clahe", 80.0, 60.0, 256, 256)
    assert r.improvement < 0.0
    assert r.improvement_ratio < 0.0


# ── 15. multiple contour records ─────────────────────────────────────────────
def test_multiple_contour_records():
    records = [
        make_contour_process_record(i, 100, 100 - i*5, 200.0, 4000.0, 0.8)
        for i in range(5)
    ]
    assert len(records) == 5
    assert records[4].n_points_after == 80


# ── 16. ContourBatchRecord default n_successful ──────────────────────────────
def test_batch_record_default_n_successful():
    r = ContourBatchRecord(5, 50, 1.0, 0.01, True)
    assert r.n_successful == 0
    assert r.success_rate == 0.0
