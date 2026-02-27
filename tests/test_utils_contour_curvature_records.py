"""Tests for puzzle_reconstruction.utils.contour_curvature_records."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.contour_curvature_records import (
    AnnealingRunRecord,
    BlendOpRecord,
    ContourSampleRecord,
    CurvatureAnalysisRecord,
    BatchCurvatureRecord,
    ContourNormRecord,
    make_annealing_run_record,
    make_contour_sample_record,
    make_curvature_analysis_record,
)

np.random.seed(99)


# ── 1. AnnealingRunRecord basic ──────────────────────────────────────────────
def test_annealing_run_record_basic():
    r = AnnealingRunRecord(
        kind="geometric", n_steps=100, t_start=1.0, t_end=0.01,
        n_temperatures=10, min_temp=0.01, max_temp=1.0,
    )
    assert r.kind == "geometric"
    assert r.n_steps == 100
    assert r.n_temperatures == 10


# ── 2. make_annealing_run_record with temperatures ───────────────────────────
def test_make_annealing_run_record():
    temps = [1.0, 0.8, 0.6, 0.4, 0.2]
    r = make_annealing_run_record("linear", 50, 1.0, 0.2, temps)
    assert r.n_temperatures == 5
    assert r.min_temp == 0.2
    assert r.max_temp == 1.0


# ── 3. make_annealing_run_record empty temperatures ──────────────────────────
def test_make_annealing_run_record_empty():
    r = make_annealing_run_record("linear", 50, 1.0, 0.0, [])
    assert r.n_temperatures == 0
    assert r.min_temp == 0.0
    assert r.max_temp == 0.0


# ── 4. BlendOpRecord basic ───────────────────────────────────────────────────
def test_blend_op_record():
    r = BlendOpRecord(
        blend_type="alpha", alpha=0.7,
        src_shape=(256, 256, 3),
        dst_shape=(256, 256, 3),
        output_shape=(256, 256, 3),
    )
    assert r.blend_type == "alpha"
    assert r.alpha == 0.7
    assert r.src_shape == (256, 256, 3)


# ── 5. ContourSampleRecord basic ─────────────────────────────────────────────
def test_contour_sample_record():
    r = ContourSampleRecord(
        strategy="uniform", n_source=200, n_sampled=100,
        closed=True, total_arc_length=150.0,
    )
    assert r.strategy == "uniform"
    assert r.n_source == 200
    assert r.n_sampled == 100
    assert r.closed is True
    assert r.total_arc_length == 150.0


# ── 6. make_contour_sample_record ────────────────────────────────────────────
def test_make_contour_sample_record():
    r = make_contour_sample_record("arc_length", 150, 75, True, 200.0)
    assert r.strategy == "arc_length"
    assert r.n_source == 150
    assert r.n_sampled == 75
    assert r.closed is True
    assert r.total_arc_length == 200.0


def test_make_contour_sample_record_defaults():
    r = make_contour_sample_record("uniform", 100, 50)
    assert r.closed is False
    assert r.total_arc_length == 0.0


# ── 8. CurvatureAnalysisRecord basic ─────────────────────────────────────────
def test_curvature_analysis_record():
    r = CurvatureAnalysisRecord(
        n_points=100, smooth_sigma=1.5, total_curvature=6.28,
        turning_angle=3.14, n_corners=4, n_inflections=2,
        corner_threshold=0.1, min_distance=3,
    )
    assert r.n_points == 100
    assert r.n_corners == 4
    assert r.n_inflections == 2


# ── 9. make_curvature_analysis_record ────────────────────────────────────────
def test_make_curvature_record():
    r = make_curvature_analysis_record(80, smooth_sigma=2.0, total_curvature=5.0,
                                        turning_angle=2.5, n_corners=3,
                                        n_inflections=1)
    assert r.n_points == 80
    assert r.smooth_sigma == 2.0
    assert r.n_corners == 3


def test_make_curvature_record_defaults():
    r = make_curvature_analysis_record(50)
    assert r.smooth_sigma == 1.0
    assert r.total_curvature == 0.0
    assert r.n_corners == 0
    assert r.n_inflections == 0


# ── 11. BatchCurvatureRecord mean_total_curvature ────────────────────────────
def test_batch_curvature_mean():
    recs = [
        make_curvature_analysis_record(100, total_curvature=6.0),
        make_curvature_analysis_record(100, total_curvature=4.0),
        make_curvature_analysis_record(100, total_curvature=2.0),
    ]
    batch = BatchCurvatureRecord(n_curves=3, records=recs)
    assert abs(batch.mean_total_curvature - 4.0) < 1e-9


# ── 12. BatchCurvatureRecord mean_n_corners ──────────────────────────────────
def test_batch_curvature_mean_corners():
    recs = [
        make_curvature_analysis_record(100, n_corners=2),
        make_curvature_analysis_record(100, n_corners=4),
        make_curvature_analysis_record(100, n_corners=6),
    ]
    batch = BatchCurvatureRecord(n_curves=3, records=recs)
    assert abs(batch.mean_n_corners - 4.0) < 1e-9


# ── 13. BatchCurvatureRecord empty records ───────────────────────────────────
def test_batch_curvature_empty():
    batch = BatchCurvatureRecord(n_curves=0)
    assert batch.mean_total_curvature == 0.0
    assert batch.mean_n_corners == 0.0


# ── 14. ContourNormRecord basic ──────────────────────────────────────────────
def test_contour_norm_record():
    r = ContourNormRecord(
        n_points=64, original_scale=100.0,
        original_centroid_x=50.0, original_centroid_y=60.0,
    )
    assert r.n_points == 64
    assert r.original_scale == 100.0
    assert r.original_centroid_x == 50.0


# ── 15. multiple run records with different kinds ────────────────────────────
def test_multiple_run_records():
    kinds = ["geometric", "linear", "exponential"]
    records = [
        make_annealing_run_record(k, 100, 1.0, 0.01,
                                   [1.0 - j * 0.1 for j in range(10)])
        for k in kinds
    ]
    assert len(records) == 3
    assert all(r.n_temperatures == 10 for r in records)
    assert records[0].kind == "geometric"
