"""Extra tests for puzzle_reconstruction/utils/contour_curvature_records.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ann_rec(kind="sa", n_steps=100, t_start=1.0, t_end=0.01,
             n_temps=10, mn=0.01, mx=1.0) -> AnnealingRunRecord:
    return AnnealingRunRecord(
        kind=kind, n_steps=n_steps, t_start=t_start, t_end=t_end,
        n_temperatures=n_temps, min_temp=mn, max_temp=mx,
    )


def _csample_rec(strategy="uniform", n_src=100, n_samp=64, closed=True,
                 arc=50.0) -> ContourSampleRecord:
    return ContourSampleRecord(
        strategy=strategy, n_source=n_src, n_sampled=n_samp,
        closed=closed, total_arc_length=arc,
    )


def _curv_rec(n_pts=50, sigma=1.0, total_curv=3.14, turning=6.28,
              corners=4, inflect=2, thresh=0.1, min_dist=3) -> CurvatureAnalysisRecord:
    return CurvatureAnalysisRecord(
        n_points=n_pts, smooth_sigma=sigma,
        total_curvature=total_curv, turning_angle=turning,
        n_corners=corners, n_inflections=inflect,
        corner_threshold=thresh, min_distance=min_dist,
    )


# ─── AnnealingRunRecord ───────────────────────────────────────────────────────

class TestAnnealingRunRecordExtra:
    def test_stores_kind(self):
        assert _ann_rec(kind="simulated").kind == "simulated"

    def test_stores_n_steps(self):
        assert _ann_rec(n_steps=500).n_steps == 500

    def test_stores_t_start(self):
        assert _ann_rec(t_start=10.0).t_start == pytest.approx(10.0)

    def test_stores_t_end(self):
        assert _ann_rec(t_end=0.001).t_end == pytest.approx(0.001)

    def test_stores_n_temperatures(self):
        assert _ann_rec(n_temps=20).n_temperatures == 20

    def test_stores_min_temp(self):
        assert _ann_rec(mn=0.05).min_temp == pytest.approx(0.05)

    def test_stores_max_temp(self):
        assert _ann_rec(mx=5.0).max_temp == pytest.approx(5.0)


# ─── BlendOpRecord ────────────────────────────────────────────────────────────

class TestBlendOpRecordExtra:
    def test_stores_blend_type(self):
        r = BlendOpRecord(blend_type="alpha", alpha=0.5,
                          src_shape=(64, 64, 3), dst_shape=(64, 64, 3),
                          output_shape=(64, 64, 3))
        assert r.blend_type == "alpha"

    def test_stores_alpha(self):
        r = BlendOpRecord(blend_type="alpha", alpha=0.3,
                          src_shape=(4, 4), dst_shape=(4, 4), output_shape=(4, 4))
        assert r.alpha == pytest.approx(0.3)

    def test_stores_shapes(self):
        r = BlendOpRecord(blend_type="add", alpha=1.0,
                          src_shape=(8, 8), dst_shape=(8, 8), output_shape=(8, 8))
        assert r.src_shape == (8, 8)


# ─── ContourSampleRecord ──────────────────────────────────────────────────────

class TestContourSampleRecordExtra:
    def test_stores_strategy(self):
        assert _csample_rec(strategy="arc").strategy == "arc"

    def test_stores_n_source(self):
        assert _csample_rec(n_src=200).n_source == 200

    def test_stores_n_sampled(self):
        assert _csample_rec(n_samp=128).n_sampled == 128

    def test_stores_closed(self):
        assert _csample_rec(closed=False).closed is False

    def test_stores_arc_length(self):
        assert _csample_rec(arc=123.4).total_arc_length == pytest.approx(123.4)


# ─── CurvatureAnalysisRecord ──────────────────────────────────────────────────

class TestCurvatureAnalysisRecordExtra:
    def test_stores_n_points(self):
        assert _curv_rec(n_pts=80).n_points == 80

    def test_stores_smooth_sigma(self):
        assert _curv_rec(sigma=2.0).smooth_sigma == pytest.approx(2.0)

    def test_stores_total_curvature(self):
        assert _curv_rec(total_curv=6.28).total_curvature == pytest.approx(6.28)

    def test_stores_n_corners(self):
        assert _curv_rec(corners=4).n_corners == 4

    def test_stores_n_inflections(self):
        assert _curv_rec(inflect=3).n_inflections == 3

    def test_stores_corner_threshold(self):
        assert _curv_rec(thresh=0.2).corner_threshold == pytest.approx(0.2)

    def test_stores_min_distance(self):
        assert _curv_rec(min_dist=5).min_distance == 5


# ─── BatchCurvatureRecord ─────────────────────────────────────────────────────

class TestBatchCurvatureRecordExtra:
    def test_stores_n_curves(self):
        r = BatchCurvatureRecord(n_curves=4)
        assert r.n_curves == 4

    def test_default_records_empty(self):
        r = BatchCurvatureRecord(n_curves=2)
        assert r.records == []

    def test_mean_total_curvature_empty(self):
        r = BatchCurvatureRecord(n_curves=0)
        assert r.mean_total_curvature == pytest.approx(0.0)

    def test_mean_total_curvature_with_records(self):
        r1 = _curv_rec(total_curv=4.0)
        r2 = _curv_rec(total_curv=6.0)
        batch = BatchCurvatureRecord(n_curves=2, records=[r1, r2])
        assert batch.mean_total_curvature == pytest.approx(5.0)

    def test_mean_n_corners_empty(self):
        assert BatchCurvatureRecord(n_curves=0).mean_n_corners == pytest.approx(0.0)

    def test_mean_n_corners(self):
        r1 = _curv_rec(corners=2)
        r2 = _curv_rec(corners=4)
        batch = BatchCurvatureRecord(n_curves=2, records=[r1, r2])
        assert batch.mean_n_corners == pytest.approx(3.0)


# ─── ContourNormRecord ────────────────────────────────────────────────────────

class TestContourNormRecordExtra:
    def test_stores_n_points(self):
        r = ContourNormRecord(n_points=64, original_scale=2.0,
                              original_centroid_x=10.0, original_centroid_y=20.0)
        assert r.n_points == 64

    def test_stores_scale(self):
        r = ContourNormRecord(n_points=32, original_scale=1.5,
                              original_centroid_x=0.0, original_centroid_y=0.0)
        assert r.original_scale == pytest.approx(1.5)

    def test_stores_centroid(self):
        r = ContourNormRecord(n_points=32, original_scale=1.0,
                              original_centroid_x=5.0, original_centroid_y=7.0)
        assert r.original_centroid_x == pytest.approx(5.0)
        assert r.original_centroid_y == pytest.approx(7.0)


# ─── make_annealing_run_record ────────────────────────────────────────────────

class TestMakeAnnealingRunRecordExtra:
    def test_returns_record(self):
        r = make_annealing_run_record("sa", 100, 1.0, 0.01, [1.0, 0.5, 0.1])
        assert isinstance(r, AnnealingRunRecord)

    def test_n_temperatures_from_list(self):
        r = make_annealing_run_record("sa", 50, 1.0, 0.01, [1.0, 0.5])
        assert r.n_temperatures == 2

    def test_empty_temps(self):
        r = make_annealing_run_record("sa", 100, 1.0, 0.01, [])
        assert r.n_temperatures == 0 and r.min_temp == pytest.approx(0.0)

    def test_min_max_temps(self):
        r = make_annealing_run_record("sa", 100, 1.0, 0.01, [0.5, 1.0, 0.2])
        assert r.min_temp == pytest.approx(0.2)
        assert r.max_temp == pytest.approx(1.0)


# ─── make_contour_sample_record ───────────────────────────────────────────────

class TestMakeContourSampleRecordExtra:
    def test_returns_record(self):
        r = make_contour_sample_record("uniform", 100, 64)
        assert isinstance(r, ContourSampleRecord)

    def test_values_stored(self):
        r = make_contour_sample_record("arc", 200, 128, closed=True,
                                       total_arc_length=75.0)
        assert r.strategy == "arc"
        assert r.n_source == 200
        assert r.n_sampled == 128
        assert r.closed is True
        assert r.total_arc_length == pytest.approx(75.0)


# ─── make_curvature_analysis_record ───────────────────────────────────────────

class TestMakeCurvatureAnalysisRecordExtra:
    def test_returns_record(self):
        r = make_curvature_analysis_record(50)
        assert isinstance(r, CurvatureAnalysisRecord)

    def test_defaults(self):
        r = make_curvature_analysis_record(50)
        assert r.n_corners == 0
        assert r.smooth_sigma == pytest.approx(1.0)

    def test_custom_values(self):
        r = make_curvature_analysis_record(
            80, smooth_sigma=2.0, total_curvature=6.28,
            n_corners=4, n_inflections=2,
        )
        assert r.n_points == 80
        assert r.n_corners == 4
