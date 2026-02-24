"""Extra tests for puzzle_reconstruction/utils/gradient_graph_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.gradient_graph_records import (
    GradientFlowRecord,
    GraphMatchRecord,
    HistogramRecord,
    HomographyRecord,
    make_gradient_flow_record,
    make_homography_record,
)


# ─── GradientFlowRecord ───────────────────────────────────────────────────────

class TestGradientFlowRecordExtra:
    def test_stores_fragment_id(self):
        r = GradientFlowRecord(fragment_id=3, mean_magnitude=25.0,
                                std_magnitude=5.0, edge_density=0.4,
                                dominant_angle=45.0)
        assert r.fragment_id == 3

    def test_is_high_texture_true(self):
        r = GradientFlowRecord(fragment_id=0, mean_magnitude=25.0,
                                std_magnitude=3.0, edge_density=0.1,
                                dominant_angle=0.0)
        assert r.is_high_texture is True

    def test_is_high_texture_false(self):
        r = GradientFlowRecord(fragment_id=0, mean_magnitude=10.0,
                                std_magnitude=3.0, edge_density=0.1,
                                dominant_angle=0.0)
        assert r.is_high_texture is False

    def test_is_edge_rich_true(self):
        r = GradientFlowRecord(fragment_id=0, mean_magnitude=5.0,
                                std_magnitude=1.0, edge_density=0.5,
                                dominant_angle=0.0)
        assert r.is_edge_rich is True

    def test_is_edge_rich_false(self):
        r = GradientFlowRecord(fragment_id=0, mean_magnitude=5.0,
                                std_magnitude=1.0, edge_density=0.1,
                                dominant_angle=0.0)
        assert r.is_edge_rich is False


# ─── GraphMatchRecord ─────────────────────────────────────────────────────────

class TestGraphMatchRecordExtra:
    def test_stores_fids(self):
        r = GraphMatchRecord(fid_a=3, fid_b=7, edge_weight=0.5)
        assert r.fid_a == 3 and r.fid_b == 7

    def test_pair_key_ordered(self):
        r = GraphMatchRecord(fid_a=5, fid_b=2, edge_weight=0.3)
        assert r.pair_key == (2, 5)

    def test_pair_key_same_order(self):
        r = GraphMatchRecord(fid_a=1, fid_b=3, edge_weight=0.3)
        assert r.pair_key == (1, 3)

    def test_is_strong_edge_true(self):
        r = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.9)
        assert r.is_strong_edge is True

    def test_is_strong_edge_false(self):
        r = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.5)
        assert r.is_strong_edge is False

    def test_is_mst_top_true(self):
        r = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.5, mst_rank=1)
        assert r.is_mst_top is True

    def test_is_mst_top_false(self):
        r = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.5, mst_rank=2)
        assert r.is_mst_top is False


# ─── HistogramRecord ──────────────────────────────────────────────────────────

class TestHistogramRecordExtra:
    def test_stores_ids(self):
        r = HistogramRecord(id_a=2, id_b=5, chi_squared=0.5,
                             intersection=0.8, emd=0.2)
        assert r.id_a == 2 and r.id_b == 5

    def test_is_similar_true(self):
        r = HistogramRecord(id_a=0, id_b=1, chi_squared=0.1,
                             intersection=0.8, emd=0.1)
        assert r.is_similar is True

    def test_is_similar_false(self):
        r = HistogramRecord(id_a=0, id_b=1, chi_squared=0.5,
                             intersection=0.4, emd=0.5)
        assert r.is_similar is False

    def test_is_dissimilar_high_chi(self):
        r = HistogramRecord(id_a=0, id_b=1, chi_squared=2.0,
                             intersection=0.5, emd=0.5)
        assert r.is_dissimilar is True

    def test_is_dissimilar_low_intersection(self):
        r = HistogramRecord(id_a=0, id_b=1, chi_squared=0.3,
                             intersection=0.2, emd=0.5)
        assert r.is_dissimilar is True


# ─── HomographyRecord ─────────────────────────────────────────────────────────

class TestHomographyRecordExtra:
    def test_stores_ids(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                              reproj_err=1.0, is_valid=True)
        assert r.id_src == 0 and r.id_dst == 1

    def test_is_good_true(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                              reproj_err=1.0, is_valid=True)
        assert r.is_good is True

    def test_is_good_false_invalid(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                              reproj_err=1.0, is_valid=False)
        assert r.is_good is False

    def test_is_good_false_high_err(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                              reproj_err=3.0, is_valid=True)
        assert r.is_good is False

    def test_quality_score_valid(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=20,
                              reproj_err=0.0, is_valid=True)
        score = r.quality_score
        assert 0.0 < score <= 1.0

    def test_quality_score_invalid_zero(self):
        r = HomographyRecord(id_src=0, id_dst=1, n_inliers=0,
                              reproj_err=5.0, is_valid=False)
        assert r.quality_score == pytest.approx(0.0)


# ─── make_gradient_flow_record ────────────────────────────────────────────────

class TestMakeGradientFlowRecordGGExtra:
    def test_returns_record(self):
        r = make_gradient_flow_record(0, 10.0, 2.0, 0.3, 45.0)
        assert isinstance(r, GradientFlowRecord)

    def test_values_stored(self):
        r = make_gradient_flow_record(5, 15.0, 3.0, 0.4, 90.0, n_boundary_points=50)
        assert r.fragment_id == 5
        assert r.n_boundary_points == 50


# ─── make_homography_record ───────────────────────────────────────────────────

class TestMakeHomographyRecordExtra:
    def test_returns_record(self):
        r = make_homography_record(0, 1, 10, 1.5, True)
        assert isinstance(r, HomographyRecord)

    def test_values_stored(self):
        r = make_homography_record(2, 3, 5, 2.0, False, method="lmeds")
        assert r.id_src == 2 and r.n_inliers == 5
        assert r.method == "lmeds"
