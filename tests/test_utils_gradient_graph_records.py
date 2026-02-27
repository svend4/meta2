"""Tests for puzzle_reconstruction.utils.gradient_graph_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.gradient_graph_records import (
    GradientFlowRecord,
    GraphMatchRecord,
    HistogramRecord,
    HomographyRecord,
    make_gradient_flow_record,
    make_homography_record,
)

np.random.seed(42)


# ── GradientFlowRecord ────────────────────────────────────────────────────────

def test_gradient_flow_is_high_texture_true():
    rec = GradientFlowRecord(fragment_id=0, mean_magnitude=25.0,
                             std_magnitude=5.0, edge_density=0.2,
                             dominant_angle=45.0)
    assert rec.is_high_texture is True


def test_gradient_flow_is_high_texture_false():
    rec = GradientFlowRecord(fragment_id=0, mean_magnitude=15.0,
                             std_magnitude=3.0, edge_density=0.1,
                             dominant_angle=0.0)
    assert rec.is_high_texture is False


def test_gradient_flow_is_edge_rich_true():
    rec = GradientFlowRecord(fragment_id=1, mean_magnitude=10.0,
                             std_magnitude=2.0, edge_density=0.4,
                             dominant_angle=90.0)
    assert rec.is_edge_rich is True


def test_gradient_flow_is_edge_rich_false():
    rec = GradientFlowRecord(fragment_id=1, mean_magnitude=10.0,
                             std_magnitude=2.0, edge_density=0.2,
                             dominant_angle=90.0)
    assert rec.is_edge_rich is False


def test_gradient_flow_n_boundary_points_default():
    rec = GradientFlowRecord(fragment_id=2, mean_magnitude=5.0,
                             std_magnitude=1.0, edge_density=0.1,
                             dominant_angle=0.0)
    assert rec.n_boundary_points == 0


def test_make_gradient_flow_record():
    rec = make_gradient_flow_record(3, 30.0, 6.0, 0.5, 135.0, n_boundary_points=50)
    assert isinstance(rec, GradientFlowRecord)
    assert rec.fragment_id == 3
    assert rec.mean_magnitude == pytest.approx(30.0)
    assert rec.n_boundary_points == 50


# ── GraphMatchRecord ──────────────────────────────────────────────────────────

def test_graph_match_pair_key_sorted():
    rec = GraphMatchRecord(fid_a=5, fid_b=2, edge_weight=0.8)
    assert rec.pair_key == (2, 5)


def test_graph_match_pair_key_same_order():
    rec = GraphMatchRecord(fid_a=1, fid_b=3, edge_weight=0.5)
    assert rec.pair_key == (1, 3)


def test_graph_match_is_strong_edge_true():
    rec = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.9)
    assert rec.is_strong_edge is True


def test_graph_match_is_strong_edge_false():
    rec = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.5)
    assert rec.is_strong_edge is False


def test_graph_match_is_mst_top_true():
    rec = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.8, mst_rank=1)
    assert rec.is_mst_top is True


def test_graph_match_is_mst_top_false():
    rec = GraphMatchRecord(fid_a=0, fid_b=1, edge_weight=0.8, mst_rank=2)
    assert rec.is_mst_top is False


# ── HistogramRecord ───────────────────────────────────────────────────────────

def test_histogram_record_is_similar_true():
    rec = HistogramRecord(id_a=0, id_b=1, chi_squared=0.2,
                          intersection=0.8, emd=0.1)
    assert rec.is_similar is True


def test_histogram_record_is_similar_false():
    rec = HistogramRecord(id_a=0, id_b=1, chi_squared=0.5,
                          intersection=0.5, emd=0.5)
    assert rec.is_similar is False


def test_histogram_record_is_dissimilar_high_chi():
    rec = HistogramRecord(id_a=0, id_b=1, chi_squared=1.5,
                          intersection=0.4, emd=0.8)
    assert rec.is_dissimilar is True


def test_histogram_record_is_dissimilar_low_intersection():
    rec = HistogramRecord(id_a=0, id_b=1, chi_squared=0.5,
                          intersection=0.2, emd=0.3)
    assert rec.is_dissimilar is True


def test_histogram_record_default_n_bins():
    rec = HistogramRecord(id_a=0, id_b=1, chi_squared=0.1,
                          intersection=0.9, emd=0.05)
    assert rec.n_bins == 256


# ── HomographyRecord ──────────────────────────────────────────────────────────

def test_homography_record_is_good_true():
    rec = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                           reproj_err=1.0, is_valid=True)
    assert rec.is_good is True


def test_homography_record_is_good_false_high_err():
    rec = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                           reproj_err=3.0, is_valid=True)
    assert rec.is_good is False


def test_homography_record_is_good_false_not_valid():
    rec = HomographyRecord(id_src=0, id_dst=1, n_inliers=10,
                           reproj_err=1.0, is_valid=False)
    assert rec.is_good is False


def test_homography_record_quality_score_invalid():
    rec = HomographyRecord(id_src=0, id_dst=1, n_inliers=0,
                           reproj_err=0.0, is_valid=False)
    assert rec.quality_score == pytest.approx(0.0)


def test_homography_record_quality_score_positive():
    rec = HomographyRecord(id_src=0, id_dst=1, n_inliers=20,
                           reproj_err=1.0, is_valid=True)
    score = rec.quality_score
    assert 0.0 < score <= 1.0


def test_make_homography_record():
    rec = make_homography_record(0, 1, 15, 0.5, True, method="lmeds")
    assert isinstance(rec, HomographyRecord)
    assert rec.n_inliers == 15
    assert rec.method == "lmeds"
