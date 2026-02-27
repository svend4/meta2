"""Tests for puzzle_reconstruction.utils.edge_fragment_records."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.edge_fragment_records import (
    EdgeCompareRecord,
    CompatMatrixRecord,
    FragmentClassifyRecord,
    BatchClassifyRecord,
    FragmentQualityRecord,
    GradientFlowRecord,
    make_edge_compare_record,
    make_fragment_classify_record,
    make_gradient_flow_record,
)

np.random.seed(37)


# ── 1. EdgeCompareRecord basic ───────────────────────────────────────────────
def test_edge_compare_basic():
    r = EdgeCompareRecord(0, 1, 0.8, 1.2, 0.7, 0.3, 0.9)
    assert r.edge_id_a == 0
    assert r.edge_id_b == 1
    assert r.score == 0.8
    assert r.pair_key == (0, 1)


def test_edge_compare_pair_key_order():
    r = EdgeCompareRecord(5, 2, 0.6, 0.5, 0.4, 0.3, 0.8)
    assert r.pair_key == (2, 5)


# ── 3. make_edge_compare_record ──────────────────────────────────────────────
def test_make_edge_compare():
    r = make_edge_compare_record(3, 7, 0.75)
    assert r.edge_id_a == 3
    assert r.edge_id_b == 7
    assert r.score == 0.75
    assert r.dtw_dist == 0.0
    assert r.css_sim == 0.0
    assert r.fd_diff == 0.0
    assert r.ifs_sim == 0.0


def test_make_edge_compare_full():
    r = make_edge_compare_record(1, 2, 0.5, 0.3, 0.7, 0.2, 0.8)
    assert r.dtw_dist == 0.3
    assert r.css_sim == 0.7
    assert r.fd_diff == 0.2
    assert r.ifs_sim == 0.8


# ── 5. CompatMatrixRecord ────────────────────────────────────────────────────
def test_compat_matrix_basic():
    r = CompatMatrixRecord(n_edges=10, min_score=0.1, max_score=0.9, mean_score=0.5)
    assert r.n_edges == 10
    assert r.min_score == 0.1
    assert r.max_score == 0.9
    assert r.mean_score == 0.5


# ── 6. FragmentClassifyRecord basic ─────────────────────────────────────────
def test_fragment_classify_basic():
    r = FragmentClassifyRecord("corner", 0.9, False, 0, 2, 0.05, 0.95)
    assert r.fragment_type == "corner"
    assert r.confidence == 0.9
    assert r.n_straight_sides == 2
    assert r.has_text is False


# ── 7. make_fragment_classify_record ─────────────────────────────────────────
def test_make_classify_record():
    r = make_fragment_classify_record("edge", 0.85, has_text=True,
                                       text_lines=2, n_straight_sides=1,
                                       texture_variance=0.1, fill_ratio=0.9)
    assert r.fragment_type == "edge"
    assert r.confidence == 0.85
    assert r.has_text is True
    assert r.text_lines == 2


def test_make_classify_defaults():
    r = make_fragment_classify_record("inner", 0.7)
    assert r.has_text is False
    assert r.text_lines == 0
    assert r.n_straight_sides == 0
    assert r.texture_variance == 0.0
    assert r.fill_ratio == 1.0


# ── 9. BatchClassifyRecord ───────────────────────────────────────────────────
def test_batch_classify_basic():
    recs = [
        make_fragment_classify_record("corner", 0.9, has_text=False),
        make_fragment_classify_record("edge", 0.8, has_text=True),
        make_fragment_classify_record("inner", 0.7, has_text=True),
    ]
    batch = BatchClassifyRecord(n_fragments=3, records=recs)
    assert batch.n_text_fragments == 2


def test_batch_classify_type_counts():
    recs = [
        make_fragment_classify_record("corner", 0.9),
        make_fragment_classify_record("corner", 0.8),
        make_fragment_classify_record("edge", 0.7),
    ]
    batch = BatchClassifyRecord(n_fragments=3, records=recs)
    counts = batch.type_counts
    assert counts["corner"] == 2
    assert counts["edge"] == 1


def test_batch_classify_empty():
    batch = BatchClassifyRecord(n_fragments=0)
    assert batch.n_text_fragments == 0
    assert batch.type_counts == {}


# ── 12. FragmentQualityRecord ────────────────────────────────────────────────
def test_fragment_quality_basic():
    r = FragmentQualityRecord(
        fragment_id=5, score=0.75, blur=0.1, contrast=0.8,
        coverage=0.95, sharpness=0.9, is_usable=True,
    )
    assert r.fragment_id == 5
    assert r.score == 0.75
    assert r.is_usable is True


# ── 13. GradientFlowRecord basic ─────────────────────────────────────────────
def test_gradient_flow_basic():
    r = GradientFlowRecord(
        height=480, width=640, mean_magnitude=25.0, std_magnitude=5.0,
        edge_density=0.1, dominant_angle=45.0, ksize=3, normalized=False,
    )
    assert r.height == 480
    assert r.width == 640
    assert r.ksize == 3


# ── 14. make_gradient_flow_record ────────────────────────────────────────────
def test_make_gradient_flow():
    r = make_gradient_flow_record(256, 256, mean_magnitude=10.0,
                                   std_magnitude=2.0, edge_density=0.05,
                                   dominant_angle=90.0, ksize=5, normalized=True)
    assert r.height == 256
    assert r.width == 256
    assert r.mean_magnitude == 10.0
    assert r.ksize == 5
    assert r.normalized is True


def test_make_gradient_flow_defaults():
    r = make_gradient_flow_record(100, 100)
    assert r.mean_magnitude == 0.0
    assert r.std_magnitude == 0.0
    assert r.edge_density == 0.0
    assert r.ksize == 3
    assert r.normalized is False


# ── 16. BatchClassifyRecord n_fragments without records ─────────────────────
def test_batch_classify_n_fragments():
    batch = BatchClassifyRecord(n_fragments=5)
    assert batch.n_fragments == 5
    assert len(batch.records) == 0
