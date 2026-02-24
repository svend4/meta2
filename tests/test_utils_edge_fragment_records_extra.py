"""Extra tests for puzzle_reconstruction/utils/edge_fragment_records.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ecr(a=0, b=1, score=0.8) -> EdgeCompareRecord:
    return EdgeCompareRecord(
        edge_id_a=a, edge_id_b=b, score=score,
        dtw_dist=0.1, css_sim=0.9, fd_diff=0.05, ifs_sim=0.85,
    )


def _fcr(ftype="corner", conf=0.9, has_text=False) -> FragmentClassifyRecord:
    return FragmentClassifyRecord(
        fragment_type=ftype, confidence=conf, has_text=has_text,
        text_lines=0, n_straight_sides=2, texture_variance=0.1, fill_ratio=0.9,
    )


def _fqr(fid=0, score=0.8) -> FragmentQualityRecord:
    return FragmentQualityRecord(
        fragment_id=fid, score=score, blur=0.1, contrast=0.7,
        coverage=0.9, sharpness=0.8, is_usable=True,
    )


def _gfr(h=64, w=64) -> GradientFlowRecord:
    return GradientFlowRecord(
        height=h, width=w, mean_magnitude=0.5, std_magnitude=0.2,
        edge_density=0.3, dominant_angle=45.0, ksize=3, normalized=False,
    )


# ─── EdgeCompareRecord ────────────────────────────────────────────────────────

class TestEdgeCompareRecordExtra:
    def test_stores_edge_ids(self):
        r = _ecr(a=2, b=5)
        assert r.edge_id_a == 2 and r.edge_id_b == 5

    def test_stores_score(self):
        assert _ecr(score=0.75).score == pytest.approx(0.75)

    def test_pair_key_ordered(self):
        r = _ecr(a=5, b=2)
        key = r.pair_key
        assert key == (2, 5)

    def test_pair_key_same_order(self):
        r = _ecr(a=1, b=3)
        assert r.pair_key == (1, 3)

    def test_stores_dtw_dist(self):
        r = EdgeCompareRecord(edge_id_a=0, edge_id_b=1, score=0.5,
                               dtw_dist=0.3, css_sim=0.7, fd_diff=0.1, ifs_sim=0.8)
        assert r.dtw_dist == pytest.approx(0.3)


# ─── CompatMatrixRecord ───────────────────────────────────────────────────────

class TestCompatMatrixRecordExtra:
    def test_stores_n_edges(self):
        r = CompatMatrixRecord(n_edges=4, min_score=0.1, max_score=0.9, mean_score=0.5)
        assert r.n_edges == 4

    def test_stores_scores(self):
        r = CompatMatrixRecord(n_edges=4, min_score=0.2, max_score=0.8, mean_score=0.5)
        assert r.min_score == pytest.approx(0.2)
        assert r.max_score == pytest.approx(0.8)
        assert r.mean_score == pytest.approx(0.5)


# ─── FragmentClassifyRecord (in edge_fragment_records) ────────────────────────

class TestEdgeFragmentClassifyRecordExtra:
    def test_stores_type(self):
        assert _fcr(ftype="edge").fragment_type == "edge"

    def test_stores_confidence(self):
        assert _fcr(conf=0.85).confidence == pytest.approx(0.85)

    def test_stores_has_text(self):
        assert _fcr(has_text=True).has_text is True

    def test_stores_text_lines(self):
        r = FragmentClassifyRecord(
            fragment_type="inner", confidence=0.6, has_text=True,
            text_lines=3, n_straight_sides=0, texture_variance=0.5, fill_ratio=0.8,
        )
        assert r.text_lines == 3

    def test_stores_fill_ratio(self):
        assert _fcr().fill_ratio == pytest.approx(0.9)


# ─── BatchClassifyRecord ──────────────────────────────────────────────────────

class TestBatchClassifyRecordExtra:
    def test_stores_n_fragments(self):
        r = BatchClassifyRecord(n_fragments=5)
        assert r.n_fragments == 5

    def test_default_records_empty(self):
        r = BatchClassifyRecord(n_fragments=3)
        assert r.records == []

    def test_n_text_fragments(self):
        recs = [_fcr(has_text=True), _fcr(has_text=False), _fcr(has_text=True)]
        r = BatchClassifyRecord(n_fragments=3, records=recs)
        assert r.n_text_fragments == 2

    def test_type_counts(self):
        recs = [_fcr(ftype="corner"), _fcr(ftype="edge"), _fcr(ftype="corner")]
        r = BatchClassifyRecord(n_fragments=3, records=recs)
        counts = r.type_counts
        assert counts.get("corner") == 2
        assert counts.get("edge") == 1

    def test_n_text_fragments_empty(self):
        assert BatchClassifyRecord(n_fragments=0).n_text_fragments == 0


# ─── FragmentQualityRecord ────────────────────────────────────────────────────

class TestFragmentQualityRecordExtra:
    def test_stores_fragment_id(self):
        assert _fqr(fid=7).fragment_id == 7

    def test_stores_score(self):
        assert _fqr(score=0.6).score == pytest.approx(0.6)

    def test_stores_is_usable(self):
        r = FragmentQualityRecord(fragment_id=0, score=0.3, blur=0.5,
                                   contrast=0.3, coverage=0.4, sharpness=0.3,
                                   is_usable=False)
        assert r.is_usable is False

    def test_stores_blur(self):
        assert _fqr().blur == pytest.approx(0.1)


# ─── GradientFlowRecord ───────────────────────────────────────────────────────

class TestGradientFlowRecordExtra:
    def test_stores_dimensions(self):
        r = _gfr(h=32, w=48)
        assert r.height == 32 and r.width == 48

    def test_stores_mean_magnitude(self):
        assert _gfr().mean_magnitude == pytest.approx(0.5)

    def test_stores_edge_density(self):
        assert _gfr().edge_density == pytest.approx(0.3)

    def test_stores_dominant_angle(self):
        assert _gfr().dominant_angle == pytest.approx(45.0)


# ─── make_edge_compare_record ─────────────────────────────────────────────────

class TestMakeEdgeCompareRecordExtra:
    def test_returns_record(self):
        r = make_edge_compare_record(0, 1, 0.7)
        assert isinstance(r, EdgeCompareRecord)

    def test_defaults_zero(self):
        r = make_edge_compare_record(0, 1, 0.7)
        assert r.dtw_dist == pytest.approx(0.0)

    def test_custom_values(self):
        r = make_edge_compare_record(2, 5, 0.9, dtw_dist=0.1, css_sim=0.8)
        assert r.edge_id_a == 2 and r.score == pytest.approx(0.9)


# ─── make_fragment_classify_record ────────────────────────────────────────────

class TestMakeEdgeFragmentClassifyRecordExtra:
    def test_returns_record(self):
        r = make_fragment_classify_record("corner", 0.9)
        assert isinstance(r, FragmentClassifyRecord)

    def test_default_has_text_false(self):
        assert make_fragment_classify_record("corner", 0.9).has_text is False

    def test_custom_values(self):
        r = make_fragment_classify_record("edge", 0.7, has_text=True, text_lines=2)
        assert r.has_text is True and r.text_lines == 2


# ─── make_gradient_flow_record ────────────────────────────────────────────────

class TestMakeGradientFlowRecordExtra:
    def test_returns_record(self):
        r = make_gradient_flow_record(64, 64)
        assert isinstance(r, GradientFlowRecord)

    def test_defaults(self):
        r = make_gradient_flow_record(32, 32)
        assert r.ksize == 3 and r.normalized is False

    def test_custom_values(self):
        r = make_gradient_flow_record(48, 96, mean_magnitude=0.3, ksize=5, normalized=True)
        assert r.height == 48 and r.width == 96
        assert r.mean_magnitude == pytest.approx(0.3)
        assert r.normalized is True
