"""Extra tests for puzzle_reconstruction/utils/image_pipeline_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.image_pipeline_utils import (
    FrequencyMatchRecord,
    FrequencyMatchSummary,
    PatchMatchRecord,
    PatchMatchSummary,
    CanvasBuildRecord,
    CanvasBuildSummary,
    summarize_frequency_matches,
    filter_frequency_matches,
    summarize_canvas_builds,
    summarize_patch_matches,
    top_frequency_matches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fmr(a=0, b=1, sim=0.7) -> FrequencyMatchRecord:
    return FrequencyMatchRecord(id_a=a, id_b=b, similarity=sim)


def _fmrs(n=4) -> list:
    return [_fmr(a=i, b=i+1, sim=float(i+1)/(n+1)) for i in range(n)]


def _pmr(sr=0, sc=0, dr=2, dc=3, score=0.8) -> PatchMatchRecord:
    return PatchMatchRecord(src_row=sr, src_col=sc,
                             dst_row=dr, dst_col=dc, score=score)


def _cbr(n=5, cov=0.8, w=100, h=100) -> CanvasBuildRecord:
    return CanvasBuildRecord(n_fragments=n, coverage=cov,
                              canvas_w=w, canvas_h=h)


# ─── FrequencyMatchRecord ─────────────────────────────────────────────────────

class TestFrequencyMatchRecordExtra:
    def test_stores_ids(self):
        r = _fmr(a=3, b=5)
        assert r.id_a == 3 and r.id_b == 5

    def test_stores_similarity(self):
        assert _fmr(sim=0.85).similarity == pytest.approx(0.85)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FrequencyMatchRecord(id_a=-1, id_b=0, similarity=0.5)

    def test_similarity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FrequencyMatchRecord(id_a=0, id_b=1, similarity=1.5)

    def test_pair_property(self):
        r = _fmr(a=2, b=7)
        assert r.pair == (2, 7)

    def test_is_similar_true(self):
        assert _fmr(sim=0.7).is_similar is True

    def test_is_similar_false(self):
        assert _fmr(sim=0.3).is_similar is False


# ─── FrequencyMatchSummary ────────────────────────────────────────────────────

class TestFrequencyMatchSummaryExtra:
    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            FrequencyMatchSummary(total_pairs=-1)

    def test_similar_ratio(self):
        s = FrequencyMatchSummary(total_pairs=4, similar_pairs=2)
        assert s.similar_ratio == pytest.approx(0.5)

    def test_similar_ratio_zero_total(self):
        s = FrequencyMatchSummary(total_pairs=0, similar_pairs=0)
        assert s.similar_ratio == pytest.approx(0.0)


# ─── PatchMatchRecord ─────────────────────────────────────────────────────────

class TestPatchMatchRecordExtra:
    def test_stores_coords(self):
        r = _pmr(sr=1, sc=2, dr=5, dc=6)
        assert r.src_row == 1 and r.dst_col == 6

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            PatchMatchRecord(src_row=-1, src_col=0, dst_row=0, dst_col=0, score=0.5)

    def test_displacement(self):
        r = _pmr(sr=1, sc=2, dr=4, dc=5)
        assert r.displacement == (3, 3)


# ─── PatchMatchSummary ────────────────────────────────────────────────────────

class TestPatchMatchSummaryExtra:
    def test_negative_pairs_raises(self):
        with pytest.raises(ValueError):
            PatchMatchSummary(n_pairs=-1)

    def test_stores_method(self):
        s = PatchMatchSummary(n_pairs=3, method="ssd")
        assert s.method == "ssd"


# ─── CanvasBuildRecord ────────────────────────────────────────────────────────

class TestCanvasBuildRecordExtra:
    def test_stores_n_fragments(self):
        assert _cbr(n=7).n_fragments == 7

    def test_negative_fragments_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildRecord(n_fragments=-1, coverage=0.5, canvas_w=100, canvas_h=100)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildRecord(n_fragments=1, coverage=1.5, canvas_w=100, canvas_h=100)

    def test_canvas_area(self):
        assert _cbr(w=50, h=40).canvas_area == 2000

    def test_is_well_covered_true(self):
        assert _cbr(cov=0.8).is_well_covered is True

    def test_is_well_covered_false(self):
        assert _cbr(cov=0.5).is_well_covered is False


# ─── CanvasBuildSummary ───────────────────────────────────────────────────────

class TestCanvasBuildSummaryExtra:
    def test_negative_canvases_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildSummary(n_canvases=-1)

    def test_well_covered_ratio(self):
        s = CanvasBuildSummary(n_canvases=4, well_covered_count=2)
        assert s.well_covered_ratio == pytest.approx(0.5)

    def test_well_covered_ratio_zero_canvases(self):
        s = CanvasBuildSummary(n_canvases=0, well_covered_count=0)
        assert s.well_covered_ratio == pytest.approx(0.0)


# ─── summarize_frequency_matches ──────────────────────────────────────────────

class TestSummarizeFrequencyMatchesExtra:
    def test_returns_summary(self):
        assert isinstance(summarize_frequency_matches(_fmrs()), FrequencyMatchSummary)

    def test_total_correct(self):
        s = summarize_frequency_matches(_fmrs(3))
        assert s.total_pairs == 3

    def test_empty_returns_default(self):
        s = summarize_frequency_matches([])
        assert s.total_pairs == 0


# ─── filter_frequency_matches ─────────────────────────────────────────────────

class TestFilterFrequencyMatchesExtra:
    def test_filters_by_min_similarity(self):
        recs = [_fmr(sim=0.3), _fmr(b=2, sim=0.8)]
        result = filter_frequency_matches(recs, min_similarity=0.5)
        assert all(r.similarity >= 0.5 for r in result)

    def test_invalid_min_similarity_raises(self):
        with pytest.raises(ValueError):
            filter_frequency_matches(_fmrs(), min_similarity=1.5)

    def test_empty_input(self):
        assert filter_frequency_matches([]) == []


# ─── summarize_canvas_builds ──────────────────────────────────────────────────

class TestSummarizeCanvasBuildsExtra:
    def test_returns_summary(self):
        recs = [_cbr(cov=0.8), _cbr(n=3, cov=0.5)]
        assert isinstance(summarize_canvas_builds(recs), CanvasBuildSummary)

    def test_n_canvases_correct(self):
        recs = [_cbr(), _cbr()]
        s = summarize_canvas_builds(recs)
        assert s.n_canvases == 2

    def test_empty_returns_default(self):
        s = summarize_canvas_builds([])
        assert s.n_canvases == 0


# ─── summarize_patch_matches ──────────────────────────────────────────────────

class TestSummarizePatchMatchesExtra:
    def test_returns_summary(self):
        batch = [[_pmr(), _pmr()], [_pmr()]]
        assert isinstance(summarize_patch_matches(batch), PatchMatchSummary)

    def test_total_matches(self):
        batch = [[_pmr(), _pmr()], [_pmr()]]
        s = summarize_patch_matches(batch)
        assert s.n_total_matches == 3

    def test_empty_input(self):
        s = summarize_patch_matches([])
        assert s.n_pairs == 0


# ─── top_frequency_matches ────────────────────────────────────────────────────

class TestTopFrequencyMatchesExtra:
    def test_returns_top_k(self):
        recs = _fmrs(5)
        result = top_frequency_matches(recs, 3)
        assert len(result) == 3

    def test_sorted_descending(self):
        recs = _fmrs(4)
        result = top_frequency_matches(recs, 4)
        sims = [r.similarity for r in result]
        assert sims == sorted(sims, reverse=True)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            top_frequency_matches(_fmrs(), k=-1)

    def test_empty_input(self):
        assert top_frequency_matches([], k=3) == []
