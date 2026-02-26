"""Tests for puzzle_reconstruction.utils.image_pipeline_utils"""
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


# ─── FrequencyMatchRecord ─────────────────────────────────────────────────────

def test_frequency_match_record_valid():
    r = FrequencyMatchRecord(id_a=0, id_b=1, similarity=0.8)
    assert r.pair == (0, 1)
    assert r.is_similar is True


def test_frequency_match_record_invalid_id_a():
    with pytest.raises(ValueError):
        FrequencyMatchRecord(id_a=-1, id_b=1, similarity=0.5)


def test_frequency_match_record_invalid_id_b():
    with pytest.raises(ValueError):
        FrequencyMatchRecord(id_a=0, id_b=-1, similarity=0.5)


def test_frequency_match_record_invalid_similarity():
    with pytest.raises(ValueError):
        FrequencyMatchRecord(id_a=0, id_b=1, similarity=1.5)


def test_frequency_match_record_not_similar():
    r = FrequencyMatchRecord(id_a=0, id_b=1, similarity=0.3)
    assert r.is_similar is False


def test_frequency_match_record_boundary_similarity():
    r = FrequencyMatchRecord(id_a=0, id_b=2, similarity=0.5)
    assert r.is_similar is True


# ─── FrequencyMatchSummary ───────────────────────────────────────────────────

def test_frequency_match_summary_similar_ratio():
    s = FrequencyMatchSummary(total_pairs=10, similar_pairs=4)
    assert abs(s.similar_ratio - 0.4) < 1e-9


def test_frequency_match_summary_zero_total():
    s = FrequencyMatchSummary(total_pairs=0, similar_pairs=0)
    assert s.similar_ratio == 0.0


def test_frequency_match_summary_invalid_total():
    with pytest.raises(ValueError):
        FrequencyMatchSummary(total_pairs=-1)


def test_frequency_match_summary_invalid_similar():
    with pytest.raises(ValueError):
        FrequencyMatchSummary(total_pairs=5, similar_pairs=-1)


# ─── summarize_frequency_matches ──────────────────────────────────────────────

def test_summarize_frequency_matches_empty():
    s = summarize_frequency_matches([])
    assert s.total_pairs == 0
    assert s.similar_pairs == 0


def test_summarize_frequency_matches_basic():
    records = [
        FrequencyMatchRecord(0, 1, 0.9),
        FrequencyMatchRecord(1, 2, 0.3),
        FrequencyMatchRecord(2, 3, 0.7),
    ]
    s = summarize_frequency_matches(records)
    assert s.total_pairs == 3
    assert s.similar_pairs == 2
    assert abs(s.mean_similarity - (0.9 + 0.3 + 0.7) / 3) < 1e-9
    assert s.max_similarity == 0.9
    assert s.min_similarity == 0.3


# ─── filter_frequency_matches ─────────────────────────────────────────────────

def test_filter_frequency_matches_basic():
    records = [
        FrequencyMatchRecord(0, 1, 0.9),
        FrequencyMatchRecord(1, 2, 0.3),
    ]
    filtered = filter_frequency_matches(records, min_similarity=0.5)
    assert len(filtered) == 1
    assert filtered[0].similarity == 0.9


def test_filter_frequency_matches_invalid():
    with pytest.raises(ValueError):
        filter_frequency_matches([], min_similarity=1.5)


def test_filter_frequency_matches_zero_threshold():
    records = [FrequencyMatchRecord(0, 1, 0.0), FrequencyMatchRecord(1, 2, 0.5)]
    filtered = filter_frequency_matches(records, min_similarity=0.0)
    assert len(filtered) == 2


# ─── PatchMatchRecord ─────────────────────────────────────────────────────────

def test_patch_match_record_displacement():
    r = PatchMatchRecord(src_row=5, src_col=3, dst_row=8, dst_col=7, score=0.9)
    assert r.displacement == (3, 4)


def test_patch_match_record_invalid_src_row():
    with pytest.raises(ValueError):
        PatchMatchRecord(src_row=-1, src_col=0, dst_row=0, dst_col=0, score=0.5)


def test_patch_match_record_zero_displacement():
    r = PatchMatchRecord(src_row=2, src_col=2, dst_row=2, dst_col=2, score=0.5)
    assert r.displacement == (0, 0)


# ─── CanvasBuildRecord ────────────────────────────────────────────────────────

def test_canvas_build_record_canvas_area():
    r = CanvasBuildRecord(n_fragments=5, coverage=0.8, canvas_w=100, canvas_h=200)
    assert r.canvas_area == 20000


def test_canvas_build_record_is_well_covered():
    r = CanvasBuildRecord(n_fragments=3, coverage=0.75, canvas_w=50, canvas_h=50)
    assert r.is_well_covered is True


def test_canvas_build_record_not_well_covered():
    r = CanvasBuildRecord(n_fragments=3, coverage=0.5, canvas_w=50, canvas_h=50)
    assert r.is_well_covered is False


def test_canvas_build_record_invalid_coverage():
    with pytest.raises(ValueError):
        CanvasBuildRecord(n_fragments=1, coverage=1.5, canvas_w=10, canvas_h=10)


def test_canvas_build_record_invalid_canvas_w():
    with pytest.raises(ValueError):
        CanvasBuildRecord(n_fragments=1, coverage=0.5, canvas_w=0, canvas_h=10)


# ─── summarize_canvas_builds ──────────────────────────────────────────────────

def test_summarize_canvas_builds_empty():
    s = summarize_canvas_builds([])
    assert s.n_canvases == 0
    assert s.well_covered_ratio == 0.0


def test_summarize_canvas_builds_basic():
    records = [
        CanvasBuildRecord(n_fragments=2, coverage=0.8, canvas_w=100, canvas_h=100),
        CanvasBuildRecord(n_fragments=3, coverage=0.6, canvas_w=100, canvas_h=100),
    ]
    s = summarize_canvas_builds(records)
    assert s.n_canvases == 2
    assert abs(s.mean_coverage - 0.7) < 1e-9
    assert s.well_covered_count == 1
    assert s.total_fragments == 5


# ─── summarize_patch_matches ──────────────────────────────────────────────────

def test_summarize_patch_matches_empty():
    s = summarize_patch_matches([])
    assert s.n_pairs == 0
    assert s.n_total_matches == 0


def test_summarize_patch_matches_basic():
    batch = [
        [PatchMatchRecord(0, 0, 1, 1, 0.5), PatchMatchRecord(1, 1, 2, 2, 0.7)],
        [PatchMatchRecord(2, 2, 3, 3, 0.8)],
    ]
    s = summarize_patch_matches(batch)
    assert s.n_pairs == 2
    assert s.n_total_matches == 3
    assert abs(s.mean_matches_per_pair - 1.5) < 1e-9


# ─── top_frequency_matches ────────────────────────────────────────────────────

def test_top_frequency_matches_basic():
    records = [
        FrequencyMatchRecord(0, 1, 0.3),
        FrequencyMatchRecord(1, 2, 0.9),
        FrequencyMatchRecord(2, 3, 0.6),
    ]
    top = top_frequency_matches(records, k=2)
    assert len(top) == 2
    assert top[0].similarity == 0.9


def test_top_frequency_matches_k_zero():
    records = [FrequencyMatchRecord(0, 1, 0.5)]
    top = top_frequency_matches(records, k=0)
    assert top == []


def test_top_frequency_matches_k_negative():
    with pytest.raises(ValueError):
        top_frequency_matches([], k=-1)
