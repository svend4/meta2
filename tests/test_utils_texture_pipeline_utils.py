"""Tests for puzzle_reconstruction.utils.texture_pipeline_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.texture_pipeline_utils import (
    TextureMatchRecord,
    TextureMatchSummary,
    make_texture_match_record,
    summarise_texture_matches,
    filter_texture_by_score,
    filter_texture_by_lbp,
    top_k_texture_records,
    best_texture_record,
    texture_score_stats,
    BatchPipelineRecord,
    BatchPipelineSummary,
    make_batch_pipeline_record,
    summarise_batch_pipeline,
    filter_batch_by_success_rate,
    filter_batch_by_stage,
    top_k_batch_records,
    batch_throughput_stats,
    compare_batch_summaries,
)

np.random.seed(77)


def _make_texture_records(n=5):
    records = []
    for i in range(n):
        r = make_texture_match_record(
            pair=(i, i + 1),
            score=float(i) / n,
            lbp_score=float(i) / (n + 1),
            gabor_score=0.5,
            gradient_score=0.3,
            side1=0, side2=1,
        )
        records.append(r)
    return records


def _make_batch_records(n=4):
    stages = ["lbp", "gabor", "gradient"]
    records = []
    for i in range(n):
        r = make_batch_pipeline_record(
            batch_id=i,
            n_items=10,
            n_done=8 + i % 3,
            n_failed=2 - i % 3,
            n_skipped=0,
            elapsed=1.0 + i * 0.5,
            stage=stages[i % len(stages)],
        )
        records.append(r)
    return records


# ── TextureMatchRecord ────────────────────────────────────────────────────────

def test_texture_match_record_basic():
    r = make_texture_match_record((0, 1), 0.8, 0.7, 0.9, 0.6)
    assert r.pair == (0, 1)
    assert r.score == pytest.approx(0.8)
    assert r.lbp_score == pytest.approx(0.7)


def test_texture_match_record_defaults():
    r = make_texture_match_record((2, 3), 0.5, 0.5, 0.5, 0.5)
    assert r.side1 == 0
    assert r.side2 == 0
    assert r.params == {}


def test_texture_match_record_with_params():
    r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4, extra="val")
    assert r.params["extra"] == "val"


def test_texture_match_record_side_values():
    r = make_texture_match_record((0, 1), 0.5, 0.5, 0.5, 0.5, side1=2, side2=3)
    assert r.side1 == 2
    assert r.side2 == 3


# ── summarise_texture_matches ─────────────────────────────────────────────────

def test_summarise_texture_empty():
    s = summarise_texture_matches([])
    assert s.n_pairs == 0
    assert s.best_pair is None
    assert s.best_score == 0.0


def test_summarise_texture_records():
    records = _make_texture_records(5)
    s = summarise_texture_matches(records)
    assert s.n_pairs == 5
    assert s.best_pair is not None
    assert s.best_score == pytest.approx(max(r.score for r in records))


def test_summarise_texture_mean_scores():
    records = _make_texture_records(4)
    s = summarise_texture_matches(records)
    expected_mean = sum(r.score for r in records) / len(records)
    assert s.mean_score == pytest.approx(expected_mean)


# ── filter functions ──────────────────────────────────────────────────────────

def test_filter_texture_by_score():
    records = _make_texture_records(6)
    filtered = filter_texture_by_score(records, 0.4)
    assert all(r.score >= 0.4 for r in filtered)


def test_filter_texture_by_lbp():
    records = _make_texture_records(6)
    filtered = filter_texture_by_lbp(records, 0.3)
    assert all(r.lbp_score >= 0.3 for r in filtered)


def test_top_k_texture_records():
    records = _make_texture_records(5)
    top = top_k_texture_records(records, 3)
    assert len(top) == 3
    assert top[0].score >= top[-1].score


def test_best_texture_record_none():
    assert best_texture_record([]) is None


def test_best_texture_record():
    records = _make_texture_records(5)
    best = best_texture_record(records)
    assert best.score == max(r.score for r in records)


# ── texture_score_stats ───────────────────────────────────────────────────────

def test_texture_score_stats_empty():
    stats = texture_score_stats([])
    assert stats["count"] == 0


def test_texture_score_stats():
    records = _make_texture_records(4)
    stats = texture_score_stats(records)
    assert stats["count"] == 4
    assert stats["max"] >= stats["min"]
    assert stats["std"] >= 0.0


# ── BatchPipelineRecord ───────────────────────────────────────────────────────

def test_batch_pipeline_record_success_rate():
    r = make_batch_pipeline_record(0, 10, 8, 2, 0, 2.0, "lbp")
    assert r.success_rate == pytest.approx(0.8)


def test_batch_pipeline_record_success_rate_zero():
    r = make_batch_pipeline_record(0, 0, 0, 0, 0, 1.0, "stage")
    assert r.success_rate == 0.0


def test_batch_pipeline_record_throughput():
    r = make_batch_pipeline_record(0, 10, 8, 2, 0, 4.0, "gabor")
    assert r.throughput == pytest.approx(2.0)


def test_batch_pipeline_record_throughput_zero_elapsed():
    r = make_batch_pipeline_record(0, 10, 10, 0, 0, 0.0, "lbp")
    assert r.throughput == 0.0


def test_batch_pipeline_record_with_params():
    r = make_batch_pipeline_record(1, 5, 4, 1, 0, 1.0, "lbp", custom=42)
    assert r.params["custom"] == 42


# ── summarise_batch_pipeline ──────────────────────────────────────────────────

def test_summarise_batch_empty():
    s = summarise_batch_pipeline([])
    assert s.n_batches == 0
    assert s.best_batch_id is None
    assert s.worst_batch_id is None


def test_summarise_batch_records():
    records = _make_batch_records(4)
    s = summarise_batch_pipeline(records)
    assert s.n_batches == 4
    assert s.total_items == 40
    assert s.best_batch_id is not None


# ── filter batch functions ────────────────────────────────────────────────────

def test_filter_batch_by_success_rate():
    records = _make_batch_records(4)
    filtered = filter_batch_by_success_rate(records, 0.9)
    assert all(r.success_rate >= 0.9 for r in filtered)


def test_filter_batch_by_stage():
    records = _make_batch_records(6)
    filtered = filter_batch_by_stage(records, "lbp")
    assert all(r.stage == "lbp" for r in filtered)


def test_top_k_batch_records():
    records = _make_batch_records(4)
    top = top_k_batch_records(records, 2)
    assert len(top) == 2
    assert top[0].success_rate >= top[-1].success_rate


# ── batch_throughput_stats ────────────────────────────────────────────────────

def test_batch_throughput_stats_empty():
    stats = batch_throughput_stats([])
    assert stats["count"] == 0


def test_batch_throughput_stats():
    records = _make_batch_records(4)
    stats = batch_throughput_stats(records)
    assert stats["count"] == 4
    assert stats["max"] >= stats["min"]


# ── compare_batch_summaries ───────────────────────────────────────────────────

def test_compare_batch_summaries():
    a = summarise_batch_pipeline(_make_batch_records(2))
    b = summarise_batch_pipeline(_make_batch_records(4))
    cmp = compare_batch_summaries(a, b)
    assert "delta_total_done" in cmp
    assert "delta_mean_success_rate" in cmp
    assert cmp["delta_total_done"] > 0
