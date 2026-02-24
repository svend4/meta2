"""Extra tests for puzzle_reconstruction/utils/texture_pipeline_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.texture_pipeline_utils import (
    TextureMatchRecord,
    TextureMatchSummary,
    BatchPipelineRecord,
    BatchPipelineSummary,
    make_texture_match_record,
    summarise_texture_matches,
    filter_texture_by_score,
    filter_texture_by_lbp,
    top_k_texture_records,
    best_texture_record,
    texture_score_stats,
    make_batch_pipeline_record,
    summarise_batch_pipeline,
    filter_batch_by_success_rate,
    filter_batch_by_stage,
    top_k_batch_records,
    batch_throughput_stats,
    compare_batch_summaries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _tex(pair=(0, 1), score=0.7, lbp=0.6, gabor=0.5,
         grad=0.4) -> TextureMatchRecord:
    return TextureMatchRecord(pair=pair, score=score, lbp_score=lbp,
                               gabor_score=gabor, gradient_score=grad)


def _batch(batch_id=0, n_items=10, n_done=8, n_failed=1,
           n_skipped=1, elapsed=2.0, stage="texture") -> BatchPipelineRecord:
    return BatchPipelineRecord(batch_id=batch_id, n_items=n_items,
                                n_done=n_done, n_failed=n_failed,
                                n_skipped=n_skipped, elapsed=elapsed,
                                stage=stage)


# ─── TextureMatchRecord ──────────────────────────────────────────────────────

class TestTextureMatchRecordExtra:
    def test_fields_stored(self):
        r = _tex(pair=(2, 5), score=0.9)
        assert r.pair == (2, 5) and r.score == pytest.approx(0.9)

    def test_params_default_empty(self):
        assert _tex().params == {}


# ─── make_texture_match_record ────────────────────────────────────────────────

class TestMakeTextureMatchRecordExtra:
    def test_returns_record(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4)
        assert isinstance(r, TextureMatchRecord)

    def test_params_stored(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4, k=3)
        assert r.params["k"] == 3


# ─── summarise_texture_matches ────────────────────────────────────────────────

class TestSummariseTextureExtra:
    def test_empty(self):
        s = summarise_texture_matches([])
        assert s.n_pairs == 0 and s.best_pair is None

    def test_mean_score(self):
        records = [_tex(score=0.4), _tex(score=0.8)]
        s = summarise_texture_matches(records)
        assert s.mean_score == pytest.approx(0.6)

    def test_best_pair(self):
        records = [_tex(pair=(0, 1), score=0.3),
                   _tex(pair=(2, 3), score=0.9)]
        s = summarise_texture_matches(records)
        assert s.best_pair == (2, 3)


# ─── filters texture ─────────────────────────────────────────────────────────

class TestFilterTextureExtra:
    def test_by_score(self):
        records = [_tex(score=0.3), _tex(score=0.8)]
        assert len(filter_texture_by_score(records, 0.5)) == 1

    def test_by_lbp(self):
        records = [_tex(lbp=0.2), _tex(lbp=0.8)]
        assert len(filter_texture_by_lbp(records, 0.5)) == 1


# ─── top_k / best / stats texture ────────────────────────────────────────────

class TestRankTextureExtra:
    def test_top_k(self):
        records = [_tex(score=0.3), _tex(score=0.9)]
        top = top_k_texture_records(records, 1)
        assert top[0].score == pytest.approx(0.9)

    def test_best(self):
        records = [_tex(score=0.3), _tex(score=0.9)]
        assert best_texture_record(records).score == pytest.approx(0.9)

    def test_best_empty(self):
        assert best_texture_record([]) is None

    def test_stats_empty(self):
        s = texture_score_stats([])
        assert s["count"] == 0


# ─── BatchPipelineRecord ─────────────────────────────────────────────────────

class TestBatchPipelineRecordExtra:
    def test_success_rate(self):
        r = _batch(n_items=10, n_done=8)
        assert r.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_items(self):
        r = _batch(n_items=0, n_done=0)
        assert r.success_rate == pytest.approx(0.0)

    def test_throughput(self):
        r = _batch(n_done=10, elapsed=2.0)
        assert r.throughput == pytest.approx(5.0)

    def test_throughput_zero_elapsed(self):
        r = _batch(n_done=10, elapsed=0.0)
        assert r.throughput == pytest.approx(0.0)


# ─── make / summarise batch pipeline ─────────────────────────────────────────

class TestBatchPipelineExtra:
    def test_make_returns_record(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2.0, "texture")
        assert isinstance(r, BatchPipelineRecord)

    def test_summarise_empty(self):
        s = summarise_batch_pipeline([])
        assert s.n_batches == 0

    def test_summarise_totals(self):
        records = [_batch(n_items=10, n_done=8, n_failed=1),
                   _batch(n_items=5, n_done=5, n_failed=0)]
        s = summarise_batch_pipeline(records)
        assert s.total_items == 15 and s.total_done == 13


# ─── filters batch ───────────────────────────────────────────────────────────

class TestFilterBatchExtra:
    def test_by_success_rate(self):
        records = [_batch(n_items=10, n_done=3), _batch(n_items=10, n_done=9)]
        assert len(filter_batch_by_success_rate(records, 0.5)) == 1

    def test_by_stage(self):
        records = [_batch(stage="texture"), _batch(stage="color")]
        assert len(filter_batch_by_stage(records, "texture")) == 1


# ─── top_k / stats batch ─────────────────────────────────────────────────────

class TestRankBatchExtra:
    def test_top_k(self):
        records = [_batch(n_items=10, n_done=3), _batch(n_items=10, n_done=9)]
        top = top_k_batch_records(records, 1)
        assert top[0].n_done == 9

    def test_throughput_stats_empty(self):
        s = batch_throughput_stats([])
        assert s["count"] == 0


# ─── compare batch ────────────────────────────────────────────────────────────

class TestCompareBatchExtra:
    def test_returns_dict(self):
        s = summarise_batch_pipeline([_batch()])
        d = compare_batch_summaries(s, s)
        assert isinstance(d, dict) and d["delta_total_done"] == 0
