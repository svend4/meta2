"""Extra tests for puzzle_reconstruction/utils/texture_pipeline_utils.py (iter-234)."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _tex(
    pair: tuple = (0, 1),
    score: float = 0.7,
    lbp_score: float = 0.6,
    gabor_score: float = 0.5,
    gradient_score: float = 0.4,
    side1: int = 0,
    side2: int = 0,
) -> TextureMatchRecord:
    return make_texture_match_record(
        pair=pair, score=score, lbp_score=lbp_score,
        gabor_score=gabor_score, gradient_score=gradient_score,
        side1=side1, side2=side2,
    )


def _batch(
    batch_id: int = 0,
    n_items: int = 10,
    n_done: int = 8,
    n_failed: int = 1,
    n_skipped: int = 1,
    elapsed: float = 2.0,
    stage: str = "texture",
) -> BatchPipelineRecord:
    return make_batch_pipeline_record(
        batch_id=batch_id, n_items=n_items, n_done=n_done,
        n_failed=n_failed, n_skipped=n_skipped,
        elapsed=elapsed, stage=stage,
    )


def _tex_records_mixed() -> list:
    return [
        _tex(pair=(0, 1), score=0.9, lbp_score=0.85, gabor_score=0.8, gradient_score=0.7),
        _tex(pair=(1, 2), score=0.3, lbp_score=0.2, gabor_score=0.4, gradient_score=0.3),
        _tex(pair=(2, 3), score=0.7, lbp_score=0.6, gabor_score=0.5, gradient_score=0.4),
        _tex(pair=(3, 4), score=0.5, lbp_score=0.45, gabor_score=0.55, gradient_score=0.5),
        _tex(pair=(4, 5), score=0.6, lbp_score=0.7, gabor_score=0.3, gradient_score=0.6),
    ]


def _batch_records_mixed() -> list:
    return [
        _batch(batch_id=0, n_items=10, n_done=9, n_failed=1, n_skipped=0, elapsed=2.0, stage="texture"),
        _batch(batch_id=1, n_items=10, n_done=5, n_failed=3, n_skipped=2, elapsed=3.0, stage="color"),
        _batch(batch_id=2, n_items=20, n_done=18, n_failed=1, n_skipped=1, elapsed=5.0, stage="texture"),
        _batch(batch_id=3, n_items=10, n_done=2, n_failed=6, n_skipped=2, elapsed=1.0, stage="edge"),
    ]


# ─── TextureMatchRecord ────────────────────────────────────────────────────

class TestTextureMatchRecordExtra:
    def test_pair_stored(self):
        r = _tex(pair=(3, 7))
        assert r.pair == (3, 7)

    def test_score_stored(self):
        r = _tex(score=0.85)
        assert r.score == pytest.approx(0.85)

    def test_lbp_score_stored(self):
        r = _tex(lbp_score=0.72)
        assert r.lbp_score == pytest.approx(0.72)

    def test_gabor_score_stored(self):
        r = _tex(gabor_score=0.63)
        assert r.gabor_score == pytest.approx(0.63)

    def test_gradient_score_stored(self):
        r = _tex(gradient_score=0.44)
        assert r.gradient_score == pytest.approx(0.44)

    def test_side1_stored(self):
        r = _tex(side1=2)
        assert r.side1 == 2

    def test_side2_stored(self):
        r = _tex(side2=3)
        assert r.side2 == 3

    def test_default_sides_zero(self):
        r = _tex()
        assert r.side1 == 0
        assert r.side2 == 0

    def test_params_default_empty(self):
        r = _tex()
        assert r.params == {}


# ─── TextureMatchSummary ───────────────────────────────────────────────────

class TestTextureMatchSummaryExtra:
    def test_fields(self):
        s = TextureMatchSummary(
            n_pairs=3, mean_score=0.5, mean_lbp=0.4,
            mean_gabor=0.3, mean_gradient=0.2,
            best_pair=(0, 1), best_score=0.9,
        )
        assert s.n_pairs == 3
        assert s.mean_score == pytest.approx(0.5)

    def test_best_pair(self):
        s = TextureMatchSummary(
            n_pairs=1, mean_score=0.7, mean_lbp=0.6,
            mean_gabor=0.5, mean_gradient=0.4,
            best_pair=(2, 5), best_score=0.7,
        )
        assert s.best_pair == (2, 5)

    def test_best_score(self):
        s = TextureMatchSummary(
            n_pairs=1, mean_score=0.7, mean_lbp=0.6,
            mean_gabor=0.5, mean_gradient=0.4,
            best_pair=(0, 1), best_score=0.95,
        )
        assert s.best_score == pytest.approx(0.95)

    def test_mean_lbp(self):
        s = TextureMatchSummary(
            n_pairs=1, mean_score=0.0, mean_lbp=0.33,
            mean_gabor=0.0, mean_gradient=0.0,
            best_pair=None, best_score=0.0,
        )
        assert s.mean_lbp == pytest.approx(0.33)

    def test_mean_gabor(self):
        s = TextureMatchSummary(
            n_pairs=1, mean_score=0.0, mean_lbp=0.0,
            mean_gabor=0.77, mean_gradient=0.0,
            best_pair=None, best_score=0.0,
        )
        assert s.mean_gabor == pytest.approx(0.77)


# ─── make_texture_match_record ──────────────────────────────────────────────

class TestMakeTextureMatchRecordExtra:
    def test_returns_record(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4)
        assert isinstance(r, TextureMatchRecord)

    def test_score_as_float(self):
        r = make_texture_match_record((0, 1), 1, 1, 1, 1)
        assert isinstance(r.score, float)

    def test_side_as_int(self):
        r = make_texture_match_record((0, 1), 0.5, 0.5, 0.5, 0.5, side1=2.0, side2=3.0)
        assert isinstance(r.side1, int)
        assert isinstance(r.side2, int)

    def test_kwargs_stored_in_params(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4, method="lbp")
        assert r.params["method"] == "lbp"

    def test_multiple_kwargs(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4, a=1, b=2)
        assert r.params == {"a": 1, "b": 2}

    def test_no_kwargs_empty_params(self):
        r = make_texture_match_record((0, 1), 0.7, 0.6, 0.5, 0.4)
        assert r.params == {}


# ─── summarise_texture_matches ──────────────────────────────────────────────

class TestSummariseTextureMatchesExtra:
    def test_empty(self):
        s = summarise_texture_matches([])
        assert s.n_pairs == 0
        assert s.best_pair is None
        assert s.best_score == pytest.approx(0.0)

    def test_single_record(self):
        s = summarise_texture_matches([_tex(pair=(0, 1), score=0.8)])
        assert s.n_pairs == 1
        assert s.mean_score == pytest.approx(0.8)
        assert s.best_pair == (0, 1)

    def test_mean_score(self):
        records = [_tex(score=0.4), _tex(score=0.8)]
        s = summarise_texture_matches(records)
        assert s.mean_score == pytest.approx(0.6)

    def test_mean_lbp(self):
        records = [_tex(lbp_score=0.3), _tex(lbp_score=0.7)]
        s = summarise_texture_matches(records)
        assert s.mean_lbp == pytest.approx(0.5)

    def test_mean_gabor(self):
        records = [_tex(gabor_score=0.2), _tex(gabor_score=0.6)]
        s = summarise_texture_matches(records)
        assert s.mean_gabor == pytest.approx(0.4)

    def test_mean_gradient(self):
        records = [_tex(gradient_score=0.1), _tex(gradient_score=0.9)]
        s = summarise_texture_matches(records)
        assert s.mean_gradient == pytest.approx(0.5)

    def test_best_pair_is_highest_score(self):
        records = [
            _tex(pair=(0, 1), score=0.3),
            _tex(pair=(2, 3), score=0.9),
        ]
        s = summarise_texture_matches(records)
        assert s.best_pair == (2, 3)
        assert s.best_score == pytest.approx(0.9)


# ─── filter_texture_by_score ────────────────────────────────────────────────

class TestFilterTextureByScoreExtra:
    def test_all_pass(self):
        records = _tex_records_mixed()
        assert len(filter_texture_by_score(records, 0.0)) == 5

    def test_high_threshold(self):
        records = _tex_records_mixed()
        result = filter_texture_by_score(records, 0.85)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_exact_boundary(self):
        records = [_tex(score=0.5)]
        assert len(filter_texture_by_score(records, 0.5)) == 1

    def test_none_pass(self):
        records = _tex_records_mixed()
        assert len(filter_texture_by_score(records, 1.0)) == 0

    def test_empty(self):
        assert filter_texture_by_score([], 0.5) == []


# ─── filter_texture_by_lbp ─────────────────────────────────────────────────

class TestFilterTextureByLbpExtra:
    def test_all_pass(self):
        records = _tex_records_mixed()
        assert len(filter_texture_by_lbp(records, 0.0)) == 5

    def test_high_threshold(self):
        records = _tex_records_mixed()
        result = filter_texture_by_lbp(records, 0.8)
        assert len(result) == 1
        assert result[0].lbp_score == pytest.approx(0.85)

    def test_exact_boundary(self):
        records = [_tex(lbp_score=0.5)]
        assert len(filter_texture_by_lbp(records, 0.5)) == 1

    def test_none_pass(self):
        records = _tex_records_mixed()
        assert len(filter_texture_by_lbp(records, 1.0)) == 0

    def test_empty(self):
        assert filter_texture_by_lbp([], 0.5) == []


# ─── top_k_texture_records ──────────────────────────────────────────────────

class TestTopKTextureRecordsExtra:
    def test_top_1(self):
        records = _tex_records_mixed()
        result = top_k_texture_records(records, 1)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_top_3(self):
        records = _tex_records_mixed()
        result = top_k_texture_records(records, 3)
        assert len(result) == 3
        assert result[0].score >= result[1].score >= result[2].score

    def test_k_greater_than_n(self):
        records = _tex_records_mixed()
        result = top_k_texture_records(records, 100)
        assert len(result) == len(records)

    def test_empty(self):
        assert top_k_texture_records([], 5) == []

    def test_descending_order(self):
        records = _tex_records_mixed()
        result = top_k_texture_records(records, 5)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


# ─── best_texture_record ───────────────────────────────────────────────────

class TestBestTextureRecordExtra:
    def test_returns_highest(self):
        records = _tex_records_mixed()
        best = best_texture_record(records)
        assert best is not None
        assert best.score == pytest.approx(0.9)

    def test_empty_returns_none(self):
        assert best_texture_record([]) is None

    def test_single(self):
        r = _tex(score=0.42)
        assert best_texture_record([r]).score == pytest.approx(0.42)

    def test_returns_record_type(self):
        records = _tex_records_mixed()
        assert isinstance(best_texture_record(records), TextureMatchRecord)

    def test_tie_returns_one(self):
        records = [_tex(score=0.5), _tex(score=0.5)]
        best = best_texture_record(records)
        assert best.score == pytest.approx(0.5)


# ─── texture_score_stats ───────────────────────────────────────────────────

class TestTextureScoreStatsExtra:
    def test_empty(self):
        stats = texture_score_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == pytest.approx(0.0)

    def test_single(self):
        stats = texture_score_stats([_tex(score=0.5)])
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["std"] == pytest.approx(0.0)

    def test_min_max(self):
        records = _tex_records_mixed()
        stats = texture_score_stats(records)
        assert stats["min"] == pytest.approx(0.3)
        assert stats["max"] == pytest.approx(0.9)

    def test_mean(self):
        records = [_tex(score=0.2), _tex(score=0.8)]
        stats = texture_score_stats(records)
        assert stats["mean"] == pytest.approx(0.5)

    def test_count(self):
        records = _tex_records_mixed()
        assert texture_score_stats(records)["count"] == 5

    def test_std_positive_for_varied(self):
        records = _tex_records_mixed()
        assert texture_score_stats(records)["std"] > 0.0


# ─── BatchPipelineRecord ───────────────────────────────────────────────────

class TestBatchPipelineRecordExtra:
    def test_success_rate(self):
        r = _batch(n_items=10, n_done=8)
        assert r.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_items(self):
        r = _batch(n_items=0, n_done=0)
        assert r.success_rate == pytest.approx(0.0)

    def test_success_rate_all_done(self):
        r = _batch(n_items=10, n_done=10)
        assert r.success_rate == pytest.approx(1.0)

    def test_throughput(self):
        r = _batch(n_done=10, elapsed=2.0)
        assert r.throughput == pytest.approx(5.0)

    def test_throughput_zero_elapsed(self):
        r = _batch(n_done=10, elapsed=0.0)
        assert r.throughput == pytest.approx(0.0)

    def test_throughput_high_elapsed(self):
        r = _batch(n_done=100, elapsed=10.0)
        assert r.throughput == pytest.approx(10.0)

    def test_fields_stored(self):
        r = _batch(batch_id=5, n_items=20, stage="color")
        assert r.batch_id == 5
        assert r.n_items == 20
        assert r.stage == "color"

    def test_params_default_empty(self):
        r = _batch()
        assert r.params == {}


# ─── BatchPipelineSummary ──────────────────────────────────────────────────

class TestBatchPipelineSummaryExtra:
    def test_fields(self):
        s = BatchPipelineSummary(
            n_batches=3, total_items=30, total_done=25,
            total_failed=3, total_elapsed=10.0,
            mean_success_rate=0.83, best_batch_id=0, worst_batch_id=2,
        )
        assert s.n_batches == 3
        assert s.total_items == 30

    def test_best_batch_id(self):
        s = BatchPipelineSummary(
            n_batches=2, total_items=20, total_done=15,
            total_failed=3, total_elapsed=5.0,
            mean_success_rate=0.75, best_batch_id=1, worst_batch_id=0,
        )
        assert s.best_batch_id == 1

    def test_worst_batch_id(self):
        s = BatchPipelineSummary(
            n_batches=2, total_items=20, total_done=15,
            total_failed=3, total_elapsed=5.0,
            mean_success_rate=0.75, best_batch_id=1, worst_batch_id=0,
        )
        assert s.worst_batch_id == 0

    def test_mean_success_rate(self):
        s = BatchPipelineSummary(
            n_batches=1, total_items=10, total_done=7,
            total_failed=2, total_elapsed=3.0,
            mean_success_rate=0.7, best_batch_id=0, worst_batch_id=0,
        )
        assert s.mean_success_rate == pytest.approx(0.7)

    def test_total_elapsed(self):
        s = BatchPipelineSummary(
            n_batches=1, total_items=10, total_done=10,
            total_failed=0, total_elapsed=4.5,
            mean_success_rate=1.0, best_batch_id=0, worst_batch_id=0,
        )
        assert s.total_elapsed == pytest.approx(4.5)


# ─── make_batch_pipeline_record ─────────────────────────────────────────────

class TestMakeBatchPipelineRecordExtra:
    def test_returns_record(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2.0, "texture")
        assert isinstance(r, BatchPipelineRecord)

    def test_batch_id_as_int(self):
        r = make_batch_pipeline_record(1.0, 10, 8, 1, 1, 2.0, "texture")
        assert isinstance(r.batch_id, int)

    def test_elapsed_as_float(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2, "texture")
        assert isinstance(r.elapsed, float)

    def test_kwargs_stored_in_params(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2.0, "texture", gpu=True)
        assert r.params["gpu"] is True

    def test_no_kwargs_empty_params(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2.0, "texture")
        assert r.params == {}

    def test_stage_stored(self):
        r = make_batch_pipeline_record(0, 10, 8, 1, 1, 2.0, "edge")
        assert r.stage == "edge"


# ─── summarise_batch_pipeline ───────────────────────────────────────────────

class TestSummariseBatchPipelineExtra:
    def test_empty(self):
        s = summarise_batch_pipeline([])
        assert s.n_batches == 0
        assert s.best_batch_id is None
        assert s.worst_batch_id is None

    def test_single_record(self):
        s = summarise_batch_pipeline([_batch(batch_id=0, n_items=10, n_done=8)])
        assert s.n_batches == 1
        assert s.total_items == 10
        assert s.total_done == 8

    def test_total_items(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        assert s.total_items == 50  # 10 + 10 + 20 + 10

    def test_total_done(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        assert s.total_done == 34  # 9 + 5 + 18 + 2

    def test_total_failed(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        assert s.total_failed == 11  # 1 + 3 + 1 + 6

    def test_total_elapsed(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        assert s.total_elapsed == pytest.approx(11.0)  # 2 + 3 + 5 + 1

    def test_best_batch_id(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        # success rates: 0.9, 0.5, 0.9, 0.2 -> best is batch_id=0 (first 0.9)
        assert s.best_batch_id in (0, 2)

    def test_worst_batch_id(self):
        records = _batch_records_mixed()
        s = summarise_batch_pipeline(records)
        assert s.worst_batch_id == 3  # success rate 0.2


# ─── filter_batch_by_success_rate ───────────────────────────────────────────

class TestFilterBatchBySuccessRateExtra:
    def test_all_pass(self):
        records = _batch_records_mixed()
        assert len(filter_batch_by_success_rate(records, 0.0)) == 4

    def test_high_threshold(self):
        records = _batch_records_mixed()
        result = filter_batch_by_success_rate(records, 0.85)
        assert len(result) == 2  # batch 0 (0.9) and batch 2 (0.9)

    def test_none_pass(self):
        records = _batch_records_mixed()
        assert len(filter_batch_by_success_rate(records, 1.0)) == 0

    def test_exact_boundary(self):
        records = [_batch(n_items=10, n_done=5)]
        assert len(filter_batch_by_success_rate(records, 0.5)) == 1

    def test_empty(self):
        assert filter_batch_by_success_rate([], 0.5) == []


# ─── filter_batch_by_stage ──────────────────────────────────────────────────

class TestFilterBatchByStageExtra:
    def test_filter_texture(self):
        records = _batch_records_mixed()
        result = filter_batch_by_stage(records, "texture")
        assert len(result) == 2
        assert all(r.stage == "texture" for r in result)

    def test_filter_color(self):
        records = _batch_records_mixed()
        result = filter_batch_by_stage(records, "color")
        assert len(result) == 1

    def test_filter_edge(self):
        records = _batch_records_mixed()
        result = filter_batch_by_stage(records, "edge")
        assert len(result) == 1

    def test_nonexistent_stage(self):
        records = _batch_records_mixed()
        assert filter_batch_by_stage(records, "unknown") == []

    def test_empty(self):
        assert filter_batch_by_stage([], "texture") == []


# ─── top_k_batch_records ───────────────────────────────────────────────────

class TestTopKBatchRecordsExtra:
    def test_top_1(self):
        records = _batch_records_mixed()
        result = top_k_batch_records(records, 1)
        assert len(result) == 1
        assert result[0].success_rate == pytest.approx(0.9)

    def test_top_2(self):
        records = _batch_records_mixed()
        result = top_k_batch_records(records, 2)
        assert len(result) == 2
        assert result[0].success_rate >= result[1].success_rate

    def test_k_greater_than_n(self):
        records = _batch_records_mixed()
        result = top_k_batch_records(records, 100)
        assert len(result) == len(records)

    def test_empty(self):
        assert top_k_batch_records([], 5) == []

    def test_descending_success_rate(self):
        records = _batch_records_mixed()
        result = top_k_batch_records(records, 4)
        rates = [r.success_rate for r in result]
        assert rates == sorted(rates, reverse=True)


# ─── batch_throughput_stats ─────────────────────────────────────────────────

class TestBatchThroughputStatsExtra:
    def test_empty(self):
        stats = batch_throughput_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == pytest.approx(0.0)

    def test_single(self):
        stats = batch_throughput_stats([_batch(n_done=10, elapsed=2.0)])
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(5.0)

    def test_min_max(self):
        records = [
            _batch(n_done=10, elapsed=2.0),  # throughput 5.0
            _batch(n_done=20, elapsed=4.0),  # throughput 5.0
            _batch(n_done=6, elapsed=3.0),   # throughput 2.0
        ]
        stats = batch_throughput_stats(records)
        assert stats["min"] == pytest.approx(2.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_count(self):
        records = _batch_records_mixed()
        assert batch_throughput_stats(records)["count"] == 4

    def test_mean_throughput(self):
        records = [
            _batch(n_done=4, elapsed=2.0),  # throughput 2.0
            _batch(n_done=6, elapsed=2.0),  # throughput 3.0
        ]
        stats = batch_throughput_stats(records)
        assert stats["mean"] == pytest.approx(2.5)


# ─── compare_batch_summaries ───────────────────────────────────────────────

class TestCompareBatchSummariesExtra:
    def test_identical(self):
        s = summarise_batch_pipeline(_batch_records_mixed())
        delta = compare_batch_summaries(s, s)
        assert delta["delta_total_done"] == 0
        assert delta["delta_total_failed"] == 0
        assert delta["delta_mean_success_rate"] == pytest.approx(0.0)

    def test_b_more_done(self):
        a = summarise_batch_pipeline([_batch(n_items=10, n_done=5)])
        b = summarise_batch_pipeline([_batch(n_items=10, n_done=9)])
        delta = compare_batch_summaries(a, b)
        assert delta["delta_total_done"] > 0

    def test_delta_total_elapsed(self):
        a = summarise_batch_pipeline([_batch(elapsed=2.0)])
        b = summarise_batch_pipeline([_batch(elapsed=5.0)])
        delta = compare_batch_summaries(a, b)
        assert delta["delta_total_elapsed"] == pytest.approx(3.0)

    def test_same_best_true(self):
        records = [_batch(batch_id=0)]
        s = summarise_batch_pipeline(records)
        delta = compare_batch_summaries(s, s)
        assert delta["same_best"] is True

    def test_same_best_false(self):
        a = summarise_batch_pipeline([_batch(batch_id=0, n_items=10, n_done=10)])
        b = summarise_batch_pipeline([_batch(batch_id=1, n_items=10, n_done=10)])
        delta = compare_batch_summaries(a, b)
        assert delta["same_best"] is False

    def test_returns_dict(self):
        s = summarise_batch_pipeline([_batch()])
        assert isinstance(compare_batch_summaries(s, s), dict)
