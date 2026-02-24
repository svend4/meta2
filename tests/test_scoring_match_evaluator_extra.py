"""Extra tests for puzzle_reconstruction/scoring/match_evaluator.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.scoring.match_evaluator import (
    EvalConfig,
    MatchEval,
    EvalReport,
    compute_precision,
    compute_recall,
    compute_f_score,
    evaluate_match,
    evaluate_batch_matches,
    aggregate_eval,
    filter_by_score,
    rank_matches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ev(pair=(0, 1), score=0.8, tp=8, fp=2, fn=1):
    return MatchEval(pair=pair, score=score, tp=tp, fp=fp, fn=fn)


# ─── EvalConfig ─────────────────────────────────────────────────────────────

class TestEvalConfigExtra:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.max_score == pytest.approx(1.0)
        assert cfg.n_levels == 10
        assert cfg.beta == pytest.approx(1.0)

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=-0.1)

    def test_max_le_min_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=0.5, max_score=0.5)

    def test_n_levels_too_small_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(n_levels=1)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=0.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=-1.0)


# ─── MatchEval ──────────────────────────────────────────────────────────────

class TestMatchEvalExtra:
    def test_precision(self):
        e = _ev(tp=8, fp=2, fn=0)
        assert e.precision == pytest.approx(0.8)

    def test_recall(self):
        e = _ev(tp=8, fp=0, fn=2)
        assert e.recall == pytest.approx(0.8)

    def test_f1(self):
        e = _ev(tp=10, fp=0, fn=0)
        assert e.f1 == pytest.approx(1.0)

    def test_f1_zero(self):
        e = _ev(tp=0, fp=0, fn=0)
        assert e.f1 == pytest.approx(0.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=-1.0)

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, tp=-1)

    def test_negative_fp_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fp=-1)

    def test_negative_fn_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fn=-1)

    def test_precision_zero_denom(self):
        e = _ev(tp=0, fp=0, fn=5)
        assert e.precision == pytest.approx(0.0)

    def test_recall_zero_denom(self):
        e = _ev(tp=0, fp=5, fn=0)
        assert e.recall == pytest.approx(0.0)


# ─── EvalReport ─────────────────────────────────────────────────────────────

class TestEvalReportExtra:
    def test_best_f1_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_f1 == pytest.approx(0.0)

    def test_best_pair_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_pair is None

    def test_best_f1_non_empty(self):
        e1 = _ev(tp=5, fp=5, fn=0)
        e2 = _ev(tp=10, fp=0, fn=0, pair=(2, 3))
        r = EvalReport(evals=[e1, e2], n_pairs=2,
                       mean_score=0.8, mean_f1=0.75)
        assert r.best_f1 == pytest.approx(1.0)
        assert r.best_pair == (2, 3)

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=-1, mean_score=0.0, mean_f1=0.0)

    def test_negative_mean_score_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=-1.0, mean_f1=0.0)

    def test_negative_mean_f1_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=-1.0)


# ─── compute_precision / recall / f_score ────────────────────────────────────

class TestComputeMetricsExtra:
    def test_precision_basic(self):
        assert compute_precision(8, 2) == pytest.approx(0.8)

    def test_precision_zero(self):
        assert compute_precision(0, 0) == pytest.approx(0.0)

    def test_precision_negative_tp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(-1, 0)

    def test_precision_negative_fp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(0, -1)

    def test_recall_basic(self):
        assert compute_recall(8, 2) == pytest.approx(0.8)

    def test_recall_zero(self):
        assert compute_recall(0, 0) == pytest.approx(0.0)

    def test_recall_negative_tp_raises(self):
        with pytest.raises(ValueError):
            compute_recall(-1, 0)

    def test_recall_negative_fn_raises(self):
        with pytest.raises(ValueError):
            compute_recall(0, -1)

    def test_f_score_perfect(self):
        assert compute_f_score(1.0, 1.0) == pytest.approx(1.0)

    def test_f_score_zero(self):
        assert compute_f_score(0.0, 0.0) == pytest.approx(0.0)

    def test_f_score_beta_zero_raises(self):
        with pytest.raises(ValueError):
            compute_f_score(0.5, 0.5, beta=0.0)

    def test_f_score_beta2(self):
        # F2 favors recall
        f = compute_f_score(0.5, 1.0, beta=2.0)
        assert 0.0 < f <= 1.0


# ─── evaluate_match ─────────────────────────────────────────────────────────

class TestEvaluateMatchExtra:
    def test_returns_eval(self):
        e = evaluate_match((0, 1), 0.8, 8, 2, 1)
        assert isinstance(e, MatchEval)
        assert e.pair == (0, 1)


# ─── evaluate_batch_matches ─────────────────────────────────────────────────

class TestEvaluateBatchExtra:
    def test_basic(self):
        evals = evaluate_batch_matches(
            [(0, 1), (2, 3)], [0.5, 0.9], [5, 8], [1, 0], [2, 1]
        )
        assert len(evals) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            evaluate_batch_matches([(0, 1)], [0.5, 0.9], [5], [1], [2])


# ─── aggregate_eval ─────────────────────────────────────────────────────────

class TestAggregateEvalExtra:
    def test_empty(self):
        r = aggregate_eval([])
        assert r.n_pairs == 0 and r.mean_f1 == pytest.approx(0.0)

    def test_single(self):
        e = _ev(tp=10, fp=0, fn=0, score=0.9)
        r = aggregate_eval([e])
        assert r.n_pairs == 1
        assert r.mean_score == pytest.approx(0.9)
        assert r.mean_f1 == pytest.approx(1.0)


# ─── filter_by_score ────────────────────────────────────────────────────────

class TestFilterByScoreEvalExtra:
    def test_basic(self):
        evals = [_ev(score=0.3), _ev(score=0.8)]
        assert len(filter_by_score(evals, 0.5)) == 1

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], -0.1)

    def test_zero_threshold_all(self):
        evals = [_ev(score=0.0), _ev(score=0.5)]
        assert len(filter_by_score(evals, 0.0)) == 2


# ─── rank_matches ────────────────────────────────────────────────────────────

class TestRankMatchesExtra:
    def test_by_f1(self):
        e1 = _ev(tp=5, fp=5, fn=0)
        e2 = _ev(tp=10, fp=0, fn=0)
        ranked = rank_matches([e1, e2], by="f1")
        assert ranked[0].f1 >= ranked[1].f1

    def test_by_score(self):
        e1 = _ev(score=0.3)
        e2 = _ev(score=0.9)
        ranked = rank_matches([e1, e2], by="score")
        assert ranked[0].score >= ranked[1].score

    def test_invalid_by_raises(self):
        with pytest.raises(ValueError):
            rank_matches([], by="invalid")
