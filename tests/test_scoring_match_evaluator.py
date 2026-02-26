"""Тесты для puzzle_reconstruction/scoring/match_evaluator.py."""
import pytest
import numpy as np

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


class TestEvalConfig:
    def test_default_values(self):
        c = EvalConfig()
        assert c.min_score == 0.0
        assert c.max_score == 1.0
        assert c.n_levels == 10
        assert c.beta == 1.0

    def test_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=-0.1)

    def test_max_score_le_min_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=0.5, max_score=0.5)

    def test_n_levels_less_than_2_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(n_levels=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=0.0)


class TestMatchEval:
    def test_precision_tp_fp(self):
        e = MatchEval(pair=(0, 1), score=0.8, tp=8, fp=2, fn=0)
        assert e.precision == pytest.approx(0.8)

    def test_recall_tp_fn(self):
        e = MatchEval(pair=(0, 1), score=0.8, tp=6, fp=0, fn=2)
        assert e.recall == pytest.approx(0.75)

    def test_f1_balanced(self):
        e = MatchEval(pair=(0, 1), score=0.8, tp=4, fp=1, fn=1)
        assert 0.0 < e.f1 <= 1.0

    def test_precision_zero_when_tp_fp_zero(self):
        e = MatchEval(pair=(0, 1), score=0.5, tp=0, fp=0, fn=0)
        assert e.precision == 0.0

    def test_recall_zero_when_tp_fn_zero(self):
        e = MatchEval(pair=(0, 1), score=0.5, tp=0, fp=0, fn=0)
        assert e.recall == 0.0

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=-0.1)

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, tp=-1)


class TestComputeMetrics:
    def test_precision_basic(self):
        assert compute_precision(3, 1) == pytest.approx(0.75)

    def test_precision_zero_denominator(self):
        assert compute_precision(0, 0) == 0.0

    def test_recall_basic(self):
        assert compute_recall(3, 1) == pytest.approx(0.75)

    def test_recall_zero_denominator(self):
        assert compute_recall(0, 0) == 0.0

    def test_f_score_symmetric_beta1(self):
        f = compute_f_score(0.8, 0.8)
        assert f == pytest.approx(0.8, abs=1e-6)

    def test_f_score_zero_when_both_zero(self):
        assert compute_f_score(0.0, 0.0) == 0.0

    def test_f_score_beta_zero_raises(self):
        with pytest.raises(ValueError):
            compute_f_score(0.5, 0.5, beta=0.0)

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(-1, 0)

    def test_negative_fp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(0, -1)


class TestEvaluateBatchMatches:
    def test_basic_batch(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        scores = [0.7, 0.5, 0.9]
        tp_list = [5, 3, 7]
        fp_list = [1, 2, 1]
        fn_list = [2, 3, 0]
        evals = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        assert len(evals) == 3
        assert all(isinstance(e, MatchEval) for e in evals)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            evaluate_batch_matches([(0, 1)], [0.5, 0.6], [1], [0], [0])

    def test_empty_batch(self):
        evals = evaluate_batch_matches([], [], [], [], [])
        assert evals == []


class TestAggregateEval:
    def test_empty_returns_zero(self):
        r = aggregate_eval([])
        assert r.n_pairs == 0
        assert r.mean_score == 0.0

    def test_mean_score_computed(self):
        e1 = MatchEval(pair=(0, 1), score=0.6, tp=3, fp=1, fn=1)
        e2 = MatchEval(pair=(1, 2), score=0.8, tp=4, fp=0, fn=1)
        r = aggregate_eval([e1, e2])
        assert r.mean_score == pytest.approx(0.7)
        assert r.n_pairs == 2

    def test_best_f1_property(self):
        e1 = MatchEval(pair=(0, 1), score=0.5, tp=1, fp=1, fn=1)
        e2 = MatchEval(pair=(1, 2), score=0.9, tp=9, fp=0, fn=0)
        r = aggregate_eval([e1, e2])
        assert r.best_f1 == pytest.approx(1.0)

    def test_best_pair_property(self):
        e1 = MatchEval(pair=(0, 1), score=0.5, tp=1, fp=2, fn=2)
        e2 = MatchEval(pair=(2, 3), score=0.9, tp=9, fp=0, fn=0)
        r = aggregate_eval([e1, e2])
        assert r.best_pair == (2, 3)


class TestRankMatches:
    def test_rank_by_f1_descending(self):
        e1 = MatchEval(pair=(0, 1), score=0.5, tp=1, fp=1, fn=1)
        e2 = MatchEval(pair=(1, 2), score=0.9, tp=9, fp=0, fn=0)
        ranked = rank_matches([e1, e2], by="f1")
        assert ranked[0].pair == (1, 2)

    def test_rank_by_score_descending(self):
        e1 = MatchEval(pair=(0, 1), score=0.3, tp=1, fp=0, fn=0)
        e2 = MatchEval(pair=(1, 2), score=0.9, tp=1, fp=0, fn=0)
        ranked = rank_matches([e1, e2], by="score")
        assert ranked[0].score > ranked[1].score

    def test_invalid_by_raises(self):
        e = MatchEval(pair=(0, 1), score=0.5)
        with pytest.raises(ValueError):
            rank_matches([e], by="invalid")

    def test_filter_by_score(self):
        evals = [
            MatchEval(pair=(0, 1), score=0.3),
            MatchEval(pair=(1, 2), score=0.7),
            MatchEval(pair=(2, 3), score=0.9),
        ]
        filtered = filter_by_score(evals, threshold=0.5)
        assert len(filtered) == 2
        assert all(e.score >= 0.5 for e in filtered)

    def test_filter_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], threshold=-0.1)
