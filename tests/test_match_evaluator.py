"""Тесты для puzzle_reconstruction.scoring.match_evaluator."""
import pytest

from puzzle_reconstruction.scoring.match_evaluator import (
    EvalConfig,
    EvalReport,
    MatchEval,
    aggregate_eval,
    compute_f_score,
    compute_precision,
    compute_recall,
    evaluate_batch_matches,
    evaluate_match,
    filter_by_score,
    rank_matches,
)


# ─── TestEvalConfig ────────────────────────────────────────────────────────────

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_score == 1.0
        assert cfg.n_levels == 10
        assert cfg.beta == 1.0

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=-0.1)

    def test_max_le_min_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=0.5, max_score=0.5)

    def test_max_lt_min_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=0.8, max_score=0.2)

    def test_n_levels_lt_2_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(n_levels=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=-1.0)

    def test_custom_valid(self):
        cfg = EvalConfig(min_score=0.1, max_score=0.9, n_levels=5, beta=2.0)
        assert cfg.min_score == 0.1
        assert cfg.max_score == 0.9
        assert cfg.n_levels == 5
        assert cfg.beta == 2.0


# ─── TestMatchEval ─────────────────────────────────────────────────────────────

class TestMatchEval:
    def _make(self, tp=3, fp=1, fn=2, score=0.8):
        return MatchEval(pair=(0, 1), score=score, tp=tp, fp=fp, fn=fn)

    def test_basic_construction(self):
        m = self._make()
        assert m.pair == (0, 1)
        assert m.score == 0.8

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=-0.1)

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, tp=-1)

    def test_negative_fp_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fp=-1)

    def test_negative_fn_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fn=-1)

    def test_precision_basic(self):
        m = self._make(tp=3, fp=1)
        assert abs(m.precision - 0.75) < 1e-9

    def test_precision_zero_denominator(self):
        m = MatchEval(pair=(0, 1), score=0.0, tp=0, fp=0)
        assert m.precision == 0.0

    def test_recall_basic(self):
        m = self._make(tp=3, fn=1)
        assert abs(m.recall - 0.75) < 1e-9

    def test_recall_zero_denominator(self):
        m = MatchEval(pair=(0, 1), score=0.0, tp=0, fn=0)
        assert m.recall == 0.0

    def test_f1_perfect(self):
        m = MatchEval(pair=(0, 1), score=1.0, tp=5, fp=0, fn=0)
        assert abs(m.f1 - 1.0) < 1e-9

    def test_f1_zero_when_no_tp(self):
        m = MatchEval(pair=(0, 1), score=0.0, tp=0, fp=5, fn=5)
        assert m.f1 == 0.0

    def test_f1_basic(self):
        m = self._make(tp=3, fp=1, fn=2)
        p = 3 / 4
        r = 3 / 5
        expected = 2 * p * r / (p + r)
        assert abs(m.f1 - expected) < 1e-9


# ─── TestEvalReport ────────────────────────────────────────────────────────────

class TestEvalReport:
    def _make_evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.9, tp=5, fp=1, fn=0),
            MatchEval(pair=(1, 2), score=0.5, tp=2, fp=3, fn=2),
            MatchEval(pair=(2, 3), score=0.3, tp=0, fp=5, fn=5),
        ]

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=-1, mean_score=0.0, mean_f1=0.0)

    def test_negative_mean_score_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=-0.1, mean_f1=0.0)

    def test_negative_mean_f1_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=-0.1)

    def test_best_f1_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_f1 == 0.0

    def test_best_pair_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_pair is None

    def test_best_f1_nonzero(self):
        evals = self._make_evals()
        r = EvalReport(evals=evals, n_pairs=3, mean_score=0.5, mean_f1=0.4)
        assert r.best_f1 == max(e.f1 for e in evals)

    def test_best_pair_returns_highest_f1(self):
        evals = self._make_evals()
        r = EvalReport(evals=evals, n_pairs=3, mean_score=0.5, mean_f1=0.4)
        best = max(evals, key=lambda e: e.f1)
        assert r.best_pair == best.pair


# ─── TestComputePrecision ──────────────────────────────────────────────────────

class TestComputePrecision:
    def test_basic(self):
        assert abs(compute_precision(3, 1) - 0.75) < 1e-9

    def test_zero_denom(self):
        assert compute_precision(0, 0) == 0.0

    def test_perfect(self):
        assert compute_precision(5, 0) == 1.0

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(-1, 0)

    def test_negative_fp_raises(self):
        with pytest.raises(ValueError):
            compute_precision(0, -1)


# ─── TestComputeRecall ─────────────────────────────────────────────────────────

class TestComputeRecall:
    def test_basic(self):
        assert abs(compute_recall(3, 1) - 0.75) < 1e-9

    def test_zero_denom(self):
        assert compute_recall(0, 0) == 0.0

    def test_perfect(self):
        assert compute_recall(5, 0) == 1.0

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError):
            compute_recall(-1, 0)

    def test_negative_fn_raises(self):
        with pytest.raises(ValueError):
            compute_recall(0, -1)


# ─── TestComputeFScore ─────────────────────────────────────────────────────────

class TestComputeFScore:
    def test_f1_basic(self):
        p, r = 0.8, 0.6
        expected = 2 * p * r / (p + r)
        assert abs(compute_f_score(p, r) - expected) < 1e-9

    def test_f1_perfect(self):
        assert abs(compute_f_score(1.0, 1.0) - 1.0) < 1e-9

    def test_f1_zero(self):
        assert compute_f_score(0.0, 0.0) == 0.0

    def test_beta_2(self):
        p, r, b = 0.5, 0.8, 2.0
        b2 = b ** 2
        expected = (1 + b2) * p * r / (b2 * p + r)
        assert abs(compute_f_score(p, r, beta=b) - expected) < 1e-9

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            compute_f_score(0.5, 0.5, beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            compute_f_score(0.5, 0.5, beta=-1.0)


# ─── TestEvaluateMatch ────────────────────────────────────────────────────────

class TestEvaluateMatch:
    def test_returns_match_eval(self):
        m = evaluate_match((0, 1), 0.7, tp=3, fp=1, fn=2)
        assert isinstance(m, MatchEval)

    def test_fields_preserved(self):
        m = evaluate_match((2, 5), 0.5, tp=4, fp=0, fn=1)
        assert m.pair == (2, 5)
        assert m.score == 0.5
        assert m.tp == 4
        assert m.fp == 0
        assert m.fn == 1


# ─── TestEvaluateBatchMatches ─────────────────────────────────────────────────

class TestEvaluateBatchMatches:
    def test_basic_batch(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.8, 0.5]
        tp_list = [3, 2]
        fp_list = [1, 2]
        fn_list = [0, 1]
        result = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        assert len(result) == 2
        assert all(isinstance(e, MatchEval) for e in result)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_batch_matches([(0, 1)], [0.5, 0.8], [1], [0], [0])

    def test_empty_batch(self):
        result = evaluate_batch_matches([], [], [], [], [])
        assert result == []

    def test_values_correct(self):
        result = evaluate_batch_matches([(0, 1)], [0.9], [5], [0], [1])
        assert result[0].pair == (0, 1)
        assert result[0].score == 0.9
        assert result[0].tp == 5


# ─── TestAggregateEval ────────────────────────────────────────────────────────

class TestAggregateEval:
    def test_empty_list(self):
        r = aggregate_eval([])
        assert isinstance(r, EvalReport)
        assert r.n_pairs == 0
        assert r.mean_score == 0.0
        assert r.mean_f1 == 0.0

    def test_n_pairs(self):
        evals = [MatchEval(pair=(0, 1), score=0.5, tp=1, fp=1, fn=1),
                 MatchEval(pair=(1, 2), score=0.7, tp=2, fp=0, fn=1)]
        r = aggregate_eval(evals)
        assert r.n_pairs == 2

    def test_mean_score(self):
        evals = [MatchEval(pair=(0, 1), score=0.4, tp=1, fp=1, fn=1),
                 MatchEval(pair=(1, 2), score=0.6, tp=1, fp=1, fn=1)]
        r = aggregate_eval(evals)
        assert abs(r.mean_score - 0.5) < 1e-9

    def test_mean_f1(self):
        evals = [MatchEval(pair=(0, 1), score=0.5, tp=5, fp=0, fn=0)]
        r = aggregate_eval(evals)
        assert abs(r.mean_f1 - 1.0) < 1e-9

    def test_evals_stored(self):
        evals = [MatchEval(pair=(0, 1), score=0.5)]
        r = aggregate_eval(evals)
        assert r.evals is evals


# ─── TestFilterByScore ────────────────────────────────────────────────────────

class TestFilterByScore:
    def _evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.9),
            MatchEval(pair=(1, 2), score=0.5),
            MatchEval(pair=(2, 3), score=0.2),
        ]

    def test_filters_below_threshold(self):
        result = filter_by_score(self._evals(), threshold=0.5)
        assert all(e.score >= 0.5 for e in result)

    def test_returns_all_above_zero(self):
        result = filter_by_score(self._evals(), threshold=0.0)
        assert len(result) == 3

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(self._evals(), threshold=-0.1)

    def test_high_threshold_empty(self):
        result = filter_by_score(self._evals(), threshold=1.0)
        assert len(result) == 0

    def test_exact_threshold_included(self):
        result = filter_by_score(self._evals(), threshold=0.9)
        assert len(result) == 1
        assert result[0].score == 0.9


# ─── TestRankMatches ──────────────────────────────────────────────────────────

class TestRankMatches:
    def _evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.3, tp=1, fp=3, fn=1),
            MatchEval(pair=(1, 2), score=0.9, tp=5, fp=0, fn=0),
            MatchEval(pair=(2, 3), score=0.6, tp=3, fp=1, fn=2),
        ]

    def test_rank_by_f1_descending(self):
        ranked = rank_matches(self._evals(), by="f1")
        f1s = [e.f1 for e in ranked]
        assert f1s == sorted(f1s, reverse=True)

    def test_rank_by_score_descending(self):
        ranked = rank_matches(self._evals(), by="score")
        scores = [e.score for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_by_raises(self):
        with pytest.raises(ValueError):
            rank_matches(self._evals(), by="precision")

    def test_empty_list(self):
        assert rank_matches([], by="f1") == []

    def test_returns_new_list(self):
        evals = self._evals()
        ranked = rank_matches(evals, by="f1")
        assert ranked is not evals
