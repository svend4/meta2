"""Extra tests for puzzle_reconstruction/scoring/match_evaluator.py"""
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


# ─── TestEvalConfigExtra ──────────────────────────────────────────────────────

class TestEvalConfigExtra:
    def test_beta_large_valid(self):
        cfg = EvalConfig(beta=10.0)
        assert cfg.beta == pytest.approx(10.0)

    def test_beta_small_positive(self):
        cfg = EvalConfig(beta=0.01)
        assert cfg.beta == pytest.approx(0.01)

    def test_n_levels_large(self):
        cfg = EvalConfig(n_levels=100)
        assert cfg.n_levels == 100

    def test_n_levels_exactly_2(self):
        cfg = EvalConfig(n_levels=2)
        assert cfg.n_levels == 2

    def test_max_score_above_one(self):
        cfg = EvalConfig(min_score=0.0, max_score=2.0)
        assert cfg.max_score == pytest.approx(2.0)

    def test_min_score_exactly_zero(self):
        cfg = EvalConfig(min_score=0.0)
        assert cfg.min_score == pytest.approx(0.0)

    def test_custom_all_params(self):
        cfg = EvalConfig(min_score=0.2, max_score=0.8, n_levels=20, beta=0.5)
        assert cfg.min_score == pytest.approx(0.2)
        assert cfg.max_score == pytest.approx(0.8)
        assert cfg.n_levels == 20
        assert cfg.beta == pytest.approx(0.5)


# ─── TestMatchEvalExtra ───────────────────────────────────────────────────────

class TestMatchEvalExtra:
    def test_all_zeros_ok(self):
        m = MatchEval(pair=(0, 1), score=0.0, tp=0, fp=0, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_perfect_match(self):
        m = MatchEval(pair=(0, 1), score=1.0, tp=10, fp=0, fn=0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_large_tp(self):
        m = MatchEval(pair=(0, 1), score=0.9, tp=1000, fp=0, fn=0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_score_exactly_one(self):
        m = MatchEval(pair=(0, 1), score=1.0)
        assert m.score == pytest.approx(1.0)

    def test_pair_stored_correctly(self):
        m = MatchEval(pair=(42, 99), score=0.5)
        assert m.pair == (42, 99)

    def test_fn_large(self):
        m = MatchEval(pair=(0, 1), score=0.5, tp=5, fp=0, fn=100)
        assert m.recall < 0.1

    def test_fp_large(self):
        m = MatchEval(pair=(0, 1), score=0.5, tp=5, fp=100, fn=0)
        assert m.precision < 0.1

    def test_f1_symmetric(self):
        m1 = MatchEval(pair=(0, 1), score=0.5, tp=3, fp=1, fn=2)
        m2 = MatchEval(pair=(0, 1), score=0.5, tp=3, fp=2, fn=1)
        # f1 is not symmetric in general unless p==r
        assert isinstance(m1.f1, float)
        assert isinstance(m2.f1, float)


# ─── TestEvalReportExtra ──────────────────────────────────────────────────────

class TestEvalReportExtra:
    def _make_evals(self, n=5):
        return [
            MatchEval(pair=(i, i + 1), score=0.5 + i * 0.05, tp=i + 1, fp=1, fn=1)
            for i in range(n)
        ]

    def test_n_pairs_stored(self):
        evals = self._make_evals(3)
        r = EvalReport(evals=evals, n_pairs=3, mean_score=0.6, mean_f1=0.5)
        assert r.n_pairs == 3

    def test_mean_score_stored(self):
        evals = self._make_evals(2)
        r = EvalReport(evals=evals, n_pairs=2, mean_score=0.75, mean_f1=0.5)
        assert r.mean_score == pytest.approx(0.75)

    def test_mean_f1_stored(self):
        evals = self._make_evals(2)
        r = EvalReport(evals=evals, n_pairs=2, mean_score=0.5, mean_f1=0.42)
        assert r.mean_f1 == pytest.approx(0.42)

    def test_best_f1_is_max(self):
        evals = self._make_evals(5)
        r = EvalReport(evals=evals, n_pairs=5, mean_score=0.6, mean_f1=0.5)
        assert r.best_f1 == pytest.approx(max(e.f1 for e in evals))

    def test_best_pair_is_highest_f1_pair(self):
        evals = self._make_evals(4)
        r = EvalReport(evals=evals, n_pairs=4, mean_score=0.6, mean_f1=0.5)
        best = max(evals, key=lambda e: e.f1)
        assert r.best_pair == best.pair

    def test_evals_stored(self):
        evals = self._make_evals(3)
        r = EvalReport(evals=evals, n_pairs=3, mean_score=0.6, mean_f1=0.5)
        assert r.evals is evals


# ─── TestComputePrecisionExtra ────────────────────────────────────────────────

class TestComputePrecisionExtra:
    def test_all_tp_no_fp(self):
        assert compute_precision(10, 0) == pytest.approx(1.0)

    def test_all_fp_no_tp(self):
        assert compute_precision(0, 10) == pytest.approx(0.0)

    def test_half(self):
        assert compute_precision(5, 5) == pytest.approx(0.5)

    def test_large_values(self):
        assert compute_precision(1000, 1000) == pytest.approx(0.5)

    def test_tp_much_larger_than_fp(self):
        v = compute_precision(999, 1)
        assert v > 0.99

    def test_result_is_float(self):
        assert isinstance(compute_precision(3, 1), float)


# ─── TestComputeRecallExtra ───────────────────────────────────────────────────

class TestComputeRecallExtra:
    def test_all_tp_no_fn(self):
        assert compute_recall(10, 0) == pytest.approx(1.0)

    def test_all_fn_no_tp(self):
        assert compute_recall(0, 10) == pytest.approx(0.0)

    def test_half(self):
        assert compute_recall(5, 5) == pytest.approx(0.5)

    def test_large_values(self):
        assert compute_recall(1000, 1000) == pytest.approx(0.5)

    def test_result_is_float(self):
        assert isinstance(compute_recall(3, 1), float)

    def test_tp_much_larger(self):
        v = compute_recall(999, 1)
        assert v > 0.99


# ─── TestComputeFScoreExtra ───────────────────────────────────────────────────

class TestComputeFScoreExtra:
    def test_beta_half(self):
        p, r, b = 0.8, 0.6, 0.5
        b2 = b ** 2
        expected = (1 + b2) * p * r / (b2 * p + r)
        assert compute_f_score(p, r, beta=b) == pytest.approx(expected, rel=1e-6)

    def test_precision_one_recall_zero(self):
        assert compute_f_score(1.0, 0.0) == pytest.approx(0.0)

    def test_precision_zero_recall_one(self):
        assert compute_f_score(0.0, 1.0) == pytest.approx(0.0)

    def test_result_in_zero_one(self):
        v = compute_f_score(0.7, 0.8)
        assert 0.0 <= v <= 1.0

    def test_f1_with_equal_p_r(self):
        assert compute_f_score(0.5, 0.5) == pytest.approx(0.5)

    def test_result_is_float(self):
        assert isinstance(compute_f_score(0.5, 0.5), float)


# ─── TestEvaluateMatchExtra ───────────────────────────────────────────────────

class TestEvaluateMatchExtra:
    def test_pair_stored(self):
        m = evaluate_match((10, 20), 0.5, tp=5, fp=1, fn=2)
        assert m.pair == (10, 20)

    def test_score_stored(self):
        m = evaluate_match((0, 1), 0.99, tp=3, fp=0, fn=0)
        assert m.score == pytest.approx(0.99)

    def test_all_zeros_no_crash(self):
        m = evaluate_match((0, 1), 0.0, tp=0, fp=0, fn=0)
        assert m.f1 == pytest.approx(0.0)

    def test_perfect_f1(self):
        m = evaluate_match((0, 1), 1.0, tp=10, fp=0, fn=0)
        assert m.f1 == pytest.approx(1.0)

    def test_returns_match_eval(self):
        assert isinstance(evaluate_match((0, 1), 0.5, tp=1, fp=0, fn=0), MatchEval)

    def test_tp_stored(self):
        m = evaluate_match((0, 1), 0.5, tp=7, fp=0, fn=0)
        assert m.tp == 7


# ─── TestEvaluateBatchMatchesExtra ────────────────────────────────────────────

class TestEvaluateBatchMatchesExtra:
    def test_five_pairs(self):
        n = 5
        pairs = [(i, i + 1) for i in range(n)]
        scores = [0.5 + i * 0.05 for i in range(n)]
        tp = [i + 1 for i in range(n)]
        fp = [1] * n
        fn = [1] * n
        result = evaluate_batch_matches(pairs, scores, tp, fp, fn)
        assert len(result) == n

    def test_all_perfect(self):
        pairs = [(0, 1), (2, 3)]
        result = evaluate_batch_matches(pairs, [1.0, 1.0], [5, 5], [0, 0], [0, 0])
        assert all(e.f1 == pytest.approx(1.0) for e in result)

    def test_all_zeros(self):
        pairs = [(0, 1)]
        result = evaluate_batch_matches(pairs, [0.0], [0], [0], [0])
        assert result[0].f1 == pytest.approx(0.0)

    def test_order_preserved(self):
        pairs = [(0, 1), (5, 6), (10, 11)]
        result = evaluate_batch_matches(pairs, [0.3, 0.7, 0.9], [1, 2, 3], [1, 1, 1], [1, 1, 1])
        assert result[0].pair == (0, 1)
        assert result[2].pair == (10, 11)

    def test_single_pair(self):
        result = evaluate_batch_matches([(0, 1)], [0.8], [4], [1], [2])
        assert len(result) == 1


# ─── TestAggregateEvalExtra ───────────────────────────────────────────────────

class TestAggregateEvalExtra:
    def test_ten_evals(self):
        evals = [MatchEval(pair=(i, i + 1), score=0.5, tp=3, fp=1, fn=1)
                 for i in range(10)]
        r = aggregate_eval(evals)
        assert r.n_pairs == 10

    def test_mean_score_correct(self):
        evals = [MatchEval(pair=(0, 1), score=0.2),
                 MatchEval(pair=(1, 2), score=0.4),
                 MatchEval(pair=(2, 3), score=0.6)]
        r = aggregate_eval(evals)
        assert r.mean_score == pytest.approx(0.4, abs=1e-9)

    def test_mean_f1_correct_all_perfect(self):
        evals = [MatchEval(pair=(i, i + 1), score=0.9, tp=5, fp=0, fn=0)
                 for i in range(3)]
        r = aggregate_eval(evals)
        assert r.mean_f1 == pytest.approx(1.0, abs=1e-9)

    def test_best_f1_from_aggregation(self):
        evals = [MatchEval(pair=(0, 1), score=0.5, tp=5, fp=0, fn=0),
                 MatchEval(pair=(1, 2), score=0.3, tp=2, fp=3, fn=2)]
        r = aggregate_eval(evals)
        assert r.best_f1 == pytest.approx(1.0)

    def test_evals_reference(self):
        evals = [MatchEval(pair=(0, 1), score=0.5)]
        r = aggregate_eval(evals)
        assert r.evals is evals


# ─── TestFilterByScoreExtra ───────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def _evals(self):
        return [
            MatchEval(pair=(i, i + 1), score=i * 0.1)
            for i in range(11)
        ]

    def test_threshold_zero_returns_all(self):
        result = filter_by_score(self._evals(), threshold=0.0)
        assert len(result) == 11

    def test_threshold_0_5_half(self):
        result = filter_by_score(self._evals(), threshold=0.5)
        assert all(e.score >= 0.5 for e in result)

    def test_single_eval_above(self):
        result = filter_by_score([MatchEval(pair=(0, 1), score=0.9)], threshold=0.5)
        assert len(result) == 1

    def test_single_eval_below(self):
        result = filter_by_score([MatchEval(pair=(0, 1), score=0.3)], threshold=0.5)
        assert len(result) == 0

    def test_exact_boundary_included(self):
        result = filter_by_score([MatchEval(pair=(0, 1), score=0.5)], threshold=0.5)
        assert len(result) == 1


# ─── TestRankMatchesExtra ─────────────────────────────────────────────────────

class TestRankMatchesExtra:
    def _evals(self, n=5):
        return [
            MatchEval(pair=(i, i + 1), score=i * 0.15,
                      tp=i + 1, fp=1, fn=max(0, 3 - i))
            for i in range(n)
        ]

    def test_by_score_descending(self):
        evals = self._evals(5)
        ranked = rank_matches(evals, by="score")
        scores = [e.score for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_by_f1_descending(self):
        evals = self._evals(5)
        ranked = rank_matches(evals, by="f1")
        f1s = [e.f1 for e in ranked]
        assert f1s == sorted(f1s, reverse=True)

    def test_empty_by_score(self):
        assert rank_matches([], by="score") == []

    def test_returns_new_list(self):
        evals = self._evals(3)
        ranked = rank_matches(evals, by="f1")
        assert ranked is not evals

    def test_all_elements_preserved(self):
        evals = self._evals(4)
        ranked = rank_matches(evals, by="score")
        assert len(ranked) == 4

    def test_single_eval_unchanged(self):
        evals = [MatchEval(pair=(0, 1), score=0.7, tp=3, fp=1, fn=1)]
        ranked = rank_matches(evals, by="f1")
        assert ranked[0].pair == (0, 1)
