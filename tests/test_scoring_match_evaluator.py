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


# ─── TestEvalConfig ───────────────────────────────────────────────────────────

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.max_score == pytest.approx(1.0)
        assert cfg.n_levels == 10
        assert cfg.beta == pytest.approx(1.0)

    def test_custom_values(self):
        cfg = EvalConfig(min_score=0.1, max_score=0.9, n_levels=5, beta=2.0)
        assert cfg.min_score == pytest.approx(0.1)
        assert cfg.n_levels == 5

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=-0.1)

    def test_max_score_le_min_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(min_score=0.5, max_score=0.5)

    def test_n_levels_lt_2_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(n_levels=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=0.0)

    def test_beta_neg_raises(self):
        with pytest.raises(ValueError):
            EvalConfig(beta=-1.0)


# ─── TestMatchEval ────────────────────────────────────────────────────────────

class TestMatchEval:
    def _make(self, tp=5, fp=2, fn=3, score=0.8) -> MatchEval:
        return MatchEval(pair=(0, 1), score=score, tp=tp, fp=fp, fn=fn)

    def test_pair_stored(self):
        m = self._make()
        assert m.pair == (0, 1)

    def test_precision_formula(self):
        m = self._make(tp=4, fp=1)
        assert m.precision == pytest.approx(4.0 / 5.0)

    def test_recall_formula(self):
        m = self._make(tp=4, fn=1)
        assert m.recall == pytest.approx(4.0 / 5.0)

    def test_f1_formula(self):
        m = self._make(tp=4, fp=0, fn=0)
        assert m.f1 == pytest.approx(1.0)

    def test_precision_zero_when_tp_fp_zero(self):
        m = self._make(tp=0, fp=0, fn=2)
        assert m.precision == pytest.approx(0.0)

    def test_recall_zero_when_tp_fn_zero(self):
        m = self._make(tp=0, fp=2, fn=0)
        assert m.recall == pytest.approx(0.0)

    def test_f1_zero_when_all_zero(self):
        m = self._make(tp=0, fp=0, fn=0)
        assert m.f1 == pytest.approx(0.0)

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=-0.1)

    def test_tp_neg_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, tp=-1)

    def test_fp_neg_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fp=-1)

    def test_fn_neg_raises(self):
        with pytest.raises(ValueError):
            MatchEval(pair=(0, 1), score=0.5, fn=-1)


# ─── TestEvalReport ───────────────────────────────────────────────────────────

class TestEvalReport:
    def _make_evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.9, tp=5, fp=1, fn=0),
            MatchEval(pair=(1, 2), score=0.7, tp=3, fp=2, fn=1),
        ]

    def _make_report(self) -> EvalReport:
        evals = self._make_evals()
        return EvalReport(evals=evals, n_pairs=2, mean_score=0.8, mean_f1=0.7)

    def test_best_f1_non_empty(self):
        r = self._make_report()
        assert r.best_f1 >= 0.0

    def test_best_f1_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_f1 == pytest.approx(0.0)

    def test_best_pair_none_when_empty(self):
        r = EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)
        assert r.best_pair is None

    def test_best_pair_returns_tuple(self):
        r = self._make_report()
        assert isinstance(r.best_pair, tuple)

    def test_n_pairs_neg_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=-1, mean_score=0.0, mean_f1=0.0)

    def test_mean_score_neg_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=-0.1, mean_f1=0.0)

    def test_mean_f1_neg_raises(self):
        with pytest.raises(ValueError):
            EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=-0.1)


# ─── TestComputePrecision ─────────────────────────────────────────────────────

class TestComputePrecision:
    def test_perfect(self):
        assert compute_precision(5, 0) == pytest.approx(1.0)

    def test_zero(self):
        assert compute_precision(0, 5) == pytest.approx(0.0)

    def test_formula(self):
        assert compute_precision(3, 2) == pytest.approx(3.0 / 5.0)

    def test_both_zero(self):
        assert compute_precision(0, 0) == pytest.approx(0.0)

    def test_tp_neg_raises(self):
        with pytest.raises(ValueError):
            compute_precision(-1, 0)

    def test_fp_neg_raises(self):
        with pytest.raises(ValueError):
            compute_precision(0, -1)


# ─── TestComputeRecall ────────────────────────────────────────────────────────

class TestComputeRecall:
    def test_perfect(self):
        assert compute_recall(5, 0) == pytest.approx(1.0)

    def test_zero(self):
        assert compute_recall(0, 5) == pytest.approx(0.0)

    def test_formula(self):
        assert compute_recall(3, 2) == pytest.approx(3.0 / 5.0)

    def test_both_zero(self):
        assert compute_recall(0, 0) == pytest.approx(0.0)

    def test_tp_neg_raises(self):
        with pytest.raises(ValueError):
            compute_recall(-1, 0)

    def test_fn_neg_raises(self):
        with pytest.raises(ValueError):
            compute_recall(0, -1)


# ─── TestComputeFScore ────────────────────────────────────────────────────────

class TestComputeFScore:
    def test_perfect_f1(self):
        assert compute_f_score(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert compute_f_score(0.0, 1.0) == pytest.approx(0.0)

    def test_zero_recall(self):
        assert compute_f_score(1.0, 0.0) == pytest.approx(0.0)

    def test_both_zero(self):
        assert compute_f_score(0.0, 0.0) == pytest.approx(0.0)

    def test_f1_formula(self):
        # F1 = 2PR/(P+R)
        p, r = 0.6, 0.8
        expected = 2 * p * r / (p + r)
        assert compute_f_score(p, r, beta=1.0) == pytest.approx(expected)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            compute_f_score(0.5, 0.5, beta=0.0)

    def test_result_in_range(self):
        score = compute_f_score(0.7, 0.6)
        assert 0.0 <= score <= 1.0


# ─── TestEvaluateMatch ────────────────────────────────────────────────────────

class TestEvaluateMatch:
    def test_returns_match_eval(self):
        m = evaluate_match((0, 1), 0.8, 5, 1, 2)
        assert isinstance(m, MatchEval)

    def test_pair_stored(self):
        m = evaluate_match((2, 3), 0.5, 3, 0, 1)
        assert m.pair == (2, 3)

    def test_score_stored(self):
        m = evaluate_match((0, 1), 0.75, 4, 2, 1)
        assert m.score == pytest.approx(0.75)

    def test_tp_fp_fn_stored(self):
        m = evaluate_match((0, 1), 0.9, 10, 2, 3)
        assert m.tp == 10
        assert m.fp == 2
        assert m.fn == 3


# ─── TestEvaluateBatchMatches ─────────────────────────────────────────────────

class TestEvaluateBatchMatches:
    def test_returns_list(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.8, 0.6]
        tp_list = [4, 3]
        fp_list = [1, 2]
        fn_list = [0, 1]
        result = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_batch_matches([(0, 1)], [0.5, 0.6], [1], [0], [0])

    def test_empty_input(self):
        result = evaluate_batch_matches([], [], [], [], [])
        assert result == []


# ─── TestAggregateEval ────────────────────────────────────────────────────────

class TestAggregateEval:
    def test_empty_returns_report(self):
        r = aggregate_eval([])
        assert isinstance(r, EvalReport)
        assert r.n_pairs == 0
        assert r.mean_score == pytest.approx(0.0)

    def test_n_pairs_matches(self):
        evals = [MatchEval(pair=(i, i+1), score=0.8, tp=4, fp=1, fn=1)
                 for i in range(3)]
        r = aggregate_eval(evals)
        assert r.n_pairs == 3

    def test_mean_score_correct(self):
        evals = [
            MatchEval(pair=(0, 1), score=0.8),
            MatchEval(pair=(1, 2), score=0.6),
        ]
        r = aggregate_eval(evals)
        assert r.mean_score == pytest.approx(0.7)


# ─── TestFilterByScore ────────────────────────────────────────────────────────

class TestFilterByScore:
    def _evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.9, tp=5),
            MatchEval(pair=(1, 2), score=0.4, tp=2),
            MatchEval(pair=(2, 3), score=0.7, tp=4),
        ]

    def test_threshold_neg_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(self._evals(), -0.1)

    def test_no_filter(self):
        result = filter_by_score(self._evals(), 0.0)
        assert len(result) == 3

    def test_filter_out_low(self):
        result = filter_by_score(self._evals(), 0.7)
        assert all(e.score >= 0.7 for e in result)

    def test_filter_all_out(self):
        result = filter_by_score(self._evals(), 1.1)
        assert result == []


# ─── TestRankMatches ──────────────────────────────────────────────────────────

class TestRankMatches:
    def _evals(self):
        return [
            MatchEval(pair=(0, 1), score=0.4, tp=2, fp=1, fn=0),
            MatchEval(pair=(1, 2), score=0.9, tp=5, fp=0, fn=1),
            MatchEval(pair=(2, 3), score=0.7, tp=4, fp=1, fn=1),
        ]

    def test_rank_by_f1_desc(self):
        ranked = rank_matches(self._evals(), by="f1")
        f1_vals = [e.f1 for e in ranked]
        assert f1_vals == sorted(f1_vals, reverse=True)

    def test_rank_by_score_desc(self):
        ranked = rank_matches(self._evals(), by="score")
        scores = [e.score for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_by_raises(self):
        with pytest.raises(ValueError):
            rank_matches(self._evals(), by="unknown")

    def test_empty_returns_empty(self):
        assert rank_matches([], by="f1") == []
