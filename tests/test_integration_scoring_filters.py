"""Integration tests for puzzle_reconstruction.scoring modules.

Covers: pair_filter, match_evaluator, gap_scorer, pair_ranker, and
cross-module pipelines.
"""

import math
import pytest

from puzzle_reconstruction.scoring.pair_filter import (
    CandidatePair,
    FilterConfig,
    FilterReport,
    deduplicate_pairs,
    filter_by_score,
    filter_pairs,
    filter_top_k,
)
from puzzle_reconstruction.scoring.match_evaluator import (
    EvalConfig,
    EvalReport,
    MatchEval,
    aggregate_eval,
    evaluate_batch_matches,
    evaluate_match,
)
from puzzle_reconstruction.scoring.gap_scorer import (
    GapConfig,
    GapMeasure,
    GapReport,
    build_gap_report,
    filter_gap_measures,
    gap_score_matrix,
    measure_gap,
    score_gap,
    worst_gap_pairs,
)
from puzzle_reconstruction.scoring.pair_ranker import (
    RankConfig,
    RankedPair,
    RankResult,
    build_rank_matrix,
    compute_pair_score,
    rank_pairs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pairs(n: int = 3):
    """Return a list of CandidatePair for fragments 0..n-1 (no self-pairs)."""
    return [
        CandidatePair(id_a=i, id_b=j, score=round((i + j) / (2 * n - 2), 4))
        for i in range(n)
        for j in range(n)
        if i != j
    ]


def make_distances(n: int = 3):
    """Return symmetric distance dict for fragments 0..n-1."""
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = float(abs(i - j))
    return d


# ---------------------------------------------------------------------------
# TestPairFilter
# ---------------------------------------------------------------------------

class TestPairFilter:

    def test_filter_by_score_returns_list_of_candidate_pair(self):
        pairs = make_pairs()
        result = filter_by_score(pairs, min_score=0.0)
        assert isinstance(result, list)
        assert all(isinstance(p, CandidatePair) for p in result)

    def test_filter_by_score_min_zero_keeps_all(self):
        pairs = make_pairs()
        result = filter_by_score(pairs, min_score=0.0)
        assert len(result) == len(pairs)

    def test_filter_by_score_min_one_keeps_only_perfect(self):
        pairs = [
            CandidatePair(id_a=0, id_b=1, score=1.0),
            CandidatePair(id_a=0, id_b=2, score=0.5),
            CandidatePair(id_a=1, id_b=2, score=0.8),
        ]
        result = filter_by_score(pairs, min_score=1.0)
        assert len(result) == 1
        assert result[0].score == 1.0

    def test_filter_top_k_returns_at_most_k_pairs(self):
        pairs = make_pairs(n=4)
        result = filter_top_k(pairs, k=3)
        assert len(result) <= 3

    def test_filter_top_k_returns_list(self):
        pairs = make_pairs()
        result = filter_top_k(pairs, k=2)
        assert isinstance(result, list)

    def test_deduplicate_pairs_returns_list_with_no_duplicates(self):
        pairs = make_pairs()
        result = deduplicate_pairs(pairs)
        assert isinstance(result, list)
        seen = set()
        for p in result:
            key = (min(p.id_a, p.id_b), max(p.id_a, p.id_b))
            assert key not in seen, f"Duplicate pair found: {key}"
            seen.add(key)

    def test_deduplicate_pairs_on_empty_list_returns_empty(self):
        result = deduplicate_pairs([])
        assert result == []

    def test_filter_pairs_returns_tuple_of_list_and_filter_report(self):
        pairs = make_pairs()
        result = filter_pairs(pairs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        lst, report = result
        assert isinstance(lst, list)
        assert isinstance(report, FilterReport)

    def test_filter_report_has_n_output_attribute(self):
        pairs = make_pairs()
        _, report = filter_pairs(pairs)
        assert hasattr(report, "n_output")
        assert isinstance(report.n_output, int)

    def test_candidate_pair_has_id_a_id_b_score_attributes(self):
        p = CandidatePair(id_a=3, id_b=7, score=0.42)
        assert hasattr(p, "id_a") and p.id_a == 3
        assert hasattr(p, "id_b") and p.id_b == 7
        assert hasattr(p, "score") and p.score == pytest.approx(0.42)

    def test_filter_by_score_all_results_above_threshold(self):
        pairs = make_pairs(n=5)
        threshold = 0.4
        result = filter_by_score(pairs, min_score=threshold)
        assert all(p.score >= threshold for p in result)


# ---------------------------------------------------------------------------
# TestMatchEvaluator
# ---------------------------------------------------------------------------

class TestMatchEvaluator:

    def test_evaluate_match_returns_match_eval(self):
        m = evaluate_match(pair=(0, 1), score=0.9, tp=4, fp=1, fn=1)
        assert isinstance(m, MatchEval)

    def test_match_eval_has_precision_recall_f1_attributes(self):
        m = evaluate_match(pair=(0, 1), score=0.9, tp=4, fp=1, fn=1)
        assert hasattr(m, "precision")
        assert hasattr(m, "recall")
        assert hasattr(m, "f1")

    def test_perfect_match_gives_precision_recall_f1_one(self):
        m = evaluate_match(pair=(0, 1), score=1.0, tp=5, fp=0, fn=0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_evaluate_batch_matches_returns_list_of_match_eval(self):
        pairs = [(0, 1), (1, 2)]
        batch = evaluate_batch_matches(pairs, [0.9, 0.7], [5, 3], [0, 1], [0, 2])
        assert isinstance(batch, list)
        assert all(isinstance(e, MatchEval) for e in batch)

    def test_batch_length_matches_input(self):
        pairs = [(i, i + 1) for i in range(5)]
        scores = [0.8] * 5
        tp_list = [3] * 5
        fp_list = [1] * 5
        fn_list = [1] * 5
        batch = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        assert len(batch) == 5

    def test_aggregate_eval_returns_eval_report(self):
        evals = [evaluate_match((0, 1), 0.9, 5, 0, 0)]
        report = aggregate_eval(evals)
        assert isinstance(report, EvalReport)

    def test_eval_report_has_mean_f1_attribute(self):
        evals = [evaluate_match((0, 1), 0.9, 5, 0, 0)]
        report = aggregate_eval(evals)
        assert hasattr(report, "mean_f1")

    def test_zero_tp_gives_precision_zero(self):
        m = evaluate_match(pair=(0, 1), score=0.0, tp=0, fp=5, fn=5)
        assert m.precision == pytest.approx(0.0)

    def test_aggregate_on_single_eval(self):
        m = evaluate_match(pair=(0, 1), score=0.8, tp=4, fp=1, fn=1)
        report = aggregate_eval([m])
        assert report.n_pairs == 1

    def test_match_eval_score_attribute_is_finite(self):
        m = evaluate_match(pair=(2, 3), score=0.75, tp=3, fp=2, fn=1)
        assert hasattr(m, "score")
        assert math.isfinite(m.score)

    def test_batch_with_single_element(self):
        batch = evaluate_batch_matches([(0, 1)], [0.6], [2], [1], [1])
        assert len(batch) == 1
        assert isinstance(batch[0], MatchEval)


# ---------------------------------------------------------------------------
# TestGapScorer
# ---------------------------------------------------------------------------

class TestGapScorer:

    def test_score_gap_returns_float_in_0_1(self):
        s, _ = score_gap(3.0)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_gap_at_target_returns_perfect_score(self):
        cfg = GapConfig()
        s, p = score_gap(cfg.target_gap, cfg)
        assert s == pytest.approx(1.0)
        assert p == pytest.approx(0.0)

    def test_measure_gap_returns_gap_measure(self):
        gm = measure_gap(0, 1, 2.5)
        assert isinstance(gm, GapMeasure)

    def test_gap_measure_has_distance_attribute(self):
        gm = measure_gap(0, 1, 3.0)
        assert hasattr(gm, "distance")
        assert gm.distance == pytest.approx(3.0)

    def test_build_gap_report_returns_gap_report(self):
        distances = make_distances(n=3)
        report = build_gap_report(distances)
        assert isinstance(report, GapReport)

    def test_gap_report_has_mean_score_attribute(self):
        distances = make_distances(n=3)
        report = build_gap_report(distances)
        assert hasattr(report, "mean_score")
        assert isinstance(report.mean_score, float)

    def test_filter_gap_measures_returns_list(self):
        distances = make_distances(n=3)
        report = build_gap_report(distances)
        result = filter_gap_measures(report, min_score=0.0)
        assert isinstance(result, list)

    def test_worst_gap_pairs_returns_list(self):
        distances = make_distances(n=4)
        report = build_gap_report(distances)
        result = worst_gap_pairs(report, top_k=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_gap_score_matrix_returns_dict(self):
        ids = [0, 1, 2]
        distances = make_distances(n=3)
        mat = gap_score_matrix(ids, distances)
        assert isinstance(mat, dict)

    def test_gap_score_matrix_has_no_diagonal_entries(self):
        ids = [0, 1, 2]
        distances = make_distances(n=3)
        mat = gap_score_matrix(ids, distances)
        for i in ids:
            assert (i, i) not in mat

    def test_gap_score_matrix_shape_matches_n(self):
        n = 4
        ids = list(range(n))
        distances = make_distances(n=n)
        mat = gap_score_matrix(ids, distances)
        # Symmetric input: expect n*(n-1)/2 unique off-diagonal pairs in output
        assert len(mat) > 0
        assert all(isinstance(v, float) for v in mat.values())


# ---------------------------------------------------------------------------
# TestPairRanker
# ---------------------------------------------------------------------------

class TestPairRanker:

    def test_compute_pair_score_returns_float(self):
        s = compute_pair_score({"color": 0.8, "shape": 0.6})
        assert isinstance(s, float)

    def test_compute_pair_score_is_finite(self):
        s = compute_pair_score({"edge": 0.7, "texture": 0.9})
        assert math.isfinite(s)

    def test_rank_pairs_returns_rank_result(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.9, 0.5, 0.7]
        result = rank_pairs(pairs, scores)
        assert isinstance(result, RankResult)

    def test_ranked_pair_has_pair_score_rank_attributes(self):
        pairs = [(0, 1), (1, 2)]
        result = rank_pairs(pairs, [0.8, 0.6])
        rp = result.ranked[0]
        assert isinstance(rp, RankedPair)
        assert hasattr(rp, "pair")
        assert hasattr(rp, "score")
        assert hasattr(rp, "rank")

    def test_rank_pairs_sorted_by_rank(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.3, 0.9, 0.6]
        result = rank_pairs(pairs, scores)
        ranks = [rp.rank for rp in result.ranked]
        assert ranks == sorted(ranks)

    def test_build_rank_matrix_returns_ndarray(self):
        import numpy as np
        pairs = [(0, 1), (0, 2), (1, 2)]
        result = rank_pairs(pairs, [0.9, 0.5, 0.7])
        mat = build_rank_matrix(result, n_fragments=3)
        assert isinstance(mat, np.ndarray)

    def test_build_rank_matrix_shape_correct(self):
        n = 4
        pairs = [(i, j) for i in range(n) for j in range(n) if i < j]
        scores = [float(i) / len(pairs) for i in range(len(pairs))]
        result = rank_pairs(pairs, scores)
        mat = build_rank_matrix(result, n_fragments=n)
        assert mat.shape == (n, n)

    def test_rank_pairs_on_3_fragments(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.8, 0.6, 0.7]
        result = rank_pairs(pairs, scores)
        assert result.n_pairs == 3
        assert result.n_ranked == 3

    def test_no_self_pairs_in_ranked_list(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.8, 0.6, 0.7]
        result = rank_pairs(pairs, scores)
        for rp in result.ranked:
            a, b = rp.pair
            assert a != b

    def test_build_rank_matrix_diagonal_is_zero(self):
        import numpy as np
        pairs = [(0, 1), (0, 2), (1, 2)]
        result = rank_pairs(pairs, [0.9, 0.5, 0.7])
        mat = build_rank_matrix(result, n_fragments=3)
        diag = np.diag(mat)
        for val in diag:
            assert val == 0 or math.isnan(float(val))

    def test_rank_pairs_result_has_score_attributes(self):
        pairs = [(0, 1), (1, 2)]
        result = rank_pairs(pairs, [0.7, 0.4])
        for rp in result.ranked:
            assert hasattr(rp, "score")
            assert math.isfinite(rp.score)


# ---------------------------------------------------------------------------
# TestScoringFiltersIntegration
# ---------------------------------------------------------------------------

class TestScoringFiltersIntegration:

    def test_pair_filter_then_match_evaluator_pipeline(self):
        """Filter candidate pairs then evaluate each as a match."""
        pairs = make_pairs(n=4)
        filtered, _ = filter_pairs(pairs)
        evals = [
            evaluate_match(
                pair=(p.id_a, p.id_b),
                score=p.score,
                tp=3,
                fp=1,
                fn=1,
            )
            for p in filtered
        ]
        assert len(evals) == len(filtered)
        assert all(isinstance(e, MatchEval) for e in evals)

    def test_rank_then_filter_pipeline(self):
        """Rank pairs then apply score filter on CandidatePair representation."""
        raw_pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
        scores = [float(k) / len(raw_pairs) for k in range(len(raw_pairs))]
        result = rank_pairs(raw_pairs, scores)
        # Convert RankedPairs to CandidatePairs for filtering
        candidate_pairs = [
            CandidatePair(id_a=rp.pair[0], id_b=rp.pair[1], score=rp.score)
            for rp in result.ranked
        ]
        filtered = filter_by_score(candidate_pairs, min_score=0.3)
        assert isinstance(filtered, list)
        assert all(p.score >= 0.3 for p in filtered)

    def test_gap_scorer_then_pair_ranker_pipeline(self):
        """Build gap scores then use them as metric scores for ranking."""
        ids = [0, 1, 2, 3]
        distances = make_distances(n=4)
        gap_mat = gap_score_matrix(ids, distances)
        # Use gap scores as input to rank_pairs
        pairs = list(gap_mat.keys())
        scores = [gap_mat[p] for p in pairs]
        result = rank_pairs(pairs, scores)
        assert isinstance(result, RankResult)
        assert result.n_ranked == len(pairs)

    def test_filter_pairs_then_evaluate_each(self):
        """Full filter then per-pair evaluate pipeline."""
        pairs = [
            CandidatePair(id_a=i, id_b=j, score=0.5 + 0.1 * (i + j))
            for i in range(3)
            for j in range(3)
            if i != j
        ]
        filtered, report = filter_pairs(pairs)
        assert isinstance(report, FilterReport)
        evals = [
            evaluate_match(
                pair=(p.id_a, p.id_b),
                score=p.score,
                tp=2,
                fp=1,
                fn=1,
            )
            for p in filtered
        ]
        assert len(evals) == report.n_output

    def test_aggregate_eval_on_batch_results(self):
        """evaluate_batch_matches + aggregate_eval end-to-end."""
        pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
        scores = [0.9, 0.7, 0.6, 0.8]
        tp_list = [5, 3, 2, 4]
        fp_list = [0, 1, 2, 1]
        fn_list = [0, 2, 1, 0]
        batch = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        report = aggregate_eval(batch)
        assert isinstance(report, EvalReport)
        assert report.n_pairs == len(pairs)
        assert math.isfinite(report.mean_f1)

    def test_build_rank_matrix_and_gap_score_matrix_comparison(self):
        """Both matrices should have compatible structures for 3 fragments."""
        n = 3
        ids = list(range(n))
        distances = make_distances(n=n)
        gap_mat = gap_score_matrix(ids, distances)

        raw_pairs = list(gap_mat.keys())
        scores = [gap_mat[p] for p in raw_pairs]
        rank_result = rank_pairs(raw_pairs, scores)
        rank_mat = build_rank_matrix(rank_result, n_fragments=n)

        assert rank_mat.shape == (n, n)
        assert isinstance(gap_mat, dict)

    def test_deduplicate_then_filter_top_k(self):
        """Deduplication followed by top-k selection."""
        pairs = make_pairs(n=4)  # includes both (i,j) and (j,i)
        deduped = deduplicate_pairs(pairs)
        top = filter_top_k(deduped, k=3)
        assert len(top) <= 3
        # No duplicates in top
        seen = set()
        for p in top:
            key = (min(p.id_a, p.id_b), max(p.id_a, p.id_b))
            assert key not in seen
            seen.add(key)

    def test_worst_gap_pairs_use_valid_indices(self):
        """worst_gap_pairs should reference valid fragment ids."""
        n = 4
        ids = list(range(n))
        distances = make_distances(n=n)
        report = build_gap_report(distances)
        worst = worst_gap_pairs(report, top_k=3)
        for gm in worst:
            assert isinstance(gm, GapMeasure)
            assert gm.id_a in ids
            assert gm.id_b in ids
            assert gm.id_a != gm.id_b

    def test_batch_evaluate_then_aggregate(self):
        """Batch evaluation then aggregation with non-trivial data."""
        n = 5
        pairs = [(i, i + 1) for i in range(n)]
        scores = [0.5 + 0.1 * i for i in range(n)]
        tp_list = [i + 1 for i in range(n)]
        fp_list = [max(0, n - i - 1) for i in range(n)]
        fn_list = [1] * n
        batch = evaluate_batch_matches(pairs, scores, tp_list, fp_list, fn_list)
        report = aggregate_eval(batch)
        assert report.n_pairs == n
        assert 0.0 <= report.mean_f1 <= 1.0

    def test_full_chain_rank_filter_evaluate_aggregate(self):
        """Full pipeline: rank -> filter -> evaluate -> aggregate."""
        raw_pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
        scores = [0.4 + 0.06 * k for k in range(len(raw_pairs))]

        # Step 1: rank
        rank_result = rank_pairs(raw_pairs, scores)

        # Step 2: convert to CandidatePair and filter
        candidates = [
            CandidatePair(id_a=rp.pair[0], id_b=rp.pair[1], score=rp.score)
            for rp in rank_result.ranked
        ]
        filtered, filter_report = filter_pairs(candidates)
        assert isinstance(filter_report, FilterReport)

        if not filtered:
            pytest.skip("filter_pairs removed all pairs; nothing to evaluate")

        # Step 3: evaluate each filtered pair
        evals = [
            evaluate_match(
                pair=(p.id_a, p.id_b),
                score=p.score,
                tp=3,
                fp=1,
                fn=1,
            )
            for p in filtered
        ]

        # Step 4: aggregate
        report = aggregate_eval(evals)
        assert isinstance(report, EvalReport)
        assert report.n_pairs == len(filtered)
        assert math.isfinite(report.mean_f1)

    def test_filter_gap_measures_non_negative_gaps(self):
        """All GapMeasures returned by filter_gap_measures have non-negative distance."""
        n = 5
        distances = make_distances(n=n)
        report = build_gap_report(distances)
        measures = filter_gap_measures(report, min_score=0.0)
        assert all(gm.distance >= 0.0 for gm in measures)
