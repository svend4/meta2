"""
Integration tests for puzzle_reconstruction scoring and verification modules.

Covers:
  - scoring/boundary_scorer.py
  - scoring/gap_scorer.py
  - scoring/global_ranker.py
  - scoring/match_evaluator.py
  - scoring/rank_fusion.py
  - scoring/score_normalizer.py
  - scoring/threshold_selector.py
  - verification/color_continuity_verifier.py
  - verification/completeness_checker.py
  - verification/statistical_coherence.py
  - verification/text_coherence.py
"""
import unittest
import numpy as np

# ─── Scoring imports ──────────────────────────────────────────────────────────
from puzzle_reconstruction.scoring.boundary_scorer import (
    BoundarySide,
    BoundaryScore,
    ScoringConfig,
    intensity_compatibility,
    gradient_compatibility,
    color_compatibility,
    score_boundary,
    score_matrix,
    batch_score_boundaries,
)
from puzzle_reconstruction.scoring.gap_scorer import (
    GapConfig,
    GapMeasure,
    GapReport,
    score_gap,
    measure_gap,
    build_gap_report,
    filter_gap_measures,
    worst_gap_pairs,
    gap_score_matrix,
)
from puzzle_reconstruction.scoring.global_ranker import (
    RankedPair,
    RankingConfig,
    normalize_matrix,
    aggregate_score_matrices,
    rank_pairs,
    top_k_candidates,
    global_rank,
    score_vector,
    batch_global_rank,
)
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
from puzzle_reconstruction.scoring.rank_fusion import (
    normalize_scores,
    reciprocal_rank_fusion,
    borda_count,
    score_fusion,
    fuse_rankings,
)
from puzzle_reconstruction.scoring.score_normalizer import (
    NormMethod,
    NormalizedMatrix,
    minmax_normalize_matrix,
    zscore_normalize_matrix,
    rank_normalize_matrix,
    softmax_normalize_matrix,
    sigmoid_normalize_matrix,
    normalize_score_matrix,
    combine_score_matrices,
    batch_normalize_matrices,
)
from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    select_fixed_threshold,
    select_percentile_threshold,
    select_otsu_threshold,
    select_f1_threshold,
    select_adaptive_threshold,
    select_threshold,
    apply_threshold,
    batch_select_thresholds,
)

# ─── Verification imports ─────────────────────────────────────────────────────
from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    ColorContinuityVerifier,
    verify_color_continuity,
)
from puzzle_reconstruction.verification.completeness_checker import (
    CompletenessReport,
    check_fragment_coverage,
    find_missing_fragments,
    check_spatial_coverage,
    find_uncovered_regions,
    completeness_score,
    generate_completeness_report,
    batch_check_coverage,
)
from puzzle_reconstruction.verification.statistical_coherence import (
    StatisticalCoherenceConfig,
    StatisticalCoherenceResult,
    StatisticalCoherenceVerifier,
    cohere_score,
)
from puzzle_reconstruction.verification.text_coherence import (
    NGramModel,
    TextCoherenceScorer,
    seam_bigram_score,
    word_boundary_score,
    build_ngram_model,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _gray_img(h: int = 40, w: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color_img(h: int = 40, w: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _uniform_img(value: int, h: int = 40, w: int = 40, channels: int = 3) -> np.ndarray:
    return np.full((h, w, channels), value, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BoundaryScorer
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryScorer(unittest.TestCase):

    def test_identical_images_same_side_score(self):
        img = _color_img(40, 40, seed=1)
        # Comparing the same strip (RIGHT to RIGHT) gives intensity_diff = 1.0
        bs = score_boundary(img, img, side1=BoundarySide.RIGHT, side2=BoundarySide.RIGHT)
        self.assertAlmostEqual(bs.intensity_diff, 1.0, places=5)
        self.assertGreater(bs.aggregate, 0.7)
        self.assertLessEqual(bs.aggregate, 1.0)

    def test_completely_different_images_lower_score(self):
        img1 = _uniform_img(0)
        img2 = _uniform_img(255)
        bs = score_boundary(img1, img2)
        # intensity_diff for black vs white should be very low (1 - 1.0 MAE = 0)
        self.assertAlmostEqual(bs.intensity_diff, 0.0, places=5)

    def test_boundary_score_fields_in_range(self):
        img1, img2 = _color_img(seed=10), _color_img(seed=11)
        bs = score_boundary(img1, img2)
        for field_val in (bs.intensity_diff, bs.gradient_score, bs.color_score, bs.aggregate):
            self.assertGreaterEqual(field_val, 0.0)
            self.assertLessEqual(field_val, 1.0)

    def test_boundary_score_sides(self):
        img1, img2 = _color_img(seed=20), _color_img(seed=21)
        bs = score_boundary(img1, img2, side1=BoundarySide.TOP, side2=BoundarySide.BOTTOM)
        self.assertEqual(bs.side1, BoundarySide.TOP)
        self.assertEqual(bs.side2, BoundarySide.BOTTOM)

    def test_scoring_config_invalid_strip_width(self):
        with self.assertRaises(ValueError):
            ScoringConfig(strip_width=0)

    def test_scoring_config_negative_weight(self):
        with self.assertRaises(ValueError):
            ScoringConfig(w_intensity=-0.1)

    def test_score_matrix_shape(self):
        imgs = [_color_img(seed=i) for i in range(4)]
        mat = score_matrix(imgs)
        self.assertEqual(mat.shape, (4, 4))
        np.testing.assert_array_equal(np.diag(mat), 0.0)

    def test_batch_score_boundaries_length(self):
        pairs = [(_color_img(seed=i), _color_img(seed=i + 10)) for i in range(5)]
        results = batch_score_boundaries(pairs)
        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0], BoundaryScore)

    def test_intensity_compatibility_identical(self):
        strip = _gray_img(4, 40, seed=5)
        score = intensity_compatibility(strip, strip)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_intensity_compatibility_shape_mismatch(self):
        s1 = _gray_img(4, 40)
        s2 = _gray_img(5, 40)
        with self.assertRaises(ValueError):
            intensity_compatibility(s1, s2)

    def test_gradient_compatibility_range(self):
        s1 = _gray_img(4, 40, seed=7)
        s2 = _gray_img(4, 40, seed=8)
        score = gradient_compatibility(s1, s2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_color_compatibility_identical_strips(self):
        strip = _color_img(4, 40, seed=9)
        score = color_compatibility(strip, strip)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_boundary_side_opposite(self):
        self.assertEqual(BoundarySide.TOP.opposite(), BoundarySide.BOTTOM)
        self.assertEqual(BoundarySide.LEFT.opposite(), BoundarySide.RIGHT)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GapScorer
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapScorer(unittest.TestCase):

    def test_score_gap_at_target(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        score, penalty = score_gap(5.0, cfg)
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(penalty, 0.0)

    def test_score_gap_within_tolerance(self):
        cfg = GapConfig(target_gap=5.0, tolerance=2.0)
        score, penalty = score_gap(6.0, cfg)
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(penalty, 0.0)

    def test_score_gap_negative_raises(self):
        with self.assertRaises(ValueError):
            score_gap(-1.0)

    def test_score_gap_beyond_max(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0, max_gap=20.0)
        score, penalty = score_gap(25.0, cfg)
        self.assertAlmostEqual(score, 0.0)
        self.assertGreater(penalty, 0.0)

    def test_gap_config_invalid_target(self):
        with self.assertRaises(ValueError):
            GapConfig(target_gap=-1.0)

    def test_gap_config_max_le_target(self):
        with self.assertRaises(ValueError):
            GapConfig(target_gap=10.0, max_gap=5.0)

    def test_measure_gap_creates_correct_object(self):
        m = measure_gap(0, 1, 5.0)
        self.assertEqual(m.id_a, 0)
        self.assertEqual(m.id_b, 1)
        self.assertAlmostEqual(m.score, 1.0)

    def test_build_gap_report_empty(self):
        report = build_gap_report({})
        self.assertEqual(report.n_pairs, 0)
        self.assertAlmostEqual(report.mean_score, 0.0)

    def test_build_gap_report_values(self):
        distances = {(0, 1): 5.0, (1, 2): 5.0, (2, 3): 30.0}
        report = build_gap_report(distances)
        self.assertEqual(report.n_pairs, 3)
        self.assertGreater(report.n_acceptable, 0)
        self.assertGreaterEqual(report.total_penalty, 0.0)

    def test_filter_gap_measures_invalid_min_score(self):
        report = build_gap_report({(0, 1): 5.0})
        with self.assertRaises(ValueError):
            filter_gap_measures(report, min_score=1.5)

    def test_worst_gap_pairs_top_k(self):
        distances = {(i, i + 1): float(i * 5) for i in range(5)}
        report = build_gap_report(distances)
        worst = worst_gap_pairs(report, top_k=2)
        self.assertEqual(len(worst), 2)
        self.assertGreaterEqual(worst[0].penalty, worst[1].penalty)

    def test_worst_gap_pairs_invalid_top_k(self):
        report = build_gap_report({(0, 1): 5.0})
        with self.assertRaises(ValueError):
            worst_gap_pairs(report, top_k=0)

    def test_gap_measure_pair_key(self):
        m = GapMeasure(id_a=3, id_b=1, distance=5.0, score=1.0, penalty=0.0)
        self.assertEqual(m.pair_key, (1, 3))

    def test_gap_score_matrix_keys(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        self.assertIn((0, 1), result)
        self.assertIn((1, 2), result)
        self.assertNotIn((2, 1), result)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GlobalRanker
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalRanker(unittest.TestCase):

    def _make_matrix(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = rng.random((n, n))
        np.fill_diagonal(m, 0.0)
        return m

    def test_normalize_matrix_range(self):
        m = self._make_matrix(5)
        nm = normalize_matrix(m)
        mask = ~np.eye(5, dtype=bool)
        self.assertAlmostEqual(nm[mask].min(), 0.0, places=6)
        self.assertAlmostEqual(nm[mask].max(), 1.0, places=6)
        np.testing.assert_array_equal(np.diag(nm), 0.0)

    def test_normalize_matrix_constant(self):
        m = np.full((4, 4), 0.5)
        np.fill_diagonal(m, 0.0)
        nm = normalize_matrix(m)
        # All off-diagonal values are equal → normalised to 0
        mask = ~np.eye(4, dtype=bool)
        np.testing.assert_array_equal(nm[mask], 0.0)

    def test_normalize_matrix_non_square_raises(self):
        with self.assertRaises(ValueError):
            normalize_matrix(np.ones((3, 4)))

    def test_aggregate_score_matrices_empty_raises(self):
        with self.assertRaises(ValueError):
            aggregate_score_matrices({})

    def test_aggregate_score_matrices_shape_mismatch(self):
        m1 = self._make_matrix(3)
        m2 = self._make_matrix(4)
        with self.assertRaises(ValueError):
            aggregate_score_matrices({"a": m1, "b": m2})

    def test_aggregate_score_matrices_result_shape(self):
        mats = {"boundary": self._make_matrix(4), "sift": self._make_matrix(4, seed=1)}
        agg = aggregate_score_matrices(mats)
        self.assertEqual(agg.shape, (4, 4))
        np.testing.assert_array_equal(np.diag(agg), 0.0)

    def test_rank_pairs_sorted_descending(self):
        m = self._make_matrix(5)
        ranked = rank_pairs(m)
        scores = [rp.score for rp in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_pairs_non_square_raises(self):
        with self.assertRaises(ValueError):
            rank_pairs(np.ones((3, 4)))

    def test_top_k_candidates_limits(self):
        m = self._make_matrix(5)
        ranked = rank_pairs(m)
        cands = top_k_candidates(ranked, n_fragments=5, k=2)
        for lst in cands.values():
            self.assertLessEqual(len(lst), 2)

    def test_top_k_candidates_invalid_k(self):
        with self.assertRaises(ValueError):
            top_k_candidates([], n_fragments=3, k=0)

    def test_global_rank_returns_list(self):
        mats = {"boundary": self._make_matrix(4)}
        result = global_rank(mats)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], RankedPair)

    def test_score_vector_shape(self):
        m = self._make_matrix(5)
        ranked = rank_pairs(m)
        sv = score_vector(ranked, n_fragments=5)
        self.assertEqual(sv.shape, (5,))
        self.assertTrue(np.all(sv >= 0.0))

    def test_batch_global_rank_multiple_groups(self):
        groups = [{"boundary": self._make_matrix(4)}, {"sift": self._make_matrix(4, seed=2)}]
        results = batch_global_rank(groups)
        self.assertEqual(len(results), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MatchEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatchEvaluator(unittest.TestCase):

    def test_compute_precision_basic(self):
        self.assertAlmostEqual(compute_precision(3, 1), 0.75)

    def test_compute_precision_zero_denom(self):
        self.assertAlmostEqual(compute_precision(0, 0), 0.0)

    def test_compute_precision_negative_raises(self):
        with self.assertRaises(ValueError):
            compute_precision(-1, 0)

    def test_compute_recall_basic(self):
        self.assertAlmostEqual(compute_recall(4, 1), 0.8)

    def test_compute_recall_zero_denom(self):
        self.assertAlmostEqual(compute_recall(0, 0), 0.0)

    def test_compute_f_score_f1(self):
        p, r = 0.8, 0.6
        f1 = compute_f_score(p, r, beta=1.0)
        expected = 2 * p * r / (p + r)
        self.assertAlmostEqual(f1, expected, places=6)

    def test_compute_f_score_beta_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_f_score(0.5, 0.5, beta=0.0)

    def test_evaluate_match_object(self):
        me = evaluate_match((0, 1), score=0.7, tp=5, fp=2, fn=1)
        self.assertAlmostEqual(me.precision, 5 / 7)
        self.assertAlmostEqual(me.recall, 5 / 6)

    def test_evaluate_batch_matches_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate_batch_matches(
                pairs=[(0, 1), (1, 2)],
                scores=[0.5],
                tp_list=[1, 1],
                fp_list=[0, 0],
                fn_list=[0, 0],
            )

    def test_aggregate_eval_empty(self):
        report = aggregate_eval([])
        self.assertEqual(report.n_pairs, 0)
        self.assertAlmostEqual(report.mean_f1, 0.0)

    def test_aggregate_eval_values(self):
        evals = [evaluate_match((i, i+1), 0.5, tp=4, fp=1, fn=1) for i in range(3)]
        report = aggregate_eval(evals)
        self.assertEqual(report.n_pairs, 3)
        self.assertGreater(report.mean_f1, 0.0)
        self.assertIsNotNone(report.best_pair)

    def test_filter_by_score_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            filter_by_score([], threshold=-0.1)

    def test_rank_matches_by_f1(self):
        evals = [
            evaluate_match((0, 1), 0.9, tp=9, fp=1, fn=1),
            evaluate_match((1, 2), 0.3, tp=1, fp=5, fn=5),
        ]
        ranked = rank_matches(evals, by="f1")
        self.assertGreater(ranked[0].f1, ranked[1].f1)

    def test_rank_matches_invalid_by_raises(self):
        with self.assertRaises(ValueError):
            rank_matches([], by="unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RankFusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankFusion(unittest.TestCase):

    def test_normalize_scores_empty_raises(self):
        with self.assertRaises(ValueError):
            normalize_scores([])

    def test_normalize_scores_range(self):
        result = normalize_scores([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(min(result), 0.0)
        self.assertAlmostEqual(max(result), 1.0)

    def test_normalize_scores_all_equal(self):
        result = normalize_scores([5.0, 5.0, 5.0])
        self.assertEqual(result, [1.0, 1.0, 1.0])

    def test_rrf_basic_consistency(self):
        lists = [[0, 1, 2], [1, 0, 2]]
        result = reciprocal_rank_fusion(lists)
        ids = [item_id for item_id, _ in result]
        # item 1 is top-ranked in first list (position 2), top in second list
        # item 0 is top-ranked in first list — both should appear
        self.assertIn(0, ids)
        self.assertIn(1, ids)

    def test_rrf_sorted_descending(self):
        lists = [[3, 1, 2], [1, 2, 3]]
        result = reciprocal_rank_fusion(lists)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[0, 1]], k=0)

    def test_rrf_empty_lists_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([])

    def test_borda_count_sorted_descending(self):
        lists = [[2, 0, 1], [0, 2, 1]]
        result = borda_count(lists)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_borda_count_empty_raises(self):
        with self.assertRaises(ValueError):
            borda_count([])

    def test_score_fusion_basic(self):
        sl1 = [(0, 0.9), (1, 0.5), (2, 0.3)]
        sl2 = [(1, 0.8), (0, 0.4), (2, 0.2)]
        result = score_fusion([sl1, sl2])
        ids = [item_id for item_id, _ in result]
        self.assertIn(0, ids)
        self.assertIn(1, ids)

    def test_score_fusion_weights_mismatch_raises(self):
        sl = [(0, 0.5)]
        with self.assertRaises(ValueError):
            score_fusion([sl, sl], weights=[1.0])

    def test_fuse_rankings_rrf(self):
        lists = [[0, 1, 2], [2, 0, 1]]
        result = fuse_rankings(lists, method="rrf")
        self.assertIsInstance(result, list)

    def test_fuse_rankings_borda(self):
        lists = [[0, 1, 2], [2, 0, 1]]
        result = fuse_rankings(lists, method="borda")
        self.assertIsInstance(result, list)

    def test_fuse_rankings_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            fuse_rankings([[0, 1]], method="unknown_method")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ScoreNormalizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreNormalizer(unittest.TestCase):

    def _mat(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random((5, 5))

    def test_minmax_range(self):
        m = self._mat()
        nm = minmax_normalize_matrix(m)
        self.assertAlmostEqual(nm.data.min(), 0.0, places=6)
        self.assertAlmostEqual(nm.data.max(), 1.0, places=6)
        self.assertEqual(nm.method, "minmax")

    def test_minmax_non_2d_raises(self):
        with self.assertRaises(ValueError):
            minmax_normalize_matrix(np.ones((3, 3, 3)))

    def test_zscore_zero_mean(self):
        m = self._mat(seed=1)
        nm = zscore_normalize_matrix(m)
        self.assertAlmostEqual(float(nm.data.mean()), 0.0, places=6)
        self.assertEqual(nm.method, "zscore")

    def test_rank_normalize_range(self):
        m = self._mat(seed=2)
        nm = rank_normalize_matrix(m)
        self.assertAlmostEqual(nm.data.min(), 0.0, places=6)
        self.assertAlmostEqual(nm.data.max(), 1.0, places=6)

    def test_softmax_normalize_sums_to_one_global(self):
        m = self._mat(seed=3)
        nm = softmax_normalize_matrix(m, axis=None)
        self.assertAlmostEqual(float(nm.data.sum()), 1.0, places=6)

    def test_softmax_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            softmax_normalize_matrix(np.ones((3, 3)), axis=2)

    def test_softmax_invalid_temperature_raises(self):
        with self.assertRaises(ValueError):
            softmax_normalize_matrix(np.ones((3, 3)), temperature=0.0)

    def test_sigmoid_normalize_range(self):
        m = np.array([[-10.0, 0.0], [10.0, 5.0]])
        nm = sigmoid_normalize_matrix(m)
        self.assertTrue(np.all(nm.data > 0.0))
        self.assertTrue(np.all(nm.data < 1.0))

    def test_normalize_score_matrix_default_minmax(self):
        m = self._mat()
        nm = normalize_score_matrix(m)
        self.assertEqual(nm.method, "minmax")

    def test_normalize_score_matrix_zscore_method(self):
        m = self._mat(seed=4)
        nm = normalize_score_matrix(m, NormMethod(method="zscore"))
        self.assertEqual(nm.method, "zscore")

    def test_combine_score_matrices_weighted(self):
        m1 = np.full((3, 3), 1.0)
        m2 = np.full((3, 3), 0.0)
        combined = combine_score_matrices([m1, m2], weights=[0.8, 0.2])
        expected = (0.8 * 1.0 + 0.2 * 0.0) / 1.0
        self.assertAlmostEqual(float(combined[0, 0]), expected, places=6)

    def test_combine_score_matrices_empty_raises(self):
        with self.assertRaises(ValueError):
            combine_score_matrices([])

    def test_combine_score_matrices_negative_weight_raises(self):
        m = np.ones((2, 2))
        with self.assertRaises(ValueError):
            combine_score_matrices([m, m], weights=[1.0, -0.5])

    def test_batch_normalize_matrices_length(self):
        mats = [self._mat(seed=i) for i in range(3)]
        results = batch_normalize_matrices(mats)
        self.assertEqual(len(results), 3)

    def test_norm_method_invalid_raises(self):
        with self.assertRaises(ValueError):
            NormMethod(method="invalid_method")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ThresholdSelector
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdSelector(unittest.TestCase):

    def _scores(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random(100)

    def test_select_fixed_threshold_value(self):
        scores = self._scores()
        result = select_fixed_threshold(scores, value=0.5)
        self.assertAlmostEqual(result.threshold, 0.5)
        self.assertEqual(result.method, "fixed")
        self.assertEqual(result.n_total, 100)

    def test_select_fixed_threshold_empty_raises(self):
        with self.assertRaises(ValueError):
            select_fixed_threshold(np.array([]))

    def test_select_fixed_threshold_negative_value_raises(self):
        with self.assertRaises(ValueError):
            select_fixed_threshold(self._scores(), value=-0.1)

    def test_select_percentile_threshold_median(self):
        scores = np.arange(10, dtype=float)
        result = select_percentile_threshold(scores, percentile=50.0)
        self.assertAlmostEqual(result.threshold, float(np.percentile(scores, 50.0)))

    def test_select_percentile_threshold_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            select_percentile_threshold(self._scores(), percentile=110.0)

    def test_select_otsu_threshold_returns_result(self):
        rng = np.random.default_rng(99)
        # Bimodal distribution for Otsu to work nicely
        low = rng.random(50) * 0.3
        high = rng.random(50) * 0.3 + 0.7
        scores = np.concatenate([low, high])
        result = select_otsu_threshold(scores)
        self.assertGreater(result.threshold, 0.3)
        self.assertLess(result.threshold, 0.7)

    def test_select_f1_threshold_returns_result(self):
        scores = np.array([0.1, 0.4, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = select_f1_threshold(scores, labels)
        self.assertEqual(result.method, "f1")

    def test_select_f1_threshold_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            select_f1_threshold(np.array([0.5, 0.6]), np.array([0, 1, 1]))

    def test_select_adaptive_threshold_returns_result(self):
        result = select_adaptive_threshold(self._scores())
        self.assertEqual(result.method, "adaptive")
        self.assertGreaterEqual(result.threshold, 0.0)

    def test_select_threshold_fixed_config(self):
        cfg = ThresholdConfig(method="fixed", fixed_value=0.3)
        result = select_threshold(self._scores(), cfg)
        self.assertAlmostEqual(result.threshold, 0.3)

    def test_select_threshold_f1_without_labels_raises(self):
        cfg = ThresholdConfig(method="f1")
        with self.assertRaises(ValueError):
            select_threshold(self._scores(), cfg)

    def test_apply_threshold_boolean_output(self):
        scores = np.array([0.1, 0.5, 0.9])
        result = select_fixed_threshold(scores, value=0.5)
        mask = apply_threshold(scores, result)
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_acceptance_ratio_correct(self):
        scores = np.array([0.1, 0.6, 0.7, 0.9])
        result = select_fixed_threshold(scores, value=0.5)
        self.assertAlmostEqual(result.acceptance_ratio, 3 / 4)
        self.assertAlmostEqual(result.rejection_ratio, 1 / 4)

    def test_threshold_config_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            ThresholdConfig(method="invalid")

    def test_batch_select_thresholds_length(self):
        arrays = [self._scores(seed=i) for i in range(4)]
        results = batch_select_thresholds(arrays)
        self.assertEqual(len(results), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ColorContinuityVerifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestColorContinuityVerifier(unittest.TestCase):

    def _pixels(self, n: int, value: float = 128.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random((n, 3)) * 255.0

    def test_identical_pixels_high_score(self):
        pixels = self._pixels(50, seed=1)
        result = verify_color_continuity(pixels, pixels)
        self.assertGreater(result.score, 0.9)
        self.assertTrue(result.is_valid)

    def test_very_different_colors_low_score(self):
        pa = np.zeros((50, 3), dtype=np.float64)         # black
        pb = np.full((50, 3), 255.0, dtype=np.float64)   # white
        cfg = ColorContinuityConfig(threshold=30.0)
        result = verify_color_continuity(pa, pb, config=cfg)
        self.assertLess(result.score, 0.5)

    def test_empty_pixels_returns_zero_score(self):
        pa = np.zeros((0, 3))
        pb = np.zeros((0, 3))
        result = verify_color_continuity(pa, pb)
        self.assertAlmostEqual(result.score, 0.0)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.n_samples, 0)

    def test_n_samples_is_min_length(self):
        pa = self._pixels(30, seed=2)
        pb = self._pixels(50, seed=3)
        result = verify_color_continuity(pa, pb)
        self.assertEqual(result.n_samples, 30)

    def test_score_from_delta_at_zero(self):
        score = ColorContinuityVerifier.score_from_delta(0.0, 30.0)
        self.assertAlmostEqual(score, 1.0)

    def test_score_from_delta_decay(self):
        s1 = ColorContinuityVerifier.score_from_delta(10.0, 30.0)
        s2 = ColorContinuityVerifier.score_from_delta(30.0, 30.0)
        self.assertGreater(s1, s2)

    def test_rgb_color_space(self):
        pa = self._pixels(20, seed=5)
        pb = self._pixels(20, seed=6)
        cfg = ColorContinuityConfig(method="rgb")
        result = verify_color_continuity(pa, pb, config=cfg)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_hsv_color_space(self):
        pa = self._pixels(20, seed=7)
        pb = self._pixels(20, seed=8)
        cfg = ColorContinuityConfig(method="hsv")
        result = verify_color_continuity(pa, pb, config=cfg)
        self.assertGreaterEqual(result.score, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CompletenessChecker
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletenessChecker(unittest.TestCase):

    def test_check_fragment_coverage_full(self):
        all_ids = [0, 1, 2, 3]
        self.assertAlmostEqual(check_fragment_coverage(all_ids, all_ids), 1.0)

    def test_check_fragment_coverage_partial(self):
        all_ids = [0, 1, 2, 3]
        placed = [0, 1]
        cov = check_fragment_coverage(placed, all_ids)
        self.assertAlmostEqual(cov, 0.5)

    def test_check_fragment_coverage_empty_all_returns_one(self):
        cov = check_fragment_coverage([], [])
        self.assertAlmostEqual(cov, 1.0)

    def test_check_fragment_coverage_invalid_id_raises(self):
        with self.assertRaises(ValueError):
            check_fragment_coverage([99], [0, 1, 2])

    def test_find_missing_fragments(self):
        missing = find_missing_fragments([0, 2], [0, 1, 2, 3])
        self.assertEqual(missing, [1, 3])

    def test_check_spatial_coverage_full(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        cov = check_spatial_coverage([mask], (10, 10))
        self.assertAlmostEqual(cov, 1.0)

    def test_check_spatial_coverage_empty(self):
        cov = check_spatial_coverage([], (10, 10))
        self.assertAlmostEqual(cov, 0.0)

    def test_check_spatial_coverage_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            check_spatial_coverage([], (0, 10))

    def test_find_uncovered_regions_all_covered(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        uncovered = find_uncovered_regions([mask], (5, 5))
        np.testing.assert_array_equal(uncovered, 0)

    def test_find_uncovered_regions_none_covered(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        uncovered = find_uncovered_regions([mask], (5, 5))
        np.testing.assert_array_equal(uncovered, 255)

    def test_completeness_score_full(self):
        score = completeness_score(10, 10, pixel_coverage=1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_completeness_score_partial(self):
        score = completeness_score(5, 10, pixel_coverage=0.5)
        self.assertAlmostEqual(score, 0.5)

    def test_completeness_score_invalid_n_total_raises(self):
        with self.assertRaises(ValueError):
            completeness_score(0, 0)

    def test_completeness_score_placed_gt_total_raises(self):
        with self.assertRaises(ValueError):
            completeness_score(11, 10)

    def test_generate_completeness_report(self):
        all_ids = [0, 1, 2, 3, 4]
        placed = [0, 1, 2]
        report = generate_completeness_report(placed, all_ids)
        self.assertAlmostEqual(report.fragment_coverage, 0.6)
        self.assertEqual(report.n_placed, 3)
        self.assertEqual(report.n_total, 5)
        self.assertIn(3, report.missing_ids)
        self.assertIn(4, report.missing_ids)

    def test_batch_check_coverage(self):
        all_ids = [0, 1, 2, 3]
        placed_sets = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
        results = batch_check_coverage(placed_sets, all_ids)
        self.assertAlmostEqual(results[0], 0.5)
        self.assertAlmostEqual(results[1], 0.75)
        self.assertAlmostEqual(results[2], 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. StatisticalCoherence
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatisticalCoherence(unittest.TestCase):

    def _patch(self, shape=(20, 20), seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, shape, dtype=np.uint8)

    def test_identical_patches_high_score(self):
        patch = self._patch(seed=10)
        result = StatisticalCoherenceVerifier().verify(patch, patch)
        self.assertGreater(result.overall_score, 0.8)
        self.assertTrue(result.is_coherent)

    def test_different_patches_result_in_range(self):
        pa = self._patch(seed=20)
        pb = self._patch(seed=21)
        verifier = StatisticalCoherenceVerifier()
        result = verifier.verify(pa, pb)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_moments_method(self):
        pa = self._patch(seed=30)
        pb = self._patch(seed=31)
        cfg = StatisticalCoherenceConfig(method="moments", use_texture=False)
        result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
        self.assertGreaterEqual(result.overall_score, 0.0)

    def test_both_method(self):
        pa = self._patch(seed=40)
        pb = self._patch(seed=41)
        cfg = StatisticalCoherenceConfig(method="both")
        result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
        self.assertGreaterEqual(result.overall_score, 0.0)

    def test_cohere_score_convenience(self):
        pa = self._patch(seed=50)
        pb = self._patch(seed=51)
        score = cohere_score(pa, pb)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_1d_patches_handled(self):
        rng = np.random.default_rng(60)
        pa = rng.random(100) * 255.0
        pb = rng.random(100) * 255.0
        score = cohere_score(pa, pb)
        self.assertGreaterEqual(score, 0.0)

    def test_3d_color_patches(self):
        pa = self._patch(shape=(20, 20, 3), seed=70)
        pb = self._patch(shape=(20, 20, 3), seed=71)
        score = cohere_score(pa, pb)
        self.assertGreaterEqual(score, 0.0)

    def test_is_coherent_above_threshold(self):
        patch = self._patch(seed=80)
        cfg = StatisticalCoherenceConfig(threshold=0.0)
        result = StatisticalCoherenceVerifier(cfg).verify(patch, patch)
        self.assertTrue(result.is_coherent)

    def test_all_similarity_fields_in_range(self):
        pa = self._patch(seed=90)
        pb = self._patch(seed=91)
        result = StatisticalCoherenceVerifier().verify(pa, pb)
        for val in (result.histogram_similarity, result.moment_similarity, result.texture_similarity):
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TextCoherence
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextCoherence(unittest.TestCase):

    CORPUS = [
        "the quick brown fox jumps over the lazy dog",
        "puzzle reconstruction requires careful matching",
        "fragments of the document are placed correctly",
        "adjacent fragments share compatible boundaries",
        "the algorithm scores each seam between fragments",
        "accurate reconstruction depends on good scoring",
        "we train a bigram model on the corpus text",
        "language model perplexity measures coherence",
    ]

    def test_build_ngram_model(self):
        model = build_ngram_model(self.CORPUS, n=2)
        self.assertIsInstance(model, NGramModel)
        self.assertGreater(model.total, 0)
        self.assertGreater(len(model.vocab), 0)

    def test_ngram_model_log_prob(self):
        model = build_ngram_model(self.CORPUS, n=2)
        lp = model.log_prob(("the",), "quick")
        self.assertLessEqual(lp, 0.0)

    def test_ngram_model_perplexity_known_sentence(self):
        model = build_ngram_model(self.CORPUS, n=2)
        pp = model.perplexity("the quick brown fox")
        self.assertGreater(pp, 1.0)
        self.assertTrue(np.isfinite(pp))

    def test_ngram_model_sentence_score_range(self):
        model = build_ngram_model(self.CORPUS, n=2)
        score = model.sentence_score("the quick brown fox")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ngram_model_short_sentence_returns_inf_perplexity(self):
        model = build_ngram_model(self.CORPUS, n=3)
        pp = model.perplexity("hi")   # too short for trigrams
        self.assertEqual(pp, float("inf"))

    def test_seam_bigram_score_no_model_returns_half(self):
        score = seam_bigram_score("hello world", "foo bar", model=None)
        self.assertAlmostEqual(score, 0.5)

    def test_seam_bigram_score_empty_text_returns_zero(self):
        score = seam_bigram_score("", "foo bar", model=None)
        self.assertAlmostEqual(score, 0.0)

    def test_seam_bigram_score_with_model(self):
        model = build_ngram_model(self.CORPUS, n=2)
        score = seam_bigram_score("the quick", "brown fox", model=model)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_word_boundary_score_full_words(self):
        # Use words longer than 3 chars to avoid broken-word heuristic
        score = word_boundary_score("hello world", "great results")
        self.assertAlmostEqual(score, 1.0)

    def test_word_boundary_score_empty_text(self):
        score = word_boundary_score("", "")
        self.assertAlmostEqual(score, 0.5)

    def test_word_boundary_score_broken_words(self):
        # Short words (1-3 chars) → broken score
        score = word_boundary_score("ab", "cd")
        self.assertLess(score, 1.0)

    def test_text_coherence_scorer_train(self):
        scorer = TextCoherenceScorer(n=2)
        scorer.train(self.CORPUS)
        self.assertTrue(scorer._trained)
        self.assertGreater(scorer.model.total, 0)

    def test_text_coherence_scorer_repr(self):
        scorer = TextCoherenceScorer(n=2)
        r = repr(scorer)
        self.assertIn("TextCoherenceScorer", r)


if __name__ == "__main__":
    unittest.main()
