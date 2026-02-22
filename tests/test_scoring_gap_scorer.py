"""Тесты для puzzle_reconstruction.scoring.gap_scorer."""
import pytest
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


# ─── TestGapConfig ────────────────────────────────────────────────────────────

class TestGapConfig:
    def test_defaults(self):
        cfg = GapConfig()
        assert cfg.target_gap == pytest.approx(5.0)
        assert cfg.tolerance == pytest.approx(1.0)
        assert cfg.penalty_scale == pytest.approx(1.0)
        assert cfg.max_gap == pytest.approx(20.0)

    def test_custom_values(self):
        cfg = GapConfig(target_gap=2.0, tolerance=0.5, penalty_scale=2.0, max_gap=10.0)
        assert cfg.target_gap == pytest.approx(2.0)

    def test_target_gap_zero_ok(self):
        cfg = GapConfig(target_gap=0.0, max_gap=1.0)
        assert cfg.target_gap == 0.0

    def test_target_gap_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=-1.0)

    def test_tolerance_zero_ok(self):
        cfg = GapConfig(tolerance=0.0)
        assert cfg.tolerance == 0.0

    def test_tolerance_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(tolerance=-0.1)

    def test_penalty_scale_zero_raises(self):
        with pytest.raises(ValueError):
            GapConfig(penalty_scale=0.0)

    def test_penalty_scale_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(penalty_scale=-1.0)

    def test_max_gap_equal_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=5.0, max_gap=5.0)

    def test_max_gap_less_than_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=10.0, max_gap=5.0)

    def test_max_gap_greater_than_target_ok(self):
        cfg = GapConfig(target_gap=3.0, max_gap=10.0)
        assert cfg.max_gap > cfg.target_gap


# ─── TestGapMeasure ───────────────────────────────────────────────────────────

class TestGapMeasure:
    def _make(self, id_a=0, id_b=1, distance=5.0,
              score=1.0, penalty=0.0) -> GapMeasure:
        return GapMeasure(id_a=id_a, id_b=id_b, distance=distance,
                          score=score, penalty=penalty)

    def test_basic(self):
        m = self._make()
        assert m.id_a == 0
        assert m.id_b == 1

    def test_pair_key_ordered(self):
        m = self._make(id_a=3, id_b=1)
        assert m.pair_key == (1, 3)

    def test_pair_key_already_ordered(self):
        m = self._make(id_a=0, id_b=5)
        assert m.pair_key == (0, 5)

    def test_is_acceptable_true(self):
        m = self._make(score=0.6)
        assert m.is_acceptable is True

    def test_is_acceptable_false(self):
        m = self._make(score=0.4)
        assert m.is_acceptable is False

    def test_is_acceptable_boundary(self):
        m = self._make(score=0.5)
        assert m.is_acceptable is False  # score > 0.5 required

    def test_distance_neg_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=-1.0, score=1.0, penalty=0.0)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=1.1, penalty=0.0)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=-0.1, penalty=0.0)

    def test_penalty_neg_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=1.0, penalty=-0.1)

    def test_score_zero_ok(self):
        m = GapMeasure(id_a=0, id_b=1, distance=100.0, score=0.0, penalty=5.0)
        assert m.score == 0.0

    def test_score_one_ok(self):
        m = self._make(score=1.0)
        assert m.score == 1.0


# ─── TestGapReport ────────────────────────────────────────────────────────────

class TestGapReport:
    def _make_measures(self, n=3):
        return [
            GapMeasure(id_a=i, id_b=i + 1, distance=5.0,
                       score=0.8, penalty=0.0)
            for i in range(n)
        ]

    def _make(self, n=3) -> GapReport:
        ms = self._make_measures(n)
        return GapReport(measures=ms, mean_score=0.8,
                         total_penalty=0.0, n_acceptable=n)

    def test_n_pairs(self):
        r = self._make(3)
        assert r.n_pairs == 3

    def test_acceptance_rate_full(self):
        r = self._make(4)
        assert r.acceptance_rate == pytest.approx(1.0)

    def test_acceptance_rate_empty(self):
        r = GapReport(measures=[], mean_score=0.0,
                      total_penalty=0.0, n_acceptable=0)
        assert r.acceptance_rate == pytest.approx(0.0)

    def test_mean_distance(self):
        ms = [
            GapMeasure(id_a=0, id_b=1, distance=4.0, score=0.9, penalty=0.0),
            GapMeasure(id_a=1, id_b=2, distance=6.0, score=0.9, penalty=0.0),
        ]
        r = GapReport(measures=ms, mean_score=0.9,
                      total_penalty=0.0, n_acceptable=2)
        assert r.mean_distance == pytest.approx(5.0)

    def test_mean_distance_empty(self):
        r = GapReport(measures=[], mean_score=0.0,
                      total_penalty=0.0, n_acceptable=0)
        assert r.mean_distance == pytest.approx(0.0)

    def test_get_measure_found(self):
        r = self._make(3)
        m = r.get_measure(0, 1)
        assert m is not None

    def test_get_measure_reverse_order(self):
        r = self._make(3)
        m = r.get_measure(1, 0)  # reversed
        assert m is not None

    def test_get_measure_not_found(self):
        r = self._make(3)
        assert r.get_measure(99, 100) is None

    def test_mean_score_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=-0.1,
                      total_penalty=0.0, n_acceptable=0)

    def test_mean_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=1.1,
                      total_penalty=0.0, n_acceptable=0)

    def test_total_penalty_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=0.5,
                      total_penalty=-1.0, n_acceptable=0)


# ─── TestScoreGap ─────────────────────────────────────────────────────────────

class TestScoreGap:
    def test_target_distance_perfect_score(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        score, penalty = score_gap(5.0, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_perfect(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        score, penalty = score_gap(5.5, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_beyond_tolerance_partial_score(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0, max_gap=20.0)
        score, penalty = score_gap(10.0, cfg)
        assert 0.0 < score < 1.0
        assert penalty > 0.0

    def test_beyond_max_gap_zero_score(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0, max_gap=10.0)
        score, penalty = score_gap(25.0, cfg)
        assert score == pytest.approx(0.0)
        assert penalty > 0.0

    def test_neg_distance_raises(self):
        with pytest.raises(ValueError):
            score_gap(-1.0)

    def test_zero_distance_ok(self):
        score, penalty = score_gap(0.0)
        assert isinstance(score, float)
        assert isinstance(penalty, float)

    def test_returns_tuple(self):
        result = score_gap(5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_cfg_used(self):
        score, _ = score_gap(5.0)
        assert score == pytest.approx(1.0)

    def test_score_in_range(self):
        for dist in [0.0, 3.0, 5.0, 7.0, 15.0, 30.0]:
            score, _ = score_gap(dist)
            assert 0.0 <= score <= 1.0


# ─── TestMeasureGap ───────────────────────────────────────────────────────────

class TestMeasureGap:
    def test_returns_gap_measure(self):
        m = measure_gap(0, 1, 5.0)
        assert isinstance(m, GapMeasure)

    def test_ids_stored(self):
        m = measure_gap(2, 7, 5.0)
        assert m.id_a == 2
        assert m.id_b == 7

    def test_distance_stored(self):
        m = measure_gap(0, 1, 8.0)
        assert m.distance == pytest.approx(8.0)

    def test_target_gap_score_one(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        m = measure_gap(0, 1, 5.0, cfg)
        assert m.score == pytest.approx(1.0)

    def test_penalty_non_negative(self):
        m = measure_gap(0, 1, 15.0)
        assert m.penalty >= 0.0


# ─── TestBuildGapReport ───────────────────────────────────────────────────────

class TestBuildGapReport:
    def test_returns_gap_report(self):
        r = build_gap_report({(0, 1): 5.0, (1, 2): 5.0})
        assert isinstance(r, GapReport)

    def test_empty_distances(self):
        r = build_gap_report({})
        assert r.n_pairs == 0
        assert r.mean_score == pytest.approx(0.0)

    def test_n_pairs_matches(self):
        r = build_gap_report({(0, 1): 5.0, (1, 2): 5.0, (2, 3): 5.0})
        assert r.n_pairs == 3

    def test_all_at_target_full_score(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        r = build_gap_report({(0, 1): 5.0, (1, 2): 5.0}, cfg)
        assert r.mean_score == pytest.approx(1.0)

    def test_mean_score_in_range(self):
        r = build_gap_report({(0, 1): 0.0, (1, 2): 100.0})
        assert 0.0 <= r.mean_score <= 1.0

    def test_n_acceptable_count(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        r = build_gap_report({(0, 1): 5.0, (1, 2): 50.0}, cfg)
        # First pair is at target (score=1.0, acceptable), second is not
        assert r.n_acceptable == 1

    def test_total_penalty_zero_at_target(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        r = build_gap_report({(0, 1): 5.0}, cfg)
        assert r.total_penalty == pytest.approx(0.0)


# ─── TestFilterGapMeasures ────────────────────────────────────────────────────

class TestFilterGapMeasures:
    def _report(self) -> GapReport:
        ms = [
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=1.0, penalty=0.0),
            GapMeasure(id_a=1, id_b=2, distance=15.0, score=0.3, penalty=2.0),
            GapMeasure(id_a=2, id_b=3, distance=7.0, score=0.7, penalty=0.5),
        ]
        return GapReport(measures=ms, mean_score=0.67,
                         total_penalty=2.5, n_acceptable=2)

    def test_returns_list(self):
        r = self._report()
        assert isinstance(filter_gap_measures(r), list)

    def test_no_filter(self):
        r = self._report()
        assert len(filter_gap_measures(r, 0.0)) == 3

    def test_filter_half(self):
        r = self._report()
        result = filter_gap_measures(r, 0.5)
        assert all(m.score >= 0.5 for m in result)

    def test_filter_all_out(self):
        r = self._report()
        assert filter_gap_measures(r, 1.1) == []

    def test_min_score_neg_raises(self):
        r = self._report()
        with pytest.raises(ValueError):
            filter_gap_measures(r, -0.1)

    def test_min_score_above_one_raises(self):
        r = self._report()
        with pytest.raises(ValueError):
            filter_gap_measures(r, 1.01)


# ─── TestWorstGapPairs ────────────────────────────────────────────────────────

class TestWorstGapPairs:
    def _report(self) -> GapReport:
        ms = [
            GapMeasure(id_a=i, id_b=i + 1, distance=5.0 + i * 2,
                       score=max(0.0, 1.0 - i * 0.2), penalty=float(i))
            for i in range(5)
        ]
        return GapReport(measures=ms, mean_score=0.5,
                         total_penalty=10.0, n_acceptable=3)

    def test_returns_list(self):
        r = self._report()
        assert isinstance(worst_gap_pairs(r), list)

    def test_sorted_by_penalty_desc(self):
        r = self._report()
        result = worst_gap_pairs(r, top_k=5)
        penalties = [m.penalty for m in result]
        assert penalties == sorted(penalties, reverse=True)

    def test_top_k_respected(self):
        r = self._report()
        assert len(worst_gap_pairs(r, top_k=3)) == 3

    def test_top_k_more_than_available(self):
        r = self._report()
        result = worst_gap_pairs(r, top_k=100)
        assert len(result) == 5

    def test_top_k_zero_raises(self):
        r = self._report()
        with pytest.raises(ValueError):
            worst_gap_pairs(r, top_k=0)

    def test_top_k_neg_raises(self):
        r = self._report()
        with pytest.raises(ValueError):
            worst_gap_pairs(r, top_k=-1)

    def test_top_k_one(self):
        r = self._report()
        result = worst_gap_pairs(r, top_k=1)
        assert len(result) == 1


# ─── TestGapScoreMatrix ───────────────────────────────────────────────────────

class TestGapScoreMatrix:
    def test_returns_dict(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        assert isinstance(result, dict)

    def test_keys_ordered(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        for a, b in result:
            assert a < b

    def test_values_in_range(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 15.0}
        result = gap_score_matrix(ids, dists)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_target_distance_score_one(self):
        cfg = GapConfig(target_gap=5.0, tolerance=1.0)
        ids = [0, 1]
        dists = {(0, 1): 5.0}
        result = gap_score_matrix(ids, dists, cfg)
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_missing_pair_excluded(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0}
        result = gap_score_matrix(ids, dists)
        # Only pair (0,1) has a known distance
        assert (0, 1) in result
        assert (0, 2) not in result

    def test_empty_ids(self):
        assert gap_score_matrix([], {}) == {}

    def test_symmetric_lookup(self):
        ids = [0, 1]
        # Provide reversed key
        dists = {(1, 0): 5.0}
        result = gap_score_matrix(ids, dists)
        assert (0, 1) in result
