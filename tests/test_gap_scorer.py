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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cfg(target=5.0, tol=1.0, scale=1.0, max_g=20.0):
    return GapConfig(target_gap=target, tolerance=tol,
                     penalty_scale=scale, max_gap=max_g)


def _measure(a=0, b=1, dist=5.0, score=1.0, penalty=0.0):
    return GapMeasure(id_a=a, id_b=b, distance=dist,
                      score=score, penalty=penalty)


def _report(measures=None, mean=0.8, total_pen=0.5, n_acc=2):
    if measures is None:
        measures = [_measure()]
    return GapReport(measures=measures, mean_score=mean,
                     total_penalty=total_pen, n_acceptable=n_acc)


# ─── TestGapConfig ────────────────────────────────────────────────────────────

class TestGapConfig:
    def test_defaults(self):
        cfg = GapConfig()
        assert cfg.target_gap == pytest.approx(5.0)
        assert cfg.tolerance == pytest.approx(1.0)
        assert cfg.penalty_scale == pytest.approx(1.0)
        assert cfg.max_gap == pytest.approx(20.0)

    def test_valid_custom(self):
        cfg = _cfg(target=2.0, tol=0.5, scale=2.0, max_g=10.0)
        assert cfg.target_gap == pytest.approx(2.0)
        assert cfg.max_gap == pytest.approx(10.0)

    def test_target_zero_ok(self):
        cfg = GapConfig(target_gap=0.0, max_gap=1.0)
        assert cfg.target_gap == 0.0

    def test_target_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=-1.0)

    def test_tolerance_zero_ok(self):
        cfg = GapConfig(tolerance=0.0)
        assert cfg.tolerance == 0.0

    def test_tolerance_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(tolerance=-0.1)

    def test_penalty_scale_small_ok(self):
        cfg = GapConfig(penalty_scale=0.001)
        assert cfg.penalty_scale == pytest.approx(0.001)

    def test_penalty_scale_zero_raises(self):
        with pytest.raises(ValueError):
            GapConfig(penalty_scale=0.0)

    def test_penalty_scale_neg_raises(self):
        with pytest.raises(ValueError):
            GapConfig(penalty_scale=-1.0)

    def test_max_gap_equals_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=5.0, max_gap=5.0)

    def test_max_gap_below_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=10.0, max_gap=5.0)

    def test_large_values_ok(self):
        cfg = GapConfig(target_gap=100.0, max_gap=500.0)
        assert cfg.target_gap == pytest.approx(100.0)


# ─── TestGapMeasure ───────────────────────────────────────────────────────────

class TestGapMeasure:
    def test_basic(self):
        m = _measure()
        assert m.id_a == 0
        assert m.id_b == 1
        assert m.distance == pytest.approx(5.0)
        assert m.score == pytest.approx(1.0)
        assert m.penalty == pytest.approx(0.0)

    def test_pair_key_ordered(self):
        m = _measure(a=5, b=2)
        assert m.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        m = _measure(a=0, b=3)
        assert m.pair_key == (0, 3)

    def test_is_acceptable_true(self):
        m = _measure(score=0.6)
        assert m.is_acceptable is True

    def test_is_acceptable_boundary(self):
        # score = 0.5 is NOT > 0.5
        m = _measure(score=0.5)
        assert m.is_acceptable is False

    def test_is_acceptable_false(self):
        m = _measure(score=0.3)
        assert m.is_acceptable is False

    def test_distance_zero_ok(self):
        m = _measure(dist=0.0)
        assert m.distance == 0.0

    def test_distance_neg_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=-1.0, score=0.5, penalty=0.0)

    def test_score_zero_ok(self):
        m = _measure(score=0.0)
        assert m.score == 0.0

    def test_score_one_ok(self):
        m = _measure(score=1.0)
        assert m.score == 1.0

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=1.0, score=-0.1, penalty=0.0)

    def test_score_above_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=1.0, score=1.1, penalty=0.0)

    def test_penalty_zero_ok(self):
        m = _measure(penalty=0.0)
        assert m.penalty == 0.0

    def test_penalty_neg_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=1.0, score=0.5, penalty=-0.1)


# ─── TestGapReport ────────────────────────────────────────────────────────────

class TestGapReport:
    def _make_report(self):
        measures = [
            _measure(a=0, b=1, dist=5.0, score=1.0, penalty=0.0),
            _measure(a=0, b=2, dist=12.0, score=0.4, penalty=2.0),
            _measure(a=1, b=2, dist=5.5, score=0.9, penalty=0.0),
        ]
        return GapReport(measures=measures, mean_score=0.77,
                         total_penalty=2.0, n_acceptable=2)

    def test_n_pairs(self):
        r = self._make_report()
        assert r.n_pairs == 3

    def test_n_pairs_empty(self):
        r = GapReport(measures=[], mean_score=0.0,
                      total_penalty=0.0, n_acceptable=0)
        assert r.n_pairs == 0

    def test_acceptance_rate(self):
        r = self._make_report()
        assert r.acceptance_rate == pytest.approx(2.0 / 3.0)

    def test_acceptance_rate_empty(self):
        r = GapReport(measures=[], mean_score=0.0,
                      total_penalty=0.0, n_acceptable=0)
        assert r.acceptance_rate == pytest.approx(0.0)

    def test_mean_distance(self):
        r = self._make_report()
        expected = (5.0 + 12.0 + 5.5) / 3.0
        assert r.mean_distance == pytest.approx(expected)

    def test_mean_distance_empty(self):
        r = GapReport(measures=[], mean_score=0.0,
                      total_penalty=0.0, n_acceptable=0)
        assert r.mean_distance == pytest.approx(0.0)

    def test_get_measure_found(self):
        r = self._make_report()
        m = r.get_measure(0, 1)
        assert m is not None
        assert m.pair_key == (0, 1)

    def test_get_measure_reversed(self):
        r = self._make_report()
        m = r.get_measure(1, 0)
        assert m is not None

    def test_get_measure_not_found(self):
        r = self._make_report()
        m = r.get_measure(99, 100)
        assert m is None

    def test_mean_score_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=-0.1,
                      total_penalty=0.0, n_acceptable=0)

    def test_mean_score_above_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=1.1,
                      total_penalty=0.0, n_acceptable=0)

    def test_total_penalty_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=0.5,
                      total_penalty=-1.0, n_acceptable=0)


# ─── TestScoreGap ─────────────────────────────────────────────────────────────

class TestScoreGap:
    def test_exact_target_score_one(self):
        cfg = _cfg(target=5.0, tol=1.0)
        score, penalty = score_gap(5.0, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_score_one(self):
        cfg = _cfg(target=5.0, tol=1.0)
        score, penalty = score_gap(5.5, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_low_end(self):
        cfg = _cfg(target=5.0, tol=1.0)
        score, penalty = score_gap(4.0, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_beyond_tolerance_positive_penalty(self):
        cfg = _cfg(target=5.0, tol=1.0, scale=1.0, max_g=20.0)
        score, penalty = score_gap(10.0, cfg)
        assert score < 1.0
        assert penalty > 0.0

    def test_beyond_max_gap_zero_score(self):
        cfg = _cfg(target=5.0, tol=1.0, max_g=20.0)
        score, penalty = score_gap(25.0, cfg)
        assert score == pytest.approx(0.0)
        assert penalty > 0.0

    def test_zero_distance_penalized(self):
        cfg = _cfg(target=5.0, tol=1.0)
        score, penalty = score_gap(0.0, cfg)
        assert score <= 1.0
        assert penalty >= 0.0

    def test_neg_distance_raises(self):
        with pytest.raises(ValueError):
            score_gap(-1.0)

    def test_default_config_used(self):
        score, penalty = score_gap(5.0)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_score_in_range(self):
        cfg = _cfg()
        for dist in (0.0, 3.0, 5.0, 8.0, 15.0, 25.0):
            score, penalty = score_gap(dist, cfg)
            assert 0.0 <= score <= 1.0
            assert penalty >= 0.0

    def test_penalty_scale_effect(self):
        cfg1 = _cfg(scale=1.0)
        cfg2 = _cfg(scale=2.0)
        _, p1 = score_gap(10.0, cfg1)
        _, p2 = score_gap(10.0, cfg2)
        assert p2 > p1

    def test_large_tolerance_zero_penalty(self):
        cfg = GapConfig(target_gap=5.0, tolerance=100.0, max_gap=200.0)
        score, penalty = score_gap(50.0, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)


# ─── TestMeasureGap ───────────────────────────────────────────────────────────

class TestMeasureGap:
    def test_returns_gap_measure(self):
        m = measure_gap(0, 1, 5.0)
        assert isinstance(m, GapMeasure)

    def test_ids_stored(self):
        m = measure_gap(3, 7, 5.0)
        assert m.id_a == 3
        assert m.id_b == 7

    def test_distance_stored(self):
        m = measure_gap(0, 1, 8.5)
        assert m.distance == pytest.approx(8.5)

    def test_within_target_score_one(self):
        cfg = _cfg(target=5.0, tol=1.0)
        m = measure_gap(0, 1, 5.0, cfg)
        assert m.score == pytest.approx(1.0)

    def test_default_config(self):
        m = measure_gap(0, 1, 5.0)
        assert m.score == pytest.approx(1.0)

    def test_neg_distance_raises(self):
        with pytest.raises(ValueError):
            measure_gap(0, 1, -1.0)


# ─── TestBuildGapReport ───────────────────────────────────────────────────────

class TestBuildGapReport:
    def test_empty_dict(self):
        r = build_gap_report({})
        assert r.n_pairs == 0
        assert r.mean_score == pytest.approx(0.0)
        assert r.total_penalty == pytest.approx(0.0)
        assert r.n_acceptable == 0

    def test_single_pair(self):
        r = build_gap_report({(0, 1): 5.0})
        assert r.n_pairs == 1
        assert r.mean_score == pytest.approx(1.0)

    def test_returns_gap_report(self):
        r = build_gap_report({(0, 1): 5.0, (1, 2): 10.0})
        assert isinstance(r, GapReport)

    def test_mean_score_average(self):
        cfg = _cfg(target=5.0, tol=0.0, max_g=15.0)
        r = build_gap_report({(0, 1): 5.0, (1, 2): 5.0}, cfg)
        assert r.mean_score == pytest.approx(1.0)

    def test_n_pairs_correct(self):
        dists = {(i, i + 1): float(i) for i in range(5)}
        r = build_gap_report(dists)
        assert r.n_pairs == 5

    def test_total_penalty_positive_for_bad(self):
        cfg = _cfg(target=5.0, tol=0.0, max_g=15.0)
        r = build_gap_report({(0, 1): 14.0}, cfg)
        assert r.total_penalty > 0.0

    def test_n_acceptable_count(self):
        cfg = _cfg(target=5.0, tol=1.0)
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 30.0}
        r = build_gap_report(dists, cfg)
        assert r.n_acceptable == 2

    def test_measures_stored(self):
        r = build_gap_report({(0, 1): 5.0})
        assert len(r.measures) == 1
        assert isinstance(r.measures[0], GapMeasure)

    def test_scores_in_range(self):
        dists = {(i, j): float(i * 3) for i in range(4) for j in range(i + 1, 4)}
        r = build_gap_report(dists)
        for m in r.measures:
            assert 0.0 <= m.score <= 1.0


# ─── TestFilterGapMeasures ────────────────────────────────────────────────────

class TestFilterGapMeasures:
    def _report(self):
        measures = [
            _measure(a=0, b=1, score=1.0),
            _measure(a=0, b=2, score=0.5),
            _measure(a=1, b=2, score=0.2),
        ]
        return GapReport(measures=measures, mean_score=0.57,
                         total_penalty=1.0, n_acceptable=1)

    def test_min_zero_all(self):
        result = filter_gap_measures(self._report(), min_score=0.0)
        assert len(result) == 3

    def test_min_one_empty(self):
        result = filter_gap_measures(self._report(), min_score=1.0)
        assert len(result) == 1
        assert result[0].score == pytest.approx(1.0)

    def test_min_half(self):
        result = filter_gap_measures(self._report(), min_score=0.5)
        assert len(result) == 2

    def test_returns_list(self):
        result = filter_gap_measures(self._report())
        assert isinstance(result, list)

    def test_empty_report(self):
        empty = GapReport(measures=[], mean_score=0.0,
                          total_penalty=0.0, n_acceptable=0)
        assert filter_gap_measures(empty) == []

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            filter_gap_measures(self._report(), min_score=-0.1)

    def test_min_score_above_raises(self):
        with pytest.raises(ValueError):
            filter_gap_measures(self._report(), min_score=1.1)

    def test_all_measures_are_gap_measure(self):
        for m in filter_gap_measures(self._report(), min_score=0.0):
            assert isinstance(m, GapMeasure)


# ─── TestWorstGapPairs ────────────────────────────────────────────────────────

class TestWorstGapPairs:
    def _report(self):
        measures = [
            _measure(a=0, b=1, penalty=0.0),
            _measure(a=0, b=2, penalty=5.0),
            _measure(a=1, b=2, penalty=2.0),
            _measure(a=2, b=3, penalty=8.0),
            _measure(a=3, b=4, penalty=1.0),
        ]
        return GapReport(measures=measures, mean_score=0.5,
                         total_penalty=16.0, n_acceptable=3)

    def test_top_1(self):
        result = worst_gap_pairs(self._report(), top_k=1)
        assert len(result) == 1
        assert result[0].penalty == pytest.approx(8.0)

    def test_top_3(self):
        result = worst_gap_pairs(self._report(), top_k=3)
        assert len(result) == 3
        penalties = [m.penalty for m in result]
        assert penalties == sorted(penalties, reverse=True)

    def test_top_larger_than_measures(self):
        result = worst_gap_pairs(self._report(), top_k=100)
        assert len(result) == 5

    def test_sorted_descending(self):
        result = worst_gap_pairs(self._report(), top_k=5)
        penalties = [m.penalty for m in result]
        assert penalties == sorted(penalties, reverse=True)

    def test_returns_list(self):
        result = worst_gap_pairs(self._report())
        assert isinstance(result, list)

    def test_empty_report(self):
        empty = GapReport(measures=[], mean_score=0.0,
                          total_penalty=0.0, n_acceptable=0)
        assert worst_gap_pairs(empty) == []

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            worst_gap_pairs(self._report(), top_k=0)

    def test_top_k_neg_raises(self):
        with pytest.raises(ValueError):
            worst_gap_pairs(self._report(), top_k=-1)


# ─── TestGapScoreMatrix ───────────────────────────────────────────────────────

class TestGapScoreMatrix:
    def test_returns_dict(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (0, 2): 6.0, (1, 2): 5.5}
        result = gap_score_matrix(ids, dists)
        assert isinstance(result, dict)

    def test_keys_are_ordered_pairs(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        for a, b in result.keys():
            assert a < b

    def test_all_pairs_present(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        assert set(result.keys()) == {(0, 1), (0, 2), (1, 2)}

    def test_scores_in_range(self):
        ids = [0, 1, 2, 3]
        dists = {(i, j): float(i * 5) for i in range(4) for j in range(i + 1, 4)}
        cfg = _cfg()
        result = gap_score_matrix(ids, dists, cfg)
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_missing_distances_skipped(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0}  # Only one pair
        result = gap_score_matrix(ids, dists)
        assert (0, 1) in result
        assert (0, 2) not in result
        assert (1, 2) not in result

    def test_empty_ids(self):
        result = gap_score_matrix([], {})
        assert result == {}

    def test_single_id_no_pairs(self):
        result = gap_score_matrix([0], {})
        assert result == {}

    def test_target_distance_score_one(self):
        ids = [0, 1]
        cfg = _cfg(target=5.0, tol=1.0)
        dists = {(0, 1): 5.0}
        result = gap_score_matrix(ids, dists, cfg)
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_reversed_key_lookup(self):
        ids = [0, 1]
        dists = {(1, 0): 5.0}  # Reversed key
        result = gap_score_matrix(ids, dists)
        assert (0, 1) in result
