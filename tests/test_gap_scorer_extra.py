"""Extra tests for puzzle_reconstruction/scoring/gap_scorer.py."""
from __future__ import annotations

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

def _cfg(**kw):
    defaults = dict(target_gap=5.0, tolerance=1.0, penalty_scale=1.0, max_gap=20.0)
    defaults.update(kw)
    return GapConfig(**defaults)


def _measure(a=0, b=1, dist=5.0, score=1.0, penalty=0.0):
    return GapMeasure(id_a=a, id_b=b, distance=dist, score=score, penalty=penalty)


def _report_with(measures, mean=0.8, total_pen=0.0, n_acc=0):
    return GapReport(measures=measures, mean_score=mean,
                     total_penalty=total_pen, n_acceptable=n_acc)


# ─── GapConfig (extra) ────────────────────────────────────────────────────────

class TestGapConfigExtra:
    def test_default_target_gap(self):
        assert GapConfig().target_gap == pytest.approx(5.0)

    def test_default_tolerance(self):
        assert GapConfig().tolerance == pytest.approx(1.0)

    def test_default_penalty_scale(self):
        assert GapConfig().penalty_scale == pytest.approx(1.0)

    def test_default_max_gap(self):
        assert GapConfig().max_gap == pytest.approx(20.0)

    def test_target_gap_large(self):
        cfg = GapConfig(target_gap=50.0, max_gap=100.0)
        assert cfg.target_gap == pytest.approx(50.0)

    def test_tolerance_large(self):
        cfg = GapConfig(tolerance=100.0, max_gap=200.0)
        assert cfg.tolerance == pytest.approx(100.0)

    def test_penalty_scale_large(self):
        cfg = GapConfig(penalty_scale=10.0)
        assert cfg.penalty_scale == pytest.approx(10.0)

    def test_max_gap_just_above_target(self):
        cfg = GapConfig(target_gap=5.0, max_gap=5.001)
        assert cfg.max_gap > cfg.target_gap

    def test_target_zero_max_positive(self):
        cfg = GapConfig(target_gap=0.0, max_gap=10.0)
        assert cfg.target_gap == pytest.approx(0.0)

    def test_penalty_scale_fractional(self):
        cfg = GapConfig(penalty_scale=0.5)
        assert cfg.penalty_scale == pytest.approx(0.5)

    def test_custom_all_fields(self):
        cfg = GapConfig(target_gap=3.0, tolerance=0.5, penalty_scale=2.0, max_gap=15.0)
        assert cfg.target_gap == pytest.approx(3.0)
        assert cfg.tolerance == pytest.approx(0.5)
        assert cfg.penalty_scale == pytest.approx(2.0)
        assert cfg.max_gap == pytest.approx(15.0)

    def test_max_gap_below_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=10.0, max_gap=9.0)

    def test_max_gap_equals_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=5.0, max_gap=5.0)


# ─── GapMeasure (extra) ───────────────────────────────────────────────────────

class TestGapMeasureExtra:
    def test_fields_stored(self):
        m = _measure(a=3, b=7, dist=12.5, score=0.4, penalty=3.2)
        assert m.id_a == 3
        assert m.id_b == 7
        assert m.distance == pytest.approx(12.5)
        assert m.score == pytest.approx(0.4)
        assert m.penalty == pytest.approx(3.2)

    def test_pair_key_a_less_b(self):
        m = _measure(a=2, b=8)
        assert m.pair_key == (2, 8)

    def test_pair_key_b_less_a(self):
        m = _measure(a=10, b=3)
        assert m.pair_key == (3, 10)

    def test_pair_key_equal_ids(self):
        m = _measure(a=5, b=5)
        assert m.pair_key == (5, 5)

    def test_is_acceptable_score_above_half(self):
        m = _measure(score=0.51)
        assert m.is_acceptable is True

    def test_is_acceptable_score_at_half(self):
        m = _measure(score=0.5)
        assert m.is_acceptable is False

    def test_is_acceptable_score_zero(self):
        m = _measure(score=0.0)
        assert m.is_acceptable is False

    def test_is_acceptable_score_one(self):
        m = _measure(score=1.0)
        assert m.is_acceptable is True

    def test_distance_large(self):
        m = _measure(dist=999.9, score=0.0, penalty=100.0)
        assert m.distance == pytest.approx(999.9)

    def test_penalty_large(self):
        m = _measure(penalty=500.0, score=0.0)
        assert m.penalty == pytest.approx(500.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=-0.01, score=0.5, penalty=0.0)

    def test_score_negative_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=-0.1, penalty=0.0)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=1.01, penalty=0.0)

    def test_penalty_negative_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=0.5, penalty=-0.1)


# ─── GapReport (extra) ────────────────────────────────────────────────────────

class TestGapReportExtra:
    def _make(self):
        measures = [
            _measure(a=0, b=1, dist=5.0, score=1.0, penalty=0.0),
            _measure(a=1, b=2, dist=8.0, score=0.7, penalty=1.0),
            _measure(a=0, b=2, dist=25.0, score=0.0, penalty=5.0),
        ]
        return GapReport(measures=measures, mean_score=0.57,
                         total_penalty=6.0, n_acceptable=1)

    def test_n_pairs_three(self):
        assert self._make().n_pairs == 3

    def test_n_pairs_empty(self):
        r = GapReport(measures=[], mean_score=0.0, total_penalty=0.0, n_acceptable=0)
        assert r.n_pairs == 0

    def test_acceptance_rate_one_third(self):
        r = self._make()
        assert r.acceptance_rate == pytest.approx(1.0 / 3.0)

    def test_acceptance_rate_full(self):
        measures = [_measure(score=1.0), _measure(a=1, b=2, score=0.8)]
        r = GapReport(measures=measures, mean_score=0.9,
                      total_penalty=0.0, n_acceptable=2)
        assert r.acceptance_rate == pytest.approx(1.0)

    def test_acceptance_rate_zero(self):
        measures = [_measure(score=0.3)]
        r = GapReport(measures=measures, mean_score=0.3,
                      total_penalty=2.0, n_acceptable=0)
        assert r.acceptance_rate == pytest.approx(0.0)

    def test_mean_distance(self):
        r = self._make()
        expected = (5.0 + 8.0 + 25.0) / 3.0
        assert r.mean_distance == pytest.approx(expected)

    def test_mean_distance_single(self):
        measures = [_measure(dist=7.5)]
        r = GapReport(measures=measures, mean_score=0.8,
                      total_penalty=0.0, n_acceptable=1)
        assert r.mean_distance == pytest.approx(7.5)

    def test_get_measure_by_pair(self):
        r = self._make()
        m = r.get_measure(0, 1)
        assert m is not None and m.distance == pytest.approx(5.0)

    def test_get_measure_reversed_pair(self):
        r = self._make()
        m = r.get_measure(2, 0)
        assert m is not None and m.distance == pytest.approx(25.0)

    def test_get_measure_missing(self):
        r = self._make()
        assert r.get_measure(99, 100) is None

    def test_mean_score_zero_ok(self):
        r = GapReport(measures=[], mean_score=0.0, total_penalty=0.0, n_acceptable=0)
        assert r.mean_score == pytest.approx(0.0)

    def test_mean_score_one_ok(self):
        r = GapReport(measures=[], mean_score=1.0, total_penalty=0.0, n_acceptable=0)
        assert r.mean_score == pytest.approx(1.0)

    def test_mean_score_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=-0.1, total_penalty=0.0, n_acceptable=0)

    def test_mean_score_above_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=1.01, total_penalty=0.0, n_acceptable=0)

    def test_total_penalty_zero_ok(self):
        r = GapReport(measures=[], mean_score=0.0, total_penalty=0.0, n_acceptable=0)
        assert r.total_penalty == pytest.approx(0.0)

    def test_total_penalty_neg_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=0.0, total_penalty=-1.0, n_acceptable=0)


# ─── score_gap (extra) ────────────────────────────────────────────────────────

class TestScoreGapExtra:
    def test_exact_target_returns_one(self):
        score, penalty = score_gap(5.0, _cfg(target_gap=5.0, tolerance=1.0))
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_low(self):
        score, penalty = score_gap(4.0, _cfg(target_gap=5.0, tolerance=1.0))
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_high(self):
        score, penalty = score_gap(6.0, _cfg(target_gap=5.0, tolerance=1.0))
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_beyond_max_score_zero(self):
        score, penalty = score_gap(100.0, _cfg(max_gap=20.0))
        assert score == pytest.approx(0.0)

    def test_score_monotone_decreasing(self):
        cfg = _cfg()
        scores = [score_gap(d, cfg)[0] for d in [5.0, 8.0, 12.0, 20.0]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_penalty_monotone_increasing(self):
        cfg = _cfg()
        penalties = [score_gap(d, cfg)[1] for d in [7.0, 10.0, 15.0]]
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1]

    def test_higher_penalty_scale_larger_penalty(self):
        _, p1 = score_gap(10.0, _cfg(penalty_scale=1.0))
        _, p2 = score_gap(10.0, _cfg(penalty_scale=3.0))
        assert p2 > p1

    def test_default_config_target_ok(self):
        score, penalty = score_gap(5.0)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            score_gap(-0.1)

    def test_zero_distance_ok(self):
        score, penalty = score_gap(0.0)
        assert 0.0 <= score <= 1.0
        assert penalty >= 0.0

    def test_returns_tuple_of_two(self):
        result = score_gap(5.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_large_tolerance_no_penalty(self):
        cfg = GapConfig(target_gap=5.0, tolerance=100.0, max_gap=200.0)
        score, penalty = score_gap(50.0, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)


# ─── measure_gap (extra) ──────────────────────────────────────────────────────

class TestMeasureGapExtra:
    def test_returns_gap_measure(self):
        assert isinstance(measure_gap(0, 1, 5.0), GapMeasure)

    def test_ids_stored_correctly(self):
        m = measure_gap(10, 20, 5.0)
        assert m.id_a == 10
        assert m.id_b == 20

    def test_distance_stored(self):
        m = measure_gap(0, 1, 12.3)
        assert m.distance == pytest.approx(12.3)

    def test_within_tolerance_score_one(self):
        m = measure_gap(0, 1, 5.0, _cfg(target_gap=5.0, tolerance=1.0))
        assert m.score == pytest.approx(1.0)
        assert m.penalty == pytest.approx(0.0)

    def test_far_distance_score_zero(self):
        m = measure_gap(0, 1, 100.0, _cfg(max_gap=20.0))
        assert m.score == pytest.approx(0.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            measure_gap(0, 1, -1.0)

    def test_default_config_works(self):
        m = measure_gap(0, 1, 5.0)
        assert m.score == pytest.approx(1.0)

    def test_zero_distance_ok(self):
        m = measure_gap(0, 1, 0.0)
        assert 0.0 <= m.score <= 1.0

    def test_pair_key_ordered(self):
        m = measure_gap(5, 2, 5.0)
        assert m.pair_key == (2, 5)


# ─── build_gap_report (extra) ─────────────────────────────────────────────────

class TestBuildGapReportExtra:
    def test_empty_dict_returns_report(self):
        r = build_gap_report({})
        assert isinstance(r, GapReport)
        assert r.n_pairs == 0

    def test_single_pair_target_distance(self):
        r = build_gap_report({(0, 1): 5.0}, _cfg())
        assert r.n_pairs == 1
        assert r.mean_score == pytest.approx(1.0)

    def test_multiple_pairs_count(self):
        dists = {(0, 1): 5.0, (1, 2): 6.0, (0, 2): 5.5}
        r = build_gap_report(dists)
        assert r.n_pairs == 3

    def test_measures_are_gap_measure_instances(self):
        r = build_gap_report({(0, 1): 5.0})
        for m in r.measures:
            assert isinstance(m, GapMeasure)

    def test_scores_in_range(self):
        dists = {(i, j): float(i * 4) for i in range(4) for j in range(i + 1, 4)}
        r = build_gap_report(dists)
        for m in r.measures:
            assert 0.0 <= m.score <= 1.0

    def test_total_penalty_sum(self):
        cfg = _cfg(target_gap=5.0, tolerance=0.0, max_gap=50.0)
        r = build_gap_report({(0, 1): 5.0}, cfg)
        total = sum(m.penalty for m in r.measures)
        assert r.total_penalty == pytest.approx(total)

    def test_n_acceptable_count(self):
        cfg = _cfg()
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 100.0}
        r = build_gap_report(dists, cfg)
        assert r.n_acceptable == 2

    def test_default_config_used(self):
        r = build_gap_report({(0, 1): 5.0})
        assert r.mean_score == pytest.approx(1.0)

    def test_mean_score_in_range(self):
        dists = {(0, 1): 3.0, (1, 2): 7.0, (0, 2): 15.0}
        r = build_gap_report(dists)
        assert 0.0 <= r.mean_score <= 1.0


# ─── filter_gap_measures (extra) ──────────────────────────────────────────────

class TestFilterGapMeasuresExtra:
    def _report(self):
        measures = [
            _measure(a=0, b=1, score=0.9),
            _measure(a=0, b=2, score=0.6),
            _measure(a=1, b=2, score=0.3),
            _measure(a=2, b=3, score=0.1),
        ]
        return GapReport(measures=measures, mean_score=0.5,
                         total_penalty=2.0, n_acceptable=2)

    def test_explicit_min_score_half(self):
        result = filter_gap_measures(self._report(), min_score=0.5)
        for m in result:
            assert m.score >= 0.5

    def test_min_zero_keeps_all(self):
        result = filter_gap_measures(self._report(), min_score=0.0)
        assert len(result) == 4

    def test_min_one_keeps_none(self):
        result = filter_gap_measures(self._report(), min_score=1.0)
        assert len(result) == 0

    def test_threshold_09_keeps_one(self):
        result = filter_gap_measures(self._report(), min_score=0.9)
        assert len(result) == 1 and result[0].score == pytest.approx(0.9)

    def test_threshold_06_keeps_two(self):
        result = filter_gap_measures(self._report(), min_score=0.6)
        assert len(result) == 2

    def test_returns_list(self):
        assert isinstance(filter_gap_measures(self._report()), list)

    def test_all_items_are_gap_measure(self):
        for m in filter_gap_measures(self._report(), min_score=0.0):
            assert isinstance(m, GapMeasure)

    def test_empty_report(self):
        empty = GapReport(measures=[], mean_score=0.0,
                          total_penalty=0.0, n_acceptable=0)
        assert filter_gap_measures(empty) == []

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            filter_gap_measures(self._report(), min_score=-0.1)

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_gap_measures(self._report(), min_score=1.1)


# ─── worst_gap_pairs (extra) ──────────────────────────────────────────────────

class TestWorstGapPairsExtra:
    def _report(self):
        measures = [
            _measure(a=0, b=1, penalty=1.0),
            _measure(a=0, b=2, penalty=3.0),
            _measure(a=1, b=2, penalty=7.0),
            _measure(a=2, b=3, penalty=2.0),
        ]
        return GapReport(measures=measures, mean_score=0.5,
                         total_penalty=13.0, n_acceptable=2)

    def test_top_1_is_worst(self):
        result = worst_gap_pairs(self._report(), top_k=1)
        assert len(result) == 1
        assert result[0].penalty == pytest.approx(7.0)

    def test_top_2_sorted_desc(self):
        result = worst_gap_pairs(self._report(), top_k=2)
        assert len(result) == 2
        assert result[0].penalty >= result[1].penalty

    def test_top_k_larger_than_n(self):
        result = worst_gap_pairs(self._report(), top_k=100)
        assert len(result) == 4

    def test_returns_list(self):
        assert isinstance(worst_gap_pairs(self._report()), list)

    def test_all_items_gap_measure(self):
        for m in worst_gap_pairs(self._report(), top_k=4):
            assert isinstance(m, GapMeasure)

    def test_sorted_descending_all(self):
        result = worst_gap_pairs(self._report(), top_k=4)
        penalties = [m.penalty for m in result]
        assert penalties == sorted(penalties, reverse=True)

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

    def test_default_top_k_5(self):
        result = worst_gap_pairs(self._report())
        assert len(result) <= 5


# ─── gap_score_matrix (extra) ─────────────────────────────────────────────────

class TestGapScoreMatrixExtra:
    def test_returns_dict(self):
        assert isinstance(gap_score_matrix([0, 1], {(0, 1): 5.0}), dict)

    def test_empty_ids(self):
        assert gap_score_matrix([], {}) == {}

    def test_single_id_no_pairs(self):
        assert gap_score_matrix([0], {}) == {}

    def test_ordered_keys(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0, (0, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        for a, b in result.keys():
            assert a < b

    def test_reversed_key_lookup(self):
        ids = [0, 1]
        dists = {(1, 0): 5.0}
        result = gap_score_matrix(ids, dists)
        assert (0, 1) in result

    def test_missing_pair_not_present(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0}
        result = gap_score_matrix(ids, dists)
        assert (0, 2) not in result
        assert (1, 2) not in result

    def test_target_score_one(self):
        ids = [0, 1]
        result = gap_score_matrix(ids, {(0, 1): 5.0}, _cfg(target_gap=5.0, tolerance=1.0))
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_far_distance_score_zero(self):
        ids = [0, 1]
        result = gap_score_matrix(ids, {(0, 1): 100.0}, _cfg(max_gap=20.0))
        assert result[(0, 1)] == pytest.approx(0.0)

    def test_scores_in_range(self):
        ids = list(range(5))
        dists = {(i, j): float(i * 3 + 2) for i in range(5) for j in range(i + 1, 5)}
        result = gap_score_matrix(ids, dists)
        for s in result.values():
            assert 0.0 <= s <= 1.0

    def test_default_config_used(self):
        ids = [0, 1]
        result = gap_score_matrix(ids, {(0, 1): 5.0})
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_two_pairs_both_present(self):
        ids = [0, 1, 2]
        dists = {(0, 1): 5.0, (1, 2): 5.0}
        result = gap_score_matrix(ids, dists)
        assert (0, 1) in result
        assert (1, 2) in result
        assert (0, 2) not in result
