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

def _measure(a=0, b=1, dist=5.0, score=0.8, penalty=0.0) -> GapMeasure:
    return GapMeasure(id_a=a, id_b=b, distance=dist,
                      score=score, penalty=penalty)


def _report(measures=None, mean=0.5, penalty=0.0, n_acc=0) -> GapReport:
    return GapReport(measures=measures or [],
                     mean_score=mean,
                     total_penalty=penalty,
                     n_acceptable=n_acc)


def _default_cfg() -> GapConfig:
    return GapConfig(target_gap=5.0, tolerance=1.0,
                     penalty_scale=1.0, max_gap=20.0)


# ─── GapConfig ────────────────────────────────────────────────────────────────

class TestGapConfigExtra:
    def test_default_target_gap(self):
        assert GapConfig().target_gap == pytest.approx(5.0)

    def test_default_tolerance(self):
        assert GapConfig().tolerance == pytest.approx(1.0)

    def test_default_penalty_scale(self):
        assert GapConfig().penalty_scale == pytest.approx(1.0)

    def test_default_max_gap(self):
        assert GapConfig().max_gap == pytest.approx(20.0)

    def test_negative_target_gap_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=-1.0)

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            GapConfig(tolerance=-0.1)

    def test_zero_penalty_scale_raises(self):
        with pytest.raises(ValueError):
            GapConfig(penalty_scale=0.0)

    def test_max_gap_le_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=10.0, max_gap=5.0)

    def test_max_gap_equal_target_raises(self):
        with pytest.raises(ValueError):
            GapConfig(target_gap=5.0, max_gap=5.0)

    def test_custom_config(self):
        cfg = GapConfig(target_gap=3.0, tolerance=0.5, max_gap=15.0)
        assert cfg.target_gap == pytest.approx(3.0)


# ─── GapMeasure ───────────────────────────────────────────────────────────────

class TestGapMeasureExtra:
    def test_ids_stored(self):
        m = _measure(a=2, b=5)
        assert m.id_a == 2 and m.id_b == 5

    def test_distance_stored(self):
        m = _measure(dist=7.5)
        assert m.distance == pytest.approx(7.5)

    def test_score_stored(self):
        m = _measure(score=0.9)
        assert m.score == pytest.approx(0.9)

    def test_penalty_stored(self):
        m = _measure(penalty=2.5)
        assert m.penalty == pytest.approx(2.5)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=-1.0, score=0.5, penalty=0.0)

    def test_score_gt_one_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=1.5, penalty=0.0)

    def test_negative_penalty_raises(self):
        with pytest.raises(ValueError):
            GapMeasure(id_a=0, id_b=1, distance=5.0, score=0.5, penalty=-1.0)

    def test_pair_key_ordered(self):
        m = _measure(a=5, b=2)
        assert m.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        m = _measure(a=1, b=3)
        assert m.pair_key == (1, 3)

    def test_is_acceptable_true(self):
        assert _measure(score=0.7).is_acceptable is True

    def test_is_acceptable_false(self):
        assert _measure(score=0.3).is_acceptable is False

    def test_is_acceptable_boundary(self):
        assert _measure(score=0.5).is_acceptable is False


# ─── GapReport ────────────────────────────────────────────────────────────────

class TestGapReportExtra:
    def test_mean_score_stored(self):
        r = _report(mean=0.7)
        assert r.mean_score == pytest.approx(0.7)

    def test_total_penalty_stored(self):
        r = _report(penalty=3.0)
        assert r.total_penalty == pytest.approx(3.0)

    def test_n_acceptable_stored(self):
        r = _report(n_acc=4)
        assert r.n_acceptable == 4

    def test_mean_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=1.5,
                      total_penalty=0.0, n_acceptable=0)

    def test_negative_total_penalty_raises(self):
        with pytest.raises(ValueError):
            GapReport(measures=[], mean_score=0.5,
                      total_penalty=-1.0, n_acceptable=0)

    def test_n_pairs_property(self):
        r = _report(measures=[_measure(), _measure()])
        assert r.n_pairs == 2

    def test_acceptance_rate_empty(self):
        r = _report()
        assert r.acceptance_rate == pytest.approx(0.0)

    def test_acceptance_rate_computed(self):
        ms = [_measure(score=0.8), _measure(score=0.3)]
        r = GapReport(measures=ms, mean_score=0.55, total_penalty=0.0, n_acceptable=1)
        assert r.acceptance_rate == pytest.approx(0.5)

    def test_mean_distance_empty(self):
        r = _report()
        assert r.mean_distance == pytest.approx(0.0)

    def test_mean_distance_computed(self):
        ms = [_measure(dist=4.0), _measure(dist=6.0)]
        r = GapReport(measures=ms, mean_score=0.5, total_penalty=0.0, n_acceptable=2)
        assert r.mean_distance == pytest.approx(5.0)

    def test_get_measure_found(self):
        ms = [_measure(a=0, b=1)]
        r = GapReport(measures=ms, mean_score=0.8, total_penalty=0.0, n_acceptable=1)
        m = r.get_measure(0, 1)
        assert m is not None

    def test_get_measure_reversed(self):
        ms = [_measure(a=0, b=1)]
        r = GapReport(measures=ms, mean_score=0.8, total_penalty=0.0, n_acceptable=1)
        m = r.get_measure(1, 0)
        assert m is not None

    def test_get_measure_not_found(self):
        r = _report()
        assert r.get_measure(99, 100) is None


# ─── score_gap ────────────────────────────────────────────────────────────────

class TestScoreGapExtra:
    def test_returns_tuple(self):
        result = score_gap(5.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_target_gap_score_one(self):
        cfg = _default_cfg()
        score, penalty = score_gap(cfg.target_gap, cfg)
        assert score == pytest.approx(1.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_score_one(self):
        cfg = _default_cfg()  # target=5, tolerance=1
        score, penalty = score_gap(5.5, cfg)
        assert score == pytest.approx(1.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            score_gap(-1.0)

    def test_beyond_max_gap_score_zero(self):
        cfg = _default_cfg()  # max_gap=20
        score, penalty = score_gap(25.0, cfg)
        assert score == pytest.approx(0.0)
        assert penalty > 0.0

    def test_none_cfg_uses_defaults(self):
        score, penalty = score_gap(5.0, cfg=None)
        assert isinstance(score, float)

    def test_score_in_range(self):
        for d in [0.0, 3.0, 5.0, 10.0, 25.0]:
            score, _ = score_gap(d)
            assert 0.0 <= score <= 1.0


# ─── measure_gap ──────────────────────────────────────────────────────────────

class TestMeasureGapExtra:
    def test_returns_gap_measure(self):
        m = measure_gap(0, 1, 5.0)
        assert isinstance(m, GapMeasure)

    def test_ids_stored(self):
        m = measure_gap(2, 7, 5.0)
        assert m.id_a == 2 and m.id_b == 7

    def test_distance_stored(self):
        m = measure_gap(0, 1, 8.0)
        assert m.distance == pytest.approx(8.0)

    def test_score_in_range(self):
        m = measure_gap(0, 1, 5.0)
        assert 0.0 <= m.score <= 1.0

    def test_none_cfg(self):
        m = measure_gap(0, 1, 5.0, cfg=None)
        assert isinstance(m, GapMeasure)


# ─── build_gap_report ─────────────────────────────────────────────────────────

class TestBuildGapReportExtra:
    def test_returns_gap_report(self):
        r = build_gap_report({(0, 1): 5.0})
        assert isinstance(r, GapReport)

    def test_empty_distances(self):
        r = build_gap_report({})
        assert r.n_pairs == 0
        assert r.mean_score == pytest.approx(0.0)

    def test_n_pairs_matches(self):
        r = build_gap_report({(0, 1): 5.0, (1, 2): 6.0})
        assert r.n_pairs == 2

    def test_mean_score_in_range(self):
        r = build_gap_report({(0, 1): 5.0, (1, 2): 5.5})
        assert 0.0 <= r.mean_score <= 1.0

    def test_none_cfg(self):
        r = build_gap_report({(0, 1): 5.0}, cfg=None)
        assert isinstance(r, GapReport)


# ─── filter_gap_measures ──────────────────────────────────────────────────────

class TestFilterGapMeasuresExtra:
    def _report(self):
        ms = [_measure(a=0, b=1, score=0.9),
              _measure(a=1, b=2, score=0.3)]
        return GapReport(measures=ms, mean_score=0.6,
                          total_penalty=0.0, n_acceptable=1)

    def test_returns_list(self):
        assert isinstance(filter_gap_measures(self._report()), list)

    def test_min_score_zero_all_pass(self):
        assert len(filter_gap_measures(self._report(), 0.0)) == 2

    def test_min_score_filters(self):
        filtered = filter_gap_measures(self._report(), 0.5)
        assert len(filtered) == 1
        assert filtered[0].score >= 0.5

    def test_min_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            filter_gap_measures(self._report(), -0.1)

    def test_empty_report(self):
        assert filter_gap_measures(_report()) == []


# ─── worst_gap_pairs ──────────────────────────────────────────────────────────

class TestWorstGapPairsExtra:
    def _report(self):
        ms = [_measure(a=0, b=1, penalty=0.5),
              _measure(a=1, b=2, penalty=2.0),
              _measure(a=2, b=3, penalty=0.1)]
        return GapReport(measures=ms, mean_score=0.5,
                          total_penalty=2.6, n_acceptable=2)

    def test_returns_list(self):
        assert isinstance(worst_gap_pairs(self._report(), 2), list)

    def test_top_k_limit(self):
        assert len(worst_gap_pairs(self._report(), 2)) == 2

    def test_sorted_by_penalty_descending(self):
        pairs = worst_gap_pairs(self._report(), 3)
        penalties = [m.penalty for m in pairs]
        assert penalties == sorted(penalties, reverse=True)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            worst_gap_pairs(self._report(), 0)

    def test_top_k_exceeds_size(self):
        pairs = worst_gap_pairs(self._report(), 100)
        assert len(pairs) == 3


# ─── gap_score_matrix ─────────────────────────────────────────────────────────

class TestGapScoreMatrixExtra:
    def test_returns_dict(self):
        result = gap_score_matrix([0, 1], {(0, 1): 5.0})
        assert isinstance(result, dict)

    def test_key_ordered(self):
        result = gap_score_matrix([0, 1], {(0, 1): 5.0})
        assert (0, 1) in result

    def test_value_in_range(self):
        result = gap_score_matrix([0, 1], {(0, 1): 5.0})
        assert 0.0 <= result[(0, 1)] <= 1.0

    def test_missing_pair_skipped(self):
        result = gap_score_matrix([0, 1, 2], {(0, 1): 5.0})
        assert (1, 2) not in result

    def test_reversed_distance_used(self):
        result = gap_score_matrix([0, 1], {(1, 0): 5.0})
        assert (0, 1) in result

    def test_none_cfg(self):
        result = gap_score_matrix([0, 1], {(0, 1): 5.0}, cfg=None)
        assert isinstance(result, dict)
