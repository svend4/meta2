"""Extra tests for puzzle_reconstruction/utils/edge_scorer.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.edge_scorer import (
    EdgeScoreConfig,
    EdgeScoreResult,
    score_edge_overlap,
    score_edge_curvature,
    score_edge_length,
    score_edge_endpoints,
    aggregate_edge_scores,
    rank_edge_pairs,
    batch_score_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _line(n=8, x_offset=0.0) -> np.ndarray:
    x = np.linspace(0.0 + x_offset, 1.0 + x_offset, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def _esr(overlap=0.8, curv=0.7, length=0.9, endpoints=0.85, total=0.8) -> EdgeScoreResult:
    return EdgeScoreResult(overlap=overlap, curvature=curv,
                            length=length, endpoints=endpoints, total=total)


# ─── EdgeScoreConfig ──────────────────────────────────────────────────────────

class TestEdgeScoreConfigExtra:
    def test_default_n_samples(self):
        assert EdgeScoreConfig().n_samples == 64

    def test_default_length_tol(self):
        assert EdgeScoreConfig().length_tol == pytest.approx(0.5)

    def test_default_endpoint_sigma(self):
        assert EdgeScoreConfig().endpoint_sigma == pytest.approx(10.0)

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            EdgeScoreConfig(n_samples=1)

    def test_length_tol_negative_raises(self):
        with pytest.raises(ValueError):
            EdgeScoreConfig(length_tol=-0.1)

    def test_endpoint_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeScoreConfig(endpoint_sigma=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            EdgeScoreConfig(weights={"overlap": -0.1, "curvature": 0.3,
                                     "length": 0.3, "endpoints": 0.3})

    def test_normalized_weights_sum_one(self):
        cfg = EdgeScoreConfig()
        total = sum(cfg.normalized_weights.values())
        assert total == pytest.approx(1.0)

    def test_custom_weights_normalize(self):
        cfg = EdgeScoreConfig(weights={"overlap": 2.0, "curvature": 2.0,
                                        "length": 0.0, "endpoints": 0.0})
        nw = cfg.normalized_weights
        assert nw["overlap"] == pytest.approx(0.5)


# ─── EdgeScoreResult ──────────────────────────────────────────────────────────

class TestEdgeScoreResultExtra:
    def test_stores_all_fields(self):
        r = _esr()
        assert r.overlap == pytest.approx(0.8)
        assert r.curvature == pytest.approx(0.7)
        assert r.length == pytest.approx(0.9)
        assert r.endpoints == pytest.approx(0.85)
        assert r.total == pytest.approx(0.8)

    def test_to_dict(self):
        d = _esr().to_dict()
        for k in ("overlap", "curvature", "length", "endpoints", "total"):
            assert k in d

    def test_to_dict_values(self):
        d = _esr(overlap=0.6).to_dict()
        assert d["overlap"] == pytest.approx(0.6)


# ─── score_edge_overlap ───────────────────────────────────────────────────────

class TestScoreEdgeOverlapExtra:
    def test_returns_float(self):
        c = _line()
        assert isinstance(score_edge_overlap(c, c), float)

    def test_reversed_near_one(self):
        """Overlap is highest when b is the reverse of a (mirrored edge)."""
        c = _line()
        score = score_edge_overlap(c, c[::-1])
        assert score >= 0.9

    def test_in_range(self):
        score = score_edge_overlap(_line(), _line(x_offset=5.0))
        assert 0.0 <= score <= 1.0

    def test_empty_curve_returns_zero(self):
        score = score_edge_overlap(np.zeros((0, 2)), _line())
        assert score == pytest.approx(0.0)


# ─── score_edge_curvature ─────────────────────────────────────────────────────

class TestScoreEdgeCurvatureExtra:
    def test_returns_float(self):
        c = _line()
        assert isinstance(score_edge_curvature(c, c), float)

    def test_identical_near_one(self):
        c = _line(n=16)
        score = score_edge_curvature(c, c)
        assert score >= 0.9

    def test_in_range(self):
        score = score_edge_curvature(_line(), _line(x_offset=1.0))
        assert 0.0 <= score <= 1.0


# ─── score_edge_length ────────────────────────────────────────────────────────

class TestScoreEdgeLengthExtra:
    def test_returns_float(self):
        c = _line()
        assert isinstance(score_edge_length(c, c), float)

    def test_identical_is_one(self):
        c = _line()
        assert score_edge_length(c, c) == pytest.approx(1.0, abs=1e-5)

    def test_nonneg(self):
        assert score_edge_length(_line(), _line(n=4)) >= 0.0


# ─── score_edge_endpoints ─────────────────────────────────────────────────────

class TestScoreEdgeEndpointsExtra:
    def test_returns_float(self):
        c = _line()
        assert isinstance(score_edge_endpoints(c, c), float)

    def test_identical_near_one(self):
        c = _line()
        score = score_edge_endpoints(c, c)
        assert score >= 0.9

    def test_in_range(self):
        score = score_edge_endpoints(_line(), _line(x_offset=100.0))
        assert 0.0 <= score <= 1.0


# ─── aggregate_edge_scores ────────────────────────────────────────────────────

class TestAggregateEdgeScoresExtra:
    def test_returns_float(self):
        result = aggregate_edge_scores(0.8, 0.7, 0.9, 0.85)
        assert isinstance(result, float)

    def test_total_in_range(self):
        r = aggregate_edge_scores(0.5, 0.6, 0.7, 0.8)
        assert 0.0 <= r <= 1.0

    def test_all_ones_is_one(self):
        r = aggregate_edge_scores(1.0, 1.0, 1.0, 1.0)
        assert r == pytest.approx(1.0)

    def test_all_zeros_is_zero(self):
        r = aggregate_edge_scores(0.0, 0.0, 0.0, 0.0)
        assert r == pytest.approx(0.0)


# ─── rank_edge_pairs ──────────────────────────────────────────────────────────

class TestRankEdgePairsExtra:
    def test_returns_list(self):
        results = [(0, 1, _esr(total=0.8)), (1, 2, _esr(total=0.5))]
        assert isinstance(rank_edge_pairs(results), list)

    def test_descending_order(self):
        results = [(0, 1, _esr(total=0.3)), (1, 2, _esr(total=0.9))]
        ranked = rank_edge_pairs(results)
        totals = [r[2].total for r in ranked]
        assert totals == sorted(totals, reverse=True)

    def test_empty_returns_empty(self):
        assert rank_edge_pairs([]) == []


# ─── batch_score_edges ────────────────────────────────────────────────────────

class TestBatchScoreEdgesExtra:
    def test_returns_list(self):
        result = batch_score_edges([_line()], [_line()])
        assert isinstance(result, list)

    def test_length_matches(self):
        a_list = [_line(), _line(x_offset=1.0)]
        b_list = [_line(), _line()]
        assert len(batch_score_edges(a_list, b_list)) == 2

    def test_each_is_result(self):
        for r in batch_score_edges([_line()], [_line()]):
            assert isinstance(r, EdgeScoreResult)

    def test_empty_returns_empty(self):
        assert batch_score_edges([], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            batch_score_edges([_line(), _line()], [_line()])
