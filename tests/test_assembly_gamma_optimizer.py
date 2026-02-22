"""Tests for puzzle_reconstruction/assembly/gamma_optimizer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSignature,
    EdgeSide,
    Fragment,
)
from puzzle_reconstruction.assembly.gamma_optimizer import (
    GammaEdgeModel,
    gamma_optimizer,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((8, 2), dtype=np.float64),
        fd=1.2,
        css_vec=np.zeros(4, dtype=np.float64),
        ifs_coeffs=np.zeros(4, dtype=np.float64),
        length=10.0,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.zeros((20, 20), dtype=np.uint8),
        contour=np.zeros((4, 2), dtype=np.float64),
        edges=[_make_edge(fid * 10 + i) for i in range(n_edges)],
    )


def _make_compat(fi: Fragment, fj: Fragment,
                 score: float = 0.5, dtw_dist: float = 0.2) -> CompatEntry:
    return CompatEntry(
        edge_i=fi.edges[0],
        edge_j=fj.edges[0],
        score=score,
        dtw_dist=dtw_dist,
        css_sim=0.8,
        fd_diff=0.1,
        text_score=0.5,
    )


def _frags(n: int):
    return [_make_fragment(i) for i in range(n)]


# ─── TestGammaEdgeModel ───────────────────────────────────────────────────────

class TestGammaEdgeModel:
    def test_construction_defaults(self):
        m = GammaEdgeModel()
        assert m.k == pytest.approx(2.0)
        assert m.theta == pytest.approx(0.5)

    def test_construction_custom(self):
        m = GammaEdgeModel(k=3.0, theta=1.5)
        assert m.k == pytest.approx(3.0)
        assert m.theta == pytest.approx(1.5)

    def test_fit_returns_self(self):
        m = GammaEdgeModel()
        devs = np.abs(np.random.randn(20)) + 0.1
        result = m.fit(devs)
        assert result is m

    def test_fit_updates_params(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        devs = np.abs(np.random.default_rng(0).standard_normal(50)) * 2 + 1.0
        m.fit(devs)
        # params should have changed (or at least be positive)
        assert m.k > 0.0
        assert m.theta > 0.0

    def test_fit_too_few_samples_keeps_defaults(self):
        m = GammaEdgeModel(k=2.0, theta=0.5)
        m.fit(np.array([0.1, 0.2, 0.3]))  # < 5 non-zero
        assert m.k == pytest.approx(2.0)
        assert m.theta == pytest.approx(0.5)

    def test_log_likelihood_returns_float(self):
        m = GammaEdgeModel()
        ll = m.log_likelihood(np.array([0.5, 1.0, 1.5]))
        assert isinstance(ll, float)

    def test_log_likelihood_finite(self):
        m = GammaEdgeModel()
        ll = m.log_likelihood(np.array([0.1, 0.5, 1.0, 2.0]))
        assert np.isfinite(ll)

    def test_log_likelihood_positive_deviations(self):
        m = GammaEdgeModel(k=2.0, theta=1.0)
        ll_low = m.log_likelihood(np.full(5, 0.01))
        ll_high = m.log_likelihood(np.full(5, 10.0))
        # Both should be finite floats
        assert np.isfinite(ll_low)
        assert np.isfinite(ll_high)

    def test_pair_score_returns_float(self):
        m = GammaEdgeModel()
        a = np.zeros((10, 2))
        b = np.zeros((10, 2))
        result = m.pair_score(a, b)
        assert isinstance(result, float)

    def test_pair_score_identical_edges(self):
        m = GammaEdgeModel()
        a = np.array([[float(i), 0.0] for i in range(10)])
        score = m.pair_score(a, a)
        assert np.isfinite(score)

    def test_pair_score_empty_returns_neg_inf(self):
        m = GammaEdgeModel()
        score = m.pair_score(np.zeros((0, 2)), np.zeros((0, 2)))
        assert score == -np.inf

    def test_pair_score_different_lengths(self):
        m = GammaEdgeModel()
        a = np.zeros((5, 2))
        b = np.zeros((8, 2))
        score = m.pair_score(a, b)
        assert isinstance(score, float)


# ─── TestGammaOptimizer ───────────────────────────────────────────────────────

class TestGammaOptimizer:
    def test_empty_fragments_returns_assembly(self):
        result = gamma_optimizer([], [], n_iter=1)
        assert isinstance(result, Assembly)

    def test_empty_fragments_empty_list(self):
        result = gamma_optimizer([], [], n_iter=1)
        assert result.fragments == []

    def test_returns_assembly_type(self):
        frags = _frags(3)
        result = gamma_optimizer(frags, [], n_iter=5, seed=0)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = _frags(3)
        result = gamma_optimizer(frags, [], n_iter=5, seed=1)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_count_matches_fragments(self):
        frags = _frags(4)
        result = gamma_optimizer(frags, [], n_iter=3, seed=2)
        assert len(result.placements) == 4

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = gamma_optimizer(frags, [], n_iter=2, seed=3)
        assert 0 in result.placements

    def test_with_compat_entries(self):
        frags = _frags(3)
        entries = [
            _make_compat(frags[0], frags[1], score=0.9, dtw_dist=0.1),
            _make_compat(frags[1], frags[2], score=0.7, dtw_dist=0.3),
        ]
        result = gamma_optimizer(frags, entries, n_iter=10, seed=42)
        assert isinstance(result, Assembly)

    def test_seed_reproducibility(self):
        frags = _frags(3)
        entries = [_make_compat(frags[0], frags[1])]
        r1 = gamma_optimizer(frags, entries, n_iter=5, seed=99)
        r2 = gamma_optimizer(frags, entries, n_iter=5, seed=99)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_placements_have_pos_and_angle(self):
        frags = _frags(2)
        result = gamma_optimizer(frags, [], n_iter=2, seed=4)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(pos, np.ndarray)
            assert isinstance(angle, float)

    def test_no_entries_runs_ok(self):
        frags = _frags(5)
        result = gamma_optimizer(frags, [], n_iter=3, seed=5)
        assert len(result.placements) == 5

    def test_few_iterations_ok(self):
        frags = _frags(2)
        result = gamma_optimizer(frags, [], n_iter=1, seed=6)
        assert isinstance(result, Assembly)

    def test_compat_entries_dtw_fit(self):
        """Many entries with varied dtw_dist values should trigger MLE fit."""
        frags = _frags(4)
        entries = [
            _make_compat(frags[i % 4], frags[(i + 1) % 4],
                         score=0.5, dtw_dist=float(i + 1) * 0.2)
            for i in range(10)
        ]
        result = gamma_optimizer(frags, entries, n_iter=5, seed=7)
        assert isinstance(result, Assembly)
