"""Additional tests for puzzle_reconstruction/assembly/gamma_optimizer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)
from puzzle_reconstruction.assembly.gamma_optimizer import (
    GammaEdgeModel,
    gamma_optimizer,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(eid: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.TOP,
        virtual_curve=np.array([[float(i), 0.0] for i in range(8)]),
        fd=1.3,
        css_vec=np.zeros(4),
        ifs_coeffs=np.zeros(4),
        length=20.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        mask=np.zeros((16, 16), dtype=np.uint8),
        contour=np.zeros((4, 2)),
        edges=[_edge(fid * 10 + i) for i in range(n_edges)],
    )


def _entry(fi: Fragment, fj: Fragment, score: float = 0.6,
           dtw_dist: float = 0.3) -> CompatEntry:
    return CompatEntry(
        edge_i=fi.edges[0],
        edge_j=fj.edges[0],
        score=score,
        dtw_dist=dtw_dist,
        css_sim=0.7,
        fd_diff=0.1,
        text_score=0.5,
    )


def _asm(frags) -> Assembly:
    return Assembly(
        fragments=frags,
        placements={f.fragment_id: (np.array([float(i * 40), 0.0]), 0.0)
                    for i, f in enumerate(frags)},
        compat_matrix=np.zeros((len(frags), len(frags))),
        total_score=0.0,
    )


# ─── TestGammaEdgeModelExtra ──────────────────────────────────────────────────

class TestGammaEdgeModelExtra:
    def test_fit_with_near_zero_all_removed(self):
        """All deviations ≤ 1e-10 → fewer than 5 kept → defaults stay."""
        m = GammaEdgeModel(k=2.0, theta=0.5)
        m.fit(np.full(20, 1e-12))
        assert m.k == pytest.approx(2.0)
        assert m.theta == pytest.approx(0.5)

    def test_fit_with_exactly_5_samples_updates(self):
        """Exactly 5 non-zero values → fit should run (or at least not crash)."""
        m = GammaEdgeModel()
        devs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        m.fit(devs)
        assert m.k > 0.0
        assert m.theta > 0.0

    def test_log_likelihood_increases_near_mode(self):
        """Mode of Gamma(k, theta) is (k-1)*theta. Values near mode → higher LL."""
        m = GammaEdgeModel(k=3.0, theta=1.0)
        mode = (m.k - 1) * m.theta  # = 2.0
        ll_near = m.log_likelihood(np.full(10, mode))
        ll_far = m.log_likelihood(np.full(10, 100.0))
        assert ll_near > ll_far

    def test_pair_score_non_inf_for_nonzero_curves(self):
        m = GammaEdgeModel()
        a = np.array([[float(i), 0.0] for i in range(10)])
        b = np.array([[float(i), 0.1] for i in range(10)])
        score = m.pair_score(a, b)
        assert np.isfinite(score)

    def test_pair_score_closer_curves_higher(self):
        """Curves that are closer should yield a higher pair score."""
        m = GammaEdgeModel(k=2.0, theta=0.5)
        base = np.array([[float(i), 0.0] for i in range(10)])
        close = np.array([[float(i), 0.01] for i in range(10)])
        far = np.array([[float(i), 5.0] for i in range(10)])
        s_close = m.pair_score(base, close)
        s_far = m.pair_score(base, far)
        assert s_close >= s_far

    def test_pair_score_unequal_lengths_no_crash(self):
        m = GammaEdgeModel()
        a = np.zeros((5, 2))
        b = np.zeros((12, 2))
        result = m.pair_score(a, b)
        assert isinstance(result, float)

    def test_log_likelihood_all_same_value(self):
        m = GammaEdgeModel(k=2.0, theta=1.0)
        ll = m.log_likelihood(np.ones(20) * 1.5)
        assert np.isfinite(ll)

    def test_fit_returns_self_identity(self):
        m = GammaEdgeModel()
        devs = np.abs(np.random.default_rng(1).standard_normal(30)) + 0.1
        returned = m.fit(devs)
        assert returned is m


# ─── TestGammaOptimizerExtra ──────────────────────────────────────────────────

class TestGammaOptimizerExtra:
    def test_result_fragments_is_same_list(self):
        frags = [_frag(i) for i in range(3)]
        result = gamma_optimizer(frags, [], n_iter=5, seed=0)
        assert result.fragments is frags

    def test_placement_positions_are_2d(self):
        frags = [_frag(i) for i in range(3)]
        result = gamma_optimizer(frags, [], n_iter=5, seed=1)
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_placement_angles_are_float(self):
        frags = [_frag(i) for i in range(3)]
        result = gamma_optimizer(frags, [], n_iter=5, seed=2)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(float(angle), float)

    def test_with_init_assembly(self):
        frags = [_frag(i) for i in range(3)]
        init = _asm(frags)
        result = gamma_optimizer(frags, [], n_iter=5,
                                 init_assembly=init, seed=3)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 3

    def test_init_assembly_preserves_all_fragments(self):
        frags = [_frag(i) for i in range(4)]
        init = _asm(frags)
        result = gamma_optimizer(frags, [_entry(frags[0], frags[1])],
                                 n_iter=10, init_assembly=init, seed=4)
        assert set(result.placements.keys()) == {f.fragment_id for f in frags}

    def test_different_seeds_no_crash(self):
        frags = [_frag(i) for i in range(4)]
        entries = [_entry(frags[i], frags[(i + 1) % 4]) for i in range(4)]
        for seed in [0, 1, 99]:
            result = gamma_optimizer(frags, entries, n_iter=10, seed=seed)
            assert isinstance(result.total_score, float)

    def test_total_score_is_finite(self):
        frags = [_frag(i) for i in range(3)]
        result = gamma_optimizer(frags, [], n_iter=10, seed=5)
        assert np.isfinite(result.total_score)

    def test_large_fragment_set(self):
        frags = [_frag(i) for i in range(8)]
        result = gamma_optimizer(frags, [], n_iter=10, seed=6)
        assert len(result.placements) == 8

    def test_many_entries_mle_fit_triggered(self):
        frags = [_frag(i) for i in range(5)]
        entries = [_entry(frags[i % 5], frags[(i + 1) % 5],
                          dtw_dist=float(i + 1) * 0.1)
                   for i in range(12)]
        result = gamma_optimizer(frags, entries, n_iter=15, seed=7)
        assert len(result.placements) == 5

    def test_zero_iterations_still_returns_assembly(self):
        frags = [_frag(i) for i in range(2)]
        result = gamma_optimizer(frags, [], n_iter=0, seed=8)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 2

    def test_single_fragment_placed(self):
        frags = [_frag(0)]
        result = gamma_optimizer(frags, [], n_iter=3, seed=9)
        assert 0 in result.placements
