"""Additional tests for puzzle_reconstruction/assembly/ant_colony.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)
from puzzle_reconstruction.assembly.ant_colony import ant_colony_assembly


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(eid: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((5, 2), dtype=np.float64),
        fd=1.2,
        css_vec=np.zeros(4),
        ifs_coeffs=np.zeros(4),
        length=10.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        mask=np.zeros((16, 16), dtype=np.uint8),
        contour=np.zeros((4, 2)),
        edges=[_edge(fid * 10 + i) for i in range(n_edges)],
    )


def _entry(fi: Fragment, fj: Fragment, score: float = 0.6) -> CompatEntry:
    return CompatEntry(
        edge_i=fi.edges[0],
        edge_j=fj.edges[0],
        score=score,
        dtw_dist=0.2,
        css_sim=0.8,
        fd_diff=0.05,
        text_score=0.7,
    )


# ─── TestAntColonyAssemblyExtra ───────────────────────────────────────────────

class TestAntColonyAssemblyExtra:
    def test_result_fragments_is_same_list(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=0)
        assert result.fragments is frags

    def test_placement_positions_are_2d(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=1)
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_placement_keys_match_fragment_ids(self):
        frags = [_frag(i * 3) for i in range(4)]
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=2)
        assert set(result.placements.keys()) == {f.fragment_id for f in frags}

    def test_no_rotation_all_angles_zero(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=3,
                                     allow_rotation=False, seed=3)
        for fid, (pos, angle) in result.placements.items():
            assert angle == pytest.approx(0.0)

    def test_different_seeds_may_differ(self):
        frags = [_frag(i) for i in range(4)]
        entries = [_entry(frags[0], frags[1])]
        r1 = ant_colony_assembly(frags, entries, n_iterations=5, seed=10)
        r2 = ant_colony_assembly(frags, entries, n_iterations=5, seed=99)
        assert isinstance(r1.total_score, float)
        assert isinstance(r2.total_score, float)

    def test_large_fragment_set(self):
        frags = [_frag(i) for i in range(8)]
        result = ant_colony_assembly(frags, [], n_iterations=3, seed=5)
        assert len(result.placements) == 8

    def test_n_ants_1_no_crash(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_ants=1, n_iterations=2, seed=6)
        assert isinstance(result, Assembly)

    def test_elite_count_0_no_crash(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2,
                                     elite_count=0, seed=7)
        assert isinstance(result, Assembly)

    def test_alpha_0_no_crash(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2,
                                     alpha=0.0, seed=8)
        assert isinstance(result, Assembly)

    def test_beta_0_no_crash(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2,
                                     beta=0.0, seed=9)
        assert isinstance(result, Assembly)

    def test_rho_0_no_evaporation_no_crash(self):
        """rho=0 means no evaporation; pheromone only grows."""
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=3,
                                     rho=0.0, seed=10)
        assert isinstance(result, Assembly)

    def test_total_score_finite(self):
        frags = [_frag(i) for i in range(4)]
        entries = [_entry(frags[i], frags[(i + 1) % 4]) for i in range(4)]
        result = ant_colony_assembly(frags, entries, n_iterations=5, seed=11)
        assert np.isfinite(result.total_score)

    def test_placements_positions_are_ndarrays(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=12)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(pos, np.ndarray)

    def test_q_param_no_crash(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2,
                                     Q=10.0, seed=13)
        assert isinstance(result, Assembly)

    def test_with_unknown_edge_entry_no_crash(self):
        frags = [_frag(0), _frag(1)]
        unknown_edge = _edge(999)
        bad_entry = CompatEntry(
            edge_i=frags[0].edges[0],
            edge_j=unknown_edge,
            score=0.9,
            dtw_dist=0.1,
            css_sim=0.9,
            fd_diff=0.05,
            text_score=0.5,
        )
        result = ant_colony_assembly(frags, [bad_entry], n_iterations=2, seed=14)
        assert isinstance(result, Assembly)
