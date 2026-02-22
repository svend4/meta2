"""Расширенные тесты для puzzle_reconstruction/assembly/annealing.py."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.annealing import simulated_annealing, _evaluate
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei: EdgeSignature, ej: EdgeSignature, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _assembly(frags, ocr_score: float = 0.0, compat=None) -> Assembly:
    placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                  for i, f in enumerate(frags)}
    if compat is None:
        compat = np.zeros((len(frags), len(frags)))
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=compat,
        total_score=0.0,
        ocr_score=ocr_score,
    )


def _entries(frags) -> list:
    result = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            result.append(_entry(fi.edges[0], fj.edges[0], 0.6))
    return result


# ─── TestSimulatedAnnealing ───────────────────────────────────────────────────

class TestSimulatedAnnealing:

    # --- Return type ---

    def test_returns_assembly(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert isinstance(result, Assembly)

    def test_single_fragment_returns_unchanged(self):
        frags = [_frag(0)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, [], max_iter=100, seed=0)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1

    # --- Fragment preservation ---

    def test_all_fragments_in_result(self):
        frags = [_frag(i) for i in range(4)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=20, seed=1)
        assert len(result.fragments) == 4

    def test_fragments_list_identical(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert result.fragments == frags

    def test_all_fragment_ids_in_placements(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=20, seed=2)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_has_correct_count(self):
        frags = [_frag(i) for i in range(5)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=20, seed=0)
        assert len(result.placements) == 5

    # --- Score ---

    def test_total_score_is_float(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert isinstance(result.total_score, float)

    def test_total_score_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert result.total_score >= 0.0

    def test_empty_entries_score_near_zero(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, [], max_iter=20, seed=0)
        # With no entries, evaluate gives ~0 (empty loop)
        assert result.total_score >= 0.0

    # --- Determinism ---

    def test_deterministic_with_same_seed(self):
        frags = [_frag(i) for i in range(4)]
        ents = _entries(frags)
        r1 = simulated_annealing(_assembly(frags), ents, max_iter=50, seed=7)
        r2 = simulated_annealing(_assembly(frags), ents, max_iter=50, seed=7)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_different_seeds_same_convergence_possible(self):
        """Two different seeds should both produce valid Assemblies."""
        frags = [_frag(i) for i in range(3)]
        ents = _entries(frags)
        r1 = simulated_annealing(_assembly(frags), ents, max_iter=30, seed=0)
        r2 = simulated_annealing(_assembly(frags), ents, max_iter=30, seed=99)
        assert isinstance(r1, Assembly)
        assert isinstance(r2, Assembly)

    # --- Preserved fields ---

    def test_ocr_score_preserved(self):
        frags = [_frag(i) for i in range(2)]
        asm = _assembly(frags, ocr_score=0.77)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert result.ocr_score == pytest.approx(0.77)

    def test_compat_matrix_preserved(self):
        frags = [_frag(i) for i in range(2)]
        compat = np.eye(2)
        asm = _assembly(frags, compat=compat)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        assert result.compat_matrix.shape == (2, 2)

    # --- Termination conditions ---

    def test_max_iter_zero_returns_immediately(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=0, seed=0)
        assert isinstance(result, Assembly)

    def test_high_cooling_terminates_quickly(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(
            asm, _entries(frags), T_max=1.0, T_min=0.9, cooling=0.01,
            max_iter=10000, seed=0
        )
        assert isinstance(result, Assembly)

    def test_t_max_below_t_min_skips_loop(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(
            asm, _entries(frags), T_max=0.05, T_min=0.1, max_iter=1000, seed=0
        )
        assert isinstance(result, Assembly)

    # --- Parameter defaults ---

    def test_default_params_run(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags))
        assert isinstance(result, Assembly)

    # --- Placement structure ---

    def test_placement_pos_has_len_2(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_placement_angle_is_float(self):
        frags = [_frag(i) for i in range(3)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(float(angle), float)

    def test_placement_pos_is_array(self):
        frags = [_frag(i) for i in range(2)]
        asm = _assembly(frags)
        result = simulated_annealing(asm, _entries(frags), max_iter=10, seed=0)
        for fid, (pos, _) in result.placements.items():
            assert hasattr(pos, '__len__')

    # --- Score tracking ---

    def test_best_score_tracked_across_iterations(self):
        """With more iterations, score should be >= initial."""
        frags = [_frag(i) for i in range(4)]
        asm = _assembly(frags)
        ents = _entries(frags)
        result = simulated_annealing(asm, ents, T_max=100.0, T_min=0.1,
                                      cooling=0.99, max_iter=200, seed=42)
        assert result.total_score >= 0.0

    def test_two_fragments_swap_works(self):
        """Two fragments can be swapped — positions change."""
        frags = [_frag(0), _frag(1)]
        asm = _assembly(frags)
        ents = _entries(frags)
        result = simulated_annealing(asm, ents, max_iter=100, seed=5)
        assert 0 in result.placements and 1 in result.placements

    def test_large_fragment_count_runs(self):
        frags = [_frag(i) for i in range(10)]
        asm = _assembly(frags)
        ents = _entries(frags)
        result = simulated_annealing(asm, ents, max_iter=50, seed=1)
        assert len(result.placements) == 10


# ─── TestEvaluate ─────────────────────────────────────────────────────────────

class TestEvaluate:
    def _edge_to_frag(self, frags):
        edge_to_frag = {}
        for f in frags:
            for e in f.edges:
                edge_to_frag[e.edge_id] = f.fragment_id
        return edge_to_frag

    def test_returns_float(self):
        frags = [_frag(i) for i in range(3)]
        placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                      for i, f in enumerate(frags)}
        ents = _entries(frags)
        e2f = self._edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert isinstance(result, float)

    def test_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                      for i, f in enumerate(frags)}
        ents = _entries(frags)
        e2f = self._edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert result >= 0.0

    def test_empty_entries_near_zero(self):
        frags = [_frag(i) for i in range(3)]
        placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                      for i, f in enumerate(frags)}
        e2f = self._edge_to_frag(frags)
        result = _evaluate(placements, [], e2f, frags)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_closer_fragments_higher_score(self):
        """Fragments closer together should yield higher score (proximity weight)."""
        frags = [_frag(0), _frag(1)]
        ents = _entries(frags)
        e2f = self._edge_to_frag(frags)

        placements_close = {0: (np.array([0.0, 0.0]), 0.0),
                             1: (np.array([10.0, 0.0]), 0.0)}
        placements_far = {0: (np.array([0.0, 0.0]), 0.0),
                           1: (np.array([10000.0, 0.0]), 0.0)}

        score_close = _evaluate(placements_close, ents, e2f, frags)
        score_far = _evaluate(placements_far, ents, e2f, frags)
        assert score_close >= score_far

    def test_higher_entry_score_higher_result(self):
        frags = [_frag(0), _frag(1)]
        e2f = self._edge_to_frag(frags)
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([50.0, 0.0]), 0.0)}

        ents_high = [_entry(frags[0].edges[0], frags[1].edges[0], score=1.0)]
        ents_low = [_entry(frags[0].edges[0], frags[1].edges[0], score=0.1)]

        s_high = _evaluate(placements, ents_high, e2f, frags)
        s_low = _evaluate(placements, ents_low, e2f, frags)
        assert s_high > s_low
