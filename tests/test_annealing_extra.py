"""Extra tests for puzzle_reconstruction.assembly.annealing."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

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


def _edge_to_frag(frags):
    e2f = {}
    for f in frags:
        for e in f.edges:
            e2f[e.edge_id] = f.fragment_id
    return e2f


# ─── TestSimulatedAnnealingExtra ──────────────────────────────────────────────

class TestSimulatedAnnealingExtra:
    def test_returns_assembly_type(self):
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=5, seed=0)
        assert type(result).__name__ == "Assembly"

    def test_placements_keys_are_fragment_ids(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=10, seed=0)
        expected = {f.fragment_id for f in frags}
        assert set(result.placements.keys()) == expected

    def test_total_score_finite(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=10, seed=0)
        assert np.isfinite(result.total_score)

    def test_compat_matrix_shape_preserved(self):
        frags = [_frag(i) for i in range(4)]
        compat = np.eye(4, dtype=np.float32)
        result = simulated_annealing(_assembly(frags, compat=compat),
                                     _entries(frags), max_iter=10, seed=0)
        assert result.compat_matrix.shape == (4, 4)

    def test_ocr_score_0_preserved(self):
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_assembly(frags, ocr_score=0.0),
                                     _entries(frags), max_iter=5, seed=0)
        assert result.ocr_score == pytest.approx(0.0)

    def test_ocr_score_1_preserved(self):
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_assembly(frags, ocr_score=1.0),
                                     _entries(frags), max_iter=5, seed=0)
        assert result.ocr_score == pytest.approx(1.0)

    def test_five_fragments_all_placed(self):
        frags = [_frag(i) for i in range(5)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=20, seed=3)
        assert len(result.placements) == 5

    def test_seed_42_reproducible(self):
        frags = [_frag(i) for i in range(3)]
        ents = _entries(frags)
        r1 = simulated_annealing(_assembly(frags), ents, max_iter=30, seed=42)
        r2 = simulated_annealing(_assembly(frags), ents, max_iter=30, seed=42)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_different_seeds_produce_assembly(self):
        frags = [_frag(i) for i in range(3)]
        ents = _entries(frags)
        for seed in [0, 1, 2, 100]:
            result = simulated_annealing(_assembly(frags), ents,
                                         max_iter=10, seed=seed)
            assert isinstance(result, Assembly)

    def test_placement_pos_array_like(self):
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=5, seed=0)
        for fid, (pos, _angle) in result.placements.items():
            assert hasattr(pos, '__len__')
            assert len(pos) == 2

    def test_placement_angle_numeric(self):
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=5, seed=0)
        for _fid, (_pos, angle) in result.placements.items():
            assert isinstance(float(angle), float)

    def test_high_temperature_still_terminates(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     T_max=1e6, T_min=1.0, cooling=0.5,
                                     max_iter=50, seed=0)
        assert isinstance(result, Assembly)

    def test_very_low_temperature_terminates(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     T_max=1e-6, T_min=1e-7, cooling=0.99,
                                     max_iter=100, seed=0)
        assert isinstance(result, Assembly)

    def test_all_edges_4_per_fragment(self):
        frags = [_frag(i, n_edges=4) for i in range(3)]
        ents = _entries(frags)
        result = simulated_annealing(_assembly(frags), ents, max_iter=20, seed=0)
        assert len(result.fragments) == 3

    def test_score_with_high_compat_entries(self):
        frags = [_frag(0), _frag(1)]
        ents = [_entry(frags[0].edges[0], frags[1].edges[0], score=1.0)]
        result = simulated_annealing(_assembly(frags), ents, max_iter=50,
                                     T_max=10.0, T_min=0.1, cooling=0.9, seed=0)
        assert result.total_score >= 0.0

    def test_score_with_zero_compat_entries(self):
        frags = [_frag(0), _frag(1)]
        ents = [_entry(frags[0].edges[0], frags[1].edges[0], score=0.0)]
        result = simulated_annealing(_assembly(frags), ents, max_iter=20, seed=0)
        assert result.total_score >= 0.0

    def test_fragments_unchanged_objects(self):
        frags = [_frag(i) for i in range(3)]
        orig_ids = [f.fragment_id for f in frags]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=10, seed=0)
        assert [f.fragment_id for f in result.fragments] == orig_ids

    def test_eight_fragments_runs(self):
        frags = [_frag(i) for i in range(8)]
        result = simulated_annealing(_assembly(frags), _entries(frags),
                                     max_iter=30, seed=0)
        assert len(result.placements) == 8


# ─── TestEvaluateExtra ────────────────────────────────────────────────────────

class TestEvaluateExtra:
    def test_returns_non_negative(self):
        frags = [_frag(0), _frag(1)]
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([100.0, 0.0]), 0.0)}
        ents = _entries(frags)
        e2f = _edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert result >= 0.0

    def test_returns_float_type(self):
        frags = [_frag(0), _frag(1)]
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([50.0, 0.0]), 0.0)}
        e2f = _edge_to_frag(frags)
        result = _evaluate(placements, [], e2f, frags)
        assert isinstance(result, float)

    def test_empty_entries_zero(self):
        frags = [_frag(i) for i in range(3)]
        placements = {f.fragment_id: (np.array([float(i * 50), 0.0]), 0.0)
                      for i, f in enumerate(frags)}
        e2f = _edge_to_frag(frags)
        assert _evaluate(placements, [], e2f, frags) == pytest.approx(0.0)

    def test_score_1_entry_positive(self):
        frags = [_frag(0), _frag(1)]
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([10.0, 0.0]), 0.0)}
        ents = [_entry(frags[0].edges[0], frags[1].edges[0], score=1.0)]
        e2f = _edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert result >= 0.0

    def test_far_apart_less_than_close(self):
        frags = [_frag(0), _frag(1)]
        ents = _entries(frags)
        e2f = _edge_to_frag(frags)
        p_close = {0: (np.array([0.0, 0.0]), 0.0),
                   1: (np.array([5.0, 0.0]), 0.0)}
        p_far = {0: (np.array([0.0, 0.0]), 0.0),
                 1: (np.array([5000.0, 0.0]), 0.0)}
        s_close = _evaluate(p_close, ents, e2f, frags)
        s_far = _evaluate(p_far, ents, e2f, frags)
        assert s_close >= s_far

    def test_three_frags_returns_float(self):
        frags = [_frag(i) for i in range(3)]
        placements = {i: (np.array([float(i * 30), 0.0]), 0.0) for i in range(3)}
        ents = _entries(frags)
        e2f = _edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert isinstance(result, float)

    def test_score_finite(self):
        frags = [_frag(i) for i in range(4)]
        placements = {f.fragment_id: (np.array([float(i * 50), 0.0]), 0.0)
                      for i, f in enumerate(frags)}
        ents = _entries(frags)
        e2f = _edge_to_frag(frags)
        result = _evaluate(placements, ents, e2f, frags)
        assert np.isfinite(result)
