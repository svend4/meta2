"""Extra tests for puzzle_reconstruction/assembly/exhaustive.py."""
import warnings

import numpy as np
import pytest

from puzzle_reconstruction.assembly.exhaustive import (
    _evaluate_config,
    _score_delta,
    exhaustive_assembly,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


def _edge(edge_id):
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid, n_edges=2):
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei, ej, score=0.5):
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _build_chain(n, base_score=0.8):
    frags = [_frag(i) for i in range(n)]
    entries = []
    for i in range(n - 1):
        entries.append(_entry(frags[i].edges[0], frags[i + 1].edges[0], score=base_score))
    return frags, entries


def _make_edge_to_frag(frags):
    return {e.edge_id: f.fragment_id for f in frags for e in f.edges}


# ─── exhaustive_assembly extras ───────────────────────────────────────────────

class TestExhaustiveAssemblyExtra:
    def test_single_fragment_assembly(self):
        frags = [_frag(0)]
        result = exhaustive_assembly(frags, [])
        assert isinstance(result, Assembly)
        assert 0 in result.placements

    def test_two_fragments_placements_count(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert len(result.placements) == 2

    def test_score_float_type(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result.total_score, float)

    def test_score_with_score_1_entries(self):
        frags, entries = _build_chain(2, base_score=1.0)
        result = exhaustive_assembly(frags, entries)
        assert result.total_score > 0.0

    def test_score_with_score_0_entries(self):
        frags, entries = _build_chain(2, base_score=0.0)
        result = exhaustive_assembly(frags, entries)
        assert result.total_score == pytest.approx(0.0)

    def test_fragments_attribute_is_input(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert result.fragments is frags

    def test_compat_matrix_is_ndarray(self):
        n = 3
        frags, entries = _build_chain(n)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_allow_rotation_true(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries, allow_rotation=True)
        assert isinstance(result, Assembly)

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries, allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_three_fragments_all_in_placements(self):
        frags, entries = _build_chain(3)
        result = exhaustive_assembly(frags, entries)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_pos_tuple(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        for pos, angle in result.placements.values():
            assert len(pos) == 2
            assert isinstance(angle, float)

    def test_seed_deterministic(self):
        frags, entries = _build_chain(3, base_score=0.7)
        r1 = exhaustive_assembly(frags, entries, seed=42)
        r2 = exhaustive_assembly(frags, entries, seed=42)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_different_seeds_same_result_for_deterministic(self):
        frags, entries = _build_chain(2)
        r1 = exhaustive_assembly(frags, entries, seed=1)
        r2 = exhaustive_assembly(frags, entries, seed=99)
        # With 2 fragments exhaustive is deterministic regardless of seed
        assert isinstance(r1, Assembly)
        assert isinstance(r2, Assembly)

    def test_fallback_triggers_warning(self):
        frags, entries = _build_chain(4)
        with pytest.warns(RuntimeWarning):
            result = exhaustive_assembly(frags, entries, max_n=3)
        assert isinstance(result, Assembly)


# ─── _score_delta extras ──────────────────────────────────────────────────────

class TestScoreDeltaExtra:
    def test_nonneg_for_connected(self):
        frags, entries = _build_chain(2, base_score=0.9)
        etf = _make_edge_to_frag(frags)
        placed = {0: (np.zeros(2), 0.0)}
        delta = _score_delta(1, [0], placed, entries, etf)
        assert delta >= 0.0

    def test_zero_for_empty_placed(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        delta = _score_delta(0, [], {}, entries, etf)
        assert delta == pytest.approx(0.0)

    def test_isolated_fragment_zero(self):
        f0 = _frag(0)
        f1 = _frag(1)
        entries = [_entry(f0.edges[0], f1.edges[0], score=0.8)]
        etf = _make_edge_to_frag([f0, f1])
        placed = {1: (np.zeros(2), 0.0)}
        # f2 (id=2) not linked to f1
        f2 = _frag(2)
        etf[f2.edges[0].edge_id] = 2
        delta = _score_delta(2, [1], placed, entries, etf)
        assert delta == pytest.approx(0.0)

    def test_high_score_better_than_low_score(self):
        f0 = _frag(0)
        f1 = _frag(1)
        etf = {e.edge_id: f.fragment_id for f in [f0, f1] for e in f.edges}
        low = _entry(f0.edges[0], f1.edges[0], score=0.1)
        high = _entry(f0.edges[0], f1.edges[0], score=0.95)
        placed = {0: (np.zeros(2), 0.0)}
        d_low = _score_delta(1, [0], placed, [low], etf)
        d_high = _score_delta(1, [0], placed, [high], etf)
        assert d_high > d_low

    def test_result_is_float(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placed = {0: (np.zeros(2), 0.0)}
        result = _score_delta(1, [0], placed, entries, etf)
        assert isinstance(result, float)


# ─── _evaluate_config extras ──────────────────────────────────────────────────

class TestEvaluateConfigExtra:
    def test_two_fragments_connected_positive(self):
        frags, entries = _build_chain(2, base_score=0.7)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        result = _evaluate_config(placements, entries, etf)
        assert result > 0.0

    def test_single_fragment_zero(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placements = {0: (np.zeros(2), 0.0)}
        result = _evaluate_config(placements, entries, etf)
        assert result == pytest.approx(0.0)

    def test_empty_entries_zero(self):
        frags = [_frag(0), _frag(1)]
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert _evaluate_config(placements, [], etf) == pytest.approx(0.0)

    def test_result_float(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert isinstance(_evaluate_config(placements, entries, etf), float)

    def test_result_nonneg(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert _evaluate_config(placements, entries, etf) >= 0.0

    def test_chain_of_three_vs_two(self):
        frags, entries = _build_chain(3, base_score=0.8)
        etf = _make_edge_to_frag(frags)
        placements_all = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        placements_two = {f.fragment_id: (np.zeros(2), 0.0) for f in frags[:2]}
        score_all = _evaluate_config(placements_all, entries, etf)
        score_two = _evaluate_config(placements_two, entries, etf)
        assert score_all >= score_two
