"""Расширенные тесты для puzzle_reconstruction/assembly/exhaustive.py."""
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


def _build_chain(n: int, base_score: float = 0.8):
    """Build n fragments with compatibility entries linking them in a chain."""
    frags = [_frag(i) for i in range(n)]
    entries = []
    for i in range(n - 1):
        e_i = frags[i].edges[0]
        e_j = frags[i + 1].edges[0]
        entries.append(_entry(e_i, e_j, score=base_score))
    return frags, entries


def _make_edge_to_frag(frags):
    return {e.edge_id: f.fragment_id for f in frags for e in f.edges}


# ─── TestExhaustiveAssembly ───────────────────────────────────────────────────

class TestExhaustiveAssembly:
    def test_returns_assembly(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_empty_fragments_raises(self):
        with pytest.raises(ValueError):
            exhaustive_assembly([], [])

    def test_single_fragment_in_placements(self):
        frags = [_frag(0)]
        result = exhaustive_assembly(frags, [])
        assert 0 in result.placements

    def test_all_fragments_in_placements(self):
        frags, entries = _build_chain(3)
        result = exhaustive_assembly(frags, entries)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_two_fragments(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert len(result.placements) == 2

    def test_score_is_float(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result.total_score, float)

    def test_score_nonneg(self):
        frags, entries = _build_chain(3)
        result = exhaustive_assembly(frags, entries)
        assert result.total_score >= 0.0

    def test_no_entries_score_zero(self):
        frags = [_frag(0), _frag(1)]
        result = exhaustive_assembly(frags, [])
        assert result.total_score == pytest.approx(0.0)

    def test_placement_is_tuple(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        for placement in result.placements.values():
            assert isinstance(placement, tuple)
            assert len(placement) == 2

    def test_placement_pos_length_2(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        for pos, angle in result.placements.values():
            assert len(pos) == 2

    def test_placement_angle_is_float(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        for pos, angle in result.placements.values():
            assert isinstance(angle, float)

    def test_compat_matrix_is_ndarray(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_fragments_stored(self):
        frags, entries = _build_chain(3)
        result = exhaustive_assembly(frags, entries)
        assert result.fragments is frags

    def test_deterministic_with_same_seed(self):
        frags, entries = _build_chain(3)
        r1 = exhaustive_assembly(frags, entries, seed=0)
        r2 = exhaustive_assembly(frags, entries, seed=0)
        assert r1.total_score == r2.total_score

    def test_n_greater_than_max_n_uses_beam(self):
        """N > max_n should fallback to beam_search with a RuntimeWarning."""
        frags, entries = _build_chain(4)
        with pytest.warns(RuntimeWarning):
            result = exhaustive_assembly(frags, entries, max_n=3)
        assert isinstance(result, Assembly)

    def test_n_equals_warn_n_triggers_warning(self):
        """N >= WARN_N (8) but <= max_n=9 should trigger RuntimeWarning."""
        frags, entries = _build_chain(8)
        with pytest.warns(RuntimeWarning):
            result = exhaustive_assembly(frags, entries, max_n=9)
        assert isinstance(result, Assembly)

    def test_small_n_no_fallback_warning(self):
        """N = 3 < WARN_N should complete without RuntimeWarning."""
        frags, entries = _build_chain(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = exhaustive_assembly(frags, entries, max_n=9)
        assert isinstance(result, Assembly)

    def test_with_high_score_entries(self):
        frags, entries = _build_chain(3, base_score=0.9)
        result = exhaustive_assembly(frags, entries)
        assert result.total_score > 0.0

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(2)
        result = exhaustive_assembly(frags, entries, allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_four_fragments_within_max_n(self):
        frags, entries = _build_chain(4)
        result = exhaustive_assembly(frags, entries, max_n=9)
        assert len(result.placements) == 4

    def test_placements_count_equals_fragment_count(self):
        n = 3
        frags, entries = _build_chain(n)
        result = exhaustive_assembly(frags, entries)
        assert len(result.placements) == n


# ─── TestScoreDelta ───────────────────────────────────────────────────────────

class TestScoreDelta:
    def test_returns_float(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placed = {0: (np.zeros(2), 0.0)}
        result = _score_delta(1, [0], placed, entries, etf)
        assert isinstance(result, float)

    def test_empty_placed_returns_zero(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        result = _score_delta(0, [], {}, entries, etf)
        assert result == pytest.approx(0.0)

    def test_connected_frags_positive_delta(self):
        frags, entries = _build_chain(2, base_score=0.7)
        etf = _make_edge_to_frag(frags)
        placed = {0: (np.zeros(2), 0.0)}
        result = _score_delta(1, [0], placed, entries, etf)
        assert result > 0.0

    def test_unconnected_frags_zero_delta(self):
        f0 = _frag(0)
        f1 = _frag(1)
        f2 = _frag(2)
        # entries only link f0-f1, not f0-f2 or f1-f2
        entries = [_entry(f0.edges[0], f1.edges[0], score=0.9)]
        etf = _make_edge_to_frag([f0, f1, f2])
        placed = {0: (np.zeros(2), 0.0)}
        result = _score_delta(2, [0], placed, entries, etf)
        assert result == pytest.approx(0.0)

    def test_nonneg(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        placed_pos = {0: (np.zeros(2), 0.0), 1: (np.array([120.0, 0.0]), 0.0)}
        result = _score_delta(2, [0, 1], placed_pos, entries, etf)
        assert result >= 0.0

    def test_higher_score_entry_higher_delta(self):
        f0 = _frag(0)
        f1 = _frag(1)
        etf = {e.edge_id: f.fragment_id for f in [f0, f1] for e in f.edges}
        entry_low = _entry(f0.edges[0], f1.edges[0], score=0.1)
        entry_high = _entry(f0.edges[0], f1.edges[0], score=0.9)
        placed = {0: (np.zeros(2), 0.0)}
        d_low = _score_delta(1, [0], placed, [entry_low], etf)
        d_high = _score_delta(1, [0], placed, [entry_high], etf)
        assert d_high > d_low


# ─── TestEvaluateConfig ───────────────────────────────────────────────────────

class TestEvaluateConfig:
    def test_returns_float(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert isinstance(_evaluate_config(placements, entries, etf), float)

    def test_empty_placements_zero(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        assert _evaluate_config({}, entries, etf) == pytest.approx(0.0)

    def test_single_fragment_zero(self):
        frags, entries = _build_chain(2)
        etf = _make_edge_to_frag(frags)
        placements = {0: (np.zeros(2), 0.0)}
        assert _evaluate_config(placements, entries, etf) == pytest.approx(0.0)

    def test_connected_pair_has_score(self):
        frags, entries = _build_chain(2, base_score=0.8)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        result = _evaluate_config(placements, entries, etf)
        assert result > 0.0

    def test_nonneg(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert _evaluate_config(placements, entries, etf) >= 0.0

    def test_no_entries_zero(self):
        frags = [_frag(0), _frag(1)]
        etf = _make_edge_to_frag(frags)
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        assert _evaluate_config(placements, [], etf) == pytest.approx(0.0)

    def test_unplaced_frag_skipped(self):
        f0 = _frag(0)
        f1 = _frag(1)
        entries = [_entry(f0.edges[0], f1.edges[0], score=0.8)]
        etf = _make_edge_to_frag([f0, f1])
        # Only f0 placed
        placements = {0: (np.zeros(2), 0.0)}
        result = _evaluate_config(placements, entries, etf)
        assert result == pytest.approx(0.0)
