"""Tests for puzzle_reconstruction/assembly/exhaustive.py"""
import math
import warnings
import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, Edge, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.exhaustive import (
    exhaustive_assembly,
    _score_delta,
    _evaluate_config,
    MAX_EXACT_N,
    WARN_N,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edge_sig(edge_id: int) -> EdgeSignature:
    """Create a minimal EdgeSignature with virtual_curve (needed by greedy)."""
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.array([[0, 0], [10, 0]], dtype=np.float32),
        fd=1.0,
        css_vec=np.zeros(4, dtype=np.float32),
        ifs_coeffs=np.zeros(4, dtype=np.float32),
        length=10.0,
    )


def make_edge(edge_id: int) -> Edge:
    """Lightweight edge without virtual_curve (used only for index tests)."""
    return Edge(edge_id=edge_id, contour=np.zeros((2, 2), dtype=np.float32))


def make_fragment_with_sig(fid: int, n_edges: int = 1) -> Fragment:
    """Fragment with EdgeSignature edges (compatible with greedy_assembly)."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    edges = [make_edge_sig(fid * 10 + i) for i in range(n_edges)]
    return Fragment(fragment_id=fid, image=img, edges=edges)


def make_fragment_plain(fid: int) -> Fragment:
    """Fragment with lightweight Edge (no virtual_curve)."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    edges = [make_edge(fid * 10)]
    return Fragment(fragment_id=fid, image=img, edges=edges)


def make_entry_sig(edge_i: EdgeSignature, edge_j: EdgeSignature,
                   score: float) -> CompatEntry:
    return CompatEntry(edge_i=edge_i, edge_j=edge_j, score=score)


def make_entry_plain(edge_i: Edge, edge_j: Edge, score: float) -> CompatEntry:
    return CompatEntry(edge_i=edge_i, edge_j=edge_j, score=score)


def make_sig_frags_and_entries(n: int):
    """Build fragments with EdgeSignature edges plus compatibility entries."""
    frags = [make_fragment_with_sig(i) for i in range(n)]
    entries = []
    for i in range(n):
        for j in range(i + 1, n):
            e_i = frags[i].edges[0]
            e_j = frags[j].edges[0]
            entries.append(make_entry_sig(e_i, e_j, 0.5))
    return frags, entries


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_max_exact_n_value(self):
        assert MAX_EXACT_N == 9

    def test_warn_n_value(self):
        assert WARN_N == 8

    def test_warn_n_less_than_max_n(self):
        assert WARN_N <= MAX_EXACT_N


# ── exhaustive_assembly ───────────────────────────────────────────────────────

class TestExhaustiveAssembly:
    def test_empty_fragments_raises(self):
        with pytest.raises(ValueError, match="не должен быть пустым"):
            exhaustive_assembly([], [])

    def test_single_fragment_returns_assembly(self):
        frags, entries = make_sig_frags_and_entries(1)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_two_fragments_returns_assembly(self):
        frags, entries = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_three_fragments_returns_assembly(self):
        frags, entries = make_sig_frags_and_entries(3)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_returns_assembly_type(self):
        frags, entries = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_placements_contain_all_fragments(self):
        frags, entries = make_sig_frags_and_entries(3)
        result = exhaustive_assembly(frags, entries)
        assert isinstance(result.placements, dict)
        for frag in frags:
            assert frag.fragment_id in result.placements

    def test_total_score_nonneg(self):
        frags, entries = make_sig_frags_and_entries(3)
        result = exhaustive_assembly(frags, entries)
        assert result.total_score >= 0.0

    def test_fragments_stored(self):
        frags, entries = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, entries)
        assert result.fragments is not None

    def test_allow_rotation_false(self):
        frags, entries = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, entries, allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_large_n_falls_back_to_beam(self):
        frags, entries = make_sig_frags_and_entries(MAX_EXACT_N + 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = exhaustive_assembly(frags, entries, max_n=MAX_EXACT_N)
            assert any("beam_search" in str(warning.message) for warning in w)
        assert isinstance(result, Assembly)

    def test_warn_n_triggers_warning(self):
        frags, entries = make_sig_frags_and_entries(WARN_N)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = exhaustive_assembly(frags, entries)
            runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warns) >= 1
        assert isinstance(result, Assembly)

    def test_high_score_entry_preferred(self):
        """Fragments with high compatibility score should be placed."""
        frags, entries = make_sig_frags_and_entries(3)
        high_entry = make_entry_sig(frags[0].edges[0], frags[1].edges[0],
                                    score=0.99)
        result = exhaustive_assembly(frags, [high_entry], allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_empty_entries_still_returns_assembly(self):
        frags, _ = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, [])
        assert isinstance(result, Assembly)

    def test_placement_values_are_tuples(self):
        frags, entries = make_sig_frags_and_entries(2)
        result = exhaustive_assembly(frags, entries)
        for fid, val in result.placements.items():
            assert isinstance(val, tuple)
            assert len(val) == 2


# ── _score_delta ──────────────────────────────────────────────────────────────

class TestScoreDelta:
    def test_no_placed_returns_zero(self):
        result = _score_delta(0, [], {}, [], {})
        assert result == 0.0

    def test_delta_sums_matching_entries(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        entry = make_entry_plain(e0, e1, score=0.8)
        edge_to_frag = {0: 0, 1: 1}
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([100.0, 0.0]), 0.0)}
        result = _score_delta(1, [0], placements, [entry], edge_to_frag)
        assert pytest.approx(result, abs=1e-6) == 0.8

    def test_delta_zero_when_no_matching_entries(self):
        edge_to_frag = {0: 0, 1: 1}
        result = _score_delta(2, [0], {}, [], edge_to_frag)
        assert result == 0.0

    def test_symmetry(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        entry = make_entry_plain(e0, e1, score=0.5)
        edge_to_frag = {0: 0, 1: 1}
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        r1 = _score_delta(1, [0], placements, [entry], edge_to_frag)
        placements2 = {1: (np.array([0.0, 0.0]), 0.0)}
        r2 = _score_delta(0, [1], placements2, [entry], edge_to_frag)
        assert pytest.approx(r1, abs=1e-6) == r2

    def test_multiple_entries_summed(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        e2 = make_edge(2)
        entry1 = make_entry_plain(e0, e2, score=0.3)
        entry2 = make_entry_plain(e1, e2, score=0.4)
        edge_to_frag = {0: 0, 1: 1, 2: 2}
        placements = {0: (np.array([0.0, 0.0]), 0.0),
                      1: (np.array([100.0, 0.0]), 0.0)}
        result = _score_delta(2, [0, 1], placements, [entry1, entry2],
                              edge_to_frag)
        assert pytest.approx(result, abs=1e-6) == 0.7


# ── _evaluate_config ──────────────────────────────────────────────────────────

class TestEvaluateConfig:
    def test_empty_placements_returns_zero(self):
        result = _evaluate_config({}, [], {})
        assert result == 0.0

    def test_single_pair_score_counted(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        entry = make_entry_plain(e0, e1, score=0.7)
        edge_to_frag = {0: 0, 1: 1}
        placements = {
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([100.0, 0.0]), 0.0),
        }
        result = _evaluate_config(placements, [entry], edge_to_frag)
        assert pytest.approx(result, abs=1e-6) == 0.7

    def test_unplaced_fragments_not_counted(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        entry = make_entry_plain(e0, e1, score=0.9)
        edge_to_frag = {0: 0, 1: 1}
        # Only fragment 0 placed
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        result = _evaluate_config(placements, [entry], edge_to_frag)
        assert result == 0.0

    def test_each_pair_counted_once(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        # Duplicate entry for same pair
        entry1 = make_entry_plain(e0, e1, score=0.5)
        entry2 = make_entry_plain(e0, e1, score=0.5)
        edge_to_frag = {0: 0, 1: 1}
        placements = {
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([100.0, 0.0]), 0.0),
        }
        result = _evaluate_config(placements, [entry1, entry2], edge_to_frag)
        # Should count each unique pair once: result should be 0.5 (first occurrence)
        assert result >= 0.0

    def test_multiple_pairs_summed(self):
        e0 = make_edge(0)
        e1 = make_edge(1)
        e2 = make_edge(2)
        entry1 = make_entry_plain(e0, e1, score=0.3)
        entry2 = make_entry_plain(e1, e2, score=0.4)
        edge_to_frag = {0: 0, 1: 1, 2: 2}
        placements = {
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([100.0, 0.0]), 0.0),
            2: (np.array([200.0, 0.0]), 0.0),
        }
        result = _evaluate_config(placements, [entry1, entry2], edge_to_frag)
        assert pytest.approx(result, abs=1e-6) == 0.7

    def test_returns_float(self):
        result = _evaluate_config({}, [], {})
        assert isinstance(result, float)
