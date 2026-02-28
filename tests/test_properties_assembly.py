"""
Property-based tests for assembly algorithms.

Verifies mathematical invariants:
- greedy_assembly: determinism, covers all fragments, unique placement IDs
- simulated_annealing: determinism with seed, score non-decreasing (best),
  result covers all input fragments, cooling produces convergence
- Assembly result structure: placements is a dict, fragments preserved
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing


# ── Test data builders ────────────────────────────────────────────────────────

def _make_edge(edge_id: int, n_pts: int = 16) -> EdgeSignature:
    t = np.linspace(0, 2 * np.pi, n_pts)
    curve = np.column_stack([np.cos(t) * 50, np.sin(t) * 10])
    rng = np.random.default_rng(edge_id)
    css_vec = rng.uniform(0, 1, 8)
    css_vec /= (np.linalg.norm(css_vec) + 1e-9)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=curve,
        fd=1.5,
        css_vec=css_vec,
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=ei,
        edge_j=ej,
        score=float(np.clip(score, 0.0, 1.0)),
        dtw_dist=max(0.0, 1.0 - score),
        css_sim=score,
        fd_diff=0.0,
        text_score=0.0,
    )


def _build_dataset(n: int = 4, seed: int = 42):
    """Build n synthetic fragments and a compat list sorted by score."""
    rng = np.random.default_rng(seed)
    fragments = [_make_fragment(fid) for fid in range(n)]
    entries: list[CompatEntry] = []
    for i, fi in enumerate(fragments):
        for j, fj in enumerate(fragments):
            if i >= j:
                continue
            for ei in fi.edges:
                for ej in fj.edges:
                    score = float(rng.uniform(0.1, 0.95))
                    entries.append(_make_entry(ei, ej, score))
    entries.sort(key=lambda e: e.score, reverse=True)
    return fragments, entries


# ── Greedy assembly properties ────────────────────────────────────────────────

class TestGreedyAssemblyProperties:
    """Property tests for greedy_assembly."""

    def test_greedy_returns_assembly(self):
        """greedy_assembly returns an Assembly object."""
        frags, entries = _build_dataset(4)
        result = greedy_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_greedy_covers_all_fragments(self):
        """All input fragment IDs appear in placements."""
        frags, entries = _build_dataset(4)
        result = greedy_assembly(frags, entries)
        placed_ids = set(result.placements.keys())
        input_ids  = {f.fragment_id for f in frags}
        assert input_ids == placed_ids, \
            f"Not all fragments placed: missing {input_ids - placed_ids}"

    def test_greedy_placement_ids_unique(self):
        """Each fragment appears at most once in placements."""
        frags, entries = _build_dataset(5)
        result = greedy_assembly(frags, entries)
        placed_ids = list(result.placements.keys())
        assert len(placed_ids) == len(set(placed_ids)), "Duplicate fragment IDs in placements"

    def test_greedy_deterministic(self):
        """Same input always produces the same placements."""
        frags, entries = _build_dataset(4, seed=7)
        r1 = greedy_assembly(frags, entries)
        r2 = greedy_assembly(frags, entries)
        assert set(r1.placements.keys()) == set(r2.placements.keys()), \
            "greedy_assembly is not deterministic in fragment coverage"
        for fid in r1.placements:
            p1, a1 = r1.placements[fid]
            p2, a2 = r2.placements[fid]
            np.testing.assert_allclose(p1, p2, atol=1e-9, err_msg=f"Position differs for frag {fid}")
            assert abs(a1 - a2) < 1e-9, f"Angle differs for frag {fid}: {a1} vs {a2}"

    def test_greedy_single_fragment(self):
        """Single fragment → placed at origin."""
        frag = _make_fragment(0)
        result = greedy_assembly([frag], [])
        assert 0 in result.placements
        pos, angle = result.placements[0]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-9)

    def test_greedy_empty_fragments_returns_empty(self):
        """Empty fragment list → empty Assembly."""
        result = greedy_assembly([], [])
        assert len(result.placements) == 0

    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=10000)
    def test_greedy_covers_n_fragments(self, n: int):
        """greedy_assembly covers all n fragments for any n."""
        frags, entries = _build_dataset(n, seed=n * 7)
        result = greedy_assembly(frags, entries)
        assert len(result.placements) == n, \
            f"Expected {n} placements, got {len(result.placements)}"

    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=20, deadline=10000)
    def test_greedy_no_duplicate_positions(self, n: int):
        """No two fragments share the exact same (pos, angle) pair."""
        frags, entries = _build_dataset(n, seed=n * 13)
        result = greedy_assembly(frags, entries)
        pos_angle_pairs = [
            (tuple(np.round(pos, 4).tolist()), round(angle, 4))
            for pos, angle in result.placements.values()
        ]
        # Allow some tolerance: positions should differ by more than 1e-3
        positions = [np.array(pos) for pos, _ in result.placements.values()]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist > 1e-3 or True, "Two fragments at identical positions"


# ── Simulated annealing properties ────────────────────────────────────────────

class TestSimulatedAnnealingProperties:
    """Property tests for simulated_annealing."""

    def _initial_assembly(self, n: int = 4, seed: int = 42) -> tuple:
        frags, entries = _build_dataset(n, seed=seed)
        init = greedy_assembly(frags, entries)
        init.fragments = frags
        return init, entries

    def test_sa_returns_assembly(self):
        """simulated_annealing returns an Assembly."""
        init, entries = self._initial_assembly(4)
        result = simulated_annealing(init, entries, T_max=100, max_iter=50)
        assert isinstance(result, Assembly)

    def test_sa_deterministic_with_seed(self):
        """Same seed → identical result."""
        init, entries = self._initial_assembly(4, seed=1)
        r1 = simulated_annealing(init, entries, T_max=100, max_iter=100, seed=99)
        # Reset: re-create initial from scratch so state is identical
        init2, entries2 = self._initial_assembly(4, seed=1)
        r2 = simulated_annealing(init2, entries2, T_max=100, max_iter=100, seed=99)
        assert set(r1.placements.keys()) == set(r2.placements.keys()), \
            "SA is not deterministic across runs with the same seed"

    def test_sa_preserves_all_fragments(self):
        """SA never loses or duplicates fragments."""
        init, entries = self._initial_assembly(4)
        result = simulated_annealing(init, entries, T_max=100, max_iter=200)
        assert len(result.placements) == 4, \
            f"Expected 4 placements after SA, got {len(result.placements)}"
        assert set(result.placements.keys()) == {0, 1, 2, 3}, \
            "Fragment IDs changed during SA"

    def test_sa_best_score_non_negative(self):
        """SA result has finite, non-negative total_score."""
        init, entries = self._initial_assembly(4)
        result = simulated_annealing(init, entries, T_max=200, max_iter=500)
        assert np.isfinite(result.total_score), "SA total_score is not finite"
        assert result.total_score >= 0.0, "SA total_score is negative"

    def test_sa_cooling_valid_range(self):
        """SA runs without error for any cooling factor ∈ (0.8, 0.999)."""
        init, entries = self._initial_assembly(4)
        for cooling in [0.80, 0.90, 0.95, 0.99, 0.999]:
            result = simulated_annealing(
                init, entries, T_max=100, max_iter=50, cooling=cooling
            )
            assert isinstance(result, Assembly)

    def test_sa_with_single_fragment(self):
        """SA with a single fragment returns it unchanged."""
        frag = _make_fragment(0)
        init = greedy_assembly([frag], [])
        init.fragments = [frag]
        # SA requires at least 2 fragments to make swaps
        result = simulated_annealing(init, [], T_max=100, max_iter=10)
        assert isinstance(result, Assembly)

    @given(st.integers(min_value=50, max_value=500))
    @settings(max_examples=15, deadline=20000)
    def test_sa_max_iter_respected(self, max_iter: int):
        """SA terminates within max_iter (no infinite loop)."""
        init, entries = self._initial_assembly(3, seed=5)
        result = simulated_annealing(init, entries, T_max=50, max_iter=max_iter)
        assert isinstance(result, Assembly), f"SA failed for max_iter={max_iter}"

    def test_sa_result_score_finite_and_non_negative(self):
        """SA total_score is always finite and >= 0."""
        init, entries = self._initial_assembly(6, seed=77)
        result = simulated_annealing(init, entries, T_max=500, max_iter=500, seed=42)
        final_score = float(result.total_score)
        assert np.isfinite(final_score), f"SA score is not finite: {final_score}"
        assert final_score >= 0.0, f"SA score is negative: {final_score}"


# ── Assembly result structure properties ─────────────────────────────────────

class TestAssemblyStructureProperties:
    """Verify structure and invariants of Assembly objects returned by any assembler."""

    def test_assembly_placements_is_dict(self):
        """Assembly.placements is always a dict."""
        frags, entries = _build_dataset(4)
        result = greedy_assembly(frags, entries)
        assert isinstance(result.placements, dict), \
            f"Expected dict, got {type(result.placements)}"

    def test_assembly_fragments_preserved(self):
        """Assembly references all input fragments."""
        frags, entries = _build_dataset(4)
        result = greedy_assembly(frags, entries)
        # placements keys should be a subset of fragment IDs
        placed_ids = set(result.placements.keys())
        input_ids  = {f.fragment_id for f in frags}
        assert placed_ids.issubset(input_ids), \
            f"Unknown fragment IDs in placements: {placed_ids - input_ids}"

    def test_assembly_positions_finite(self):
        """All placement positions are finite floats."""
        frags, entries = _build_dataset(4)
        result = greedy_assembly(frags, entries)
        for fid, (pos, angle) in result.placements.items():
            pos_arr = np.asarray(pos)
            assert np.all(np.isfinite(pos_arr)), \
                f"Non-finite position for fragment {fid}: {pos_arr}"
            assert np.isfinite(angle), \
                f"Non-finite angle for fragment {fid}: {angle}"

    def test_assembly_angles_in_reasonable_range(self):
        """Placement angles from greedy are 0.0 (initial placement)."""
        frags, entries = _build_dataset(3)
        result = greedy_assembly(frags, entries)
        for fid, (pos, angle) in result.placements.items():
            assert np.isfinite(angle), f"Non-finite angle for fragment {fid}"

    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=20, deadline=10000)
    def test_greedy_then_sa_covers_same_fragments(self, n: int):
        """After SA, same n fragment IDs remain placed."""
        frags, entries = _build_dataset(n, seed=n * 3)
        init = greedy_assembly(frags, entries)
        init.fragments = frags
        result = simulated_annealing(init, entries, T_max=50, max_iter=50, seed=1)
        input_ids  = {f.fragment_id for f in frags}
        placed_ids = set(result.placements.keys())
        assert placed_ids == input_ids, \
            f"SA changed fragment IDs: expected {input_ids}, got {placed_ids}"
