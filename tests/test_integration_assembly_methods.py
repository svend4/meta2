"""
Integration tests: compare all 8 assembly algorithms on the same synthetic data.

Each algorithm is run on 4 synthetic fragments with a constructed compat matrix.
Tests verify that each method:
  - returns an Assembly object
  - covers all 4 fragments (no orphans)
  - produces non-overlapping placement IDs
  - has a finite total_score

Additionally, the cross-method utilities (pick_best, run_all_methods,
summary_table) are tested against the same data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)
from puzzle_reconstruction.assembly.parallel import (
    ALL_METHODS,
    DEFAULT_METHODS,
    MethodResult,
    AssemblyRacer,
    pick_best,
    pick_best_k,
    run_all_methods,
    run_selected,
    summary_table,
)

# ─── Helper builders ──────────────────────────────────────────────────────────

N_FRAGS = 4


def _make_edge(edge_id: int, n_points: int = 16) -> EdgeSignature:
    """Create a minimal EdgeSignature with a realistic virtual_curve."""
    t = np.linspace(0, 2 * np.pi, n_points)
    curve = np.column_stack([np.cos(t) * 50, np.sin(t) * 10])
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=curve,
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int) -> Fragment:
    """
    Fragment with 2 EdgeSignature edges.

    Edge IDs use the convention edge_id = fid * 10 + i so that
    mcts._build_score_map (which uses edge_id // 10) reconstructs the
    correct fragment_id.
    """
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(2)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=ei,
        edge_j=ej,
        score=score,
        dtw_dist=max(0.01, 1.0 - score),
        css_sim=score,
        fd_diff=0.0,
        text_score=0.0,
    )


def _build_dataset(n: int = N_FRAGS, seed: int = 42):
    """
    Build n synthetic fragments and a compat list with pseudo-random scores.

    Returns (fragments, entries) sorted by score descending.
    """
    rng = np.random.RandomState(seed)
    fragments = [_make_fragment(fid) for fid in range(n)]

    entries = []
    for i, fi in enumerate(fragments):
        for j, fj in enumerate(fragments):
            if i >= j:
                continue
            # Give different pairs different scores so algorithms can rank them
            score = float(rng.uniform(0.3, 0.95))
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score))
            entries.append(_make_entry(fi.edges[1], fj.edges[1], score * 0.9))

    entries.sort(key=lambda e: e.score, reverse=True)
    return fragments, entries


# ─── Module-scoped fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def four_fragment_data():
    """Shared 4-fragment dataset used by all tests in this module."""
    fragments, entries = _build_dataset(n=N_FRAGS, seed=7)
    return {"fragments": fragments, "entries": entries}


# ─── Individual assembler runner ──────────────────────────────────────────────

def _run_method(method: str, fragments, entries) -> Assembly:
    """
    Run a single assembler and return the Assembly.

    'sa' is handled specially because parallel.py calls
    simulated_annealing(fragments, entries, ...) but the function
    signature is simulated_annealing(assembly, entries, ...).
    We therefore call greedy first and pass the result to SA.
    """
    if method == "greedy":
        from puzzle_reconstruction.assembly.greedy import greedy_assembly
        return greedy_assembly(fragments, entries)

    if method == "sa":
        from puzzle_reconstruction.assembly.greedy import greedy_assembly
        from puzzle_reconstruction.assembly.annealing import simulated_annealing
        init = greedy_assembly(fragments, entries)
        return simulated_annealing(init, entries, T_max=100, T_min=0.1,
                                   max_iter=300, seed=0)

    if method == "beam":
        from puzzle_reconstruction.assembly.beam_search import beam_search
        return beam_search(fragments, entries, beam_width=3)

    if method == "gamma":
        from puzzle_reconstruction.assembly.gamma_optimizer import gamma_optimizer
        return gamma_optimizer(fragments, entries, n_iter=100, seed=0)

    if method == "genetic":
        from puzzle_reconstruction.assembly.genetic import genetic_assembly
        return genetic_assembly(fragments, entries, population_size=10,
                                n_generations=20, seed=0)

    if method == "exhaustive":
        from puzzle_reconstruction.assembly.exhaustive import exhaustive_assembly
        return exhaustive_assembly(fragments, entries)

    if method == "ant_colony":
        from puzzle_reconstruction.assembly.ant_colony import ant_colony_assembly
        return ant_colony_assembly(fragments, entries, n_ants=5,
                                   n_iterations=10, seed=0)

    if method == "mcts":
        from puzzle_reconstruction.assembly.mcts import mcts_assembly
        return mcts_assembly(fragments, entries, n_simulations=20, seed=0)

    raise ValueError(f"Unknown method: {method!r}")


# ─── TestAssemblyMethodsConsistency ──────────────────────────────────────────

class TestAssemblyMethodsConsistency:
    """Each algorithm is expected to return a valid Assembly for 4 fragments."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_returns_assembly(self, method, four_fragment_data):
        """Each method must return an Assembly instance."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        assert isinstance(asm, Assembly), (
            f"Method {method!r} did not return Assembly, got {type(asm)}"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_covers_all_fragments(self, method, four_fragment_data):
        """Placements dict must contain an entry for every fragment_id."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        expected_ids = {f.fragment_id for f in frags}
        placed_ids = set(asm.placements.keys())
        assert placed_ids == expected_ids, (
            f"Method {method!r}: expected {expected_ids}, got {placed_ids}"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_no_duplicate_placement_keys(self, method, four_fragment_data):
        """Placement keys (fragment_ids) must be unique (dict guarantees this)."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        # A dict cannot have duplicate keys – verify count equals n_frags
        assert len(asm.placements) == len(frags), (
            f"Method {method!r}: placements count mismatch"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_placements_are_tuples(self, method, four_fragment_data):
        """Each placement value must be a 2-tuple (position, angle)."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        for fid, placement in asm.placements.items():
            assert isinstance(placement, (tuple, list)) and len(placement) == 2, (
                f"Method {method!r}, fragment {fid}: "
                f"placement {placement!r} is not a (pos, angle) pair"
            )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_positions_are_2d_vectors(self, method, four_fragment_data):
        """Position component of each placement must be a length-2 array."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        for fid, (pos, angle) in asm.placements.items():
            pos_arr = np.asarray(pos)
            assert pos_arr.shape == (2,), (
                f"Method {method!r}, fragment {fid}: "
                f"position shape {pos_arr.shape} != (2,)"
            )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_angles_are_finite(self, method, four_fragment_data):
        """Rotation angle for each placement must be a finite float."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        for fid, (pos, angle) in asm.placements.items():
            assert np.isfinite(float(angle)), (
                f"Method {method!r}, fragment {fid}: angle {angle!r} is not finite"
            )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_positions_are_finite(self, method, four_fragment_data):
        """Position coordinates for each placement must be finite."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        for fid, (pos, angle) in asm.placements.items():
            pos_arr = np.asarray(pos)
            assert np.all(np.isfinite(pos_arr)), (
                f"Method {method!r}, fragment {fid}: position {pos_arr} has non-finite values"
            )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_total_score_is_finite(self, method, four_fragment_data):
        """total_score must be a finite float."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        assert np.isfinite(asm.total_score), (
            f"Method {method!r}: total_score {asm.total_score!r} is not finite"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_fragments_field_is_set(self, method, four_fragment_data):
        """assembly.fragments must reference the input fragment list."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        assert asm.fragments is not None, (
            f"Method {method!r}: Assembly.fragments is None"
        )
        assert len(asm.fragments) == len(frags), (
            f"Method {method!r}: Assembly.fragments length mismatch"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_no_overlap_on_fragment_ids(self, method, four_fragment_data):
        """Every input fragment_id must appear exactly once in placements."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method(method, frags, entries)
        all_fids = [f.fragment_id for f in frags]
        for fid in all_fids:
            assert fid in asm.placements, (
                f"Method {method!r}: fragment_id {fid} missing from placements"
            )


# ─── TestAssemblyEdgeCases ────────────────────────────────────────────────────

class TestAssemblyEdgeCases:
    """Tests that check edge-case handling common to all methods."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_empty_entries(self, method):
        """Methods must not crash when entries list is empty."""
        frags = [_make_fragment(fid) for fid in range(3)]
        entries = []
        asm = _run_method(method, frags, entries)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(frags)

    @pytest.mark.parametrize("method", ["greedy", "beam", "gamma", "genetic",
                                         "ant_colony", "mcts"])
    def test_method_single_fragment(self, method):
        """Methods must handle a single fragment without error."""
        frags = [_make_fragment(0)]
        entries = []
        asm = _run_method(method, frags, entries)
        assert isinstance(asm, Assembly)
        assert 0 in asm.placements

    def test_exhaustive_works_for_4_fragments(self, four_fragment_data):
        """exhaustive_assembly must succeed for exactly 4 fragments."""
        from puzzle_reconstruction.assembly.exhaustive import exhaustive_assembly
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = exhaustive_assembly(frags, entries)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(frags)

    def test_sa_handles_single_fragment(self):
        """SA must return early (no-op) for a single fragment."""
        from puzzle_reconstruction.assembly.greedy import greedy_assembly
        from puzzle_reconstruction.assembly.annealing import simulated_annealing
        frags = [_make_fragment(0)]
        entries = []
        init = greedy_assembly(frags, entries)
        asm = simulated_annealing(init, entries, max_iter=50, seed=0)
        assert isinstance(asm, Assembly)


# ─── TestParallelUtilities ────────────────────────────────────────────────────

class TestParallelUtilities:
    """Tests for run_all_methods, pick_best, and summary_table."""

    def test_pick_best_selects_highest_score(self, four_fragment_data):
        """pick_best must return the Assembly with the highest total_score."""
        frags = four_fragment_data["fragments"]

        def _asm(score):
            placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
            return Assembly(fragments=frags, placements=placements, total_score=score)

        results = [
            MethodResult(name="greedy", assembly=_asm(0.30)),
            MethodResult(name="beam",   assembly=_asm(0.85)),
            MethodResult(name="sa",     assembly=_asm(0.60)),
        ]
        best = pick_best(results)
        assert best is not None
        assert abs(best.total_score - 0.85) < 1e-9

    def test_pick_best_ignores_failures(self, four_fragment_data):
        """pick_best must ignore MethodResult entries without a valid Assembly."""
        frags = four_fragment_data["fragments"]
        placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
        good_asm = Assembly(fragments=frags, placements=placements, total_score=0.5)
        results = [
            MethodResult(name="greedy", assembly=good_asm),
            MethodResult(name="sa",     error="crashed"),
            MethodResult(name="beam",   timed_out=True),
        ]
        best = pick_best(results)
        assert best is good_asm

    def test_pick_best_returns_none_when_all_failed(self):
        """pick_best must return None when every result failed."""
        results = [
            MethodResult(name="greedy", error="fail"),
            MethodResult(name="sa",     timed_out=True),
        ]
        assert pick_best(results) is None

    def test_run_all_methods_returns_list(self, four_fragment_data):
        """run_all_methods must return a list."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0,
                                  timeout=30.0)
        assert isinstance(results, list)

    def test_run_all_methods_returns_method_results(self, four_fragment_data):
        """Each element in the returned list must be a MethodResult."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0,
                                  timeout=30.0)
        assert all(isinstance(r, MethodResult) for r in results)

    def test_run_all_methods_greedy_succeeds(self, four_fragment_data):
        """Greedy method must succeed when run through run_all_methods."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0,
                                  timeout=30.0)
        assert len(results) == 1
        r = results[0]
        assert r.name == "greedy"
        assert r.success

    def test_run_all_methods_beam_succeeds(self, four_fragment_data):
        """Beam method must succeed when run through run_all_methods."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["beam"], seed=0,
                                  timeout=30.0)
        assert results[0].success

    def test_run_all_methods_genetic_succeeds(self, four_fragment_data):
        """Genetic method must succeed when run through run_all_methods."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["genetic"], seed=0,
                                  timeout=30.0, n_iterations=50)
        assert results[0].success

    def test_run_all_methods_elapsed_nonneg(self, four_fragment_data):
        """Elapsed time for each result must be non-negative."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0,
                                  timeout=30.0)
        assert all(r.elapsed >= 0.0 for r in results)

    def test_summary_table_covers_all_methods(self, four_fragment_data):
        """summary_table output must mention every method name in the results."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]

        method_names = ["greedy", "beam", "gamma"]
        results = []
        for m in method_names:
            asm = _run_method(m, frags, entries)
            results.append(MethodResult(name=m, assembly=asm, elapsed=0.01))

        table = summary_table(results)
        assert isinstance(table, str)
        for m in method_names:
            assert m in table, f"Method {m!r} not found in summary_table output"

    def test_summary_table_is_markdown(self, four_fragment_data):
        """summary_table must return a Markdown table (contains '|')."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method("greedy", frags, entries)
        results = [MethodResult(name="greedy", assembly=asm)]
        table = summary_table(results)
        assert "|" in table

    def test_summary_table_shows_ok_status(self, four_fragment_data):
        """Successful results must have 'OK' in the summary table."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        asm = _run_method("greedy", frags, entries)
        results = [MethodResult(name="greedy", assembly=asm)]
        table = summary_table(results)
        assert "OK" in table

    def test_summary_table_shows_timeout_status(self):
        """Timed-out results must have 'TIMEOUT' in the summary table."""
        results = [MethodResult(name="sa", timed_out=True)]
        table = summary_table(results)
        assert "TIMEOUT" in table

    def test_summary_table_shows_error_status(self):
        """Failed results must have 'ERROR' in the summary table."""
        results = [MethodResult(name="gamma", error="division by zero")]
        table = summary_table(results)
        assert "ERROR" in table

    def test_run_selected_raises_on_unknown_method(self, four_fragment_data):
        """run_selected must raise ValueError for unknown method names."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        with pytest.raises(ValueError):
            run_selected(frags, entries, methods=["not_a_real_method"])

    def test_pick_best_k_returns_sorted(self, four_fragment_data):
        """pick_best_k must return assemblies sorted by score descending."""
        frags = four_fragment_data["fragments"]

        def _asm(score):
            placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
            return Assembly(fragments=frags, placements=placements, total_score=score)

        results = [
            MethodResult(name="a", assembly=_asm(0.1)),
            MethodResult(name="b", assembly=_asm(0.9)),
            MethodResult(name="c", assembly=_asm(0.5)),
        ]
        top = pick_best_k(results, k=3)
        scores = [a.total_score for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_pick_best_k_respects_k(self, four_fragment_data):
        """pick_best_k(k=2) must return at most 2 assemblies."""
        frags = four_fragment_data["fragments"]

        def _asm(score):
            placements = {f.fragment_id: (np.zeros(2), 0.0) for f in frags}
            return Assembly(fragments=frags, placements=placements, total_score=score)

        results = [MethodResult(name=str(i), assembly=_asm(float(i) / 10))
                   for i in range(5)]
        top2 = pick_best_k(results, k=2)
        assert len(top2) == 2


# ─── TestAllMethodsConstants ──────────────────────────────────────────────────

class TestAllMethodsConstants:
    """Tests that verify the ALL_METHODS constant has the expected content."""

    def test_all_methods_is_list(self):
        assert isinstance(ALL_METHODS, list)

    def test_all_methods_length(self):
        assert len(ALL_METHODS) == 8

    def test_all_methods_contains_greedy(self):
        assert "greedy" in ALL_METHODS

    def test_all_methods_contains_sa(self):
        assert "sa" in ALL_METHODS

    def test_all_methods_contains_beam(self):
        assert "beam" in ALL_METHODS

    def test_all_methods_contains_gamma(self):
        assert "gamma" in ALL_METHODS

    def test_all_methods_contains_genetic(self):
        assert "genetic" in ALL_METHODS

    def test_all_methods_contains_exhaustive(self):
        assert "exhaustive" in ALL_METHODS

    def test_all_methods_contains_ant_colony(self):
        assert "ant_colony" in ALL_METHODS

    def test_all_methods_contains_mcts(self):
        assert "mcts" in ALL_METHODS

    def test_default_methods_subset_of_all(self):
        assert all(m in ALL_METHODS for m in DEFAULT_METHODS)


# ─── TestMethodResultModel ────────────────────────────────────────────────────

class TestMethodResultModel:
    """Tests for the MethodResult dataclass."""

    def test_success_true_when_assembly_present(self):
        frags = [_make_fragment(0)]
        placements = {0: (np.zeros(2), 0.0)}
        asm = Assembly(fragments=frags, placements=placements, total_score=0.5)
        mr = MethodResult(name="greedy", assembly=asm)
        assert mr.success is True

    def test_success_false_when_no_assembly(self):
        mr = MethodResult(name="sa", assembly=None)
        assert mr.success is False

    def test_success_false_on_timeout(self):
        frags = [_make_fragment(0)]
        placements = {0: (np.zeros(2), 0.0)}
        asm = Assembly(fragments=frags, placements=placements, total_score=0.5)
        mr = MethodResult(name="beam", assembly=asm, timed_out=True)
        assert mr.success is False

    def test_score_zero_without_assembly(self):
        mr = MethodResult(name="gamma")
        assert mr.score == 0.0

    def test_score_matches_assembly(self):
        frags = [_make_fragment(0)]
        placements = {0: (np.zeros(2), 0.0)}
        asm = Assembly(fragments=frags, placements=placements, total_score=0.77)
        mr = MethodResult(name="mcts", assembly=asm)
        assert abs(mr.score - 0.77) < 1e-9

    def test_method_property_alias(self):
        mr = MethodResult(name="ant_colony")
        assert mr.method == "ant_colony"

    def test_repr_contains_name(self):
        mr = MethodResult(name="genetic")
        assert "genetic" in repr(mr)

    def test_error_stored_correctly(self):
        mr = MethodResult(name="exhaustive", error="out of memory")
        assert mr.error == "out of memory"
        assert mr.success is False

    def test_timed_out_flag(self):
        mr = MethodResult(name="mcts", timed_out=True)
        assert mr.timed_out is True
        assert mr.success is False


# ─── TestAssemblyRacer ────────────────────────────────────────────────────────

class TestAssemblyRacer:
    """Tests for the AssemblyRacer class."""

    def test_racer_construction(self, four_fragment_data):
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        racer = AssemblyRacer(frags, entries, seed=0)
        assert racer.fragments is frags
        assert racer.entries is entries

    def test_race_returns_list(self, four_fragment_data):
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["greedy"], timeout=30.0)
        assert isinstance(results, list)

    def test_race_greedy_succeeds(self, four_fragment_data):
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["greedy"], timeout=30.0)
        assert len(results) >= 1
        successful = [r for r in results if r.success]
        assert len(successful) >= 1

    def test_race_first_only(self, four_fragment_data):
        """first_only=True should return as soon as one method completes."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["greedy", "beam"], timeout=30.0,
                              first_only=True)
        # At least one result (the first completed one)
        assert len(results) >= 1

    def test_race_beam_method(self, four_fragment_data):
        """AssemblyRacer can run beam method."""
        frags = four_fragment_data["fragments"]
        entries = four_fragment_data["entries"]
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["beam"], timeout=30.0)
        assert isinstance(results, list)
        assert len(results) >= 1
