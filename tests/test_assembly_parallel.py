"""Tests for puzzle_reconstruction/assembly/parallel.py."""
import pytest
import numpy as np

from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide
from puzzle_reconstruction.assembly.parallel import (
    ALL_METHODS,
    DEFAULT_METHODS,
    MethodResult,
    run_all_methods,
    run_selected,
    pick_best,
    pick_best_k,
    summary_table,
    AssemblyRacer,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(2)]
    return frag


def _make_entry(ei, ej, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=0.5))
    return entries


def _make_assembly(frags, score=0.5):
    placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                  for i, f in enumerate(frags)}
    return Assembly(
        fragments=frags,
        placements=list(placements.values()),
        compat_matrix=np.zeros((len(frags), len(frags))),
        total_score=score,
    )


# ─── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_all_methods_is_list(self):
        assert isinstance(ALL_METHODS, list)

    def test_all_methods_not_empty(self):
        assert len(ALL_METHODS) > 0

    def test_default_methods_subset_of_all(self):
        assert all(m in ALL_METHODS for m in DEFAULT_METHODS)

    def test_greedy_in_methods(self):
        assert "greedy" in ALL_METHODS

    def test_default_methods_not_empty(self):
        assert len(DEFAULT_METHODS) > 0


# ─── MethodResult ─────────────────────────────────────────────────────────────

class TestMethodResult:
    def test_success_with_assembly(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags, score=0.8)
        mr = MethodResult(name="greedy", assembly=asm, elapsed=0.1)
        assert mr.success is True

    def test_success_false_without_assembly(self):
        mr = MethodResult(name="greedy", assembly=None)
        assert mr.success is False

    def test_success_false_on_timeout(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags)
        mr = MethodResult(name="greedy", assembly=asm, timed_out=True)
        assert mr.success is False

    def test_score_returns_total_score(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags, score=0.75)
        mr = MethodResult(name="greedy", assembly=asm)
        assert abs(mr.score - 0.75) < 1e-9

    def test_score_zero_when_no_assembly(self):
        mr = MethodResult(name="greedy")
        assert mr.score == 0.0

    def test_method_property_alias(self):
        mr = MethodResult(name="sa")
        assert mr.method == "sa"

    def test_repr_contains_name(self):
        mr = MethodResult(name="beam")
        assert "beam" in repr(mr)

    def test_error_stored(self):
        mr = MethodResult(name="genetic", error="Something went wrong")
        assert mr.error == "Something went wrong"


# ─── pick_best ────────────────────────────────────────────────────────────────

class TestPickBest:
    def test_returns_none_on_empty(self):
        assert pick_best([]) is None

    def test_returns_none_when_all_failed(self):
        results = [
            MethodResult(name="greedy", error="fail"),
            MethodResult(name="sa", timed_out=True),
        ]
        assert pick_best(results) is None

    def test_returns_highest_score(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm_low = _make_assembly(frags, score=0.3)
        asm_high = _make_assembly(frags, score=0.9)
        results = [
            MethodResult(name="greedy", assembly=asm_low),
            MethodResult(name="sa", assembly=asm_high),
        ]
        best = pick_best(results)
        assert best is asm_high

    def test_single_successful_result(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags, score=0.5)
        results = [MethodResult(name="greedy", assembly=asm)]
        assert pick_best(results) is asm


# ─── pick_best_k ──────────────────────────────────────────────────────────────

class TestPickBestK:
    def test_returns_up_to_k_results(self):
        frags = [_make_fragment(i) for i in range(2)]
        results = [
            MethodResult(name="greedy", assembly=_make_assembly(frags, score=0.3)),
            MethodResult(name="sa", assembly=_make_assembly(frags, score=0.8)),
            MethodResult(name="beam", assembly=_make_assembly(frags, score=0.5)),
        ]
        top2 = pick_best_k(results, k=2)
        assert len(top2) == 2

    def test_returns_sorted_by_score_descending(self):
        frags = [_make_fragment(i) for i in range(2)]
        results = [
            MethodResult(name="a", assembly=_make_assembly(frags, score=0.1)),
            MethodResult(name="b", assembly=_make_assembly(frags, score=0.9)),
            MethodResult(name="c", assembly=_make_assembly(frags, score=0.5)),
        ]
        top = pick_best_k(results, k=3)
        scores = [a.total_score for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self):
        assert pick_best_k([], k=3) == []

    def test_fewer_successful_than_k(self):
        frags = [_make_fragment(0)]
        results = [MethodResult(name="greedy", assembly=_make_assembly(frags))]
        top = pick_best_k(results, k=5)
        assert len(top) == 1


# ─── summary_table ────────────────────────────────────────────────────────────

class TestSummaryTable:
    def test_returns_string(self):
        results = [MethodResult(name="greedy")]
        table = summary_table(results)
        assert isinstance(table, str)

    def test_contains_method_name(self):
        results = [MethodResult(name="genetic")]
        table = summary_table(results)
        assert "genetic" in table

    def test_ok_status_shown(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags, score=0.5)
        results = [MethodResult(name="greedy", assembly=asm)]
        table = summary_table(results)
        assert "OK" in table

    def test_timeout_status_shown(self):
        results = [MethodResult(name="sa", timed_out=True)]
        table = summary_table(results)
        assert "TIMEOUT" in table

    def test_empty_results(self):
        table = summary_table([])
        assert isinstance(table, str)


# ─── run_selected ─────────────────────────────────────────────────────────────

class TestRunSelected:
    def test_unknown_method_raises(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        with pytest.raises(ValueError):
            run_selected(frags, entries, methods=["unknown_method"])

    def test_valid_method_runs(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        results = run_selected(frags, entries, methods=["greedy"], seed=0)
        assert len(results) == 1
        assert results[0].name == "greedy"

    def test_returns_list_of_method_results(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        results = run_selected(frags, entries, methods=["greedy"], seed=0)
        assert isinstance(results, list)
        assert all(isinstance(r, MethodResult) for r in results)


# ─── run_all_methods ──────────────────────────────────────────────────────────

class TestRunAllMethods:
    def test_greedy_method_runs(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0)
        assert len(results) == 1
        assert results[0].name == "greedy"

    def test_elapsed_is_nonnegative(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        results = run_all_methods(frags, entries, methods=["greedy"], seed=0)
        assert all(r.elapsed >= 0.0 for r in results)


# ─── AssemblyRacer ────────────────────────────────────────────────────────────

class TestAssemblyRacer:
    def test_racer_creation(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        racer = AssemblyRacer(frags, entries, seed=0)
        assert racer.fragments is frags

    def test_race_returns_list(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["greedy"])
        assert isinstance(results, list)

    def test_race_with_greedy(self):
        frags = [_make_fragment(i) for i in range(2)]
        entries = _make_entries(frags)
        racer = AssemblyRacer(frags, entries, seed=0)
        results = racer.race(methods=["greedy"], timeout=30.0)
        assert len(results) >= 1
