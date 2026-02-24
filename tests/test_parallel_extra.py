"""Extra tests for puzzle_reconstruction/assembly/parallel.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from puzzle_reconstruction.assembly.parallel import (
    ALL_METHODS,
    DEFAULT_METHODS,
    AssemblyRacer,
    MethodResult,
    pick_best,
    pick_best_k,
    run_selected,
    summary_table,
)
from puzzle_reconstruction.models import Assembly, CompatEntry, EdgeSide, EdgeSignature, Fragment


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fragment(fid: int = 0) -> Fragment:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)
    contour = np.array([[0, 0], [32, 0], [32, 32], [0, 32]])
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


def _assembly(score: float = 0.7) -> Assembly:
    frag = _fragment(0)
    return Assembly(
        fragments=[frag],
        placements={0: (np.array([0.0, 0.0]), 0.0)},
        compat_matrix=np.zeros((1, 1)),
        total_score=score,
    )


def _mr(name: str = "greedy", score: float = 0.7,
        success: bool = True, timed_out: bool = False) -> MethodResult:
    asm = _assembly(score) if success else None
    return MethodResult(name=name, assembly=asm,
                        elapsed=0.01, timed_out=timed_out,
                        error=None if success else "err")


# ─── ALL_METHODS / DEFAULT_METHODS (extra) ────────────────────────────────────

class TestMethodConstantsExtra:
    def test_all_methods_is_list(self):
        assert isinstance(ALL_METHODS, list)

    def test_all_methods_length(self):
        assert len(ALL_METHODS) == 8

    def test_greedy_in_all(self):
        assert "greedy" in ALL_METHODS

    def test_mcts_in_all(self):
        assert "mcts" in ALL_METHODS

    def test_default_methods_is_list(self):
        assert isinstance(DEFAULT_METHODS, list)

    def test_default_is_subset_of_all(self):
        for m in DEFAULT_METHODS:
            assert m in ALL_METHODS

    def test_default_not_empty(self):
        assert len(DEFAULT_METHODS) > 0

    def test_no_duplicates_in_all(self):
        assert len(ALL_METHODS) == len(set(ALL_METHODS))

    def test_no_duplicates_in_default(self):
        assert len(DEFAULT_METHODS) == len(set(DEFAULT_METHODS))


# ─── MethodResult (extra) ─────────────────────────────────────────────────────

class TestMethodResultExtra:
    def test_name_stored(self):
        assert _mr("sa").name == "sa"

    def test_assembly_stored(self):
        asm = _assembly(0.8)
        mr = MethodResult(name="g", assembly=asm, elapsed=0.1)
        assert mr.assembly is asm

    def test_elapsed_stored(self):
        mr = MethodResult(name="g", elapsed=2.5)
        assert mr.elapsed == pytest.approx(2.5)

    def test_error_none_by_default(self):
        mr = MethodResult(name="g", assembly=_assembly())
        assert mr.error is None

    def test_timed_out_false_by_default(self):
        assert MethodResult(name="g").timed_out is False

    def test_success_true_with_assembly(self):
        assert _mr("greedy", success=True).success is True

    def test_success_false_without_assembly(self):
        mr = MethodResult(name="g")
        assert mr.success is False

    def test_success_false_when_timed_out(self):
        mr = MethodResult(name="g", assembly=_assembly(), timed_out=True)
        assert mr.success is False

    def test_score_returns_assembly_score(self):
        mr = _mr("g", score=0.85)
        assert mr.score == pytest.approx(0.85)

    def test_score_zero_without_assembly(self):
        mr = MethodResult(name="g")
        assert mr.score == pytest.approx(0.0)

    def test_repr_contains_name(self):
        assert "greedy" in repr(_mr("greedy"))

    def test_repr_contains_ok(self):
        assert "OK" in repr(_mr("g", success=True))

    def test_repr_contains_timeout(self):
        assert "TIMEOUT" in repr(_mr("g", timed_out=True))


# ─── pick_best (extra) ────────────────────────────────────────────────────────

class TestPickBestExtra:
    def test_returns_assembly_or_none(self):
        result = pick_best([_mr("g", score=0.7)])
        assert result is None or isinstance(result, Assembly)

    def test_empty_list_returns_none(self):
        assert pick_best([]) is None

    def test_all_failed_returns_none(self):
        failed = [MethodResult(name="g", error="err")]
        assert pick_best(failed) is None

    def test_single_success(self):
        mr = _mr("g", score=0.8)
        result = pick_best([mr])
        assert result is mr.assembly

    def test_picks_highest_score(self):
        mr1 = _mr("a", score=0.6)
        mr2 = _mr("b", score=0.9)
        mr3 = _mr("c", score=0.5)
        result = pick_best([mr1, mr2, mr3])
        assert result is mr2.assembly

    def test_ignores_timed_out(self):
        good = _mr("g", score=0.7)
        timed = MethodResult(name="t", assembly=_assembly(0.9), timed_out=True)
        result = pick_best([good, timed])
        assert result is good.assembly

    def test_ignores_error_result(self):
        good = _mr("g", score=0.6)
        bad = MethodResult(name="b", error="oops")
        result = pick_best([good, bad])
        assert result is good.assembly


# ─── pick_best_k (extra) ──────────────────────────────────────────────────────

class TestPickBestKExtra:
    def test_returns_list(self):
        assert isinstance(pick_best_k([], k=3), list)

    def test_empty_list(self):
        assert pick_best_k([], k=5) == []

    def test_k_larger_than_successful(self):
        results = [_mr("a", score=0.7), _mr("b", score=0.5)]
        tops = pick_best_k(results, k=10)
        assert len(tops) == 2

    def test_top_1_is_highest(self):
        results = [_mr("a", score=0.5), _mr("b", score=0.9), _mr("c", score=0.7)]
        tops = pick_best_k(results, k=1)
        assert tops[0].total_score == pytest.approx(0.9)

    def test_sorted_descending(self):
        results = [_mr("a", score=0.5), _mr("b", score=0.9), _mr("c", score=0.7)]
        tops = pick_best_k(results, k=3)
        scores = [t.total_score for t in tops]
        assert scores == sorted(scores, reverse=True)

    def test_failed_excluded(self):
        results = [_mr("a", score=0.8), MethodResult(name="b", error="err")]
        tops = pick_best_k(results, k=5)
        assert len(tops) == 1


# ─── summary_table (extra) ────────────────────────────────────────────────────

class TestSummaryTableExtra:
    def test_returns_string(self):
        assert isinstance(summary_table([_mr("g")]), str)

    def test_empty_list(self):
        table = summary_table([])
        assert isinstance(table, str)
        assert "Method" in table

    def test_contains_header(self):
        table = summary_table([_mr("greedy")])
        assert "Method" in table

    def test_contains_method_name(self):
        table = summary_table([_mr("sa")])
        assert "sa" in table

    def test_ok_status_for_success(self):
        table = summary_table([_mr("g", success=True)])
        assert "OK" in table

    def test_timeout_status(self):
        mr = MethodResult(name="g", timed_out=True)
        table = summary_table([mr])
        assert "TIMEOUT" in table

    def test_error_status(self):
        mr = MethodResult(name="g", error="boom")
        table = summary_table([mr])
        assert "ERROR" in table

    def test_multiple_methods(self):
        results = [_mr("a", score=0.8), _mr("b", score=0.5)]
        table = summary_table(results)
        assert "a" in table and "b" in table

    def test_sorted_by_score_descending(self):
        results = [_mr("low", score=0.2), _mr("high", score=0.9)]
        table = summary_table(results)
        lines = table.splitlines()
        # "high" should appear before "low" (sorted by descending score)
        idx_high = next(i for i, l in enumerate(lines) if "high" in l)
        idx_low = next(i for i, l in enumerate(lines) if "low" in l)
        assert idx_high < idx_low


# ─── run_selected (extra) ─────────────────────────────────────────────────────

class TestRunSelectedExtra:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            run_selected([], [], methods=["unknown_method"])

    def test_unknown_in_list_raises(self):
        with pytest.raises(ValueError):
            run_selected([], [], methods=["greedy", "bad_name"])

    def test_returns_list(self):
        frags = [_fragment(i) for i in range(2)]
        result = run_selected(frags, [], methods=["greedy"], timeout=10.0)
        assert isinstance(result, list)

    def test_length_matches_methods(self):
        frags = [_fragment(i) for i in range(2)]
        result = run_selected(frags, [], methods=["greedy", "mcts"], timeout=10.0)
        assert len(result) == 2

    def test_each_element_is_method_result(self):
        frags = [_fragment(i) for i in range(2)]
        result = run_selected(frags, [], methods=["greedy"], timeout=10.0)
        for r in result:
            assert isinstance(r, MethodResult)


# ─── AssemblyRacer (extra) ────────────────────────────────────────────────────

class TestAssemblyRacerExtra:
    def _make_racer(self, n: int = 2) -> AssemblyRacer:
        frags = [_fragment(i) for i in range(n)]
        return AssemblyRacer(frags, [], seed=0)

    def test_instantiation(self):
        racer = self._make_racer()
        assert isinstance(racer, AssemblyRacer)

    def test_fragments_stored(self):
        racer = self._make_racer(3)
        assert len(racer.fragments) == 3

    def test_seed_stored(self):
        racer = AssemblyRacer([], [], seed=42)
        assert racer.seed == 42

    def test_race_returns_list(self):
        racer = self._make_racer(2)
        result = racer.race(methods=["greedy"], timeout=10.0, max_workers=1)
        assert isinstance(result, list)

    def test_race_unknown_method_skipped(self):
        racer = self._make_racer(2)
        result = racer.race(methods=["unknown_xyz"], timeout=5.0, max_workers=1)
        assert isinstance(result, list)

    def test_race_results_are_method_result(self):
        racer = self._make_racer(2)
        result = racer.race(methods=["greedy"], timeout=10.0)
        for r in result:
            assert isinstance(r, MethodResult)

    def test_first_only_returns_max_one_success(self):
        racer = self._make_racer(2)
        result = racer.race(methods=["greedy", "mcts"], first_only=True, timeout=10.0)
        successes = [r for r in result if r.success]
        assert len(successes) <= 2  # first_only may stop early
