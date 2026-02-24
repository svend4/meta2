"""Extra tests for puzzle_reconstruction/assembly/parallel.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly
from puzzle_reconstruction.assembly.parallel import (
    MethodResult,
    ALL_METHODS,
    DEFAULT_METHODS,
    pick_best,
    pick_best_k,
    summary_table,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _asm(score=0.5):
    return Assembly(
        fragments=[], placements={},
        compat_matrix=np.array([]), total_score=score,
    )


def _mr(name, score=0.5, success=True, timed_out=False):
    if success:
        return MethodResult(name=name, assembly=_asm(score))
    return MethodResult(name=name, timed_out=timed_out,
                        error=None if not timed_out else None)


# ─── MethodResult ───────────────────────────────────────────────────────────

class TestMethodResultExtra:
    def test_success(self):
        mr = _mr("greedy", 0.8)
        assert mr.success is True
        assert mr.score == pytest.approx(0.8)

    def test_failure(self):
        mr = MethodResult(name="bad", error="oops")
        assert mr.success is False
        assert mr.score == pytest.approx(0.0)

    def test_timeout(self):
        mr = MethodResult(name="slow", timed_out=True)
        assert mr.success is False

    def test_repr(self):
        mr = _mr("greedy", 0.7)
        s = repr(mr)
        assert "greedy" in s
        assert "OK" in s

    def test_repr_timeout(self):
        mr = MethodResult(name="slow", timed_out=True)
        s = repr(mr)
        assert "TIMEOUT" in s

    def test_repr_error(self):
        mr = MethodResult(name="bad", error="fail")
        s = repr(mr)
        assert "ERR" in s


# ─── constants ──────────────────────────────────────────────────────────────

class TestConstantsExtra:
    def test_all_methods(self):
        assert len(ALL_METHODS) >= 4
        for m in DEFAULT_METHODS:
            assert m in ALL_METHODS

    def test_default_methods(self):
        assert "greedy" in DEFAULT_METHODS


# ─── pick_best ──────────────────────────────────────────────────────────────

class TestPickBestExtra:
    def test_empty(self):
        assert pick_best([]) is None

    def test_all_failed(self):
        results = [MethodResult(name="a", error="fail"),
                   MethodResult(name="b", timed_out=True)]
        assert pick_best(results) is None

    def test_picks_highest(self):
        results = [_mr("a", 0.5), _mr("b", 0.9), _mr("c", 0.7)]
        best = pick_best(results)
        assert best is not None
        assert best.total_score == pytest.approx(0.9)

    def test_single_success(self):
        results = [_mr("a", 0.5)]
        best = pick_best(results)
        assert best is not None


# ─── pick_best_k ────────────────────────────────────────────────────────────

class TestPickBestKExtra:
    def test_empty(self):
        assert pick_best_k([], 3) == []

    def test_k_larger(self):
        results = [_mr("a", 0.5)]
        top = pick_best_k(results, 5)
        assert len(top) == 1

    def test_k_exact(self):
        results = [_mr("a", 0.3), _mr("b", 0.9), _mr("c", 0.7)]
        top = pick_best_k(results, 2)
        assert len(top) == 2
        assert top[0].total_score >= top[1].total_score


# ─── summary_table ──────────────────────────────────────────────────────────

class TestSummaryTableExtra:
    def test_empty(self):
        s = summary_table([])
        assert "Method" in s

    def test_with_data(self):
        results = [_mr("greedy", 0.7), _mr("sa", 0.8)]
        s = summary_table(results)
        assert "greedy" in s
        assert "sa" in s
        assert "OK" in s

    def test_timeout(self):
        results = [MethodResult(name="slow", timed_out=True)]
        s = summary_table(results)
        assert "TIMEOUT" in s
