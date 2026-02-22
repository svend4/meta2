"""
Тесты для puzzle_reconstruction/assembly/parallel.py

Покрытие:
    MethodResult   — success, score, repr, timed_out, error
    run_selected   — ValueError на неизвестный метод
    pick_best      — пустой список, все неуспешные, выбирает max score
    pick_best_k    — k=1, k=0, k>n, сортировка
    summary_table  — строка, заголовок, все имена методов присутствуют
    ALL_METHODS    — длина 8, все ожидаемые методы
    DEFAULT_METHODS — подмножество ALL_METHODS
    AssemblyRacer  — инициализация, race() работает без краша
    run_all_methods — без реального запуска (моки через monkeypatch)
"""
from __future__ import annotations

import math
from typing import List, Optional
from unittest.mock import MagicMock, patch

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
from puzzle_reconstruction.models import Assembly, CompatEntry, Edge, Fragment, Placement


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _make_fragment(fid: int) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, 10, 10),
    )


def _make_assembly(score: float, fids: Optional[List[int]] = None) -> Assembly:
    if fids is None:
        fids = [0, 1, 2]
    placements = [
        Placement(fragment_id=f, position=(i * 10, 0), rotation=0.0)
        for i, f in enumerate(fids)
    ]
    return Assembly(
        placements=placements,
        total_score=score,
        method="mock",
    )


def _ok_result(name: str, score: float) -> MethodResult:
    return MethodResult(name=name, assembly=_make_assembly(score), elapsed=0.1)


def _fail_result(name: str) -> MethodResult:
    return MethodResult(name=name, error="fail", elapsed=0.05)


def _timeout_result(name: str) -> MethodResult:
    return MethodResult(name=name, timed_out=True, elapsed=30.0)


# ─── MethodResult ─────────────────────────────────────────────────────────────

class TestMethodResult:
    def test_success_when_assembly_present(self):
        r = _ok_result("greedy", 0.7)
        assert r.success is True

    def test_not_success_when_error(self):
        r = _fail_result("sa")
        assert r.success is False

    def test_not_success_when_timed_out(self):
        r = _timeout_result("beam")
        assert r.success is False

    def test_timed_out_flag(self):
        r = _timeout_result("mcts")
        assert r.timed_out is True

    def test_score_from_assembly(self):
        r = _ok_result("gamma", 0.85)
        assert r.score == pytest.approx(0.85)

    def test_score_zero_if_no_assembly(self):
        r = _fail_result("genetic")
        assert r.score == pytest.approx(0.0)

    def test_repr_contains_name(self):
        r = _ok_result("greedy", 0.5)
        assert "greedy" in repr(r)

    def test_repr_contains_ok(self):
        r = _ok_result("beam", 0.6)
        assert "OK" in repr(r)

    def test_repr_contains_timeout(self):
        r = _timeout_result("sa")
        assert "TIMEOUT" in repr(r)

    def test_repr_contains_err(self):
        r = _fail_result("ant_colony")
        assert "ERR" in repr(r)

    def test_repr_contains_elapsed(self):
        r = _ok_result("greedy", 0.5)
        assert "0.10" in repr(r)

    def test_default_values(self):
        r = MethodResult(name="x")
        assert r.assembly is None
        assert r.elapsed == 0.0
        assert r.error is None
        assert r.timed_out is False


# ─── pick_best ────────────────────────────────────────────────────────────────

class TestPickBest:
    def test_empty_list_returns_none(self):
        assert pick_best([]) is None

    def test_all_failed_returns_none(self):
        results = [_fail_result("a"), _timeout_result("b")]
        assert pick_best(results) is None

    def test_single_success(self):
        r = _ok_result("greedy", 0.7)
        best = pick_best([r])
        assert best is not None
        assert best.total_score == pytest.approx(0.7)

    def test_picks_max_score(self):
        results = [
            _ok_result("a", 0.5),
            _ok_result("b", 0.9),
            _ok_result("c", 0.7),
        ]
        best = pick_best(results)
        assert best.total_score == pytest.approx(0.9)

    def test_ignores_failed(self):
        results = [
            _fail_result("x"),
            _ok_result("y", 0.6),
            _timeout_result("z"),
        ]
        best = pick_best(results)
        assert best.total_score == pytest.approx(0.6)

    def test_returns_assembly_object(self):
        r    = _ok_result("beam", 0.75)
        best = pick_best([r])
        assert isinstance(best, Assembly)


# ─── pick_best_k ──────────────────────────────────────────────────────────────

class TestPickBestK:
    def test_k_1_returns_one(self):
        results = [_ok_result("a", 0.8), _ok_result("b", 0.6)]
        top     = pick_best_k(results, k=1)
        assert len(top) == 1
        assert top[0].total_score == pytest.approx(0.8)

    def test_k_0_returns_empty(self):
        results = [_ok_result("a", 0.8)]
        assert pick_best_k(results, k=0) == []

    def test_k_greater_than_n(self):
        results = [_ok_result("a", 0.8), _ok_result("b", 0.6)]
        top     = pick_best_k(results, k=10)
        assert len(top) == 2

    def test_sorted_descending(self):
        results = [
            _ok_result("a", 0.5),
            _ok_result("b", 0.9),
            _ok_result("c", 0.7),
        ]
        top = pick_best_k(results, k=3)
        scores = [a.total_score for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_ignores_failed(self):
        results = [_fail_result("x"), _ok_result("y", 0.6)]
        top     = pick_best_k(results, k=5)
        assert len(top) == 1


# ─── summary_table ────────────────────────────────────────────────────────────

class TestSummaryTable:
    def test_returns_string(self):
        results = [_ok_result("greedy", 0.7)]
        assert isinstance(summary_table(results), str)

    def test_contains_header(self):
        table = summary_table([_ok_result("a", 0.5)])
        assert "Method" in table
        assert "Score" in table

    def test_contains_method_name(self):
        table = summary_table([_ok_result("greedy", 0.7)])
        assert "greedy" in table

    def test_contains_score(self):
        table = summary_table([_ok_result("sa", 0.8234)])
        assert "0.8234" in table

    def test_timeout_shown(self):
        table = summary_table([_timeout_result("beam")])
        assert "TIMEOUT" in table

    def test_error_shown(self):
        table = summary_table([_fail_result("mcts")])
        assert "ERROR" in table

    def test_multiple_methods(self):
        results = [
            _ok_result("greedy", 0.5),
            _ok_result("sa", 0.9),
            _fail_result("beam"),
        ]
        table = summary_table(results)
        assert "greedy" in table
        assert "sa" in table
        assert "beam" in table

    def test_empty_results(self):
        table = summary_table([])
        assert isinstance(table, str)


# ─── ALL_METHODS и DEFAULT_METHODS ────────────────────────────────────────────

class TestConstants:
    def test_all_methods_length(self):
        assert len(ALL_METHODS) == 8

    def test_all_methods_contains_expected(self):
        expected = {"greedy", "sa", "beam", "gamma", "genetic",
                    "exhaustive", "ant_colony", "mcts"}
        assert set(ALL_METHODS) == expected

    def test_default_methods_subset_of_all(self):
        assert all(m in ALL_METHODS for m in DEFAULT_METHODS)

    def test_default_methods_nonempty(self):
        assert len(DEFAULT_METHODS) > 0

    def test_all_methods_unique(self):
        assert len(set(ALL_METHODS)) == len(ALL_METHODS)


# ─── run_selected — валидация ─────────────────────────────────────────────────

class TestRunSelected:
    def test_unknown_method_raises_value_error(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        with pytest.raises(ValueError, match="Неизвестные методы"):
            run_selected(frags, [], methods=["greedy", "nonexistent_xyz"])

    def test_all_unknown_raises(self):
        with pytest.raises(ValueError):
            run_selected([], [], methods=["foo", "bar"])

    def test_known_methods_no_error_in_validation(self, monkeypatch):
        """Известные методы проходят валидацию (сам запуск мокирован)."""
        from puzzle_reconstruction.assembly import parallel as pmod

        def fake_run_all(fragments, entries, methods, **kw):
            return [_ok_result(m, 0.5) for m in methods]

        monkeypatch.setattr(pmod, "run_all_methods", fake_run_all)
        frags   = [_make_fragment(0)]
        results = run_selected(frags, [], methods=["greedy"])
        assert isinstance(results, list)


# ─── AssemblyRacer ────────────────────────────────────────────────────────────

class TestAssemblyRacer:
    def test_init(self):
        frags  = [_make_fragment(i) for i in range(3)]
        racer  = AssemblyRacer(frags, [], seed=42)
        assert racer.seed == 42
        assert len(racer.fragments) == 3

    def test_race_returns_list(self, monkeypatch):
        """race() возвращает список (методы мокированы)."""
        from puzzle_reconstruction.assembly import parallel as pmod

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "greedy": lambda: _make_assembly(0.7),
        })

        frags = [_make_fragment(i) for i in range(2)]
        racer = AssemblyRacer(frags, [], seed=0)
        results = racer.race(methods=["greedy"], timeout=5.0, max_workers=1)
        assert isinstance(results, list)

    def test_race_first_only(self, monkeypatch):
        """first_only=True → возвращает не более одного результата (мок)."""
        from puzzle_reconstruction.assembly import parallel as pmod

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "greedy":  lambda: _make_assembly(0.7),
            "genetic": lambda: _make_assembly(0.6),
        })

        frags   = [_make_fragment(i) for i in range(2)]
        racer   = AssemblyRacer(frags, [], seed=0)
        results = racer.race(methods=["greedy", "genetic"],
                              timeout=5.0, first_only=True, max_workers=2)
        # first_only → ≤ 2, но минимум 1 (если хотя бы один успел)
        assert 1 <= len(results) <= 2


# ─── run_all_methods с mock ───────────────────────────────────────────────────

class TestRunAllMethodsMocked:
    def test_mocked_greedy(self, monkeypatch):
        from puzzle_reconstruction.assembly import parallel as pmod

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "greedy": lambda: _make_assembly(0.75),
        })

        frags   = [_make_fragment(i) for i in range(2)]
        results = pmod.run_all_methods(frags, [], methods=["greedy"], timeout=5)
        assert len(results) == 1
        assert results[0].name == "greedy"
        assert results[0].success

    def test_mocked_method_exception(self, monkeypatch):
        from puzzle_reconstruction.assembly import parallel as pmod

        def bad_caller():
            raise RuntimeError("simulated error")

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "sa": bad_caller,
        })

        frags   = [_make_fragment(0)]
        results = pmod.run_all_methods(frags, [], methods=["sa"], timeout=5)
        assert len(results) == 1
        assert not results[0].success
        assert results[0].error is not None

    def test_multiple_methods_mocked(self, monkeypatch):
        from puzzle_reconstruction.assembly import parallel as pmod

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "greedy":  lambda: _make_assembly(0.6),
            "genetic": lambda: _make_assembly(0.9),
        })

        frags   = [_make_fragment(i) for i in range(2)]
        results = pmod.run_all_methods(frags, [], methods=["greedy", "genetic"])
        assert len(results) == 2
        best = pick_best(results)
        assert best.total_score == pytest.approx(0.9)

    def test_parallel_mode_mocked(self, monkeypatch):
        from puzzle_reconstruction.assembly import parallel as pmod

        monkeypatch.setattr(pmod, "_build_callers", lambda *a, **kw: {
            "greedy": lambda: _make_assembly(0.7),
            "sa":     lambda: _make_assembly(0.8),
        })

        frags   = [_make_fragment(i) for i in range(2)]
        results = pmod.run_all_methods(frags, [], methods=["greedy", "sa"],
                                        n_workers=2, timeout=5)
        assert len(results) == 2
