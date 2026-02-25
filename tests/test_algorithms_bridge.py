"""Тесты для puzzle_reconstruction/algorithms/bridge.py (Bridge #5).

Проверяет что реестр алгоритмов строится корректно и все
зарегистрированные функции доступны как callable.
"""
import pytest

from puzzle_reconstruction.algorithms.bridge import (
    ALGORITHM_CATEGORIES,
    build_algorithm_registry,
    get_algorithm,
    list_algorithms,
)


# ─── build_algorithm_registry ─────────────────────────────────────────────────

class TestBuildAlgorithmRegistry:
    def test_returns_dict(self):
        reg = build_algorithm_registry()
        assert isinstance(reg, dict)

    def test_not_empty(self):
        reg = build_algorithm_registry()
        assert len(reg) > 0

    def test_values_are_callable(self):
        reg = build_algorithm_registry()
        for name, fn in reg.items():
            assert callable(fn), f"algorithm '{name}' is not callable"

    def test_keys_are_strings(self):
        reg = build_algorithm_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_idempotent_call(self):
        r1 = build_algorithm_registry()
        r2 = build_algorithm_registry()
        assert set(r1.keys()) == set(r2.keys())

    def test_at_least_20_algorithms(self):
        reg = build_algorithm_registry()
        assert len(reg) >= 20


# ─── ALGORITHM_CATEGORIES ─────────────────────────────────────────────────────

class TestAlgorithmCategories:
    def test_is_dict(self):
        assert isinstance(ALGORITHM_CATEGORIES, dict)

    def test_not_empty(self):
        assert len(ALGORITHM_CATEGORIES) > 0

    def test_expected_categories_present(self):
        for cat in ("fragment", "pair", "assembly"):
            assert cat in ALGORITHM_CATEGORIES, f"missing category '{cat}'"

    def test_each_value_is_list(self):
        for cat, names in ALGORITHM_CATEGORIES.items():
            assert isinstance(names, list)

    def test_names_are_strings(self):
        for cat, names in ALGORITHM_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str)

    def test_fragment_category_not_empty(self):
        assert len(ALGORITHM_CATEGORIES["fragment"]) > 0

    def test_pair_category_not_empty(self):
        assert len(ALGORITHM_CATEGORIES["pair"]) > 0


# ─── list_algorithms ──────────────────────────────────────────────────────────

class TestListAlgorithms:
    def test_returns_list(self):
        result = list_algorithms()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_algorithms()) > 0

    def test_sorted(self):
        names = list_algorithms()
        assert names == sorted(names)

    def test_filter_by_fragment_category(self):
        names = list_algorithms(category="fragment")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_filter_by_pair_category(self):
        names = list_algorithms(category="pair")
        assert isinstance(names, list)

    def test_filter_by_assembly_category(self):
        names = list_algorithms(category="assembly")
        assert isinstance(names, list)

    def test_filter_unknown_category_empty(self):
        names = list_algorithms(category="__no_category__")
        assert names == []

    def test_none_returns_all(self):
        assert len(list_algorithms(category=None)) > 0


# ─── get_algorithm ────────────────────────────────────────────────────────────

class TestGetAlgorithm:
    def test_known_algorithm_callable_or_none(self):
        for name in list_algorithms():
            fn = get_algorithm(name)
            assert fn is None or callable(fn), f"'{name}' not callable"

    def test_unknown_algorithm_returns_none(self):
        assert get_algorithm("__no_such_alg__") is None

    def test_empty_string_returns_none(self):
        assert get_algorithm("") is None

    def test_fragment_category_algorithms(self):
        for name in ALGORITHM_CATEGORIES.get("fragment", []):
            fn = get_algorithm(name)
            assert fn is None or callable(fn)

    def test_pair_category_algorithms(self):
        for name in ALGORITHM_CATEGORIES.get("pair", []):
            fn = get_algorithm(name)
            assert fn is None or callable(fn)

    def test_assembly_category_algorithms(self):
        for name in ALGORITHM_CATEGORIES.get("assembly", []):
            fn = get_algorithm(name)
            assert fn is None or callable(fn)
