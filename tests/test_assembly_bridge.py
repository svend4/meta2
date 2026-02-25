"""Тесты для puzzle_reconstruction/assembly/bridge.py (Bridge #10).

Проверяет что реестр assembly строится корректно и все
зарегистрированные функции доступны как callable.
"""
import pytest

from puzzle_reconstruction.assembly.bridge import (
    ASSEMBLY_CATEGORIES,
    build_assembly_registry,
    get_assembly_fn,
    list_assembly_fns,
)


# ─── build_assembly_registry ──────────────────────────────────────────────────

class TestBuildAssemblyRegistry:
    def test_returns_dict(self):
        reg = build_assembly_registry()
        assert isinstance(reg, dict)

    def test_not_empty(self):
        reg = build_assembly_registry()
        assert len(reg) > 0

    def test_values_are_callable(self):
        reg = build_assembly_registry()
        for name, fn in reg.items():
            assert callable(fn), f"assembly fn '{name}' is not callable"

    def test_keys_are_strings(self):
        reg = build_assembly_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_idempotent_call(self):
        r1 = build_assembly_registry()
        r2 = build_assembly_registry()
        assert set(r1.keys()) == set(r2.keys())

    def test_at_least_10_functions(self):
        reg = build_assembly_registry()
        assert len(reg) >= 10


# ─── ASSEMBLY_CATEGORIES ──────────────────────────────────────────────────────

class TestAssemblyCategories:
    def test_is_dict(self):
        assert isinstance(ASSEMBLY_CATEGORIES, dict)

    def test_not_empty(self):
        assert len(ASSEMBLY_CATEGORIES) > 0

    def test_each_category_is_list(self):
        for cat, names in ASSEMBLY_CATEGORIES.items():
            assert isinstance(names, list)

    def test_names_are_strings(self):
        for cat, names in ASSEMBLY_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str)

    def test_expected_categories(self):
        expected = {"state", "filter", "geometry", "cost", "layout",
                    "scoring", "sequencing", "tracking"}
        for cat in expected:
            assert cat in ASSEMBLY_CATEGORIES, f"missing category '{cat}'"


# ─── list_assembly_fns ────────────────────────────────────────────────────────

class TestListAssemblyFns:
    def test_returns_list(self):
        result = list_assembly_fns()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_assembly_fns()) > 0

    def test_sorted(self):
        names = list_assembly_fns()
        assert names == sorted(names)

    def test_filter_by_state_category(self):
        names = list_assembly_fns(category="state")
        assert isinstance(names, list)

    def test_filter_by_layout_category(self):
        names = list_assembly_fns(category="layout")
        assert isinstance(names, list)

    def test_filter_unknown_category_empty(self):
        names = list_assembly_fns(category="__nonexistent__")
        assert names == []

    def test_none_returns_all(self):
        all_names = list_assembly_fns(category=None)
        assert len(all_names) > 0


# ─── get_assembly_fn ──────────────────────────────────────────────────────────

class TestGetAssemblyFn:
    def test_known_fn_callable_or_none(self):
        for name in list_assembly_fns():
            fn = get_assembly_fn(name)
            assert fn is None or callable(fn), f"'{name}' not callable"

    def test_unknown_name_returns_none(self):
        assert get_assembly_fn("__no_such_fn__") is None

    def test_empty_string_returns_none(self):
        assert get_assembly_fn("") is None

    def test_state_category_functions(self):
        for name in ASSEMBLY_CATEGORIES.get("state", []):
            fn = get_assembly_fn(name)
            assert fn is None or callable(fn)

    def test_layout_category_functions(self):
        for name in ASSEMBLY_CATEGORIES.get("layout", []):
            fn = get_assembly_fn(name)
            assert fn is None or callable(fn)

    def test_scoring_category_functions(self):
        for name in ASSEMBLY_CATEGORIES.get("scoring", []):
            fn = get_assembly_fn(name)
            assert fn is None or callable(fn)
