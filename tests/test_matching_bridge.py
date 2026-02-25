"""Тесты для puzzle_reconstruction/matching/bridge.py (Bridge #9).

Проверяет что реестр matching строится корректно и все
зарегистрированные функции доступны как callable.
"""
import pytest

from puzzle_reconstruction.matching.bridge import (
    MATCHER_CATEGORIES,
    build_matcher_registry,
    get_matcher,
    get_matcher_category,
    list_matchers,
)


# ─── build_matcher_registry ───────────────────────────────────────────────────

class TestBuildMatcherRegistry:
    def test_returns_dict(self):
        reg = build_matcher_registry()
        assert isinstance(reg, dict)

    def test_not_empty(self):
        reg = build_matcher_registry()
        assert len(reg) > 0

    def test_values_are_callable(self):
        reg = build_matcher_registry()
        for name, fn in reg.items():
            assert callable(fn), f"matcher '{name}' is not callable"

    def test_keys_are_strings(self):
        reg = build_matcher_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_idempotent_call(self):
        r1 = build_matcher_registry()
        r2 = build_matcher_registry()
        assert set(r1.keys()) == set(r2.keys())


# ─── MATCHER_CATEGORIES ────────────────────────────────────────────────

class TestMatcherBridgeCategories:
    def test_is_dict(self):
        assert isinstance(MATCHER_CATEGORIES, dict)

    def test_not_empty(self):
        assert len(MATCHER_CATEGORIES) > 0

    def test_each_category_is_list(self):
        for cat, names in MATCHER_CATEGORIES.items():
            assert isinstance(names, list), f"category '{cat}' is not a list"

    def test_names_are_strings(self):
        for cat, names in MATCHER_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str)


# ─── list_matchers ────────────────────────────────────────────────────────────

class TestListMatchers:
    def test_returns_list(self):
        result = list_matchers()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_matchers()) > 0

    def test_sorted(self):
        names = list_matchers()
        assert names == sorted(names)

    def test_filter_by_category(self):
        for cat in MATCHER_CATEGORIES:
            names = list_matchers(category=cat)
            assert isinstance(names, list)

    def test_filter_unknown_category_empty(self):
        names = list_matchers(category="__no_such_category__")
        assert names == []

    def test_none_category_returns_all(self):
        all_names = list_matchers(category=None)
        assert len(all_names) > 0


# ─── get_matcher ───────────────────────────────────────────────────────────

class TestGetMatcherFn:
    def test_known_matcher_callable_or_none(self):
        for name in list_matchers():
            fn = get_matcher(name)
            assert fn is None or callable(fn), f"'{name}' is not callable"

    def test_unknown_matcher_returns_none(self):
        assert get_matcher("__does_not_exist__") is None

    def test_empty_name_returns_none(self):
        assert get_matcher("") is None

    def test_distance_category_functions(self):
        for name in MATCHER_CATEGORIES.get("distance", []):
            fn = get_matcher(name)
            assert fn is None or callable(fn)

    def test_geometric_category_functions(self):
        for name in MATCHER_CATEGORIES.get("geometric", []):
            fn = get_matcher(name)
            assert fn is None or callable(fn)

    def test_appearance_category_functions(self):
        for name in MATCHER_CATEGORIES.get("appearance", []):
            fn = get_matcher(name)
            assert fn is None or callable(fn)
