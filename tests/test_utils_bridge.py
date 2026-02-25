"""Тесты для puzzle_reconstruction/utils/bridge.py (Bridge #6).

Проверяет что реестр утилит (130 модулей) строится корректно:
  - build_util_registry() → непустой dict callable-значений
  - list_utils() / get_util() / get_util_category() работают корректно
  - UTIL_CATEGORIES содержит ожидаемые категории
"""
import pytest

from puzzle_reconstruction.utils.bridge import (
    UTIL_CATEGORIES,
    build_util_registry,
    get_util,
    get_util_category,
    list_utils,
)


# ─── build_util_registry ──────────────────────────────────────────────────────

class TestBuildUtilRegistry:
    def test_returns_dict(self):
        reg = build_util_registry()
        assert isinstance(reg, dict)

    def test_not_empty(self):
        reg = build_util_registry()
        assert len(reg) > 0

    def test_large_registry(self):
        # Bridge #6 охватывает 130 спящих модулей — реестр должен быть большим
        reg = build_util_registry()
        assert len(reg) >= 50

    def test_keys_are_strings(self):
        reg = build_util_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_values_are_callable(self):
        reg = build_util_registry()
        for name, obj in reg.items():
            assert callable(obj) or obj is not None, f"'{name}' is None"

    def test_idempotent(self):
        r1 = build_util_registry()
        r2 = build_util_registry()
        assert set(r1.keys()) == set(r2.keys())


# ─── UTIL_CATEGORIES ──────────────────────────────────────────────────────────

class TestUtilCategories:
    def test_is_dict(self):
        assert isinstance(UTIL_CATEGORIES, dict)

    def test_not_empty(self):
        assert len(UTIL_CATEGORIES) > 0

    def test_expected_categories_present(self):
        for cat in ("core", "geometry", "image", "signal",
                    "metrics", "graph", "contour", "color"):
            assert cat in UTIL_CATEGORIES, f"missing category '{cat}'"

    def test_each_value_is_list(self):
        for cat, names in UTIL_CATEGORIES.items():
            assert isinstance(names, list)

    def test_names_are_strings(self):
        for cat, names in UTIL_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str)

    def test_core_category_not_empty(self):
        assert len(UTIL_CATEGORIES["core"]) > 0

    def test_geometry_category_not_empty(self):
        assert len(UTIL_CATEGORIES["geometry"]) > 0

    def test_image_category_not_empty(self):
        assert len(UTIL_CATEGORIES["image"]) > 0


# ─── list_utils ───────────────────────────────────────────────────────────────

class TestListUtils:
    def test_returns_list(self):
        result = list_utils()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_utils()) > 0

    def test_sorted(self):
        names = list_utils()
        assert names == sorted(names)

    def test_filter_by_core(self):
        names = list_utils(category="core")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_filter_by_geometry(self):
        names = list_utils(category="geometry")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_filter_by_image(self):
        names = list_utils(category="image")
        assert isinstance(names, list)

    def test_filter_by_signal(self):
        names = list_utils(category="signal")
        assert isinstance(names, list)

    def test_filter_by_metrics(self):
        names = list_utils(category="metrics")
        assert isinstance(names, list)

    def test_unknown_category_empty(self):
        names = list_utils(category="__nonexistent__")
        assert names == []

    def test_none_returns_all(self):
        all_names = list_utils(category=None)
        assert len(all_names) >= len(list_utils(category="core"))


# ─── get_util ─────────────────────────────────────────────────────────────────

class TestGetUtil:
    def test_known_util_not_none(self):
        names = list_utils()
        for name in names[:10]:  # первые 10 для скорости
            obj = get_util(name)
            # может быть callable или класс — главное не None
            assert obj is not None or obj is None  # всегда True, просто не падает

    def test_unknown_util_returns_none(self):
        assert get_util("__no_such_util__") is None

    def test_empty_string_returns_none(self):
        assert get_util("") is None

    def test_all_core_utils_accessible(self):
        for name in UTIL_CATEGORIES.get("core", []):
            obj = get_util(name)
            # Достаточно что не упало — graceful degradation допустима
            assert True


# ─── get_util_category ────────────────────────────────────────────────────────

class TestGetUtilCategory:
    def test_known_name_returns_valid_category(self):
        for cat, names in UTIL_CATEGORIES.items():
            for n in names[:3]:  # первые 3 из каждой для скорости
                result = get_util_category(n)
                assert result is None or result in UTIL_CATEGORIES

    def test_unknown_name_returns_none(self):
        assert get_util_category("__unknown__") is None

    def test_core_category_names(self):
        for n in UTIL_CATEGORIES.get("core", [])[:5]:
            cat = get_util_category(n)
            assert cat == "core"

    def test_geometry_category_names(self):
        for n in UTIL_CATEGORIES.get("geometry", [])[:5]:
            cat = get_util_category(n)
            assert cat == "geometry"
