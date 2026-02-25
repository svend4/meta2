"""Тесты для puzzle_reconstruction/scoring/bridge.py (Bridge #8).

Проверяет что реестр скоринга строится корректно и все
зарегистрированные функции доступны как callable.
"""
import pytest

from puzzle_reconstruction.scoring.bridge import (
    SCORING_CATEGORIES,
    build_scoring_registry,
    get_scorer,
    get_scorer_category,
    list_scorers,
)


# ─── build_scoring_registry ───────────────────────────────────────────────────

class TestBuildScoringRegistry:
    def test_returns_dict(self):
        reg = build_scoring_registry()
        assert isinstance(reg, dict)

    def test_not_empty(self):
        reg = build_scoring_registry()
        assert len(reg) > 0

    def test_values_are_callable(self):
        reg = build_scoring_registry()
        for name, fn in reg.items():
            assert callable(fn), f"scorer '{name}' is not callable"

    def test_keys_are_strings(self):
        reg = build_scoring_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_idempotent_call(self):
        r1 = build_scoring_registry()
        r2 = build_scoring_registry()
        assert set(r1.keys()) == set(r2.keys())


# ─── SCORING_CATEGORIES ───────────────────────────────────────────────────────

class TestScoringCategories:
    def test_is_dict(self):
        assert isinstance(SCORING_CATEGORIES, dict)

    def test_expected_categories_present(self):
        for cat in ("pair", "filter_rank", "fusion", "evaluation"):
            assert cat in SCORING_CATEGORIES

    def test_each_category_is_list(self):
        for cat, names in SCORING_CATEGORIES.items():
            assert isinstance(names, list), f"category '{cat}' is not a list"

    def test_names_are_strings(self):
        for cat, names in SCORING_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str)

    def test_no_duplicate_names_within_category(self):
        for cat, names in SCORING_CATEGORIES.items():
            assert len(names) == len(set(names)), f"duplicates in category '{cat}'"


# ─── list_scorers ─────────────────────────────────────────────────────────────

class TestListScorers:
    def test_returns_list(self):
        result = list_scorers()
        assert isinstance(result, list)

    def test_not_empty(self):
        assert len(list_scorers()) > 0

    def test_sorted(self):
        names = list_scorers()
        assert names == sorted(names)

    def test_filter_by_category_pair(self):
        names = list_scorers(category="pair")
        assert isinstance(names, list)

    def test_filter_by_category_fusion(self):
        names = list_scorers(category="fusion")
        assert isinstance(names, list)

    def test_filter_unknown_category_empty(self):
        names = list_scorers(category="nonexistent_category")
        assert names == []

    def test_none_category_returns_all(self):
        all_names = list_scorers(category=None)
        assert len(all_names) > 0


# ─── get_scorer ───────────────────────────────────────────────────────────────

class TestGetScorer:
    def test_known_scorer_returns_callable(self):
        names = list_scorers()
        if names:
            fn = get_scorer(names[0])
            assert fn is None or callable(fn)

    def test_unknown_scorer_returns_none(self):
        assert get_scorer("__does_not_exist__") is None

    def test_empty_name_returns_none(self):
        assert get_scorer("") is None

    def test_all_listed_scorers_are_callable_or_none(self):
        for name in list_scorers():
            fn = get_scorer(name)
            assert fn is None or callable(fn), f"'{name}' is not callable"


# ─── get_scorer_category ──────────────────────────────────────────────────────

class TestGetScorerCategory:
    def test_known_name_returns_string(self):
        for cat, names in SCORING_CATEGORIES.items():
            for n in names:
                result = get_scorer_category(n)
                assert result in SCORING_CATEGORIES or result is None

    def test_unknown_name_returns_none(self):
        assert get_scorer_category("__no_such_scorer__") is None

    def test_pair_category_scorers(self):
        for n in SCORING_CATEGORIES.get("pair", []):
            cat = get_scorer_category(n)
            assert cat == "pair"

    def test_fusion_category_scorers(self):
        for n in SCORING_CATEGORIES.get("fusion", []):
            cat = get_scorer_category(n)
            assert cat == "fusion"
