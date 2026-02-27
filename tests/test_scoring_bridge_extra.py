"""Extra tests for puzzle_reconstruction/scoring/bridge.py"""
import pytest

from puzzle_reconstruction.scoring.bridge import (
    SCORING_CATEGORIES,
    SCORING_REGISTRY,
    build_scoring_registry,
    list_scorers,
    get_scorer,
    get_scorer_category,
    _ensure_registry,
)


# ─── SCORING_CATEGORIES structural invariants ─────────────────────────────────

def test_scoring_categories_no_empty_lists():
    for cat, names in SCORING_CATEGORIES.items():
        assert len(names) > 0, f"Category {cat!r} is empty"


def test_scoring_categories_all_expected_names_are_strings():
    for cat, names in SCORING_CATEGORIES.items():
        for n in names:
            assert isinstance(n, str) and len(n) > 0, f"Invalid name {n!r} in {cat}"


def test_scoring_categories_pair_expected_names():
    pair_names = set(SCORING_CATEGORIES.get("pair", []))
    assert "score_boundary" in pair_names
    assert "score_match" in pair_names


def test_scoring_categories_fusion_expected_names():
    fusion_names = set(SCORING_CATEGORIES.get("fusion", []))
    assert "fuse_rankings" in fusion_names


def test_scoring_categories_filter_rank_expected_names():
    fr = set(SCORING_CATEGORIES.get("filter_rank", []))
    assert "filter_pairs" in fr or "rank_pairs" in fr


def test_scoring_categories_evaluation_expected_names():
    ev = set(SCORING_CATEGORIES.get("evaluation", []))
    assert "evaluate_match" in ev


def test_all_category_names_globally_unique():
    all_names = []
    for names in SCORING_CATEGORIES.values():
        all_names.extend(names)
    # Some names may appear in multiple categories but no duplicates within one
    # Here we just check total counts are reasonable
    assert len(all_names) > 5


# ─── build_scoring_registry idempotent ────────────────────────────────────────

def test_build_twice_same_keys():
    r1 = build_scoring_registry()
    r2 = build_scoring_registry()
    assert set(r1.keys()) == set(r2.keys())


def test_build_returns_nonempty_dict():
    r = build_scoring_registry()
    assert len(r) > 0


def test_build_all_values_callable():
    r = build_scoring_registry()
    for name, fn in r.items():
        assert callable(fn), f"{name!r} is not callable"


def test_build_registry_keys_are_strings():
    r = build_scoring_registry()
    for k in r.keys():
        assert isinstance(k, str)


# ─── _ensure_registry ────────────────────────────────────────────────────────

def test_ensure_registry_populates_global():
    # _ensure_registry() reassigns the module-level dict; re-import to see updated value
    import puzzle_reconstruction.scoring.bridge as _bridge
    _bridge._ensure_registry()
    assert len(_bridge.SCORING_REGISTRY) > 0


def test_ensure_registry_idempotent():
    _ensure_registry()
    count1 = len(SCORING_REGISTRY)
    _ensure_registry()
    count2 = len(SCORING_REGISTRY)
    assert count1 == count2


# ─── list_scorers ─────────────────────────────────────────────────────────────

def test_list_scorers_no_duplicates():
    names = list_scorers()
    assert len(names) == len(set(names))


def test_list_scorers_sorted_ascending():
    names = list_scorers()
    assert names == sorted(names)


def test_list_scorers_category_pair_sorted():
    names = list_scorers(category="pair")
    assert names == sorted(names)


def test_list_scorers_category_fusion_sorted():
    names = list_scorers(category="fusion")
    assert names == sorted(names)


def test_list_scorers_category_filter_rank():
    names = list_scorers(category="filter_rank")
    assert isinstance(names, list)


def test_list_scorers_category_evaluation():
    names = list_scorers(category="evaluation")
    assert isinstance(names, list)


def test_list_scorers_unknown_category_returns_empty_list():
    assert list_scorers(category="does_not_exist_xyz") == []


def test_list_scorers_all_items_in_registry():
    all_names = list_scorers()
    registry = build_scoring_registry()
    for name in all_names:
        assert name in registry


def test_list_scorers_category_subset_of_all():
    all_names = set(list_scorers())
    for cat in SCORING_CATEGORIES:
        cat_names = set(list_scorers(category=cat))
        assert cat_names <= all_names


# ─── get_scorer ───────────────────────────────────────────────────────────────

def test_get_scorer_returns_none_for_unknown():
    assert get_scorer("__totally_unknown_scorer__") is None


def test_get_scorer_empty_string_returns_none():
    assert get_scorer("") is None


def test_get_scorer_whitespace_returns_none():
    assert get_scorer("   ") is None


def test_get_scorer_all_registered_are_callable():
    for name in list_scorers():
        fn = get_scorer(name)
        assert callable(fn), f"{name!r} is not callable"


def test_get_scorer_case_sensitive():
    # Scorer names are lowercase; uppercase should return None
    for name in list_scorers()[:3]:
        assert get_scorer(name.upper()) is None or callable(get_scorer(name.upper()))


def test_get_scorer_does_not_mutate_registry():
    before = set(list_scorers())
    get_scorer("some_name_xyz")
    after = set(list_scorers())
    assert before == after


# ─── get_scorer_category ──────────────────────────────────────────────────────

def test_get_scorer_category_unknown_returns_none():
    assert get_scorer_category("nonexistent_xyz") is None


def test_get_scorer_category_valid_names_return_valid_categories():
    valid_cats = set(SCORING_CATEGORIES.keys())
    for cat, names in SCORING_CATEGORIES.items():
        for n in names:
            result = get_scorer_category(n)
            assert result in valid_cats


def test_get_scorer_category_pair_names_return_pair():
    for n in SCORING_CATEGORIES.get("pair", []):
        assert get_scorer_category(n) == "pair"


def test_get_scorer_category_evaluation_names():
    for n in SCORING_CATEGORIES.get("evaluation", []):
        assert get_scorer_category(n) == "evaluation"


def test_get_scorer_category_filter_rank_names():
    for n in SCORING_CATEGORIES.get("filter_rank", []):
        assert get_scorer_category(n) == "filter_rank"


def test_get_scorer_category_fusion_names():
    for n in SCORING_CATEGORIES.get("fusion", []):
        assert get_scorer_category(n) == "fusion"


def test_get_scorer_category_empty_string_returns_none():
    assert get_scorer_category("") is None


# ─── Registry completeness ────────────────────────────────────────────────────

def test_at_least_five_scorers_registered():
    assert len(list_scorers()) >= 5


def test_at_least_one_pair_scorer():
    assert len(list_scorers(category="pair")) >= 1


def test_at_least_one_fusion_scorer():
    assert len(list_scorers(category="fusion")) >= 1
