"""Extra tests for puzzle_reconstruction/algorithms/bridge.py"""
from __future__ import annotations

import pytest

from puzzle_reconstruction.algorithms.bridge import (
    ALGORITHM_CATEGORIES,
    build_algorithm_registry,
    get_algorithm,
    get_category,
    list_algorithms,
)


# ── ALGORITHM_CATEGORIES structure ──────────────────────────────────────────

class TestAlgorithmCategoriesStructure:

    def test_three_exact_categories(self):
        assert set(ALGORITHM_CATEGORIES.keys()) == {"fragment", "pair", "assembly"}

    def test_assembly_category_not_empty(self):
        assert len(ALGORITHM_CATEGORIES["assembly"]) > 0

    def test_no_duplicate_names_within_fragment(self):
        names = ALGORITHM_CATEGORIES["fragment"]
        assert len(names) == len(set(names))

    def test_no_duplicate_names_within_pair(self):
        names = ALGORITHM_CATEGORIES["pair"]
        assert len(names) == len(set(names))

    def test_no_duplicate_names_within_assembly(self):
        names = ALGORITHM_CATEGORIES["assembly"]
        assert len(names) == len(set(names))

    def test_fragment_count_gte_10(self):
        assert len(ALGORITHM_CATEGORIES["fragment"]) >= 10

    def test_pair_count_gte_5(self):
        assert len(ALGORITHM_CATEGORIES["pair"]) >= 5

    def test_all_names_non_empty_strings(self):
        for cat, names in ALGORITHM_CATEGORIES.items():
            for n in names:
                assert n.strip() != "", f"Empty name in category {cat}"

    def test_names_use_underscores_not_dashes(self):
        for cat, names in ALGORITHM_CATEGORIES.items():
            for n in names:
                assert "-" not in n, f"Name {n!r} uses dash, expected underscore"

    def test_known_fragment_algorithms_present(self):
        fragment_names = ALGORITHM_CATEGORIES["fragment"]
        for expected in ("boundary_descriptor", "fragment_classifier", "gradient_flow"):
            assert expected in fragment_names

    def test_known_pair_algorithms_present(self):
        pair_names = ALGORITHM_CATEGORIES["pair"]
        for expected in ("edge_comparator", "edge_scorer", "seam_evaluator"):
            assert expected in pair_names

    def test_known_assembly_algorithms_present(self):
        assembly_names = ALGORITHM_CATEGORIES["assembly"]
        for expected in ("path_planner", "position_estimator"):
            assert expected in assembly_names


# ── build_algorithm_registry uniqueness and content ────────────────────────

class TestBuildAlgorithmRegistryContent:

    def test_all_keys_unique(self):
        reg = build_algorithm_registry()
        keys = list(reg.keys())
        assert len(keys) == len(set(keys))

    def test_multiple_calls_same_length(self):
        r1 = build_algorithm_registry()
        r2 = build_algorithm_registry()
        assert len(r1) == len(r2)

    def test_no_none_values(self):
        reg = build_algorithm_registry()
        for name, fn in reg.items():
            assert fn is not None, f"None callable for {name!r}"

    def test_all_callables_are_truly_callable(self):
        reg = build_algorithm_registry()
        for name, fn in reg.items():
            assert hasattr(fn, "__call__"), f"{name!r} has no __call__"

    def test_registered_names_in_some_category(self):
        """Every registered algorithm should belong to at least one category."""
        reg = build_algorithm_registry()
        all_category_names = set(
            n for names in ALGORITHM_CATEGORIES.values() for n in names
        )
        for name in reg:
            assert name in all_category_names, f"{name!r} not in any category"


# ── list_algorithms category filtering ─────────────────────────────────────

class TestListAlgorithmsCategoryFiltering:

    def test_filter_fragment_subset_of_all(self):
        all_names = set(list_algorithms())
        fragment_names = set(list_algorithms(category="fragment"))
        assert fragment_names.issubset(all_names)

    def test_filter_pair_subset_of_all(self):
        all_names = set(list_algorithms())
        pair_names = set(list_algorithms(category="pair"))
        assert pair_names.issubset(all_names)

    def test_filter_assembly_subset_of_all(self):
        all_names = set(list_algorithms())
        assembly_names = set(list_algorithms(category="assembly"))
        assert assembly_names.issubset(all_names)

    def test_categories_are_disjoint_in_result(self):
        """Names filtered by different categories should be disjoint."""
        fragment = set(list_algorithms(category="fragment"))
        pair = set(list_algorithms(category="pair"))
        assembly = set(list_algorithms(category="assembly"))
        assert fragment & pair == set()
        assert fragment & assembly == set()
        assert pair & assembly == set()

    def test_union_of_categories_equals_all(self):
        all_names = set(list_algorithms())
        union = (
            set(list_algorithms(category="fragment"))
            | set(list_algorithms(category="pair"))
            | set(list_algorithms(category="assembly"))
        )
        assert union == all_names

    def test_each_result_element_is_string(self):
        for name in list_algorithms(category="fragment"):
            assert isinstance(name, str)

    def test_filter_result_is_sorted(self):
        names = list_algorithms(category="pair")
        assert names == sorted(names)


# ── get_algorithm behaviour ─────────────────────────────────────────────────

class TestGetAlgorithmBehaviour:

    def test_case_sensitive_lookup(self):
        """Lookup is case-sensitive; uppercase should return None."""
        reg = build_algorithm_registry()
        for name in list(reg.keys())[:3]:
            assert get_algorithm(name.upper()) is None

    def test_whitespace_name_returns_none(self):
        assert get_algorithm("   ") is None

    def test_returns_same_object_on_repeated_calls(self):
        name = list_algorithms()[0]
        fn1 = get_algorithm(name)
        fn2 = get_algorithm(name)
        assert fn1 is fn2

    def test_each_fragment_alg_callable_or_none(self):
        for name in ALGORITHM_CATEGORIES["fragment"]:
            fn = get_algorithm(name)
            if fn is not None:
                assert callable(fn)

    def test_each_pair_alg_callable_or_none(self):
        for name in ALGORITHM_CATEGORIES["pair"]:
            fn = get_algorithm(name)
            if fn is not None:
                assert callable(fn)

    def test_each_assembly_alg_callable_or_none(self):
        for name in ALGORITHM_CATEGORIES["assembly"]:
            fn = get_algorithm(name)
            if fn is not None:
                assert callable(fn)


# ── get_category ─────────────────────────────────────────────────────────────

class TestGetCategory:

    def test_fragment_algorithm_category(self):
        assert get_category("boundary_descriptor") == "fragment"

    def test_pair_algorithm_category(self):
        assert get_category("edge_comparator") == "pair"

    def test_assembly_algorithm_category(self):
        assert get_category("path_planner") == "assembly"

    def test_unknown_name_returns_none(self):
        assert get_category("__no_such_algorithm__") is None

    def test_empty_string_returns_none(self):
        assert get_category("") is None

    def test_each_registered_name_has_category(self):
        for name in list_algorithms():
            cat = get_category(name)
            assert cat in ("fragment", "pair", "assembly"), (
                f"{name!r} has unexpected category {cat!r}"
            )

    def test_all_fragment_names_get_fragment_category(self):
        for name in ALGORITHM_CATEGORIES["fragment"]:
            cat = get_category(name)
            assert cat == "fragment", f"{name!r} → {cat!r}"

    def test_all_pair_names_get_pair_category(self):
        for name in ALGORITHM_CATEGORIES["pair"]:
            cat = get_category(name)
            assert cat == "pair", f"{name!r} → {cat!r}"

    def test_all_assembly_names_get_assembly_category(self):
        for name in ALGORITHM_CATEGORIES["assembly"]:
            cat = get_category(name)
            assert cat == "assembly", f"{name!r} → {cat!r}"
