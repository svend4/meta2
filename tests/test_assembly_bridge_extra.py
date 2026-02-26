"""Extra tests for puzzle_reconstruction/assembly/bridge.py"""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.bridge import (
    ASSEMBLY_CATEGORIES,
    build_assembly_registry,
    get_assembly_category,
    get_assembly_fn,
    list_assembly_fns,
)


# ── ASSEMBLY_CATEGORIES structure ────────────────────────────────────────────

class TestAssemblyCategoriesStructure:

    def test_eight_expected_categories(self):
        expected = {"state", "filter", "geometry", "cost",
                    "layout", "scoring", "sequencing", "tracking"}
        assert set(ASSEMBLY_CATEGORIES.keys()) == expected

    def test_state_has_expected_members(self):
        state = ASSEMBLY_CATEGORIES["state"]
        for name in ("create_state", "place_fragment", "make_empty_canvas"):
            assert name in state

    def test_filter_has_expected_members(self):
        fil = ASSEMBLY_CATEGORIES["filter"]
        for name in ("filter_by_threshold", "filter_top_k", "filter_by_rank"):
            assert name in fil

    def test_geometry_has_expected_members(self):
        geo = ASSEMBLY_CATEGORIES["geometry"]
        for name in ("aabb_overlap", "detect_collisions", "analyze_all_gaps"):
            assert name in geo

    def test_scoring_has_expected_members(self):
        scoring = ASSEMBLY_CATEGORIES["scoring"]
        for name in ("score_fragment", "score_assembly", "find_best_next"):
            assert name in scoring

    def test_no_duplicate_names_per_category(self):
        for cat, names in ASSEMBLY_CATEGORIES.items():
            assert len(names) == len(set(names)), f"Duplicates in {cat}"

    def test_all_names_are_nonempty_strings(self):
        for cat, names in ASSEMBLY_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str) and len(n) > 0

    def test_names_use_underscores(self):
        for cat, names in ASSEMBLY_CATEGORIES.items():
            for n in names:
                assert "-" not in n, f"{n!r} has dash"

    def test_sequencing_category(self):
        seq = ASSEMBLY_CATEGORIES["sequencing"]
        for name in ("sequence_greedy", "sort_by_score", "build_placement_plan"):
            assert name in seq

    def test_tracking_category(self):
        tr = ASSEMBLY_CATEGORIES["tracking"]
        for name in ("create_tracker", "record_snapshot", "detect_convergence"):
            assert name in tr


# ── build_assembly_registry content ──────────────────────────────────────────

class TestBuildAssemblyRegistryContent:

    def test_all_keys_unique(self):
        reg = build_assembly_registry()
        keys = list(reg.keys())
        assert len(keys) == len(set(keys))

    def test_values_not_none(self):
        reg = build_assembly_registry()
        for name, fn in reg.items():
            assert fn is not None, f"None fn for {name!r}"

    def test_values_have_call(self):
        reg = build_assembly_registry()
        for name, fn in reg.items():
            assert hasattr(fn, "__call__"), f"{name!r} not callable"

    def test_all_registered_names_in_some_category(self):
        reg = build_assembly_registry()
        all_cat_names = {n for names in ASSEMBLY_CATEGORIES.values() for n in names}
        for name in reg:
            assert name in all_cat_names, f"{name!r} not in any category"

    def test_second_call_same_keys(self):
        r1 = build_assembly_registry()
        r2 = build_assembly_registry()
        assert set(r1.keys()) == set(r2.keys())

    def test_at_least_15_functions(self):
        reg = build_assembly_registry()
        assert len(reg) >= 15


# ── list_assembly_fns filtering ───────────────────────────────────────────────

class TestListAssemblyFnsFiltering:

    def test_state_filter_subset_of_all(self):
        all_names = set(list_assembly_fns())
        state_names = set(list_assembly_fns(category="state"))
        assert state_names.issubset(all_names)

    def test_scoring_filter_subset_of_all(self):
        all_names = set(list_assembly_fns())
        scoring_names = set(list_assembly_fns(category="scoring"))
        assert scoring_names.issubset(all_names)

    def test_geometry_filter_subset_of_all(self):
        all_names = set(list_assembly_fns())
        geo_names = set(list_assembly_fns(category="geometry"))
        assert geo_names.issubset(all_names)

    def test_tracking_filter_result_sorted(self):
        names = list_assembly_fns(category="tracking")
        assert names == sorted(names)

    def test_cost_filter(self):
        cost_names = list_assembly_fns(category="cost")
        assert isinstance(cost_names, list)

    def test_layout_filter(self):
        layout_names = list_assembly_fns(category="layout")
        assert isinstance(layout_names, list)

    def test_disjoint_categories(self):
        state = set(list_assembly_fns(category="state"))
        filter_ = set(list_assembly_fns(category="filter"))
        geometry = set(list_assembly_fns(category="geometry"))
        assert state & filter_ == set()
        assert state & geometry == set()
        assert filter_ & geometry == set()

    def test_union_of_all_categories_equals_all(self):
        all_names = set(list_assembly_fns())
        union = set()
        for cat in ASSEMBLY_CATEGORIES:
            union |= set(list_assembly_fns(category=cat))
        assert union == all_names


# ── get_assembly_fn behaviour ─────────────────────────────────────────────────

class TestGetAssemblyFnBehaviour:

    def test_returns_none_for_unknown(self):
        assert get_assembly_fn("__not_a_real_function__") is None

    def test_empty_string_returns_none(self):
        assert get_assembly_fn("") is None

    def test_whitespace_returns_none(self):
        assert get_assembly_fn("   ") is None

    def test_case_sensitive(self):
        reg = build_assembly_registry()
        for name in list(reg.keys())[:3]:
            assert get_assembly_fn(name.upper()) is None

    def test_known_fns_return_same_object(self):
        for name in list_assembly_fns():
            fn1 = get_assembly_fn(name)
            fn2 = get_assembly_fn(name)
            assert fn1 is fn2

    def test_all_filter_category_fns(self):
        for name in ASSEMBLY_CATEGORIES.get("filter", []):
            fn = get_assembly_fn(name)
            if fn is not None:
                assert callable(fn)

    def test_all_geometry_category_fns(self):
        for name in ASSEMBLY_CATEGORIES.get("geometry", []):
            fn = get_assembly_fn(name)
            if fn is not None:
                assert callable(fn)

    def test_all_cost_category_fns(self):
        for name in ASSEMBLY_CATEGORIES.get("cost", []):
            fn = get_assembly_fn(name)
            if fn is not None:
                assert callable(fn)


# ── get_assembly_category ─────────────────────────────────────────────────────

class TestGetAssemblyCategory:

    def test_create_state_is_state(self):
        assert get_assembly_category("create_state") == "state"

    def test_filter_by_threshold_is_filter(self):
        assert get_assembly_category("filter_by_threshold") == "filter"

    def test_aabb_overlap_is_geometry(self):
        assert get_assembly_category("aabb_overlap") == "geometry"

    def test_score_fragment_is_scoring(self):
        assert get_assembly_category("score_fragment") == "scoring"

    def test_sequence_greedy_is_sequencing(self):
        assert get_assembly_category("sequence_greedy") == "sequencing"

    def test_create_tracker_is_tracking(self):
        assert get_assembly_category("create_tracker") == "tracking"

    def test_unknown_returns_none(self):
        assert get_assembly_category("__nonexistent__") is None

    def test_empty_string_returns_none(self):
        assert get_assembly_category("") is None

    def test_all_registered_have_category(self):
        for name in list_assembly_fns():
            cat = get_assembly_category(name)
            assert cat in ASSEMBLY_CATEGORIES, f"{name!r} has bad cat {cat!r}"
