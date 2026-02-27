"""Extra tests for puzzle_reconstruction/matching/bridge.py"""
from __future__ import annotations

import pytest

from puzzle_reconstruction.matching.bridge import (
    MATCHER_CATEGORIES,
    build_matcher_registry,
    get_matcher,
    get_matcher_category,
    list_matchers,
)


# ── MATCHER_CATEGORIES structure ─────────────────────────────────────────────

class TestMatcherCategoriesStructure:

    def test_exact_category_keys(self):
        expected = {"distance", "geometric", "appearance", "patch",
                    "ranking", "graph", "registry"}
        assert set(MATCHER_CATEGORIES.keys()) == expected

    def test_distance_category_members(self):
        dist = MATCHER_CATEGORIES["distance"]
        for name in ("dtw_distance", "compute_fourier_descriptor",
                     "compute_seam_score", "extract_edge_sample"):
            assert name in dist

    def test_geometric_category_members(self):
        geo = MATCHER_CATEGORIES["geometric"]
        for name in ("estimate_affine", "icp_align", "hu_moments"):
            assert name in geo

    def test_appearance_category_members(self):
        app = MATCHER_CATEGORIES["appearance"]
        for name in ("compute_color_histogram", "extract_features",
                     "compute_lbp_histogram"):
            assert name in app

    def test_ranking_category_members(self):
        rank = MATCHER_CATEGORIES["ranking"]
        for name in ("rank_pairs", "score_pair", "global_match"):
            assert name in rank

    def test_graph_category_members(self):
        graph = MATCHER_CATEGORIES["graph"]
        assert "build_fragment_graph" in graph

    def test_registry_category_members(self):
        reg = MATCHER_CATEGORIES["registry"]
        for name in ("register_matcher", "get_matcher_fn"):
            assert name in reg

    def test_no_duplicates_within_category(self):
        for cat, names in MATCHER_CATEGORIES.items():
            assert len(names) == len(set(names)), f"Duplicates in {cat}"

    def test_all_names_nonempty_strings(self):
        for cat, names in MATCHER_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str) and len(n) > 0

    def test_names_use_underscores_not_dashes(self):
        for cat, names in MATCHER_CATEGORIES.items():
            for n in names:
                assert "-" not in n, f"{n!r} uses dash"


# ── build_matcher_registry content ───────────────────────────────────────────

class TestBuildMatcherRegistryContent:

    def test_keys_are_strings(self):
        reg = build_matcher_registry()
        for k in reg:
            assert isinstance(k, str)

    def test_values_are_callable(self):
        reg = build_matcher_registry()
        for name, fn in reg.items():
            assert callable(fn), f"{name!r} is not callable"

    def test_no_none_values(self):
        reg = build_matcher_registry()
        for name, fn in reg.items():
            assert fn is not None

    def test_all_unique_keys(self):
        reg = build_matcher_registry()
        keys = list(reg.keys())
        assert len(keys) == len(set(keys))

    def test_all_registered_names_in_some_category(self):
        reg = build_matcher_registry()
        all_cat_names = {n for names in MATCHER_CATEGORIES.values() for n in names}
        for name in reg:
            assert name in all_cat_names, f"{name!r} not in any category"

    def test_repeated_calls_same_result(self):
        r1 = build_matcher_registry()
        r2 = build_matcher_registry()
        assert set(r1.keys()) == set(r2.keys())

    def test_at_least_5_matchers(self):
        reg = build_matcher_registry()
        assert len(reg) >= 5


# ── list_matchers filtering extra ─────────────────────────────────────────────

class TestListMatchersFilteringExtra:

    def test_distance_filter_subset_of_all(self):
        all_names = set(list_matchers())
        dist_names = set(list_matchers(category="distance"))
        assert dist_names.issubset(all_names)

    def test_geometric_filter_subset_of_all(self):
        all_names = set(list_matchers())
        geo_names = set(list_matchers(category="geometric"))
        assert geo_names.issubset(all_names)

    def test_appearance_filter_subset_of_all(self):
        all_names = set(list_matchers())
        app_names = set(list_matchers(category="appearance"))
        assert app_names.issubset(all_names)

    def test_result_sorted_for_each_category(self):
        for cat in MATCHER_CATEGORIES:
            names = list_matchers(category=cat)
            assert names == sorted(names), f"Not sorted for category {cat!r}"

    def test_patch_filter(self):
        patch_names = list_matchers(category="patch")
        assert isinstance(patch_names, list)

    def test_ranking_filter(self):
        rank_names = list_matchers(category="ranking")
        assert isinstance(rank_names, list)

    def test_graph_filter(self):
        graph_names = list_matchers(category="graph")
        assert isinstance(graph_names, list)

    def test_registry_filter(self):
        reg_names = list_matchers(category="registry")
        assert isinstance(reg_names, list)

    def test_union_of_categories_equals_all(self):
        all_names = set(list_matchers())
        union = set()
        for cat in MATCHER_CATEGORIES:
            union |= set(list_matchers(category=cat))
        assert union == all_names

    def test_categories_pairwise_disjoint(self):
        category_results = {cat: set(list_matchers(category=cat))
                            for cat in MATCHER_CATEGORIES}
        cats = list(MATCHER_CATEGORIES.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                intersection = category_results[cats[i]] & category_results[cats[j]]
                assert intersection == set(), (
                    f"Overlap between {cats[i]} and {cats[j]}: {intersection}"
                )


# ── get_matcher behaviour extra ───────────────────────────────────────────────

class TestGetMatcherBehaviourExtra:

    def test_returns_none_for_unknown_name(self):
        assert get_matcher("__no_such_matcher__") is None

    def test_empty_string_returns_none(self):
        assert get_matcher("") is None

    def test_whitespace_returns_none(self):
        assert get_matcher("  ") is None

    def test_case_sensitive_lookup(self):
        reg = build_matcher_registry()
        for name in list(reg.keys())[:3]:
            assert get_matcher(name.upper()) is None

    def test_same_object_on_repeated_calls(self):
        for name in list_matchers():
            fn1 = get_matcher(name)
            fn2 = get_matcher(name)
            assert fn1 is fn2

    def test_all_distance_category_callable_or_none(self):
        for name in MATCHER_CATEGORIES.get("distance", []):
            fn = get_matcher(name)
            if fn is not None:
                assert callable(fn)

    def test_all_ranking_category_callable_or_none(self):
        for name in MATCHER_CATEGORIES.get("ranking", []):
            fn = get_matcher(name)
            if fn is not None:
                assert callable(fn)

    def test_all_graph_category_callable_or_none(self):
        for name in MATCHER_CATEGORIES.get("graph", []):
            fn = get_matcher(name)
            if fn is not None:
                assert callable(fn)

    def test_all_registry_category_callable_or_none(self):
        for name in MATCHER_CATEGORIES.get("registry", []):
            fn = get_matcher(name)
            if fn is not None:
                assert callable(fn)


# ── get_matcher_category extra ────────────────────────────────────────────────

class TestGetMatcherCategoryExtra:

    def test_dtw_distance_is_distance(self):
        assert get_matcher_category("dtw_distance") == "distance"

    def test_estimate_affine_is_geometric(self):
        assert get_matcher_category("estimate_affine") == "geometric"

    def test_icp_align_is_geometric(self):
        assert get_matcher_category("icp_align") == "geometric"

    def test_hu_moments_is_geometric(self):
        assert get_matcher_category("hu_moments") == "geometric"

    def test_rank_pairs_is_ranking(self):
        assert get_matcher_category("rank_pairs") == "ranking"

    def test_score_pair_is_ranking(self):
        assert get_matcher_category("score_pair") == "ranking"

    def test_global_match_is_ranking(self):
        assert get_matcher_category("global_match") == "ranking"

    def test_build_fragment_graph_is_graph(self):
        assert get_matcher_category("build_fragment_graph") == "graph"

    def test_register_matcher_is_registry(self):
        assert get_matcher_category("register_matcher") == "registry"

    def test_unknown_returns_none(self):
        assert get_matcher_category("__nonexistent__") is None

    def test_empty_returns_none(self):
        assert get_matcher_category("") is None

    def test_all_registered_matchers_have_valid_category(self):
        for name in list_matchers():
            cat = get_matcher_category(name)
            assert cat in MATCHER_CATEGORIES, (
                f"{name!r} has bad category {cat!r}"
            )

    def test_compute_fourier_descriptor_is_distance(self):
        assert get_matcher_category("compute_fourier_descriptor") == "distance"

    def test_compute_seam_score_is_distance(self):
        assert get_matcher_category("compute_seam_score") == "distance"

    def test_extract_features_is_appearance(self):
        assert get_matcher_category("extract_features") == "appearance"
