"""Additional tests for puzzle_reconstruction.utils.bridge (Bridge #6)."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.bridge import (
    UTIL_CATEGORIES,
    build_util_registry,
    get_util,
    get_util_category,
    list_utils,
)

# ─── Expected categories ──────────────────────────────────────────────────────

EXPECTED_CATEGORIES = [
    "core", "geometry", "image", "signal", "metrics",
    "graph", "contour", "color", "keypoint", "fragment",
    "scoring", "assembly", "records", "io", "annealing",
]

# Spot-check: a few known utility names per category
CATEGORY_SPOT_CHECKS = {
    "core":     ["get_logger", "PipelineProfiler", "ResultCache"],
    "geometry": ["rotation_matrix_2d", "BoundingBox", "polygon_area"],
    "image":    ["load_image", "compute_image_stats", "PatchSet"],
    "signal":   ["smooth_signal", "moving_average", "FrequencyConfig"],
    "metrics":  ["ReconstructionMetrics", "euclidean_distance", "kmeans_cluster"],
    "graph":    ["build_graph", "to_sparse_entries", "TopologyConfig"],
    "contour":  ["simplify_contour", "compute_curvature", "curve_l2"],
    "color":    ["to_gray", "compute_1d_histogram", "ColorHistConfig"],
    "keypoint": ["detect_keypoints", "DescriptorConfig"],
    "fragment": ["FragmentMetrics", "FragmentFilterConfig"],
    "scoring":  ["sequence_utils", "voting_utils", "sampling_utils"],
    "assembly": ["assembly_score_utils", "canvas_build_utils"],
    "io":       ["load_image_dir", "ConfigSpec", "normalize_array"],
    "annealing": ["linear_schedule", "AnnealingScoreEntry"],
}


# ─── UTIL_CATEGORIES structure ────────────────────────────────────────────────

class TestUtilCategoriesExtra:
    def test_all_expected_categories_present(self):
        for cat in EXPECTED_CATEGORIES:
            assert cat in UTIL_CATEGORIES, f"Missing category: {cat}"

    def test_no_empty_category(self):
        for cat, names in UTIL_CATEGORIES.items():
            assert len(names) > 0, f"Category '{cat}' is empty"

    def test_total_utilities_count(self):
        total = sum(len(v) for v in UTIL_CATEGORIES.values())
        assert total >= 100

    def test_no_duplicate_names_within_category(self):
        for cat, names in UTIL_CATEGORIES.items():
            assert len(names) == len(set(names)), f"Duplicates in category '{cat}'"

    def test_all_names_nonempty_strings(self):
        for cat, names in UTIL_CATEGORIES.items():
            for n in names:
                assert isinstance(n, str) and len(n) > 0

    def test_core_count(self):
        assert len(UTIL_CATEGORIES["core"]) >= 10

    def test_geometry_count(self):
        assert len(UTIL_CATEGORIES["geometry"]) >= 5

    def test_records_count(self):
        assert len(UTIL_CATEGORIES["records"]) >= 10

    def test_scoring_count(self):
        assert len(UTIL_CATEGORIES["scoring"]) >= 10

    def test_spot_check_core_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["core"]:
            assert name in UTIL_CATEGORIES["core"], f"Missing '{name}' in core"

    def test_spot_check_geometry_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["geometry"]:
            assert name in UTIL_CATEGORIES["geometry"], f"Missing '{name}' in geometry"

    def test_spot_check_image_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["image"]:
            assert name in UTIL_CATEGORIES["image"], f"Missing '{name}' in image"

    def test_spot_check_signal_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["signal"]:
            assert name in UTIL_CATEGORIES["signal"], f"Missing '{name}' in signal"

    def test_spot_check_metrics_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["metrics"]:
            assert name in UTIL_CATEGORIES["metrics"], f"Missing '{name}' in metrics"

    def test_spot_check_io_names_present(self):
        for name in CATEGORY_SPOT_CHECKS["io"]:
            assert name in UTIL_CATEGORIES["io"], f"Missing '{name}' in io"


# ─── build_util_registry ──────────────────────────────────────────────────────

class TestBuildUtilRegistryExtra:
    def test_registry_size_matches_categories_total(self):
        reg = build_util_registry()
        total_in_cats = sum(len(v) for v in UTIL_CATEGORIES.values())
        # Registry size should be close to or equal to total
        assert len(reg) >= total_in_cats * 0.8  # at least 80% available

    def test_all_expected_categories_contribute(self):
        reg = build_util_registry()
        for cat, names in UTIL_CATEGORIES.items():
            available = [n for n in names if n in reg]
            assert len(available) > 0, f"Category '{cat}' has 0 available utilities"

    def test_core_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["core"]:
            assert name in reg, f"core util '{name}' not in registry"

    def test_geometry_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["geometry"]:
            assert name in reg, f"geometry util '{name}' not in registry"

    def test_image_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["image"]:
            assert name in reg, f"image util '{name}' not in registry"

    def test_signal_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["signal"]:
            assert name in reg, f"signal util '{name}' not in registry"

    def test_metrics_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["metrics"]:
            assert name in reg, f"metrics util '{name}' not in registry"

    def test_graph_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["graph"]:
            assert name in reg, f"graph util '{name}' not in registry"

    def test_contour_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["contour"]:
            assert name in reg, f"contour util '{name}' not in registry"

    def test_color_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["color"]:
            assert name in reg, f"color util '{name}' not in registry"

    def test_fragment_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["fragment"]:
            assert name in reg, f"fragment util '{name}' not in registry"

    def test_io_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["io"]:
            assert name in reg, f"io util '{name}' not in registry"

    def test_annealing_utils_all_available(self):
        reg = build_util_registry()
        for name in UTIL_CATEGORIES["annealing"]:
            assert name in reg, f"annealing util '{name}' not in registry"

    def test_all_values_not_none(self):
        reg = build_util_registry()
        for name, val in reg.items():
            assert val is not None, f"Util '{name}' is None"

    def test_get_logger_callable(self):
        reg = build_util_registry()
        fn = reg.get("get_logger")
        assert fn is not None and callable(fn)

    def test_rotation_matrix_2d_callable(self):
        reg = build_util_registry()
        fn = reg.get("rotation_matrix_2d")
        assert fn is not None and callable(fn)

    def test_euclidean_distance_callable(self):
        reg = build_util_registry()
        fn = reg.get("euclidean_distance")
        assert fn is not None and callable(fn)

    def test_build_graph_callable(self):
        reg = build_util_registry()
        fn = reg.get("build_graph")
        assert fn is not None and callable(fn)

    def test_simplify_contour_callable(self):
        reg = build_util_registry()
        fn = reg.get("simplify_contour")
        assert fn is not None and callable(fn)

    def test_to_gray_callable(self):
        reg = build_util_registry()
        fn = reg.get("to_gray")
        assert fn is not None and callable(fn)


# ─── list_utils ───────────────────────────────────────────────────────────────

class TestListUtilsExtra:
    def test_all_categories_return_nonempty_list(self):
        for cat in EXPECTED_CATEGORIES:
            names = list_utils(category=cat)
            assert len(names) > 0, f"list_utils('{cat}') returned empty"

    def test_category_core_sorted(self):
        names = list_utils(category="core")
        assert names == sorted(names)

    def test_category_geometry_sorted(self):
        names = list_utils(category="geometry")
        assert names == sorted(names)

    def test_category_image_contains_expected(self):
        names = list_utils(category="image")
        assert "load_image" in names

    def test_category_metrics_contains_expected(self):
        names = list_utils(category="metrics")
        assert "euclidean_distance" in names

    def test_category_graph_contains_expected(self):
        names = list_utils(category="graph")
        assert "build_graph" in names

    def test_category_contour_contains_expected(self):
        names = list_utils(category="contour")
        assert "simplify_contour" in names

    def test_category_color_contains_expected(self):
        names = list_utils(category="color")
        assert "to_gray" in names

    def test_category_signal_contains_expected(self):
        names = list_utils(category="signal")
        assert "smooth_signal" in names

    def test_all_none_superset_of_category(self):
        all_names = set(list_utils())
        for cat in EXPECTED_CATEGORIES:
            cat_names = set(list_utils(category=cat))
            assert cat_names <= all_names, f"Category '{cat}' has names not in all"

    def test_category_subsets_union_equals_all(self):
        all_names = set(list_utils())
        union = set()
        for cat in EXPECTED_CATEGORIES:
            union |= set(list_utils(category=cat))
        assert union == all_names

    def test_elements_are_strings(self):
        for name in list_utils():
            assert isinstance(name, str)

    def test_no_duplicates_in_all(self):
        names = list_utils()
        assert len(names) == len(set(names))


# ─── get_util ─────────────────────────────────────────────────────────────────

class TestGetUtilExtra:
    def test_get_logger(self):
        fn = get_util("get_logger")
        assert fn is not None

    def test_rotation_matrix_2d(self):
        fn = get_util("rotation_matrix_2d")
        assert fn is not None

    def test_euclidean_distance(self):
        fn = get_util("euclidean_distance")
        assert fn is not None

    def test_build_graph(self):
        fn = get_util("build_graph")
        assert fn is not None

    def test_simplify_contour(self):
        fn = get_util("simplify_contour")
        assert fn is not None

    def test_to_gray(self):
        fn = get_util("to_gray")
        assert fn is not None

    def test_smooth_signal(self):
        fn = get_util("smooth_signal")
        assert fn is not None

    def test_load_image(self):
        fn = get_util("load_image")
        assert fn is not None

    def test_detect_keypoints(self):
        fn = get_util("detect_keypoints")
        assert fn is not None

    def test_all_registered_utils_accessible(self):
        all_names = list_utils()
        for name in all_names:
            obj = get_util(name)
            assert obj is not None, f"get_util('{name}') returned None"

    def test_missing_name_returns_none(self):
        assert get_util("__missing_util_xyz__") is None

    def test_empty_name_returns_none(self):
        assert get_util("") is None


# ─── get_util_category ────────────────────────────────────────────────────────

class TestGetUtilCategoryExtra:
    def test_all_categories_correct(self):
        for cat, names in UTIL_CATEGORIES.items():
            for name in names:
                result = get_util_category(name)
                assert result == cat, (
                    f"get_util_category('{name}') returned '{result}', expected '{cat}'"
                )

    def test_unknown_returns_none(self):
        assert get_util_category("__totally_unknown_xyz__") is None

    def test_core_names_return_core(self):
        for name in UTIL_CATEGORIES["core"]:
            assert get_util_category(name) == "core"

    def test_geometry_names_return_geometry(self):
        for name in UTIL_CATEGORIES["geometry"]:
            assert get_util_category(name) == "geometry"

    def test_image_names_return_image(self):
        for name in UTIL_CATEGORIES["image"]:
            assert get_util_category(name) == "image"

    def test_signal_names_return_signal(self):
        for name in UTIL_CATEGORIES["signal"]:
            assert get_util_category(name) == "signal"

    def test_metrics_names_return_metrics(self):
        for name in UTIL_CATEGORIES["metrics"]:
            assert get_util_category(name) == "metrics"

    def test_graph_names_return_graph(self):
        for name in UTIL_CATEGORIES["graph"]:
            assert get_util_category(name) == "graph"

    def test_contour_names_return_contour(self):
        for name in UTIL_CATEGORIES["contour"]:
            assert get_util_category(name) == "contour"

    def test_color_names_return_color(self):
        for name in UTIL_CATEGORIES["color"]:
            assert get_util_category(name) == "color"

    def test_io_names_return_io(self):
        for name in UTIL_CATEGORIES["io"]:
            assert get_util_category(name) == "io"

    def test_annealing_names_return_annealing(self):
        for name in UTIL_CATEGORIES["annealing"]:
            assert get_util_category(name) == "annealing"

    def test_returns_string_for_known(self):
        result = get_util_category("get_logger")
        assert isinstance(result, str)

    def test_result_is_valid_category(self):
        result = get_util_category("euclidean_distance")
        assert result in UTIL_CATEGORIES
