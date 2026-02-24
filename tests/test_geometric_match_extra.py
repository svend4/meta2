"""Extra tests for puzzle_reconstruction/matching/geometric_match.py."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from puzzle_reconstruction.matching.geometric_match import (
    FragmentGeometry,
    GeometricMatchResult,
    compute_fragment_geometry,
    aspect_ratio_similarity,
    area_ratio_similarity,
    hu_moments_similarity,
    edge_length_similarity,
    match_geometry,
    batch_geometry_match,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle_mask(h=128, w=128, r=40):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return mask


def _rect_mask(h=128, w=128, rh=60, rw=80):
    mask = np.zeros((h, w), dtype=np.uint8)
    y = (h - rh) // 2
    x = (w - rw) // 2
    mask[y:y + rh, x:x + rw] = 255
    return mask


def _empty_mask(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _make_geom(area=1000.0, perim=120.0, ar=1.5, hull_area=1100.0, solidity=0.9):
    return FragmentGeometry(
        area=area, perimeter=perim, aspect_ratio=ar,
        hull_area=hull_area, solidity=solidity,
        hu_moments=np.zeros(7, dtype=np.float64),
        bbox=(10, 10, 60, 40),
    )


def _geom_with_hu(hu):
    return FragmentGeometry(
        area=100.0, perimeter=40.0, aspect_ratio=1.0,
        hull_area=100.0, solidity=1.0,
        hu_moments=np.array(hu, dtype=np.float64),
        bbox=(0, 0, 10, 10),
    )


# ─── FragmentGeometry (extra) ─────────────────────────────────────────────────

class TestFragmentGeometryExtra:
    def test_area_stored(self):
        g = _make_geom(area=2500.0)
        assert g.area == pytest.approx(2500.0)

    def test_perimeter_stored(self):
        g = _make_geom(perim=200.0)
        assert g.perimeter == pytest.approx(200.0)

    def test_aspect_ratio_stored(self):
        g = _make_geom(ar=2.0)
        assert g.aspect_ratio == pytest.approx(2.0)

    def test_hull_area_stored(self):
        g = _make_geom(hull_area=3000.0)
        assert g.hull_area == pytest.approx(3000.0)

    def test_solidity_stored(self):
        g = _make_geom(solidity=0.75)
        assert g.solidity == pytest.approx(0.75)

    def test_hu_moments_shape_7(self):
        g = _make_geom()
        assert g.hu_moments.shape == (7,)

    def test_hu_moments_dtype_float64(self):
        g = _make_geom()
        assert g.hu_moments.dtype == np.float64

    def test_bbox_length_4(self):
        g = _make_geom()
        assert len(g.bbox) == 4

    def test_bbox_values(self):
        g = FragmentGeometry(
            area=0.0, perimeter=0.0, aspect_ratio=1.0,
            hull_area=0.0, solidity=0.0,
            hu_moments=np.zeros(7), bbox=(5, 10, 50, 80),
        )
        assert g.bbox == (5, 10, 50, 80)

    def test_params_default_empty_dict(self):
        g = _make_geom()
        assert g.params == {}

    def test_params_custom_stored(self):
        g = FragmentGeometry(
            area=0.0, perimeter=0.0, aspect_ratio=1.0,
            hull_area=0.0, solidity=0.0,
            hu_moments=np.zeros(7), bbox=(0, 0, 0, 0),
            params={"epsilon_frac": 0.02, "n_contours": 1},
        )
        assert g.params["epsilon_frac"] == pytest.approx(0.02)
        assert g.params["n_contours"] == 1

    def test_zero_area_allowed(self):
        g = _make_geom(area=0.0)
        assert g.area == pytest.approx(0.0)


# ─── GeometricMatchResult (extra) ─────────────────────────────────────────────

class TestGeometricMatchResultExtra:
    def test_score_stored(self):
        r = GeometricMatchResult(score=0.75, aspect_score=0.8,
                                  area_score=0.7, hu_score=0.75)
        assert r.score == pytest.approx(0.75)

    def test_aspect_score_stored(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.9,
                                  area_score=0.5, hu_score=0.5)
        assert r.aspect_score == pytest.approx(0.9)

    def test_area_score_stored(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.65, hu_score=0.5)
        assert r.area_score == pytest.approx(0.65)

    def test_hu_score_stored(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.88)
        assert r.hu_score == pytest.approx(0.88)

    def test_method_default_geometric(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert r.method == "geometric"

    def test_params_default_empty(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert isinstance(r.params, dict)

    def test_params_custom(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5,
                                  params={"w_aspect": 0.3})
        assert r.params["w_aspect"] == pytest.approx(0.3)

    def test_repr_contains_class_name(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert "GeometricMatchResult" in repr(r)

    def test_scores_all_in_range(self):
        r = GeometricMatchResult(score=0.6, aspect_score=0.7,
                                  area_score=0.5, hu_score=0.8)
        for val in [r.score, r.aspect_score, r.area_score, r.hu_score]:
            assert 0.0 <= val <= 1.0


# ─── compute_fragment_geometry (extra) ────────────────────────────────────────

class TestComputeFragmentGeometryExtra:
    def test_returns_fragment_geometry(self):
        g = compute_fragment_geometry(_circle_mask())
        assert isinstance(g, FragmentGeometry)

    def test_area_positive_circle(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.area > 0.0

    def test_area_positive_rect(self):
        g = compute_fragment_geometry(_rect_mask())
        assert g.area > 0.0

    def test_perimeter_positive(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.perimeter > 0.0

    def test_aspect_ratio_gte_one(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.aspect_ratio >= 1.0

    def test_hull_area_gte_contour_area(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.hull_area >= g.area - 1.0

    def test_solidity_in_0_1(self):
        g = compute_fragment_geometry(_circle_mask())
        assert 0.0 < g.solidity <= 1.0

    def test_hu_moments_shape(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.hu_moments.shape == (7,)

    def test_hu_moments_dtype(self):
        g = compute_fragment_geometry(_circle_mask())
        assert g.hu_moments.dtype == np.float64

    def test_bbox_has_positive_dims(self):
        g = compute_fragment_geometry(_circle_mask())
        x, y, w, h = g.bbox
        assert w > 0 and h > 0

    def test_empty_mask_zero_area(self):
        g = compute_fragment_geometry(_empty_mask())
        assert g.area == pytest.approx(0.0)

    def test_empty_mask_no_crash(self):
        g = compute_fragment_geometry(_empty_mask())
        assert g.hu_moments.shape == (7,)

    def test_epsilon_frac_in_params(self):
        g = compute_fragment_geometry(_circle_mask(), epsilon_frac=0.05)
        assert g.params.get("epsilon_frac") == pytest.approx(0.05)

    def test_n_contours_in_params(self):
        g = compute_fragment_geometry(_circle_mask())
        assert "n_contours" in g.params
        assert g.params["n_contours"] >= 1

    def test_rect_aspect_ratio(self):
        g = compute_fragment_geometry(_rect_mask(rh=40, rw=80))
        assert g.aspect_ratio >= 1.5

    def test_square_solidity_near_one(self):
        g = compute_fragment_geometry(_rect_mask(rh=60, rw=60))
        assert g.solidity > 0.9


# ─── aspect_ratio_similarity (extra) ──────────────────────────────────────────

class TestAspectRatioSimilarityExtra:
    def test_same_ar_is_one(self):
        g1 = _make_geom(ar=2.0)
        g2 = _make_geom(ar=2.0)
        assert aspect_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_range_0_1(self):
        g1 = _make_geom(ar=1.0)
        g2 = _make_geom(ar=5.0)
        s = aspect_ratio_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        g1 = _make_geom(ar=1.2)
        g2 = _make_geom(ar=3.0)
        assert aspect_ratio_similarity(g1, g2) == pytest.approx(
            aspect_ratio_similarity(g2, g1)
        )

    def test_returns_float(self):
        g1 = _make_geom(ar=1.5)
        g2 = _make_geom(ar=2.5)
        assert isinstance(aspect_ratio_similarity(g1, g2), float)

    def test_ar_one_vs_two(self):
        g1 = _make_geom(ar=1.0)
        g2 = _make_geom(ar=2.0)
        s = aspect_ratio_similarity(g1, g2)
        assert 0.0 < s < 1.0


# ─── area_ratio_similarity (extra) ────────────────────────────────────────────

class TestAreaRatioSimilarityExtra:
    def test_same_area_is_one(self):
        g = _make_geom(area=2000.0)
        assert area_ratio_similarity(g, g) == pytest.approx(1.0)

    def test_range_0_1(self):
        g1 = _make_geom(area=500.0)
        g2 = _make_geom(area=5000.0)
        assert 0.0 <= area_ratio_similarity(g1, g2) <= 1.0

    def test_half_area_returns_half(self):
        g1 = _make_geom(area=1000.0)
        g2 = _make_geom(area=2000.0)
        assert area_ratio_similarity(g1, g2) == pytest.approx(0.5)

    def test_symmetric(self):
        g1 = _make_geom(area=400.0)
        g2 = _make_geom(area=1600.0)
        assert area_ratio_similarity(g1, g2) == pytest.approx(
            area_ratio_similarity(g2, g1)
        )

    def test_zero_areas_return_one(self):
        g1 = _make_geom(area=0.0)
        g2 = _make_geom(area=0.0)
        assert area_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(area_ratio_similarity(_make_geom(), _make_geom(area=3000.0)), float)

    def test_large_ratio_small_score(self):
        g1 = _make_geom(area=1.0)
        g2 = _make_geom(area=1000.0)
        s = area_ratio_similarity(g1, g2)
        assert s < 0.1


# ─── hu_moments_similarity (extra) ────────────────────────────────────────────

class TestHuMomentsSimilarityExtra:
    def test_identical_returns_one(self):
        g = _geom_with_hu([0.5, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        assert hu_moments_similarity(g, g) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vectors_return_one(self):
        g = _geom_with_hu([0.0] * 7)
        assert hu_moments_similarity(g, g) == pytest.approx(1.0, abs=1e-6)

    def test_range_0_1(self):
        g1 = _geom_with_hu([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        g2 = _geom_with_hu([7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0])
        s = hu_moments_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        g1 = _geom_with_hu([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        g2 = _geom_with_hu([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert hu_moments_similarity(g1, g2) == pytest.approx(
            hu_moments_similarity(g2, g1), abs=1e-6
        )

    def test_returns_float(self):
        g1 = _geom_with_hu([1.0] * 7)
        g2 = _geom_with_hu([2.0] * 7)
        assert isinstance(hu_moments_similarity(g1, g2), float)

    def test_very_different_moments_low_score(self):
        g1 = _geom_with_hu([0.0] * 7)
        g2 = _geom_with_hu([100.0] * 7)
        s = hu_moments_similarity(g1, g2)
        assert s < 0.5


# ─── edge_length_similarity (extra) ───────────────────────────────────────────

class TestEdgeLengthSimilarityExtra:
    def test_equal_is_one(self):
        assert edge_length_similarity(100.0, 100.0) == pytest.approx(1.0)

    def test_zero_both_is_one(self):
        assert edge_length_similarity(0.0, 0.0) == pytest.approx(1.0)

    def test_half_ratio(self):
        assert edge_length_similarity(10.0, 20.0) == pytest.approx(0.5)

    def test_symmetric(self):
        assert edge_length_similarity(30.0, 90.0) == pytest.approx(
            edge_length_similarity(90.0, 30.0)
        )

    def test_range(self):
        s = edge_length_similarity(5.0, 100.0)
        assert 0.0 <= s <= 1.0

    def test_returns_float(self):
        assert isinstance(edge_length_similarity(10.0, 30.0), float)

    def test_large_values(self):
        assert edge_length_similarity(1000.0, 1000.0) == pytest.approx(1.0)

    def test_near_equal_near_one(self):
        s = edge_length_similarity(10.0, 10.1)
        assert s > 0.9


# ─── match_geometry (extra) ───────────────────────────────────────────────────

class TestMatchGeometryExtra:
    def test_returns_geometric_match_result(self):
        g = _make_geom()
        assert isinstance(match_geometry(g, g), GeometricMatchResult)

    def test_method_geometric(self):
        g = _make_geom()
        assert match_geometry(g, g).method == "geometric"

    def test_score_in_range(self):
        g1 = _make_geom()
        g2 = _make_geom(area=5000.0, ar=3.0)
        r = match_geometry(g1, g2)
        assert 0.0 <= r.score <= 1.0

    def test_identical_high_score(self):
        g = compute_fragment_geometry(_circle_mask())
        r = match_geometry(g, g)
        assert r.score > 0.5

    def test_component_scores_in_range(self):
        g1 = compute_fragment_geometry(_circle_mask())
        g2 = compute_fragment_geometry(_rect_mask())
        r = match_geometry(g1, g2)
        for score in [r.aspect_score, r.area_score, r.hu_score]:
            assert 0.0 <= score <= 1.0

    def test_weights_stored_in_params(self):
        g = _make_geom()
        r = match_geometry(g, g, w_aspect=0.1, w_area=0.6, w_hu=0.3)
        assert r.params.get("w_aspect") == pytest.approx(0.1)
        assert r.params.get("w_area") == pytest.approx(0.6)
        assert r.params.get("w_hu") == pytest.approx(0.3)

    def test_edge_len_score_stored_when_provided(self):
        g = _make_geom()
        r = match_geometry(g, g, edge_len1=50.0, edge_len2=50.0)
        assert "edge_len_score" in r.params

    def test_edge_len_score_in_range(self):
        g = _make_geom()
        r = match_geometry(g, g, edge_len1=30.0, edge_len2=60.0)
        assert 0.0 <= r.params["edge_len_score"] <= 1.0

    def test_no_edge_len_no_key(self):
        g = _make_geom()
        r = match_geometry(g, g)
        assert "edge_len_score" not in r.params

    def test_equal_edge_lengths_score_one(self):
        g = _make_geom()
        r = match_geometry(g, g, edge_len1=50.0, edge_len2=50.0)
        assert r.params["edge_len_score"] == pytest.approx(1.0)


# ─── batch_geometry_match (extra) ─────────────────────────────────────────────

class TestBatchGeometryMatchExtra:
    def test_returns_list(self):
        g1 = _make_geom()
        g2 = _make_geom(area=2000.0)
        r = batch_geometry_match([g1, g2], [(0, 1)])
        assert isinstance(r, list)

    def test_length_matches_pairs(self):
        g = _make_geom()
        pairs = [(0, 0), (0, 0), (0, 0)]
        r = batch_geometry_match([g], pairs)
        assert len(r) == 3

    def test_each_is_geometric_match_result(self):
        g = _make_geom()
        for item in batch_geometry_match([g, g], [(0, 1)]):
            assert isinstance(item, GeometricMatchResult)

    def test_empty_pairs(self):
        g = _make_geom()
        assert batch_geometry_match([g], []) == []

    def test_scores_in_range(self):
        geoms = [compute_fragment_geometry(_circle_mask()),
                 compute_fragment_geometry(_rect_mask())]
        pairs = [(0, 0), (0, 1), (1, 1)]
        for r in batch_geometry_match(geoms, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_self_pair_high_score(self):
        g = compute_fragment_geometry(_circle_mask())
        results = batch_geometry_match([g], [(0, 0)])
        assert results[0].score > 0.5

    def test_weights_forwarded(self):
        g = _make_geom()
        results = batch_geometry_match([g], [(0, 0)], w_aspect=0.2, w_area=0.6, w_hu=0.2)
        assert results[0].params.get("w_aspect") == pytest.approx(0.2)

    def test_single_pair(self):
        g1 = _make_geom()
        g2 = _make_geom(area=3000.0)
        r = batch_geometry_match([g1, g2], [(0, 1)])
        assert len(r) == 1
