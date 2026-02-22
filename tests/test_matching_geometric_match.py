"""Тесты для puzzle_reconstruction.matching.geometric_match."""
import pytest
import numpy as np
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

def _blank_mask(h=64, w=64) -> np.ndarray:
    """All-black: no contours."""
    return np.zeros((h, w), dtype=np.uint8)


def _rect_mask(h=64, w=64, top=10, left=10, bottom=50, right=50) -> np.ndarray:
    """White rectangle on black background."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def _square_mask() -> np.ndarray:
    return _rect_mask(64, 64, 10, 10, 50, 50)


def _wide_mask() -> np.ndarray:
    """Wide rectangle (AR > 1)."""
    return _rect_mask(64, 128, 10, 10, 30, 100)


def _rgb_mask() -> np.ndarray:
    """BGR version of a square mask."""
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    return mask


def _geom(area=1000.0, perimeter=120.0, aspect_ratio=1.0) -> FragmentGeometry:
    hu = np.zeros(7, dtype=np.float64)
    return FragmentGeometry(
        area=area, perimeter=perimeter, aspect_ratio=aspect_ratio,
        hull_area=area, solidity=1.0, hu_moments=hu,
        bbox=(0, 0, 32, 32),
    )


# ─── TestFragmentGeometry ─────────────────────────────────────────────────────

class TestFragmentGeometry:
    def test_basic_fields(self):
        g = _geom()
        assert g.area == pytest.approx(1000.0)
        assert g.perimeter == pytest.approx(120.0)
        assert g.aspect_ratio == pytest.approx(1.0)

    def test_hu_moments_shape(self):
        g = _geom()
        assert g.hu_moments.shape == (7,)

    def test_hu_moments_dtype(self):
        g = _geom()
        assert g.hu_moments.dtype == np.float64

    def test_bbox_stored(self):
        g = _geom()
        assert g.bbox == (0, 0, 32, 32)

    def test_solidity_stored(self):
        g = _geom()
        assert g.solidity == pytest.approx(1.0)

    def test_params_default_empty(self):
        g = _geom()
        assert g.params == {}

    def test_params_stored(self):
        hu = np.zeros(7, dtype=np.float64)
        g = FragmentGeometry(
            area=100.0, perimeter=40.0, aspect_ratio=1.0,
            hull_area=100.0, solidity=1.0, hu_moments=hu,
            bbox=(0, 0, 10, 10), params={"n_contours": 1},
        )
        assert g.params["n_contours"] == 1


# ─── TestGeometricMatchResult ─────────────────────────────────────────────────

class TestGeometricMatchResult:
    def test_basic_fields(self):
        r = GeometricMatchResult(score=0.8, aspect_score=0.9,
                                  area_score=0.7, hu_score=0.8)
        assert r.score == pytest.approx(0.8)

    def test_method_geometric(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert r.method == "geometric"

    def test_params_default_empty(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert isinstance(r.params, dict)

    def test_repr_ok(self):
        r = GeometricMatchResult(score=0.7, aspect_score=0.8,
                                  area_score=0.6, hu_score=0.7)
        s = repr(r)
        assert "GeometricMatchResult" in s


# ─── TestComputeFragmentGeometry ──────────────────────────────────────────────

class TestComputeFragmentGeometry:
    def test_returns_fragment_geometry(self):
        g = compute_fragment_geometry(_square_mask())
        assert isinstance(g, FragmentGeometry)

    def test_area_positive_for_rect(self):
        g = compute_fragment_geometry(_square_mask())
        assert g.area > 0.0

    def test_perimeter_positive_for_rect(self):
        g = compute_fragment_geometry(_square_mask())
        assert g.perimeter > 0.0

    def test_hu_moments_shape(self):
        g = compute_fragment_geometry(_square_mask())
        assert g.hu_moments.shape == (7,)

    def test_blank_mask_returns_zero_area(self):
        g = compute_fragment_geometry(_blank_mask())
        assert g.area == pytest.approx(0.0)

    def test_blank_mask_returns_zero_perimeter(self):
        g = compute_fragment_geometry(_blank_mask())
        assert g.perimeter == pytest.approx(0.0)

    def test_bbox_non_zero_for_rect(self):
        g = compute_fragment_geometry(_square_mask())
        x, y, w, h = g.bbox
        assert w > 0 and h > 0

    def test_solidity_leq_one(self):
        g = compute_fragment_geometry(_square_mask())
        assert g.solidity <= 1.0 + 1e-5

    def test_aspect_ratio_square_close_to_one(self):
        g = compute_fragment_geometry(_square_mask())
        assert g.aspect_ratio == pytest.approx(1.0, abs=0.1)

    def test_wide_rect_higher_ar(self):
        g_wide = compute_fragment_geometry(_wide_mask())
        g_sq = compute_fragment_geometry(_square_mask())
        assert g_wide.aspect_ratio > g_sq.aspect_ratio

    def test_rgb_mask_ok(self):
        g = compute_fragment_geometry(_rgb_mask())
        assert g.area > 0.0

    def test_params_has_n_contours(self):
        g = compute_fragment_geometry(_square_mask())
        assert "n_contours" in g.params

    def test_blank_has_zero_contours(self):
        g = compute_fragment_geometry(_blank_mask())
        assert g.params["n_contours"] == 0


# ─── TestSimilarityFunctions ──────────────────────────────────────────────────

class TestAspectRatioSimilarity:
    def test_identical_aspect_ratio_one(self):
        g1 = _geom(aspect_ratio=1.5)
        g2 = _geom(aspect_ratio=1.5)
        assert aspect_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_result_in_range(self):
        g1 = _geom(aspect_ratio=1.0)
        g2 = _geom(aspect_ratio=3.0)
        s = aspect_ratio_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_very_different_low_score(self):
        g1 = _geom(aspect_ratio=1.0)
        g2 = _geom(aspect_ratio=10.0)
        assert aspect_ratio_similarity(g1, g2) < 0.5

    def test_symmetric(self):
        g1 = _geom(aspect_ratio=1.5)
        g2 = _geom(aspect_ratio=2.5)
        assert (aspect_ratio_similarity(g1, g2) ==
                pytest.approx(aspect_ratio_similarity(g2, g1), abs=1e-6))


class TestAreaRatioSimilarity:
    def test_identical_areas_one(self):
        g1 = _geom(area=500.0)
        g2 = _geom(area=500.0)
        assert area_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_result_in_range(self):
        g1 = _geom(area=100.0)
        g2 = _geom(area=900.0)
        s = area_ratio_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_zero_areas_return_one(self):
        g1 = _geom(area=0.0)
        g2 = _geom(area=0.0)
        assert area_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_symmetric(self):
        g1 = _geom(area=200.0)
        g2 = _geom(area=800.0)
        assert (area_ratio_similarity(g1, g2) ==
                pytest.approx(area_ratio_similarity(g2, g1), abs=1e-6))


class TestHuMomentsSimilarity:
    def test_identical_moments_one(self):
        hu = np.array([-1.0, 2.0, 0.5, -0.3, 0.1, 0.0, 0.0])
        g1 = FragmentGeometry(area=100.0, perimeter=40.0, aspect_ratio=1.0,
                               hull_area=100.0, solidity=1.0, hu_moments=hu.copy(),
                               bbox=(0, 0, 10, 10))
        g2 = FragmentGeometry(area=100.0, perimeter=40.0, aspect_ratio=1.0,
                               hull_area=100.0, solidity=1.0, hu_moments=hu.copy(),
                               bbox=(0, 0, 10, 10))
        assert hu_moments_similarity(g1, g2) == pytest.approx(1.0, abs=1e-6)

    def test_result_in_range(self):
        g1 = compute_fragment_geometry(_square_mask())
        g2 = compute_fragment_geometry(_wide_mask())
        s = hu_moments_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        g1 = compute_fragment_geometry(_square_mask())
        g2 = compute_fragment_geometry(_wide_mask())
        assert (hu_moments_similarity(g1, g2) ==
                pytest.approx(hu_moments_similarity(g2, g1), abs=1e-6))


class TestEdgeLengthSimilarity:
    def test_identical_lengths_one(self):
        assert edge_length_similarity(50.0, 50.0) == pytest.approx(1.0)

    def test_zero_lengths_return_one(self):
        assert edge_length_similarity(0.0, 0.0) == pytest.approx(1.0)

    def test_result_in_range(self):
        s = edge_length_similarity(10.0, 90.0)
        assert 0.0 <= s <= 1.0

    def test_smaller_over_larger(self):
        # 10/100 = 0.1
        s = edge_length_similarity(10.0, 100.0)
        assert s == pytest.approx(0.1, abs=1e-5)

    def test_symmetric(self):
        assert (edge_length_similarity(30.0, 70.0) ==
                pytest.approx(edge_length_similarity(70.0, 30.0), abs=1e-6))


# ─── TestMatchGeometry ────────────────────────────────────────────────────────

class TestMatchGeometry:
    def test_returns_geometric_match_result(self):
        g1 = compute_fragment_geometry(_square_mask())
        g2 = compute_fragment_geometry(_square_mask())
        r = match_geometry(g1, g2)
        assert isinstance(r, GeometricMatchResult)

    def test_score_in_range(self):
        g1 = compute_fragment_geometry(_square_mask())
        g2 = compute_fragment_geometry(_wide_mask())
        r = match_geometry(g1, g2)
        assert 0.0 <= r.score <= 1.0

    def test_method_geometric(self):
        g = _geom()
        r = match_geometry(g, g)
        assert r.method == "geometric"

    def test_identical_geom_high_score(self):
        g = compute_fragment_geometry(_square_mask())
        r = match_geometry(g, g)
        assert r.score > 0.8

    def test_aspect_score_in_range(self):
        g1 = _geom(aspect_ratio=1.0)
        g2 = _geom(aspect_ratio=2.0)
        r = match_geometry(g1, g2)
        assert 0.0 <= r.aspect_score <= 1.0

    def test_area_score_in_range(self):
        g1 = _geom(area=100.0)
        g2 = _geom(area=500.0)
        r = match_geometry(g1, g2)
        assert 0.0 <= r.area_score <= 1.0

    def test_hu_score_in_range(self):
        g1 = _geom()
        g2 = _geom()
        r = match_geometry(g1, g2)
        assert 0.0 <= r.hu_score <= 1.0

    def test_with_edge_lengths(self):
        g = _geom()
        r = match_geometry(g, g, edge_len1=50.0, edge_len2=50.0)
        assert "edge_len_score" in r.params
        assert 0.0 <= r.score <= 1.0

    def test_no_edge_lengths(self):
        g = _geom()
        r = match_geometry(g, g)
        assert "edge_len_score" not in r.params

    def test_custom_weights(self):
        g1 = _geom()
        g2 = _geom()
        r = match_geometry(g1, g2, w_aspect=1.0, w_area=0.0, w_hu=0.0)
        assert r.score == pytest.approx(r.aspect_score, abs=1e-4)

    def test_params_has_weights(self):
        g = _geom()
        r = match_geometry(g, g)
        assert "w_aspect" in r.params
        assert "w_area" in r.params
        assert "w_hu" in r.params


# ─── TestBatchGeometryMatch ───────────────────────────────────────────────────

class TestBatchGeometryMatch:
    def test_returns_list(self):
        geoms = [compute_fragment_geometry(_square_mask()),
                 compute_fragment_geometry(_wide_mask()),
                 compute_fragment_geometry(_square_mask())]
        result = batch_geometry_match(geoms, [(0, 1), (1, 2)])
        assert isinstance(result, list)

    def test_length_matches_pairs(self):
        geoms = [_geom() for _ in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_geometry_match(geoms, pairs)
        assert len(result) == 3

    def test_empty_pairs(self):
        geoms = [_geom()]
        result = batch_geometry_match(geoms, [])
        assert result == []

    def test_all_geometric_match_results(self):
        geoms = [_geom() for _ in range(3)]
        for r in batch_geometry_match(geoms, [(0, 1), (1, 2)]):
            assert isinstance(r, GeometricMatchResult)

    def test_scores_in_range(self):
        geoms = [compute_fragment_geometry(_square_mask()),
                 compute_fragment_geometry(_wide_mask())]
        for r in batch_geometry_match(geoms, [(0, 1)]):
            assert 0.0 <= r.score <= 1.0

    def test_out_of_range_index_raises(self):
        geoms = [_geom()]
        with pytest.raises(IndexError):
            batch_geometry_match(geoms, [(0, 5)])

    def test_custom_weights_passed(self):
        geoms = [_geom() for _ in range(2)]
        result = batch_geometry_match(geoms, [(0, 1)],
                                       w_aspect=0.5, w_area=0.3, w_hu=0.2)
        assert result[0].params["w_aspect"] == pytest.approx(0.5)
