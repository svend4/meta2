"""Extra tests for puzzle_reconstruction.matching.geometric_match."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.geometric_match import (
    FragmentGeometry,
    GeometricMatchResult,
    area_ratio_similarity,
    aspect_ratio_similarity,
    batch_geometry_match,
    compute_fragment_geometry,
    edge_length_similarity,
    hu_moments_similarity,
    match_geometry,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)

def _rect_mask(h=64, w=64, top=10, left=10, bottom=50, right=50):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask

def _square():
    return _rect_mask(64, 64, 10, 10, 50, 50)

def _wide():
    return _rect_mask(64, 128, 10, 10, 30, 100)

def _tall():
    return _rect_mask(128, 64, 10, 10, 110, 30)

def _geom(area=1000.0, perimeter=120.0, aspect_ratio=1.0,
          solidity=1.0, hull_area=None):
    hu = np.zeros(7, dtype=np.float64)
    return FragmentGeometry(
        area=area, perimeter=perimeter, aspect_ratio=aspect_ratio,
        hull_area=hull_area if hull_area is not None else area,
        solidity=solidity, hu_moments=hu, bbox=(0, 0, 32, 32),
    )


# ─── TestFragmentGeometryExtra ────────────────────────────────────────────────

class TestFragmentGeometryExtra:
    def test_large_area(self):
        g = _geom(area=100000.0)
        assert g.area == pytest.approx(100000.0)

    def test_hull_area_stored(self):
        g = _geom(area=800.0, hull_area=1000.0)
        assert g.hull_area == pytest.approx(1000.0)

    def test_bbox_four_values(self):
        g = _geom()
        assert len(g.bbox) == 4

    def test_hu_moments_7_values(self):
        g = _geom()
        assert len(g.hu_moments) == 7

    def test_solidity_range(self):
        g = _geom(solidity=0.8)
        assert 0.0 <= g.solidity <= 1.0

    def test_perimeter_positive(self):
        g = _geom(perimeter=200.0)
        assert g.perimeter == pytest.approx(200.0)

    def test_params_empty_default(self):
        g = _geom()
        assert g.params == {}


# ─── TestGeometricMatchResultExtra ────────────────────────────────────────────

class TestGeometricMatchResultExtra:
    def test_score_0_valid(self):
        r = GeometricMatchResult(score=0.0, aspect_score=0.0,
                                  area_score=0.0, hu_score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_1_valid(self):
        r = GeometricMatchResult(score=1.0, aspect_score=1.0,
                                  area_score=1.0, hu_score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_sub_scores_in_range(self):
        r = GeometricMatchResult(score=0.7, aspect_score=0.6,
                                  area_score=0.8, hu_score=0.7)
        for v in (r.aspect_score, r.area_score, r.hu_score):
            assert 0.0 <= v <= 1.0

    def test_custom_params_stored(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5,
                                  params={"edge_len_score": 0.9})
        assert r.params["edge_len_score"] == pytest.approx(0.9)

    def test_method_is_string(self):
        r = GeometricMatchResult(score=0.5, aspect_score=0.5,
                                  area_score=0.5, hu_score=0.5)
        assert isinstance(r.method, str)


# ─── TestComputeFragmentGeometryExtra ─────────────────────────────────────────

class TestComputeFragmentGeometryExtra:
    def test_tall_rect_ar_positive(self):
        g = compute_fragment_geometry(_tall())
        assert g.aspect_ratio > 0.0

    def test_wide_rect_ar_gt_one(self):
        g = compute_fragment_geometry(_wide())
        assert g.aspect_ratio > 1.0

    def test_large_mask(self):
        mask = np.full((256, 256), 255, dtype=np.uint8)
        g = compute_fragment_geometry(mask)
        assert g.area > 0.0

    def test_small_mask(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        g = compute_fragment_geometry(mask)
        assert g.area > 0.0

    def test_hu_moments_float64(self):
        g = compute_fragment_geometry(_square())
        assert g.hu_moments.dtype == np.float64

    def test_solidity_nonneg(self):
        g = compute_fragment_geometry(_square())
        assert g.solidity >= 0.0

    def test_bbox_all_positive_for_rect(self):
        g = compute_fragment_geometry(_square())
        x, y, w, h = g.bbox
        assert x >= 0 and y >= 0 and w > 0 and h > 0


# ─── TestAspectRatioSimilarityExtra ──────────────────────────────────────────

class TestAspectRatioSimilarityExtra:
    def test_both_equal_non_1(self):
        g1 = _geom(aspect_ratio=2.0)
        g2 = _geom(aspect_ratio=2.0)
        assert aspect_ratio_similarity(g1, g2) == pytest.approx(1.0)

    def test_very_similar(self):
        g1 = _geom(aspect_ratio=1.0)
        g2 = _geom(aspect_ratio=1.05)
        assert aspect_ratio_similarity(g1, g2) > 0.9

    def test_five_different_combos_in_range(self):
        pairs = [(1.0, 2.0), (0.5, 0.6), (1.0, 10.0), (2.0, 3.0), (0.1, 0.2)]
        for a, b in pairs:
            s = aspect_ratio_similarity(_geom(aspect_ratio=a), _geom(aspect_ratio=b))
            assert 0.0 <= s <= 1.0

    def test_extremes_give_low_score(self):
        g1 = _geom(aspect_ratio=0.01)
        g2 = _geom(aspect_ratio=100.0)
        assert aspect_ratio_similarity(g1, g2) < 0.1


# ─── TestAreaRatioSimilarityExtra ─────────────────────────────────────────────

class TestAreaRatioSimilarityExtra:
    def test_large_ratio_low_score(self):
        g1 = _geom(area=10.0)
        g2 = _geom(area=10000.0)
        assert area_ratio_similarity(g1, g2) < 0.1

    def test_one_zero_other_nonzero(self):
        g1 = _geom(area=0.0)
        g2 = _geom(area=100.0)
        s = area_ratio_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_ratio_0_5(self):
        g1 = _geom(area=50.0)
        g2 = _geom(area=100.0)
        s = area_ratio_similarity(g1, g2)
        assert s == pytest.approx(0.5, abs=1e-5)

    def test_various_pairs_in_range(self):
        pairs = [(100.0, 200.0), (10.0, 1000.0), (500.0, 500.0)]
        for a, b in pairs:
            s = area_ratio_similarity(_geom(area=a), _geom(area=b))
            assert 0.0 <= s <= 1.0


# ─── TestHuMomentsSimilarityExtra ────────────────────────────────────────────

class TestHuMomentsSimilarityExtra:
    def test_both_zeros_returns_one(self):
        hu = np.zeros(7, dtype=np.float64)
        g1 = FragmentGeometry(area=100.0, perimeter=40.0, aspect_ratio=1.0,
                               hull_area=100.0, solidity=1.0, hu_moments=hu.copy(),
                               bbox=(0, 0, 10, 10))
        g2 = FragmentGeometry(area=100.0, perimeter=40.0, aspect_ratio=1.0,
                               hull_area=100.0, solidity=1.0, hu_moments=hu.copy(),
                               bbox=(0, 0, 10, 10))
        assert hu_moments_similarity(g1, g2) == pytest.approx(1.0, abs=1e-6)

    def test_square_vs_blank_in_range(self):
        g1 = compute_fragment_geometry(_square())
        g2 = compute_fragment_geometry(_blank())
        s = hu_moments_similarity(g1, g2)
        assert 0.0 <= s <= 1.0

    def test_same_shape_different_size(self):
        g1 = compute_fragment_geometry(_rect_mask(64, 64, 10, 10, 50, 50))
        g2 = compute_fragment_geometry(_rect_mask(128, 128, 20, 20, 100, 100))
        s = hu_moments_similarity(g1, g2)
        assert 0.0 <= s <= 1.0


# ─── TestEdgeLengthSimilarityExtra ───────────────────────────────────────────

class TestEdgeLengthSimilarityExtra:
    def test_equal_large_values(self):
        assert edge_length_similarity(1000.0, 1000.0) == pytest.approx(1.0)

    def test_ratio_0_5(self):
        s = edge_length_similarity(50.0, 100.0)
        assert s == pytest.approx(0.5, abs=1e-5)

    def test_in_range_various(self):
        pairs = [(1.0, 2.0), (5.0, 50.0), (100.0, 101.0), (0.0, 0.0)]
        for a, b in pairs:
            s = edge_length_similarity(a, b)
            assert 0.0 <= s <= 1.0


# ─── TestMatchGeometryExtra ───────────────────────────────────────────────────

class TestMatchGeometryExtra:
    def test_different_aspects_lower_score(self):
        g_sq = compute_fragment_geometry(_square())
        g_wide = compute_fragment_geometry(_wide())
        r_same = match_geometry(g_sq, g_sq)
        r_diff = match_geometry(g_sq, g_wide)
        assert r_same.score >= r_diff.score

    def test_edge_lengths_identical_high_score(self):
        g = _geom()
        r = match_geometry(g, g, edge_len1=40.0, edge_len2=40.0)
        assert r.score > 0.8

    def test_blank_vs_rect_lower_than_rect_vs_rect(self):
        g_rect = compute_fragment_geometry(_square())
        g_blank = compute_fragment_geometry(_blank())
        r_same = match_geometry(g_rect, g_rect)
        r_diff = match_geometry(g_rect, g_blank)
        assert r_same.score >= r_diff.score

    def test_aspect_score_all_in_01(self):
        for ar in (0.5, 1.0, 2.0):
            g1 = _geom(aspect_ratio=ar)
            g2 = _geom(aspect_ratio=ar * 2)
            r = match_geometry(g1, g2)
            assert 0.0 <= r.aspect_score <= 1.0

    def test_edge_score_stored_when_given(self):
        g = _geom()
        r = match_geometry(g, g, edge_len1=30.0, edge_len2=30.0)
        assert r.params.get("edge_len_score") is not None


# ─── TestBatchGeometryMatchExtra ─────────────────────────────────────────────

class TestBatchGeometryMatchExtra:
    def test_single_pair(self):
        geoms = [_geom(), _geom()]
        result = batch_geometry_match(geoms, [(0, 1)])
        assert len(result) == 1

    def test_blank_vs_rect(self):
        geoms = [compute_fragment_geometry(_blank()),
                 compute_fragment_geometry(_square())]
        result = batch_geometry_match(geoms, [(0, 1)])
        assert 0.0 <= result[0].score <= 1.0

    def test_five_pairs(self):
        geoms = [_geom() for _ in range(5)]
        pairs = [(i, i + 1) for i in range(4)]
        result = batch_geometry_match(geoms, pairs)
        assert len(result) == 4

    def test_same_idx_pair_high_score(self):
        geoms = [compute_fragment_geometry(_square()),
                 compute_fragment_geometry(_square())]
        result = batch_geometry_match(geoms, [(0, 1)])
        assert result[0].score > 0.8

    def test_all_results_are_gmr(self):
        geoms = [_geom() for _ in range(3)]
        for r in batch_geometry_match(geoms, [(0, 1), (0, 2), (1, 2)]):
            assert isinstance(r, GeometricMatchResult)
