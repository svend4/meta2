"""Extra tests for puzzle_reconstruction/utils/region_score_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.region_score_utils import (
    RegionScoreConfig,
    RegionScore,
    region_compactness,
    region_extent,
    mask_perimeter,
    score_region,
    evaluate_region,
    filter_by_score,
    rank_regions,
    top_k_regions,
    region_score_stats,
    batch_evaluate_regions,
    normalize_scores,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rscore(label=0, area=100, compactness=0.8, extent=0.9,
            score=0.7) -> RegionScore:
    return RegionScore(label=label, area=area, compactness=compactness,
                       extent=extent, score=score)


def _square_mask(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=np.uint8)


def _square_bbox(size: int):
    return (0, 0, size, size)


# ─── RegionScoreConfig ────────────────────────────────────────────────────────

class TestRegionScoreConfigExtra:
    def test_default_weights(self):
        cfg = RegionScoreConfig()
        assert cfg.w_area == pytest.approx(0.4)
        assert cfg.w_compactness == pytest.approx(0.3)
        assert cfg.w_extent == pytest.approx(0.3)

    def test_default_min_area(self):
        assert RegionScoreConfig().min_area == 1

    def test_custom_config(self):
        cfg = RegionScoreConfig(w_area=0.5, min_area=10)
        assert cfg.w_area == pytest.approx(0.5) and cfg.min_area == 10


# ─── RegionScore ──────────────────────────────────────────────────────────────

class TestRegionScoreExtra:
    def test_repr_contains_label(self):
        r = _rscore(label=5)
        assert "label=5" in repr(r)

    def test_meta_default_empty(self):
        r = _rscore()
        assert r.meta == {}

    def test_fields_stored(self):
        r = _rscore(area=200, score=0.5)
        assert r.area == 200 and r.score == pytest.approx(0.5)


# ─── region_compactness ───────────────────────────────────────────────────────

class TestRegionCompactnessExtra:
    def test_zero_perimeter_returns_zero(self):
        assert region_compactness(100, 0.0) == pytest.approx(0.0)

    def test_circle_approx_one(self):
        # Circle: area=πr², perimeter=2πr → compactness ≈ 1
        import math
        r = 10.0
        area = math.pi * r ** 2
        perimeter = 2 * math.pi * r
        c = region_compactness(int(area), perimeter)
        assert c == pytest.approx(1.0, abs=1e-3)

    def test_result_in_unit_range(self):
        c = region_compactness(50, 30.0)
        assert 0.0 <= c <= 1.0


# ─── region_extent ────────────────────────────────────────────────────────────

class TestRegionExtentExtra:
    def test_full_fill(self):
        bbox = (0, 0, 10, 10)
        assert region_extent(100, bbox) == pytest.approx(1.0)

    def test_partial_fill(self):
        bbox = (0, 0, 10, 10)
        assert region_extent(50, bbox) == pytest.approx(0.5)

    def test_zero_bbox_area_returns_one(self):
        bbox = (5, 5, 5, 5)
        assert region_extent(100, bbox) == pytest.approx(1.0)


# ─── mask_perimeter ───────────────────────────────────────────────────────────

class TestMaskPerimeterExtra:
    def test_empty_mask_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_perimeter(mask) == pytest.approx(0.0)

    def test_single_pixel(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        p = mask_perimeter(mask)
        assert p > 0

    def test_solid_square_has_perimeter(self):
        mask = _square_mask(10)
        p = mask_perimeter(mask)
        assert p > 0


# ─── score_region ─────────────────────────────────────────────────────────────

class TestScoreRegionExtra:
    def test_returns_float(self):
        mask = _square_mask(10)
        s = score_region(100, _square_bbox(10), mask)
        assert isinstance(s, float)

    def test_result_in_unit_range(self):
        mask = _square_mask(10)
        s = score_region(100, _square_bbox(10), mask)
        assert 0.0 <= s <= 1.0

    def test_zero_weights_returns_zero(self):
        mask = _square_mask(10)
        cfg = RegionScoreConfig(w_area=0.0, w_compactness=0.0, w_extent=0.0)
        s = score_region(100, _square_bbox(10), mask, cfg=cfg)
        assert s == pytest.approx(0.0)


# ─── evaluate_region ─────────────────────────────────────────────────────────

class TestEvaluateRegionExtra:
    def test_returns_region_score(self):
        mask = _square_mask(10)
        rs = evaluate_region(1, 100, _square_bbox(10), mask)
        assert isinstance(rs, RegionScore)

    def test_label_stored(self):
        mask = _square_mask(5)
        rs = evaluate_region(7, 25, _square_bbox(5), mask)
        assert rs.label == 7

    def test_score_in_range(self):
        mask = _square_mask(8)
        rs = evaluate_region(0, 64, _square_bbox(8), mask)
        assert 0.0 <= rs.score <= 1.0


# ─── filter_by_score ─────────────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_threshold_filter(self):
        scores = [_rscore(score=0.3), _rscore(score=0.8)]
        result = filter_by_score(scores, threshold=0.5)
        assert len(result) == 1

    def test_zero_threshold_keeps_all(self):
        scores = [_rscore(score=0.1), _rscore(score=0.9)]
        result = filter_by_score(scores, threshold=0.0)
        assert len(result) == 2


# ─── rank_regions ────────────────────────────────────────────────────────────

class TestRankRegionsExtra:
    def test_descending_order(self):
        scores = [_rscore(score=0.3), _rscore(score=0.9), _rscore(score=0.5)]
        ranked = rank_regions(scores)
        assert ranked[0].score == pytest.approx(0.9)

    def test_ascending_order(self):
        scores = [_rscore(score=0.3), _rscore(score=0.9)]
        ranked = rank_regions(scores, reverse=False)
        assert ranked[0].score == pytest.approx(0.3)


# ─── top_k_regions ───────────────────────────────────────────────────────────

class TestTopKRegionsExtra:
    def test_top_k_count(self):
        scores = [_rscore(score=0.3), _rscore(score=0.9), _rscore(score=0.5)]
        top = top_k_regions(scores, 2)
        assert len(top) == 2

    def test_top_k_best_first(self):
        scores = [_rscore(score=0.3), _rscore(score=0.9)]
        top = top_k_regions(scores, 1)
        assert top[0].score == pytest.approx(0.9)


# ─── region_score_stats ──────────────────────────────────────────────────────

class TestRegionScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = region_score_stats([])
        assert s["n"] == 0

    def test_count(self):
        scores = [_rscore(score=0.3, area=50), _rscore(score=0.7, area=100)]
        s = region_score_stats(scores)
        assert s["n"] == 2

    def test_total_area(self):
        scores = [_rscore(area=50), _rscore(area=100)]
        s = region_score_stats(scores)
        assert s["total_area"] == 150


# ─── batch_evaluate_regions ──────────────────────────────────────────────────

class TestBatchEvaluateRegionsExtra:
    def test_returns_list(self):
        mask = _square_mask(10)
        regions = [{"label": 0, "area": 100,
                    "bbox": _square_bbox(10), "mask": mask}]
        result = batch_evaluate_regions(regions)
        assert len(result) == 1

    def test_empty_input(self):
        assert batch_evaluate_regions([]) == []


# ─── normalize_scores ────────────────────────────────────────────────────────

class TestNormalizeScoresExtra:
    def test_empty_returns_empty(self):
        assert normalize_scores([]) == []

    def test_constant_scores_to_one(self):
        scores = [_rscore(score=0.5), _rscore(score=0.5)]
        norm = normalize_scores(scores)
        assert all(r.score == pytest.approx(1.0) for r in norm)

    def test_min_max_normalized(self):
        scores = [_rscore(score=0.0), _rscore(score=1.0)]
        norm = normalize_scores(scores)
        vals = sorted(r.score for r in norm)
        assert vals[0] == pytest.approx(0.0) and vals[1] == pytest.approx(1.0)
