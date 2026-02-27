"""Tests for puzzle_reconstruction.utils.region_score_utils."""
import pytest
import numpy as np

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

np.random.seed(42)


def _square_mask(size=10):
    m = np.zeros((size, size), dtype=np.uint8)
    m[1:-1, 1:-1] = 1
    return m


def _circle_mask(r=10):
    sz = r * 2 + 1
    m = np.zeros((sz, sz), dtype=np.uint8)
    cx, cy = r, r
    for i in range(sz):
        for j in range(sz):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                m[i, j] = 1
    return m


# ── RegionScoreConfig ─────────────────────────────────────────────────────────

def test_region_score_config_defaults():
    cfg = RegionScoreConfig()
    assert cfg.w_area == pytest.approx(0.4)
    assert cfg.w_compactness == pytest.approx(0.3)
    assert cfg.w_extent == pytest.approx(0.3)
    assert cfg.min_area == 1
    assert cfg.max_area == 10_000_000


# ── region_compactness ────────────────────────────────────────────────────────

def test_region_compactness_zero_perimeter():
    assert region_compactness(100, 0.0) == 0.0


def test_region_compactness_circle_approaches_one():
    # Large area / small perimeter → high compactness
    comp = region_compactness(314, 63.0)  # ~circle r=10
    assert comp > 0.9


def test_region_compactness_in_range():
    comp = region_compactness(50, 30.0)
    assert 0.0 <= comp <= 1.0


# ── region_extent ─────────────────────────────────────────────────────────────

def test_region_extent_full_bbox():
    ext = region_extent(100, (0, 0, 10, 10))
    assert ext == pytest.approx(1.0)


def test_region_extent_partial():
    ext = region_extent(50, (0, 0, 10, 10))
    assert ext == pytest.approx(0.5)


def test_region_extent_zero_bbox():
    ext = region_extent(10, (5, 5, 5, 5))
    assert ext == pytest.approx(1.0)


# ── mask_perimeter ────────────────────────────────────────────────────────────

def test_mask_perimeter_empty():
    m = np.zeros((10, 10), dtype=np.uint8)
    assert mask_perimeter(m) == 0.0


def test_mask_perimeter_square():
    m = _square_mask(10)
    p = mask_perimeter(m)
    assert p > 0.0


def test_mask_perimeter_single_pixel():
    m = np.zeros((5, 5), dtype=np.uint8)
    m[2, 2] = 1
    p = mask_perimeter(m)
    assert p == 1.0


# ── score_region ──────────────────────────────────────────────────────────────

def test_score_region_in_range():
    m = _square_mask(10)
    area = int(m.sum())
    bbox = (1, 1, 9, 9)
    s = score_region(area, bbox, m)
    assert 0.0 <= s <= 1.0


def test_score_region_zero_weights():
    cfg = RegionScoreConfig(w_area=0.0, w_compactness=0.0, w_extent=0.0)
    m = _square_mask(10)
    s = score_region(int(m.sum()), (1, 1, 9, 9), m, cfg)
    assert s == pytest.approx(0.0)


def test_score_region_custom_weights():
    cfg = RegionScoreConfig(w_area=1.0, w_compactness=0.0, w_extent=0.0)
    m = _square_mask(10)
    s = score_region(int(m.sum()), (1, 1, 9, 9), m, cfg)
    assert 0.0 <= s <= 1.0


# ── evaluate_region ───────────────────────────────────────────────────────────

def test_evaluate_region_returns_region_score():
    m = _square_mask(10)
    rs = evaluate_region(label=1, area=int(m.sum()), bbox=(1, 1, 9, 9), mask=m)
    assert isinstance(rs, RegionScore)
    assert rs.label == 1
    assert 0.0 <= rs.score <= 1.0


def test_evaluate_region_compactness_in_range():
    m = _circle_mask(5)
    rs = evaluate_region(label=0, area=int(m.sum()), bbox=(0, 0, 11, 11), mask=m)
    assert 0.0 <= rs.compactness <= 1.0


# ── filter_by_score ───────────────────────────────────────────────────────────

def test_filter_by_score_threshold():
    regions = [
        RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=float(i) * 0.2)
        for i in range(6)
    ]
    filtered = filter_by_score(regions, threshold=0.4)
    assert all(r.score >= 0.4 for r in filtered)


def test_filter_by_score_all_pass():
    regions = [RegionScore(label=0, area=10, compactness=0.5, extent=0.8, score=1.0)]
    assert len(filter_by_score(regions, 0.0)) == 1


# ── rank_regions ──────────────────────────────────────────────────────────────

def test_rank_regions_descending():
    regions = [RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=float(i) * 0.1)
               for i in range(5)]
    ranked = rank_regions(regions)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_regions_ascending():
    regions = [RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=float(i) * 0.1)
               for i in range(5)]
    ranked = rank_regions(regions, reverse=False)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores)


# ── top_k_regions ─────────────────────────────────────────────────────────────

def test_top_k_regions():
    regions = [RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=float(i) * 0.1)
               for i in range(6)]
    top = top_k_regions(regions, 3)
    assert len(top) == 3
    assert top[0].score >= top[2].score


# ── region_score_stats ────────────────────────────────────────────────────────

def test_region_score_stats_empty():
    stats = region_score_stats([])
    assert stats["n"] == 0


def test_region_score_stats():
    regions = [RegionScore(label=i, area=10 + i, compactness=0.5, extent=0.8, score=0.1 * i)
               for i in range(5)]
    stats = region_score_stats(regions)
    assert stats["n"] == 5
    assert stats["total_area"] == sum(10 + i for i in range(5))
    assert stats["max_score"] >= stats["min_score"]


# ── batch_evaluate_regions ────────────────────────────────────────────────────

def test_batch_evaluate_regions():
    m = _square_mask(8)
    regions = [
        {"label": i, "area": int(m.sum()), "bbox": (1, 1, 7, 7), "mask": m}
        for i in range(3)
    ]
    results = batch_evaluate_regions(regions)
    assert len(results) == 3
    assert all(isinstance(r, RegionScore) for r in results)


# ── normalize_scores ──────────────────────────────────────────────────────────

def test_normalize_scores_empty():
    assert normalize_scores([]) == []


def test_normalize_scores_range():
    regions = [RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=0.1 * i)
               for i in range(5)]
    normalized = normalize_scores(regions)
    scores = [r.score for r in normalized]
    assert min(scores) == pytest.approx(0.0)
    assert max(scores) == pytest.approx(1.0)


def test_normalize_scores_all_same():
    regions = [RegionScore(label=i, area=10, compactness=0.5, extent=0.8, score=0.5)
               for i in range(3)]
    normalized = normalize_scores(regions)
    assert all(r.score == pytest.approx(1.0) for r in normalized)
