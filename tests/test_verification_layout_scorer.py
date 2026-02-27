"""Tests for puzzle_reconstruction.verification.layout_scorer"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.layout_scorer import (
    PlacedFragment,
    LayoutScorerConfig,
    LayoutScoreResult,
    compute_coverage,
    compute_overlap_ratio,
    compute_uniformity,
    score_layout,
    rank_layouts,
    batch_score_layouts,
)


def make_fragment(fid, x, y, w=50, h=50, angle=0.0, score=0.5):
    return PlacedFragment(fragment_id=fid, x=x, y=y, w=w, h=h, angle=angle, score=score)


# ─── PlacedFragment ───────────────────────────────────────────────────────────

def test_placed_fragment_area():
    f = make_fragment(0, 0, 0, w=40, h=30)
    assert f.area == 1200


def test_placed_fragment_x2_y2():
    f = make_fragment(0, 10, 20, w=30, h=40)
    assert f.x2 == 40
    assert f.y2 == 60


def test_placed_fragment_center():
    f = make_fragment(0, 0, 0, w=100, h=100)
    cx, cy = f.center
    assert cx == 50.0
    assert cy == 50.0


def test_placed_fragment_invalid_id():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=-1, x=0, y=0, w=10, h=10)


def test_placed_fragment_invalid_x():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=0, x=-1, y=0, w=10, h=10)


def test_placed_fragment_invalid_w():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=0, x=0, y=0, w=0, h=10)


def test_placed_fragment_invalid_score():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10, score=-0.1)


# ─── LayoutScorerConfig ───────────────────────────────────────────────────────

def test_layout_scorer_config_defaults():
    cfg = LayoutScorerConfig()
    assert cfg.canvas_w == 512
    assert cfg.canvas_h == 512


def test_layout_scorer_config_invalid_canvas():
    with pytest.raises(ValueError):
        LayoutScorerConfig(canvas_w=0)


def test_layout_scorer_config_invalid_weight():
    with pytest.raises(ValueError):
        LayoutScorerConfig(coverage_weight=-1.0)


# ─── LayoutScoreResult ────────────────────────────────────────────────────────

def test_layout_score_result_quality_poor():
    r = LayoutScoreResult(
        total_score=0.1, coverage=0.5, overlap_ratio=0.0,
        uniformity=0.5, mean_frag_score=0.5, n_fragments=1
    )
    assert r.quality_level == "poor"


def test_layout_score_result_quality_excellent():
    r = LayoutScoreResult(
        total_score=0.9, coverage=0.9, overlap_ratio=0.0,
        uniformity=0.9, mean_frag_score=0.9, n_fragments=1
    )
    assert r.quality_level == "excellent"


def test_layout_score_result_quality_fair():
    r = LayoutScoreResult(
        total_score=0.35, coverage=0.5, overlap_ratio=0.0,
        uniformity=0.5, mean_frag_score=0.5, n_fragments=1
    )
    assert r.quality_level == "fair"


def test_layout_score_result_quality_good():
    r = LayoutScoreResult(
        total_score=0.65, coverage=0.5, overlap_ratio=0.0,
        uniformity=0.5, mean_frag_score=0.5, n_fragments=1
    )
    assert r.quality_level == "good"


def test_layout_score_result_invalid_coverage():
    with pytest.raises(ValueError):
        LayoutScoreResult(
            total_score=0.5, coverage=1.5, overlap_ratio=0.0,
            uniformity=0.5, mean_frag_score=0.5, n_fragments=1
        )


# ─── compute_coverage ─────────────────────────────────────────────────────────

def test_compute_coverage_empty():
    assert compute_coverage([], 100, 100) == 0.0


def test_compute_coverage_full():
    f = make_fragment(0, 0, 0, w=100, h=100)
    cov = compute_coverage([f], 100, 100)
    assert cov == pytest.approx(1.0)


def test_compute_coverage_half():
    f = make_fragment(0, 0, 0, w=50, h=100)
    cov = compute_coverage([f], 100, 100)
    assert cov == pytest.approx(0.5)


def test_compute_coverage_invalid_canvas():
    with pytest.raises(ValueError):
        compute_coverage([], 0, 100)


def test_compute_coverage_out_of_bounds():
    # Fragment extends beyond canvas - should be clipped
    f = make_fragment(0, 80, 80, w=50, h=50)
    cov = compute_coverage([f], 100, 100)
    assert 0.0 < cov < 1.0


def test_compute_coverage_two_non_overlapping():
    f1 = make_fragment(0, 0, 0, w=50, h=100)
    f2 = make_fragment(1, 50, 0, w=50, h=100)
    cov = compute_coverage([f1, f2], 100, 100)
    assert cov == pytest.approx(1.0)


# ─── compute_overlap_ratio ────────────────────────────────────────────────────

def test_compute_overlap_ratio_no_overlap():
    f1 = make_fragment(0, 0, 0, w=50, h=50)
    f2 = make_fragment(1, 50, 0, w=50, h=50)
    ratio = compute_overlap_ratio([f1, f2])
    assert ratio == pytest.approx(0.0)


def test_compute_overlap_ratio_full_overlap():
    f1 = make_fragment(0, 0, 0, w=50, h=50)
    f2 = make_fragment(1, 0, 0, w=50, h=50)
    ratio = compute_overlap_ratio([f1, f2])
    assert ratio > 0.0


def test_compute_overlap_ratio_single():
    f = make_fragment(0, 0, 0)
    assert compute_overlap_ratio([f]) == 0.0


def test_compute_overlap_ratio_empty():
    assert compute_overlap_ratio([]) == 0.0


def test_compute_overlap_ratio_range():
    f1 = make_fragment(0, 0, 0, w=50, h=50)
    f2 = make_fragment(1, 25, 25, w=50, h=50)
    ratio = compute_overlap_ratio([f1, f2])
    assert 0.0 <= ratio <= 1.0


# ─── compute_uniformity ───────────────────────────────────────────────────────

def test_compute_uniformity_single():
    f = make_fragment(0, 0, 0)
    assert compute_uniformity([f], 100, 100) == pytest.approx(1.0)


def test_compute_uniformity_empty():
    assert compute_uniformity([], 100, 100) == pytest.approx(1.0)


def test_compute_uniformity_symmetric():
    # Symmetric placement should be more uniform
    f1 = make_fragment(0, 0, 0, w=100, h=100)
    f2 = make_fragment(1, 300, 300, w=100, h=100)
    u = compute_uniformity([f1, f2], 400, 400)
    assert 0.0 <= u <= 1.0


# ─── score_layout ─────────────────────────────────────────────────────────────

def test_score_layout_empty():
    result = score_layout([])
    assert result.total_score == 0.0
    assert result.n_fragments == 0


def test_score_layout_basic():
    frags = [make_fragment(i, i * 50, 0) for i in range(4)]
    cfg = LayoutScorerConfig(canvas_w=200, canvas_h=100)
    result = score_layout(frags, cfg)
    assert 0.0 <= result.total_score <= 1.0
    assert result.n_fragments == 4


def test_score_layout_no_overlap_good():
    f1 = make_fragment(0, 0, 0, w=50, h=50, score=0.8)
    f2 = make_fragment(1, 50, 0, w=50, h=50, score=0.8)
    cfg = LayoutScorerConfig(canvas_w=100, canvas_h=50)
    result = score_layout([f1, f2], cfg)
    assert result.overlap_ratio == pytest.approx(0.0)
    assert result.coverage == pytest.approx(1.0)


def test_score_layout_default_config():
    f = make_fragment(0, 0, 0, score=0.9)
    result = score_layout([f])
    assert isinstance(result, LayoutScoreResult)


# ─── rank_layouts ─────────────────────────────────────────────────────────────

def test_rank_layouts_ordering():
    good = [make_fragment(i, i * 50, 0, score=0.9) for i in range(4)]
    bad = [make_fragment(i, 0, 0, score=0.1) for i in range(4)]
    cfg = LayoutScorerConfig(canvas_w=200, canvas_h=50)
    ranked = rank_layouts([bad, good], cfg)
    assert len(ranked) == 2
    # First should have higher score
    assert ranked[0][1].total_score >= ranked[1][1].total_score


def test_rank_layouts_empty():
    results = rank_layouts([])
    assert results == []


def test_rank_layouts_preserves_index():
    layouts = [[make_fragment(0, 0, 0)], [make_fragment(0, 10, 10)]]
    ranked = rank_layouts(layouts)
    indices = [idx for idx, _ in ranked]
    assert set(indices) == {0, 1}


# ─── batch_score_layouts ──────────────────────────────────────────────────────

def test_batch_score_layouts_basic():
    layouts = [
        [make_fragment(0, 0, 0)],
        [make_fragment(0, 10, 10)],
    ]
    results = batch_score_layouts(layouts)
    assert len(results) == 2


def test_batch_score_layouts_empty():
    results = batch_score_layouts([])
    assert results == []
