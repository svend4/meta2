"""Extra tests for puzzle_reconstruction/verification/layout_scorer.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pf(fid=0, x=0, y=0, w=50, h=50, angle=0.0, score=0.5):
    return PlacedFragment(
        fragment_id=fid, x=x, y=y, w=w, h=h, angle=angle, score=score)


def _grid_frags():
    """2x2 non-overlapping 50x50 fragments on a 100x100 canvas."""
    return [
        _pf(0, 0, 0, 50, 50),
        _pf(1, 50, 0, 50, 50),
        _pf(2, 0, 50, 50, 50),
        _pf(3, 50, 50, 50, 50),
    ]


# ─── PlacedFragment ─────────────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_properties(self):
        f = _pf(0, 10, 20, 30, 40)
        assert f.x2 == 40
        assert f.y2 == 60
        assert f.area == 1200
        assert f.center == pytest.approx((25.0, 40.0))

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=-1, x=0, y=0, w=10, h=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=-1, y=0, w=10, h=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=-1, w=10, h=10)

    def test_zero_w_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10, score=-0.1)

    def test_angle(self):
        f = _pf(0, 0, 0, 10, 10, angle=45.0)
        assert f.angle == 45.0


# ─── LayoutScorerConfig ─────────────────────────────────────────────────────

class TestLayoutScorerConfigExtra:
    def test_defaults(self):
        c = LayoutScorerConfig()
        assert c.canvas_w == 512
        assert c.canvas_h == 512
        assert c.overlap_penalty == 1.0
        assert c.coverage_weight == 1.0
        assert c.uniformity_weight == 0.5
        assert c.score_weight == 1.0

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(canvas_w=0)

    def test_zero_canvas_h_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(canvas_h=0)

    def test_negative_overlap_penalty_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(overlap_penalty=-1.0)

    def test_negative_coverage_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(coverage_weight=-1.0)

    def test_negative_uniformity_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(uniformity_weight=-1.0)

    def test_negative_score_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(score_weight=-1.0)


# ─── LayoutScoreResult ──────────────────────────────────────────────────────

class TestLayoutScoreResultExtra:
    def test_quality_levels(self):
        assert LayoutScoreResult(0.1, 0.0, 0.0, 0.5, 0.0, 0).quality_level == "poor"
        assert LayoutScoreResult(0.3, 0.0, 0.0, 0.5, 0.0, 0).quality_level == "fair"
        assert LayoutScoreResult(0.6, 0.5, 0.0, 0.5, 0.5, 1).quality_level == "good"
        assert LayoutScoreResult(0.9, 0.8, 0.0, 0.9, 0.8, 2).quality_level == "excellent"

    def test_negative_total_score_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(-0.1, 0.0, 0.0, 0.0, 0.0, 0)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(0.0, 1.5, 0.0, 0.0, 0.0, 0)

    def test_overlap_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(0.0, 0.0, -0.1, 0.0, 0.0, 0)

    def test_uniformity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(0.0, 0.0, 0.0, 1.5, 0.0, 0)

    def test_negative_mean_frag_score_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(0.0, 0.0, 0.0, 0.5, -0.1, 0)

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(0.0, 0.0, 0.0, 0.5, 0.0, -1)


# ─── compute_coverage ────────────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_full(self):
        frags = _grid_frags()
        cov = compute_coverage(frags, 100, 100)
        assert cov == pytest.approx(1.0)

    def test_empty(self):
        assert compute_coverage([], 100, 100) == pytest.approx(0.0)

    def test_partial(self):
        cov = compute_coverage([_pf(0, 0, 0, 50, 50)], 100, 100)
        assert cov == pytest.approx(0.25)

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([], 0, 100)

    def test_clipped_to_canvas(self):
        # Fragment extends beyond canvas
        cov = compute_coverage([_pf(0, 80, 80, 50, 50)], 100, 100)
        # Only 20x20 = 400 pixels inside canvas, coverage = 400/10000 = 0.04
        assert cov == pytest.approx(0.04)


# ─── compute_overlap_ratio ───────────────────────────────────────────────────

class TestComputeOverlapRatioExtra:
    def test_no_overlap(self):
        assert compute_overlap_ratio(_grid_frags()) == pytest.approx(0.0)

    def test_single(self):
        assert compute_overlap_ratio([_pf()]) == pytest.approx(0.0)

    def test_empty(self):
        assert compute_overlap_ratio([]) == pytest.approx(0.0)

    def test_with_overlap(self):
        frags = [_pf(0, 0, 0, 100, 100), _pf(1, 50, 50, 100, 100)]
        ratio = compute_overlap_ratio(frags)
        assert ratio > 0.0


# ─── compute_uniformity ─────────────────────────────────────────────────────

class TestComputeUniformityExtra:
    def test_single(self):
        assert compute_uniformity([_pf()], 100, 100) == pytest.approx(1.0)

    def test_empty(self):
        assert compute_uniformity([], 100, 100) == pytest.approx(1.0)

    def test_symmetric(self):
        frags = _grid_frags()
        u = compute_uniformity(frags, 100, 100)
        assert 0.0 <= u <= 1.0

    def test_asymmetric(self):
        frags = [_pf(0, 0, 0, 10, 10), _pf(1, 90, 90, 10, 10)]
        u = compute_uniformity(frags, 100, 100)
        assert 0.0 <= u <= 1.0


# ─── score_layout ───────────────────────────────────────────────────────────

class TestScoreLayoutExtra:
    def test_empty(self):
        r = score_layout([])
        assert r.total_score == pytest.approx(0.0)
        assert r.n_fragments == 0

    def test_full_coverage(self):
        cfg = LayoutScorerConfig(canvas_w=100, canvas_h=100)
        r = score_layout(_grid_frags(), cfg)
        assert r.coverage == pytest.approx(1.0)
        assert r.n_fragments == 4

    def test_no_overlap(self):
        cfg = LayoutScorerConfig(canvas_w=100, canvas_h=100)
        r = score_layout(_grid_frags(), cfg)
        assert r.overlap_ratio == pytest.approx(0.0)

    def test_positive_score(self):
        cfg = LayoutScorerConfig(canvas_w=100, canvas_h=100)
        r = score_layout(_grid_frags(), cfg)
        assert r.total_score > 0.0

    def test_default_config(self):
        r = score_layout([_pf(0, 0, 0, 50, 50)])
        assert isinstance(r, LayoutScoreResult)


# ─── rank_layouts ───────────────────────────────────────────────────────────

class TestRankLayoutsExtra:
    def test_empty(self):
        assert rank_layouts([]) == []

    def test_order(self):
        layout_a = _grid_frags()  # full coverage
        layout_b = [_pf(0, 0, 0, 10, 10)]  # tiny
        cfg = LayoutScorerConfig(canvas_w=100, canvas_h=100)
        ranked = rank_layouts([layout_a, layout_b], cfg)
        assert len(ranked) == 2
        # First should have higher score
        assert ranked[0][1].total_score >= ranked[1][1].total_score

    def test_single(self):
        ranked = rank_layouts([_grid_frags()])
        assert len(ranked) == 1
        assert ranked[0][0] == 0


# ─── batch_score_layouts ────────────────────────────────────────────────────

class TestBatchScoreLayoutsExtra:
    def test_empty(self):
        assert batch_score_layouts([]) == []

    def test_multiple(self):
        results = batch_score_layouts([_grid_frags(), [_pf()]])
        assert len(results) == 2
        assert all(isinstance(r, LayoutScoreResult) for r in results)

    def test_preserves_order(self):
        cfg = LayoutScorerConfig(canvas_w=100, canvas_h=100)
        results = batch_score_layouts([_grid_frags(), []], cfg)
        assert results[0].n_fragments == 4
        assert results[1].n_fragments == 0
