"""Extra tests for puzzle_reconstruction/verification/layout_scorer.py"""
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

def _frag(fid=0, x=0, y=0, w=10, h=10, score=0.5, angle=0.0):
    return PlacedFragment(fragment_id=fid, x=x, y=y, w=w, h=h,
                          score=score, angle=angle)


def _cfg(**kw):
    return LayoutScorerConfig(**kw)


# ─── TestPlacedFragmentExtra ──────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_angle_stored(self):
        f = _frag(angle=45.0)
        assert f.angle == pytest.approx(45.0)

    def test_angle_default_zero(self):
        f = _frag()
        assert f.angle == pytest.approx(0.0)

    def test_score_default_zero(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=5, h=5)
        assert f.score == pytest.approx(0.0)

    def test_center_non_zero_position(self):
        f = _frag(x=10, y=20, w=20, h=10)
        cx, cy = f.center
        assert cx == pytest.approx(20.0)
        assert cy == pytest.approx(25.0)

    def test_area_large_fragment(self):
        f = _frag(w=100, h=200)
        assert f.area == 20000

    def test_x2_y2_correct(self):
        f = _frag(x=5, y=3, w=15, h=7)
        assert f.x2 == 20
        assert f.y2 == 10


# ─── TestLayoutScorerConfigExtra ─────────────────────────────────────────────

class TestLayoutScorerConfigExtra:
    def test_custom_canvas_size(self):
        cfg = _cfg(canvas_w=800, canvas_h=600)
        assert cfg.canvas_w == 800
        assert cfg.canvas_h == 600

    def test_zero_overlap_penalty_valid(self):
        cfg = _cfg(overlap_penalty=0.0)
        assert cfg.overlap_penalty == pytest.approx(0.0)

    def test_zero_score_weight_valid(self):
        cfg = _cfg(score_weight=0.0)
        assert cfg.score_weight == pytest.approx(0.0)

    def test_large_weights_valid(self):
        cfg = _cfg(coverage_weight=10.0, uniformity_weight=5.0)
        assert cfg.coverage_weight == pytest.approx(10.0)

    def test_canvas_1x1_valid(self):
        cfg = _cfg(canvas_w=1, canvas_h=1)
        assert cfg.canvas_w == 1


# ─── TestLayoutScoreResultExtra ──────────────────────────────────────────────

class TestLayoutScoreResultExtra:
    def _make(self, total=0.5, coverage=0.5, overlap=0.0,
              uniformity=1.0, mean_frag=0.5, n=1):
        return LayoutScoreResult(
            total_score=total,
            coverage=coverage,
            overlap_ratio=overlap,
            uniformity=uniformity,
            mean_frag_score=mean_frag,
            n_fragments=n,
        )

    def test_quality_poor_at_0(self):
        r = self._make(total=0.0)
        assert r.quality_level == "poor"

    def test_quality_fair_at_025(self):
        r = self._make(total=0.25)
        assert r.quality_level == "fair"

    def test_quality_good_at_05(self):
        r = self._make(total=0.5)
        assert r.quality_level == "good"

    def test_quality_excellent_at_075(self):
        r = self._make(total=0.75)
        assert r.quality_level == "excellent"

    def test_quality_excellent_at_1(self):
        r = self._make(total=1.0)
        assert r.quality_level == "excellent"

    def test_n_fragments_zero_valid(self):
        r = self._make(n=0)
        assert r.n_fragments == 0

    def test_total_score_zero_valid(self):
        r = self._make(total=0.0)
        assert r.total_score == pytest.approx(0.0)

    def test_uniformity_zero_valid(self):
        r = self._make(uniformity=0.0)
        assert r.uniformity == pytest.approx(0.0)


# ─── TestComputeCoverageExtra ─────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_two_non_overlapping(self):
        f1 = PlacedFragment(fragment_id=0, x=0, y=0, w=50, h=100)
        f2 = PlacedFragment(fragment_id=1, x=50, y=0, w=50, h=100)
        cov = compute_coverage([f1, f2], 100, 100)
        assert cov == pytest.approx(1.0, abs=1e-9)

    def test_small_fragment_in_large_canvas(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10)
        cov = compute_coverage([f], 100, 100)
        assert cov == pytest.approx(0.01, abs=1e-6)

    def test_fragment_entirely_outside(self):
        # Fragment starts at x=110 on canvas_w=100 → no coverage
        f = PlacedFragment(fragment_id=0, x=110, y=0, w=10, h=10)
        cov = compute_coverage([f], 100, 100)
        assert cov == pytest.approx(0.0)

    def test_canvas_h_raises_zero(self):
        with pytest.raises(ValueError):
            compute_coverage([], 100, 0)


# ─── TestComputeOverlapRatioExtra ─────────────────────────────────────────────

class TestComputeOverlapRatioExtra:
    def test_partial_overlap(self):
        f1 = PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10)
        f2 = PlacedFragment(fragment_id=1, x=5, y=5, w=10, h=10)
        ratio = compute_overlap_ratio([f1, f2])
        assert 0.0 < ratio <= 1.0

    def test_touching_no_overlap(self):
        f1 = PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10)
        f2 = PlacedFragment(fragment_id=1, x=10, y=0, w=10, h=10)
        # x2 of f1 = 10, x of f2 = 10 → touching, not overlapping
        ratio = compute_overlap_ratio([f1, f2])
        assert ratio == pytest.approx(0.0)

    def test_three_all_same_position(self):
        frags = [PlacedFragment(fragment_id=i, x=0, y=0, w=10, h=10)
                 for i in range(3)]
        ratio = compute_overlap_ratio(frags)
        assert 0.0 < ratio <= 1.0


# ─── TestComputeUniformityExtra ───────────────────────────────────────────────

class TestComputeUniformityExtra:
    def test_two_fragments_symmetric(self):
        # Placed symmetrically around canvas center → should have some uniformity
        f1 = PlacedFragment(fragment_id=0, x=40, y=45, w=10, h=10)
        f2 = PlacedFragment(fragment_id=1, x=50, y=45, w=10, h=10)
        u = compute_uniformity([f1, f2], 100, 100)
        assert 0.0 <= u <= 1.0

    def test_all_at_same_position(self):
        frags = [PlacedFragment(fragment_id=i, x=10, y=10, w=5, h=5)
                 for i in range(4)]
        u = compute_uniformity(frags, 100, 100)
        # std of identical distances = 0 → uniformity = 1
        assert u == pytest.approx(1.0, abs=1e-9)

    def test_result_in_0_1_always(self):
        frags = [_frag(fid=i, x=i * 30, y=i * 20) for i in range(5)]
        u = compute_uniformity(frags, 200, 200)
        assert 0.0 <= u <= 1.0


# ─── TestScoreLayoutExtra ─────────────────────────────────────────────────────

class TestScoreLayoutExtra:
    def test_coverage_zero_when_all_outside(self):
        f = PlacedFragment(fragment_id=0, x=200, y=200, w=10, h=10)
        r = score_layout([f], _cfg(canvas_w=100, canvas_h=100))
        assert r.coverage == pytest.approx(0.0)

    def test_mean_frag_score_computed(self):
        frags = [_frag(fid=i, x=i * 15, score=0.6) for i in range(3)]
        r = score_layout(frags, _cfg())
        assert r.mean_frag_score == pytest.approx(0.6, abs=1e-6)

    def test_zero_weights_no_crash(self):
        cfg = _cfg(coverage_weight=0.0, uniformity_weight=0.0,
                   score_weight=0.0, overlap_penalty=0.0)
        frags = [_frag()]
        r = score_layout(frags, cfg)
        assert 0.0 <= r.total_score <= 1.0

    def test_non_default_canvas(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=200, h=200)
        r = score_layout([f], _cfg(canvas_w=200, canvas_h=200))
        assert r.coverage == pytest.approx(1.0, abs=1e-9)


# ─── TestRankLayoutsExtra ─────────────────────────────────────────────────────

class TestRankLayoutsExtra:
    def test_single_layout(self):
        result = rank_layouts([[_frag()]])
        assert len(result) == 1
        idx, res = result[0]
        assert idx == 0

    def test_three_layouts_all_indices_present(self):
        layouts = [[_frag(score=0.8)], [_frag(score=0.3)], [_frag(score=0.5)]]
        result = rank_layouts(layouts, _cfg())
        indices = [idx for idx, _ in result]
        assert set(indices) == {0, 1, 2}

    def test_best_score_first(self):
        layouts = [[_frag(score=0.1)], [_frag(score=0.9)]]
        result = rank_layouts(layouts, _cfg())
        _, best_res = result[0]
        _, worst_res = result[1]
        assert best_res.total_score >= worst_res.total_score

    def test_results_are_tuples(self):
        result = rank_layouts([[_frag()], [_frag()]])
        assert all(isinstance(item, tuple) and len(item) == 2
                   for item in result)


# ─── TestBatchScoreLayoutsExtra ───────────────────────────────────────────────

class TestBatchScoreLayoutsExtra:
    def test_five_layouts(self):
        layouts = [[_frag(fid=i)] for i in range(5)]
        result = batch_score_layouts(layouts, _cfg())
        assert len(result) == 5

    def test_all_have_n_fragments(self):
        layouts = [[_frag(fid=0), _frag(fid=1, x=20)],
                   [_frag(fid=0)]]
        result = batch_score_layouts(layouts, _cfg())
        assert result[0].n_fragments == 2
        assert result[1].n_fragments == 1

    def test_default_config_applied(self):
        result = batch_score_layouts([[_frag()]])
        assert isinstance(result[0], LayoutScoreResult)
