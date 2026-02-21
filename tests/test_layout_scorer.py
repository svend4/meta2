"""Тесты для puzzle_reconstruction.verification.layout_scorer."""
import pytest

from puzzle_reconstruction.verification.layout_scorer import (
    LayoutScoreResult,
    LayoutScorerConfig,
    PlacedFragment,
    batch_score_layouts,
    compute_coverage,
    compute_overlap_ratio,
    compute_uniformity,
    rank_layouts,
    score_layout,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frag(fid=0, x=0, y=0, w=10, h=10, score=0.5):
    return PlacedFragment(fragment_id=fid, x=x, y=y, w=w, h=h, score=score)


def _cfg(canvas_w=100, canvas_h=100):
    return LayoutScorerConfig(canvas_w=canvas_w, canvas_h=canvas_h)


# ─── TestPlacedFragment ───────────────────────────────────────────────────────

class TestPlacedFragment:
    def test_basic_construction(self):
        f = _frag()
        assert f.fragment_id == 0
        assert f.x == 0
        assert f.y == 0
        assert f.w == 10
        assert f.h == 10

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=-1, x=0, y=0, w=10, h=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=-1, y=0, w=10, h=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=-1, w=10, h=10)

    def test_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, w=10, h=10, score=-0.1)

    def test_x2_property(self):
        f = _frag(x=5, w=20)
        assert f.x2 == 25

    def test_y2_property(self):
        f = _frag(y=3, h=15)
        assert f.y2 == 18

    def test_area_property(self):
        f = _frag(w=10, h=20)
        assert f.area == 200

    def test_center_property(self):
        f = _frag(x=0, y=0, w=10, h=20)
        cx, cy = f.center
        assert abs(cx - 5.0) < 1e-9
        assert abs(cy - 10.0) < 1e-9


# ─── TestLayoutScorerConfig ───────────────────────────────────────────────────

class TestLayoutScorerConfig:
    def test_defaults(self):
        cfg = LayoutScorerConfig()
        assert cfg.canvas_w == 512
        assert cfg.canvas_h == 512

    def test_canvas_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(canvas_w=0)

    def test_canvas_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(canvas_h=0)

    def test_negative_overlap_penalty_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(overlap_penalty=-1.0)

    def test_negative_coverage_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(coverage_weight=-0.1)

    def test_negative_uniformity_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(uniformity_weight=-0.1)

    def test_negative_score_weight_raises(self):
        with pytest.raises(ValueError):
            LayoutScorerConfig(score_weight=-0.1)


# ─── TestLayoutScoreResult ────────────────────────────────────────────────────

class TestLayoutScoreResult:
    def _make(self, total=0.8, coverage=0.7, overlap=0.05,
              uniformity=0.9, mean_frag=0.6, n=3):
        return LayoutScoreResult(
            total_score=total,
            coverage=coverage,
            overlap_ratio=overlap,
            uniformity=uniformity,
            mean_frag_score=mean_frag,
            n_fragments=n,
        )

    def test_basic_construction(self):
        r = self._make()
        assert r.n_fragments == 3

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=-0.1, coverage=0.5,
                              overlap_ratio=0.0, uniformity=1.0,
                              mean_frag_score=0.5, n_fragments=1)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=0.5, coverage=1.1,
                              overlap_ratio=0.0, uniformity=1.0,
                              mean_frag_score=0.5, n_fragments=1)

    def test_overlap_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=0.5, coverage=0.5,
                              overlap_ratio=-0.1, uniformity=1.0,
                              mean_frag_score=0.5, n_fragments=1)

    def test_uniformity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=0.5, coverage=0.5,
                              overlap_ratio=0.0, uniformity=1.5,
                              mean_frag_score=0.5, n_fragments=1)

    def test_negative_mean_frag_score_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=0.5, coverage=0.5,
                              overlap_ratio=0.0, uniformity=0.5,
                              mean_frag_score=-0.1, n_fragments=1)

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            LayoutScoreResult(total_score=0.5, coverage=0.5,
                              overlap_ratio=0.0, uniformity=0.5,
                              mean_frag_score=0.5, n_fragments=-1)

    def test_quality_level_poor(self):
        r = self._make(total=0.1)
        assert r.quality_level == "poor"

    def test_quality_level_fair(self):
        r = self._make(total=0.4)
        assert r.quality_level == "fair"

    def test_quality_level_good(self):
        r = self._make(total=0.6)
        assert r.quality_level == "good"

    def test_quality_level_excellent(self):
        r = self._make(total=0.9)
        assert r.quality_level == "excellent"


# ─── TestComputeCoverage ──────────────────────────────────────────────────────

class TestComputeCoverage:
    def test_empty_fragments_zero(self):
        assert compute_coverage([], 100, 100) == 0.0

    def test_full_canvas_one(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=100, h=100)
        assert abs(compute_coverage([f], 100, 100) - 1.0) < 1e-9

    def test_partial_coverage(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=50, h=100)
        cov = compute_coverage([f], 100, 100)
        assert abs(cov - 0.5) < 1e-9

    def test_no_double_count(self):
        # Two overlapping fragments covering the whole canvas
        f1 = PlacedFragment(fragment_id=0, x=0, y=0, w=100, h=100)
        f2 = PlacedFragment(fragment_id=1, x=0, y=0, w=100, h=100)
        cov = compute_coverage([f1, f2], 100, 100)
        assert abs(cov - 1.0) < 1e-9

    def test_out_of_bounds_clipped(self):
        f = PlacedFragment(fragment_id=0, x=90, y=0, w=100, h=100)
        cov = compute_coverage([f], 100, 100)
        assert cov <= 1.0

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([], 0, 100)


# ─── TestComputeOverlapRatio ──────────────────────────────────────────────────

class TestComputeOverlapRatio:
    def test_no_overlap(self):
        f1 = _frag(fid=0, x=0, y=0, w=10, h=10)
        f2 = _frag(fid=1, x=20, y=0, w=10, h=10)
        assert compute_overlap_ratio([f1, f2]) == 0.0

    def test_full_overlap(self):
        f1 = _frag(fid=0, x=0, y=0, w=10, h=10)
        f2 = _frag(fid=1, x=0, y=0, w=10, h=10)
        ratio = compute_overlap_ratio([f1, f2])
        assert ratio > 0.0

    def test_single_fragment_no_overlap(self):
        assert compute_overlap_ratio([_frag()]) == 0.0

    def test_empty_no_overlap(self):
        assert compute_overlap_ratio([]) == 0.0

    def test_ratio_in_0_1(self):
        frags = [_frag(fid=i, x=i * 3, y=0) for i in range(5)]
        r = compute_overlap_ratio(frags)
        assert 0.0 <= r <= 1.0


# ─── TestComputeUniformity ────────────────────────────────────────────────────

class TestComputeUniformity:
    def test_single_fragment_returns_one(self):
        assert compute_uniformity([_frag()], 100, 100) == 1.0

    def test_empty_returns_one(self):
        assert compute_uniformity([], 100, 100) == 1.0

    def test_uniformly_distributed(self):
        # Four fragments at corners — should have low uniformity (uneven distances)
        frags = [
            _frag(fid=0, x=0, y=0),
            _frag(fid=1, x=90, y=0),
            _frag(fid=2, x=0, y=90),
            _frag(fid=3, x=90, y=90),
        ]
        u = compute_uniformity(frags, 100, 100)
        assert 0.0 <= u <= 1.0

    def test_result_in_0_1(self):
        frags = [_frag(fid=i, x=i * 20, y=i * 10) for i in range(4)]
        u = compute_uniformity(frags, 100, 100)
        assert 0.0 <= u <= 1.0


# ─── TestScoreLayout ──────────────────────────────────────────────────────────

class TestScoreLayout:
    def test_returns_layout_score_result(self):
        r = score_layout([_frag()])
        assert isinstance(r, LayoutScoreResult)

    def test_empty_fragments_zero_total(self):
        r = score_layout([])
        assert r.total_score == 0.0
        assert r.n_fragments == 0

    def test_n_fragments_correct(self):
        frags = [_frag(fid=i) for i in range(5)]
        r = score_layout(frags, _cfg())
        assert r.n_fragments == 5

    def test_total_in_0_1(self):
        frags = [_frag(fid=i, x=i * 10, y=0, score=0.8) for i in range(5)]
        r = score_layout(frags, _cfg())
        assert 0.0 <= r.total_score <= 1.0

    def test_high_score_frags_better(self):
        frags_good = [_frag(fid=i, x=i * 10, y=0, score=0.9) for i in range(4)]
        frags_bad = [_frag(fid=i, x=i * 10, y=0, score=0.1) for i in range(4)]
        r_good = score_layout(frags_good, _cfg())
        r_bad = score_layout(frags_bad, _cfg())
        assert r_good.total_score >= r_bad.total_score

    def test_coverage_computed(self):
        f = PlacedFragment(fragment_id=0, x=0, y=0, w=100, h=100)
        r = score_layout([f], _cfg(100, 100))
        assert abs(r.coverage - 1.0) < 1e-9

    def test_overlap_ratio_computed(self):
        f1 = _frag(fid=0, x=0, y=0, w=10, h=10)
        f2 = _frag(fid=1, x=5, y=5, w=10, h=10)
        r = score_layout([f1, f2], _cfg())
        assert r.overlap_ratio > 0.0

    def test_default_config_used(self):
        r = score_layout([_frag()])
        assert isinstance(r, LayoutScoreResult)


# ─── TestRankLayouts ──────────────────────────────────────────────────────────

class TestRankLayouts:
    def test_returns_list(self):
        r = rank_layouts([[_frag()], [_frag(score=0.9)]])
        assert isinstance(r, list)

    def test_length_matches(self):
        layouts = [[_frag(fid=i)] for i in range(4)]
        r = rank_layouts(layouts, _cfg())
        assert len(r) == 4

    def test_descending_order(self):
        layouts = [[_frag(score=0.1)], [_frag(score=0.9)], [_frag(score=0.5)]]
        ranked = rank_layouts(layouts, _cfg())
        scores = [res.total_score for _, res in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_original_indices_preserved(self):
        layouts = [[_frag(score=0.1)], [_frag(score=0.9)]]
        ranked = rank_layouts(layouts, _cfg())
        indices = [idx for idx, _ in ranked]
        assert set(indices) == {0, 1}

    def test_empty_list(self):
        assert rank_layouts([]) == []


# ─── TestBatchScoreLayouts ────────────────────────────────────────────────────

class TestBatchScoreLayouts:
    def test_returns_list(self):
        r = batch_score_layouts([[_frag()], [_frag(score=0.8)]])
        assert isinstance(r, list)

    def test_length_matches(self):
        layouts = [[_frag(fid=i)] for i in range(3)]
        r = batch_score_layouts(layouts, _cfg())
        assert len(r) == 3

    def test_each_is_layout_score_result(self):
        r = batch_score_layouts([[_frag()]], _cfg())
        assert all(isinstance(x, LayoutScoreResult) for x in r)

    def test_empty_list(self):
        assert batch_score_layouts([]) == []
