"""Extra tests for puzzle_reconstruction/scoring/boundary_scorer.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.scoring.boundary_scorer import (
    BoundarySide,
    BoundaryScore,
    ScoringConfig,
    intensity_compatibility,
    gradient_compatibility,
    color_compatibility,
    score_boundary,
    score_matrix,
    batch_score_boundaries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=32) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32, val=100) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _score(i=0.5, g=0.5, c=0.5, agg=0.5) -> BoundaryScore:
    return BoundaryScore(intensity_diff=i, gradient_score=g,
                         color_score=c, aggregate=agg)


# ─── BoundarySide ─────────────────────────────────────────────────────────────

class TestBoundarySideExtra:
    def test_top_opposite_bottom(self):
        assert BoundarySide.TOP.opposite() == BoundarySide.BOTTOM

    def test_bottom_opposite_top(self):
        assert BoundarySide.BOTTOM.opposite() == BoundarySide.TOP

    def test_left_opposite_right(self):
        assert BoundarySide.LEFT.opposite() == BoundarySide.RIGHT

    def test_right_opposite_left(self):
        assert BoundarySide.RIGHT.opposite() == BoundarySide.LEFT

    def test_double_opposite_is_self(self):
        for side in BoundarySide:
            assert side.opposite().opposite() == side

    def test_string_values(self):
        assert BoundarySide.TOP.value == "top"
        assert BoundarySide.RIGHT.value == "right"


# ─── BoundaryScore ────────────────────────────────────────────────────────────

class TestBoundaryScoreExtra:
    def test_stores_intensity_diff(self):
        s = _score(i=0.7)
        assert s.intensity_diff == pytest.approx(0.7)

    def test_stores_gradient_score(self):
        s = _score(g=0.6)
        assert s.gradient_score == pytest.approx(0.6)

    def test_stores_color_score(self):
        s = _score(c=0.4)
        assert s.color_score == pytest.approx(0.4)

    def test_stores_aggregate(self):
        s = _score(agg=0.8)
        assert s.aggregate == pytest.approx(0.8)

    def test_intensity_diff_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=1.5, gradient_score=0.5,
                          color_score=0.5, aggregate=0.5)

    def test_gradient_score_negative_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=-0.1,
                          color_score=0.5, aggregate=0.5)

    def test_aggregate_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=0.5, aggregate=1.5)

    def test_default_sides(self):
        s = _score()
        assert s.side1 == BoundarySide.RIGHT
        assert s.side2 == BoundarySide.LEFT

    def test_custom_sides(self):
        s = BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=0.5, aggregate=0.5,
                          side1=BoundarySide.TOP, side2=BoundarySide.BOTTOM)
        assert s.side1 == BoundarySide.TOP


# ─── ScoringConfig ────────────────────────────────────────────────────────────

class TestScoringConfigExtra:
    def test_default_strip_width(self):
        assert ScoringConfig().strip_width == 4

    def test_default_weights(self):
        cfg = ScoringConfig()
        assert cfg.w_intensity == pytest.approx(0.4)
        assert cfg.w_gradient == pytest.approx(0.3)
        assert cfg.w_color == pytest.approx(0.3)

    def test_default_normalize_weights(self):
        assert ScoringConfig().normalize_weights is True

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(strip_width=0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_intensity=-0.1)

    def test_weights_property_sums_to_one(self):
        cfg = ScoringConfig(w_intensity=0.5, w_gradient=0.3, w_color=0.2,
                             normalize_weights=True)
        wi, wg, wc = cfg.weights
        assert wi + wg + wc == pytest.approx(1.0)

    def test_weights_without_normalization(self):
        cfg = ScoringConfig(w_intensity=0.5, w_gradient=0.3, w_color=0.2,
                             normalize_weights=False)
        wi, wg, wc = cfg.weights
        assert wi == pytest.approx(0.5)

    def test_custom_strip_width(self):
        cfg = ScoringConfig(strip_width=8)
        assert cfg.strip_width == 8


# ─── intensity_compatibility ─────────────────────────────────────────────────

class TestIntensityCompatibilityExtra:
    def test_identical_strips_one(self):
        s = np.full((4, 4), 128, dtype=np.uint8)
        assert intensity_compatibility(s, s) == pytest.approx(1.0)

    def test_opposite_strips_zero(self):
        s1 = np.zeros((4, 4), dtype=np.uint8)
        s2 = np.full((4, 4), 255, dtype=np.uint8)
        result = intensity_compatibility(s1, s2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_result_in_range(self):
        s1 = np.full((4, 8), 100, dtype=np.uint8)
        s2 = np.full((4, 8), 150, dtype=np.uint8)
        result = intensity_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_shape_mismatch_raises(self):
        s1 = np.zeros((4, 4), dtype=np.uint8)
        s2 = np.zeros((4, 8), dtype=np.uint8)
        with pytest.raises(ValueError):
            intensity_compatibility(s1, s2)

    def test_returns_float(self):
        s = np.zeros((4, 4), dtype=np.uint8)
        assert isinstance(intensity_compatibility(s, s), float)


# ─── gradient_compatibility ──────────────────────────────────────────────────

class TestGradientCompatibilityExtra:
    def test_returns_float(self):
        s = _gray(4, 8)
        assert isinstance(gradient_compatibility(s, s), float)

    def test_result_in_range(self):
        s1 = _gray(4, 8, val=100)
        s2 = _gray(4, 8, val=200)
        result = gradient_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_identical_uniform_strips(self):
        s = _gray(4, 8, val=128)
        result = gradient_compatibility(s, s)
        assert 0.0 <= result <= 1.0

    def test_bgr_input_ok(self):
        s = _bgr(4, 8)
        result = gradient_compatibility(s, s)
        assert 0.0 <= result <= 1.0


# ─── color_compatibility ─────────────────────────────────────────────────────

class TestColorCompatibilityExtra:
    def test_returns_float(self):
        s = _gray(4, 8)
        assert isinstance(color_compatibility(s, s), float)

    def test_result_in_range(self):
        s1 = _gray(4, 8, val=100)
        s2 = _gray(4, 8, val=200)
        result = color_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_identical_strips_high(self):
        s = _gray(4, 8, val=128)
        result = color_compatibility(s, s)
        assert result > 0.5

    def test_bgr_input_ok(self):
        s = _bgr(4, 8)
        result = color_compatibility(s, s)
        assert 0.0 <= result <= 1.0


# ─── score_boundary ───────────────────────────────────────────────────────────

class TestScoreBoundaryExtra:
    def test_returns_boundary_score(self):
        r = score_boundary(_gray(), _gray())
        assert isinstance(r, BoundaryScore)

    def test_aggregate_in_range(self):
        r = score_boundary(_gray(), _ramp())
        assert 0.0 <= r.aggregate <= 1.0

    def test_default_sides(self):
        r = score_boundary(_gray(), _gray())
        assert r.side1 == BoundarySide.RIGHT
        assert r.side2 == BoundarySide.LEFT

    def test_auto_opposite_side(self):
        r = score_boundary(_gray(), _gray(), side1=BoundarySide.TOP)
        assert r.side2 == BoundarySide.BOTTOM

    def test_custom_sides(self):
        r = score_boundary(_gray(), _gray(),
                           side1=BoundarySide.LEFT,
                           side2=BoundarySide.RIGHT)
        assert r.side1 == BoundarySide.LEFT
        assert r.side2 == BoundarySide.RIGHT

    def test_identical_images_high_score(self):
        img = _gray(val=128)
        r = score_boundary(img, img.copy())
        assert r.aggregate >= 0.5

    def test_none_cfg_uses_defaults(self):
        r = score_boundary(_gray(), _gray(), cfg=None)
        assert isinstance(r, BoundaryScore)

    def test_bgr_images(self):
        r = score_boundary(_bgr(), _bgr())
        assert 0.0 <= r.aggregate <= 1.0


# ─── score_matrix ─────────────────────────────────────────────────────────────

class TestScoreMatrixExtra:
    def test_returns_ndarray(self):
        imgs = [_gray(), _gray()]
        mat = score_matrix(imgs)
        assert isinstance(mat, np.ndarray)

    def test_shape_n_by_n(self):
        imgs = [_gray(), _gray(), _gray()]
        mat = score_matrix(imgs)
        assert mat.shape == (3, 3)

    def test_diagonal_zero(self):
        imgs = [_gray(), _gray()]
        mat = score_matrix(imgs)
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_values_in_range(self):
        imgs = [_gray(val=100), _gray(val=200)]
        mat = score_matrix(imgs)
        assert np.all((mat >= 0.0) & (mat <= 1.0))

    def test_single_image(self):
        mat = score_matrix([_gray()])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.0)


# ─── batch_score_boundaries ──────────────────────────────────────────────────

class TestBatchScoreBoundariesExtra:
    def test_returns_list(self):
        result = batch_score_boundaries([(_gray(), _gray())])
        assert isinstance(result, list)

    def test_length_matches(self):
        pairs = [(_gray(), _gray()), (_ramp(), _gray())]
        result = batch_score_boundaries(pairs)
        assert len(result) == 2

    def test_empty_pairs(self):
        assert batch_score_boundaries([]) == []

    def test_each_element_is_boundary_score(self):
        result = batch_score_boundaries([(_gray(), _gray())])
        assert isinstance(result[0], BoundaryScore)

    def test_aggregate_in_range(self):
        result = batch_score_boundaries([(_gray(), _ramp())])
        assert 0.0 <= result[0].aggregate <= 1.0
