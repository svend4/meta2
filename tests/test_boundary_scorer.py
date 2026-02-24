"""Тесты для puzzle_reconstruction.scoring.boundary_scorer."""
import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=None, seed=0):
    if val is not None:
        return np.full((h, w), val, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=64, w=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestBoundarySide ─────────────────────────────────────────────────────────

class TestBoundarySide:
    def test_four_values(self):
        sides = list(BoundarySide)
        assert len(sides) == 4

    def test_string_values(self):
        assert BoundarySide.TOP == "top"
        assert BoundarySide.BOTTOM == "bottom"
        assert BoundarySide.LEFT == "left"
        assert BoundarySide.RIGHT == "right"

    def test_opposite_top_bottom(self):
        assert BoundarySide.TOP.opposite() == BoundarySide.BOTTOM

    def test_opposite_bottom_top(self):
        assert BoundarySide.BOTTOM.opposite() == BoundarySide.TOP

    def test_opposite_left_right(self):
        assert BoundarySide.LEFT.opposite() == BoundarySide.RIGHT

    def test_opposite_right_left(self):
        assert BoundarySide.RIGHT.opposite() == BoundarySide.LEFT

    def test_opposite_involutive(self):
        for side in BoundarySide:
            assert side.opposite().opposite() == side


# ─── TestBoundaryScore ────────────────────────────────────────────────────────

class TestBoundaryScore:
    def _make(self, i=0.8, g=0.7, c=0.9, agg=0.8):
        return BoundaryScore(
            intensity_diff=i, gradient_score=g,
            color_score=c, aggregate=agg
        )

    def test_basic_creation(self):
        bs = self._make()
        assert bs.aggregate == pytest.approx(0.8)

    def test_intensity_below_zero_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=-0.1, gradient_score=0.5,
                          color_score=0.5, aggregate=0.5)

    def test_intensity_above_one_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=1.1, gradient_score=0.5,
                          color_score=0.5, aggregate=0.5)

    def test_gradient_below_zero_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=-0.1,
                          color_score=0.5, aggregate=0.5)

    def test_color_above_one_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=1.1, aggregate=0.5)

    def test_aggregate_above_one_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=0.5, aggregate=1.1)

    def test_boundary_values_valid(self):
        bs = BoundaryScore(intensity_diff=0.0, gradient_score=0.0,
                           color_score=0.0, aggregate=0.0)
        assert bs.aggregate == 0.0
        bs2 = BoundaryScore(intensity_diff=1.0, gradient_score=1.0,
                            color_score=1.0, aggregate=1.0)
        assert bs2.aggregate == 1.0

    def test_default_sides(self):
        bs = self._make()
        assert bs.side1 == BoundarySide.RIGHT
        assert bs.side2 == BoundarySide.LEFT


# ─── TestScoringConfig ────────────────────────────────────────────────────────

class TestScoringConfig:
    def test_default_values(self):
        cfg = ScoringConfig()
        assert cfg.strip_width == 4
        assert cfg.w_intensity == pytest.approx(0.4)

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(strip_width=0)

    def test_negative_w_intensity_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_intensity=-0.1)

    def test_negative_w_gradient_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_gradient=-0.1)

    def test_negative_w_color_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_color=-0.1)

    def test_weights_sum_to_one_when_normalized(self):
        cfg = ScoringConfig(w_intensity=2.0, w_gradient=1.0,
                            w_color=1.0, normalize_weights=True)
        wi, wg, wc = cfg.weights
        assert wi + wg + wc == pytest.approx(1.0, abs=1e-9)

    def test_weights_not_normalized_when_flag_false(self):
        cfg = ScoringConfig(w_intensity=0.4, w_gradient=0.3,
                            w_color=0.3, normalize_weights=False)
        wi, wg, wc = cfg.weights
        assert wi == pytest.approx(0.4)


# ─── TestIntensityCompatibility ───────────────────────────────────────────────

class TestIntensityCompatibility:
    def test_identical_strips_returns_one(self):
        s = _gray(4, 64, val=128)
        score = intensity_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_opposite_extremes_near_zero(self):
        s1 = _gray(4, 64, val=0)
        s2 = _gray(4, 64, val=255)
        score = intensity_compatibility(s1, s2)
        assert score < 0.05

    def test_result_in_range(self):
        s1 = _gray(4, 64, seed=0)
        s2 = _gray(4, 64, seed=1)
        score = intensity_compatibility(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            intensity_compatibility(_gray(4, 64), _gray(8, 32))

    def test_color_strips(self):
        s = _color(4, 64)
        score = intensity_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-6)


# ─── TestGradientCompatibility ────────────────────────────────────────────────

class TestGradientCompatibility:
    def test_identical_strips_near_one(self):
        s = _gray(4, 64, seed=7)
        score = gradient_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_result_in_range(self):
        score = gradient_compatibility(_gray(4, 64, seed=0), _gray(4, 64, seed=1))
        assert 0.0 <= score <= 1.0

    def test_constant_strips(self):
        # Постоянная полоса: нулевой градиент → ZNCC неопределён, должен не упасть
        s = _gray(4, 64, val=128)
        score = gradient_compatibility(s, s)
        assert isinstance(score, float)

    def test_color_strips(self):
        s = _color(4, 64)
        score = gradient_compatibility(s, s)
        assert 0.0 <= score <= 1.0


# ─── TestColorCompatibility ───────────────────────────────────────────────────

class TestColorCompatibility:
    def test_identical_image_near_one(self):
        img = _gray(4, 64, seed=3)
        score = color_compatibility(img, img)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_result_in_range(self):
        score = color_compatibility(_gray(4, 64, seed=0), _gray(4, 64, seed=1))
        assert 0.0 <= score <= 1.0

    def test_color_image_identical(self):
        img = _color(4, 64)
        score = color_compatibility(img, img)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_white_vs_black_low_score(self):
        s1 = _gray(4, 64, val=0)
        s2 = _gray(4, 64, val=255)
        score = color_compatibility(s1, s2)
        # Black vs white differ only in L channel; A and B channels are identical
        # (both neutral gray). After 3-channel normalization, max score for L=0 is 2/3.
        assert score < 0.8


# ─── TestScoreBoundary ────────────────────────────────────────────────────────

class TestScoreBoundary:
    def test_returns_boundary_score(self):
        img = _gray()
        bs = score_boundary(img, img)
        assert isinstance(bs, BoundaryScore)

    def test_aggregate_in_range(self):
        bs = score_boundary(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= bs.aggregate <= 1.0

    def test_identical_images_high_aggregate(self):
        img = _gray()
        bs = score_boundary(img, img)
        assert bs.aggregate >= 0.5

    def test_side1_stored(self):
        img = _gray()
        bs = score_boundary(img, img, side1=BoundarySide.TOP)
        assert bs.side1 == BoundarySide.TOP

    def test_side2_defaults_to_opposite(self):
        img = _gray()
        bs = score_boundary(img, img, side1=BoundarySide.LEFT)
        assert bs.side2 == BoundarySide.RIGHT

    def test_all_sides(self):
        img = _gray()
        for side in BoundarySide:
            bs = score_boundary(img, img, side1=side)
            assert isinstance(bs, BoundaryScore)

    def test_color_images(self):
        img = _color()
        bs = score_boundary(img, img)
        assert 0.0 <= bs.aggregate <= 1.0

    def test_custom_config(self):
        cfg = ScoringConfig(strip_width=8, w_intensity=1.0,
                            w_gradient=0.0, w_color=0.0)
        img = _gray()
        bs = score_boundary(img, img, cfg=cfg)
        assert bs.aggregate == pytest.approx(bs.intensity_diff, abs=1e-6)


# ─── TestScoreMatrix ──────────────────────────────────────────────────────────

class TestScoreMatrix:
    def test_shape(self):
        images = [_gray(seed=i) for i in range(4)]
        mat = score_matrix(images)
        assert mat.shape == (4, 4)

    def test_diagonal_zero(self):
        images = [_gray(seed=i) for i in range(3)]
        mat = score_matrix(images)
        for i in range(3):
            assert mat[i, i] == pytest.approx(0.0)

    def test_dtype_float64(self):
        images = [_gray(seed=i) for i in range(2)]
        mat = score_matrix(images)
        assert mat.dtype == np.float64

    def test_values_in_range(self):
        images = [_gray(seed=i) for i in range(3)]
        mat = score_matrix(images)
        assert (mat >= 0.0).all()
        assert (mat <= 1.0 + 1e-9).all()

    def test_single_image(self):
        mat = score_matrix([_gray()])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_empty_list(self):
        mat = score_matrix([])
        assert mat.shape == (0, 0)


# ─── TestBatchScoreBoundaries ─────────────────────────────────────────────────

class TestBatchScoreBoundaries:
    def test_returns_list(self):
        pairs = [(_gray(seed=0), _gray(seed=1))]
        result = batch_score_boundaries(pairs)
        assert isinstance(result, list)

    def test_correct_length(self):
        pairs = [(_gray(seed=i), _gray(seed=i + 1)) for i in range(4)]
        result = batch_score_boundaries(pairs)
        assert len(result) == 4

    def test_empty_list(self):
        result = batch_score_boundaries([])
        assert result == []

    def test_each_boundary_score(self):
        pairs = [(_gray(), _color())]
        result = batch_score_boundaries(pairs)
        assert all(isinstance(bs, BoundaryScore) for bs in result)

    def test_aggregate_in_range(self):
        pairs = [(_gray(seed=i), _gray(seed=i + 5)) for i in range(3)]
        result = batch_score_boundaries(pairs)
        for bs in result:
            assert 0.0 <= bs.aggregate <= 1.0
