"""Tests for puzzle_reconstruction/scoring/boundary_scorer.py"""
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=40, w=40, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=40, w=40, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


def make_gradient(h=40, w=40):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_strip(h=4, w=40, value=128):
    return np.full((h, w), value, dtype=np.uint8)


# ─── BoundarySide ─────────────────────────────────────────────────────────────

class TestBoundarySide:
    def test_opposite_top_bottom(self):
        assert BoundarySide.TOP.opposite() == BoundarySide.BOTTOM

    def test_opposite_bottom_top(self):
        assert BoundarySide.BOTTOM.opposite() == BoundarySide.TOP

    def test_opposite_left_right(self):
        assert BoundarySide.LEFT.opposite() == BoundarySide.RIGHT

    def test_opposite_right_left(self):
        assert BoundarySide.RIGHT.opposite() == BoundarySide.LEFT

    def test_double_opposite_identity(self):
        for side in BoundarySide:
            assert side.opposite().opposite() == side

    def test_string_values(self):
        assert BoundarySide.TOP == "top"
        assert BoundarySide.RIGHT == "right"


# ─── BoundaryScore ────────────────────────────────────────────────────────────

class TestBoundaryScore:
    def test_basic_creation(self):
        bs = BoundaryScore(
            intensity_diff=0.8, gradient_score=0.7,
            color_score=0.9, aggregate=0.8
        )
        assert bs.aggregate == pytest.approx(0.8)

    def test_boundary_values_ok(self):
        bs = BoundaryScore(
            intensity_diff=0.0, gradient_score=1.0,
            color_score=0.0, aggregate=1.0
        )
        assert bs.gradient_score == 1.0

    def test_intensity_diff_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=1.5, gradient_score=0.5,
                          color_score=0.5, aggregate=0.5)

    def test_gradient_score_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=-0.1,
                          color_score=0.5, aggregate=0.5)

    def test_color_score_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=2.0, aggregate=0.5)

    def test_aggregate_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                          color_score=0.5, aggregate=-0.1)

    def test_default_sides(self):
        bs = BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                           color_score=0.5, aggregate=0.5)
        assert bs.side1 == BoundarySide.RIGHT
        assert bs.side2 == BoundarySide.LEFT

    def test_custom_sides(self):
        bs = BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                           color_score=0.5, aggregate=0.5,
                           side1=BoundarySide.TOP, side2=BoundarySide.BOTTOM)
        assert bs.side1 == BoundarySide.TOP


# ─── ScoringConfig ────────────────────────────────────────────────────────────

class TestScoringConfig:
    def test_defaults(self):
        cfg = ScoringConfig()
        assert cfg.strip_width == 4
        assert cfg.w_intensity == pytest.approx(0.4)
        assert cfg.w_gradient == pytest.approx(0.3)
        assert cfg.w_color == pytest.approx(0.3)
        assert cfg.normalize_weights is True

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(strip_width=0)

    def test_negative_w_intensity_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_intensity=-0.1)

    def test_negative_w_gradient_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_gradient=-1.0)

    def test_negative_w_color_raises(self):
        with pytest.raises(ValueError):
            ScoringConfig(w_color=-0.5)

    def test_weights_normalized(self):
        cfg = ScoringConfig(w_intensity=2.0, w_gradient=2.0, w_color=2.0,
                            normalize_weights=True)
        wi, wg, wc = cfg.weights
        assert wi + wg + wc == pytest.approx(1.0)

    def test_weights_not_normalized(self):
        cfg = ScoringConfig(w_intensity=0.5, w_gradient=0.3, w_color=0.2,
                            normalize_weights=False)
        wi, wg, wc = cfg.weights
        assert wi == pytest.approx(0.5)
        assert wg == pytest.approx(0.3)
        assert wc == pytest.approx(0.2)

    def test_strip_width_one_ok(self):
        cfg = ScoringConfig(strip_width=1)
        assert cfg.strip_width == 1


# ─── intensity_compatibility ─────────────────────────────────────────────────

class TestIntensityCompatibility:
    def test_identical_strips(self):
        s = make_strip(value=100)
        assert intensity_compatibility(s, s) == pytest.approx(1.0)

    def test_max_difference(self):
        s1 = make_strip(value=0)
        s2 = make_strip(value=255)
        result = intensity_compatibility(s1, s2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_range(self):
        s1 = make_strip(value=80)
        s2 = make_strip(value=180)
        result = intensity_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_shape_mismatch_raises(self):
        s1 = np.zeros((4, 30), dtype=np.uint8)
        s2 = np.zeros((4, 40), dtype=np.uint8)
        with pytest.raises(ValueError):
            intensity_compatibility(s1, s2)

    def test_close_values_high_score(self):
        s1 = make_strip(value=128)
        s2 = make_strip(value=130)
        result = intensity_compatibility(s1, s2)
        assert result > 0.99

    def test_non_negative(self):
        s1 = make_strip(value=0)
        s2 = make_strip(value=255)
        assert intensity_compatibility(s1, s2) >= 0.0


# ─── gradient_compatibility ──────────────────────────────────────────────────

class TestGradientCompatibility:
    def test_identical_strips(self):
        strip = make_gradient()[:4, :]
        result = gradient_compatibility(strip, strip.copy())
        assert result == pytest.approx(1.0)

    def test_range(self):
        s1 = make_gradient()[:4, :]
        s2 = np.zeros((4, 40), dtype=np.uint8)
        result = gradient_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_output_in_range(self):
        s1 = make_strip(value=100)
        s2 = make_strip(value=200)
        result = gradient_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_constant_strips_gives_half(self):
        """Constant strips → zero Sobel everywhere → zncc=0 → (0+1)/2 = 0.5."""
        s1 = make_strip(value=100)
        s2 = make_strip(value=200)
        result = gradient_compatibility(s1, s2)
        assert result == pytest.approx(0.5)

    def test_non_negative(self):
        s1 = make_gradient()[:4, :]
        s2 = np.flipud(make_gradient()[:4, :])
        result = gradient_compatibility(s1, s2)
        assert result >= 0.0

    def test_bgr_input(self):
        s1 = make_bgr(h=4, w=40)
        s2 = make_bgr(h=4, w=40, value=(100, 100, 100))
        result = gradient_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0


# ─── color_compatibility ─────────────────────────────────────────────────────

class TestColorCompatibility:
    def test_identical_strips(self):
        s = make_strip(value=128)
        result = color_compatibility(s, s.copy())
        assert result == pytest.approx(1.0, abs=0.01)

    def test_range(self):
        s1 = make_strip(value=50)
        s2 = make_strip(value=200)
        result = color_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0

    def test_same_bgr_strips(self):
        s = make_bgr(h=4, w=40)
        result = color_compatibility(s, s.copy())
        assert result > 0.9

    def test_non_negative(self):
        s1 = make_strip(value=0)
        s2 = make_strip(value=255)
        assert color_compatibility(s1, s2) >= 0.0

    def test_gray_and_bgr_accepted(self):
        s1 = make_strip(h=4, w=40, value=128)
        s2 = make_bgr(h=4, w=40, value=(128, 128, 128))
        result = color_compatibility(s1, s2)
        assert 0.0 <= result <= 1.0


# ─── score_boundary ───────────────────────────────────────────────────────────

class TestScoreBoundary:
    def test_returns_boundary_score(self):
        img = make_gray()
        result = score_boundary(img, img)
        assert isinstance(result, BoundaryScore)

    def test_score_in_range(self):
        img1 = make_gray(value=80)
        img2 = make_gray(value=180)
        result = score_boundary(img1, img2)
        assert 0.0 <= result.aggregate <= 1.0

    def test_identical_images_high_score(self):
        img = make_gray(value=150)
        result = score_boundary(img, img)
        assert result.aggregate > 0.5

    def test_default_side2_is_opposite(self):
        img = make_gray()
        result = score_boundary(img, img, side1=BoundarySide.TOP)
        assert result.side2 == BoundarySide.BOTTOM

    def test_explicit_sides(self):
        img = make_gray()
        result = score_boundary(img, img,
                                side1=BoundarySide.LEFT,
                                side2=BoundarySide.RIGHT)
        assert result.side1 == BoundarySide.LEFT
        assert result.side2 == BoundarySide.RIGHT

    def test_custom_config(self):
        img = make_gray()
        cfg = ScoringConfig(strip_width=2)
        result = score_boundary(img, img, cfg=cfg)
        assert isinstance(result, BoundaryScore)

    def test_all_sides(self):
        img = make_gray(50, 50, 128)
        for side in BoundarySide:
            result = score_boundary(img, img, side1=side)
            assert 0.0 <= result.aggregate <= 1.0

    def test_bgr_images(self):
        img = make_bgr()
        result = score_boundary(img, img)
        assert isinstance(result, BoundaryScore)

    def test_component_scores_in_range(self):
        img1 = make_gradient()
        img2 = make_gray(value=128)
        result = score_boundary(img1, img2)
        assert 0.0 <= result.intensity_diff <= 1.0
        assert 0.0 <= result.gradient_score <= 1.0
        assert 0.0 <= result.color_score <= 1.0


# ─── score_matrix ─────────────────────────────────────────────────────────────

class TestScoreMatrix:
    def test_shape(self):
        imgs = [make_gray(value=v) for v in [100, 150, 200]]
        mat = score_matrix(imgs)
        assert mat.shape == (3, 3)

    def test_diagonal_zero(self):
        imgs = [make_gray(value=v) for v in [100, 150, 200]]
        mat = score_matrix(imgs)
        np.testing.assert_array_almost_equal(np.diag(mat), 0.0)

    def test_dtype_float64(self):
        imgs = [make_gray() for _ in range(2)]
        mat = score_matrix(imgs)
        assert mat.dtype == np.float64

    def test_values_in_range(self):
        imgs = [make_gray(value=v) for v in [50, 100, 200]]
        mat = score_matrix(imgs)
        # off-diagonal entries should be in [0, 1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0.0 <= mat[i, j] <= 1.0

    def test_single_image(self):
        imgs = [make_gray()]
        mat = score_matrix(imgs)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0


# ─── batch_score_boundaries ──────────────────────────────────────────────────

class TestBatchScoreBoundaries:
    def test_returns_list(self):
        img = make_gray()
        pairs = [(img, img) for _ in range(3)]
        results = batch_score_boundaries(pairs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_boundary_scores(self):
        img = make_gray()
        pairs = [(img, img) for _ in range(4)]
        results = batch_score_boundaries(pairs)
        assert all(isinstance(r, BoundaryScore) for r in results)

    def test_empty_list(self):
        results = batch_score_boundaries([])
        assert results == []

    def test_custom_config(self):
        img = make_gray()
        cfg = ScoringConfig(strip_width=3)
        results = batch_score_boundaries([(img, img)], cfg=cfg)
        assert isinstance(results[0], BoundaryScore)

    def test_scores_in_range(self):
        imgs = [make_gray(value=v) for v in [50, 100, 200, 150]]
        pairs = [(imgs[i], imgs[j]) for i in range(2) for j in range(2, 4)]
        results = batch_score_boundaries(pairs)
        for r in results:
            assert 0.0 <= r.aggregate <= 1.0
