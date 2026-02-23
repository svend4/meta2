"""Extra tests for puzzle_reconstruction.scoring.boundary_scorer."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=None, seed=0):
    if val is not None:
        return np.full((h, w), val, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=64, w=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestBoundarySideExtra ──────────────────────────────────────────────────

class TestBoundarySideExtra:
    def test_top_string_value(self):
        assert BoundarySide.TOP == "top"

    def test_bottom_string_value(self):
        assert BoundarySide.BOTTOM == "bottom"

    def test_left_string_value(self):
        assert BoundarySide.LEFT == "left"

    def test_right_string_value(self):
        assert BoundarySide.RIGHT == "right"

    def test_all_have_opposite(self):
        for side in BoundarySide:
            assert side.opposite() in list(BoundarySide)

    def test_opposite_involutive_top(self):
        assert BoundarySide.TOP.opposite().opposite() == BoundarySide.TOP

    def test_opposite_involutive_left(self):
        assert BoundarySide.LEFT.opposite().opposite() == BoundarySide.LEFT


# ─── TestBoundaryScoreExtra ─────────────────────────────────────────────────

class TestBoundaryScoreExtra:
    def _make(self, **kw):
        defaults = dict(intensity_diff=0.8, gradient_score=0.7,
                        color_score=0.9, aggregate=0.8)
        defaults.update(kw)
        return BoundaryScore(**defaults)

    def test_zero_all_fields_ok(self):
        bs = self._make(intensity_diff=0.0, gradient_score=0.0,
                        color_score=0.0, aggregate=0.0)
        assert bs.aggregate == pytest.approx(0.0)

    def test_one_all_fields_ok(self):
        bs = self._make(intensity_diff=1.0, gradient_score=1.0,
                        color_score=1.0, aggregate=1.0)
        assert bs.aggregate == pytest.approx(1.0)

    def test_mid_values(self):
        bs = self._make(intensity_diff=0.5, gradient_score=0.5,
                        color_score=0.5, aggregate=0.5)
        assert bs.intensity_diff == pytest.approx(0.5)
        assert bs.gradient_score == pytest.approx(0.5)
        assert bs.color_score == pytest.approx(0.5)

    def test_default_sides_right_left(self):
        bs = self._make()
        assert bs.side1 == BoundarySide.RIGHT
        assert bs.side2 == BoundarySide.LEFT

    def test_custom_side_top_bottom(self):
        bs = BoundaryScore(intensity_diff=0.5, gradient_score=0.5,
                           color_score=0.5, aggregate=0.5,
                           side1=BoundarySide.TOP, side2=BoundarySide.BOTTOM)
        assert bs.side1 == BoundarySide.TOP
        assert bs.side2 == BoundarySide.BOTTOM


# ─── TestScoringConfigExtra ─────────────────────────────────────────────────

class TestScoringConfigExtra:
    def test_default_strip_width(self):
        cfg = ScoringConfig()
        assert cfg.strip_width == 4

    def test_default_w_intensity(self):
        cfg = ScoringConfig()
        assert cfg.w_intensity == pytest.approx(0.4)

    def test_strip_width_10_ok(self):
        cfg = ScoringConfig(strip_width=10)
        assert cfg.strip_width == 10

    def test_w_intensity_zero_ok(self):
        cfg = ScoringConfig(w_intensity=0.0, w_gradient=0.5, w_color=0.5)
        assert cfg.w_intensity == pytest.approx(0.0)

    def test_normalize_weights_true(self):
        cfg = ScoringConfig(w_intensity=1.0, w_gradient=1.0,
                            w_color=1.0, normalize_weights=True)
        wi, wg, wc = cfg.weights
        assert wi + wg + wc == pytest.approx(1.0)

    def test_normalize_weights_false(self):
        cfg = ScoringConfig(w_intensity=0.3, w_gradient=0.3,
                            w_color=0.4, normalize_weights=False)
        wi, wg, wc = cfg.weights
        assert wi == pytest.approx(0.3)
        assert wg == pytest.approx(0.3)
        assert wc == pytest.approx(0.4)

    def test_weights_tuple_len_3(self):
        cfg = ScoringConfig()
        assert len(cfg.weights) == 3


# ─── TestIntensityCompatibilityExtra ────────────────────────────────────────

class TestIntensityCompatibilityExtra:
    def test_gray_same(self):
        s = _gray(4, 32, val=100)
        score = intensity_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_color_same(self):
        s = _color(4, 32, seed=5)
        score = intensity_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_random_pairs_in_range(self):
        for seed in range(5):
            s1 = _gray(4, 32, seed=seed)
            s2 = _gray(4, 32, seed=seed + 10)
            assert 0.0 <= intensity_compatibility(s1, s2) <= 1.0

    def test_all_same_val_vs_random(self):
        s1 = _gray(4, 32, val=128)
        s2 = _gray(4, 32, seed=0)
        score = intensity_compatibility(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_complementary_vals(self):
        s1 = _gray(4, 32, val=100)
        s2 = _gray(4, 32, val=155)
        score = intensity_compatibility(s1, s2)
        assert 0.0 <= score <= 1.0


# ─── TestGradientCompatibilityExtra ─────────────────────────────────────────

class TestGradientCompatibilityExtra:
    def test_identical_random_near_one(self):
        s = _gray(4, 32, seed=3)
        score = gradient_compatibility(s, s)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_constant_strip_no_crash(self):
        s = _gray(4, 32, val=64)
        score = gradient_compatibility(s, s)
        assert isinstance(score, float)

    def test_color_identical(self):
        s = _color(4, 32, seed=2)
        score = gradient_compatibility(s, s)
        assert 0.0 <= score <= 1.0

    def test_random_different(self):
        s1 = _gray(4, 32, seed=0)
        s2 = _gray(4, 32, seed=7)
        score = gradient_compatibility(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_all_zero_vs_all_255(self):
        s1 = _gray(4, 32, val=0)
        s2 = _gray(4, 32, val=255)
        score = gradient_compatibility(s1, s2)
        assert isinstance(score, float)


# ─── TestColorCompatibilityExtra ────────────────────────────────────────────

class TestColorCompatibilityExtra:
    def test_identical_gray(self):
        s = _gray(4, 32, seed=9)
        assert color_compatibility(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_identical_color(self):
        s = _color(4, 32, seed=4)
        assert color_compatibility(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_in_range_random(self):
        for seed in range(5):
            s1 = _gray(4, 32, seed=seed)
            s2 = _gray(4, 32, seed=seed + 20)
            assert 0.0 <= color_compatibility(s1, s2) <= 1.0

    def test_black_vs_white(self):
        s1 = _gray(4, 32, val=0)
        s2 = _gray(4, 32, val=255)
        score = color_compatibility(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_near_same_vals(self):
        s1 = _gray(4, 32, val=127)
        s2 = _gray(4, 32, val=128)
        score = color_compatibility(s1, s2)
        assert score > 0.8


# ─── TestScoreBoundaryExtra ─────────────────────────────────────────────────

class TestScoreBoundaryExtra:
    def test_all_sides_no_crash(self):
        img = _gray()
        for side in BoundarySide:
            bs = score_boundary(img, img, side1=side)
            assert isinstance(bs, BoundaryScore)

    def test_aggregate_in_range_color(self):
        img = _color()
        bs = score_boundary(img, img)
        assert 0.0 <= bs.aggregate <= 1.0

    def test_custom_config_intensity_only(self):
        cfg = ScoringConfig(strip_width=4, w_intensity=1.0,
                            w_gradient=0.0, w_color=0.0)
        img = _gray()
        bs = score_boundary(img, img, cfg=cfg)
        assert bs.aggregate == pytest.approx(bs.intensity_diff, abs=1e-6)

    def test_side2_is_opposite_of_side1(self):
        img = _gray()
        bs = score_boundary(img, img, side1=BoundarySide.TOP)
        assert bs.side2 == BoundarySide.BOTTOM

    def test_identical_gray_high_score(self):
        img = _gray(seed=42)
        bs = score_boundary(img, img)
        assert bs.aggregate >= 0.5

    def test_extreme_contrast_lower(self):
        s1 = _gray(val=0)
        s2 = _gray(val=255)
        bs = score_boundary(s1, s2)
        assert bs.aggregate < 0.9

    def test_component_scores_in_range(self):
        bs = score_boundary(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= bs.intensity_diff <= 1.0
        assert 0.0 <= bs.gradient_score <= 1.0
        assert 0.0 <= bs.color_score <= 1.0


# ─── TestScoreMatrixExtra ───────────────────────────────────────────────────

class TestScoreMatrixExtra:
    def test_1x1(self):
        mat = score_matrix([_gray()])
        assert mat.shape == (1, 1)

    def test_2x2(self):
        imgs = [_gray(seed=0), _gray(seed=1)]
        mat = score_matrix(imgs)
        assert mat.shape == (2, 2)

    def test_5x5(self):
        imgs = [_gray(seed=i) for i in range(5)]
        mat = score_matrix(imgs)
        assert mat.shape == (5, 5)

    def test_diagonal_zero(self):
        imgs = [_gray(seed=i) for i in range(4)]
        mat = score_matrix(imgs)
        for i in range(4):
            assert mat[i, i] == pytest.approx(0.0)

    def test_all_in_range(self):
        imgs = [_gray(seed=i) for i in range(3)]
        mat = score_matrix(imgs)
        assert (mat >= 0.0).all()
        assert (mat <= 1.0 + 1e-9).all()

    def test_dtype(self):
        imgs = [_gray(seed=i) for i in range(3)]
        mat = score_matrix(imgs)
        assert mat.dtype == np.float64


# ─── TestBatchScoreBoundariesExtra ──────────────────────────────────────────

class TestBatchScoreBoundariesExtra:
    def test_single_pair(self):
        result = batch_score_boundaries([(_gray(seed=0), _gray(seed=1))])
        assert len(result) == 1
        assert isinstance(result[0], BoundaryScore)

    def test_five_pairs(self):
        pairs = [(_gray(seed=i), _gray(seed=i + 5)) for i in range(5)]
        result = batch_score_boundaries(pairs)
        assert len(result) == 5

    def test_identical_pairs_high_score(self):
        img = _gray(seed=7)
        result = batch_score_boundaries([(img, img)])
        assert result[0].aggregate >= 0.5

    def test_color_pairs(self):
        pairs = [(_color(seed=i), _color(seed=i + 3)) for i in range(3)]
        result = batch_score_boundaries(pairs)
        for bs in result:
            assert 0.0 <= bs.aggregate <= 1.0

    def test_each_aggregate_in_range(self):
        pairs = [(_gray(seed=i), _gray(seed=i + 1)) for i in range(4)]
        result = batch_score_boundaries(pairs)
        for bs in result:
            assert 0.0 <= bs.aggregate <= 1.0
