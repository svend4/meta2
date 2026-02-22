"""Tests for puzzle_reconstruction/algorithms/seam_evaluator.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.seam_evaluator import (
    SeamConfig,
    SeamScore,
    extract_seam_strip,
    color_continuity,
    gradient_continuity,
    texture_continuity,
    evaluate_seam,
    batch_evaluate_seams,
    rank_seams,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray_img(h=50, w=50, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr_img(h=50, w=50, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


def make_gradient_img(h=50, w=50):
    """Image with a horizontal gradient (0→255)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def make_noisy_img(h=30, w=30, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 255, (h, w)) * 255).astype(np.uint8).clip(0, 255)


# ─── SeamConfig ───────────────────────────────────────────────────────────────

class TestSeamConfig:
    def test_default(self):
        cfg = SeamConfig()
        assert cfg.w_color == pytest.approx(0.40)
        assert cfg.w_gradient == pytest.approx(0.35)
        assert cfg.w_texture == pytest.approx(0.25)
        assert cfg.blend_width == 8

    def test_total_weight(self):
        cfg = SeamConfig(w_color=0.5, w_gradient=0.3, w_texture=0.2)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_custom(self):
        cfg = SeamConfig(w_color=1.0, w_gradient=0.0, w_texture=0.5, blend_width=4)
        assert cfg.blend_width == 4

    def test_negative_w_color_raises(self):
        with pytest.raises(ValueError):
            SeamConfig(w_color=-0.1)

    def test_negative_w_gradient_raises(self):
        with pytest.raises(ValueError):
            SeamConfig(w_gradient=-1.0)

    def test_negative_w_texture_raises(self):
        with pytest.raises(ValueError):
            SeamConfig(w_texture=-0.5)

    def test_blend_width_zero_raises(self):
        with pytest.raises(ValueError):
            SeamConfig(blend_width=0)

    def test_blend_width_negative_raises(self):
        with pytest.raises(ValueError):
            SeamConfig(blend_width=-3)

    def test_zero_weights_allowed(self):
        cfg = SeamConfig(w_color=0.0, w_gradient=0.0, w_texture=0.0)
        assert cfg.total_weight == 0.0

    def test_blend_width_one_ok(self):
        cfg = SeamConfig(blend_width=1)
        assert cfg.blend_width == 1


# ─── SeamScore ────────────────────────────────────────────────────────────────

class TestSeamScore:
    def test_basic(self):
        s = SeamScore(score=0.8, color_score=0.9, gradient_score=0.7, texture_score=0.8)
        assert s.score == pytest.approx(0.8)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SeamScore(score=1.1, color_score=0.5, gradient_score=0.5, texture_score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            SeamScore(score=-0.1, color_score=0.5, gradient_score=0.5, texture_score=0.5)

    def test_color_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SeamScore(score=0.5, color_score=1.5, gradient_score=0.5, texture_score=0.5)

    def test_gradient_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SeamScore(score=0.5, color_score=0.5, gradient_score=-0.1, texture_score=0.5)

    def test_texture_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SeamScore(score=0.5, color_score=0.5, gradient_score=0.5, texture_score=2.0)

    def test_params_stored(self):
        s = SeamScore(score=0.5, color_score=0.5,
                      gradient_score=0.5, texture_score=0.5,
                      params={"key": "val"})
        assert s.params["key"] == "val"

    def test_boundary_values_ok(self):
        s = SeamScore(score=0.0, color_score=1.0, gradient_score=0.0, texture_score=1.0)
        assert s.score == 0.0
        assert s.color_score == 1.0


# ─── extract_seam_strip ───────────────────────────────────────────────────────

class TestExtractSeamStrip:
    def test_side_0_top(self):
        img = np.arange(50 * 60, dtype=np.uint8).reshape(50, 60)
        strip = extract_seam_strip(img, side=0, width=5)
        assert strip.shape[0] <= 5
        assert strip.shape[1] == 60

    def test_side_1_right(self):
        img = make_gray_img(50, 60)
        strip = extract_seam_strip(img, side=1, width=4)
        assert strip.shape[0] == 50
        assert strip.shape[1] <= 4

    def test_side_2_bottom(self):
        img = make_gray_img(50, 60)
        strip = extract_seam_strip(img, side=2, width=6)
        assert strip.shape[0] <= 6
        assert strip.shape[1] == 60

    def test_side_3_left(self):
        img = make_gray_img(50, 60)
        strip = extract_seam_strip(img, side=3, width=3)
        assert strip.shape[0] == 50
        assert strip.shape[1] <= 3

    def test_invalid_side_raises(self):
        img = make_gray_img()
        with pytest.raises(ValueError):
            extract_seam_strip(img, side=4)

    def test_negative_side_raises(self):
        img = make_gray_img()
        with pytest.raises(ValueError):
            extract_seam_strip(img, side=-1)

    def test_width_zero_raises(self):
        img = make_gray_img()
        with pytest.raises(ValueError):
            extract_seam_strip(img, side=0, width=0)

    def test_output_float32(self):
        img = make_gray_img()
        strip = extract_seam_strip(img, side=0)
        assert strip.dtype == np.float32

    def test_bgr_strip(self):
        img = make_bgr_img(50, 60)
        strip = extract_seam_strip(img, side=0, width=5)
        assert strip.ndim == 3
        assert strip.shape[2] == 3

    def test_width_one_ok(self):
        img = make_gray_img(20, 20)
        strip = extract_seam_strip(img, side=0, width=1)
        assert strip.shape[0] >= 1

    def test_values_match_source(self):
        """Top strip should match top rows of source."""
        img = make_gradient_img(20, 30)
        strip = extract_seam_strip(img, side=0, width=3)
        np.testing.assert_array_equal(strip[:3], img[:3].astype(np.float32))


# ─── color_continuity ─────────────────────────────────────────────────────────

class TestColorContinuity:
    def test_identical_strips_near_one(self):
        strip = np.full((5, 10), 100.0, dtype=np.float32)
        assert color_continuity(strip, strip.copy()) == pytest.approx(1.0)

    def test_range(self):
        a = np.full((5, 5), 0.0, dtype=np.float32)
        b = np.full((5, 5), 255.0, dtype=np.float32)
        result = color_continuity(a, b)
        assert 0.0 <= result <= 1.0

    def test_max_difference_near_zero(self):
        a = np.full((5, 5), 0.0, dtype=np.float32)
        b = np.full((5, 5), 255.0, dtype=np.float32)
        result = color_continuity(a, b)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_empty_a_returns_one(self):
        a = np.zeros((0, 5), dtype=np.float32)
        b = np.full((5, 5), 100.0, dtype=np.float32)
        assert color_continuity(a, b) == 1.0

    def test_empty_b_returns_one(self):
        a = np.full((5, 5), 100.0, dtype=np.float32)
        b = np.zeros((0, 5), dtype=np.float32)
        assert color_continuity(a, b) == 1.0

    def test_close_means_high_score(self):
        a = np.full((5, 5), 128.0, dtype=np.float32)
        b = np.full((5, 5), 130.0, dtype=np.float32)
        assert color_continuity(a, b) > 0.99

    def test_non_negative(self):
        a = np.full((5, 5), 255.0, dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        assert color_continuity(a, b) >= 0.0


# ─── gradient_continuity ──────────────────────────────────────────────────────

class TestGradientContinuity:
    def test_identical_strips(self):
        img = make_gradient_img(10, 20).astype(np.float32)
        assert gradient_continuity(img, img.copy()) == pytest.approx(1.0)

    def test_range(self):
        a = make_gradient_img(10, 20).astype(np.float32)
        b = np.zeros((10, 20), dtype=np.float32)
        result = gradient_continuity(a, b)
        assert 0.0 <= result <= 1.0

    def test_empty_a_returns_one(self):
        a = np.zeros((0, 5), dtype=np.float32)
        b = np.full((5, 5), 100.0, dtype=np.float32)
        assert gradient_continuity(a, b) == 1.0

    def test_empty_b_returns_one(self):
        a = np.full((5, 5), 100.0, dtype=np.float32)
        b = np.zeros((0, 5), dtype=np.float32)
        assert gradient_continuity(a, b) == 1.0

    def test_constant_strips_near_one(self):
        """Constant strips have zero gradient difference → similarity ≈ 1."""
        a = np.full((10, 10), 100.0, dtype=np.float32)
        b = np.full((10, 10), 100.0, dtype=np.float32)
        assert gradient_continuity(a, b) == pytest.approx(1.0)

    def test_bgr_accepted(self):
        a = make_bgr_img(10, 10).astype(np.float32)
        b = make_bgr_img(10, 10).astype(np.float32)
        result = gradient_continuity(a, b)
        assert 0.0 <= result <= 1.0


# ─── texture_continuity ───────────────────────────────────────────────────────

class TestTextureContinuity:
    def test_identical_strips(self):
        img = make_noisy_img(10, 10).astype(np.float32)
        assert texture_continuity(img, img.copy()) == pytest.approx(1.0)

    def test_range(self):
        a = make_noisy_img(10, 10).astype(np.float32)
        b = np.full((10, 10), 100.0, dtype=np.float32)
        result = texture_continuity(a, b)
        assert 0.0 <= result <= 1.0

    def test_both_zero_returns_one(self):
        a = np.full((5, 5), 128.0, dtype=np.float32)
        b = np.full((5, 5), 200.0, dtype=np.float32)
        # both constant → std=0 → returns 1.0
        assert texture_continuity(a, b) == pytest.approx(1.0)

    def test_empty_a_returns_one(self):
        a = np.zeros((0, 5), dtype=np.float32)
        b = make_noisy_img(5, 5).astype(np.float32)
        assert texture_continuity(a, b) == 1.0

    def test_empty_b_returns_one(self):
        a = make_noisy_img(5, 5).astype(np.float32)
        b = np.zeros((0, 5), dtype=np.float32)
        assert texture_continuity(a, b) == 1.0

    def test_symmetric(self):
        a = make_noisy_img(10, 10, seed=1).astype(np.float32)
        b = make_noisy_img(10, 10, seed=2).astype(np.float32)
        assert texture_continuity(a, b) == pytest.approx(texture_continuity(b, a))

    def test_non_negative(self):
        a = make_noisy_img(10, 10, seed=10).astype(np.float32)
        b = np.full((10, 10), 255.0, dtype=np.float32)
        assert texture_continuity(a, b) >= 0.0


# ─── evaluate_seam ────────────────────────────────────────────────────────────

class TestEvaluateSeam:
    def test_returns_seam_score(self):
        img = make_gray_img()
        result = evaluate_seam(img, 0, img, 2)
        assert isinstance(result, SeamScore)

    def test_score_in_range(self):
        a = make_gray_img(50, 50, 100)
        b = make_gray_img(50, 50, 200)
        result = evaluate_seam(a, 0, b, 2)
        assert 0.0 <= result.score <= 1.0

    def test_identical_images_high_score(self):
        img = make_gray_img(50, 50, 150)
        result = evaluate_seam(img, 0, img, 2)
        assert result.score > 0.8

    def test_params_stored(self):
        img = make_gray_img()
        result = evaluate_seam(img, 1, img, 3)
        assert "side_a" in result.params
        assert result.params["side_a"] == 1
        assert result.params["side_b"] == 3

    def test_default_config(self):
        img = make_gray_img()
        result = evaluate_seam(img, 0, img, 2)
        assert result.params["blend_width"] == 8

    def test_custom_config(self):
        img = make_gray_img()
        cfg = SeamConfig(blend_width=4)
        result = evaluate_seam(img, 0, img, 2, cfg=cfg)
        assert result.params["blend_width"] == 4

    def test_bgr_images(self):
        a = make_bgr_img(40, 40, (100, 120, 80))
        b = make_bgr_img(40, 40, (100, 120, 80))
        result = evaluate_seam(a, 1, b, 3)
        assert isinstance(result, SeamScore)

    def test_all_sides(self):
        img = make_gray_img(60, 60, 100)
        for side_a in range(4):
            for side_b in range(4):
                result = evaluate_seam(img, side_a, img, side_b)
                assert 0.0 <= result.score <= 1.0

    def test_zero_weights_score_zero(self):
        img = make_gray_img()
        cfg = SeamConfig(w_color=0.0, w_gradient=0.0, w_texture=0.0)
        result = evaluate_seam(img, 0, img, 2, cfg=cfg)
        assert result.score == pytest.approx(0.0)


# ─── batch_evaluate_seams ─────────────────────────────────────────────────────

class TestBatchEvaluateSeams:
    def test_returns_list(self):
        img = make_gray_img()
        pairs = [(img, 0, img, 2), (img, 1, img, 3)]
        results = batch_evaluate_seams(pairs)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_all_seam_scores(self):
        img = make_gray_img()
        pairs = [(img, i, img, (i + 2) % 4) for i in range(4)]
        results = batch_evaluate_seams(pairs)
        assert all(isinstance(r, SeamScore) for r in results)

    def test_empty_list(self):
        results = batch_evaluate_seams([])
        assert results == []

    def test_custom_config(self):
        img = make_gray_img()
        cfg = SeamConfig(blend_width=3)
        results = batch_evaluate_seams([(img, 0, img, 2)], cfg=cfg)
        assert results[0].params["blend_width"] == 3

    def test_scores_in_range(self):
        a = make_gray_img(40, 40, 80)
        b = make_gray_img(40, 40, 180)
        pairs = [(a, 0, b, 2), (b, 1, a, 3), (a, 2, a, 0)]
        results = batch_evaluate_seams(pairs)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ─── rank_seams ───────────────────────────────────────────────────────────────

class TestRankSeams:
    def _make_scores(self, values):
        return [SeamScore(score=v, color_score=v, gradient_score=v, texture_score=v)
                for v in values]

    def test_returns_list_of_tuples(self):
        scores = self._make_scores([0.5, 0.8, 0.3])
        ranked = rank_seams(scores)
        assert isinstance(ranked, list)
        for item in ranked:
            assert len(item) == 2

    def test_sorted_descending(self):
        scores = self._make_scores([0.3, 0.9, 0.5, 0.7])
        ranked = rank_seams(scores)
        vals = [s for _, s in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_default_indices(self):
        scores = self._make_scores([0.4, 0.6, 0.2])
        ranked = rank_seams(scores)
        indices = [i for i, _ in ranked]
        assert set(indices) == {0, 1, 2}

    def test_custom_indices(self):
        scores = self._make_scores([0.3, 0.7, 0.5])
        ranked = rank_seams(scores, indices=[10, 20, 30])
        indices = [i for i, _ in ranked]
        assert set(indices) == {10, 20, 30}

    def test_len_mismatch_raises(self):
        scores = self._make_scores([0.5, 0.8])
        with pytest.raises(ValueError):
            rank_seams(scores, indices=[0])

    def test_empty_scores(self):
        ranked = rank_seams([])
        assert ranked == []

    def test_highest_first(self):
        scores = self._make_scores([0.2, 0.9, 0.5])
        ranked = rank_seams(scores)
        assert ranked[0][0] == 1  # index of 0.9
        assert ranked[0][1] == pytest.approx(0.9)

    def test_tie_stable(self):
        scores = self._make_scores([0.7, 0.7, 0.7])
        ranked = rank_seams(scores)
        vals = [s for _, s in ranked]
        assert all(v == pytest.approx(0.7) for v in vals)
