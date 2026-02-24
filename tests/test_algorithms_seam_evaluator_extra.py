"""Extra tests for puzzle_reconstruction/algorithms/seam_evaluator.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=50, w=50, val=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = val
    return img


def _gradient(h=50, w=50):
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _noisy(h=30, w=30, seed=42):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _strip(h=5, w=10, val=100.0):
    return np.full((h, w), val, dtype=np.float32)


def _seam_score(v=0.5):
    return SeamScore(score=v, color_score=v, gradient_score=v, texture_score=v)


# ─── SeamConfig (extra) ──────────────────────────────────────────────────────

class TestSeamConfigExtra:
    def test_large_weights_ok(self):
        cfg = SeamConfig(w_color=10.0, w_gradient=5.0, w_texture=3.0)
        assert cfg.total_weight == pytest.approx(18.0)

    def test_single_weight_nonzero(self):
        cfg = SeamConfig(w_color=1.0, w_gradient=0.0, w_texture=0.0)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_blend_width_large(self):
        cfg = SeamConfig(blend_width=100)
        assert cfg.blend_width == 100

    def test_independent_instances(self):
        c1 = SeamConfig(w_color=0.5)
        c2 = SeamConfig(w_color=0.9)
        assert c1.w_color != c2.w_color

    def test_equal_weights(self):
        cfg = SeamConfig(w_color=0.33, w_gradient=0.33, w_texture=0.34)
        assert cfg.total_weight == pytest.approx(1.0)


# ─── SeamScore (extra) ───────────────────────────────────────────────────────

class TestSeamScoreExtra:
    def test_score_zero_ok(self):
        s = _seam_score(0.0)
        assert s.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        s = _seam_score(1.0)
        assert s.score == pytest.approx(1.0)

    def test_all_channels_zero(self):
        s = SeamScore(score=0.0, color_score=0.0, gradient_score=0.0, texture_score=0.0)
        assert s.gradient_score == pytest.approx(0.0)

    def test_all_channels_one(self):
        s = SeamScore(score=1.0, color_score=1.0, gradient_score=1.0, texture_score=1.0)
        assert s.texture_score == pytest.approx(1.0)

    def test_params_default_empty(self):
        s = _seam_score()
        assert s.params == {}

    def test_params_custom(self):
        s = SeamScore(score=0.5, color_score=0.5, gradient_score=0.5,
                      texture_score=0.5, params={"side_a": 0, "side_b": 2})
        assert s.params["side_a"] == 0


# ─── extract_seam_strip (extra) ──────────────────────────────────────────────

class TestExtractSeamStripExtra:
    def test_all_sides_produce_float32(self):
        img = _gray(40, 40)
        for side in range(4):
            strip = extract_seam_strip(img, side=side, width=5)
            assert strip.dtype == np.float32

    def test_all_sides_nonempty(self):
        img = _gray(40, 40)
        for side in range(4):
            strip = extract_seam_strip(img, side=side, width=3)
            assert strip.size > 0

    def test_side_5_raises(self):
        with pytest.raises(ValueError):
            extract_seam_strip(_gray(), side=5)

    def test_bgr_side_1_has_channels(self):
        img = _bgr(50, 60)
        strip = extract_seam_strip(img, side=1, width=5)
        assert strip.ndim == 3
        assert strip.shape[2] == 3

    def test_large_width_clipped(self):
        img = _gray(10, 10)
        strip = extract_seam_strip(img, side=0, width=20)
        assert strip.shape[0] <= 20

    def test_bottom_strip_values(self):
        img = _gradient(20, 30)
        strip = extract_seam_strip(img, side=2, width=3)
        np.testing.assert_array_equal(strip[-3:], img[-3:].astype(np.float32))


# ─── color_continuity (extra) ────────────────────────────────────────────────

class TestColorContinuityExtra:
    def test_result_float(self):
        result = color_continuity(_strip(), _strip())
        assert isinstance(result, float)

    def test_consistent_calls(self):
        a, b = _strip(val=50.0), _strip(val=200.0)
        assert color_continuity(a, b) == pytest.approx(color_continuity(a, b))

    def test_close_strips_high(self):
        a = _strip(val=100.0)
        b = _strip(val=101.0)
        assert color_continuity(a, b) > 0.99

    def test_far_strips_low(self):
        a = _strip(val=0.0)
        b = _strip(val=255.0)
        assert color_continuity(a, b) < 0.05

    def test_symmetric(self):
        a = _strip(val=50.0)
        b = _strip(val=200.0)
        assert color_continuity(a, b) == pytest.approx(color_continuity(b, a))


# ─── gradient_continuity (extra) ─────────────────────────────────────────────

class TestGradientContinuityExtra:
    def test_result_float(self):
        a = _gradient(10, 20).astype(np.float32)
        result = gradient_continuity(a, a.copy())
        assert isinstance(result, float)

    def test_consistent_calls(self):
        a = _gradient(10, 20).astype(np.float32)
        b = _strip(val=128.0)
        r1 = gradient_continuity(a, b)
        r2 = gradient_continuity(a, b)
        assert r1 == pytest.approx(r2)

    def test_same_gradient_gives_one(self):
        g = _gradient(10, 20).astype(np.float32)
        assert gradient_continuity(g, g.copy()) == pytest.approx(1.0)

    def test_symmetric(self):
        a = _gradient(10, 20).astype(np.float32)
        b = np.zeros((10, 20), dtype=np.float32)
        r1 = gradient_continuity(a, b)
        r2 = gradient_continuity(b, a)
        assert r1 == pytest.approx(r2, abs=1e-6)

    def test_nonneg(self):
        a = _gradient(10, 20).astype(np.float32)
        b = _strip(5, 20, val=0.0)
        assert gradient_continuity(a, b) >= 0.0


# ─── texture_continuity (extra) ──────────────────────────────────────────────

class TestTextureContinuityExtra:
    def test_result_float(self):
        a = _noisy(10, 10).astype(np.float32)
        result = texture_continuity(a, a.copy())
        assert isinstance(result, float)

    def test_consistent_calls(self):
        a = _noisy(10, 10, seed=1).astype(np.float32)
        b = _noisy(10, 10, seed=2).astype(np.float32)
        assert texture_continuity(a, b) == pytest.approx(texture_continuity(a, b))

    def test_same_texture_gives_one(self):
        n = _noisy(10, 10).astype(np.float32)
        assert texture_continuity(n, n.copy()) == pytest.approx(1.0)

    def test_constant_vs_constant_one(self):
        a = _strip(val=50.0)
        b = _strip(val=200.0)
        assert texture_continuity(a, b) == pytest.approx(1.0)

    def test_nonneg(self):
        a = _noisy(10, 10, seed=0).astype(np.float32)
        b = _strip(val=128.0)
        assert texture_continuity(a, b) >= 0.0


# ─── evaluate_seam (extra) ───────────────────────────────────────────────────

class TestEvaluateSeamExtra:
    def test_all_channels_in_range(self):
        img = _gray(50, 50, 128)
        r = evaluate_seam(img, 0, img, 2)
        assert 0.0 <= r.color_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0
        assert 0.0 <= r.texture_score <= 1.0

    def test_bgr_score_in_range(self):
        a = _bgr(40, 40, (100, 120, 80))
        b = _bgr(40, 40, (200, 180, 220))
        r = evaluate_seam(a, 0, b, 2)
        assert 0.0 <= r.score <= 1.0

    def test_params_contains_sides(self):
        img = _gray()
        r = evaluate_seam(img, 2, img, 0)
        assert r.params["side_a"] == 2
        assert r.params["side_b"] == 0

    def test_side_1_and_3(self):
        img = _gray(60, 60)
        r = evaluate_seam(img, 1, img, 3)
        assert 0.0 <= r.score <= 1.0

    def test_identical_images_score_high(self):
        img = _gray(50, 50, 150)
        r = evaluate_seam(img, 0, img, 2)
        assert r.score > 0.7

    def test_very_different_images_lower(self):
        a = _gray(50, 50, 0)
        b = _gray(50, 50, 255)
        r = evaluate_seam(a, 0, b, 2)
        r_same = evaluate_seam(a, 0, a, 2)
        assert r_same.score >= r.score


# ─── batch_evaluate_seams (extra) ────────────────────────────────────────────

class TestBatchEvaluateSeamsExtra:
    def test_single_pair(self):
        img = _gray()
        results = batch_evaluate_seams([(img, 0, img, 2)])
        assert len(results) == 1

    def test_large_batch(self):
        img = _gray()
        pairs = [(img, i % 4, img, (i + 2) % 4) for i in range(8)]
        results = batch_evaluate_seams(pairs)
        assert len(results) == 8

    def test_all_scores_in_range(self):
        a = _gray(40, 40, 80)
        b = _gray(40, 40, 180)
        pairs = [(a, 0, b, 2), (b, 1, a, 3)]
        for r in batch_evaluate_seams(pairs):
            assert 0.0 <= r.score <= 1.0

    def test_consistent_with_individual(self):
        img = _gray(40, 40, 128)
        batch_r = batch_evaluate_seams([(img, 0, img, 2)])[0]
        indiv_r = evaluate_seam(img, 0, img, 2)
        assert batch_r.score == pytest.approx(indiv_r.score, abs=1e-6)

    def test_custom_config_passed(self):
        img = _gray()
        cfg = SeamConfig(blend_width=5)
        results = batch_evaluate_seams([(img, 0, img, 2)], cfg=cfg)
        assert results[0].params["blend_width"] == 5


# ─── rank_seams (extra) ──────────────────────────────────────────────────────

class TestRankSeamsExtra:
    def test_single_score(self):
        ranked = rank_seams([_seam_score(0.7)])
        assert len(ranked) == 1

    def test_first_is_highest(self):
        scores = [_seam_score(0.3), _seam_score(0.9), _seam_score(0.6)]
        ranked = rank_seams(scores)
        assert ranked[0][1] == pytest.approx(0.9)

    def test_last_is_lowest(self):
        scores = [_seam_score(0.3), _seam_score(0.9), _seam_score(0.6)]
        ranked = rank_seams(scores)
        assert ranked[-1][1] == pytest.approx(0.3)

    def test_all_same_all_present(self):
        scores = [_seam_score(0.5) for _ in range(4)]
        ranked = rank_seams(scores)
        assert len(ranked) == 4

    def test_tuple_format(self):
        ranked = rank_seams([_seam_score(0.7)])
        idx, val = ranked[0]
        assert isinstance(idx, int)
        assert isinstance(val, float)

    def test_custom_indices(self):
        scores = [_seam_score(0.3), _seam_score(0.7)]
        ranked = rank_seams(scores, indices=[10, 20])
        ids = {i for i, _ in ranked}
        assert ids == {10, 20}
