"""Additional tests for puzzle_reconstruction.algorithms.edge_scorer."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.edge_scorer import (
    EdgeScore,
    score_color_compat,
    score_gradient_compat,
    score_texture_compat,
    score_edge_pair,
    batch_score_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _uniform(val=128, h=64, w=64):
    return np.full((h, w), val, dtype=np.uint8)


def _make_es(**kw):
    defaults = dict(idx1=0, idx2=1, side1=0, side2=2,
                    color_score=0.5, gradient_score=0.5,
                    texture_score=0.5, total_score=0.5)
    defaults.update(kw)
    return EdgeScore(**defaults)


# ─── TestEdgeScoreExtra ───────────────────────────────────────────────────────

class TestEdgeScoreExtra:
    def test_idx_zero(self):
        e = _make_es(idx1=0, idx2=0)
        assert e.idx1 == 0

    def test_large_idx(self):
        e = _make_es(idx1=999, idx2=1000)
        assert e.idx2 == 1000

    def test_color_score_zero(self):
        e = _make_es(color_score=0.0, total_score=0.0)
        assert e.color_score == pytest.approx(0.0)

    def test_total_score_zero(self):
        e = _make_es(total_score=0.0)
        assert e.total_score == pytest.approx(0.0)

    def test_total_score_one(self):
        e = _make_es(color_score=1.0, gradient_score=1.0,
                     texture_score=1.0, total_score=1.0)
        assert e.total_score == pytest.approx(1.0)

    def test_method_custom(self):
        e = EdgeScore(idx1=0, idx2=1, side1=0, side2=0,
                      color_score=0.5, gradient_score=0.5,
                      texture_score=0.5, total_score=0.5,
                      method="max")
        assert e.method == "max"

    def test_params_multiple_entries(self):
        e = EdgeScore(idx1=0, idx2=1, side1=0, side2=0,
                      color_score=0.5, gradient_score=0.5,
                      texture_score=0.5, total_score=0.5,
                      params={"a": 1, "b": 2, "c": 3})
        assert e.params["a"] == 1
        assert e.params["c"] == 3


# ─── TestScoreColorCompatExtra ────────────────────────────────────────────────

class TestScoreColorCompatExtra:
    def test_border_px_1(self):
        img = _gray()
        s = score_color_compat(img, img, border_px=1)
        assert 0.0 <= s <= 1.0

    def test_border_px_20(self):
        img = _gray()
        s = score_color_compat(img, img, border_px=20)
        assert 0.0 <= s <= 1.0

    def test_uniform_same_image_high(self):
        u = _uniform(200)
        s = score_color_compat(u, u)
        assert s >= 0.8

    def test_result_float(self):
        s = score_color_compat(_gray(seed=0), _gray(seed=1))
        assert isinstance(s, float)

    def test_side_0_1_both_in_range(self):
        img = _gray()
        s0 = score_color_compat(img, img, side1=0, side2=0)
        s1 = score_color_compat(img, img, side1=1, side2=1)
        assert 0.0 <= s0 <= 1.0
        assert 0.0 <= s1 <= 1.0

    def test_rgb_different_seeds(self):
        s = score_color_compat(_rgb(seed=3), _rgb(seed=4))
        assert 0.0 <= s <= 1.0

    def test_deterministic(self):
        img1 = _gray(seed=0)
        img2 = _gray(seed=1)
        assert score_color_compat(img1, img2) == score_color_compat(img1, img2)


# ─── TestScoreGradientCompatExtra ────────────────────────────────────────────

class TestScoreGradientCompatExtra:
    def test_uniform_vs_uniform_in_range(self):
        u = _uniform(100)
        s = score_gradient_compat(u, u)
        assert 0.0 <= s <= 1.0

    def test_border_side_2(self):
        img = _gray()
        s = score_gradient_compat(img, img, side1=2, side2=2)
        assert 0.0 <= s <= 1.0

    def test_border_side_3(self):
        img = _gray()
        s = score_gradient_compat(img, img, side1=3, side2=3)
        assert 0.0 <= s <= 1.0

    def test_rgb_same_in_range(self):
        img = _rgb(seed=5)
        s = score_gradient_compat(img, img)
        assert 0.0 <= s <= 1.0

    def test_result_is_float(self):
        s = score_gradient_compat(_gray(seed=0), _gray(seed=1))
        assert isinstance(s, float)

    def test_deterministic(self):
        img1 = _gray(seed=0)
        img2 = _gray(seed=1)
        assert score_gradient_compat(img1, img2) == score_gradient_compat(img1, img2)


# ─── TestScoreTextureCompatExtra ─────────────────────────────────────────────

class TestScoreTextureCompatExtra:
    def test_uniform_img_in_range(self):
        u = _uniform(64)
        s = score_texture_compat(u, u)
        assert 0.0 <= s <= 1.0

    def test_large_image_no_crash(self):
        img = _gray(h=128, w=128, seed=7)
        s = score_texture_compat(img, img)
        assert 0.0 <= s <= 1.0

    def test_side_1_2_in_range(self):
        img = _gray()
        s1 = score_texture_compat(img, img, side1=1, side2=1)
        s2 = score_texture_compat(img, img, side1=2, side2=2)
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0

    def test_rgb_different_in_range(self):
        s = score_texture_compat(_rgb(seed=0), _rgb(seed=5))
        assert 0.0 <= s <= 1.0

    def test_result_float(self):
        s = score_texture_compat(_gray(seed=2), _gray(seed=3))
        assert isinstance(s, float)


# ─── TestScoreEdgePairExtra ───────────────────────────────────────────────────

class TestScoreEdgePairExtra:
    def test_equal_weights_total_is_mean(self):
        img = _gray(seed=0)
        weights = {"color": 1.0, "gradient": 1.0, "texture": 1.0}
        r = score_edge_pair(img, img, weights=weights)
        expected = (r.color_score + r.gradient_score + r.texture_score) / 3.0
        assert r.total_score == pytest.approx(expected, abs=1e-4)

    def test_gradient_only_weight(self):
        img1 = _gray(seed=0)
        img2 = _gray(seed=1)
        weights = {"color": 0.0, "gradient": 1.0, "texture": 0.0}
        r = score_edge_pair(img1, img2, weights=weights)
        assert r.total_score == pytest.approx(r.gradient_score, abs=1e-5)

    def test_texture_only_weight(self):
        img1 = _gray(seed=0)
        img2 = _gray(seed=1)
        weights = {"color": 0.0, "gradient": 0.0, "texture": 1.0}
        r = score_edge_pair(img1, img2, weights=weights)
        assert r.total_score == pytest.approx(r.texture_score, abs=1e-5)

    def test_side_cross_combination(self):
        img = _gray()
        r = score_edge_pair(img, img, side1=0, side2=2)
        assert 0.0 <= r.total_score <= 1.0

    def test_large_border_px(self):
        img = _gray()
        r = score_edge_pair(img, img, border_px=30)
        assert 0.0 <= r.total_score <= 1.0

    def test_bins_32(self):
        img = _gray()
        r = score_edge_pair(img, img, bins=32)
        assert isinstance(r, EdgeScore)

    def test_rgb_all_channels_in_range(self):
        r = score_edge_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.color_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0
        assert 0.0 <= r.texture_score <= 1.0

    def test_idx_default_stored(self):
        r = score_edge_pair(_gray(), _gray())
        assert r.idx1 == 0
        assert r.idx2 == 1


# ─── TestBatchScoreEdgesExtra ─────────────────────────────────────────────────

class TestBatchScoreEdgesExtra:
    def test_single_pair(self):
        imgs = [_gray(seed=i) for i in range(2)]
        result = batch_score_edges(imgs, [(0, 1)])
        assert len(result) == 1
        assert isinstance(result[0], EdgeScore)

    def test_six_images_all_pairs(self):
        n = 6
        imgs = [_gray(seed=i) for i in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        result = batch_score_edges(imgs, pairs)
        assert len(result) == len(pairs)

    def test_side_pairs_all_zeros(self):
        imgs = [_gray(seed=i) for i in range(3)]
        pairs = [(0, 1), (1, 2)]
        side_pairs = [(0, 0), (0, 0)]
        result = batch_score_edges(imgs, pairs, side_pairs=side_pairs)
        assert result[0].side1 == 0
        assert result[0].side2 == 0

    def test_total_scores_all_in_range(self):
        imgs = [_gray(seed=i) for i in range(4)]
        result = batch_score_edges(imgs, [(0, 1), (0, 2), (1, 3)])
        for r in result:
            assert 0.0 <= r.total_score <= 1.0

    def test_idx1_idx2_match_pair_indices(self):
        imgs = [_gray(seed=i) for i in range(5)]
        pairs = [(2, 4), (0, 3)]
        result = batch_score_edges(imgs, pairs)
        assert result[0].idx1 == 2
        assert result[0].idx2 == 4
        assert result[1].idx1 == 0
        assert result[1].idx2 == 3
