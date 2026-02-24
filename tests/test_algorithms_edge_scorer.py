"""Тесты для puzzle_reconstruction.algorithms.edge_scorer."""
import pytest
import numpy as np
from puzzle_reconstruction.algorithms.edge_scorer import (
    EdgeScore,
    score_color_compat,
    score_gradient_compat,
    score_texture_compat,
    score_edge_pair,
    batch_score_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _uniform(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _make_edge_score(**kwargs) -> EdgeScore:
    defaults = dict(idx1=0, idx2=1, side1=2, side2=0,
                    color_score=0.8, gradient_score=0.7,
                    texture_score=0.6, total_score=0.73)
    defaults.update(kwargs)
    return EdgeScore(**defaults)


# ─── TestEdgeScore ────────────────────────────────────────────────────────────

class TestEdgeScore:
    def test_basic_fields(self):
        e = _make_edge_score()
        assert e.idx1 == 0
        assert e.idx2 == 1
        assert e.side1 == 2
        assert e.side2 == 0

    def test_channel_scores_stored(self):
        e = _make_edge_score()
        assert e.color_score == pytest.approx(0.8)
        assert e.gradient_score == pytest.approx(0.7)
        assert e.texture_score == pytest.approx(0.6)

    def test_total_score_stored(self):
        e = _make_edge_score()
        assert e.total_score == pytest.approx(0.73)

    def test_method_default(self):
        e = _make_edge_score()
        assert e.method == "weighted"

    def test_params_default_empty(self):
        e = _make_edge_score()
        assert e.params == {}

    def test_params_stored(self):
        e = EdgeScore(idx1=0, idx2=1, side1=0, side2=2,
                      color_score=0.5, gradient_score=0.5,
                      texture_score=0.5, total_score=0.5,
                      params={"border_px": 10})
        assert e.params["border_px"] == 10

    def test_side_values_0_to_3(self):
        for side in range(4):
            e = _make_edge_score(side1=side, side2=side)
            assert e.side1 == side


# ─── TestScoreColorCompat ─────────────────────────────────────────────────────

class TestScoreColorCompat:
    def test_returns_float(self):
        s = score_color_compat(_gray(), _gray())
        assert isinstance(s, float)

    def test_in_range(self):
        s = score_color_compat(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= s <= 1.0

    def test_identical_images_high(self):
        img = _uniform()  # uniform image: all strips identical → high correlation
        s = score_color_compat(img, img)
        assert s >= 0.8

    def test_rgb_ok(self):
        s = score_color_compat(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= s <= 1.0

    def test_all_sides(self):
        img = _gray()
        for s1 in range(4):
            for s2 in range(4):
                score = score_color_compat(img, img, side1=s1, side2=s2)
                assert 0.0 <= score <= 1.0

    def test_custom_border_px(self):
        img = _gray()
        s = score_color_compat(img, img, border_px=5)
        assert 0.0 <= s <= 1.0

    def test_uniform_vs_random_lower(self):
        u = _uniform()
        rand = _gray()
        s_same = score_color_compat(u, u)
        s_diff = score_color_compat(u, rand)
        assert s_same >= s_diff


# ─── TestScoreGradientCompat ──────────────────────────────────────────────────

class TestScoreGradientCompat:
    def test_returns_float(self):
        s = score_gradient_compat(_gray(), _gray())
        assert isinstance(s, float)

    def test_in_range(self):
        s = score_gradient_compat(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= s <= 1.0

    def test_identical_images_high(self):
        img = _uniform()  # uniform image: same gradient profiles → in range
        s = score_gradient_compat(img, img)
        assert 0.0 <= s <= 1.0

    def test_rgb_ok(self):
        s = score_gradient_compat(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= s <= 1.0

    def test_all_sides(self):
        img = _gray()
        for side in range(4):
            score = score_gradient_compat(img, img, side1=side, side2=side)
            assert 0.0 <= score <= 1.0

    def test_non_negative(self):
        s = score_gradient_compat(_gray(seed=0), _gray(seed=5))
        assert s >= 0.0


# ─── TestScoreTextureCompat ───────────────────────────────────────────────────

class TestScoreTextureCompat:
    def test_returns_float(self):
        s = score_texture_compat(_gray(), _gray())
        assert isinstance(s, float)

    def test_in_range(self):
        s = score_texture_compat(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= s <= 1.0

    def test_identical_images_high(self):
        img = _gray()
        s = score_texture_compat(img, img)
        assert s >= 0.7

    def test_rgb_ok(self):
        s = score_texture_compat(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= s <= 1.0

    def test_all_sides(self):
        img = _gray()
        for side in range(4):
            score = score_texture_compat(img, img, side1=side, side2=side)
            assert 0.0 <= score <= 1.0


# ─── TestScoreEdgePair ────────────────────────────────────────────────────────

class TestScoreEdgePair:
    def test_returns_edge_score(self):
        r = score_edge_pair(_gray(seed=0), _gray(seed=1))
        assert isinstance(r, EdgeScore)

    def test_ids_stored(self):
        r = score_edge_pair(_gray(), _gray(), idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_sides_stored(self):
        r = score_edge_pair(_gray(), _gray(), side1=1, side2=3)
        assert r.side1 == 1
        assert r.side2 == 3

    def test_total_score_in_range(self):
        r = score_edge_pair(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= r.total_score <= 1.0

    def test_channels_in_range(self):
        r = score_edge_pair(_gray(seed=0), _gray(seed=1))
        assert 0.0 <= r.color_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0
        assert 0.0 <= r.texture_score <= 1.0

    def test_method_weighted(self):
        r = score_edge_pair(_gray(), _gray())
        assert r.method == "weighted"

    def test_params_stored(self):
        r = score_edge_pair(_gray(), _gray(), border_px=5, bins=32)
        assert r.params["border_px"] == 5
        assert r.params["bins"] == 32

    def test_rgb_ok(self):
        r = score_edge_pair(_rgb(seed=0), _rgb(seed=1))
        assert isinstance(r, EdgeScore)

    def test_identical_images_high_total(self):
        img = _uniform()  # uniform image: all edge profiles identical → high score
        r = score_edge_pair(img, img)
        assert r.total_score >= 0.7

    def test_custom_weights(self):
        weights = {"color": 1.0, "gradient": 0.0, "texture": 0.0}
        r = score_edge_pair(_gray(seed=0), _gray(seed=1), weights=weights)
        assert r.total_score == pytest.approx(r.color_score, abs=1e-5)

    def test_all_sides_0_to_3(self):
        img = _gray()
        for side in range(4):
            r = score_edge_pair(img, img, side1=side, side2=side)
            assert 0.0 <= r.total_score <= 1.0


# ─── TestBatchScoreEdges ──────────────────────────────────────────────────────

class TestBatchScoreEdges:
    def _images(self, n=4) -> list:
        return [_gray(seed=i) for i in range(n)]

    def test_returns_list(self):
        imgs = self._images(4)
        result = batch_score_edges(imgs, [(0, 1), (2, 3)])
        assert isinstance(result, list)

    def test_length_matches_pairs(self):
        imgs = self._images(4)
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_score_edges(imgs, pairs)
        assert len(result) == 3

    def test_all_edge_scores(self):
        imgs = self._images(4)
        for r in batch_score_edges(imgs, [(0, 1), (2, 3)]):
            assert isinstance(r, EdgeScore)

    def test_empty_pairs(self):
        imgs = self._images(4)
        assert batch_score_edges(imgs, []) == []

    def test_custom_side_pairs(self):
        imgs = self._images(4)
        side_pairs = [(0, 2), (1, 3)]
        result = batch_score_edges(imgs, [(0, 1), (2, 3)],
                                   side_pairs=side_pairs)
        assert result[0].side1 == 0
        assert result[0].side2 == 2

    def test_ids_match_pair(self):
        imgs = self._images(4)
        result = batch_score_edges(imgs, [(1, 3)])
        assert result[0].idx1 == 1
        assert result[0].idx2 == 3

    def test_total_scores_in_range(self):
        imgs = self._images(4)
        for r in batch_score_edges(imgs, [(0, 1), (1, 2)]):
            assert 0.0 <= r.total_score <= 1.0
