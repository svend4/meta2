"""Extra tests for puzzle_reconstruction.algorithms.edge_scorer."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64, b=200, g=120, r=60):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _noisy(h=64, w=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _gradient(h=64, w=64):
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


# ─── TestEdgeScoreExtra ─────────────────────────────────────────────────────

class TestEdgeScoreExtra:
    def _make(self, **kw):
        defaults = dict(idx1=0, idx2=1, side1=2, side2=0,
                        color_score=0.7, gradient_score=0.6,
                        texture_score=0.8, total_score=0.68)
        defaults.update(kw)
        return EdgeScore(**defaults)

    def test_perfect_scores(self):
        r = self._make(color_score=1.0, gradient_score=1.0,
                       texture_score=1.0, total_score=1.0)
        assert r.total_score == pytest.approx(1.0)

    def test_zero_scores(self):
        r = self._make(color_score=0.0, gradient_score=0.0,
                       texture_score=0.0, total_score=0.0)
        assert r.total_score == pytest.approx(0.0)

    def test_idx_large(self):
        r = self._make(idx1=999, idx2=1000)
        assert r.idx1 == 999
        assert r.idx2 == 1000

    def test_side_0_to_3_ok(self):
        for s in range(4):
            r = self._make(side1=s, side2=s)
            assert r.side1 == s
            assert r.side2 == s

    def test_method_weighted(self):
        assert self._make().method == "weighted"

    def test_params_border_px(self):
        r = self._make(params={"border_px": 16})
        assert r.params["border_px"] == 16

    def test_params_bins(self):
        r = self._make(params={"bins": 128})
        assert r.params["bins"] == 128

    def test_repr_is_string(self):
        assert isinstance(repr(self._make()), str)

    def test_all_scores_floats(self):
        r = self._make()
        for v in (r.color_score, r.gradient_score,
                  r.texture_score, r.total_score):
            assert isinstance(v, float)


# ─── TestScoreColorCompatExtra ──────────────────────────────────────────────

class TestScoreColorCompatExtra:
    def test_black_vs_white(self):
        img1 = _gray(val=0)
        img2 = _gray(val=255)
        v = score_color_compat(img1, img2)
        assert 0.0 <= v <= 1.0

    def test_same_bgr(self):
        img = _bgr()
        v = score_color_compat(img, img)
        assert v > 0.9

    def test_bins_16(self):
        v = score_color_compat(_noisy(), _noisy(seed=5), bins=16)
        assert 0.0 <= v <= 1.0

    def test_bins_256(self):
        v = score_color_compat(_noisy(), _noisy(seed=5), bins=256)
        assert 0.0 <= v <= 1.0

    def test_border_px_1(self):
        v = score_color_compat(_gray(), _gray(), border_px=1)
        assert 0.0 <= v <= 1.0

    def test_border_px_20(self):
        v = score_color_compat(_gray(128, 128), _gray(128, 128), border_px=20)
        assert 0.0 <= v <= 1.0

    def test_side_0(self):
        v = score_color_compat(_gray(), _gray(), side1=0, side2=0)
        assert 0.0 <= v <= 1.0

    def test_side_1(self):
        v = score_color_compat(_gray(), _gray(), side1=1, side2=1)
        assert 0.0 <= v <= 1.0

    def test_side_3(self):
        v = score_color_compat(_gray(), _gray(), side1=3, side2=3)
        assert 0.0 <= v <= 1.0

    def test_large_image(self):
        img = _gray(128, 128)
        v = score_color_compat(img, img)
        assert v > 0.9


# ─── TestScoreGradientCompatExtra ──────────────────────────────────────────

class TestScoreGradientCompatExtra:
    def test_gradient_vs_gradient(self):
        g = _gradient()
        v = score_gradient_compat(g, g)
        assert v >= 0.5

    def test_gradient_vs_flat(self):
        g1 = _gradient()
        g2 = _gray()
        v = score_gradient_compat(g1, g2)
        assert 0.0 <= v <= 1.0

    def test_noisy_identical(self):
        n = _noisy()
        v = score_gradient_compat(n, n)
        assert v >= 0.5

    def test_border_px_small(self):
        v = score_gradient_compat(_noisy(), _noisy(seed=9), border_px=2)
        assert 0.0 <= v <= 1.0

    def test_bgr_identical(self):
        b = _bgr()
        v = score_gradient_compat(b, b)
        assert 0.0 <= v <= 1.0

    def test_side_2(self):
        v = score_gradient_compat(_gray(), _gray(), side1=2, side2=2)
        assert 0.0 <= v <= 1.0

    def test_side_3(self):
        v = score_gradient_compat(_gradient(), _gradient(), side1=3, side2=3)
        assert 0.0 <= v <= 1.0


# ─── TestScoreTextureCompatExtra ────────────────────────────────────────────

class TestScoreTextureCompatExtra:
    def test_noisy_vs_noisy_diff_seeds(self):
        v = score_texture_compat(_noisy(seed=0), _noisy(seed=42))
        assert 0.0 <= v <= 1.0

    def test_bgr_same(self):
        b = _bgr()
        v = score_texture_compat(b, b)
        assert v == pytest.approx(1.0, abs=1e-5)

    def test_uniform_diff_vals_returns_one(self):
        v = score_texture_compat(_gray(val=50), _gray(val=200))
        assert v == pytest.approx(1.0, abs=1e-5)

    def test_side_0(self):
        v = score_texture_compat(_gray(), _noisy(), side1=0, side2=0)
        assert 0.0 <= v <= 1.0

    def test_side_1(self):
        v = score_texture_compat(_gray(), _noisy(), side1=1, side2=1)
        assert 0.0 <= v <= 1.0

    def test_side_2(self):
        v = score_texture_compat(_gray(), _noisy(), side1=2, side2=2)
        assert 0.0 <= v <= 1.0

    def test_large_image(self):
        v = score_texture_compat(_noisy(128, 128), _noisy(128, 128, seed=5))
        assert 0.0 <= v <= 1.0


# ─── TestScoreEdgePairExtra ─────────────────────────────────────────────────

class TestScoreEdgePairExtra:
    def test_all_scores_in_range_noisy(self):
        r = score_edge_pair(_noisy(seed=1), _noisy(seed=2))
        for v in (r.color_score, r.gradient_score,
                  r.texture_score, r.total_score):
            assert 0.0 <= v <= 1.0

    def test_identical_gray(self):
        img = _gray(val=100)
        r = score_edge_pair(img, img)
        assert r.total_score >= 0.5

    def test_identical_bgr(self):
        img = _bgr()
        r = score_edge_pair(img, img)
        assert r.total_score >= 0.5

    def test_color_only_weight(self):
        r = score_edge_pair(_gray(), _gray(),
                            weights={"color": 1.0, "gradient": 0.0,
                                     "texture": 0.0})
        assert r.total_score == pytest.approx(r.color_score, abs=1e-6)

    def test_texture_only_weight(self):
        r = score_edge_pair(_gray(), _gray(),
                            weights={"color": 0.0, "gradient": 0.0,
                                     "texture": 1.0})
        assert r.total_score == pytest.approx(r.texture_score, abs=1e-6)

    def test_large_border_px(self):
        r = score_edge_pair(_gray(128, 128), _gray(128, 128), border_px=20)
        assert isinstance(r, EdgeScore)

    def test_different_sides(self):
        for s in range(4):
            r = score_edge_pair(_gray(), _gray(), side1=s, side2=s)
            assert isinstance(r, EdgeScore)

    def test_bins_32(self):
        r = score_edge_pair(_noisy(), _noisy(seed=4), bins=32)
        assert 0.0 <= r.total_score <= 1.0

    def test_weights_stored_in_params(self):
        w = {"color": 0.5, "gradient": 0.3, "texture": 0.2}
        r = score_edge_pair(_gray(), _gray(), weights=w)
        assert r.params["weights"] == w


# ─── TestBatchScoreEdgesExtra ───────────────────────────────────────────────

class TestBatchScoreEdgesExtra:
    def _imgs(self, n=4):
        return [_gray() if i % 2 == 0 else _noisy(seed=i)
                for i in range(n)]

    def test_single_pair(self):
        imgs = self._imgs(2)
        result = batch_score_edges(imgs, [(0, 1)])
        assert len(result) == 1

    def test_three_pairs(self):
        imgs = self._imgs(4)
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_score_edges(imgs, pairs)
        assert len(result) == 3

    def test_idx_stored(self):
        imgs = self._imgs(4)
        result = batch_score_edges(imgs, [(1, 3)])
        assert result[0].idx1 == 1
        assert result[0].idx2 == 3

    def test_side_pair_custom(self):
        imgs = self._imgs(2)
        result = batch_score_edges(imgs, [(0, 1)], side_pairs=[(0, 2)])
        assert result[0].side1 == 0
        assert result[0].side2 == 2

    def test_all_total_in_range(self):
        imgs = self._imgs(5)
        pairs = [(i, i + 1) for i in range(4)]
        result = batch_score_edges(imgs, pairs)
        for r in result:
            assert 0.0 <= r.total_score <= 1.0

    def test_bins_forwarded(self):
        imgs = self._imgs(2)
        result = batch_score_edges(imgs, [(0, 1)], bins=64)
        assert result[0].params.get("bins") == 64

    def test_border_px_forwarded(self):
        imgs = self._imgs(2)
        result = batch_score_edges(imgs, [(0, 1)], border_px=10)
        assert result[0].params.get("border_px") == 10

    def test_identical_high_score(self):
        img = _noisy(seed=7)
        result = batch_score_edges([img, img], [(0, 1)])
        assert result[0].total_score > 0.5
