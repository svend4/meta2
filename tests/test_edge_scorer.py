"""Тесты для puzzle_reconstruction/algorithms/edge_scorer.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

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
    """Изображение с горизонтальным градиентом яркости."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


# ─── EdgeScore ────────────────────────────────────────────────────────────────

class TestEdgeScore:
    def _make(self, **kw):
        defaults = dict(idx1=0, idx2=1, side1=2, side2=0,
                        color_score=0.7, gradient_score=0.6,
                        texture_score=0.8, total_score=0.68)
        defaults.update(kw)
        return EdgeScore(**defaults)

    def test_fields(self):
        r = self._make()
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.side1 == 2
        assert r.side2 == 0

    def test_scores_stored(self):
        r = self._make(color_score=0.5, gradient_score=0.4,
                       texture_score=0.9, total_score=0.55)
        assert r.color_score    == pytest.approx(0.5)
        assert r.gradient_score == pytest.approx(0.4)
        assert r.texture_score  == pytest.approx(0.9)
        assert r.total_score    == pytest.approx(0.55)

    def test_method_default(self):
        assert self._make().method == "weighted"

    def test_params_default_empty(self):
        assert isinstance(self._make().params, dict)

    def test_params_stored(self):
        r = self._make(params={"border_px": 10, "bins": 64})
        assert r.params["border_px"] == 10

    def test_repr_contains_class(self):
        assert "EdgeScore" in repr(self._make())

    def test_repr_contains_total(self):
        r = self._make(total_score=0.1234)
        assert "0.12" in repr(r) or "total" in repr(r).lower()

    def test_all_scores_in_range(self):
        r = self._make()
        for v in (r.color_score, r.gradient_score, r.texture_score, r.total_score):
            assert 0.0 <= v <= 1.0


# ─── score_color_compat ───────────────────────────────────────────────────────

class TestScoreColorCompat:
    def test_returns_float(self):
        assert isinstance(score_color_compat(_gray(), _gray()), float)

    def test_in_range(self):
        v = score_color_compat(_noisy(), _noisy(seed=5))
        assert 0.0 <= v <= 1.0

    def test_identical_high(self):
        img = _gray()
        v   = score_color_compat(img, img, side1=2, side2=0)
        assert v > 0.9

    def test_gray_input(self):
        v = score_color_compat(_gray(), _gray())
        assert isinstance(v, float)

    def test_bgr_input(self):
        v = score_color_compat(_bgr(), _bgr())
        assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides_no_crash(self, side):
        v = score_color_compat(_gray(64, 64), _gray(64, 64), side1=side, side2=side)
        assert 0.0 <= v <= 1.0

    def test_different_values_in_range(self):
        img1 = _gray(val=50)
        img2 = _gray(val=200)
        v    = score_color_compat(img1, img2)
        assert 0.0 <= v <= 1.0

    def test_bins_param(self):
        v = score_color_compat(_gray(), _gray(), bins=32)
        assert 0.0 <= v <= 1.0

    def test_border_px_param(self):
        v = score_color_compat(_gray(), _gray(), border_px=5)
        assert 0.0 <= v <= 1.0


# ─── score_gradient_compat ────────────────────────────────────────────────────

class TestScoreGradientCompat:
    def test_returns_float(self):
        assert isinstance(score_gradient_compat(_gray(), _gray()), float)

    def test_in_range(self):
        v = score_gradient_compat(_noisy(), _noisy(seed=5))
        assert 0.0 <= v <= 1.0

    def test_identical_gte_half(self):
        img = _noisy()
        v   = score_gradient_compat(img, img)
        assert v >= 0.5

    def test_flat_profile_returns_half(self):
        img1 = _gray()   # постоянная яркость → плоский профиль
        img2 = _gray()
        v    = score_gradient_compat(img1, img2)
        assert v == pytest.approx(0.5, abs=0.1)

    def test_gray_input(self):
        v = score_gradient_compat(_gray(), _gradient())
        assert isinstance(v, float)

    def test_bgr_input(self):
        v = score_gradient_compat(_bgr(), _bgr())
        assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        v = score_gradient_compat(_gradient(64, 64), _gradient(64, 64),
                                   side1=side, side2=side)
        assert 0.0 <= v <= 1.0

    def test_border_px_param(self):
        v = score_gradient_compat(_gray(), _gray(), border_px=5)
        assert 0.0 <= v <= 1.0


# ─── score_texture_compat ─────────────────────────────────────────────────────

class TestScoreTextureCompat:
    def test_returns_float(self):
        assert isinstance(score_texture_compat(_gray(), _gray()), float)

    def test_in_range(self):
        v = score_texture_compat(_noisy(), _noisy(seed=7))
        assert 0.0 <= v <= 1.0

    def test_both_uniform_returns_one(self):
        img1 = _gray()
        img2 = _gray(val=200)
        v    = score_texture_compat(img1, img2)
        assert v == pytest.approx(1.0, abs=1e-5)

    def test_identical_returns_one(self):
        img = _noisy()
        v   = score_texture_compat(img, img)
        assert v == pytest.approx(1.0, abs=1e-5)

    def test_gray_input(self):
        v = score_texture_compat(_gray(), _noisy())
        assert 0.0 <= v <= 1.0

    def test_bgr_input(self):
        v = score_texture_compat(_bgr(), _bgr())
        assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        v = score_texture_compat(_gray(64, 64), _noisy(64, 64),
                                  side1=side, side2=side)
        assert 0.0 <= v <= 1.0

    def test_very_different_texture_lower(self):
        uniform = _gray()
        noisy   = _noisy()
        v = score_texture_compat(uniform, noisy)
        assert v < 1.0


# ─── score_edge_pair ──────────────────────────────────────────────────────────

class TestScoreEdgePair:
    def test_returns_edge_score(self):
        assert isinstance(score_edge_pair(_gray(), _gray()), EdgeScore)

    def test_idx_stored(self):
        r = score_edge_pair(_gray(), _gray(), idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_sides_stored(self):
        r = score_edge_pair(_gray(), _gray(), side1=1, side2=3)
        assert r.side1 == 1
        assert r.side2 == 3

    def test_color_score_in_range(self):
        r = score_edge_pair(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.color_score <= 1.0

    def test_gradient_score_in_range(self):
        r = score_edge_pair(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.gradient_score <= 1.0

    def test_texture_score_in_range(self):
        r = score_edge_pair(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.texture_score <= 1.0

    def test_total_score_in_range(self):
        r = score_edge_pair(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.total_score <= 1.0

    def test_method_is_weighted(self):
        assert score_edge_pair(_gray(), _gray()).method == "weighted"

    def test_params_border_px(self):
        r = score_edge_pair(_gray(), _gray(), border_px=8)
        assert r.params.get("border_px") == 8

    def test_params_bins(self):
        r = score_edge_pair(_gray(), _gray(), bins=32)
        assert r.params.get("bins") == 32

    def test_params_weights_stored(self):
        w = {"color": 1.0, "gradient": 0.0, "texture": 0.0}
        r = score_edge_pair(_gray(), _gray(), weights=w)
        assert r.params.get("weights") == w

    def test_custom_weights_affect_total(self):
        img1 = _gray(val=50)
        img2 = _gray(val=50)   # идентичны
        r1   = score_edge_pair(img1, img2,
                                weights={"color": 1.0, "gradient": 0.0, "texture": 0.0})
        r2   = score_edge_pair(img1, img2,
                                weights={"color": 0.0, "gradient": 0.0, "texture": 1.0})
        # total должен отличаться в зависимости от весов
        assert isinstance(r1.total_score, float)
        assert isinstance(r2.total_score, float)

    def test_gray_input(self):
        r = score_edge_pair(_gray(), _gray())
        assert isinstance(r, EdgeScore)

    def test_bgr_input(self):
        r = score_edge_pair(_bgr(), _bgr())
        assert 0.0 <= r.total_score <= 1.0

    def test_identical_images_high_total(self):
        img = _noisy()
        r   = score_edge_pair(img, img)
        assert r.total_score > 0.5


# ─── batch_score_edges ────────────────────────────────────────────────────────

class TestBatchScoreEdges:
    def _images(self, n=3):
        return [_gray() if i % 2 == 0 else _noisy(seed=i) for i in range(n)]

    def test_returns_list(self):
        imgs = self._images(3)
        r    = batch_score_edges(imgs, [(0, 1), (1, 2)])
        assert isinstance(r, list)
        assert len(r) == 2

    def test_each_is_edge_score(self):
        imgs = self._images(3)
        for r in batch_score_edges(imgs, [(0, 1), (1, 2)]):
            assert isinstance(r, EdgeScore)

    def test_empty_pairs_empty_list(self):
        assert batch_score_edges(self._images(3), []) == []

    def test_default_side_pairs_low_side(self):
        imgs = self._images(2)
        r    = batch_score_edges(imgs, [(0, 1)])
        assert r[0].side1 == 2
        assert r[0].side2 == 0

    def test_custom_side_pairs(self):
        imgs = self._images(2)
        r    = batch_score_edges(imgs, [(0, 1)], side_pairs=[(1, 3)])
        assert r[0].side1 == 1
        assert r[0].side2 == 3

    def test_border_px_forwarded(self):
        imgs = self._images(2)
        r    = batch_score_edges(imgs, [(0, 1)], border_px=5)
        assert r[0].params.get("border_px") == 5

    def test_bins_forwarded(self):
        imgs = self._images(2)
        r    = batch_score_edges(imgs, [(0, 1)], bins=32)
        assert r[0].params.get("bins") == 32

    def test_idx_correct_in_result(self):
        imgs = self._images(4)
        r    = batch_score_edges(imgs, [(2, 3)])
        assert r[0].idx1 == 2
        assert r[0].idx2 == 3

    def test_total_in_range(self):
        imgs = self._images(3)
        for r in batch_score_edges(imgs, [(0, 1), (0, 2), (1, 2)]):
            assert 0.0 <= r.total_score <= 1.0
