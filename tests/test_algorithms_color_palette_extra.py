"""Extra tests for puzzle_reconstruction/algorithms/color_palette.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.color_palette import (
    ColorPalette,
    ColorPaletteConfig,
    batch_compute_palettes,
    compute_palette,
    extract_dominant_colors,
    palette_distance,
    rank_by_palette,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=16, w=16, val=100):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=16, w=16, color=(50, 100, 150)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _gradient(h=16, w=32):
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def _palette(colors, weights, frag_id=0):
    c = np.array(colors, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    return ColorPalette(fragment_id=frag_id, colors=c, weights=w, n_colors=len(colors))


# ─── ColorPaletteConfig (extra) ───────────────────────────────────────────────

class TestColorPaletteConfigExtra:
    def test_seed_stored(self):
        cfg = ColorPaletteConfig(seed=123)
        assert cfg.seed == 123

    def test_tol_large_valid(self):
        cfg = ColorPaletteConfig(tol=1.0)
        assert cfg.tol == pytest.approx(1.0)

    def test_max_iter_large_valid(self):
        cfg = ColorPaletteConfig(max_iter=1000)
        assert cfg.max_iter == 1000

    def test_n_colors_large_valid(self):
        cfg = ColorPaletteConfig(n_colors=64)
        assert cfg.n_colors == 64

    def test_default_n_colors_8(self):
        assert ColorPaletteConfig().n_colors == 8

    def test_default_max_iter_20(self):
        assert ColorPaletteConfig().max_iter == 20

    def test_default_seed_0(self):
        assert ColorPaletteConfig().seed == 0


# ─── ColorPalette (extra) ─────────────────────────────────────────────────────

class TestColorPaletteExtra:
    def test_dominant_returns_array(self):
        p = _palette([[10.0, 20.0], [30.0, 40.0]], [0.3, 0.7])
        assert isinstance(p.dominant, np.ndarray)

    def test_dominant_is_highest_weight(self):
        p = _palette([[0.0], [255.0], [128.0]], [0.1, 0.7, 0.2])
        assert p.dominant[0] == pytest.approx(255.0)

    def test_n_colors_matches_colors_len(self):
        p = _palette([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.3, 0.3, 0.4])
        assert p.n_colors == 3

    def test_large_fragment_id(self):
        p = _palette([[1.0], [2.0]], [0.5, 0.5], frag_id=999)
        assert p.fragment_id == 999

    def test_colors_shape_correct(self):
        p = _palette([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0.5, 0.5])
        assert p.colors.shape == (2, 3)

    def test_weights_shape_1d(self):
        p = _palette([[1.0], [2.0]], [0.6, 0.4])
        assert p.weights.ndim == 1

    def test_fragment_id_zero_valid(self):
        p = _palette([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5], frag_id=0)
        assert p.fragment_id == 0


# ─── extract_dominant_colors (extra) ──────────────────────────────────────────

class TestExtractDominantColorsExtra:
    def test_n_colors_2_valid(self):
        colors, weights = extract_dominant_colors(_gray(), n_colors=2)
        assert colors.shape[0] == 2
        assert weights.shape[0] == 2

    def test_gradient_image_weights_summed(self):
        colors, weights = extract_dominant_colors(_gradient(), n_colors=4)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_bgr_image_3_channels(self):
        colors, weights = extract_dominant_colors(_bgr(), n_colors=3)
        assert colors.shape == (3, 3)

    def test_grayscale_1_channel(self):
        colors, _ = extract_dominant_colors(_gray(), n_colors=3)
        assert colors.shape[1] == 1

    def test_seed_reproducibility(self):
        img = _gradient()
        c1, w1 = extract_dominant_colors(img, n_colors=3, seed=42)
        c2, w2 = extract_dominant_colors(img, n_colors=3, seed=42)
        np.testing.assert_array_equal(c1, c2)

    def test_colors_values_nonneg(self):
        colors, _ = extract_dominant_colors(_gray(val=200), n_colors=3)
        assert (colors >= 0).all()

    def test_weights_nonneg(self):
        _, weights = extract_dominant_colors(_gradient(), n_colors=4)
        assert (weights >= 0).all()

    def test_different_seeds_may_differ(self):
        img = _gradient()
        c1, _ = extract_dominant_colors(img, n_colors=4, seed=1)
        c2, _ = extract_dominant_colors(img, n_colors=4, seed=99)
        # They could be same or different; just check types
        assert c1.shape == c2.shape


# ─── palette_distance (extra) ─────────────────────────────────────────────────

class TestPaletteDistanceExtra:
    def test_self_distance_zero(self):
        p = _palette([[50.0, 100.0, 150.0], [200.0, 10.0, 5.0]], [0.7, 0.3])
        assert palette_distance(p, p) == pytest.approx(0.0)

    def test_symmetric(self):
        p1 = _palette([[0.0, 0.0], [100.0, 200.0]], [0.5, 0.5])
        p2 = _palette([[50.0, 50.0], [150.0, 150.0]], [0.4, 0.6])
        d12 = palette_distance(p1, p2)
        d21 = palette_distance(p2, p1)
        assert d12 == pytest.approx(d21)

    def test_different_palettes_positive(self):
        p1 = _palette([[0.0], [0.0]], [1.0, 0.0])
        p2 = _palette([[200.0], [200.0]], [1.0, 0.0])
        assert palette_distance(p1, p2) > 0.0

    def test_distance_returns_float(self):
        p = _palette([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5])
        assert isinstance(palette_distance(p, p), float)

    def test_distance_nonneg(self):
        rng = np.random.default_rng(0)
        c1 = rng.uniform(0, 255, (4, 3)).astype(np.float32)
        c2 = rng.uniform(0, 255, (4, 3)).astype(np.float32)
        w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        p1 = ColorPalette(fragment_id=0, colors=c1, weights=w, n_colors=4)
        p2 = ColorPalette(fragment_id=1, colors=c2, weights=w, n_colors=4)
        assert palette_distance(p1, p2) >= 0.0

    def test_identical_weights_different_colors(self):
        p1 = _palette([[0.0], [255.0]], [0.5, 0.5])
        p2 = _palette([[100.0], [200.0]], [0.5, 0.5])
        d = palette_distance(p1, p2)
        assert d > 0.0


# ─── compute_palette (extra) ──────────────────────────────────────────────────

class TestComputePaletteExtra:
    def test_returns_color_palette(self):
        result = compute_palette(_gray())
        assert isinstance(result, ColorPalette)

    def test_default_fragment_id_zero(self):
        result = compute_palette(_gray())
        assert result.fragment_id == 0

    def test_custom_fragment_id(self):
        result = compute_palette(_gray(), fragment_id=7)
        assert result.fragment_id == 7

    def test_bgr_colors_3_channels(self):
        result = compute_palette(_bgr())
        assert result.colors.shape[1] == 3

    def test_gray_colors_1_channel(self):
        result = compute_palette(_gray())
        assert result.colors.shape[1] == 1

    def test_weights_sum_to_one(self):
        result = compute_palette(_gradient())
        assert float(result.weights.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_n_colors_custom_config(self):
        cfg = ColorPaletteConfig(n_colors=3)
        result = compute_palette(_gray(), cfg=cfg)
        assert result.n_colors == 3

    def test_params_has_n_pixels(self):
        result = compute_palette(_gray(h=8, w=8))
        assert "n_pixels" in result.params
        assert result.params["n_pixels"] == 64

    def test_params_has_img_shape(self):
        result = compute_palette(_gray(h=5, w=7))
        assert "img_shape" in result.params
        assert result.params["img_shape"][:2] == (5, 7)


# ─── batch_compute_palettes (extra) ───────────────────────────────────────────

class TestBatchComputePalettesExtra:
    def test_single_image_returns_one_palette(self):
        result = batch_compute_palettes([_gray()])
        assert len(result) == 1

    def test_fragment_ids_assigned_sequentially(self):
        imgs = [_gray(), _bgr(), _gradient()]
        result = batch_compute_palettes(imgs)
        for i, p in enumerate(result):
            assert p.fragment_id == i

    def test_mixed_gray_bgr(self):
        result = batch_compute_palettes([_gray(), _bgr()])
        assert len(result) == 2

    def test_custom_config_n_colors(self):
        cfg = ColorPaletteConfig(n_colors=2)
        result = batch_compute_palettes([_gray(), _gray()], cfg=cfg)
        for p in result:
            assert p.n_colors == 2


# ─── rank_by_palette (extra) ──────────────────────────────────────────────────

class TestRankByPaletteExtra:
    def test_self_is_most_similar(self):
        query = _palette([[50.0, 50.0], [200.0, 200.0]], [0.5, 0.5])
        other = _palette([[0.0, 0.0], [255.0, 255.0]], [0.5, 0.5], frag_id=1)
        result = rank_by_palette(query, [query, other])
        # First result should be query itself (highest similarity)
        assert result[0][1] >= result[1][1]

    def test_returns_list(self):
        query = _palette([[1.0], [2.0]], [0.5, 0.5])
        result = rank_by_palette(query, [query])
        assert isinstance(result, list)

    def test_tuple_format(self):
        query = _palette([[1.0], [2.0]], [0.5, 0.5])
        result = rank_by_palette(query, [query])
        idx, sim = result[0]
        assert isinstance(idx, int)
        assert isinstance(sim, float)

    def test_default_indices_are_0_to_n(self):
        query = _palette([[1.0], [2.0]], [0.5, 0.5])
        cands = [_palette([[1.0], [2.0]], [0.5, 0.5], frag_id=i) for i in range(3)]
        result = rank_by_palette(query, cands)
        returned_indices = [r[0] for r in result]
        assert sorted(returned_indices) == [0, 1, 2]

    def test_similarity_in_0_1(self):
        query = _palette([[0.0, 0.0], [255.0, 255.0]], [0.5, 0.5])
        cands = [
            _palette([[10.0, 10.0], [240.0, 240.0]], [0.5, 0.5]),
            _palette([[100.0, 100.0], [150.0, 150.0]], [0.5, 0.5]),
        ]
        result = rank_by_palette(query, cands)
        for _, sim in result:
            assert 0.0 <= sim <= 1.0

    def test_custom_index_mapped(self):
        query = _palette([[1.0], [2.0]], [0.5, 0.5])
        cand = _palette([[1.0], [2.0]], [0.5, 0.5])
        result = rank_by_palette(query, [cand], indices=[99])
        assert result[0][0] == 99

    def test_sorted_descending_by_similarity(self):
        query = _palette([[0.0], [100.0]], [0.5, 0.5])
        near = _palette([[5.0], [95.0]], [0.5, 0.5], frag_id=0)
        far = _palette([[255.0], [0.0]], [0.5, 0.5], frag_id=1)
        result = rank_by_palette(query, [far, near])
        similarities = [r[1] for r in result]
        assert similarities[0] >= similarities[1]
