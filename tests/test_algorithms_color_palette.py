"""Tests for algorithms/color_palette.py."""
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

def make_gray(h=20, w=20, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def make_bgr(h=20, w=20, color=(128, 64, 32)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def make_gradient_gray(h=20, w=20):
    img = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return img


def make_palette(colors, weights, frag_id=0, n_colors=None):
    c = np.array(colors, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    n = n_colors if n_colors is not None else len(colors)
    return ColorPalette(fragment_id=frag_id, colors=c, weights=w, n_colors=n)


# ─── ColorPaletteConfig ───────────────────────────────────────────────────────

class TestColorPaletteConfig:
    def test_defaults(self):
        cfg = ColorPaletteConfig()
        assert cfg.n_colors == 8
        assert cfg.max_iter == 20
        assert cfg.tol == pytest.approx(1e-4)
        assert cfg.seed == 0

    def test_n_colors_too_small_raises(self):
        with pytest.raises(ValueError, match="n_colors"):
            ColorPaletteConfig(n_colors=1)

    def test_n_colors_minimum_valid(self):
        cfg = ColorPaletteConfig(n_colors=2)
        assert cfg.n_colors == 2

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            ColorPaletteConfig(max_iter=0)

    def test_max_iter_negative_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            ColorPaletteConfig(max_iter=-1)

    def test_tol_negative_raises(self):
        with pytest.raises(ValueError, match="tol"):
            ColorPaletteConfig(tol=-0.001)

    def test_tol_zero_valid(self):
        cfg = ColorPaletteConfig(tol=0.0)
        assert cfg.tol == 0.0

    def test_custom_values(self):
        cfg = ColorPaletteConfig(n_colors=4, max_iter=10, tol=0.01, seed=42)
        assert cfg.n_colors == 4
        assert cfg.max_iter == 10
        assert cfg.seed == 42


# ─── ColorPalette ─────────────────────────────────────────────────────────────

class TestColorPalette:
    def test_basic_creation(self):
        p = make_palette([[100.0, 200.0, 50.0], [10.0, 20.0, 5.0]], [0.6, 0.4])
        assert p.fragment_id == 0
        assert p.n_colors == 2

    def test_fragment_id_negative_raises(self):
        with pytest.raises(ValueError, match="fragment_id"):
            make_palette([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0.5, 0.5], frag_id=-1)

    def test_n_colors_too_small_raises(self):
        with pytest.raises(ValueError, match="n_colors"):
            ColorPalette(
                fragment_id=0,
                colors=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                weights=np.array([1.0], dtype=np.float32),
                n_colors=1,
            )

    def test_colors_must_be_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            ColorPalette(
                fragment_id=0,
                colors=np.array([1.0, 2.0], dtype=np.float32),
                weights=np.array([0.5, 0.5], dtype=np.float32),
                n_colors=2,
            )

    def test_weights_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            ColorPalette(
                fragment_id=0,
                colors=np.array([[1.0], [2.0]], dtype=np.float32),
                weights=np.array([[0.5], [0.5]], dtype=np.float32),
                n_colors=2,
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Длины"):
            ColorPalette(
                fragment_id=0,
                colors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                weights=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                n_colors=2,
            )

    def test_colors_dtype_is_float32(self):
        p = make_palette([[100.0, 50.0], [200.0, 10.0]], [0.7, 0.3])
        assert p.colors.dtype == np.float32

    def test_weights_dtype_is_float32(self):
        p = make_palette([[100.0, 50.0], [200.0, 10.0]], [0.7, 0.3])
        assert p.weights.dtype == np.float32

    def test_dominant_returns_highest_weight_color(self):
        colors = [[10.0, 0.0], [200.0, 0.0], [50.0, 0.0]]
        weights = [0.1, 0.8, 0.1]
        p = make_palette(colors, weights)
        dom = p.dominant
        assert dom[0] == pytest.approx(200.0)

    def test_dominant_shape_is_1d(self):
        p = make_palette([[10.0, 20.0], [30.0, 40.0]], [0.4, 0.6])
        assert p.dominant.ndim == 1

    def test_grayscale_palette(self):
        # single-channel colors
        p = ColorPalette(
            fragment_id=0,
            colors=np.array([[100.0], [200.0]], dtype=np.float32),
            weights=np.array([0.3, 0.7], dtype=np.float32),
            n_colors=2,
        )
        assert p.colors.shape == (2, 1)


# ─── extract_dominant_colors ──────────────────────────────────────────────────

class TestExtractDominantColors:
    def test_n_colors_too_small_raises(self):
        with pytest.raises(ValueError, match="n_colors"):
            extract_dominant_colors(make_gray(), n_colors=1)

    def test_returns_tuple_of_two(self):
        result = extract_dominant_colors(make_gray())
        assert len(result) == 2

    def test_grayscale_colors_shape(self):
        colors, weights = extract_dominant_colors(make_gray(), n_colors=4)
        assert colors.shape == (4, 1)

    def test_bgr_colors_shape(self):
        colors, weights = extract_dominant_colors(make_bgr(), n_colors=4)
        assert colors.shape == (4, 3)

    def test_weights_shape(self):
        _, weights = extract_dominant_colors(make_gray(), n_colors=4)
        assert weights.shape == (4,)

    def test_weights_sum_to_one(self):
        _, weights = extract_dominant_colors(make_gradient_gray(), n_colors=4)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self):
        _, weights = extract_dominant_colors(make_gradient_gray(), n_colors=4)
        assert np.all(weights >= 0.0)

    def test_colors_dtype_float32(self):
        colors, _ = extract_dominant_colors(make_gray(), n_colors=4)
        assert colors.dtype == np.float32

    def test_weights_dtype_float32(self):
        _, weights = extract_dominant_colors(make_gray(), n_colors=4)
        assert weights.dtype == np.float32

    def test_reproducible_with_seed(self):
        img = make_gradient_gray()
        c1, w1 = extract_dominant_colors(img, n_colors=4, seed=7)
        c2, w2 = extract_dominant_colors(img, n_colors=4, seed=7)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(w1, w2)

    def test_constant_image_single_cluster(self):
        img = make_gray(val=128)
        colors, weights = extract_dominant_colors(img, n_colors=4)
        # dominant center should be near 128
        dominant_idx = int(np.argmax(weights))
        assert colors[dominant_idx, 0] == pytest.approx(128.0, abs=5.0)

    def test_3d_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            extract_dominant_colors(np.zeros((4, 4, 4, 4), dtype=np.uint8), n_colors=2)


# ─── palette_distance ─────────────────────────────────────────────────────────

class TestPaletteDistance:
    def _make(self, colors, weights, frag_id=0):
        c = np.array(colors, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        return ColorPalette(fragment_id=frag_id, colors=c, weights=w, n_colors=len(colors))

    def test_identical_palettes_zero_distance(self):
        p = self._make([[100.0, 50.0, 25.0], [200.0, 150.0, 100.0]], [0.6, 0.4])
        assert palette_distance(p, p) == pytest.approx(0.0)

    def test_different_n_colors_raises(self):
        p2 = self._make([[1.0], [2.0]], [0.5, 0.5])
        p3 = self._make([[1.0], [2.0], [3.0]], [0.33, 0.33, 0.34])
        with pytest.raises(ValueError, match="n_colors"):
            palette_distance(p2, p3)

    def test_distance_is_non_negative(self):
        p1 = self._make([[0.0, 0.0], [255.0, 255.0]], [0.5, 0.5])
        p2 = self._make([[100.0, 100.0], [200.0, 200.0]], [0.5, 0.5])
        dist = palette_distance(p1, p2)
        assert dist >= 0.0

    def test_distance_is_symmetric(self):
        p1 = self._make([[0.0], [255.0]], [0.3, 0.7])
        p2 = self._make([[100.0], [150.0]], [0.6, 0.4])
        assert palette_distance(p1, p2) == pytest.approx(palette_distance(p2, p1))

    def test_distance_returns_float(self):
        p = self._make([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5])
        assert isinstance(palette_distance(p, p), float)

    def test_different_palettes_positive_distance(self):
        p1 = self._make([[0.0], [0.0]], [1.0, 0.0])
        p2 = self._make([[255.0], [255.0]], [1.0, 0.0])
        assert palette_distance(p1, p2) > 0.0


# ─── compute_palette ──────────────────────────────────────────────────────────

class TestComputePalette:
    def test_returns_color_palette(self):
        result = compute_palette(make_gray())
        assert isinstance(result, ColorPalette)

    def test_fragment_id_set_correctly(self):
        result = compute_palette(make_gray(), fragment_id=5)
        assert result.fragment_id == 5

    def test_n_colors_matches_config(self):
        cfg = ColorPaletteConfig(n_colors=4)
        result = compute_palette(make_gray(), cfg=cfg)
        assert result.n_colors == 4

    def test_default_config(self):
        result = compute_palette(make_gray())
        assert result.n_colors == 8

    def test_bgr_image(self):
        result = compute_palette(make_bgr(), fragment_id=2)
        assert result.fragment_id == 2
        assert result.colors.shape[1] == 3

    def test_params_contains_n_pixels(self):
        result = compute_palette(make_gray(h=10, w=10))
        assert "n_pixels" in result.params
        assert result.params["n_pixels"] == 100

    def test_params_contains_img_shape(self):
        result = compute_palette(make_gray(h=10, w=20))
        assert "img_shape" in result.params
        h, w = result.params["img_shape"][:2]
        assert h == 10
        assert w == 20


# ─── batch_compute_palettes ───────────────────────────────────────────────────

class TestBatchComputePalettes:
    def test_empty_list_returns_empty(self):
        assert batch_compute_palettes([]) == []

    def test_length_preserved(self):
        imgs = [make_gray(), make_bgr(), make_gradient_gray()]
        result = batch_compute_palettes(imgs)
        assert len(result) == 3

    def test_fragment_ids_sequential(self):
        imgs = [make_gray(), make_gray(), make_gray()]
        result = batch_compute_palettes(imgs)
        for i, p in enumerate(result):
            assert p.fragment_id == i

    def test_all_are_color_palettes(self):
        imgs = [make_gray(), make_bgr()]
        result = batch_compute_palettes(imgs)
        for p in result:
            assert isinstance(p, ColorPalette)

    def test_custom_config_applied(self):
        cfg = ColorPaletteConfig(n_colors=3)
        imgs = [make_gray(), make_gray()]
        result = batch_compute_palettes(imgs, cfg=cfg)
        for p in result:
            assert p.n_colors == 3


# ─── rank_by_palette ──────────────────────────────────────────────────────────

class TestRankByPalette:
    def _make(self, colors, weights, frag_id=0):
        c = np.array(colors, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        return ColorPalette(fragment_id=frag_id, colors=c, weights=w, n_colors=len(colors))

    def test_empty_candidates_returns_empty(self):
        query = self._make([[100.0], [200.0]], [0.5, 0.5])
        assert rank_by_palette(query, []) == []

    def test_indices_length_mismatch_raises(self):
        query = self._make([[100.0], [200.0]], [0.5, 0.5])
        cands = [self._make([[100.0], [200.0]], [0.5, 0.5], frag_id=1)]
        with pytest.raises(ValueError, match="Длины"):
            rank_by_palette(query, cands, indices=[0, 1])

    def test_returns_list_of_tuples(self):
        query = self._make([[100.0], [200.0]], [0.5, 0.5])
        cand = self._make([[100.0], [200.0]], [0.5, 0.5])
        result = rank_by_palette(query, [cand])
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_sorted_descending(self):
        query = self._make([[0.0], [0.0]], [1.0, 0.0])
        near = self._make([[5.0], [0.0]], [1.0, 0.0], frag_id=0)
        far = self._make([[200.0], [0.0]], [1.0, 0.0], frag_id=1)
        result = rank_by_palette(query, [far, near])
        # near should appear first (higher similarity)
        assert result[0][1] >= result[1][1]

    def test_identical_query_and_candidate_max_similarity(self):
        query = self._make([[100.0, 50.0], [200.0, 10.0]], [0.6, 0.4])
        result = rank_by_palette(query, [query])
        _, sim = result[0]
        assert sim == pytest.approx(1.0)

    def test_custom_indices_used(self):
        query = self._make([[100.0], [200.0]], [0.5, 0.5])
        cand = self._make([[100.0], [200.0]], [0.5, 0.5])
        result = rank_by_palette(query, [cand], indices=[42])
        assert result[0][0] == 42

    def test_similarity_in_zero_one(self):
        query = self._make([[0.0, 0.0], [255.0, 255.0]], [0.5, 0.5])
        cands = [
            self._make([[10.0, 10.0], [245.0, 245.0]], [0.5, 0.5]),
            self._make([[50.0, 50.0], [200.0, 200.0]], [0.5, 0.5]),
        ]
        result = rank_by_palette(query, cands)
        for _, sim in result:
            assert 0.0 <= sim <= 1.0
