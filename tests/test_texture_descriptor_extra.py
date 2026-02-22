"""Additional tests for puzzle_reconstruction.algorithms.texture_descriptor."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.texture_descriptor import (
    TextureDescriptor,
    compute_lbp,
    compute_glcm_features,
    compute_stats_descriptor,
    compute_texture_descriptor,
    normalize_descriptor,
    descriptor_distance,
    batch_compute_descriptors,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const_gray(val=128, h=16, w=16):
    return np.full((h, w), val, dtype=np.uint8)


# ─── TestTextureDescriptorExtra ───────────────────────────────────────────────

class TestTextureDescriptorExtra:
    def test_image_id_nonzero_stored(self):
        td = TextureDescriptor(vector=np.ones(8), image_id=42)
        assert td.image_id == 42

    def test_large_vector_stored(self):
        v = np.random.default_rng(0).random(1024).astype(np.float32)
        td = TextureDescriptor(vector=v)
        assert len(td) == 1024

    def test_method_is_string(self):
        td = TextureDescriptor(vector=np.ones(4))
        assert isinstance(td.method, str)

    def test_float64_converted_to_float32(self):
        v = np.ones(6, dtype=np.float64)
        td = TextureDescriptor(vector=v)
        assert td.vector.dtype == np.float32

    def test_len_matches_vector_size(self):
        for n in (1, 10, 64, 256):
            td = TextureDescriptor(vector=np.zeros(n))
            assert len(td) == n

    def test_zero_image_id_ok(self):
        td = TextureDescriptor(vector=np.ones(4), image_id=0)
        assert td.image_id == 0


# ─── TestComputeLbpExtra ─────────────────────────────────────────────────────

class TestComputeLbpExtra:
    def test_small_image_8x8(self):
        img = _gray(h=8, w=8)
        hist = compute_lbp(img, n_bins=32)
        assert hist.shape == (32,)
        assert hist.dtype == np.float32

    def test_n_bins_16(self):
        hist = compute_lbp(_gray(), n_bins=16)
        assert hist.shape == (16,)

    def test_n_bins_128(self):
        hist = compute_lbp(_gray(), n_bins=128)
        assert hist.shape == (128,)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_color_image_same_shape_as_gray(self):
        h_g = compute_lbp(_gray(seed=0), n_bins=32)
        h_c = compute_lbp(_color(seed=0), n_bins=32)
        assert h_g.shape == h_c.shape

    def test_different_seeds_different_histograms(self):
        h1 = compute_lbp(_gray(seed=0), n_bins=64)
        h2 = compute_lbp(_gray(seed=99), n_bins=64)
        assert not np.allclose(h1, h2)

    def test_all_nonneg(self):
        hist = compute_lbp(_gray(seed=5), n_bins=64)
        assert np.all(hist >= 0)


# ─── TestComputeGlcmFeaturesExtra ─────────────────────────────────────────────

class TestComputeGlcmFeaturesExtra:
    def test_constant_image_returns_float32(self):
        feat = compute_glcm_features(_const_gray(), levels=8)
        assert feat.dtype == np.float32

    def test_levels_4_ok(self):
        feat = compute_glcm_features(_gray(), levels=4)
        assert feat.shape == (4,)

    def test_levels_32_ok(self):
        feat = compute_glcm_features(_gray(), levels=32)
        assert feat.shape == (4,)

    def test_color_input_works(self):
        feat = compute_glcm_features(_color(), levels=8)
        assert feat.dtype == np.float32
        assert feat.shape == (4,)

    def test_energy_nonneg(self):
        feat = compute_glcm_features(_gray(seed=2), levels=16)
        assert feat[2] >= 0.0

    def test_contrast_increases_with_noise(self):
        flat = _const_gray(val=128, h=32, w=32)
        noisy = _gray(h=32, w=32, seed=7)
        feat_flat = compute_glcm_features(flat, levels=8)
        feat_noisy = compute_glcm_features(noisy, levels=8)
        # noisy image should have higher contrast than flat
        assert feat_noisy[0] >= feat_flat[0] - 1e-6

    def test_all_values_finite(self):
        feat = compute_glcm_features(_gray(seed=4), levels=16)
        assert np.all(np.isfinite(feat))


# ─── TestComputeStatsDescriptorExtra ──────────────────────────────────────────

class TestComputeStatsDescriptorExtra:
    def test_small_image_gray(self):
        img = _gray(h=4, w=4)
        desc = compute_stats_descriptor(img)
        assert desc.shape == (2,)

    def test_all_zeros_gray(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        desc = compute_stats_descriptor(img)
        assert desc[0] == pytest.approx(0.0, abs=1e-5)
        assert desc[1] == pytest.approx(0.0, abs=1e-5)

    def test_all_255_mean(self):
        img = np.full((16, 16), 255, dtype=np.uint8)
        desc = compute_stats_descriptor(img)
        assert desc[0] == pytest.approx(255.0, abs=1e-4)

    def test_color_shape_6(self):
        img = _color()
        desc = compute_stats_descriptor(img)
        assert desc.shape == (6,)
        assert desc.dtype == np.float32

    def test_different_images_different_descs(self):
        d1 = compute_stats_descriptor(_gray(seed=0))
        d2 = compute_stats_descriptor(_gray(seed=99))
        assert not np.allclose(d1, d2)

    def test_values_finite(self):
        desc = compute_stats_descriptor(_color(seed=3))
        assert np.all(np.isfinite(desc))


# ─── TestComputeTextureDescriptorExtra ────────────────────────────────────────

class TestComputeTextureDescriptorExtra:
    def test_lbp_color_image(self):
        td = compute_texture_descriptor(_color(), method="lbp", lbp_bins=32)
        assert isinstance(td, TextureDescriptor)
        assert len(td) == 32

    def test_glcm_color_image(self):
        td = compute_texture_descriptor(_color(), method="glcm", glcm_levels=8)
        assert len(td) == 4

    def test_stats_color_image(self):
        td = compute_texture_descriptor(_color(), method="stats")
        assert len(td) == 6

    def test_combined_color(self):
        td = compute_texture_descriptor(_color(), method="combined",
                                        lbp_bins=16, glcm_levels=4)
        assert len(td) == 16 + 4 + 6

    def test_image_id_large(self):
        td = compute_texture_descriptor(_gray(), image_id=999)
        assert td.image_id == 999

    def test_method_stored_in_result(self):
        td = compute_texture_descriptor(_gray(), method="lbp", lbp_bins=32)
        assert td.method == "lbp"

    def test_output_always_float32(self):
        for method in ("lbp", "glcm", "stats", "combined"):
            td = compute_texture_descriptor(_gray(), method=method,
                                            lbp_bins=16, glcm_levels=4)
            assert td.vector.dtype == np.float32


# ─── TestNormalizeDescriptorExtra ────────────────────────────────────────────

class TestNormalizeDescriptorExtra:
    def test_already_normalized_stays_normalized(self):
        v = np.array([0.6, 0.8], dtype=np.float32)
        td = TextureDescriptor(vector=v)
        norm = normalize_descriptor(td)
        assert np.linalg.norm(norm.vector) == pytest.approx(1.0, abs=1e-6)

    def test_large_vector_normalized(self):
        v = np.random.default_rng(0).random(256).astype(np.float32) * 1000
        td = TextureDescriptor(vector=v)
        norm = normalize_descriptor(td)
        assert np.linalg.norm(norm.vector) == pytest.approx(1.0, abs=1e-5)

    def test_single_element(self):
        td = TextureDescriptor(vector=np.array([7.0]))
        norm = normalize_descriptor(td)
        assert norm.vector[0] == pytest.approx(1.0, abs=1e-6)

    def test_result_is_new_object(self):
        td = TextureDescriptor(vector=np.array([1.0, 2.0, 3.0]))
        norm = normalize_descriptor(td)
        assert norm is not td


# ─── TestDescriptorDistanceExtra ─────────────────────────────────────────────

class TestDescriptorDistanceExtra:
    def test_symmetry(self):
        a = TextureDescriptor(vector=np.array([1.0, 0.0, 0.0]))
        b = TextureDescriptor(vector=np.array([0.0, 1.0, 0.0]))
        assert descriptor_distance(a, b) == pytest.approx(descriptor_distance(b, a))

    def test_zero_distance_same_vector(self):
        v = np.random.default_rng(3).random(20).astype(np.float32)
        td = TextureDescriptor(vector=v)
        assert descriptor_distance(td, td) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_sqrt2(self):
        a = TextureDescriptor(vector=np.array([1.0, 0.0], dtype=np.float32))
        b = TextureDescriptor(vector=np.array([0.0, 1.0], dtype=np.float32))
        assert descriptor_distance(a, b) == pytest.approx(math.sqrt(2), abs=1e-5)

    def test_result_is_float(self):
        a = TextureDescriptor(vector=np.ones(5, dtype=np.float32))
        b = TextureDescriptor(vector=np.zeros(5, dtype=np.float32))
        assert isinstance(descriptor_distance(a, b), float)


import math


# ─── TestBatchComputeDescriptorsExtra ─────────────────────────────────────────

class TestBatchComputeDescriptorsExtra:
    def test_mixed_gray_color(self):
        images = [_gray(), _color()]
        result = batch_compute_descriptors(images, method="stats")
        assert len(result) == 2

    def test_single_image(self):
        result = batch_compute_descriptors([_gray()], method="lbp", lbp_bins=32)
        assert len(result) == 1
        assert isinstance(result[0], TextureDescriptor)

    def test_lbp_method_descriptor_length(self):
        images = [_gray(seed=i) for i in range(3)]
        result = batch_compute_descriptors(images, method="lbp", lbp_bins=64)
        assert all(len(td) == 64 for td in result)

    def test_ids_assigned_sequentially(self):
        images = [_gray(seed=i) for i in range(6)]
        result = batch_compute_descriptors(images, method="glcm", glcm_levels=8)
        for i, td in enumerate(result):
            assert td.image_id == i

    def test_all_float32(self):
        images = [_gray(seed=i) for i in range(4)]
        result = batch_compute_descriptors(images, method="stats")
        assert all(td.vector.dtype == np.float32 for td in result)
