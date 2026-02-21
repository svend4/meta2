"""Тесты для puzzle_reconstruction.algorithms.texture_descriptor."""
import pytest
import numpy as np

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=32, w=32):
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestTextureDescriptor ────────────────────────────────────────────────────

class TestTextureDescriptor:
    def test_basic_creation(self):
        td = TextureDescriptor(vector=np.ones(10, dtype=np.float32))
        assert len(td) == 10

    def test_dtype_float32(self):
        td = TextureDescriptor(vector=np.array([1.0, 2.0, 3.0]))
        assert td.vector.dtype == np.float32

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            TextureDescriptor(vector=np.ones((3, 3)))

    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError):
            TextureDescriptor(vector=np.ones(5), image_id=-1)

    def test_len(self):
        td = TextureDescriptor(vector=np.zeros(8))
        assert len(td) == 8

    def test_image_id_default_zero(self):
        td = TextureDescriptor(vector=np.ones(4))
        assert td.image_id == 0

    def test_method_stored(self):
        td = TextureDescriptor(vector=np.ones(4), method="lbp")
        assert td.method == "lbp"


# ─── TestComputeLbp ───────────────────────────────────────────────────────────

class TestComputeLbp:
    def test_returns_float32(self):
        hist = compute_lbp(_gray(), n_bins=32)
        assert hist.dtype == np.float32

    def test_shape(self):
        hist = compute_lbp(_gray(), n_bins=64)
        assert hist.shape == (64,)

    def test_sums_to_one(self):
        hist = compute_lbp(_gray(), n_bins=256)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_nonnegative(self):
        hist = compute_lbp(_gray(), n_bins=32)
        assert (hist >= 0).all()

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            compute_lbp(_gray(), n_bins=1)

    def test_color_image(self):
        hist = compute_lbp(_color(), n_bins=32)
        assert hist.shape == (32,)

    def test_constant_image(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        hist = compute_lbp(img, n_bins=256)
        # Константное изображение → один бин заполнен
        assert hist.max() > 0


# ─── TestComputeGlcmFeatures ──────────────────────────────────────────────────

class TestComputeGlcmFeatures:
    def test_returns_float32(self):
        feat = compute_glcm_features(_gray(), levels=16)
        assert feat.dtype == np.float32

    def test_shape_four(self):
        feat = compute_glcm_features(_gray(), levels=16)
        assert feat.shape == (4,)

    def test_contrast_nonnegative(self):
        feat = compute_glcm_features(_gray(), levels=16)
        assert feat[0] >= 0.0

    def test_homogeneity_in_range(self):
        feat = compute_glcm_features(_gray(), levels=16)
        assert 0.0 <= feat[1] <= 1.0 + 1e-6

    def test_energy_in_range(self):
        feat = compute_glcm_features(_gray(), levels=8)
        assert 0.0 <= feat[2] <= 1.0 + 1e-6

    def test_levels_too_small_raises(self):
        with pytest.raises(ValueError):
            compute_glcm_features(_gray(), levels=1)

    def test_distance_zero_raises(self):
        with pytest.raises(ValueError):
            compute_glcm_features(_gray(), levels=8, distance=0)

    def test_color_image(self):
        feat = compute_glcm_features(_color(), levels=8)
        assert feat.shape == (4,)


# ─── TestComputeStatsDescriptor ───────────────────────────────────────────────

class TestComputeStatsDescriptor:
    def test_gray_image_two_values(self):
        desc = compute_stats_descriptor(_gray())
        assert desc.shape == (2,)

    def test_color_image_six_values(self):
        desc = compute_stats_descriptor(_color())
        assert desc.shape == (6,)

    def test_returns_float32(self):
        desc = compute_stats_descriptor(_gray())
        assert desc.dtype == np.float32

    def test_mean_in_valid_range(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        desc = compute_stats_descriptor(img)
        assert desc[0] == pytest.approx(100.0)

    def test_constant_image_zero_std(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        desc = compute_stats_descriptor(img)
        assert desc[1] == pytest.approx(0.0, abs=1e-5)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            compute_stats_descriptor(np.ones((2, 2, 2, 2), dtype=np.uint8))


# ─── TestComputeTextureDescriptor ─────────────────────────────────────────────

class TestComputeTextureDescriptor:
    def test_lbp_method(self):
        td = compute_texture_descriptor(_gray(), method="lbp", lbp_bins=32)
        assert isinstance(td, TextureDescriptor)
        assert len(td) == 32

    def test_glcm_method(self):
        td = compute_texture_descriptor(_gray(), method="glcm", glcm_levels=8)
        assert len(td) == 4

    def test_stats_method_gray(self):
        td = compute_texture_descriptor(_gray(), method="stats")
        assert len(td) == 2

    def test_combined_method(self):
        td = compute_texture_descriptor(_gray(), method="combined",
                                        lbp_bins=32, glcm_levels=8)
        assert len(td) == 32 + 4 + 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            compute_texture_descriptor(_gray(), method="unknown")

    def test_image_id_stored(self):
        td = compute_texture_descriptor(_gray(), image_id=7)
        assert td.image_id == 7

    def test_output_dtype_float32(self):
        td = compute_texture_descriptor(_gray())
        assert td.vector.dtype == np.float32


# ─── TestNormalizeDescriptor ──────────────────────────────────────────────────

class TestNormalizeDescriptor:
    def test_l2_norm_one(self):
        td = TextureDescriptor(vector=np.array([3.0, 4.0], dtype=np.float32))
        norm_td = normalize_descriptor(td)
        assert np.linalg.norm(norm_td.vector) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_unchanged(self):
        td = TextureDescriptor(vector=np.zeros(4, dtype=np.float32))
        norm_td = normalize_descriptor(td)
        np.testing.assert_array_equal(norm_td.vector, np.zeros(4, dtype=np.float32))

    def test_method_preserved(self):
        td = TextureDescriptor(vector=np.ones(4), method="lbp")
        norm_td = normalize_descriptor(td)
        assert norm_td.method == "lbp"

    def test_image_id_preserved(self):
        td = TextureDescriptor(vector=np.ones(4), image_id=5)
        norm_td = normalize_descriptor(td)
        assert norm_td.image_id == 5

    def test_output_float32(self):
        td = TextureDescriptor(vector=np.array([1.0, 2.0, 3.0]))
        norm_td = normalize_descriptor(td)
        assert norm_td.vector.dtype == np.float32


# ─── TestDescriptorDistance ───────────────────────────────────────────────────

class TestDescriptorDistance:
    def test_same_descriptor_zero(self):
        td = TextureDescriptor(vector=np.array([1.0, 2.0, 3.0]))
        assert descriptor_distance(td, td) == pytest.approx(0.0)

    def test_known_distance(self):
        a = TextureDescriptor(vector=np.array([0.0, 0.0], dtype=np.float32))
        b = TextureDescriptor(vector=np.array([3.0, 4.0], dtype=np.float32))
        assert descriptor_distance(a, b) == pytest.approx(5.0, abs=1e-5)

    def test_nonnegative(self):
        a = TextureDescriptor(vector=np.random.rand(10).astype(np.float32))
        b = TextureDescriptor(vector=np.random.rand(10).astype(np.float32))
        assert descriptor_distance(a, b) >= 0.0

    def test_mismatched_length_raises(self):
        a = TextureDescriptor(vector=np.ones(3))
        b = TextureDescriptor(vector=np.ones(5))
        with pytest.raises(ValueError):
            descriptor_distance(a, b)


# ─── TestBatchComputeDescriptors ──────────────────────────────────────────────

class TestBatchComputeDescriptors:
    def test_returns_list(self):
        images = [_gray() for _ in range(3)]
        result = batch_compute_descriptors(images, method="lbp", lbp_bins=32)
        assert isinstance(result, list)

    def test_correct_length(self):
        images = [_gray() for _ in range(5)]
        result = batch_compute_descriptors(images, method="stats")
        assert len(result) == 5

    def test_empty_list(self):
        result = batch_compute_descriptors([])
        assert result == []

    def test_image_ids_sequential(self):
        images = [_gray() for _ in range(4)]
        result = batch_compute_descriptors(images, method="glcm", glcm_levels=8)
        for i, td in enumerate(result):
            assert td.image_id == i

    def test_each_texture_descriptor(self):
        images = [_gray(), _color()]
        result = batch_compute_descriptors(images, method="stats")
        assert all(isinstance(td, TextureDescriptor) for td in result)
