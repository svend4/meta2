"""Tests for puzzle_reconstruction.algorithms.texture_descriptor."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.texture_descriptor import (
    TextureDescriptor,
    batch_compute_descriptors,
    compute_glcm_features,
    compute_lbp,
    compute_stats_descriptor,
    compute_texture_descriptor,
    descriptor_distance,
    normalize_descriptor,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, fill=None, rng_seed=None):
    if rng_seed is not None:
        return np.random.default_rng(rng_seed).integers(0, 255, (h, w),
                                                        dtype=np.uint8)
    if fill is not None:
        return np.full((h, w), fill, dtype=np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h=32, w=32, rng_seed=0):
    return np.random.default_rng(rng_seed).integers(0, 255, (h, w, 3),
                                                    dtype=np.uint8)


# ─── TestTextureDescriptor ────────────────────────────────────────────────────

class TestTextureDescriptor:
    def test_basic_creation(self):
        v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        td = TextureDescriptor(vector=v, method="stats", image_id=0)
        assert td.method == "stats"
        assert td.image_id == 0

    def test_vector_cast_to_float32(self):
        v = [1, 2, 3]
        td = TextureDescriptor(vector=v)
        assert td.vector.dtype == np.float32

    def test_2d_vector_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            TextureDescriptor(vector=np.zeros((3, 4)))

    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError, match="image_id"):
            TextureDescriptor(vector=np.zeros(4), image_id=-1)

    def test_len_returns_vector_length(self):
        td = TextureDescriptor(vector=np.zeros(10))
        assert len(td) == 10

    def test_default_method_combined(self):
        td = TextureDescriptor(vector=np.zeros(5))
        assert td.method == "combined"

    def test_params_default_empty(self):
        td = TextureDescriptor(vector=np.zeros(5))
        assert td.params == {}


# ─── TestComputeLbp ───────────────────────────────────────────────────────────

class TestComputeLbp:
    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            compute_lbp(_gray(), n_bins=1)

    def test_returns_float32_1d(self):
        hist = compute_lbp(_gray(rng_seed=0))
        assert hist.dtype == np.float32
        assert hist.ndim == 1

    def test_output_length_equals_n_bins(self):
        hist = compute_lbp(_gray(rng_seed=0), n_bins=32)
        assert len(hist) == 32

    def test_histogram_sums_to_1(self):
        hist = compute_lbp(_gray(rng_seed=0))
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_constant_image_valid(self):
        """Constant image: all LBP values are 0 → bin 0 gets all mass."""
        hist = compute_lbp(_gray(fill=128))
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_color_image_accepted(self):
        hist = compute_lbp(_bgr())
        assert hist.ndim == 1

    def test_default_n_bins_256(self):
        hist = compute_lbp(_gray(rng_seed=1))
        assert len(hist) == 256


# ─── TestComputeGlcmFeatures ──────────────────────────────────────────────────

class TestComputeGlcmFeatures:
    def test_levels_less_than_2_raises(self):
        with pytest.raises(ValueError, match="levels"):
            compute_glcm_features(_gray(), levels=1)

    def test_levels_greater_than_256_raises(self):
        with pytest.raises(ValueError, match="levels"):
            compute_glcm_features(_gray(), levels=257)

    def test_distance_less_than_1_raises(self):
        with pytest.raises(ValueError, match="distance"):
            compute_glcm_features(_gray(), distance=0)

    def test_returns_4_features(self):
        feats = compute_glcm_features(_gray(rng_seed=0))
        assert feats.shape == (4,)

    def test_returns_float32(self):
        feats = compute_glcm_features(_gray(rng_seed=0))
        assert feats.dtype == np.float32

    def test_energy_in_range(self):
        """Energy is sum of squared probabilities → in (0, 1]."""
        feats = compute_glcm_features(_gray(rng_seed=0))
        energy = float(feats[2])
        assert 0.0 <= energy <= 1.0 + 1e-6

    def test_homogeneity_nonneg(self):
        feats = compute_glcm_features(_gray(rng_seed=0))
        assert float(feats[1]) >= 0.0

    def test_color_image_accepted(self):
        feats = compute_glcm_features(_bgr())
        assert feats.shape == (4,)


# ─── TestComputeStatsDescriptor ───────────────────────────────────────────────

class TestComputeStatsDescriptor:
    def test_grayscale_returns_2_values(self):
        v = compute_stats_descriptor(_gray())
        assert v.shape == (2,)

    def test_bgr_returns_6_values(self):
        v = compute_stats_descriptor(_bgr())
        assert v.shape == (6,)

    def test_returns_float32(self):
        v = compute_stats_descriptor(_gray())
        assert v.dtype == np.float32

    def test_4d_raises(self):
        img = np.zeros((4, 4, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D или 3-D"):
            compute_stats_descriptor(img)

    def test_mean_value_correct(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        v = compute_stats_descriptor(img)
        assert float(v[0]) == pytest.approx(100.0, abs=0.5)

    def test_std_zero_for_constant_image(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        v = compute_stats_descriptor(img)
        assert float(v[1]) == pytest.approx(0.0, abs=1e-4)


# ─── TestComputeTextureDescriptor ─────────────────────────────────────────────

class TestComputeTextureDescriptor:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="метод"):
            compute_texture_descriptor(_gray(), method="unknown")

    def test_lbp_method(self):
        td = compute_texture_descriptor(_gray(rng_seed=0), method="lbp",
                                        lbp_bins=16)
        assert td.method == "lbp"
        assert len(td) == 16

    def test_glcm_method(self):
        td = compute_texture_descriptor(_gray(rng_seed=0), method="glcm")
        assert td.method == "glcm"
        assert len(td) == 4

    def test_stats_method_grayscale(self):
        td = compute_texture_descriptor(_gray(), method="stats")
        assert td.method == "stats"
        assert len(td) == 2

    def test_combined_method_length(self):
        td = compute_texture_descriptor(_gray(rng_seed=0), method="combined",
                                        lbp_bins=16)
        # lbp(16) + glcm(4) + stats(2) = 22
        assert len(td) == 22

    def test_image_id_set(self):
        td = compute_texture_descriptor(_gray(), image_id=5)
        assert td.image_id == 5

    def test_vector_is_float32(self):
        td = compute_texture_descriptor(_gray(rng_seed=0))
        assert td.vector.dtype == np.float32

    def test_returns_texture_descriptor_instance(self):
        td = compute_texture_descriptor(_gray())
        assert isinstance(td, TextureDescriptor)


# ─── TestNormalizeDescriptor ──────────────────────────────────────────────────

class TestNormalizeDescriptor:
    def test_unit_norm_after_normalize(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        td = TextureDescriptor(vector=v)
        nd = normalize_descriptor(td)
        assert float(np.linalg.norm(nd.vector)) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_unchanged(self):
        v = np.zeros(5, dtype=np.float32)
        td = TextureDescriptor(vector=v)
        nd = normalize_descriptor(td)
        np.testing.assert_array_equal(nd.vector, np.zeros(5, dtype=np.float32))

    def test_metadata_preserved(self):
        v = np.array([1.0, 2.0, 3.0])
        td = TextureDescriptor(vector=v, method="lbp", image_id=7,
                               params={"x": 1})
        nd = normalize_descriptor(td)
        assert nd.method == "lbp"
        assert nd.image_id == 7
        assert nd.params == {"x": 1}

    def test_original_unchanged(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        td = TextureDescriptor(vector=v.copy())
        normalize_descriptor(td)
        np.testing.assert_array_almost_equal(td.vector, v)


# ─── TestDescriptorDistance ───────────────────────────────────────────────────

class TestDescriptorDistance:
    def test_identical_descriptors_zero_distance(self):
        v = np.array([1.0, 2.0, 3.0])
        a = TextureDescriptor(vector=v)
        b = TextureDescriptor(vector=v.copy())
        assert descriptor_distance(a, b) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        a = TextureDescriptor(vector=np.zeros(3))
        b = TextureDescriptor(vector=np.zeros(4))
        with pytest.raises(ValueError):
            descriptor_distance(a, b)

    def test_known_distance(self):
        a = TextureDescriptor(vector=np.array([0.0, 0.0]))
        b = TextureDescriptor(vector=np.array([3.0, 4.0]))
        assert descriptor_distance(a, b) == pytest.approx(5.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        a = TextureDescriptor(vector=rng.random(10).astype(np.float32))
        b = TextureDescriptor(vector=rng.random(10).astype(np.float32))
        assert descriptor_distance(a, b) >= 0.0

    def test_symmetric(self):
        rng = np.random.default_rng(7)
        a = TextureDescriptor(vector=rng.random(8).astype(np.float32))
        b = TextureDescriptor(vector=rng.random(8).astype(np.float32))
        assert descriptor_distance(a, b) == pytest.approx(
            descriptor_distance(b, a), abs=1e-6
        )


# ─── TestBatchComputeDescriptors ──────────────────────────────────────────────

class TestBatchComputeDescriptors:
    def test_returns_list_of_correct_length(self):
        images = [_gray(rng_seed=i) for i in range(4)]
        results = batch_compute_descriptors(images)
        assert len(results) == 4

    def test_image_ids_set_correctly(self):
        images = [_gray(rng_seed=i) for i in range(3)]
        results = batch_compute_descriptors(images)
        for i, td in enumerate(results):
            assert td.image_id == i

    def test_empty_list(self):
        assert batch_compute_descriptors([]) == []

    def test_all_same_method(self):
        images = [_gray(rng_seed=i) for i in range(2)]
        results = batch_compute_descriptors(images, method="glcm")
        for td in results:
            assert td.method == "glcm"

    def test_all_same_length(self):
        images = [_gray(rng_seed=i) for i in range(3)]
        results = batch_compute_descriptors(images, method="lbp", lbp_bins=16)
        lengths = [len(td) for td in results]
        assert len(set(lengths)) == 1
