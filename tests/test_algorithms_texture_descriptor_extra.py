"""Extra tests for puzzle_reconstruction.algorithms.texture_descriptor."""
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

def _gray(h=32, w=32, fill=None, seed=None):
    if seed is not None:
        return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)
    if fill is not None:
        return np.full((h, w), fill, dtype=np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _td(values, method="combined", image_id=0):
    return TextureDescriptor(
        vector=np.array(values, dtype=np.float32),
        method=method, image_id=image_id,
    )


# ─── TextureDescriptor extras ─────────────────────────────────────────────────

class TestTextureDescriptorExtra:
    def test_repr_is_string(self):
        td = _td([0.1, 0.2, 0.3])
        assert isinstance(repr(td), str)

    def test_large_vector(self):
        v = np.zeros(512, dtype=np.float32)
        td = TextureDescriptor(vector=v)
        assert len(td) == 512

    def test_method_lbp_stored(self):
        td = _td([0.1], method="lbp")
        assert td.method == "lbp"

    def test_method_glcm_stored(self):
        td = _td([0.1, 0.2, 0.3, 0.4], method="glcm")
        assert td.method == "glcm"

    def test_method_stats_stored(self):
        td = _td([10.0, 5.0], method="stats")
        assert td.method == "stats"

    def test_image_id_zero(self):
        td = _td([0.1], image_id=0)
        assert td.image_id == 0

    def test_image_id_large(self):
        td = _td([0.1], image_id=999)
        assert td.image_id == 999

    def test_params_stored(self):
        td = TextureDescriptor(vector=np.zeros(4), params={"lbp_bins": 16})
        assert td.params["lbp_bins"] == 16

    def test_vector_dtype_float32(self):
        td = TextureDescriptor(vector=[1.0, 2.0, 3.0])
        assert td.vector.dtype == np.float32

    def test_len_zero_vector(self):
        td = TextureDescriptor(vector=np.zeros(0, dtype=np.float32))
        assert len(td) == 0


# ─── compute_lbp extras ───────────────────────────────────────────────────────

class TestComputeLbpExtra:
    def test_n_bins_16(self):
        hist = compute_lbp(_gray(seed=0), n_bins=16)
        assert hist.shape == (16,)

    def test_n_bins_8(self):
        hist = compute_lbp(_gray(seed=1), n_bins=8)
        assert hist.shape == (8,)

    def test_n_bins_64(self):
        hist = compute_lbp(_gray(seed=2), n_bins=64)
        assert hist.shape == (64,)

    def test_sums_to_1(self):
        hist = compute_lbp(_gray(seed=3), n_bins=32)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_non_square_image(self):
        img = _gray(h=16, w=48, seed=4)
        hist = compute_lbp(img, n_bins=32)
        assert hist.shape == (32,)

    def test_small_image(self):
        img = _gray(h=8, w=8, seed=5)
        hist = compute_lbp(img, n_bins=16)
        assert hist.ndim == 1

    def test_all_values_nonneg(self):
        hist = compute_lbp(_gray(seed=6), n_bins=32)
        assert np.all(hist >= 0.0)

    def test_bgr_accepted(self):
        hist = compute_lbp(_bgr(), n_bins=32)
        assert hist.shape == (32,)


# ─── compute_glcm_features extras ─────────────────────────────────────────────

class TestComputeGlcmFeaturesExtra:
    def test_distance_2(self):
        feats = compute_glcm_features(_gray(seed=0), distance=2)
        assert feats.shape == (4,)

    def test_distance_3(self):
        feats = compute_glcm_features(_gray(seed=1), distance=3)
        assert feats.shape == (4,)

    def test_levels_8(self):
        feats = compute_glcm_features(_gray(seed=2), levels=8)
        assert feats.shape == (4,)
        assert feats.dtype == np.float32

    def test_levels_64(self):
        feats = compute_glcm_features(_gray(seed=3), levels=64)
        assert feats.shape == (4,)

    def test_non_square_image(self):
        img = _gray(h=16, w=48, seed=4)
        feats = compute_glcm_features(img)
        assert feats.shape == (4,)

    def test_constant_image(self):
        img = _gray(fill=100)
        feats = compute_glcm_features(img)
        assert feats.dtype == np.float32

    def test_contrast_nonneg(self):
        feats = compute_glcm_features(_gray(seed=5))
        assert float(feats[0]) >= 0.0

    def test_correlation_in_range(self):
        feats = compute_glcm_features(_gray(seed=6))
        corr = float(feats[3])
        assert -1.0 - 1e-5 <= corr <= 1.0 + 1e-5


# ─── compute_stats_descriptor extras ─────────────────────────────────────────

class TestComputeStatsDescriptorExtra:
    def test_bgr_returns_6_values(self):
        v = compute_stats_descriptor(_bgr())
        assert v.shape == (6,)

    def test_constant_bgr_std_zero(self):
        img = np.full((8, 8, 3), 100, dtype=np.uint8)
        v = compute_stats_descriptor(img)
        # std channels should all be 0
        for i in [1, 3, 5]:
            assert float(v[i]) == pytest.approx(0.0, abs=1e-4)

    def test_non_square_gray(self):
        img = _gray(h=8, w=24, fill=50)
        v = compute_stats_descriptor(img)
        assert v.shape == (2,)

    def test_float32_output(self):
        v = compute_stats_descriptor(_gray(seed=0))
        assert v.dtype == np.float32

    def test_all_zeros_mean_zero(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        v = compute_stats_descriptor(img)
        assert float(v[0]) == pytest.approx(0.0, abs=1e-5)

    def test_mean_255_channel(self):
        img = np.full((8, 8), 255, dtype=np.uint8)
        v = compute_stats_descriptor(img)
        assert float(v[0]) == pytest.approx(255.0, abs=0.5)


# ─── compute_texture_descriptor extras ───────────────────────────────────────

class TestComputeTextureDescriptorExtra:
    def test_lbp_default_bins(self):
        td = compute_texture_descriptor(_gray(seed=0), method="lbp")
        assert td.method == "lbp"
        assert len(td) > 0  # bins depend on implementation default

    def test_stats_gray_length_2(self):
        td = compute_texture_descriptor(_gray(seed=0), method="stats")
        assert len(td) == 2

    def test_stats_bgr_length_6(self):
        td = compute_texture_descriptor(_bgr(), method="stats")
        assert len(td) == 6

    def test_glcm_length_4(self):
        td = compute_texture_descriptor(_gray(seed=1), method="glcm")
        assert len(td) == 4

    def test_combined_gray_length_with_16_bins(self):
        # lbp(16) + glcm(4) + stats(2) = 22
        td = compute_texture_descriptor(_gray(seed=2), method="combined", lbp_bins=16)
        assert len(td) == 22

    def test_image_id_stored(self):
        td = compute_texture_descriptor(_gray(), image_id=42)
        assert td.image_id == 42

    def test_returns_texture_descriptor(self):
        td = compute_texture_descriptor(_gray())
        assert isinstance(td, TextureDescriptor)

    def test_vector_float32(self):
        td = compute_texture_descriptor(_gray(seed=3))
        assert td.vector.dtype == np.float32

    def test_bgr_combined(self):
        td = compute_texture_descriptor(_bgr(), method="combined", lbp_bins=16)
        assert len(td) > 0


# ─── normalize_descriptor extras ─────────────────────────────────────────────

class TestNormalizeDescriptorExtra:
    def test_already_unit_norm(self):
        v = np.array([0.6, 0.8], dtype=np.float32)  # norm=1.0
        td = TextureDescriptor(vector=v)
        nd = normalize_descriptor(td)
        assert float(np.linalg.norm(nd.vector)) == pytest.approx(1.0, abs=1e-5)

    def test_method_preserved(self):
        td = _td([1.0, 2.0, 3.0], method="lbp")
        nd = normalize_descriptor(td)
        assert nd.method == "lbp"

    def test_image_id_preserved(self):
        td = _td([1.0, 2.0], image_id=7)
        nd = normalize_descriptor(td)
        assert nd.image_id == 7

    def test_params_preserved(self):
        td = TextureDescriptor(vector=np.array([1.0]), params={"x": 5})
        nd = normalize_descriptor(td)
        assert nd.params["x"] == 5

    def test_large_vector_unit_norm(self):
        v = np.arange(1, 11, dtype=np.float32)
        td = TextureDescriptor(vector=v)
        nd = normalize_descriptor(td)
        assert float(np.linalg.norm(nd.vector)) == pytest.approx(1.0, abs=1e-5)

    def test_single_value_positive(self):
        td = _td([5.0])
        nd = normalize_descriptor(td)
        assert float(nd.vector[0]) == pytest.approx(1.0, abs=1e-5)


# ─── descriptor_distance extras ──────────────────────────────────────────────

class TestDescriptorDistanceExtra:
    def test_known_distance_3_4_5(self):
        a = _td([0.0, 0.0])
        b = _td([3.0, 4.0])
        assert descriptor_distance(a, b) == pytest.approx(5.0)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        a = _td(rng.random(8).tolist())
        b = _td(rng.random(8).tolist())
        assert descriptor_distance(a, b) == pytest.approx(
            descriptor_distance(b, a), abs=1e-5)

    def test_nonneg_random(self):
        rng = np.random.default_rng(2)
        a = _td(rng.random(16).tolist())
        b = _td(rng.random(16).tolist())
        assert descriptor_distance(a, b) >= 0.0

    def test_identical_zero(self):
        a = _td([1.0, 2.0, 3.0])
        b = _td([1.0, 2.0, 3.0])
        assert descriptor_distance(a, b) == pytest.approx(0.0)

    def test_length_1_vectors(self):
        a = _td([5.0])
        b = _td([8.0])
        assert descriptor_distance(a, b) == pytest.approx(3.0)


# ─── batch_compute_descriptors extras ────────────────────────────────────────

class TestBatchComputeDescriptorsExtra:
    def test_stats_method(self):
        images = [_gray(seed=i) for i in range(3)]
        results = batch_compute_descriptors(images, method="stats")
        assert all(r.method == "stats" for r in results)
        assert all(len(r) == 2 for r in results)

    def test_lbp_bins_16(self):
        images = [_gray(seed=i) for i in range(3)]
        results = batch_compute_descriptors(images, method="lbp", lbp_bins=16)
        assert all(len(r) == 16 for r in results)

    def test_glcm_method(self):
        images = [_gray(seed=i) for i in range(2)]
        results = batch_compute_descriptors(images, method="glcm")
        assert all(len(r) == 4 for r in results)

    def test_image_ids_sequential(self):
        images = [_gray(seed=i) for i in range(5)]
        results = batch_compute_descriptors(images)
        for i, r in enumerate(results):
            assert r.image_id == i

    def test_single_image(self):
        results = batch_compute_descriptors([_gray(seed=0)])
        assert len(results) == 1
        assert isinstance(results[0], TextureDescriptor)

    def test_bgr_images(self):
        images = [_bgr(seed=i) for i in range(3)]
        results = batch_compute_descriptors(images, method="stats")
        assert all(len(r) == 6 for r in results)

    def test_all_same_length(self):
        images = [_gray(seed=i) for i in range(4)]
        results = batch_compute_descriptors(images, method="glcm")
        assert len(set(len(r) for r in results)) == 1
