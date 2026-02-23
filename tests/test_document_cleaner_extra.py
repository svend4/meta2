"""Extra tests for puzzle_reconstruction/preprocessing/document_cleaner.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.document_cleaner import (
    CleanResult,
    auto_clean,
    batch_clean,
    normalize_illumination,
    remove_blobs,
    remove_border_artifacts,
    remove_shadow,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=180):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=7):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _with_blobs(h=64, w=64):
    img = np.full((h, w), 240, dtype=np.uint8)
    img[10:14, 10:14] = 20
    img[40:46, 40:46] = 20
    return img


# ─── TestCleanResultExtra ─────────────────────────────────────────────────────

class TestCleanResultExtra:
    def test_multiple_params(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="shadow",
                        params={"block_size": 21, "c": 15.0, "mode": "rgb"})
        assert r.params["block_size"] == 21
        assert r.params["c"] == pytest.approx(15.0)
        assert r.params["mode"] == "rgb"

    def test_artifacts_removed_large(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="blobs", artifacts_removed=42)
        assert r.artifacts_removed == 42

    def test_method_blobs(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="blobs")
        assert r.method == "blobs"

    def test_method_auto(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="auto")
        assert r.method == "auto"

    def test_method_illumination(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="illumination")
        assert r.method == "illumination"

    def test_bgr_cleaned_shape(self):
        arr = np.zeros((30, 40, 3), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="border")
        assert r.cleaned.shape == (30, 40, 3)

    def test_repr_contains_artifacts(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="blobs", artifacts_removed=3)
        s = repr(r)
        assert "blobs" in s

    def test_params_empty_by_default(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="shadow")
        assert r.params == {}


# ─── TestRemoveShadowExtra ────────────────────────────────────────────────────

class TestRemoveShadowExtra:
    def test_various_block_sizes(self):
        img = _noisy()
        for bs in (11, 21, 31, 51):
            r = remove_shadow(img, block_size=bs)
            assert r.cleaned.dtype == np.uint8
            assert r.cleaned.shape == img.shape

    def test_c_zero(self):
        r = remove_shadow(_noisy(), c=0.0)
        assert r.cleaned.dtype == np.uint8

    def test_c_large(self):
        r = remove_shadow(_noisy(), c=50.0)
        assert r.cleaned.dtype == np.uint8

    def test_output_range(self):
        r = remove_shadow(_noisy())
        assert int(r.cleaned.min()) >= 0
        assert int(r.cleaned.max()) <= 255

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        r = remove_shadow(img)
        assert r.cleaned.shape == (16, 16)

    def test_non_square(self):
        img = _noisy(h=32, w=48)
        r = remove_shadow(img)
        assert r.cleaned.shape == (32, 48)

    def test_bgr_non_square(self):
        img = _bgr(h=32, w=48)
        r = remove_shadow(img)
        assert r.cleaned.shape == (32, 48, 3)

    def test_block_size_one_less_than_even_corrected(self):
        r = remove_shadow(_noisy(), block_size=20)
        assert r.params.get("block_size") % 2 == 1

    def test_all_black_no_crash(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        r = remove_shadow(img)
        assert isinstance(r, CleanResult)


# ─── TestRemoveBorderArtifactsExtra ───────────────────────────────────────────

class TestRemoveBorderArtifactsExtra:
    def test_large_border(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=15, fill=0)
        assert np.all(r.cleaned[:15, :] == 0)
        assert np.all(r.cleaned[-15:, :] == 0)
        assert np.all(r.cleaned[:, :15] == 0)
        assert np.all(r.cleaned[:, -15:] == 0)

    def test_fill_128(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=5, fill=128)
        assert np.all(r.cleaned[:5, :] == 128)

    def test_interior_not_filled(self):
        img = _gray(val=100)
        r = remove_border_artifacts(img, border_px=5, fill=0)
        interior = r.cleaned[5:-5, 5:-5]
        assert np.all(interior == 100)

    def test_non_square(self):
        img = _noisy(h=40, w=80)
        r = remove_border_artifacts(img, border_px=5)
        assert r.cleaned.shape == (40, 80)

    def test_border_larger_than_half_no_crash(self):
        img = _noisy(h=20, w=20)
        r = remove_border_artifacts(img, border_px=8, fill=0)
        assert r.cleaned.dtype == np.uint8

    def test_method_border(self):
        r = remove_border_artifacts(_noisy())
        assert r.method == "border"

    def test_bgr_non_square(self):
        img = _bgr(h=40, w=80)
        r = remove_border_artifacts(img, border_px=5, fill=0)
        assert r.cleaned.shape == (40, 80, 3)

    def test_params_fill_stored(self):
        r = remove_border_artifacts(_noisy(), border_px=6, fill=42)
        assert r.params.get("fill") == 42


# ─── TestNormalizeIlluminationExtra ───────────────────────────────────────────

class TestNormalizeIlluminationExtra:
    def test_various_sigmas(self):
        img = _noisy()
        for s in (10.0, 20.0, 40.0, 80.0):
            r = normalize_illumination(img, sigma=s)
            assert r.cleaned.dtype == np.uint8
            assert r.cleaned.shape == img.shape

    def test_non_square(self):
        img = _noisy(h=40, w=80)
        r = normalize_illumination(img)
        assert r.cleaned.shape == (40, 80)

    def test_all_black_no_crash(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        r = normalize_illumination(img)
        assert isinstance(r, CleanResult)

    def test_all_white_no_crash(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        r = normalize_illumination(img)
        assert isinstance(r, CleanResult)

    def test_bgr_non_square(self):
        img = _bgr(h=40, w=60)
        r = normalize_illumination(img)
        assert r.cleaned.shape == (40, 60, 3)

    def test_input_unchanged(self):
        img = _noisy()
        orig = img.copy()
        normalize_illumination(img)
        np.testing.assert_array_equal(img, orig)

    def test_sigma_stored(self):
        r = normalize_illumination(_noisy(), sigma=50.0)
        assert r.params.get("sigma") == pytest.approx(50.0)

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        r = normalize_illumination(img)
        assert r.cleaned.shape == (16, 16)


# ─── TestRemoveBlobsExtra ─────────────────────────────────────────────────────

class TestRemoveBlobsExtra:
    def test_large_min_area_removes_nothing(self):
        img = _with_blobs()
        r = remove_blobs(img, min_area=10000, max_area=100000, fill=255)
        assert r.artifacts_removed == 0

    def test_small_max_area_removes_small_blobs(self):
        img = _with_blobs()
        r = remove_blobs(img, min_area=1, max_area=50, fill=255)
        assert r.artifacts_removed >= 0

    def test_non_square(self):
        img = _noisy(h=40, w=80)
        r = remove_blobs(img)
        assert r.cleaned.shape == (40, 80)

    def test_fill_white(self):
        img = _with_blobs()
        r = remove_blobs(img, min_area=1, max_area=100, fill=255)
        assert r.cleaned.dtype == np.uint8

    def test_fill_black(self):
        img = _with_blobs()
        r = remove_blobs(img, min_area=1, max_area=100, fill=0)
        assert r.cleaned.dtype == np.uint8

    def test_no_modify_input_blobs(self):
        img = _with_blobs()
        orig = img.copy()
        remove_blobs(img)
        np.testing.assert_array_equal(img, orig)

    def test_all_white_no_blobs(self):
        img = _gray(val=255)
        r = remove_blobs(img)
        assert r.artifacts_removed == 0

    def test_output_range(self):
        r = remove_blobs(_noisy())
        assert int(r.cleaned.min()) >= 0
        assert int(r.cleaned.max()) <= 255


# ─── TestAutoCleanExtra ───────────────────────────────────────────────────────

class TestAutoCleanExtra:
    def test_non_square(self):
        img = _noisy(h=40, w=80)
        r = auto_clean(img)
        assert r.cleaned.shape == (40, 80)

    def test_all_black_no_crash(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        r = auto_clean(img)
        assert isinstance(r, CleanResult)

    def test_all_white_no_crash(self):
        img = _gray(val=255)
        r = auto_clean(img)
        assert isinstance(r, CleanResult)

    def test_output_range(self):
        r = auto_clean(_noisy())
        assert int(r.cleaned.min()) >= 0
        assert int(r.cleaned.max()) <= 255

    def test_method_auto(self):
        r = auto_clean(_noisy())
        assert r.method == "auto"

    def test_bgr_non_square(self):
        img = _bgr(h=40, w=60)
        r = auto_clean(img)
        assert r.cleaned.shape == (40, 60, 3)

    def test_custom_shadow_block(self):
        r = auto_clean(_noisy(), shadow_block=31)
        assert r.params.get("shadow_block") is not None

    def test_custom_illum_sigma(self):
        r = auto_clean(_noisy(), illum_sigma=30.0)
        assert r.params.get("illum_sigma") is not None

    def test_dtype_preserved(self):
        r = auto_clean(_noisy())
        assert r.cleaned.dtype == np.uint8


# ─── TestBatchCleanExtra ─────────────────────────────────────────────────────

class TestBatchCleanExtra:
    def test_ten_images(self):
        imgs = [_noisy(seed=i) for i in range(10)]
        results = batch_clean(imgs)
        assert len(results) == 10

    def test_mixed_sizes(self):
        imgs = [_noisy(h=30, w=30), _noisy(h=50, w=60), _noisy(h=64, w=32)]
        results = batch_clean(imgs)
        assert results[0].cleaned.shape == (30, 30)
        assert results[1].cleaned.shape == (50, 60)
        assert results[2].cleaned.shape == (64, 32)

    def test_shadow_method_all_uint8(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        results = batch_clean(imgs, method="shadow")
        assert all(r.cleaned.dtype == np.uint8 for r in results)

    def test_illumination_method(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        results = batch_clean(imgs, method="illumination")
        assert all(r.method == "illumination" for r in results)

    def test_blobs_method(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        results = batch_clean(imgs, method="blobs")
        assert all(r.method == "blobs" for r in results)

    def test_border_method(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        results = batch_clean(imgs, method="border")
        assert all(r.method == "border" for r in results)

    def test_bgr_batch(self):
        imgs = [_bgr() for _ in range(3)]
        results = batch_clean(imgs, method="shadow")
        assert all(r.cleaned.shape == (64, 64, 3) for r in results)

    def test_kwargs_forwarded_to_all(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        results = batch_clean(imgs, method="border", border_px=6, fill=0)
        for r in results:
            assert r.params.get("border_px") == 6
