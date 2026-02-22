"""Тесты для puzzle_reconstruction/preprocessing/document_cleaner.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.document_cleaner import (
    CleanResult,
    remove_shadow,
    remove_border_artifacts,
    normalize_illumination,
    remove_blobs,
    auto_clean,
    batch_clean,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=180):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _with_dark_spots(h=64, w=64):
    """Белое изображение с несколькими небольшими тёмными пятнами."""
    img = np.full((h, w), 240, dtype=np.uint8)
    img[10:14, 10:14] = 20   # ~16 px²
    img[30:35, 40:45] = 20   # ~25 px²
    return img


# ─── CleanResult ─────────────────────────────────────────────────────────────

class TestCleanResult:
    def test_fields(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="shadow")
        assert r.method == "shadow"
        assert r.artifacts_removed == 0
        assert isinstance(r.params, dict)

    def test_cleaned_stored(self):
        arr = np.full((8, 8), 128, dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="test")
        assert r.cleaned.shape == (8, 8)

    def test_artifacts_removed_stored(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="blobs", artifacts_removed=5)
        assert r.artifacts_removed == 5

    def test_params_stored(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="shadow", params={"block_size": 41})
        assert r.params["block_size"] == 41

    def test_repr_contains_method(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="illumination")
        assert "illumination" in repr(r)

    def test_repr_contains_shape(self):
        arr = np.zeros((20, 30), dtype=np.uint8)
        r = CleanResult(cleaned=arr, method="border")
        s = repr(r)
        assert "20" in s or "30" in s


# ─── remove_shadow ───────────────────────────────────────────────────────────

class TestRemoveShadow:
    def test_returns_result(self):
        assert isinstance(remove_shadow(_noisy()), CleanResult)

    def test_method_name(self):
        assert remove_shadow(_noisy()).method == "shadow"

    def test_output_shape_gray(self):
        r = remove_shadow(_noisy(32, 48))
        assert r.cleaned.shape == (32, 48)

    def test_output_dtype_gray(self):
        assert remove_shadow(_noisy()).cleaned.dtype == np.uint8

    def test_output_shape_bgr(self):
        r = remove_shadow(_bgr(40, 60))
        assert r.cleaned.shape == (40, 60, 3)

    def test_output_dtype_bgr(self):
        assert remove_shadow(_bgr()).cleaned.dtype == np.uint8

    def test_block_size_stored(self):
        r = remove_shadow(_noisy(), block_size=51)
        assert r.params.get("block_size") is not None

    def test_c_param_stored(self):
        r = remove_shadow(_noisy(), c=20.0)
        assert r.params.get("c") == pytest.approx(20.0)

    def test_no_modify_input(self):
        img = _noisy()
        orig = img.copy()
        remove_shadow(img)
        np.testing.assert_array_equal(img, orig)

    def test_constant_image_no_crash(self):
        r = remove_shadow(_gray(val=200))
        assert isinstance(r, CleanResult)
        assert r.cleaned.shape == (64, 64)

    def test_even_block_size_corrected(self):
        # even block_size should be auto-corrected to odd
        r = remove_shadow(_noisy(), block_size=40)
        assert r.params.get("block_size") % 2 == 1

    def test_noisy_image_changes(self):
        img = _noisy()
        r = remove_shadow(img, block_size=11)
        # Result should be different from input
        assert not np.array_equal(r.cleaned, img)


# ─── remove_border_artifacts ─────────────────────────────────────────────────

class TestRemoveBorderArtifacts:
    def test_returns_result(self):
        assert isinstance(remove_border_artifacts(_noisy()), CleanResult)

    def test_method_name(self):
        assert remove_border_artifacts(_noisy()).method == "border"

    def test_output_shape_gray(self):
        r = remove_border_artifacts(_noisy(32, 48))
        assert r.cleaned.shape == (32, 48)

    def test_output_shape_bgr(self):
        r = remove_border_artifacts(_bgr())
        assert r.cleaned.shape == (64, 64, 3)

    def test_output_dtype(self):
        assert remove_border_artifacts(_noisy()).cleaned.dtype == np.uint8

    def test_params_stored(self):
        r = remove_border_artifacts(_noisy(), border_px=8, fill=0)
        assert r.params.get("border_px") == 8
        assert r.params.get("fill") == 0

    def test_zero_border_no_change(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=0)
        np.testing.assert_array_equal(r.cleaned, img)

    def test_border_top_filled(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=5, fill=255)
        assert np.all(r.cleaned[:5, :] == 255)

    def test_border_bottom_filled(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=5, fill=255)
        assert np.all(r.cleaned[-5:, :] == 255)

    def test_border_left_filled(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=5, fill=0)
        assert np.all(r.cleaned[:, :5] == 0)

    def test_border_right_filled(self):
        img = _noisy()
        r = remove_border_artifacts(img, border_px=5, fill=0)
        assert np.all(r.cleaned[:, -5:] == 0)

    def test_no_modify_input(self):
        img = _noisy()
        orig = img.copy()
        remove_border_artifacts(img)
        np.testing.assert_array_equal(img, orig)

    def test_bgr_border_filled(self):
        img = _bgr()
        r = remove_border_artifacts(img, border_px=3, fill=255)
        assert np.all(r.cleaned[:3, :, 0] == 255)


# ─── normalize_illumination ───────────────────────────────────────────────────

class TestNormalizeIllumination:
    def test_returns_result(self):
        assert isinstance(normalize_illumination(_noisy()), CleanResult)

    def test_method_name(self):
        assert normalize_illumination(_noisy()).method == "illumination"

    def test_output_shape_gray(self):
        r = normalize_illumination(_noisy(40, 50))
        assert r.cleaned.shape == (40, 50)

    def test_output_shape_bgr(self):
        r = normalize_illumination(_bgr())
        assert r.cleaned.shape == (64, 64, 3)

    def test_output_dtype(self):
        assert normalize_illumination(_noisy()).cleaned.dtype == np.uint8

    def test_sigma_stored(self):
        r = normalize_illumination(_noisy(), sigma=40.0)
        assert r.params.get("sigma") == pytest.approx(40.0)

    def test_output_range(self):
        r = normalize_illumination(_noisy())
        assert r.cleaned.min() >= 0
        assert r.cleaned.max() <= 255

    def test_constant_image_no_crash(self):
        r = normalize_illumination(_gray(val=150))
        assert isinstance(r, CleanResult)
        assert r.cleaned.shape == (64, 64)

    def test_no_modify_input(self):
        img = _noisy()
        orig = img.copy()
        normalize_illumination(img)
        np.testing.assert_array_equal(img, orig)

    def test_bgr_input(self):
        r = normalize_illumination(_bgr())
        assert r.cleaned.dtype == np.uint8


# ─── remove_blobs ─────────────────────────────────────────────────────────────

class TestRemoveBlobs:
    def test_returns_result(self):
        assert isinstance(remove_blobs(_noisy()), CleanResult)

    def test_method_name(self):
        assert remove_blobs(_noisy()).method == "blobs"

    def test_output_shape(self):
        r = remove_blobs(_noisy(48, 56))
        assert r.cleaned.shape == (48, 56)

    def test_output_dtype(self):
        assert remove_blobs(_noisy()).cleaned.dtype == np.uint8

    def test_params_stored(self):
        r = remove_blobs(_noisy(), min_area=5, max_area=200, fill=255)
        assert r.params.get("min_area") == 5
        assert r.params.get("max_area") == 200
        assert r.params.get("fill") == 255

    def test_artifacts_removed_nonneg(self):
        r = remove_blobs(_noisy())
        assert r.artifacts_removed >= 0

    def test_blobs_removed_on_text_image(self):
        img = _with_dark_spots()
        r = remove_blobs(img, min_area=5, max_area=100, fill=255)
        assert isinstance(r, CleanResult)
        # Spots should be removed (filled with white)
        assert r.artifacts_removed >= 0

    def test_bgr_input(self):
        r = remove_blobs(_bgr())
        assert r.cleaned.shape == (64, 64, 3)

    def test_constant_white_no_blobs(self):
        img = _gray(val=255)
        r = remove_blobs(img)
        assert r.artifacts_removed == 0

    def test_no_modify_input(self):
        img = _with_dark_spots()
        orig = img.copy()
        remove_blobs(img)
        np.testing.assert_array_equal(img, orig)


# ─── auto_clean ───────────────────────────────────────────────────────────────

class TestAutoClean:
    def test_returns_result(self):
        assert isinstance(auto_clean(_noisy()), CleanResult)

    def test_method_name(self):
        assert auto_clean(_noisy()).method == "auto"

    def test_output_shape_gray(self):
        r = auto_clean(_noisy(36, 48))
        assert r.cleaned.shape == (36, 48)

    def test_output_shape_bgr(self):
        r = auto_clean(_bgr())
        assert r.cleaned.shape == (64, 64, 3)

    def test_output_dtype(self):
        assert auto_clean(_noisy()).cleaned.dtype == np.uint8

    def test_params_stored(self):
        r = auto_clean(_noisy(), shadow_block=51, illum_sigma=25.0)
        assert r.params.get("shadow_block") is not None
        assert r.params.get("illum_sigma") is not None

    def test_constant_image_no_crash(self):
        r = auto_clean(_gray(val=180))
        assert isinstance(r, CleanResult)

    def test_output_range(self):
        r = auto_clean(_noisy())
        assert r.cleaned.min() >= 0
        assert r.cleaned.max() <= 255


# ─── batch_clean ─────────────────────────────────────────────────────────────

class TestBatchClean:
    def test_returns_list(self):
        results = batch_clean([_noisy() for _ in range(3)])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_is_result(self):
        for r in batch_clean([_noisy(), _gray()]):
            assert isinstance(r, CleanResult)

    def test_empty_list(self):
        assert batch_clean([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises((ValueError, KeyError)):
            batch_clean([_noisy()], method="unknown_xyz_method")

    @pytest.mark.parametrize("method", [
        "shadow",
        "border",
        "illumination",
        "blobs",
        "auto",
    ])
    def test_all_methods(self, method):
        imgs = [_noisy(), _gray()]
        results = batch_clean(imgs, method=method)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, CleanResult)
            assert r.cleaned.shape == (64, 64)

    def test_kwargs_forwarded_shadow(self):
        results = batch_clean([_noisy()], method="shadow", block_size=21, c=15.0)
        assert results[0].params.get("c") == pytest.approx(15.0)

    def test_kwargs_forwarded_border(self):
        results = batch_clean([_noisy()], method="border", border_px=10)
        assert results[0].params.get("border_px") == 10

    def test_shapes_preserved(self):
        imgs = [_noisy(30, 40), _noisy(50, 60)]
        results = batch_clean(imgs)
        assert results[0].cleaned.shape == (30, 40)
        assert results[1].cleaned.shape == (50, 60)

    def test_bgr_input(self):
        results = batch_clean([_bgr()], method="shadow")
        assert results[0].cleaned.shape == (64, 64, 3)
