"""Тесты для puzzle_reconstruction/preprocessing/binarizer.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.binarizer import (
    BinarizeResult,
    binarize_otsu,
    binarize_adaptive,
    binarize_sauvola,
    binarize_niblack,
    binarize_bernsen,
    auto_binarize,
    batch_binarize,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _text_like(h=64, w=64):
    img = np.full((h, w), 200, dtype=np.uint8)
    img[10:20, 5:60] = 30
    img[30:40, 5:60] = 30
    return img


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h // 2, :] = [200, 150, 100]
    return img


# ─── BinarizeResult ──────────────────────────────────────────────────────────

class TestBinarizeResult:
    def test_fields(self):
        b = np.zeros((10, 10), dtype=np.uint8)
        r = BinarizeResult(binary=b, method="otsu", threshold=128.0)
        assert r.method == "otsu"
        assert r.threshold == pytest.approx(128.0)
        assert r.inverted is False
        assert isinstance(r.params, dict)

    def test_foreground_ratio_all_black(self):
        b = np.zeros((10, 10), dtype=np.uint8)
        r = BinarizeResult(binary=b, method="otsu", threshold=0.0)
        assert r.foreground_ratio == pytest.approx(0.0)

    def test_foreground_ratio_all_white(self):
        b = np.full((10, 10), 255, dtype=np.uint8)
        r = BinarizeResult(binary=b, method="otsu", threshold=0.0)
        assert r.foreground_ratio == pytest.approx(1.0)

    def test_foreground_ratio_half(self):
        b = np.zeros((4, 4), dtype=np.uint8)
        b[:2, :] = 255
        r = BinarizeResult(binary=b, method="x", threshold=0.0)
        assert 0.4 < r.foreground_ratio < 0.6

    def test_repr_contains_method(self):
        b = np.zeros((8, 8), dtype=np.uint8)
        r = BinarizeResult(binary=b, method="sauvola", threshold=100.0)
        s = repr(r)
        assert "sauvola" in s

    def test_inverted_flag(self):
        b = np.zeros((4, 4), dtype=np.uint8)
        r = BinarizeResult(binary=b, method="m", threshold=0.0, inverted=True)
        assert r.inverted is True

    def test_params_stored(self):
        b = np.zeros((4, 4), dtype=np.uint8)
        r = BinarizeResult(binary=b, method="m", threshold=0.0, params={"k": 3})
        assert r.params["k"] == 3

    def test_foreground_ratio_range(self):
        b = _noisy()
        b = np.where(b > 128, 255, 0).astype(np.uint8)
        r = BinarizeResult(binary=b, method="m", threshold=128.0)
        assert 0.0 <= r.foreground_ratio <= 1.0


# ─── binarize_otsu ───────────────────────────────────────────────────────────

class TestBinarizeOtsu:
    def test_returns_result(self):
        assert isinstance(binarize_otsu(_text_like()), BinarizeResult)

    def test_method_name(self):
        assert binarize_otsu(_noisy()).method == "otsu"

    def test_output_shape_gray(self):
        r = binarize_otsu(_noisy(32, 48))
        assert r.binary.shape == (32, 48)

    def test_output_dtype(self):
        assert binarize_otsu(_noisy()).binary.dtype == np.uint8

    def test_binary_values(self):
        unique = np.unique(binarize_otsu(_noisy()).binary)
        assert set(unique).issubset({0, 255})

    def test_threshold_in_range(self):
        r = binarize_otsu(_text_like())
        assert 0.0 <= r.threshold <= 255.0

    def test_bgr_input(self):
        r = binarize_otsu(_bgr())
        assert r.binary.ndim == 2
        assert r.binary.shape == (64, 64)

    def test_invert_flips_result(self):
        img = _text_like()
        r_n = binarize_otsu(img, invert=False)
        r_i = binarize_otsu(img, invert=True)
        assert r_i.inverted is True
        combined = r_n.binary.astype(np.int32) + r_i.binary.astype(np.int32)
        assert np.all(combined == 255)

    def test_no_modify_input(self):
        img = _text_like()
        orig = img.copy()
        binarize_otsu(img)
        np.testing.assert_array_equal(img, orig)

    def test_text_has_both_values(self):
        r = binarize_otsu(_text_like())
        assert 0 in r.binary and 255 in r.binary


# ─── binarize_adaptive ───────────────────────────────────────────────────────

class TestBinarizeAdaptive:
    def test_returns_result(self):
        assert isinstance(binarize_adaptive(_text_like()), BinarizeResult)

    def test_method_contains_adaptive(self):
        assert "adaptive" in binarize_adaptive(_text_like()).method

    def test_block_size_stored(self):
        r = binarize_adaptive(_text_like(), block_size=15)
        assert r.params.get("block_size") == 15

    def test_c_param_stored(self):
        r = binarize_adaptive(_text_like(), c=5.0)
        assert r.params.get("c") == pytest.approx(5.0)

    def test_adaptive_mean_method(self):
        r = binarize_adaptive(_text_like(), adaptive_method="mean")
        assert "mean" in r.method or r.params.get("adaptive_method") == "mean"

    def test_adaptive_gaussian_method(self):
        r = binarize_adaptive(_text_like(), adaptive_method="gaussian")
        assert "gaussian" in r.method or r.params.get("adaptive_method") == "gaussian"

    def test_output_shape(self):
        r = binarize_adaptive(_noisy(40, 50))
        assert r.binary.shape == (40, 50)

    def test_output_dtype(self):
        assert binarize_adaptive(_noisy()).binary.dtype == np.uint8

    def test_binary_values(self):
        unique = np.unique(binarize_adaptive(_noisy()).binary)
        assert set(unique).issubset({0, 255})

    def test_invert(self):
        r = binarize_adaptive(_text_like(), invert=True)
        assert r.inverted is True

    def test_bgr_input(self):
        r = binarize_adaptive(_bgr())
        assert r.binary.shape == (64, 64)


# ─── binarize_sauvola ────────────────────────────────────────────────────────

class TestBinarizeSauvola:
    def test_returns_result(self):
        assert isinstance(binarize_sauvola(_text_like()), BinarizeResult)

    def test_method_name(self):
        assert binarize_sauvola(_text_like()).method == "sauvola"

    def test_params_stored(self):
        r = binarize_sauvola(_text_like(), window_size=21, k=0.3, r=64.0)
        assert r.params.get("window_size") == 21
        assert r.params.get("k") == pytest.approx(0.3)
        assert r.params.get("r") == pytest.approx(64.0)

    def test_output_shape(self):
        r = binarize_sauvola(_noisy(50, 60))
        assert r.binary.shape == (50, 60)

    def test_output_dtype(self):
        assert binarize_sauvola(_noisy()).binary.dtype == np.uint8

    def test_binary_values(self):
        unique = np.unique(binarize_sauvola(_noisy()).binary)
        assert set(unique).issubset({0, 255})

    def test_text_image_splits(self):
        r = binarize_sauvola(_text_like())
        assert 0 in r.binary and 255 in r.binary

    def test_constant_image_no_crash(self):
        r = binarize_sauvola(_gray(val=128))
        assert isinstance(r, BinarizeResult)
        assert r.binary.shape == (64, 64)

    def test_invert(self):
        assert binarize_sauvola(_text_like(), invert=True).inverted is True

    def test_bgr_input(self):
        assert binarize_sauvola(_bgr()).binary.shape == (64, 64)

    def test_foreground_ratio_in_range(self):
        r = binarize_sauvola(_text_like())
        assert 0.0 <= r.foreground_ratio <= 1.0


# ─── binarize_niblack ────────────────────────────────────────────────────────

class TestBinarizeNiblack:
    def test_returns_result(self):
        assert isinstance(binarize_niblack(_text_like()), BinarizeResult)

    def test_method_name(self):
        assert binarize_niblack(_text_like()).method == "niblack"

    def test_params_stored(self):
        r = binarize_niblack(_text_like(), window_size=19, k=-0.3)
        assert r.params.get("window_size") == 19
        assert r.params.get("k") == pytest.approx(-0.3)

    def test_output_shape(self):
        r = binarize_niblack(_noisy(45, 55))
        assert r.binary.shape == (45, 55)

    def test_output_dtype(self):
        assert binarize_niblack(_noisy()).binary.dtype == np.uint8

    def test_binary_values(self):
        unique = np.unique(binarize_niblack(_noisy()).binary)
        assert set(unique).issubset({0, 255})

    def test_invert(self):
        assert binarize_niblack(_text_like(), invert=True).inverted is True

    def test_constant_image_no_crash(self):
        r = binarize_niblack(_gray(val=100))
        assert isinstance(r, BinarizeResult)

    def test_bgr_input(self):
        assert binarize_niblack(_bgr()).binary.shape == (64, 64)


# ─── binarize_bernsen ────────────────────────────────────────────────────────

class TestBinarizeBernsen:
    def test_returns_result(self):
        assert isinstance(binarize_bernsen(_text_like()), BinarizeResult)

    def test_method_name(self):
        assert binarize_bernsen(_text_like()).method == "bernsen"

    def test_params_stored(self):
        r = binarize_bernsen(_text_like(), window_size=21, contrast_thresh=20.0)
        assert r.params.get("window_size") == 21
        assert r.params.get("contrast_thresh") == pytest.approx(20.0)

    def test_output_shape(self):
        r = binarize_bernsen(_noisy(36, 48))
        assert r.binary.shape == (36, 48)

    def test_output_dtype(self):
        assert binarize_bernsen(_noisy()).binary.dtype == np.uint8

    def test_binary_values(self):
        unique = np.unique(binarize_bernsen(_noisy()).binary)
        assert set(unique).issubset({0, 255})

    def test_low_contrast_uniform_image(self):
        # Uniform image: zero local contrast → classified as background
        r = binarize_bernsen(_gray(val=128), contrast_thresh=15.0)
        assert isinstance(r, BinarizeResult)
        assert set(np.unique(r.binary)).issubset({0, 255})

    def test_invert(self):
        assert binarize_bernsen(_text_like(), invert=True).inverted is True

    def test_bgr_input(self):
        assert binarize_bernsen(_bgr()).binary.shape == (64, 64)


# ─── auto_binarize ───────────────────────────────────────────────────────────

class TestAutoBinarize:
    def test_returns_result(self):
        assert isinstance(auto_binarize(_text_like()), BinarizeResult)

    def test_high_entropy_uses_otsu(self):
        # Uniformly random image → high entropy → Otsu
        r = auto_binarize(_noisy())
        assert "otsu" in r.method or r.method == "otsu"

    def test_low_entropy_uses_sauvola(self):
        # Nearly uniform image → low entropy → Sauvola
        img = _gray(val=200)
        img[5, 5] = 201  # tiny variation to prevent zero-division
        r = auto_binarize(img)
        assert "sauvola" in r.method or r.method == "sauvola"

    def test_invert_forwarded(self):
        assert auto_binarize(_text_like(), invert=True).inverted is True

    def test_output_shape(self):
        r = auto_binarize(_text_like(48, 64))
        assert r.binary.shape == (48, 64)

    def test_binary_values(self):
        unique = np.unique(auto_binarize(_text_like()).binary)
        assert set(unique).issubset({0, 255})

    def test_bgr_input(self):
        assert auto_binarize(_bgr()).binary.shape == (64, 64)


# ─── batch_binarize ──────────────────────────────────────────────────────────

class TestBatchBinarize:
    def test_returns_list(self):
        results = batch_binarize([_noisy() for _ in range(3)])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_is_result(self):
        for r in batch_binarize([_text_like() for _ in range(2)]):
            assert isinstance(r, BinarizeResult)

    def test_empty_list(self):
        assert batch_binarize([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises((ValueError, KeyError)):
            batch_binarize([_noisy()], method="unknown_xyz")

    @pytest.mark.parametrize("method", [
        "otsu",
        "adaptive_mean",
        "adaptive_gaussian",
        "adaptive",
        "sauvola",
        "niblack",
        "bernsen",
        "auto",
    ])
    def test_all_methods(self, method):
        imgs = [_text_like(), _noisy()]
        results = batch_binarize(imgs, method=method)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, BinarizeResult)
            assert r.binary.shape == (64, 64)

    def test_kwargs_forwarded_sauvola(self):
        results = batch_binarize([_text_like()], method="sauvola", k=0.5, window_size=21)
        assert results[0].params.get("k") == pytest.approx(0.5)
        assert results[0].params.get("window_size") == 21

    def test_shapes_preserved(self):
        imgs = [_noisy(30, 40), _noisy(50, 60)]
        results = batch_binarize(imgs)
        assert results[0].binary.shape == (30, 40)
        assert results[1].binary.shape == (50, 60)

    def test_single_image(self):
        results = batch_binarize([_text_like()])
        assert len(results) == 1
        assert isinstance(results[0], BinarizeResult)
