"""
Tests for puzzle_reconstruction.preprocessing.illumination_equalizer
"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_equalizer import (
    IlluminationEqualizerConfig,
    IlluminationEqualizerResult,
    IlluminationEqualizer,
    equalize_fragments,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_gray(h=64, w=64, value=None, seed=0):
    if value is not None:
        return np.full((h, w), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, (h, w), dtype=np.uint8)


def _make_rgb(h=64, w=64, value=None, seed=0):
    if value is not None:
        return np.full((h, w, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, (h, w, 3), dtype=np.uint8)


def _three_gray_images():
    return [_make_gray(seed=i) for i in range(3)]


def _three_rgb_images():
    return [_make_rgb(seed=i) for i in range(3)]


# ─── Config tests ─────────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = IlluminationEqualizerConfig()
    assert cfg.method == "histogram"
    assert cfg.reference_idx == 0
    assert cfg.retinex_scales == [15.0, 80.0, 250.0]
    assert cfg.clahe_clip == 2.0
    assert cfg.clahe_grid == (8, 8)


def test_config_invalid_method():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(method="unknown")


def test_config_invalid_reference_idx():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(reference_idx=-1)


def test_config_empty_retinex_scales():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(retinex_scales=[])


def test_config_invalid_clahe_clip():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(clahe_clip=0.0)


# ─── Result field tests ───────────────────────────────────────────────────────

def test_result_fields():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    assert hasattr(result, "images")
    assert hasattr(result, "uniformity_scores")
    assert hasattr(result, "method")


def test_equalize_returns_result_type():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    assert isinstance(result, IlluminationEqualizerResult)


def test_equalize_single_image():
    images = [_make_gray()]
    result = IlluminationEqualizer().equalize(images)
    assert len(result.images) == 1
    assert result.images[0].shape == images[0].shape


def test_equalize_three_images():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    assert len(result.images) == 3


def test_output_images_same_shape_as_input():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    for orig, out in zip(images, result.images):
        assert out.shape == orig.shape


def test_uniformity_scores_in_01():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    for score in result.uniformity_scores:
        assert 0.0 <= score <= 1.0


def test_method_histogram_works():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="histogram")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    assert result.method == "histogram"
    assert len(result.images) == 3


def test_method_retinex_works():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="retinex")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    assert result.method == "retinex"
    assert len(result.images) == 3


def test_method_clahe_works():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="clahe")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    assert result.method == "clahe"
    assert len(result.images) == 3


def test_identical_images_high_uniformity():
    img = _make_gray(value=128)
    images = [img.copy(), img.copy(), img.copy()]
    result = IlluminationEqualizer().equalize(images)
    for score in result.uniformity_scores:
        assert score > 0.5


def test_very_dark_image_equalized():
    dark = _make_gray(value=5)
    ref = _make_gray(seed=0)
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=1)
    result = IlluminationEqualizer(config=cfg).equalize([dark, ref])
    # The dark image should have a higher mean after equalization
    assert result.images[0].mean() > dark.mean()


def test_very_bright_image_equalized():
    bright = _make_gray(value=250)
    ref = _make_gray(seed=0)
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=1)
    result = IlluminationEqualizer(config=cfg).equalize([bright, ref])
    # The bright image mean should decrease after equalization
    assert result.images[0].mean() < bright.mean()


def test_equalize_fragments_module_level():
    images = _three_gray_images()
    out = equalize_fragments(images)
    assert len(out) == 3


def test_result_images_count_matches_input():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    assert len(result.images) == len(images)


def test_result_method_matches_config():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="retinex")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    assert result.method == "retinex"


def test_grayscale_images_work():
    images = [_make_gray(seed=i) for i in range(2)]
    result = IlluminationEqualizer().equalize(images)
    for img in result.images:
        assert img.ndim == 2


def test_rgb_images_work():
    images = _three_rgb_images()
    result = IlluminationEqualizer().equalize(images)
    for img in result.images:
        assert img.ndim == 3
        assert img.shape[2] == 3


def test_small_images_16x16():
    images = [_make_gray(h=16, w=16, seed=i) for i in range(3)]
    result = IlluminationEqualizer().equalize(images)
    for orig, out in zip(images, result.images):
        assert out.shape == orig.shape


def test_reference_idx_1():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=1)
    result = IlluminationEqualizer(config=cfg).equalize(images)
    # Reference image (index 1) should be unchanged
    np.testing.assert_array_equal(result.images[1], images[1])


def test_reference_image_unchanged():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=0)
    result = IlluminationEqualizer(config=cfg).equalize(images)
    np.testing.assert_array_equal(result.images[0], images[0])


def test_deterministic_histogram():
    images = _three_gray_images()
    result1 = IlluminationEqualizer().equalize(images)
    result2 = IlluminationEqualizer().equalize(images)
    for a, b in zip(result1.images, result2.images):
        np.testing.assert_array_equal(a, b)


def test_deterministic_retinex():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="retinex")
    result1 = IlluminationEqualizer(config=cfg).equalize(images)
    result2 = IlluminationEqualizer(config=cfg).equalize(images)
    for a, b in zip(result1.images, result2.images):
        np.testing.assert_array_equal(a, b)


def test_uniformity_scores_count_matches_images():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    assert len(result.uniformity_scores) == len(images)


def test_empty_images_raises():
    with pytest.raises(ValueError):
        IlluminationEqualizer().equalize([])


def test_reference_idx_out_of_range_raises():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=10)
    with pytest.raises(ValueError):
        IlluminationEqualizer(config=cfg).equalize(images)


def test_equalize_fragments_method_arg():
    images = _three_gray_images()
    out = equalize_fragments(images, method="retinex")
    assert len(out) == 3
    for orig, img in zip(images, out):
        assert img.shape == orig.shape


def test_output_dtype_uint8():
    images = _three_gray_images()
    result = IlluminationEqualizer().equalize(images)
    for img in result.images:
        assert img.dtype == np.uint8


def test_retinex_output_in_valid_range():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="retinex")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    for img in result.images:
        assert img.min() >= 0
        assert img.max() <= 255


def test_clahe_output_in_valid_range():
    images = _three_gray_images()
    cfg = IlluminationEqualizerConfig(method="clahe")
    result = IlluminationEqualizer(config=cfg).equalize(images)
    for img in result.images:
        assert int(img.min()) >= 0
        assert int(img.max()) <= 255
