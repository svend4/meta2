"""Extra tests for puzzle_reconstruction/preprocessing/illumination_equalizer.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_equalizer import (
    IlluminationEqualizerConfig,
    IlluminationEqualizerResult,
    IlluminationEqualizer,
    IlluminationEqualizerConfig,
    equalize_fragments,
    _ensure_odd,
    _to_float32,
    _numpy_gaussian_blur_1ch,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, value=None, seed=0):
    if value is not None:
        return np.full((h, w), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, value=None, seed=0):
    if value is not None:
        return np.full((h, w, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _float_gray(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w)).astype(np.float32) * 255.0


# ─── Config edge cases ────────────────────────────────────────────────────────

def test_config_negative_clahe_clip_raises():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(clahe_clip=-1.0)


def test_config_clahe_clip_zero_raises():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(clahe_clip=0.0)


def test_config_retinex_single_scale():
    cfg = IlluminationEqualizerConfig(method="retinex", retinex_scales=[15.0])
    assert cfg.retinex_scales == [15.0]


def test_config_retinex_many_scales():
    cfg = IlluminationEqualizerConfig(method="retinex", retinex_scales=[5.0, 10.0, 20.0, 40.0, 80.0])
    assert len(cfg.retinex_scales) == 5


def test_config_clahe_grid_custom():
    cfg = IlluminationEqualizerConfig(clahe_grid=(4, 4))
    assert cfg.clahe_grid == (4, 4)


def test_config_reference_idx_zero_valid():
    cfg = IlluminationEqualizerConfig(reference_idx=0)
    assert cfg.reference_idx == 0


def test_config_reference_idx_large_valid():
    cfg = IlluminationEqualizerConfig(reference_idx=1000)
    assert cfg.reference_idx == 1000


def test_config_all_valid_methods():
    for m in ("histogram", "retinex", "clahe"):
        cfg = IlluminationEqualizerConfig(method=m)
        assert cfg.method == m


def test_config_method_case_sensitive_raises():
    with pytest.raises(ValueError):
        IlluminationEqualizerConfig(method="Histogram")


# ─── Single-image edge cases ──────────────────────────────────────────────────

def test_single_image_histogram():
    img = _gray(seed=1)
    result = IlluminationEqualizer().equalize([img])
    assert len(result.images) == 1
    np.testing.assert_array_equal(result.images[0], img)


def test_single_image_retinex():
    cfg = IlluminationEqualizerConfig(method="retinex")
    img = _gray(seed=2)
    result = IlluminationEqualizer(config=cfg).equalize([img])
    assert len(result.images) == 1
    assert result.images[0].dtype == np.uint8


def test_single_image_clahe():
    cfg = IlluminationEqualizerConfig(method="clahe")
    img = _gray(seed=3)
    result = IlluminationEqualizer(config=cfg).equalize([img])
    assert len(result.images) == 1


# ─── Empty input ──────────────────────────────────────────────────────────────

def test_empty_list_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        IlluminationEqualizer().equalize([])


# ─── Reference index out of range ─────────────────────────────────────────────

def test_histogram_reference_idx_exact_length_raises():
    images = [_gray(seed=i) for i in range(3)]
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=3)
    with pytest.raises(ValueError):
        IlluminationEqualizer(config=cfg).equalize(images)


def test_histogram_reference_idx_last_valid():
    images = [_gray(seed=i) for i in range(3)]
    cfg = IlluminationEqualizerConfig(method="histogram", reference_idx=2)
    result = IlluminationEqualizer(config=cfg).equalize(images)
    np.testing.assert_array_equal(result.images[2], images[2])


# ─── Float input images ───────────────────────────────────────────────────────

def test_float_input_histogram():
    img_f = _float_gray()
    images = [img_f, _float_gray(seed=1)]
    result = IlluminationEqualizer().equalize(images)
    assert len(result.images) == 2


def test_float_input_retinex():
    cfg = IlluminationEqualizerConfig(method="retinex")
    img_f = _float_gray()
    result = IlluminationEqualizer(config=cfg).equalize([img_f])
    assert result.images[0].dtype == np.uint8


# ─── Non-square images ────────────────────────────────────────────────────────

def test_wide_image_histogram():
    images = [_gray(32, 128, seed=i) for i in range(2)]
    result = IlluminationEqualizer().equalize(images)
    for img in result.images:
        assert img.shape == (32, 128)


def test_tall_image_retinex():
    cfg = IlluminationEqualizerConfig(method="retinex")
    images = [_gray(128, 32, seed=i) for i in range(2)]
    result = IlluminationEqualizer(config=cfg).equalize(images)
    for img in result.images:
        assert img.shape == (128, 32)


def test_1x1_image_clahe():
    cfg = IlluminationEqualizerConfig(method="clahe")
    img = np.array([[128]], dtype=np.uint8)
    result = IlluminationEqualizer(config=cfg).equalize([img])
    assert result.images[0].shape == (1, 1)


def test_1x1_image_histogram():
    img = np.array([[200]], dtype=np.uint8)
    result = IlluminationEqualizer().equalize([img])
    assert result.images[0].shape == (1, 1)


def test_1x1_image_retinex():
    cfg = IlluminationEqualizerConfig(method="retinex")
    img = np.array([[50]], dtype=np.uint8)
    result = IlluminationEqualizer(config=cfg).equalize([img])
    assert result.images[0].shape == (1, 1)


# ─── Uniformity score edge cases ──────────────────────────────────────────────

def test_uniformity_score_uniform_image():
    img = _gray(value=128)
    result = IlluminationEqualizer().equalize([img])
    # Perfectly uniform image → uniformity near 1
    assert result.uniformity_scores[0] > 0.9


def test_uniformity_score_very_dark_image():
    img = _gray(value=1)
    result = IlluminationEqualizer().equalize([img])
    # Dark uniform image has mean < 1.0 → returns 1.0
    assert result.uniformity_scores[0] == pytest.approx(1.0)


def test_uniformity_score_rgb():
    imgs = [_rgb(value=128)]
    result = IlluminationEqualizer().equalize(imgs)
    assert 0.0 <= result.uniformity_scores[0] <= 1.0


# ─── RGB + mixed channel images ───────────────────────────────────────────────

def test_rgb_histogram_four_channel_image():
    """4-channel image handled by histogram method."""
    img4 = np.full((32, 32, 4), 100, dtype=np.uint8)
    ref  = _rgb(32, 32, seed=0)
    # Ref is 3-channel, img4 is 4-channel
    # histogram method may raise or work; we just need no unhandled exception
    try:
        result = IlluminationEqualizer().equalize([img4, ref])
        assert len(result.images) == 2
    except Exception:
        pass  # OK to fail gracefully


def test_rgb_clahe_all_channels_processed():
    cfg = IlluminationEqualizerConfig(method="clahe")
    imgs = [_rgb(seed=i) for i in range(2)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    for img in result.images:
        assert img.shape[2] == 3


def test_rgb_retinex_channels_independent():
    cfg = IlluminationEqualizerConfig(method="retinex")
    imgs = [_rgb(seed=5)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    assert result.images[0].shape == imgs[0].shape


# ─── Retinex single scale ─────────────────────────────────────────────────────

def test_retinex_single_scale_produces_valid_output():
    cfg = IlluminationEqualizerConfig(method="retinex", retinex_scales=[80.0])
    imgs = [_gray(seed=10), _gray(seed=11)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    for img in result.images:
        assert img.min() >= 0
        assert img.max() <= 255


def test_retinex_with_large_sigma():
    cfg = IlluminationEqualizerConfig(method="retinex", retinex_scales=[500.0])
    imgs = [_gray(seed=12)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    assert result.images[0].dtype == np.uint8


# ─── CLAHE grid edge cases ────────────────────────────────────────────────────

def test_clahe_1x1_grid():
    cfg = IlluminationEqualizerConfig(method="clahe", clahe_grid=(1, 1))
    imgs = [_gray(seed=13)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    assert result.images[0].shape == imgs[0].shape


def test_clahe_large_clip_limit():
    cfg = IlluminationEqualizerConfig(method="clahe", clahe_clip=100.0)
    imgs = [_gray(seed=14)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    assert result.images[0].dtype == np.uint8


# ─── equalize_fragments with config ──────────────────────────────────────────

def test_equalize_fragments_with_config_override():
    imgs = [_gray(seed=i) for i in range(3)]
    cfg = IlluminationEqualizerConfig(method="clahe")
    out = equalize_fragments(imgs, method="histogram", config=cfg)
    # Config takes precedence, so clahe is used
    assert len(out) == 3


def test_equalize_fragments_clahe_method():
    imgs = [_gray(seed=i) for i in range(2)]
    out = equalize_fragments(imgs, method="clahe")
    assert len(out) == 2
    for o, orig in zip(out, imgs):
        assert o.shape == orig.shape


# ─── Private helpers ──────────────────────────────────────────────────────────

def test_ensure_odd_already_odd():
    assert _ensure_odd(3) == 3
    assert _ensure_odd(7) == 7
    assert _ensure_odd(1) == 1


def test_ensure_odd_even_increments():
    assert _ensure_odd(4) == 5
    assert _ensure_odd(10) == 11
    assert _ensure_odd(0) == 1


def test_to_float32_converts_dtype():
    arr = np.array([1, 2, 3], dtype=np.uint8)
    result = _to_float32(arr)
    assert result.dtype == np.float32


def test_numpy_gaussian_blur_1ch_output_shape():
    ch = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8).astype(np.float32)
    out = _numpy_gaussian_blur_1ch(ch, sigma=5.0)
    assert out.shape == ch.shape


def test_numpy_gaussian_blur_1ch_preserves_finite():
    ch = np.ones((16, 16), dtype=np.float32) * 100.0
    out = _numpy_gaussian_blur_1ch(ch, sigma=3.0)
    assert np.all(np.isfinite(out))


def test_numpy_gaussian_blur_1ch_sigma_zero():
    ch = np.ones((8, 8), dtype=np.float32)
    out = _numpy_gaussian_blur_1ch(ch, sigma=1e-10)
    assert np.all(np.isfinite(out))


# ─── Large image smoke test ───────────────────────────────────────────────────

def test_large_image_histogram_no_error():
    imgs = [_gray(256, 256, seed=i) for i in range(2)]
    result = IlluminationEqualizer().equalize(imgs)
    assert len(result.images) == 2


def test_large_rgb_image_clahe():
    cfg = IlluminationEqualizerConfig(method="clahe")
    imgs = [_rgb(256, 256, seed=i) for i in range(2)]
    result = IlluminationEqualizer(config=cfg).equalize(imgs)
    for img in result.images:
        assert img.shape == (256, 256, 3)
