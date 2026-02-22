"""Additional tests for puzzle_reconstruction.algorithms.fragment_aligner."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fragment_aligner import (
    AlignmentResult,
    apply_shift,
    batch_align,
    estimate_shift,
    phase_correlation_align,
    template_match_align,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _bgr(h=64, w=64, fill=160):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _striped(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(0, w, 8):
        img[:, col:col + 4] = 200
    return img


def _rand(h=32, w=32, d=None, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (h, w) if d is None else (h, w, d), dtype=np.uint8)
    return arr


# ─── TestAlignmentResultExtra ─────────────────────────────────────────────────

class TestAlignmentResultExtra:
    def test_zero_confidence(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase")
        assert r.confidence == pytest.approx(0.0)

    def test_large_shift(self):
        r = AlignmentResult(dx=1000.0, dy=-500.0, confidence=0.9, method="template")
        assert r.dx == pytest.approx(1000.0)
        assert r.dy == pytest.approx(-500.0)

    def test_multiple_params(self):
        r = AlignmentResult(dx=1.0, dy=2.0, confidence=0.8, method="phase",
                            params={"a": 1, "b": 2, "c": 3})
        assert len(r.params) == 3

    def test_method_template(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=1.0, method="template")
        assert r.method == "template"

    def test_repr_contains_confidence(self):
        r = AlignmentResult(dx=0.5, dy=-0.5, confidence=0.75, method="phase")
        assert "confidence" in repr(r) or "0.75" in repr(r)

    def test_separate_instances_no_shared_params(self):
        r1 = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5, method="phase")
        r2 = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5, method="phase")
        r1.params["x"] = 99
        assert "x" not in r2.params


# ─── TestEstimateShiftExtra ───────────────────────────────────────────────────

class TestEstimateShiftExtra:
    def test_all_zeros_returns_shift_and_conf(self):
        a = np.zeros(32, dtype=np.float32)
        shift, conf = estimate_shift(a, a.copy())
        assert isinstance(shift, float) and isinstance(conf, float)

    def test_float64_input(self):
        a = np.linspace(0, 1, 64).astype(np.float64)
        shift, conf = estimate_shift(a, a.copy())
        assert isinstance(shift, float)

    def test_different_lengths_short(self):
        a = np.arange(3, dtype=np.float32)
        b = np.arange(3, dtype=np.float32)
        shift, conf = estimate_shift(a, b)
        assert isinstance(shift, float)

    def test_large_constant_input(self):
        a = np.full(64, 200, dtype=np.float32)
        b = np.full(64, 200, dtype=np.float32)
        shift, conf = estimate_shift(a, b)
        assert math.isfinite(shift) and math.isfinite(conf)

    def test_random_pairs_finite(self):
        for seed in range(5):
            a = np.random.default_rng(seed).random(64).astype(np.float32)
            b = np.random.default_rng(seed + 10).random(64).astype(np.float32)
            shift, conf = estimate_shift(a, b)
            assert math.isfinite(shift) and math.isfinite(conf)

    def test_confidence_in_01(self):
        a = np.sin(np.linspace(0, 4 * math.pi, 64)).astype(np.float32)
        _, conf = estimate_shift(a, a.copy())
        assert 0.0 <= conf <= 1.0


# ─── TestPhaseCorrelationAlignExtra ──────────────────────────────────────────

class TestPhaseCorrelationAlignExtra:
    def test_large_image(self):
        img = _striped(128, 128)
        r = phase_correlation_align(img, img)
        assert isinstance(r, AlignmentResult)

    def test_non_square_image(self):
        img = _gray(32, 80)
        r = phase_correlation_align(img, img)
        assert isinstance(r, AlignmentResult)

    def test_n_samples_param_stored(self):
        img = _gray()
        r = phase_correlation_align(img, img, n_samples=64)
        assert r.params["n_samples"] == 64

    def test_border_px_param_stored(self):
        img = _striped()
        r = phase_correlation_align(img, img, border_px=8)
        assert r.params["border_px"] == 8

    def test_identical_images_confidence_in_range(self):
        img = _striped()
        r = phase_correlation_align(img, img)
        assert 0.0 <= r.confidence <= 1.0

    def test_different_fills_returns_result(self):
        img1 = _gray(fill=50)
        img2 = _gray(fill=200)
        r = phase_correlation_align(img1, img2)
        assert isinstance(r, AlignmentResult)

    def test_bgr_large_image(self):
        img = _bgr(64, 96)
        r = phase_correlation_align(img, img)
        assert r.method == "phase"

    def test_side_params_stored(self):
        img = _striped()
        r = phase_correlation_align(img, img, side1=0, side2=2)
        assert r.params.get("side1") == 0
        assert r.params.get("side2") == 2


# ─── TestTemplateMatchAlignExtra ─────────────────────────────────────────────

class TestTemplateMatchAlignExtra:
    def test_large_search_range(self):
        img = _striped()
        r = template_match_align(img, img, search_range=20)
        assert isinstance(r, AlignmentResult)

    def test_non_square_image(self):
        img = _gray(32, 80)
        r = template_match_align(img, img)
        assert isinstance(r, AlignmentResult)

    def test_border_px_stored(self):
        img = _gray()
        r = template_match_align(img, img, border_px=10)
        assert r.params["border_px"] == 10

    def test_search_range_stored(self):
        img = _gray()
        r = template_match_align(img, img, search_range=8)
        assert r.params["search_range"] == 8

    def test_confidence_in_range_random(self):
        img = _rand(32, 32, seed=7)
        r = template_match_align(img, img)
        assert 0.0 <= r.confidence <= 1.0

    def test_bgr_input_ok(self):
        img = _bgr()
        r = template_match_align(img, img)
        assert isinstance(r, AlignmentResult)

    def test_method_is_template(self):
        r = template_match_align(_gray(), _gray())
        assert r.method == "template"


# ─── TestApplyShiftExtra ─────────────────────────────────────────────────────

class TestApplyShiftExtra:
    def test_large_dx(self):
        img = _gray()
        out = apply_shift(img, dx=40.0, dy=0.0)
        assert out.shape == img.shape

    def test_negative_dx(self):
        img = _gray()
        out = apply_shift(img, dx=-10.0, dy=0.0)
        assert out.shape == img.shape

    def test_negative_dy(self):
        img = _gray()
        out = apply_shift(img, dx=0.0, dy=-15.0)
        assert out.shape == img.shape

    def test_rgb_shift_shape(self):
        img = _bgr()
        out = apply_shift(img, dx=5.0, dy=5.0)
        assert out.shape == img.shape

    def test_rgb_dtype_uint8(self):
        img = _bgr()
        out = apply_shift(img, dx=5.0, dy=-5.0)
        assert out.dtype == np.uint8

    def test_output_values_in_range(self):
        img = _rand(64, 64, seed=3)
        out = apply_shift(img, dx=10.0, dy=5.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_non_square_shape_preserved(self):
        img = np.zeros((48, 80), dtype=np.uint8)
        out = apply_shift(img, dx=5.0, dy=0.0)
        assert out.shape == (48, 80)

    def test_large_shift_border_fill(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        out = apply_shift(img, dx=28.0, dy=0.0)
        assert out[16, 0] == 255


# ─── TestBatchAlignExtra ─────────────────────────────────────────────────────

class TestBatchAlignExtra:
    def test_single_pair_phase(self):
        imgs = [_striped(), _striped()]
        results = batch_align(imgs, [(0, 1, 0, 2)], method="phase")
        assert len(results) == 1

    def test_five_pairs(self):
        imgs = [_gray() for _ in range(6)]
        pairs = [(i, i + 1, 0, 2) for i in range(5)]
        results = batch_align(imgs, pairs, method="phase")
        assert len(results) == 5

    def test_all_template(self):
        imgs = [_striped(), _striped(), _striped()]
        pairs = [(0, 1, 0, 2), (1, 2, 1, 3)]
        results = batch_align(imgs, pairs, method="template")
        assert all(r.method == "template" for r in results)

    def test_results_all_alignment_result_type(self):
        imgs = [_gray(), _gray()]
        results = batch_align(imgs, [(0, 1, 0, 2)], method="template")
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_empty_pairs_phase(self):
        imgs = [_gray()]
        assert batch_align(imgs, [], method="phase") == []

    def test_kwargs_to_template(self):
        imgs = [_gray(), _gray()]
        results = batch_align(imgs, [(0, 1, 0, 2)],
                              method="template", border_px=6)
        assert results[0].params["border_px"] == 6

    def test_invalid_method_error(self):
        with pytest.raises(ValueError):
            batch_align([_gray(), _gray()], [(0, 1, 0, 2)], method="fft_magic")
