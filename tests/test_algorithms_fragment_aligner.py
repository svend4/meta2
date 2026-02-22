"""Tests for puzzle_reconstruction.algorithms.fragment_aligner."""
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


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 120
    img[:, :, 2] = 160
    return img


def _striped(h=64, w=64):
    """Grayscale with vertical stripes — gives a clear correlation peak."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(0, w, 8):
        img[:, col : col + 4] = 200
    return img


# ─── TestAlignmentResult ──────────────────────────────────────────────────────

class TestAlignmentResult:
    def test_basic_creation(self):
        r = AlignmentResult(dx=1.5, dy=-0.3, confidence=0.9, method="phase")
        assert r.dx == pytest.approx(1.5)
        assert r.dy == pytest.approx(-0.3)
        assert r.confidence == pytest.approx(0.9)
        assert r.method == "phase"
        assert r.params == {}

    def test_params_stored(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5,
                            method="template", params={"border_px": 8})
        assert r.params["border_px"] == 8

    def test_repr_contains_fields(self):
        r = AlignmentResult(dx=2.0, dy=-1.0, confidence=0.75, method="phase")
        s = repr(r)
        assert "dx=" in s
        assert "dy=" in s
        assert "phase" in s

    def test_default_params_is_dict(self):
        r1 = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase")
        r2 = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase")
        r1.params["key"] = 1
        assert "key" not in r2.params  # no shared default


# ─── TestEstimateShift ────────────────────────────────────────────────────────

class TestEstimateShift:
    def test_too_short_returns_zero(self):
        shift, conf = estimate_shift(np.array([1.0]), np.array([1.0]))
        assert shift == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)

    def test_zero_std_returns_half_confidence(self):
        a = np.ones(32, dtype=np.float32)
        b = np.ones(32, dtype=np.float32)
        shift, conf = estimate_shift(a, b)
        assert conf == pytest.approx(0.5)

    def test_identical_strips_near_zero_shift(self):
        a = np.sin(np.linspace(0, 4 * np.pi, 64)).astype(np.float32)
        shift, conf = estimate_shift(a, a.copy())
        assert abs(shift) < 2.0
        assert conf > 0.0

    def test_returns_tuple_of_floats(self):
        a = np.random.default_rng(0).random(32).astype(np.float32)
        b = np.random.default_rng(1).random(32).astype(np.float32)
        result = estimate_shift(a, b)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_known_shift(self):
        """Shift b by +4 → estimate_shift should return near -4 or +4."""
        base = np.array([0, 10, 50, 200, 50, 10, 0] * 9 + [0], dtype=np.float32)
        shifted = np.roll(base, 4)
        shift, conf = estimate_shift(base, shifted)
        assert abs(abs(shift) - 4.0) < 2.5  # subpixel accuracy within ±2.5


# ─── TestPhaseCorrelationAlign ────────────────────────────────────────────────

class TestPhaseCorrelationAlign:
    def test_returns_alignment_result(self):
        img1 = _gray()
        img2 = _gray()
        result = phase_correlation_align(img1, img2)
        assert isinstance(result, AlignmentResult)
        assert result.method == "phase"

    def test_method_param_stored(self):
        img1 = _gray()
        img2 = _gray()
        result = phase_correlation_align(img1, img2, border_px=4, n_samples=32)
        assert result.params["border_px"] == 4
        assert result.params["n_samples"] == 32

    def test_horizontal_side_sets_dx(self):
        """Side 0 (top) is horizontal → shift maps to dx, dy==0."""
        img1 = _striped()
        img2 = _striped()
        result = phase_correlation_align(img1, img2, side1=0, side2=2)
        assert result.dy == pytest.approx(0.0)
        assert result.params["side1"] == 0

    def test_vertical_side_sets_dy(self):
        """Side 1 (right) is vertical → shift maps to dy, dx==0."""
        img1 = _striped()
        img2 = _striped()
        result = phase_correlation_align(img1, img2, side1=1, side2=3)
        assert result.dx == pytest.approx(0.0)

    def test_bgr_input_accepted(self):
        img1 = _bgr()
        img2 = _bgr()
        result = phase_correlation_align(img1, img2)
        assert isinstance(result, AlignmentResult)

    def test_confidence_in_range(self):
        img1 = _striped()
        img2 = _striped()
        result = phase_correlation_align(img1, img2)
        assert 0.0 <= result.confidence <= 1.0


# ─── TestTemplateMatchAlign ───────────────────────────────────────────────────

class TestTemplateMatchAlign:
    def test_returns_alignment_result(self):
        img1 = _gray()
        img2 = _gray()
        result = template_match_align(img1, img2)
        assert isinstance(result, AlignmentResult)
        assert result.method == "template"

    def test_params_stored(self):
        img1 = _gray()
        img2 = _gray()
        result = template_match_align(img1, img2, border_px=6, search_range=5)
        assert result.params["border_px"] == 6
        assert result.params["search_range"] == 5

    def test_constant_image_returns_half_confidence(self):
        """Constant strips → zero std → confidence=0.5."""
        img1 = _gray(fill=100)
        img2 = _gray(fill=100)
        result = template_match_align(img1, img2, border_px=4)
        assert result.confidence == pytest.approx(0.5)

    def test_horizontal_side_gives_dy_zero(self):
        img1 = _striped()
        img2 = _striped()
        result = template_match_align(img1, img2, side1=0, side2=2)
        assert result.dy == pytest.approx(0.0)

    def test_vertical_side_gives_dx_zero(self):
        img1 = _striped()
        img2 = _striped()
        result = template_match_align(img1, img2, side1=1, side2=3)
        assert result.dx == pytest.approx(0.0)

    def test_confidence_in_range(self):
        img1 = _striped()
        img2 = _striped()
        result = template_match_align(img1, img2)
        assert 0.0 <= result.confidence <= 1.0


# ─── TestApplyShift ───────────────────────────────────────────────────────────

class TestApplyShift:
    def test_zero_shift_identity(self):
        img = _striped()
        out = apply_shift(img, dx=0.0, dy=0.0)
        assert out.shape == img.shape
        np.testing.assert_array_equal(out, img)

    def test_output_shape_preserved(self):
        img = _bgr()
        out = apply_shift(img, dx=5.0, dy=-3.0)
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        out = apply_shift(img, dx=2.0, dy=0.0)
        assert out.dtype == np.uint8

    def test_border_filled_with_255(self):
        """Shifting right → left column should be filled with 255."""
        img = np.zeros((32, 32), dtype=np.uint8)
        out = apply_shift(img, dx=10.0, dy=0.0)
        assert int(out[16, 0]) == 255  # border area

    def test_shift_moves_content(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10, 10] = 200
        out = apply_shift(img, dx=5.0, dy=0.0)
        # Content should appear around column 15 now
        assert out[10, 15] > 100  # billinear interpolation, close enough


# ─── TestBatchAlign ───────────────────────────────────────────────────────────

class TestBatchAlign:
    def test_unknown_method_raises(self):
        imgs = [_gray(), _gray()]
        with pytest.raises(ValueError, match="Unknown"):
            batch_align(imgs, [(0, 1, 1, 3)], method="invalid")

    def test_returns_list_of_results(self):
        imgs = [_gray(), _gray(), _gray()]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        results = batch_align(imgs, pairs, method="phase")
        assert len(results) == 2
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_empty_pairs(self):
        imgs = [_gray()]
        results = batch_align(imgs, [], method="phase")
        assert results == []

    def test_template_method(self):
        imgs = [_striped(), _striped()]
        results = batch_align(imgs, [(0, 1, 1, 3)], method="template")
        assert len(results) == 1
        assert results[0].method == "template"

    def test_kwargs_passed_through(self):
        imgs = [_gray(), _gray()]
        results = batch_align(imgs, [(0, 0, 1, 2)], method="phase",
                              border_px=4, n_samples=32)
        assert results[0].params["border_px"] == 4
