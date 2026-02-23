"""Extra tests for puzzle_reconstruction.algorithms.fragment_aligner."""
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

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    return img


def _profile(n=64, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 200, n).astype(np.float32)


# ─── TestAlignmentResultExtra ─────────────────────────────────────────────────

class TestAlignmentResultExtra:
    def test_dx_float(self):
        r = AlignmentResult(dx=0.5, dy=0.0, confidence=1.0, method="phase")
        assert isinstance(r.dx, float)

    def test_dy_float(self):
        r = AlignmentResult(dx=0.0, dy=-2.7, confidence=0.9, method="template")
        assert isinstance(r.dy, float)

    def test_confidence_stored(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.42, method="phase")
        assert r.confidence == pytest.approx(0.42)

    def test_method_phase(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase")
        assert r.method == "phase"

    def test_method_template(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="template")
        assert r.method == "template"

    def test_params_empty_default(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase")
        assert r.params == {}

    def test_params_multi_key(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.0, method="phase",
                             params={"side1": 0, "side2": 2, "border_px": 8})
        assert r.params["side1"] == 0
        assert r.params["border_px"] == 8

    def test_repr_is_string(self):
        r = AlignmentResult(dx=1.0, dy=2.0, confidence=0.9, method="phase")
        assert isinstance(repr(r), str)

    def test_repr_contains_method(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5, method="template")
        assert "template" in repr(r)

    def test_dx_negative(self):
        r = AlignmentResult(dx=-5.5, dy=0.0, confidence=0.5, method="phase")
        assert r.dx == pytest.approx(-5.5)


# ─── TestEstimateShiftExtra ───────────────────────────────────────────────────

class TestEstimateShiftExtra:
    def test_returns_tuple_length_2(self):
        p = _profile()
        result = estimate_shift(p, p)
        assert len(result) == 2

    def test_shift_float(self):
        p = _profile()
        sh, _ = estimate_shift(p, p)
        assert isinstance(sh, float)

    def test_confidence_float(self):
        p = _profile()
        _, cf = estimate_shift(p, p)
        assert isinstance(cf, float)

    def test_identical_near_zero_shift(self):
        p = _profile(n=128, seed=10)
        sh, _ = estimate_shift(p, p)
        assert abs(sh) < 2.0

    def test_confidence_in_0_1(self):
        p1 = _profile(seed=1)
        p2 = _profile(seed=2)
        _, cf = estimate_shift(p1, p2)
        assert 0.0 <= cf <= 1.0

    def test_empty_returns_zero_shift(self):
        sh, cf = estimate_shift(np.array([]), np.array([]))
        assert sh == pytest.approx(0.0)

    def test_empty_returns_zero_confidence(self):
        _, cf = estimate_shift(np.array([]), np.array([]))
        assert cf == pytest.approx(0.0)

    def test_constant_profile_confidence_half(self):
        p = np.full(64, 100.0, dtype=np.float32)
        _, cf = estimate_shift(p, p)
        assert cf == pytest.approx(0.5)

    def test_single_element(self):
        sh, cf = estimate_shift(np.array([5.0]), np.array([5.0]))
        assert isinstance(sh, float)


# ─── TestPhaseCorrelationAlignExtra ───────────────────────────────────────────

class TestPhaseCorrelationAlignExtra:
    def test_method_is_phase(self):
        r = phase_correlation_align(_noisy(), _noisy(seed=10))
        assert r.method == "phase"

    def test_confidence_bounded(self):
        r = phase_correlation_align(_noisy(), _noisy(seed=11))
        assert 0.0 <= r.confidence <= 1.0

    def test_dx_is_float(self):
        r = phase_correlation_align(_noisy(), _noisy())
        assert isinstance(r.dx, float)

    def test_dy_is_float(self):
        r = phase_correlation_align(_noisy(), _noisy())
        assert isinstance(r.dy, float)

    def test_gray_same_image(self):
        img = _gray()
        r = phase_correlation_align(img, img)
        assert isinstance(r, AlignmentResult)

    def test_bgr_images_accepted(self):
        r = phase_correlation_align(_bgr(), _bgr())
        assert isinstance(r, AlignmentResult)

    def test_side1_stored(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=1, side2=3)
        assert r.params.get("side1") == 1

    def test_side2_stored(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.params.get("side2") == 2

    def test_border_px_stored(self):
        r = phase_correlation_align(_noisy(), _noisy(), border_px=16)
        assert r.params.get("border_px") == 16

    def test_n_samples_stored(self):
        r = phase_correlation_align(_noisy(), _noisy(), n_samples=48)
        assert r.params.get("n_samples") == 48

    def test_horizontal_dy_zero(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.dy == pytest.approx(0.0)

    def test_vertical_dx_zero(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=1, side2=3)
        assert r.dx == pytest.approx(0.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_side_parametrize(self, side):
        r = phase_correlation_align(_noisy(seed=side), _noisy(seed=side + 10),
                                     side1=side, side2=side)
        assert isinstance(r, AlignmentResult)


# ─── TestTemplateMatchAlignExtra ──────────────────────────────────────────────

class TestTemplateMatchAlignExtra:
    def test_method_is_template(self):
        r = template_match_align(_noisy(), _noisy(seed=20))
        assert r.method == "template"

    def test_confidence_bounded(self):
        r = template_match_align(_noisy(), _noisy(seed=21))
        assert 0.0 <= r.confidence <= 1.0

    def test_dx_is_float(self):
        r = template_match_align(_noisy(), _noisy())
        assert isinstance(r.dx, float)

    def test_dy_is_float(self):
        r = template_match_align(_noisy(), _noisy())
        assert isinstance(r.dy, float)

    def test_gray_images(self):
        r = template_match_align(_gray(), _gray())
        assert isinstance(r, AlignmentResult)

    def test_bgr_images(self):
        r = template_match_align(_bgr(), _bgr())
        assert isinstance(r, AlignmentResult)

    def test_side_params_stored(self):
        r = template_match_align(_noisy(), _noisy(), side1=3, side2=1)
        assert r.params.get("side1") == 3
        assert r.params.get("side2") == 1

    def test_border_px_stored(self):
        r = template_match_align(_noisy(), _noisy(), border_px=12)
        assert r.params.get("border_px") == 12

    def test_search_range_stored(self):
        r = template_match_align(_noisy(), _noisy(), search_range=6)
        assert r.params.get("search_range") == 6

    def test_horizontal_dy_zero(self):
        r = template_match_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.dy == pytest.approx(0.0)

    def test_vertical_dx_zero(self):
        r = template_match_align(_noisy(), _noisy(), side1=1, side2=3)
        assert r.dx == pytest.approx(0.0)


# ─── TestApplyShiftExtra ──────────────────────────────────────────────────────

class TestApplyShiftExtra:
    def test_gray_shape_preserved(self):
        img = _gray(32, 48)
        out = apply_shift(img, 3.0, 0.0)
        assert out.shape == (32, 48)

    def test_bgr_shape_preserved(self):
        img = _bgr(32, 48)
        out = apply_shift(img, 0.0, 2.0)
        assert out.shape == (32, 48, 3)

    def test_dtype_uint8(self):
        out = apply_shift(_gray(), 1.0, 1.0)
        assert out.dtype == np.uint8

    def test_zero_shift_similar_to_original(self):
        img = _noisy(seed=100)
        out = apply_shift(img, 0.0, 0.0)
        diff = np.abs(img.astype(float) - out.astype(float))
        assert diff.mean() < 5.0

    def test_large_shift_changes_image(self):
        img = _noisy(seed=200)
        out = apply_shift(img, 30.0, 0.0)
        assert not np.array_equal(img, out)

    def test_returns_ndarray(self):
        out = apply_shift(_gray(), 0.0, 0.0)
        assert isinstance(out, np.ndarray)

    def test_bgr_3channels(self):
        out = apply_shift(_bgr(), 5.0, 5.0)
        assert out.ndim == 3


# ─── TestBatchAlignExtra ──────────────────────────────────────────────────────

class TestBatchAlignExtra:
    def test_empty_pairs_empty_list(self):
        assert batch_align([_noisy()], [], method="phase") == []

    def test_single_pair_phase(self):
        imgs = [_noisy(seed=0), _noisy(seed=1)]
        result = batch_align(imgs, [(0, 1, 1, 3)], method="phase")
        assert len(result) == 1
        assert isinstance(result[0], AlignmentResult)

    def test_single_pair_template(self):
        imgs = [_noisy(seed=0), _noisy(seed=1)]
        result = batch_align(imgs, [(0, 1, 0, 2)], method="template")
        assert result[0].method == "template"

    def test_multiple_pairs_count(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        pairs = [(0, 1, 0, 2), (1, 2, 1, 3), (2, 3, 2, 0)]
        result = batch_align(imgs, pairs, method="phase")
        assert len(result) == 3

    def test_unknown_method_raises(self):
        imgs = [_noisy(), _noisy(seed=1)]
        with pytest.raises(ValueError):
            batch_align(imgs, [(0, 1, 0, 2)], method="non_existent")

    def test_all_results_are_alignment_result(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        results = batch_align(imgs, [(0, 1, 0, 2), (1, 2, 1, 3)], method="phase")
        for r in results:
            assert isinstance(r, AlignmentResult)

    def test_n_samples_forwarded(self):
        imgs = [_noisy(), _noisy(seed=1)]
        result = batch_align(imgs, [(0, 1, 0, 2)], method="phase", n_samples=16)
        assert result[0].params.get("n_samples") == 16

    def test_search_range_forwarded(self):
        imgs = [_noisy(), _noisy(seed=2)]
        result = batch_align(imgs, [(0, 1, 1, 3)], method="template",
                              search_range=4)
        assert result[0].params.get("search_range") == 4

    @pytest.mark.parametrize("method", ["phase", "template"])
    def test_methods_return_correct_method_name(self, method):
        imgs = [_noisy(seed=i) for i in range(2)]
        result = batch_align(imgs, [(0, 1, 0, 2)], method=method)
        assert result[0].method == method
