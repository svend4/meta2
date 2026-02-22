"""Тесты для puzzle_reconstruction/algorithms/fragment_aligner.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fragment_aligner import (
    AlignmentResult,
    estimate_shift,
    phase_correlation_align,
    template_match_align,
    apply_shift,
    batch_align,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
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


def _profile(n=64, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 200, n).astype(np.float32)


# ─── AlignmentResult ──────────────────────────────────────────────────────────

class TestAlignmentResult:
    def test_fields(self):
        r = AlignmentResult(dx=1.5, dy=-0.3, confidence=0.8, method="phase")
        assert r.dx == pytest.approx(1.5)
        assert r.dy == pytest.approx(-0.3)
        assert r.confidence == pytest.approx(0.8)
        assert r.method == "phase"

    def test_params_default_empty(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5, method="phase")
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.5, method="template",
                             params={"search_range": 10})
        assert r.params["search_range"] == 10

    def test_repr(self):
        r = AlignmentResult(dx=2.1, dy=0.0, confidence=0.75, method="phase")
        s = repr(r)
        assert "AlignmentResult" in s
        assert "phase" in s

    def test_repr_contains_dx(self):
        r = AlignmentResult(dx=3.14, dy=0.0, confidence=0.5, method="phase")
        s = repr(r)
        assert "3.14" in s or "dx" in s.lower()

    def test_confidence_stored(self):
        r = AlignmentResult(dx=0.0, dy=0.0, confidence=0.92, method="template")
        assert r.confidence == pytest.approx(0.92)


# ─── estimate_shift ───────────────────────────────────────────────────────────

class TestEstimateShift:
    def test_returns_tuple(self):
        p = _profile()
        result = estimate_shift(p, p)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self):
        p       = _profile()
        sh, cf  = estimate_shift(p, p)
        assert isinstance(sh, float)
        assert isinstance(cf, float)

    def test_same_profile_near_zero(self):
        p      = _profile()
        sh, _  = estimate_shift(p, p)
        assert abs(sh) < 2.0   # identical → shift ≈ 0

    def test_confidence_in_range(self):
        p1 = _profile(seed=1)
        p2 = _profile(seed=2)
        _, cf = estimate_shift(p1, p2)
        assert 0.0 <= cf <= 1.0

    def test_constant_profiles(self):
        p1 = np.full(32, 128.0, dtype=np.float32)
        p2 = np.full(32, 128.0, dtype=np.float32)
        sh, cf = estimate_shift(p1, p2)
        assert isinstance(sh, float)
        assert cf == pytest.approx(0.5)

    def test_empty_profiles(self):
        sh, cf = estimate_shift(np.array([]), np.array([]))
        assert sh == pytest.approx(0.0)
        assert cf == pytest.approx(0.0)

    def test_single_element(self):
        sh, cf = estimate_shift(
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
        )
        assert isinstance(sh, float)

    def test_noisy_identical_profiles(self):
        p      = _profile(n=64, seed=42)
        sh, _  = estimate_shift(p, p)
        assert abs(sh) < 2.0


# ─── phase_correlation_align ──────────────────────────────────────────────────

class TestPhaseCorrelationAlign:
    def test_returns_result(self):
        r = phase_correlation_align(_noisy(), _noisy(seed=9))
        assert isinstance(r, AlignmentResult)

    def test_method(self):
        assert phase_correlation_align(_noisy(), _noisy()).method == "phase"

    def test_confidence_in_range(self):
        r = phase_correlation_align(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.confidence <= 1.0

    def test_dx_is_float(self):
        r = phase_correlation_align(_noisy(), _noisy())
        assert isinstance(r.dx, float)

    def test_dy_is_float(self):
        r = phase_correlation_align(_noisy(), _noisy())
        assert isinstance(r.dy, float)

    def test_params_side1_stored(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.params.get("side1") == 0
        assert r.params.get("side2") == 2

    def test_params_border_px(self):
        r = phase_correlation_align(_noisy(), _noisy(), border_px=12)
        assert r.params.get("border_px") == 12

    def test_params_n_samples(self):
        r = phase_correlation_align(_noisy(), _noisy(), n_samples=32)
        assert r.params.get("n_samples") == 32

    def test_gray_input(self):
        r = phase_correlation_align(_gray(), _gray())
        assert isinstance(r, AlignmentResult)

    def test_bgr_input(self):
        r = phase_correlation_align(_bgr(), _bgr())
        assert 0.0 <= r.confidence <= 1.0

    def test_horizontal_edge_zero_dy(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.dy == pytest.approx(0.0)

    def test_vertical_edge_zero_dx(self):
        r = phase_correlation_align(_noisy(), _noisy(), side1=1, side2=3)
        assert r.dx == pytest.approx(0.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = phase_correlation_align(_noisy(64, 64), _noisy(64, 64, seed=7),
                                     side1=side, side2=side)
        assert isinstance(r, AlignmentResult)


# ─── template_match_align ─────────────────────────────────────────────────────

class TestTemplateMatchAlign:
    def test_returns_result(self):
        r = template_match_align(_noisy(), _noisy(seed=9))
        assert isinstance(r, AlignmentResult)

    def test_method(self):
        assert template_match_align(_noisy(), _noisy()).method == "template"

    def test_confidence_in_range(self):
        r = template_match_align(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.confidence <= 1.0

    def test_dx_is_float(self):
        assert isinstance(template_match_align(_noisy(), _noisy()).dx, float)

    def test_dy_is_float(self):
        assert isinstance(template_match_align(_noisy(), _noisy()).dy, float)

    def test_params_stored(self):
        r = template_match_align(_noisy(), _noisy(), side1=2, side2=0,
                                  border_px=10, search_range=8)
        assert r.params.get("side1") == 2
        assert r.params.get("side2") == 0
        assert r.params.get("border_px") == 10
        assert r.params.get("search_range") == 8

    def test_gray_input(self):
        r = template_match_align(_gray(), _gray())
        assert isinstance(r, AlignmentResult)

    def test_bgr_input(self):
        r = template_match_align(_bgr(), _bgr())
        assert 0.0 <= r.confidence <= 1.0

    def test_horizontal_edge_zero_dy(self):
        r = template_match_align(_noisy(), _noisy(), side1=0, side2=2)
        assert r.dy == pytest.approx(0.0)

    def test_vertical_edge_zero_dx(self):
        r = template_match_align(_noisy(), _noisy(), side1=1, side2=3)
        assert r.dx == pytest.approx(0.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = template_match_align(_noisy(64, 64), _noisy(64, 64, seed=9),
                                  side1=side, side2=side)
        assert isinstance(r, AlignmentResult)


# ─── apply_shift ──────────────────────────────────────────────────────────────

class TestApplyShift:
    def test_returns_ndarray(self):
        assert isinstance(apply_shift(_gray(), 0.0, 0.0), np.ndarray)

    def test_same_shape_gray(self):
        img = _gray(40, 60)
        r   = apply_shift(img, 5.0, 0.0)
        assert r.shape == (40, 60)

    def test_same_shape_bgr(self):
        img = _bgr(40, 60)
        r   = apply_shift(img, 0.0, 3.0)
        assert r.shape == (40, 60, 3)

    def test_dtype_uint8(self):
        r = apply_shift(_gray(), 2.0, 1.0)
        assert r.dtype == np.uint8

    def test_zero_shift_similar(self):
        img = _noisy()
        r   = apply_shift(img, 0.0, 0.0)
        # Zero shift should keep image nearly identical
        diff = np.abs(img.astype(np.float32) - r.astype(np.float32))
        assert diff.mean() < 5.0

    def test_nonzero_shift_changes_image(self):
        img = _noisy()
        r   = apply_shift(img, 10.0, 0.0)
        assert not np.array_equal(img, r)

    def test_bgr_input(self):
        r = apply_shift(_bgr(), 3.0, 2.0)
        assert r.ndim == 3


# ─── batch_align ──────────────────────────────────────────────────────────────

class TestBatchAlign:
    def test_returns_list(self):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        r     = batch_align(imgs, pairs, method="phase")
        assert isinstance(r, list)
        assert len(r) == 2

    def test_each_is_result(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        for r in batch_align(imgs, [(0, 1, 1, 3)], method="phase"):
            assert isinstance(r, AlignmentResult)

    def test_empty_pairs(self):
        assert batch_align([_noisy()], [], method="phase") == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_align([_noisy(), _noisy()], [(0, 1, 1, 3)], method="magic_xyz")

    @pytest.mark.parametrize("method", ["phase", "template"])
    def test_both_methods(self, method):
        imgs  = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2)]
        r     = batch_align(imgs, pairs, method=method)
        assert len(r) == 2
        for result in r:
            assert isinstance(result, AlignmentResult)
            assert result.method == method

    def test_kwargs_forwarded_n_samples(self):
        imgs  = [_noisy(), _noisy(seed=1)]
        r = batch_align(imgs, [(0, 1, 1, 3)], method="phase", n_samples=32)
        assert r[0].params.get("n_samples") == 32

    def test_kwargs_forwarded_search_range(self):
        imgs = [_noisy(), _noisy(seed=1)]
        r = batch_align(imgs, [(0, 1, 1, 3)], method="template", search_range=5)
        assert r[0].params.get("search_range") == 5

    def test_bgr_input(self):
        imgs  = [_bgr(), _bgr()]
        r     = batch_align(imgs, [(0, 1, 1, 3)], method="phase")
        assert isinstance(r[0], AlignmentResult)
