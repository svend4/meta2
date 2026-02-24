"""Extra tests for puzzle_reconstruction/preprocessing/warp_corrector.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.warp_corrector import (
    WarpConfig,
    WarpEstimate,
    WarpResult,
    apply_warp,
    batch_correct_warp,
    correct_warp,
    estimate_warp,
    warp_score,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _est(skew=0.0, sx=1.0, sy=1.0, confidence=1.0):
    import math
    c, s = math.cos(math.radians(skew)), math.sin(math.radians(skew))
    m = np.array([[c * sx, -s * sy, 0.0], [s * sx, c * sy, 0.0]])
    return WarpEstimate(matrix=m, skew_angle=skew, scale_x=sx, scale_y=sy,
                        translation=(0.0, 0.0), confidence=confidence)


# ─── TestWarpConfigExtra ──────────────────────────────────────────────────────

class TestWarpConfigExtra:
    def test_max_iter_100_valid(self):
        cfg = WarpConfig(max_iter=100)
        assert cfg.max_iter == 100

    def test_border_mode_zero_valid(self):
        cfg = WarpConfig(border_mode="zero")
        assert cfg.border_mode == "zero"

    def test_output_size_1x1_valid(self):
        cfg = WarpConfig(output_size=(1, 1))
        assert cfg.output_size == (1, 1)

    def test_output_size_non_square(self):
        cfg = WarpConfig(output_size=(200, 100))
        assert cfg.output_size == (200, 100)

    def test_convergence_eps_large_valid(self):
        cfg = WarpConfig(convergence_eps=1.0)
        assert cfg.convergence_eps == pytest.approx(1.0)

    def test_border_mode_replicate_stored(self):
        cfg = WarpConfig(border_mode="replicate")
        assert cfg.border_mode == "replicate"

    def test_default_output_size(self):
        cfg = WarpConfig()
        assert cfg.output_size == (128, 128)


# ─── TestWarpEstimateExtra ────────────────────────────────────────────────────

class TestWarpEstimateExtra:
    def test_confidence_1_valid(self):
        e = WarpEstimate(matrix=_identity(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=1.0)
        assert e.confidence == pytest.approx(1.0)

    def test_scale_x_2_valid(self):
        m = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        e = WarpEstimate(matrix=m, skew_angle=0.0,
                         scale_x=2.0, scale_y=2.0, translation=(0.0, 0.0))
        assert e.scale_x == pytest.approx(2.0)

    def test_large_skew_valid(self):
        e = _est(skew=45.0)
        assert e.skew_angle == pytest.approx(45.0)

    def test_translation_stored(self):
        m = _identity().copy()
        m[0, 2] = 10.0
        m[1, 2] = 5.0
        e = WarpEstimate(matrix=m, skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                         translation=(10.0, 5.0))
        assert e.translation == (10.0, 5.0)

    def test_is_identity_skew_zero(self):
        e = _est(skew=0.0)
        assert isinstance(e.is_identity, bool)

    def test_rotation_deg_is_float(self):
        e = _est(skew=30.0)
        assert isinstance(e.rotation_deg, float)


# ─── TestWarpResultExtra ──────────────────────────────────────────────────────

class TestWarpResultExtra:
    def test_large_n_iter(self):
        r = WarpResult(corrected=_gray(), estimate=_est(),
                       n_iter=10, converged=True)
        assert r.n_iter == 10

    def test_converged_false(self):
        r = WarpResult(corrected=_gray(), estimate=_est(),
                       n_iter=10, converged=False)
        assert r.converged is False

    def test_output_shape_rgb(self):
        r = WarpResult(corrected=_rgb(32, 64), estimate=_est(),
                       n_iter=1, converged=True)
        # output_shape returns the full corrected array shape
        assert r.output_shape[:2] == (32, 64)

    def test_was_modified_is_bool(self):
        r = WarpResult(corrected=_gray(), estimate=_est(),
                       n_iter=1, converged=True)
        assert isinstance(r.was_modified, bool)

    def test_estimate_stored(self):
        est = _est(skew=10.0)
        r = WarpResult(corrected=_gray(), estimate=est,
                       n_iter=1, converged=True)
        assert r.estimate.skew_angle == pytest.approx(10.0)


# ─── TestEstimateWarpExtra ────────────────────────────────────────────────────

class TestEstimateWarpExtra:
    def test_five_seeds_no_crash(self):
        for s in range(5):
            e = estimate_warp(_gray(seed=s))
            assert e.matrix.shape == (2, 3)

    def test_noisy_image(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        e = estimate_warp(img)
        assert isinstance(e, WarpEstimate)

    def test_non_square_various(self):
        for h, w in [(32, 64), (64, 32), (48, 96)]:
            e = estimate_warp(_gray(h, w))
            assert e.matrix.shape == (2, 3)

    def test_scale_x_not_zero(self):
        for s in range(3):
            e = estimate_warp(_gray(seed=s))
            assert e.scale_x != 0.0

    def test_rgb_returns_valid(self):
        e = estimate_warp(_rgb(seed=1))
        assert 0.0 <= e.confidence <= 1.0


# ─── TestApplyWarpExtra ───────────────────────────────────────────────────────

class TestApplyWarpExtra:
    def test_various_output_sizes(self):
        for w, h in [(16, 16), (32, 48), (64, 32)]:
            cfg = WarpConfig(output_size=(w, h))
            result = apply_warp(_gray(), _est(), cfg)
            assert result.shape == (h, w)

    def test_rgb_preserves_channels(self):
        result = apply_warp(_rgb(), _est())
        assert result.shape[2] == 3

    def test_dtype_uint8(self):
        result = apply_warp(_gray(), _est())
        assert result.dtype == np.uint8

    def test_various_seeds_no_crash(self):
        for s in range(5):
            result = apply_warp(_gray(seed=s), _est())
            assert isinstance(result, np.ndarray)

    def test_border_mode_replicate(self):
        cfg = WarpConfig(border_mode="replicate", output_size=(64, 64))
        result = apply_warp(_gray(), _est(), cfg)
        assert result.shape == (64, 64)


# ─── TestCorrectWarpExtra ─────────────────────────────────────────────────────

class TestCorrectWarpExtra:
    def test_five_seeds(self):
        for s in range(5):
            r = correct_warp(_gray(seed=s))
            assert isinstance(r, WarpResult)

    def test_constant_image_ok(self):
        img = np.full((64, 64), 128, dtype=np.uint8)
        r = correct_warp(img)
        assert isinstance(r, WarpResult)

    def test_non_square(self):
        r = correct_warp(_gray(h=32, w=64))
        assert isinstance(r, WarpResult)

    def test_corrected_dtype_uint8(self):
        r = correct_warp(_gray())
        assert r.corrected.dtype == np.uint8

    def test_config_output_size_applied(self):
        cfg = WarpConfig(output_size=(48, 48))
        r = correct_warp(_gray(), cfg)
        assert r.corrected.shape[0] == 48
        assert r.corrected.shape[1] == 48


# ─── TestWarpScoreExtra ───────────────────────────────────────────────────────

class TestWarpScoreExtra:
    def test_confidence_0_5(self):
        s = warp_score(_est(confidence=0.5))
        assert 0.0 <= s <= 1.0

    def test_very_small_skew(self):
        s = warp_score(_est(skew=0.1))
        assert 0.0 <= s <= 1.0

    def test_zero_skew_high_score(self):
        e = WarpEstimate(matrix=_identity(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=1.0)
        s = warp_score(e)
        assert s >= 0.9

    def test_five_estimates_in_range(self):
        for skew in (0.0, 10.0, 20.0, 30.0, 45.0):
            s = warp_score(_est(skew=skew))
            assert 0.0 <= s <= 1.0

    def test_very_large_skew(self):
        s = warp_score(_est(skew=89.0))
        assert s >= 0.0


# ─── TestBatchCorrectWarpExtra ────────────────────────────────────────────────

class TestBatchCorrectWarpExtra:
    def test_five_images(self):
        imgs = [_gray(seed=i) for i in range(5)]
        results = batch_correct_warp(imgs)
        assert len(results) == 5

    def test_shapes_match_config(self):
        cfg = WarpConfig(output_size=(32, 32))
        imgs = [_gray(seed=i) for i in range(3)]
        for r in batch_correct_warp(imgs, cfg):
            assert r.corrected.shape == (32, 32)

    def test_rgb_batch(self):
        imgs = [_rgb(seed=i) for i in range(3)]
        results = batch_correct_warp(imgs)
        for r in results:
            assert isinstance(r, WarpResult)

    def test_mixed_shapes_ok(self):
        imgs = [_gray(32, 32), _gray(64, 64), _gray(48, 48)]
        results = batch_correct_warp(imgs)
        assert len(results) == 3

    def test_single_image(self):
        results = batch_correct_warp([_gray()])
        assert len(results) == 1
