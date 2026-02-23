"""Extra tests for puzzle_reconstruction.preprocessing.warp_corrector (iter-179)."""
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

def _image(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)

def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def _est(skew=0.0, sx=1.0, sy=1.0, confidence=1.0):
    import math
    c, s = math.cos(math.radians(skew)), math.sin(math.radians(skew))
    m = np.array([[c * sx, -s * sy, 0.0], [s * sx, c * sy, 0.0]])
    return WarpEstimate(matrix=m, skew_angle=skew, scale_x=sx, scale_y=sy,
                        translation=(0.0, 0.0), confidence=confidence)


# ─── TestWarpConfigExtra2 ─────────────────────────────────────────────────────

class TestWarpConfigExtra2:
    def test_max_iter_1_valid(self):
        cfg = WarpConfig(max_iter=1)
        assert cfg.max_iter == 1

    def test_max_iter_50(self):
        cfg = WarpConfig(max_iter=50)
        assert cfg.max_iter == 50

    def test_large_output_size(self):
        cfg = WarpConfig(output_size=(512, 512))
        assert cfg.output_size == (512, 512)

    def test_very_small_eps(self):
        cfg = WarpConfig(convergence_eps=1e-9)
        assert cfg.convergence_eps == pytest.approx(1e-9)

    def test_negative_output_w_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(-1, 64))

    def test_negative_output_h_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(64, -1))

    def test_border_mode_stored(self):
        cfg = WarpConfig(border_mode="zero")
        assert cfg.border_mode == "zero"


# ─── TestWarpEstimateExtra2 ───────────────────────────────────────────────────

class TestWarpEstimateExtra2:
    def test_small_scale_x_valid(self):
        e = _est(sx=0.1)
        assert e.scale_x == pytest.approx(0.1)

    def test_small_scale_y_valid(self):
        e = _est(sy=0.1)
        assert e.scale_y == pytest.approx(0.1)

    def test_negative_skew_valid(self):
        e = _est(skew=-15.0)
        assert e.skew_angle == pytest.approx(-15.0)

    def test_confidence_0_valid(self):
        e = WarpEstimate(matrix=_identity(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=0.0)
        assert e.confidence == pytest.approx(0.0)

    def test_translation_negative(self):
        m = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -3.0]])
        e = WarpEstimate(matrix=m, skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                         translation=(-5.0, -3.0))
        assert e.translation == (-5.0, -3.0)

    def test_matrix_shape_2x3(self):
        e = _est()
        assert e.matrix.shape == (2, 3)

    def test_both_scale_2(self):
        m = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        e = WarpEstimate(matrix=m, skew_angle=0.0, scale_x=2.0, scale_y=2.0,
                         translation=(0.0, 0.0))
        assert e.scale_x == pytest.approx(2.0)
        assert e.scale_y == pytest.approx(2.0)


# ─── TestWarpResultExtra2 ─────────────────────────────────────────────────────

class TestWarpResultExtra2:
    def test_n_iter_0_valid(self):
        r = WarpResult(corrected=np.zeros((4, 4), dtype=np.uint8),
                       estimate=_est(), n_iter=0, converged=True)
        assert r.n_iter == 0

    def test_corrected_float32(self):
        img = np.zeros((8, 8), dtype=np.float32)
        r = WarpResult(corrected=img, estimate=_est(), n_iter=1, converged=True)
        assert r.corrected.dtype == np.float32

    def test_estimate_is_warp_estimate(self):
        r = WarpResult(corrected=np.zeros((8, 8), dtype=np.uint8),
                       estimate=_est(), n_iter=1, converged=True)
        assert isinstance(r.estimate, WarpEstimate)

    def test_output_shape_rgb(self):
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        r = WarpResult(corrected=img, estimate=_est(), n_iter=1, converged=True)
        assert r.output_shape[:2] == (24, 32)

    def test_output_shape_2d(self):
        img = np.zeros((16, 24), dtype=np.uint8)
        r = WarpResult(corrected=img, estimate=_est(), n_iter=1, converged=True)
        assert r.output_shape == (16, 24)


# ─── TestEstimateWarpExtra2 ───────────────────────────────────────────────────

class TestEstimateWarpExtra2:
    def test_128x128_image(self):
        est = estimate_warp(_image(128, 128))
        assert est.matrix.shape == (2, 3)

    def test_8x8_image(self):
        est = estimate_warp(_image(8, 8))
        assert isinstance(est, WarpEstimate)

    def test_all_zeros_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        est = estimate_warp(img)
        assert est.matrix.shape == (2, 3)

    def test_all_max_image(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        est = estimate_warp(img)
        assert est.matrix.shape == (2, 3)

    def test_float64_input(self):
        img = _image(32, 32).astype(np.float64)
        est = estimate_warp(img)
        assert isinstance(est, WarpEstimate)

    def test_confidence_returns_float(self):
        est = estimate_warp(_image(32, 32, seed=99))
        assert isinstance(est.confidence, float)


# ─── TestApplyWarpExtra2 ─────────────────────────────────────────────────────

class TestApplyWarpExtra2:
    def test_identity_small_image(self):
        img = _image(8, 8)
        cfg = WarpConfig(output_size=(8, 8))
        out = apply_warp(img, _est(), cfg)
        assert out.shape == (8, 8)

    def test_float64_input_ok(self):
        img = _image(16, 16).astype(np.float64)
        out = apply_warp(img, _est(), WarpConfig(output_size=(16, 16)))
        assert isinstance(out, np.ndarray)

    def test_large_output(self):
        img = _image(32, 32)
        cfg = WarpConfig(output_size=(256, 256))
        out = apply_warp(img, _est(), cfg)
        assert out.shape == (256, 256)

    def test_default_config_output(self):
        img = _image(32, 32)
        out = apply_warp(img, _est())
        assert out.shape == (128, 128)

    def test_non_square_output(self):
        img = _image(32, 32)
        cfg = WarpConfig(output_size=(48, 32))
        out = apply_warp(img, _est(), cfg)
        assert out.shape == (32, 48)


# ─── TestCorrectWarpExtra2 ───────────────────────────────────────────────────

class TestCorrectWarpExtra2:
    def test_all_white_image(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        r = correct_warp(img)
        assert isinstance(r, WarpResult)

    def test_all_black_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        r = correct_warp(img)
        assert isinstance(r, WarpResult)

    def test_custom_config_max_iter(self):
        img = _image(32, 32)
        cfg = WarpConfig(output_size=(32, 32), max_iter=5)
        r = correct_warp(img, cfg)
        assert r.n_iter <= 5

    def test_estimate_scale_positive(self):
        img = _image(32, 32)
        r = correct_warp(img)
        assert r.estimate.scale_x > 0.0
        assert r.estimate.scale_y > 0.0

    def test_different_seeds_all_ok(self):
        for s in range(5):
            r = correct_warp(_image(32, 32, seed=s))
            assert isinstance(r, WarpResult)


# ─── TestWarpScoreExtra2 ─────────────────────────────────────────────────────

class TestWarpScoreExtra2:
    def test_various_skews_monotone_decreasing(self):
        scores = [warp_score(_est(skew=float(a))) for a in (0, 10, 20, 30)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9

    def test_various_confidences_monotone(self):
        scores = [warp_score(_est(confidence=c)) for c in (0.2, 0.5, 0.8, 1.0)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-9

    def test_all_return_float(self):
        for s in range(5):
            r = correct_warp(_image(32, 32, seed=s))
            assert isinstance(warp_score(r.estimate), float)

    def test_identity_is_max(self):
        identity_score = warp_score(_est(skew=0.0, confidence=1.0))
        skewed_score = warp_score(_est(skew=30.0, confidence=0.5))
        assert identity_score >= skewed_score


# ─── TestBatchCorrectWarpExtra2 ───────────────────────────────────────────────

class TestBatchCorrectWarpExtra2:
    def test_10_images(self):
        imgs = [_image(32, 32, seed=i) for i in range(10)]
        cfg = WarpConfig(output_size=(32, 32))
        results = batch_correct_warp(imgs, cfg)
        assert len(results) == 10

    def test_all_white_batch(self):
        imgs = [np.full((32, 32), 255, dtype=np.uint8)] * 3
        cfg = WarpConfig(output_size=(32, 32))
        results = batch_correct_warp(imgs, cfg)
        assert len(results) == 3

    def test_various_sizes_same_output(self):
        imgs = [_image(16, 16), _image(32, 32), _image(64, 64)]
        cfg = WarpConfig(output_size=(32, 32))
        results = batch_correct_warp(imgs, cfg)
        for r in results:
            assert r.corrected.shape[:2] == (32, 32)

    def test_single_image_ok(self):
        results = batch_correct_warp([_image(16, 16)],
                                      WarpConfig(output_size=(16, 16)))
        assert len(results) == 1
        assert isinstance(results[0], WarpResult)
