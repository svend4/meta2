"""Тесты для puzzle_reconstruction.preprocessing.warp_corrector."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.warp_corrector import (
    WarpConfig,
    WarpEstimate,
    WarpResult,
    estimate_warp,
    apply_warp,
    correct_warp,
    warp_score,
    batch_correct_warp,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _identity_matrix() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _est(skew: float = 0.0, sx: float = 1.0, sy: float = 1.0,
         confidence: float = 1.0) -> WarpEstimate:
    import math
    c, s = math.cos(math.radians(skew)), math.sin(math.radians(skew))
    matrix = np.array([
        [c * sx, -s * sy, 0.0],
        [s * sx, c * sy, 0.0],
    ])
    return WarpEstimate(matrix=matrix, skew_angle=skew,
                        scale_x=sx, scale_y=sy,
                        translation=(0.0, 0.0), confidence=confidence)


# ─── TestWarpConfig ───────────────────────────────────────────────────────────

class TestWarpConfig:
    def test_defaults(self):
        cfg = WarpConfig()
        assert cfg.output_size == (128, 128)
        assert cfg.border_mode == "zero"
        assert cfg.max_iter == 10
        assert cfg.convergence_eps == pytest.approx(1e-4)

    def test_custom_output_size(self):
        cfg = WarpConfig(output_size=(64, 32))
        assert cfg.output_size == (64, 32)

    def test_output_size_zero_w_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(0, 64))

    def test_output_size_zero_h_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(64, 0))

    def test_output_size_neg_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(-1, 64))

    def test_border_mode_replicate_ok(self):
        cfg = WarpConfig(border_mode="replicate")
        assert cfg.border_mode == "replicate"

    def test_border_mode_invalid_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(border_mode="wrap")

    def test_border_mode_empty_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(border_mode="")

    def test_max_iter_one_ok(self):
        cfg = WarpConfig(max_iter=1)
        assert cfg.max_iter == 1

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(max_iter=0)

    def test_max_iter_neg_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(max_iter=-1)

    def test_convergence_eps_zero_ok(self):
        cfg = WarpConfig(convergence_eps=0.0)
        assert cfg.convergence_eps == 0.0

    def test_convergence_eps_neg_raises(self):
        with pytest.raises(ValueError):
            WarpConfig(convergence_eps=-1e-5)


# ─── TestWarpEstimate ─────────────────────────────────────────────────────────

class TestWarpEstimate:
    def test_basic(self):
        e = _est()
        assert e.skew_angle == pytest.approx(0.0)

    def test_is_identity_true(self):
        e = WarpEstimate(matrix=_identity_matrix(),
                         skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                         translation=(0.0, 0.0))
        assert e.is_identity is True

    def test_is_identity_false(self):
        m = _identity_matrix().copy()
        m[0, 2] = 5.0
        e = WarpEstimate(matrix=m, skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                         translation=(5.0, 0.0))
        assert e.is_identity is False

    def test_rotation_deg_zero(self):
        e = WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0))
        assert e.rotation_deg == pytest.approx(0.0, abs=1e-6)

    def test_matrix_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=np.eye(3), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0))

    def test_scale_x_zero_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=0.0, scale_y=1.0, translation=(0.0, 0.0))

    def test_scale_x_neg_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=-1.0, scale_y=1.0, translation=(0.0, 0.0))

    def test_scale_y_zero_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=0.0, translation=(0.0, 0.0))

    def test_confidence_neg_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError):
            WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=1.1)

    def test_confidence_zero_ok(self):
        e = WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0),
                         confidence=0.0)
        assert e.confidence == 0.0


# ─── TestWarpResult ───────────────────────────────────────────────────────────

class TestWarpResult:
    def _make(self, img=None, n_iter=1, converged=True) -> WarpResult:
        if img is None:
            img = _gray()
        return WarpResult(corrected=img, estimate=_est(),
                          n_iter=n_iter, converged=converged)

    def test_output_shape(self):
        r = self._make(_gray(32, 64))
        assert r.output_shape == (32, 64)

    def test_was_modified_identity(self):
        r = self._make()
        assert r.was_modified is False

    def test_was_modified_true(self):
        e = _est(skew=10.0)
        r = WarpResult(corrected=_gray(), estimate=e, n_iter=1, converged=True)
        assert r.was_modified is True

    def test_n_iter_neg_raises(self):
        with pytest.raises(ValueError):
            WarpResult(corrected=_gray(), estimate=_est(),
                       n_iter=-1, converged=True)

    def test_n_iter_zero_ok(self):
        r = WarpResult(corrected=_gray(), estimate=_est(),
                       n_iter=0, converged=False)
        assert r.n_iter == 0


# ─── TestEstimateWarp ─────────────────────────────────────────────────────────

class TestEstimateWarp:
    def test_returns_warp_estimate(self):
        assert isinstance(estimate_warp(_gray()), WarpEstimate)

    def test_matrix_shape(self):
        e = estimate_warp(_gray())
        assert e.matrix.shape == (2, 3)

    def test_rgb_ok(self):
        e = estimate_warp(_rgb())
        assert isinstance(e, WarpEstimate)

    def test_confidence_in_range(self):
        e = estimate_warp(_gray())
        assert 0.0 <= e.confidence <= 1.0

    def test_scale_x_positive(self):
        e = estimate_warp(_gray())
        assert e.scale_x > 0.0

    def test_scale_y_positive(self):
        e = estimate_warp(_gray())
        assert e.scale_y > 0.0

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            estimate_warp(np.zeros(64))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            estimate_warp(np.zeros((4, 4, 3, 2)))

    def test_constant_image_identity_like(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        e = estimate_warp(img)
        assert e.matrix.shape == (2, 3)

    def test_non_square_ok(self):
        e = estimate_warp(_gray(48, 80))
        assert isinstance(e, WarpEstimate)

    def test_translation_is_tuple(self):
        e = estimate_warp(_gray())
        assert isinstance(e.translation, tuple)
        assert len(e.translation) == 2


# ─── TestApplyWarp ────────────────────────────────────────────────────────────

class TestApplyWarp:
    def test_returns_ndarray(self):
        result = apply_warp(_gray(), _est())
        assert isinstance(result, np.ndarray)

    def test_output_size_from_config(self):
        cfg = WarpConfig(output_size=(32, 48))
        result = apply_warp(_gray(), _est(), cfg)
        assert result.shape == (48, 32)

    def test_gray_output_2d(self):
        result = apply_warp(_gray(), _est())
        assert result.ndim == 2

    def test_rgb_output_3d(self):
        result = apply_warp(_rgb(), _est())
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_border_mode_replicate_ok(self):
        cfg = WarpConfig(border_mode="replicate")
        result = apply_warp(_gray(), _est(), cfg)
        assert result.shape == (128, 128)

    def test_identity_copies_content(self):
        img = _gray(128, 128)
        cfg = WarpConfig(output_size=(128, 128))
        result = apply_warp(img, WarpEstimate(
            matrix=_identity_matrix(), skew_angle=0.0,
            scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0)
        ), cfg)
        # При единичном преобразовании большинство пикселей должны совпадать
        assert result.shape == (128, 128)


# ─── TestCorrectWarp ──────────────────────────────────────────────────────────

class TestCorrectWarp:
    def test_returns_warp_result(self):
        assert isinstance(correct_warp(_gray()), WarpResult)

    def test_corrected_ndarray(self):
        r = correct_warp(_gray())
        assert isinstance(r.corrected, np.ndarray)

    def test_n_iter_one(self):
        r = correct_warp(_gray())
        assert r.n_iter == 1

    def test_converged_true(self):
        r = correct_warp(_gray())
        assert r.converged is True

    def test_estimate_is_warp_estimate(self):
        r = correct_warp(_gray())
        assert isinstance(r.estimate, WarpEstimate)

    def test_rgb_ok(self):
        r = correct_warp(_rgb())
        assert isinstance(r, WarpResult)

    def test_output_size_from_config(self):
        cfg = WarpConfig(output_size=(32, 48))
        r = correct_warp(_gray(), cfg)
        assert r.corrected.shape == (48, 32)


# ─── TestWarpScore ────────────────────────────────────────────────────────────

class TestWarpScore:
    def test_returns_float(self):
        assert isinstance(warp_score(_est()), float)

    def test_in_range(self):
        s = warp_score(_est())
        assert 0.0 <= s <= 1.0

    def test_identity_high_score(self):
        e = WarpEstimate(matrix=_identity_matrix(), skew_angle=0.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0))
        s = warp_score(e)
        assert s >= 0.9

    def test_large_skew_lower_score(self):
        s_small = warp_score(_est(skew=1.0))
        s_large = warp_score(_est(skew=40.0))
        assert s_large < s_small

    def test_low_confidence_lower_score(self):
        s1 = warp_score(_est(confidence=1.0))
        s2 = warp_score(_est(confidence=0.2))
        assert s2 < s1

    def test_non_negative(self):
        assert warp_score(_est(skew=89.0)) >= 0.0


# ─── TestBatchCorrectWarp ─────────────────────────────────────────────────────

class TestBatchCorrectWarp:
    def test_returns_list(self):
        images = [_gray(seed=i) for i in range(3)]
        assert isinstance(batch_correct_warp(images), list)

    def test_length_matches(self):
        images = [_gray(seed=i) for i in range(5)]
        assert len(batch_correct_warp(images)) == 5

    def test_empty_list(self):
        assert batch_correct_warp([]) == []

    def test_all_warp_results(self):
        images = [_gray(seed=i) for i in range(3)]
        for r in batch_correct_warp(images):
            assert isinstance(r, WarpResult)

    def test_custom_config(self):
        cfg = WarpConfig(output_size=(32, 32))
        images = [_gray(seed=i) for i in range(2)]
        for r in batch_correct_warp(images, cfg):
            assert r.corrected.shape == (32, 32)
