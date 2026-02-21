"""Тесты для puzzle_reconstruction.preprocessing.warp_corrector."""
import numpy as np
import pytest
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

def _image(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)


def _identity_estimate() -> WarpEstimate:
    return WarpEstimate(
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        skew_angle=0.0,
        scale_x=1.0,
        scale_y=1.0,
        translation=(0.0, 0.0),
        confidence=1.0,
    )


# ─── TestWarpConfig ───────────────────────────────────────────────────────────

class TestWarpConfig:
    def test_defaults(self):
        cfg = WarpConfig()
        assert cfg.output_size == (128, 128)
        assert cfg.border_mode == "zero"
        assert cfg.max_iter == 10
        assert cfg.convergence_eps == 1e-4

    def test_valid_custom(self):
        cfg = WarpConfig(output_size=(64, 32), border_mode="replicate",
                         max_iter=5, convergence_eps=1e-3)
        assert cfg.output_size == (64, 32)
        assert cfg.border_mode == "replicate"
        assert cfg.max_iter == 5

    def test_invalid_output_size_zero_w(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(0, 64))

    def test_invalid_output_size_zero_h(self):
        with pytest.raises(ValueError):
            WarpConfig(output_size=(64, 0))

    def test_invalid_border_mode(self):
        with pytest.raises(ValueError):
            WarpConfig(border_mode="mirror")

    def test_invalid_max_iter_zero(self):
        with pytest.raises(ValueError):
            WarpConfig(max_iter=0)

    def test_invalid_convergence_eps_neg(self):
        with pytest.raises(ValueError):
            WarpConfig(convergence_eps=-1e-5)

    def test_convergence_eps_zero_ok(self):
        cfg = WarpConfig(convergence_eps=0.0)
        assert cfg.convergence_eps == 0.0


# ─── TestWarpEstimate ─────────────────────────────────────────────────────────

class TestWarpEstimate:
    def test_basic(self):
        e = _identity_estimate()
        assert e.skew_angle == 0.0
        assert e.scale_x == 1.0
        assert e.scale_y == 1.0
        assert e.confidence == 1.0

    def test_is_identity_true(self):
        e = _identity_estimate()
        assert e.is_identity is True

    def test_is_identity_false(self):
        e = WarpEstimate(
            matrix=np.array([[1.1, 0.0, 5.0], [0.0, 0.9, -3.0]]),
            skew_angle=5.0,
            scale_x=1.1,
            scale_y=0.9,
            translation=(5.0, -3.0),
        )
        assert e.is_identity is False

    def test_rotation_deg_identity(self):
        e = _identity_estimate()
        assert abs(e.rotation_deg) < 1e-6

    def test_rotation_deg_90(self):
        theta = np.radians(90.0)
        m = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
        ])
        e = WarpEstimate(matrix=m, skew_angle=90.0,
                         scale_x=1.0, scale_y=1.0, translation=(0.0, 0.0))
        assert abs(e.rotation_deg - 90.0) < 1e-4

    def test_invalid_matrix_shape(self):
        with pytest.raises(ValueError):
            WarpEstimate(
                matrix=np.eye(3),
                skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                translation=(0.0, 0.0),
            )

    def test_invalid_scale_x_zero(self):
        with pytest.raises(ValueError):
            WarpEstimate(
                matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                skew_angle=0.0, scale_x=0.0, scale_y=1.0,
                translation=(0.0, 0.0),
            )

    def test_invalid_scale_y_neg(self):
        with pytest.raises(ValueError):
            WarpEstimate(
                matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                skew_angle=0.0, scale_x=1.0, scale_y=-0.5,
                translation=(0.0, 0.0),
            )

    def test_invalid_confidence_above(self):
        with pytest.raises(ValueError):
            WarpEstimate(
                matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                translation=(0.0, 0.0), confidence=1.5,
            )

    def test_invalid_confidence_below(self):
        with pytest.raises(ValueError):
            WarpEstimate(
                matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                skew_angle=0.0, scale_x=1.0, scale_y=1.0,
                translation=(0.0, 0.0), confidence=-0.1,
            )


# ─── TestWarpResult ───────────────────────────────────────────────────────────

class TestWarpResult:
    def _make(self, shape=(32, 32), identity=True):
        img = np.zeros(shape, dtype=np.uint8)
        est = _identity_estimate()
        if not identity:
            est = WarpEstimate(
                matrix=np.array([[1.1, 0.0, 0.0], [0.0, 0.9, 0.0]]),
                skew_angle=5.0, scale_x=1.1, scale_y=0.9,
                translation=(0.0, 0.0),
            )
        return WarpResult(corrected=img, estimate=est, n_iter=1, converged=True)

    def test_output_shape_2d(self):
        r = self._make((16, 32))
        assert r.output_shape == (16, 32)

    def test_output_shape_3d(self):
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        r = WarpResult(corrected=img, estimate=_identity_estimate(),
                       n_iter=1, converged=True)
        assert r.output_shape == (16, 32, 3)

    def test_was_modified_false(self):
        r = self._make(identity=True)
        assert r.was_modified is False

    def test_was_modified_true(self):
        r = self._make(identity=False)
        assert r.was_modified is True

    def test_converged(self):
        r = self._make()
        assert r.converged is True

    def test_n_iter_valid(self):
        r = self._make()
        assert r.n_iter >= 0

    def test_invalid_n_iter_neg(self):
        with pytest.raises(ValueError):
            WarpResult(corrected=np.zeros((4, 4)), estimate=_identity_estimate(),
                       n_iter=-1, converged=True)


# ─── TestEstimateWarp ─────────────────────────────────────────────────────────

class TestEstimateWarp:
    def test_basic_2d(self):
        img = _image(32, 32)
        est = estimate_warp(img)
        assert isinstance(est, WarpEstimate)

    def test_basic_3d(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        est = estimate_warp(img)
        assert est.matrix.shape == (2, 3)

    def test_confidence_in_range(self):
        img = _image(32, 32, seed=2)
        est = estimate_warp(img)
        assert 0.0 <= est.confidence <= 1.0

    def test_scale_positive(self):
        img = _image(32, 32, seed=3)
        est = estimate_warp(img)
        assert est.scale_x > 0
        assert est.scale_y > 0

    def test_uniform_image_near_identity(self):
        img = np.ones((32, 32), dtype=np.uint8) * 128
        est = estimate_warp(img)
        assert est.matrix.shape == (2, 3)

    def test_invalid_image_1d(self):
        with pytest.raises(ValueError):
            estimate_warp(np.zeros(32))

    def test_default_config(self):
        img = _image(16, 16)
        est = estimate_warp(img)
        assert est is not None

    def test_custom_config(self):
        img = _image(32, 32)
        cfg = WarpConfig(output_size=(64, 64))
        est = estimate_warp(img, cfg)
        assert est.matrix.shape == (2, 3)


# ─── TestApplyWarp ────────────────────────────────────────────────────────────

class TestApplyWarp:
    def test_output_size_matches_config(self):
        img = _image(32, 32)
        est = _identity_estimate()
        cfg = WarpConfig(output_size=(64, 48))
        out = apply_warp(img, est, cfg)
        assert out.shape == (48, 64)

    def test_identity_preserves_dtype(self):
        img = _image(32, 32).astype(np.float32)
        out = apply_warp(img, _identity_estimate(), WarpConfig(output_size=(32, 32)))
        assert out.dtype == np.float32

    def test_3d_input(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        out = apply_warp(img, _identity_estimate(), WarpConfig(output_size=(32, 32)))
        assert out.shape == (32, 32, 3)

    def test_border_mode_zero(self):
        img = np.ones((32, 32), dtype=np.uint8) * 100
        cfg = WarpConfig(output_size=(64, 64), border_mode="zero")
        est = _identity_estimate()
        out = apply_warp(img, est, cfg)
        assert out.shape == (64, 64)

    def test_border_mode_replicate(self):
        img = np.ones((32, 32), dtype=np.uint8) * 100
        cfg = WarpConfig(output_size=(64, 64), border_mode="replicate")
        out = apply_warp(img, _identity_estimate(), cfg)
        assert out.shape == (64, 64)


# ─── TestCorrectWarp ──────────────────────────────────────────────────────────

class TestCorrectWarp:
    def test_returns_warp_result(self):
        img = _image(32, 32)
        r = correct_warp(img)
        assert isinstance(r, WarpResult)

    def test_output_shape_matches_config(self):
        img = _image(32, 32)
        cfg = WarpConfig(output_size=(48, 48))
        r = correct_warp(img, cfg)
        assert r.corrected.shape[:2] == (48, 48)

    def test_converged_is_true(self):
        img = _image(16, 16)
        r = correct_warp(img)
        assert r.converged is True

    def test_n_iter_positive(self):
        img = _image(16, 16)
        r = correct_warp(img)
        assert r.n_iter >= 1

    def test_3d_image(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        r = correct_warp(img, WarpConfig(output_size=(32, 32)))
        assert r.corrected.shape == (32, 32, 3)

    def test_default_config(self):
        img = _image(32, 32)
        r = correct_warp(img)
        assert r.corrected.shape == (128, 128)


# ─── TestWarpScore ────────────────────────────────────────────────────────────

class TestWarpScore:
    def test_identity_gives_high_score(self):
        e = _identity_estimate()
        score = warp_score(e)
        assert score > 0.8

    def test_output_range(self):
        img = _image(32, 32, seed=7)
        e = estimate_warp(img)
        score = warp_score(e)
        assert 0.0 <= score <= 1.0

    def test_large_skew_lowers_score(self):
        e_low = WarpEstimate(
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            skew_angle=0.0, scale_x=1.0, scale_y=1.0,
            translation=(0.0, 0.0), confidence=1.0,
        )
        e_high = WarpEstimate(
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            skew_angle=44.0, scale_x=1.0, scale_y=1.0,
            translation=(0.0, 0.0), confidence=1.0,
        )
        assert warp_score(e_high) < warp_score(e_low)

    def test_low_confidence_lowers_score(self):
        e_hi = WarpEstimate(
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            skew_angle=0.0, scale_x=1.0, scale_y=1.0,
            translation=(0.0, 0.0), confidence=1.0,
        )
        e_lo = WarpEstimate(
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            skew_angle=0.0, scale_x=1.0, scale_y=1.0,
            translation=(0.0, 0.0), confidence=0.2,
        )
        assert warp_score(e_lo) < warp_score(e_hi)


# ─── TestBatchCorrectWarp ─────────────────────────────────────────────────────

class TestBatchCorrectWarp:
    def test_basic(self):
        imgs = [_image(32, 32, seed=i) for i in range(4)]
        cfg = WarpConfig(output_size=(32, 32))
        results = batch_correct_warp(imgs, cfg)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, WarpResult)

    def test_empty_list(self):
        assert batch_correct_warp([]) == []

    def test_output_shapes_consistent(self):
        imgs = [_image(16, 16, seed=i) for i in range(3)]
        cfg = WarpConfig(output_size=(24, 24))
        results = batch_correct_warp(imgs, cfg)
        for r in results:
            assert r.corrected.shape[:2] == (24, 24)

    def test_default_config(self):
        imgs = [_image(32, 32)]
        results = batch_correct_warp(imgs)
        assert results[0].corrected.shape == (128, 128)
