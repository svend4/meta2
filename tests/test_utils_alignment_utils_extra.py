"""Extra tests for puzzle_reconstruction/utils/alignment_utils.py."""
from __future__ import annotations

import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.alignment_utils import (
    AlignmentConfig,
    AlignmentResult,
    normalize_for_alignment,
    find_best_rotation,
    find_best_translation,
    compute_alignment_error,
    align_curves_procrustes,
    align_curves_icp,
    alignment_score,
    batch_align_curves,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(n=8) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def _line(n=8) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.zeros(n)])


# ─── AlignmentConfig ──────────────────────────────────────────────────────────

class TestAlignmentConfigExtra:
    def test_default_n_samples(self):
        assert AlignmentConfig().n_samples == 64

    def test_default_max_icp_iter(self):
        assert AlignmentConfig().max_icp_iter == 50

    def test_default_icp_tol(self):
        assert AlignmentConfig().icp_tol == pytest.approx(1e-6)

    def test_default_allow_reflection(self):
        assert AlignmentConfig().allow_reflection is False

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            AlignmentConfig(n_samples=1)

    def test_max_icp_iter_lt_1_raises(self):
        with pytest.raises(ValueError):
            AlignmentConfig(max_icp_iter=0)

    def test_icp_tol_zero_raises(self):
        with pytest.raises(ValueError):
            AlignmentConfig(icp_tol=0.0)

    def test_icp_tol_negative_raises(self):
        with pytest.raises(ValueError):
            AlignmentConfig(icp_tol=-1e-6)

    def test_custom_values(self):
        cfg = AlignmentConfig(n_samples=32, max_icp_iter=10)
        assert cfg.n_samples == 32 and cfg.max_icp_iter == 10


# ─── AlignmentResult ──────────────────────────────────────────────────────────

class TestAlignmentResultExtra:
    def _make(self, error=0.01) -> AlignmentResult:
        return AlignmentResult(
            rotation=0.0,
            translation=np.zeros(2),
            scale=1.0,
            error=error,
            aligned=_square(),
        )

    def test_stores_rotation(self):
        r = self._make()
        assert r.rotation == pytest.approx(0.0)

    def test_stores_scale(self):
        r = self._make()
        assert r.scale == pytest.approx(1.0)

    def test_stores_error(self):
        r = self._make(error=0.05)
        assert r.error == pytest.approx(0.05)

    def test_default_converged(self):
        r = self._make()
        assert r.converged is True

    def test_to_dict_returns_dict(self):
        r = self._make()
        assert isinstance(r.to_dict(), dict)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for k in ("rotation", "translation", "scale", "error", "converged"):
            assert k in d


# ─── normalize_for_alignment ──────────────────────────────────────────────────

class TestNormalizeForAlignmentExtra:
    def test_returns_tuple_3(self):
        result = normalize_for_alignment(_square())
        assert isinstance(result, tuple) and len(result) == 3

    def test_normalized_mean_near_zero(self):
        normalized, _, _ = normalize_for_alignment(_square())
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-10)

    def test_scale_positive(self):
        _, _, scale = normalize_for_alignment(_square())
        assert scale > 0.0

    def test_centroid_shape(self):
        _, centroid, _ = normalize_for_alignment(_square())
        assert centroid.shape == (2,)

    def test_single_point(self):
        pts = np.array([[3.0, 4.0]])
        norm, centroid, scale = normalize_for_alignment(pts)
        np.testing.assert_allclose(centroid, [3.0, 4.0], atol=1e-10)


# ─── find_best_rotation ───────────────────────────────────────────────────────

class TestFindBestRotationExtra:
    def test_returns_tuple(self):
        s = _square()
        result = find_best_rotation(s, s)
        assert isinstance(result, tuple) and len(result) == 2

    def test_identical_points_zero_rotation(self):
        s = _square()
        angle, R = find_best_rotation(s, s)
        assert abs(angle) < 1e-6

    def test_rotation_matrix_shape(self):
        s = _square()
        _, R = find_best_rotation(s, s)
        assert R.shape == (2, 2)

    def test_rotation_matrix_orthogonal(self):
        s = _square()
        _, R = find_best_rotation(s, s)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)

    def test_known_90_degree_rotation(self):
        s = np.array([[1.0, 0.0], [0.0, 1.0]])
        t = np.array([[0.0, 1.0], [-1.0, 0.0]])  # 90° CCW rotation
        angle, _ = find_best_rotation(s, t)
        assert abs(abs(angle) - math.pi / 2) < 0.05


# ─── find_best_translation ────────────────────────────────────────────────────

class TestFindBestTranslationExtra:
    def test_returns_ndarray(self):
        s = _square()
        assert isinstance(find_best_translation(s, s), np.ndarray)

    def test_shape_2(self):
        assert find_best_translation(_square(), _square()).shape == (2,)

    def test_identical_zero_translation(self):
        s = _square()
        t = find_best_translation(s, s)
        np.testing.assert_allclose(t, 0.0, atol=1e-10)

    def test_known_translation(self):
        s = _square()
        target = s + np.array([3.0, -2.0])
        t = find_best_translation(s, target)
        np.testing.assert_allclose(t, [3.0, -2.0], atol=1e-9)


# ─── compute_alignment_error ──────────────────────────────────────────────────

class TestComputeAlignmentErrorExtra:
    def test_returns_float(self):
        assert isinstance(compute_alignment_error(_square(), _square()), float)

    def test_identical_is_zero(self):
        s = _square()
        assert compute_alignment_error(s, s) == pytest.approx(0.0)

    def test_nonneg(self):
        assert compute_alignment_error(_square(), _line()) >= 0.0

    def test_shape_mismatch_returns_inf(self):
        s = _square(4)
        t = _square(8)
        assert compute_alignment_error(s, t) == float("inf")

    def test_empty_returns_inf(self):
        result = compute_alignment_error(np.zeros((0, 2)), np.zeros((0, 2)))
        assert result == float("inf")

    def test_translation_gives_constant_error(self):
        s = _square()
        t = s + np.array([1.0, 0.0])
        err = compute_alignment_error(s, t)
        assert err == pytest.approx(1.0)


# ─── align_curves_procrustes ──────────────────────────────────────────────────

class TestAlignCurvesProcrustesExtra:
    def test_returns_alignment_result(self):
        s = _square()
        r = align_curves_procrustes(s, s)
        assert isinstance(r, AlignmentResult)

    def test_identical_error_near_zero(self):
        s = _square()
        r = align_curves_procrustes(s, s)
        assert r.error < 1e-6

    def test_aligned_shape(self):
        s = _square()
        r = align_curves_procrustes(s, s)
        # Procrustes may resample to n_samples; check 2 columns
        assert r.aligned.ndim == 2 and r.aligned.shape[1] == 2

    def test_scale_positive(self):
        s = _square()
        r = align_curves_procrustes(s, s)
        assert r.scale > 0.0

    def test_none_cfg(self):
        s = _square()
        r = align_curves_procrustes(s, s, cfg=None)
        assert isinstance(r, AlignmentResult)

    def test_translated_source(self):
        s = _square()
        t = s + np.array([5.0, 3.0])
        r = align_curves_procrustes(s, t)
        assert r.error < 1.0


# ─── align_curves_icp ─────────────────────────────────────────────────────────

class TestAlignCurvesIcpExtra:
    def test_returns_alignment_result(self):
        s = _square()
        r = align_curves_icp(s, s)
        assert isinstance(r, AlignmentResult)

    def test_identical_error_near_zero(self):
        s = _square()
        r = align_curves_icp(s, s)
        assert r.error < 1e-4

    def test_scale_is_one(self):
        s = _square()
        r = align_curves_icp(s, s)
        assert r.scale == pytest.approx(1.0)

    def test_aligned_shape(self):
        s = _square()
        r = align_curves_icp(s, s)
        assert r.aligned.ndim == 2 and r.aligned.shape[1] == 2

    def test_none_cfg(self):
        s = _square()
        r = align_curves_icp(s, s, cfg=None)
        assert isinstance(r, AlignmentResult)


# ─── alignment_score ──────────────────────────────────────────────────────────

class TestAlignmentScoreExtra:
    def _result(self, error=0.0) -> AlignmentResult:
        return AlignmentResult(
            rotation=0.0, translation=np.zeros(2), scale=1.0,
            error=error, aligned=_square()
        )

    def test_returns_float(self):
        assert isinstance(alignment_score(self._result(0.0)), float)

    def test_zero_error_is_one(self):
        assert alignment_score(self._result(0.0)) == pytest.approx(1.0)

    def test_in_range(self):
        s = alignment_score(self._result(1.0))
        assert 0.0 < s <= 1.0

    def test_higher_error_lower_score(self):
        s_low = alignment_score(self._result(0.1))
        s_high = alignment_score(self._result(10.0))
        assert s_high < s_low

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            alignment_score(self._result(), sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            alignment_score(self._result(), sigma=-1.0)


# ─── batch_align_curves ───────────────────────────────────────────────────────

class TestBatchAlignCurvesExtra:
    def test_returns_list(self):
        s = _square()
        result = batch_align_curves([s], [s])
        assert isinstance(result, list)

    def test_length_matches(self):
        s = _square()
        result = batch_align_curves([s, s], [s, s])
        assert len(result) == 2

    def test_each_alignment_result(self):
        s = _square()
        for r in batch_align_curves([s], [s]):
            assert isinstance(r, AlignmentResult)

    def test_length_mismatch_raises(self):
        s = _square()
        with pytest.raises(ValueError):
            batch_align_curves([s], [s, s])

    def test_invalid_method_raises(self):
        s = _square()
        with pytest.raises(ValueError):
            batch_align_curves([s], [s], method="umeyama")

    def test_icp_method(self):
        s = _square()
        result = batch_align_curves([s], [s], method="icp")
        assert len(result) == 1
