"""Extra tests for puzzle_reconstruction/utils/curve_metrics.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.curve_metrics import (
    CurveMetricConfig,
    CurveComparisonResult,
    curve_l2,
    curve_l2_mirror,
    hausdorff_distance,
    frechet_distance_approx,
    curve_length,
    length_ratio,
    compare_curves,
    batch_compare_curves,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=8, r=1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _line(n=8) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.zeros(n)])


# ─── CurveMetricConfig ────────────────────────────────────────────────────────

class TestCurveMetricConfigExtra:
    def test_default_n_samples(self):
        assert CurveMetricConfig().n_samples == 64

    def test_default_eps(self):
        assert CurveMetricConfig().eps > 0.0

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            CurveMetricConfig(n_samples=1)

    def test_eps_zero_raises(self):
        with pytest.raises(ValueError):
            CurveMetricConfig(eps=0.0)

    def test_eps_negative_raises(self):
        with pytest.raises(ValueError):
            CurveMetricConfig(eps=-1e-6)

    def test_custom_values(self):
        cfg = CurveMetricConfig(n_samples=32, eps=1e-8)
        assert cfg.n_samples == 32


# ─── curve_l2 ─────────────────────────────────────────────────────────────────

class TestCurveL2Extra:
    def test_returns_float(self):
        s = _circle()
        assert isinstance(curve_l2(s, s), float)

    def test_identical_is_zero(self):
        s = _circle()
        assert curve_l2(s, s) == pytest.approx(0.0, abs=1e-10)

    def test_nonneg(self):
        assert curve_l2(_circle(), _line()) >= 0.0

    def test_symmetric(self):
        a, b = _circle(), _line()
        assert abs(curve_l2(a, b) - curve_l2(b, a)) < 1e-10

    def test_none_cfg_uses_default(self):
        s = _circle()
        r = curve_l2(s, s, cfg=None)
        assert r == pytest.approx(0.0, abs=1e-10)


# ─── curve_l2_mirror ──────────────────────────────────────────────────────────

class TestCurveL2MirrorExtra:
    def test_returns_float(self):
        s = _circle()
        assert isinstance(curve_l2_mirror(s, s), float)

    def test_identical_is_zero(self):
        s = _circle()
        assert curve_l2_mirror(s, s) == pytest.approx(0.0, abs=1e-10)

    def test_le_curve_l2(self):
        a, b = _circle(), _line()
        assert curve_l2_mirror(a, b) <= curve_l2(a, b) + 1e-10


# ─── hausdorff_distance ───────────────────────────────────────────────────────

class TestHausdorffDistanceExtra:
    def test_returns_float(self):
        s = _circle()
        assert isinstance(hausdorff_distance(s, s), float)

    def test_identical_is_zero(self):
        s = _circle()
        assert hausdorff_distance(s, s) == pytest.approx(0.0, abs=1e-10)

    def test_nonneg(self):
        assert hausdorff_distance(_circle(), _line()) >= 0.0

    def test_symmetric(self):
        a, b = _circle(), _line()
        assert abs(hausdorff_distance(a, b) - hausdorff_distance(b, a)) < 1e-9


# ─── frechet_distance_approx ──────────────────────────────────────────────────

class TestFrechetDistanceApproxExtra:
    def test_returns_float(self):
        s = _circle()
        assert isinstance(frechet_distance_approx(s, s), float)

    def test_identical_is_zero(self):
        s = _circle()
        assert frechet_distance_approx(s, s) == pytest.approx(0.0, abs=1e-8)

    def test_nonneg(self):
        assert frechet_distance_approx(_circle(), _line()) >= 0.0


# ─── curve_length ─────────────────────────────────────────────────────────────

class TestCurveLengthExtra:
    def test_returns_float(self):
        assert isinstance(curve_length(_line()), float)

    def test_line_length(self):
        assert curve_length(_line(8)) == pytest.approx(1.0, abs=0.01)

    def test_nonneg(self):
        assert curve_length(_circle()) >= 0.0

    def test_single_point(self):
        pts = np.array([[0.0, 0.0]])
        assert curve_length(pts) == pytest.approx(0.0)


# ─── length_ratio ─────────────────────────────────────────────────────────────

class TestLengthRatioExtra:
    def test_returns_float(self):
        s = _circle()
        assert isinstance(length_ratio(s, s), float)

    def test_identical_is_one(self):
        s = _circle()
        assert length_ratio(s, s) == pytest.approx(1.0, abs=0.01)

    def test_in_range(self):
        r = length_ratio(_circle(), _line())
        assert r >= 0.0


# ─── CurveComparisonResult ────────────────────────────────────────────────────

class TestCurveComparisonResultExtra:
    def _make(self, l2=0.1, hd=0.2, fr=0.15, lr=1.0) -> CurveComparisonResult:
        return CurveComparisonResult(l2=l2, hausdorff=hd, frechet=fr,
                                     length_ratio=lr)

    def test_stores_l2(self):
        assert self._make(l2=0.5).l2 == pytest.approx(0.5)

    def test_stores_hausdorff(self):
        assert self._make(hd=0.3).hausdorff == pytest.approx(0.3)

    def test_stores_frechet(self):
        assert self._make(fr=0.2).frechet == pytest.approx(0.2)

    def test_stores_length_ratio(self):
        assert self._make(lr=0.8).length_ratio == pytest.approx(0.8)

    def test_similarity_in_range(self):
        r = self._make(l2=0.0, hd=0.0, lr=1.0)
        s = r.similarity(sigma=1.0)
        assert 0.0 <= s <= 1.0

    def test_similarity_zero_error_is_lr(self):
        r = self._make(l2=0.0, hd=0.0, lr=0.8)
        assert r.similarity(sigma=1.0) == pytest.approx(0.8)

    def test_similarity_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            self._make().similarity(sigma=0.0)

    def test_to_dict(self):
        d = self._make().to_dict()
        for k in ("l2", "hausdorff", "frechet", "length_ratio"):
            assert k in d


# ─── compare_curves ───────────────────────────────────────────────────────────

class TestCompareCurvesExtra:
    def test_returns_result(self):
        s = _circle()
        assert isinstance(compare_curves(s, s), CurveComparisonResult)

    def test_identical_l2_near_zero(self):
        s = _circle()
        r = compare_curves(s, s)
        assert r.l2 == pytest.approx(0.0, abs=1e-9)

    def test_none_cfg(self):
        s = _circle()
        r = compare_curves(s, s, cfg=None)
        assert isinstance(r, CurveComparisonResult)


# ─── batch_compare_curves ─────────────────────────────────────────────────────

class TestBatchCompareCurvesExtra:
    def test_returns_list(self):
        s = _circle()
        result = batch_compare_curves([(s, s)])
        assert isinstance(result, list)

    def test_length_matches(self):
        s = _circle()
        result = batch_compare_curves([(s, s), (s, _line())])
        assert len(result) == 2

    def test_each_is_result(self):
        s = _circle()
        for r in batch_compare_curves([(s, s)]):
            assert isinstance(r, CurveComparisonResult)

    def test_empty_returns_empty(self):
        assert batch_compare_curves([]) == []
