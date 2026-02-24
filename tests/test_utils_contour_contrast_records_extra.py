"""Extra tests for puzzle_reconstruction/utils/contour_contrast_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.contour_contrast_records import (
    ContourProcessRecord,
    ContrastEnhanceRecord,
    CostMatrixRecord,
    ContourBatchRecord,
    make_contour_process_record,
    make_contrast_enhance_record,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cpr(fid=0, before=100, after=80, peri=50.0, area=200.0, compact=0.8) -> ContourProcessRecord:
    return ContourProcessRecord(
        fragment_id=fid,
        n_points_before=before,
        n_points_after=after,
        perimeter=peri,
        area=area,
        compactness=compact,
    )


def _cer(method="clahe", cb=0.3, ca=0.8, h=64, w=64, nc=1) -> ContrastEnhanceRecord:
    return ContrastEnhanceRecord(
        method=method,
        contrast_before=cb,
        contrast_after=ca,
        image_height=h,
        image_width=w,
        n_channels=nc,
    )


def _cmr(n_frags=4, method="l2", mn=0.1, mx=0.9, mean=0.5) -> CostMatrixRecord:
    return CostMatrixRecord(
        n_fragments=n_frags, method=method,
        min_cost=mn, max_cost=mx, mean_cost=mean,
    )


def _cbr(n=5, npc=64, sigma=1.0, eps=0.01, normalize=True, n_succ=4) -> ContourBatchRecord:
    return ContourBatchRecord(
        n_contours=n, n_points_config=npc,
        smooth_sigma=sigma, rdp_epsilon=eps,
        normalize=normalize, n_successful=n_succ,
    )


# ─── ContourProcessRecord ─────────────────────────────────────────────────────

class TestContourProcessRecordExtra:
    def test_stores_fragment_id(self):
        assert _cpr(fid=7).fragment_id == 7

    def test_stores_n_points_before(self):
        assert _cpr(before=200).n_points_before == 200

    def test_stores_n_points_after(self):
        assert _cpr(after=150).n_points_after == 150

    def test_stores_perimeter(self):
        assert _cpr(peri=75.5).perimeter == pytest.approx(75.5)

    def test_stores_area(self):
        assert _cpr(area=400.0).area == pytest.approx(400.0)

    def test_stores_compactness(self):
        assert _cpr(compact=0.6).compactness == pytest.approx(0.6)

    def test_compression_ratio(self):
        r = _cpr(before=100, after=80)
        assert r.compression_ratio == pytest.approx(0.8)

    def test_compression_ratio_zero_before(self):
        r = _cpr(before=0, after=0)
        assert r.compression_ratio == pytest.approx(0.0)

    def test_default_normalized_true(self):
        assert _cpr().normalized is True

    def test_default_simplified_false(self):
        assert _cpr().simplified is False


# ─── ContrastEnhanceRecord ────────────────────────────────────────────────────

class TestContrastEnhanceRecordExtra:
    def test_stores_method(self):
        assert _cer(method="hist_eq").method == "hist_eq"

    def test_stores_contrast_before(self):
        assert _cer(cb=0.25).contrast_before == pytest.approx(0.25)

    def test_stores_contrast_after(self):
        assert _cer(ca=0.9).contrast_after == pytest.approx(0.9)

    def test_improvement_computed(self):
        r = _cer(cb=0.3, ca=0.8)
        assert r.improvement == pytest.approx(0.5)

    def test_improvement_ratio(self):
        r = _cer(cb=0.4, ca=0.8)
        assert r.improvement_ratio == pytest.approx(1.0)

    def test_improvement_ratio_zero_before(self):
        r = _cer(cb=0.0, ca=0.5)
        assert r.improvement_ratio == pytest.approx(0.0)

    def test_is_grayscale_true(self):
        assert _cer(nc=1).is_grayscale is True

    def test_is_grayscale_false(self):
        assert _cer(nc=3).is_grayscale is False

    def test_stores_dimensions(self):
        r = _cer(h=48, w=96)
        assert r.image_height == 48 and r.image_width == 96


# ─── CostMatrixRecord ─────────────────────────────────────────────────────────

class TestCostMatrixRecordExtra:
    def test_stores_n_fragments(self):
        assert _cmr(n_frags=8).n_fragments == 8

    def test_stores_method(self):
        assert _cmr(method="cosine").method == "cosine"

    def test_stores_min_max_mean(self):
        r = _cmr(mn=0.1, mx=0.9, mean=0.5)
        assert r.min_cost == pytest.approx(0.1)
        assert r.max_cost == pytest.approx(0.9)
        assert r.mean_cost == pytest.approx(0.5)

    def test_cost_range(self):
        r = _cmr(mn=0.2, mx=0.8)
        assert r.cost_range == pytest.approx(0.6)

    def test_default_n_forbidden_zero(self):
        assert _cmr().n_forbidden == 0

    def test_default_normalized_false(self):
        assert _cmr().normalized is False


# ─── ContourBatchRecord ───────────────────────────────────────────────────────

class TestContourBatchRecordExtra:
    def test_stores_n_contours(self):
        assert _cbr(n=10).n_contours == 10

    def test_stores_n_points_config(self):
        assert _cbr(npc=128).n_points_config == 128

    def test_stores_smooth_sigma(self):
        assert _cbr(sigma=2.0).smooth_sigma == pytest.approx(2.0)

    def test_stores_rdp_epsilon(self):
        assert _cbr(eps=0.05).rdp_epsilon == pytest.approx(0.05)

    def test_success_rate(self):
        r = _cbr(n=5, n_succ=4)
        assert r.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_contours(self):
        r = _cbr(n=0, n_succ=0)
        assert r.success_rate == pytest.approx(0.0)

    def test_stores_normalize(self):
        assert _cbr(normalize=False).normalize is False


# ─── make_contour_process_record ──────────────────────────────────────────────

class TestMakeContourProcessRecordExtra:
    def test_returns_record(self):
        r = make_contour_process_record(0, 100, 80, 50.0, 200.0, 0.8)
        assert isinstance(r, ContourProcessRecord)

    def test_values_stored(self):
        r = make_contour_process_record(3, 120, 90, 60.0, 300.0, 0.7)
        assert r.fragment_id == 3
        assert r.n_points_before == 120
        assert r.perimeter == pytest.approx(60.0)

    def test_normalized_flag(self):
        r = make_contour_process_record(0, 100, 80, 50.0, 200.0, 0.8, normalized=False)
        assert r.normalized is False

    def test_simplified_flag(self):
        r = make_contour_process_record(0, 100, 80, 50.0, 200.0, 0.8, simplified=True)
        assert r.simplified is True


# ─── make_contrast_enhance_record ─────────────────────────────────────────────

class TestMakeContrastEnhanceRecordExtra:
    def test_returns_record(self):
        r = make_contrast_enhance_record("clahe", 0.3, 0.8, (64, 64))
        assert isinstance(r, ContrastEnhanceRecord)

    def test_grayscale_shape(self):
        r = make_contrast_enhance_record("clahe", 0.3, 0.8, (64, 64))
        assert r.n_channels == 1
        assert r.is_grayscale is True

    def test_bgr_shape(self):
        r = make_contrast_enhance_record("clahe", 0.3, 0.8, (64, 64, 3))
        assert r.n_channels == 3
        assert r.is_grayscale is False

    def test_dimensions_from_shape(self):
        r = make_contrast_enhance_record("eq", 0.2, 0.7, (48, 96))
        assert r.image_height == 48 and r.image_width == 96
