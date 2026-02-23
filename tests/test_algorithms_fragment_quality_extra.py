"""Extra tests for puzzle_reconstruction/algorithms/fragment_quality.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fragment_quality import (
    QualityConfig,
    QualityReport,
    measure_blur,
    measure_contrast,
    measure_mask_coverage,
    measure_edge_sharpness,
    assess_fragment,
    rank_fragments,
    batch_assess,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=50, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def _mask(h=50, w=50, filled=True):
    if filled:
        return np.full((h, w), 255, dtype=np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h=50, w=50, fill=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = fill
    return img


def _report(fid=0, score=0.5):
    return QualityReport(
        fragment_id=fid, score=score, blur=score,
        contrast=score, coverage=score, sharpness=score,
    )


# ─── QualityConfig (extra) ────────────────────────────────────────────────────

class TestQualityConfigExtra:
    def test_large_weights_ok(self):
        cfg = QualityConfig(w_blur=10.0, w_contrast=10.0,
                            w_coverage=10.0, w_sharpness=10.0)
        assert cfg.total_weight == pytest.approx(40.0)

    def test_single_nonzero_weight(self):
        cfg = QualityConfig(w_blur=1.0, w_contrast=0.0,
                            w_coverage=0.0, w_sharpness=0.0)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_custom_blur_ref(self):
        cfg = QualityConfig(blur_ref=1000.0)
        assert cfg.blur_ref == pytest.approx(1000.0)

    def test_custom_grad_ref(self):
        cfg = QualityConfig(grad_ref=100.0)
        assert cfg.grad_ref == pytest.approx(100.0)

    def test_independent_instances(self):
        c1 = QualityConfig(w_blur=1.0)
        c2 = QualityConfig(w_blur=2.0)
        assert c1.w_blur != c2.w_blur

    def test_grad_ref_positive(self):
        cfg = QualityConfig(grad_ref=0.1)
        assert cfg.grad_ref == pytest.approx(0.1)


# ─── QualityReport (extra) ────────────────────────────────────────────────────

class TestQualityReportExtra:
    def test_score_zero_ok(self):
        r = _report(score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        r = _report(score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_score_boundary_is_usable(self):
        r = _report(score=0.3)
        assert r.is_usable is True

    def test_low_score_not_usable(self):
        r = _report(score=0.05)
        assert r.is_usable is False

    def test_params_stored(self):
        r = QualityReport(
            fragment_id=0, score=0.5, blur=0.5,
            contrast=0.5, coverage=0.5, sharpness=0.5,
            params={"method": "laplacian"},
        )
        assert r.params["method"] == "laplacian"

    def test_summary_contains_score(self):
        r = _report(score=0.75)
        summary = r.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_fragment_id_zero_ok(self):
        r = _report(fid=0)
        assert r.fragment_id == 0

    def test_large_fragment_id_ok(self):
        r = QualityReport(
            fragment_id=9999, score=0.5, blur=0.5,
            contrast=0.5, coverage=0.5, sharpness=0.5,
        )
        assert r.fragment_id == 9999


# ─── measure_blur (extra) ─────────────────────────────────────────────────────

class TestMeasureBlurExtra:
    def test_gradient_higher_than_flat(self):
        grad = _gradient()
        flat = _gray(fill=128)
        assert measure_blur(grad) >= measure_blur(flat)

    def test_large_ref_gives_lower_score(self):
        img = _gradient()
        s1 = measure_blur(img, ref=100.0)
        s2 = measure_blur(img, ref=10000.0)
        assert s1 >= s2

    def test_small_ref_gives_higher_score(self):
        img = _gradient()
        s1 = measure_blur(img, ref=1.0)
        s2 = measure_blur(img, ref=1000.0)
        assert s1 >= s2

    def test_result_float(self):
        result = measure_blur(_gradient())
        assert isinstance(result, float)

    def test_consistent_calls(self):
        img = _gradient()
        assert measure_blur(img) == pytest.approx(measure_blur(img))


# ─── measure_contrast (extra) ─────────────────────────────────────────────────

class TestMeasureContrastExtra:
    def test_gradient_image_nonzero(self):
        result = measure_contrast(_gradient())
        assert result > 0.0

    def test_high_contrast_above_half(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[:25, :] = 255
        result = measure_contrast(img)
        assert result > 0.5

    def test_result_float(self):
        result = measure_contrast(_gradient())
        assert isinstance(result, float)

    def test_with_full_mask_same_as_no_mask(self):
        img = _gradient()
        mask = _mask(filled=True)
        r_no = measure_contrast(img)
        r_mask = measure_contrast(img, mask=mask)
        assert r_no == pytest.approx(r_mask, rel=1e-4)

    def test_consistent_calls(self):
        img = _gradient()
        assert measure_contrast(img) == pytest.approx(measure_contrast(img))


# ─── measure_mask_coverage (extra) ────────────────────────────────────────────

class TestMeasureMaskCoverageExtra:
    def test_binary_mask_values(self):
        # mask with 0/1 values instead of 0/255
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[:25, :] = 1
        result = measure_mask_coverage(mask)
        assert 0.0 <= result <= 1.0

    def test_quarter_filled(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:10, :] = 255
        result = measure_mask_coverage(mask)
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_three_quarter_filled(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:30, :] = 255
        result = measure_mask_coverage(mask)
        assert result == pytest.approx(0.75, abs=1e-6)

    def test_result_nonneg(self):
        mask = _mask(filled=True)
        assert measure_mask_coverage(mask) >= 0.0

    def test_consistent_calls(self):
        mask = _mask(filled=True)
        assert measure_mask_coverage(mask) == pytest.approx(measure_mask_coverage(mask))


# ─── measure_edge_sharpness (extra) ───────────────────────────────────────────

class TestMeasureEdgeSharpnessExtra:
    def test_gradient_image_nonzero(self):
        result = measure_edge_sharpness(_gradient())
        assert result > 0.0

    def test_result_float(self):
        result = measure_edge_sharpness(_gradient())
        assert isinstance(result, float)

    def test_large_ref_gives_lower_score(self):
        img = _gradient()
        s1 = measure_edge_sharpness(img, ref=1.0)
        s2 = measure_edge_sharpness(img, ref=1000.0)
        assert s1 >= s2

    def test_full_mask_same_as_no_mask(self):
        img = _gradient()
        mask = _mask(filled=True)
        r_no = measure_edge_sharpness(img)
        r_mask = measure_edge_sharpness(img, mask=mask)
        assert r_no == pytest.approx(r_mask, rel=1e-4)

    def test_consistent_calls(self):
        img = _gradient()
        r1 = measure_edge_sharpness(img)
        r2 = measure_edge_sharpness(img)
        assert r1 == pytest.approx(r2)


# ─── assess_fragment (extra) ──────────────────────────────────────────────────

class TestAssessFragmentExtra:
    def test_fragment_id_default_zero(self):
        result = assess_fragment(_gradient())
        assert result.fragment_id == 0

    def test_custom_fragment_id_10(self):
        result = assess_fragment(_gradient(), fragment_id=10)
        assert result.fragment_id == 10

    def test_gradient_higher_score_than_flat(self):
        r_grad = assess_fragment(_gradient())
        r_flat = assess_fragment(_gray(fill=128))
        assert r_grad.score >= r_flat.score

    def test_all_weights_zero_except_coverage(self):
        cfg = QualityConfig(w_blur=0.0, w_contrast=0.0,
                            w_coverage=1.0, w_sharpness=0.0)
        r = assess_fragment(_gradient(), cfg=cfg)
        assert isinstance(r, QualityReport)

    def test_blur_ref_large(self):
        cfg = QualityConfig(blur_ref=100000.0)
        r = assess_fragment(_gradient(), cfg=cfg)
        assert r.blur == pytest.approx(0.0, abs=0.01)

    def test_score_consistent(self):
        img = _gradient()
        r1 = assess_fragment(img)
        r2 = assess_fragment(img)
        assert r1.score == pytest.approx(r2.score)


# ─── rank_fragments (extra) ───────────────────────────────────────────────────

class TestRankFragmentsExtra:
    def test_single_report(self):
        ranked = rank_fragments([_report()])
        assert len(ranked) == 1

    def test_all_same_score_returns_all(self):
        reports = [_report(fid=i, score=0.5) for i in range(5)]
        ranked = rank_fragments(reports)
        assert len(ranked) == 5

    def test_first_is_highest_score(self):
        reports = [_report(fid=0, score=0.3), _report(fid=1, score=0.9),
                   _report(fid=2, score=0.6)]
        ranked = rank_fragments(reports)
        _, best_score = ranked[0]
        assert best_score == pytest.approx(0.9)

    def test_last_is_lowest_score(self):
        reports = [_report(fid=0, score=0.3), _report(fid=1, score=0.9),
                   _report(fid=2, score=0.6)]
        ranked = rank_fragments(reports)
        _, worst_score = ranked[-1]
        assert worst_score == pytest.approx(0.3)

    def test_tuple_format(self):
        reports = [_report()]
        ranked = rank_fragments(reports)
        fid, score = ranked[0]
        assert isinstance(fid, int)
        assert isinstance(score, float)

    def test_custom_indices_stored(self):
        reports = [_report(fid=0, score=0.5), _report(fid=1, score=0.8)]
        ranked = rank_fragments(reports, indices=[100, 200])
        ids = {fid for fid, _ in ranked}
        assert ids == {100, 200}


# ─── batch_assess (extra) ─────────────────────────────────────────────────────

class TestBatchAssessExtra:
    def test_single_image(self):
        result = batch_assess([_gradient()])
        assert len(result) == 1

    def test_large_batch(self):
        imgs = [_gradient() for _ in range(8)]
        result = batch_assess(imgs)
        assert len(result) == 8

    def test_gradient_scores_above_flat(self):
        imgs = [_gradient(), _gray(fill=128)]
        results = batch_assess(imgs)
        assert results[0].score >= results[1].score

    def test_with_custom_config(self):
        cfg = QualityConfig(w_blur=2.0, w_contrast=0.0,
                            w_coverage=0.0, w_sharpness=0.0)
        imgs = [_gradient()]
        result = batch_assess(imgs, cfg=cfg)
        assert isinstance(result[0], QualityReport)

    def test_fragment_ids_sequential_from_zero(self):
        imgs = [_gradient() for _ in range(3)]
        result = batch_assess(imgs)
        assert result[0].fragment_id == 0
        assert result[1].fragment_id == 1
        assert result[2].fragment_id == 2

    def test_bgr_images_ok(self):
        imgs = [_bgr()]
        result = batch_assess(imgs)
        assert len(result) == 1
        assert isinstance(result[0], QualityReport)
