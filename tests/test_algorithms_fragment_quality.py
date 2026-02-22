"""Tests for puzzle_reconstruction/algorithms/fragment_quality.py"""
import pytest
import numpy as np

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=50, w=50, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=50, w=50, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


def make_gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_mask(h=50, w=50, foreground=True):
    if foreground:
        return np.full((h, w), 255, dtype=np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


# ─── QualityConfig ────────────────────────────────────────────────────────────

class TestQualityConfig:
    def test_defaults(self):
        cfg = QualityConfig()
        assert cfg.w_blur == pytest.approx(1.0)
        assert cfg.w_contrast == pytest.approx(1.0)
        assert cfg.w_coverage == pytest.approx(1.0)
        assert cfg.w_sharpness == pytest.approx(1.0)
        assert cfg.blur_ref == pytest.approx(500.0)
        assert cfg.grad_ref == pytest.approx(50.0)

    def test_negative_w_blur_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(w_blur=-0.1)

    def test_negative_w_contrast_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(w_contrast=-1.0)

    def test_negative_w_coverage_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(w_coverage=-0.5)

    def test_negative_w_sharpness_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(w_sharpness=-0.01)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(w_blur=0.0, w_contrast=0.0,
                          w_coverage=0.0, w_sharpness=0.0)

    def test_blur_ref_zero_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(blur_ref=0.0)

    def test_blur_ref_negative_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(blur_ref=-10.0)

    def test_grad_ref_zero_raises(self):
        with pytest.raises(ValueError):
            QualityConfig(grad_ref=0.0)

    def test_total_weight(self):
        cfg = QualityConfig(w_blur=1.0, w_contrast=2.0,
                            w_coverage=3.0, w_sharpness=4.0)
        assert cfg.total_weight == pytest.approx(10.0)

    def test_custom_config_valid(self):
        cfg = QualityConfig(w_blur=0.0, w_contrast=1.0,
                            w_coverage=1.0, w_sharpness=0.0)
        assert cfg.total_weight == pytest.approx(2.0)


# ─── QualityReport ────────────────────────────────────────────────────────────

class TestQualityReport:
    def test_basic_creation(self):
        r = QualityReport(fragment_id=0, score=0.8, blur=0.7,
                          contrast=0.6, coverage=0.9, sharpness=0.5)
        assert r.score == pytest.approx(0.8)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=-1, score=0.5, blur=0.5,
                          contrast=0.5, coverage=0.5, sharpness=0.5)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=0, score=1.5, blur=0.5,
                          contrast=0.5, coverage=0.5, sharpness=0.5)

    def test_blur_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=0, score=0.5, blur=-0.1,
                          contrast=0.5, coverage=0.5, sharpness=0.5)

    def test_contrast_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=0, score=0.5, blur=0.5,
                          contrast=1.2, coverage=0.5, sharpness=0.5)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=0, score=0.5, blur=0.5,
                          contrast=0.5, coverage=-0.5, sharpness=0.5)

    def test_sharpness_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityReport(fragment_id=0, score=0.5, blur=0.5,
                          contrast=0.5, coverage=0.5, sharpness=2.0)

    def test_is_usable_true(self):
        r = QualityReport(fragment_id=0, score=0.5, blur=0.5,
                          contrast=0.5, coverage=0.5, sharpness=0.5)
        assert r.is_usable is True

    def test_is_usable_false(self):
        r = QualityReport(fragment_id=0, score=0.1, blur=0.1,
                          contrast=0.1, coverage=0.1, sharpness=0.1)
        assert r.is_usable is False

    def test_is_usable_at_boundary(self):
        r = QualityReport(fragment_id=0, score=0.3, blur=0.3,
                          contrast=0.3, coverage=0.3, sharpness=0.3)
        assert r.is_usable is True

    def test_summary_contains_id(self):
        r = QualityReport(fragment_id=42, score=0.7, blur=0.7,
                          contrast=0.7, coverage=0.7, sharpness=0.7)
        summary = r.summary()
        assert "42" in summary

    def test_default_params_empty(self):
        r = QualityReport(fragment_id=0, score=0.5, blur=0.5,
                          contrast=0.5, coverage=0.5, sharpness=0.5)
        assert r.params == {}


# ─── measure_blur ─────────────────────────────────────────────────────────────

class TestMeasureBlur:
    def test_range(self):
        img = make_gradient()
        result = measure_blur(img)
        assert 0.0 <= result <= 1.0

    def test_sharp_image_higher_than_blurred(self):
        sharp = make_gradient()
        blurred = np.full((50, 50), 128, dtype=np.uint8)
        assert measure_blur(sharp) >= measure_blur(blurred)

    def test_bgr_input(self):
        img = make_bgr()
        result = measure_blur(img)
        assert 0.0 <= result <= 1.0

    def test_ref_zero_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            measure_blur(img, ref=0.0)

    def test_ref_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            measure_blur(img, ref=-10.0)

    def test_constant_image_low_score(self):
        img = make_gray(value=128)
        score = measure_blur(img, ref=500.0)
        assert score == pytest.approx(0.0)


# ─── measure_contrast ─────────────────────────────────────────────────────────

class TestMeasureContrast:
    def test_range(self):
        img = make_gradient()
        result = measure_contrast(img)
        assert 0.0 <= result <= 1.0

    def test_constant_image_zero(self):
        img = make_gray(value=128)
        result = measure_contrast(img)
        assert result == pytest.approx(0.0)

    def test_bgr_input(self):
        img = make_bgr()
        result = measure_contrast(img)
        assert 0.0 <= result <= 1.0

    def test_with_mask(self):
        img = make_gradient()
        mask = make_mask(foreground=True)
        result = measure_contrast(img, mask=mask)
        assert 0.0 <= result <= 1.0

    def test_empty_mask_region_returns_zero(self):
        img = make_gradient()
        mask = make_mask(foreground=False)
        result = measure_contrast(img, mask=mask)
        assert result == pytest.approx(0.0)

    def test_high_contrast_image(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[:25, :] = 255
        result = measure_contrast(img)
        assert result > 0.5


# ─── measure_mask_coverage ───────────────────────────────────────────────────

class TestMeasureMaskCoverage:
    def test_full_mask_gives_one(self):
        mask = make_mask(foreground=True)
        assert measure_mask_coverage(mask) == pytest.approx(1.0)

    def test_empty_mask_gives_zero(self):
        mask = make_mask(foreground=False)
        assert measure_mask_coverage(mask) == pytest.approx(0.0)

    def test_partial_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[:25, :] = 255
        result = measure_mask_coverage(mask)
        assert result == pytest.approx(0.5)

    def test_3d_mask_raises(self):
        mask = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            measure_mask_coverage(mask)

    def test_range(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        result = measure_mask_coverage(mask)
        assert 0.0 <= result <= 1.0


# ─── measure_edge_sharpness ──────────────────────────────────────────────────

class TestMeasureEdgeSharpness:
    def test_range(self):
        img = make_gradient()
        result = measure_edge_sharpness(img)
        assert 0.0 <= result <= 1.0

    def test_constant_image_low_score(self):
        img = make_gray(value=128)
        result = measure_edge_sharpness(img)
        assert result == pytest.approx(0.0)

    def test_bgr_input(self):
        img = make_bgr()
        result = measure_edge_sharpness(img)
        assert 0.0 <= result <= 1.0

    def test_ref_zero_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            measure_edge_sharpness(img, ref=0.0)

    def test_ref_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            measure_edge_sharpness(img, ref=-5.0)

    def test_with_mask(self):
        img = make_gradient()
        mask = make_mask(foreground=True)
        result = measure_edge_sharpness(img, mask=mask)
        assert 0.0 <= result <= 1.0

    def test_empty_mask_returns_zero(self):
        img = make_gradient()
        mask = make_mask(foreground=False)
        result = measure_edge_sharpness(img, mask=mask)
        assert result == pytest.approx(0.0)


# ─── assess_fragment ─────────────────────────────────────────────────────────

class TestAssessFragment:
    def test_returns_quality_report(self):
        img = make_gradient()
        result = assess_fragment(img)
        assert isinstance(result, QualityReport)

    def test_score_in_range(self):
        img = make_gradient()
        result = assess_fragment(img)
        assert 0.0 <= result.score <= 1.0

    def test_fragment_id_stored(self):
        img = make_gradient()
        result = assess_fragment(img, fragment_id=7)
        assert result.fragment_id == 7

    def test_with_mask(self):
        img = make_gradient()
        mask = make_mask(foreground=True)
        result = assess_fragment(img, mask=mask)
        assert isinstance(result, QualityReport)

    def test_bgr_input(self):
        img = make_bgr()
        result = assess_fragment(img)
        assert isinstance(result, QualityReport)

    def test_custom_config(self):
        img = make_gradient()
        cfg = QualityConfig(w_blur=2.0, w_contrast=0.0,
                            w_coverage=0.0, w_sharpness=0.0)
        result = assess_fragment(img, cfg=cfg)
        assert isinstance(result, QualityReport)

    def test_all_metrics_in_range(self):
        img = make_gradient()
        result = assess_fragment(img)
        assert 0.0 <= result.blur <= 1.0
        assert 0.0 <= result.contrast <= 1.0
        assert 0.0 <= result.coverage <= 1.0
        assert 0.0 <= result.sharpness <= 1.0


# ─── rank_fragments ──────────────────────────────────────────────────────────

class TestRankFragments:
    def _make_reports(self, scores):
        return [
            QualityReport(fragment_id=i, score=s, blur=s,
                          contrast=s, coverage=s, sharpness=s)
            for i, s in enumerate(scores)
        ]

    def test_sorted_desc(self):
        reports = self._make_reports([0.3, 0.9, 0.6])
        ranked = rank_fragments(reports)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_returns_list_of_tuples(self):
        reports = self._make_reports([0.5, 0.8])
        ranked = rank_fragments(reports)
        assert isinstance(ranked, list)
        assert all(isinstance(item, tuple) for item in ranked)

    def test_fragment_ids_preserved(self):
        reports = self._make_reports([0.5, 0.8, 0.3])
        ranked = rank_fragments(reports)
        ids = {fid for fid, _ in ranked}
        assert ids == {0, 1, 2}

    def test_empty_reports(self):
        ranked = rank_fragments([])
        assert ranked == []

    def test_indices_mismatch_raises(self):
        reports = self._make_reports([0.5, 0.8])
        with pytest.raises(ValueError):
            rank_fragments(reports, indices=[0, 1, 2])

    def test_custom_indices(self):
        reports = self._make_reports([0.5, 0.8])
        ranked = rank_fragments(reports, indices=[10, 20])
        ids = {fid for fid, _ in ranked}
        assert ids == {10, 20}


# ─── batch_assess ────────────────────────────────────────────────────────────

class TestBatchAssess:
    def test_returns_list(self):
        imgs = [make_gradient() for _ in range(3)]
        results = batch_assess(imgs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_quality_reports(self):
        imgs = [make_gradient() for _ in range(3)]
        results = batch_assess(imgs)
        assert all(isinstance(r, QualityReport) for r in results)

    def test_fragment_ids_sequential(self):
        imgs = [make_gradient() for _ in range(4)]
        results = batch_assess(imgs)
        assert [r.fragment_id for r in results] == [0, 1, 2, 3]

    def test_empty_list(self):
        results = batch_assess([])
        assert results == []

    def test_masks_length_mismatch_raises(self):
        imgs = [make_gradient() for _ in range(3)]
        masks = [make_mask() for _ in range(2)]
        with pytest.raises(ValueError):
            batch_assess(imgs, masks=masks)

    def test_with_masks(self):
        imgs = [make_gradient() for _ in range(2)]
        masks = [make_mask() for _ in range(2)]
        results = batch_assess(imgs, masks=masks)
        assert len(results) == 2

    def test_scores_in_range(self):
        imgs = [make_gradient(), make_gray(value=128)]
        results = batch_assess(imgs)
        for r in results:
            assert 0.0 <= r.score <= 1.0
