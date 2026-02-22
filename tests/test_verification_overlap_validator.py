"""Tests for puzzle_reconstruction/verification/overlap_validator.py."""
import numpy as np
import pytest

from puzzle_reconstruction.verification.overlap_validator import (
    OverlapRecord,
    ValidationReport,
    bbox_overlap,
    mask_iou,
    check_pair_overlap,
    validate_assembly,
    overlap_area_matrix,
    batch_validate,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mask(h=32, w=32, filled=True):
    if filled:
        return np.ones((h, w), dtype=np.uint8) * 255
    return np.zeros((h, w), dtype=np.uint8)


# ─── TestOverlapRecord ────────────────────────────────────────────────────────

class TestOverlapRecord:
    def test_valid_creation(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=10.0, iou=0.5)
        assert r.idx1 == 0 and r.idx2 == 1

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=-1, idx2=1, overlap_area=0.0, iou=0.0)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=-1, overlap_area=0.0, iou=0.0)

    def test_negative_overlap_area_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=-1.0, iou=0.0)

    def test_iou_above_1_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=1.5)

    def test_iou_below_0_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=-0.1)

    def test_iou_at_zero_boundary(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=0.0)
        assert r.iou == 0.0

    def test_iou_at_one_boundary(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=1.0)
        assert r.iou == 1.0

    def test_pair_property(self):
        r = OverlapRecord(idx1=2, idx2=5, overlap_area=0.0, iou=0.0)
        assert r.pair == (2, 5)

    def test_default_params_dict(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=0.0)
        assert isinstance(r.params, dict)

    def test_stores_overlap_area(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=42.5, iou=0.3)
        assert r.overlap_area == pytest.approx(42.5)

    def test_zero_idx_zero_area_zero_iou(self):
        r = OverlapRecord(idx1=0, idx2=0, overlap_area=0.0, iou=0.0)
        assert r.idx1 == 0 and r.idx2 == 0


# ─── TestValidationReport ─────────────────────────────────────────────────────

class TestValidationReport:
    def _make(self, n_overlaps=0, total_area=0.0, max_iou=0.0, is_valid=True):
        return ValidationReport(
            overlaps=[],
            n_overlaps=n_overlaps,
            total_area=total_area,
            max_iou=max_iou,
            is_valid=is_valid,
        )

    def test_valid_creation(self):
        r = self._make()
        assert r.is_valid is True

    def test_negative_n_overlaps_raises(self):
        with pytest.raises(ValueError):
            self._make(n_overlaps=-1)

    def test_negative_total_area_raises(self):
        with pytest.raises(ValueError):
            self._make(total_area=-1.0)

    def test_len_equals_n_overlaps(self):
        r = self._make(n_overlaps=3)
        assert len(r) == 3

    def test_zero_len(self):
        r = self._make(n_overlaps=0)
        assert len(r) == 0

    def test_stores_overlaps_list(self):
        rec = OverlapRecord(idx1=0, idx2=1, overlap_area=5.0, iou=0.3)
        r = ValidationReport(
            overlaps=[rec],
            n_overlaps=1,
            total_area=5.0,
            max_iou=0.3,
            is_valid=False,
        )
        assert len(r.overlaps) == 1

    def test_default_params_dict(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_is_valid_false(self):
        r = self._make(n_overlaps=1, is_valid=False)
        assert r.is_valid is False


# ─── TestBboxOverlap ──────────────────────────────────────────────────────────

class TestBboxOverlap:
    def test_non_overlapping(self):
        result = bbox_overlap((0, 0, 10, 10), (20, 0, 10, 10))
        assert result == pytest.approx(0.0)

    def test_fully_overlapping(self):
        result = bbox_overlap((0, 0, 10, 10), (0, 0, 10, 10))
        assert result == pytest.approx(100.0)

    def test_partial_overlap(self):
        result = bbox_overlap((0, 0, 10, 10), (5, 0, 10, 10))
        assert result == pytest.approx(50.0)

    def test_touching_edges_zero(self):
        result = bbox_overlap((0, 0, 10, 10), (10, 0, 10, 10))
        assert result == pytest.approx(0.0)

    def test_negative_w1_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, -1, 10), (0, 0, 10, 10))

    def test_negative_h1_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, 10, -1), (0, 0, 10, 10))

    def test_negative_w2_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, 10, 10), (0, 0, -1, 10))

    def test_negative_h2_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, 10, 10), (0, 0, 10, -1))

    def test_returns_float(self):
        result = bbox_overlap((0, 0, 5, 5), (2, 2, 5, 5))
        assert isinstance(result, float)

    def test_zero_width_zero_overlap(self):
        result = bbox_overlap((0, 0, 0, 10), (0, 0, 10, 10))
        assert result == pytest.approx(0.0)

    def test_result_nonneg(self):
        result = bbox_overlap((0, 0, 8, 8), (4, 4, 8, 8))
        assert result >= 0.0


# ─── TestMaskIou ──────────────────────────────────────────────────────────────

class TestMaskIou:
    def test_identical_masks_iou_one(self):
        m = _mask(16, 16, filled=True)
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_empty_masks_iou_zero(self):
        m = _mask(16, 16, filled=False)
        assert mask_iou(m, m) == pytest.approx(0.0)

    def test_no_overlap_iou_zero(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_different_shapes_raises(self):
        with pytest.raises(ValueError):
            mask_iou(_mask(8, 8), _mask(16, 16))

    def test_non_2d_mask1_raises(self):
        with pytest.raises(ValueError):
            mask_iou(np.ones((4, 4, 1), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8))

    def test_non_2d_mask2_raises(self):
        with pytest.raises(ValueError):
            mask_iou(np.ones((4, 4), dtype=np.uint8), np.ones((4, 4, 1), dtype=np.uint8))

    def test_result_in_zero_one(self):
        m1 = _mask(8, 8, filled=True)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[:4, :4] = 255
        result = mask_iou(m1, m2)
        assert 0.0 <= result <= 1.0

    def test_partial_overlap(self):
        # m1: full 8x8 (64 px), m2: top-left 4x4 (16 px)
        # intersection=16, union=64 → IoU=0.25
        m1 = _mask(8, 8, filled=True)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[:4, :4] = 255
        result = mask_iou(m1, m2)
        assert result == pytest.approx(0.25)


# ─── TestCheckPairOverlap ─────────────────────────────────────────────────────

class TestCheckPairOverlap:
    def test_returns_overlap_record(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (32, 32))
        assert isinstance(r, OverlapRecord)

    def test_no_overlap(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (32, 0))
        assert r.overlap_area == pytest.approx(0.0)
        assert r.iou == pytest.approx(0.0)

    def test_full_overlap(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (0, 0))
        assert r.overlap_area > 0.0
        assert r.iou == pytest.approx(1.0)

    def test_negative_idx1_raises(self):
        m = _mask(16, 16)
        with pytest.raises(ValueError):
            check_pair_overlap(-1, 1, m, m, (64, 64))

    def test_negative_idx2_raises(self):
        m = _mask(16, 16)
        with pytest.raises(ValueError):
            check_pair_overlap(0, -1, m, m, (64, 64))

    def test_non_2d_mask1_raises(self):
        m2d = _mask(16, 16)
        m3d = np.ones((16, 16, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            check_pair_overlap(0, 1, m3d, m2d, (64, 64))

    def test_non_2d_mask2_raises(self):
        m2d = _mask(16, 16)
        m3d = np.ones((16, 16, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            check_pair_overlap(0, 1, m2d, m3d, (64, 64))

    def test_stores_indices(self):
        m = _mask(16, 16)
        r = check_pair_overlap(3, 7, m, m, (64, 64), (0, 0), (32, 0))
        assert r.idx1 == 3 and r.idx2 == 7

    def test_iou_in_range(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (8, 0))
        assert 0.0 <= r.iou <= 1.0


# ─── TestValidateAssembly ─────────────────────────────────────────────────────

class TestValidateAssembly:
    def test_returns_validation_report(self):
        m = _mask(16, 16)
        r = validate_assembly([m], [(0, 0)], (64, 64))
        assert isinstance(r, ValidationReport)

    def test_single_fragment_valid(self):
        m = _mask(16, 16)
        r = validate_assembly([m], [(0, 0)], (64, 64))
        assert r.is_valid is True
        assert r.n_overlaps == 0

    def test_non_overlapping_valid(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (32, 0)], (64, 64))
        assert r.is_valid is True

    def test_overlapping_detected(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (0, 0)], (64, 64))
        assert r.n_overlaps > 0
        assert r.is_valid is False

    def test_length_mismatch_raises(self):
        m = _mask(16, 16)
        with pytest.raises(ValueError):
            validate_assembly([m, m], [(0, 0)], (64, 64))

    def test_negative_iou_threshold_raises(self):
        m = _mask(16, 16)
        with pytest.raises(ValueError):
            validate_assembly([m], [(0, 0)], (64, 64), iou_threshold=-0.1)

    def test_empty_assembly_valid(self):
        r = validate_assembly([], [], (64, 64))
        assert r.is_valid is True
        assert r.n_overlaps == 0

    def test_total_area_nonneg(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (0, 0)], (64, 64))
        assert r.total_area >= 0.0

    def test_max_iou_nonneg(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (0, 0)], (64, 64))
        assert r.max_iou >= 0.0

    def test_iou_threshold_filters_small_overlaps(self):
        m = _mask(16, 16)
        # Partial overlap: second mask shifted by half
        r_low = validate_assembly([m, m], [(0, 0), (8, 0)], (64, 64), iou_threshold=0.0)
        r_high = validate_assembly([m, m], [(0, 0), (8, 0)], (64, 64), iou_threshold=0.9)
        assert r_low.n_overlaps >= r_high.n_overlaps


# ─── TestOverlapAreaMatrix ────────────────────────────────────────────────────

class TestOverlapAreaMatrix:
    def test_shape(self):
        m = _mask(16, 16)
        result = overlap_area_matrix([m, m, m], [(0, 0), (32, 0), (0, 32)], (64, 64))
        assert result.shape == (3, 3)

    def test_symmetric(self):
        m = _mask(16, 16)
        result = overlap_area_matrix([m, m], [(0, 0), (8, 0)], (64, 64))
        assert result[0, 1] == pytest.approx(result[1, 0])

    def test_no_overlap_zeros(self):
        m = _mask(16, 16)
        result = overlap_area_matrix([m, m], [(0, 0), (32, 0)], (64, 64))
        assert result[0, 1] == pytest.approx(0.0)

    def test_dtype_float64(self):
        m = _mask(8, 8)
        result = overlap_area_matrix([m, m], [(0, 0), (32, 0)], (64, 64))
        assert result.dtype == np.float64

    def test_single_element_zero_diagonal(self):
        m = _mask(8, 8)
        result = overlap_area_matrix([m], [(0, 0)], (32, 32))
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_overlap_positive(self):
        m = _mask(16, 16)
        result = overlap_area_matrix([m, m], [(0, 0), (0, 0)], (64, 64))
        assert result[0, 1] > 0.0

    def test_nonneg_values(self):
        m = _mask(16, 16)
        result = overlap_area_matrix([m, m, m], [(0, 0), (8, 0), (32, 0)], (64, 64))
        assert (result >= 0.0).all()


# ─── TestBatchValidate ────────────────────────────────────────────────────────

class TestBatchValidate:
    def test_returns_list(self):
        m = _mask(16, 16)
        result = batch_validate([([m], [(0, 0)], (64, 64))])
        assert isinstance(result, list)

    def test_length_matches(self):
        m = _mask(16, 16)
        assemblies = [
            ([m], [(0, 0)], (64, 64)),
            ([m, m], [(0, 0), (32, 0)], (64, 64)),
        ]
        result = batch_validate(assemblies)
        assert len(result) == 2

    def test_each_is_validation_report(self):
        m = _mask(16, 16)
        for r in batch_validate([([m], [(0, 0)], (64, 64))]):
            assert isinstance(r, ValidationReport)

    def test_empty_list_returns_empty(self):
        result = batch_validate([])
        assert result == []

    def test_iou_threshold_applied(self):
        m = _mask(16, 16)
        assemblies = [([m, m], [(0, 0), (0, 0)], (64, 64))]
        r_thresh = batch_validate(assemblies, iou_threshold=0.99)
        r_no_thresh = batch_validate(assemblies, iou_threshold=0.0)
        assert r_no_thresh[0].n_overlaps >= r_thresh[0].n_overlaps
