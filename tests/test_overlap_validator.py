"""Тесты для puzzle_reconstruction.verification.overlap_validator."""
import pytest
import numpy as np

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

def _square_mask(size=16, fill=True):
    m = np.zeros((size, size), dtype=np.uint8)
    if fill:
        m[:] = 255
    return m


# ─── TestOverlapRecord ────────────────────────────────────────────────────────

class TestOverlapRecord:
    def test_basic_creation(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=10.0, iou=0.5)
        assert r.overlap_area == 10.0
        assert r.iou == 0.5

    def test_pair_property(self):
        r = OverlapRecord(idx1=2, idx2=5, overlap_area=0.0, iou=0.0)
        assert r.pair == (2, 5)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=-1, idx2=0, overlap_area=0.0, iou=0.0)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=-1, overlap_area=0.0, iou=0.0)

    def test_negative_overlap_area_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=-1.0, iou=0.0)

    def test_iou_below_zero_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=-0.1)

    def test_iou_above_one_raises(self):
        with pytest.raises(ValueError):
            OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=1.1)

    def test_boundary_values_valid(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=0.0)
        assert r.iou == 0.0
        r2 = OverlapRecord(idx1=0, idx2=1, overlap_area=100.0, iou=1.0)
        assert r2.iou == 1.0


# ─── TestValidationReport ─────────────────────────────────────────────────────

class TestValidationReport:
    def _make(self, n=0):
        return ValidationReport(
            overlaps=[],
            n_overlaps=n,
            total_area=0.0,
            max_iou=0.0,
            is_valid=(n == 0),
        )

    def test_len(self):
        rpt = self._make(3)
        assert len(rpt) == 3

    def test_is_valid_when_no_overlaps(self):
        rpt = self._make(0)
        assert rpt.is_valid is True

    def test_negative_n_overlaps_raises(self):
        with pytest.raises(ValueError):
            ValidationReport(
                overlaps=[], n_overlaps=-1,
                total_area=0.0, max_iou=0.0, is_valid=True,
            )

    def test_negative_total_area_raises(self):
        with pytest.raises(ValueError):
            ValidationReport(
                overlaps=[], n_overlaps=0,
                total_area=-1.0, max_iou=0.0, is_valid=True,
            )


# ─── TestBboxOverlap ──────────────────────────────────────────────────────────

class TestBboxOverlap:
    def test_full_overlap(self):
        area = bbox_overlap((0, 0, 10, 10), (0, 0, 10, 10))
        assert area == pytest.approx(100.0)

    def test_no_overlap(self):
        area = bbox_overlap((0, 0, 5, 5), (10, 10, 5, 5))
        assert area == pytest.approx(0.0)

    def test_partial_overlap(self):
        # (0,0)-(10,10) ∩ (5,5)-(15,15) = 5×5 = 25
        area = bbox_overlap((0, 0, 10, 10), (5, 5, 10, 10))
        assert area == pytest.approx(25.0)

    def test_touching_edges_no_overlap(self):
        area = bbox_overlap((0, 0, 5, 5), (5, 0, 5, 5))
        assert area == pytest.approx(0.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, -1, 5), (0, 0, 5, 5))

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            bbox_overlap((0, 0, 5, 5), (0, 0, 5, -1))


# ─── TestMaskIou ──────────────────────────────────────────────────────────────

class TestMaskIou:
    def test_identical_masks_iou_one(self):
        m = _square_mask(8)
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m1[:4, :] = 255
        m2[4:, :] = 255
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_empty_masks_iou_zero(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m2 = np.zeros((4, 4), dtype=np.uint8)
        m1[:, :2] = 255  # левая половина
        m2[:, 1:3] = 255  # сдвинута на 1
        iou = mask_iou(m1, m2)
        assert 0.0 < iou < 1.0

    def test_non_2d_mask1_raises(self):
        with pytest.raises(ValueError):
            mask_iou(np.ones((3, 3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mask_iou(np.ones((4, 4), dtype=np.uint8), np.ones((5, 5), dtype=np.uint8))

    def test_iou_in_range(self):
        m1 = _square_mask(8)
        m2 = _square_mask(8)
        m2[:2, :] = 0
        iou = mask_iou(m1, m2)
        assert 0.0 <= iou <= 1.0


# ─── TestCheckPairOverlap ─────────────────────────────────────────────────────

class TestCheckPairOverlap:
    def test_complete_overlap(self):
        m = _square_mask(8)
        rec = check_pair_overlap(0, 1, m, m, canvas_size=(16, 16))
        assert rec.iou == pytest.approx(1.0)
        assert rec.overlap_area > 0

    def test_no_overlap_disjoint_positions(self):
        m = _square_mask(8)
        rec = check_pair_overlap(
            0, 1, m, m, canvas_size=(32, 32),
            pos1=(0, 0), pos2=(16, 16)
        )
        assert rec.iou == pytest.approx(0.0)
        assert rec.overlap_area == pytest.approx(0.0)

    def test_negative_idx_raises(self):
        m = _square_mask(8)
        with pytest.raises(ValueError):
            check_pair_overlap(-1, 0, m, m, canvas_size=(16, 16))

    def test_non_2d_mask_raises(self):
        m2d = _square_mask(8)
        m3d = np.ones((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            check_pair_overlap(0, 1, m3d, m2d, canvas_size=(16, 16))

    def test_returns_overlap_record(self):
        m = _square_mask(8)
        rec = check_pair_overlap(0, 1, m, m, canvas_size=(16, 16))
        assert isinstance(rec, OverlapRecord)

    def test_idx_preserved(self):
        m = _square_mask(8)
        rec = check_pair_overlap(3, 7, m, m, canvas_size=(16, 16))
        assert rec.idx1 == 3 and rec.idx2 == 7


# ─── TestValidateAssembly ─────────────────────────────────────────────────────

class TestValidateAssembly:
    def _non_overlapping(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m1[:] = 255
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[:] = 255
        masks = [m1, m2]
        positions = [(0, 0), (0, 16)]  # разнесены
        return masks, positions

    def test_no_overlap_valid(self):
        masks, positions = self._non_overlapping()
        rpt = validate_assembly(masks, positions, canvas_size=(32, 32))
        assert rpt.is_valid is True
        assert rpt.n_overlaps == 0

    def test_full_overlap_detected(self):
        m = _square_mask(8)
        rpt = validate_assembly([m, m], [(0, 0), (0, 0)], canvas_size=(16, 16))
        assert rpt.n_overlaps > 0
        assert rpt.is_valid is False

    def test_length_mismatch_raises(self):
        m = _square_mask(8)
        with pytest.raises(ValueError):
            validate_assembly([m, m], [(0, 0)], canvas_size=(16, 16))

    def test_negative_iou_threshold_raises(self):
        m = _square_mask(8)
        with pytest.raises(ValueError):
            validate_assembly([m], [(0, 0)], canvas_size=(16, 16), iou_threshold=-0.1)

    def test_returns_validation_report(self):
        m = _square_mask(8)
        rpt = validate_assembly([m], [(0, 0)], canvas_size=(16, 16))
        assert isinstance(rpt, ValidationReport)

    def test_total_area_nonnegative(self):
        m = _square_mask(8)
        rpt = validate_assembly([m, m], [(0, 0), (0, 0)], canvas_size=(16, 16))
        assert rpt.total_area >= 0.0


# ─── TestOverlapAreaMatrix ────────────────────────────────────────────────────

class TestOverlapAreaMatrix:
    def test_shape(self):
        masks = [_square_mask(8) for _ in range(3)]
        positions = [(0, 0), (0, 0), (0, 16)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(32, 32))
        assert mat.shape == (3, 3)

    def test_symmetric(self):
        masks = [_square_mask(8) for _ in range(3)]
        positions = [(0, 0), (4, 4), (0, 20)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(32, 32))
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_diagonal_zero(self):
        masks = [_square_mask(8) for _ in range(2)]
        positions = [(0, 0), (0, 0)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(16, 16))
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_dtype_float64(self):
        masks = [_square_mask(8) for _ in range(2)]
        positions = [(0, 0), (0, 0)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(16, 16))
        assert mat.dtype == np.float64


# ─── TestBatchValidate ────────────────────────────────────────────────────────

class TestBatchValidate:
    def test_returns_list(self):
        m = _square_mask(8)
        assemblies = [([m, m], [(0, 0), (0, 16)], (32, 32))]
        result = batch_validate(assemblies)
        assert isinstance(result, list)

    def test_correct_length(self):
        m = _square_mask(8)
        assemblies = [
            ([m], [(0, 0)], (16, 16)),
            ([m, m], [(0, 0), (0, 0)], (16, 16)),
        ]
        result = batch_validate(assemblies)
        assert len(result) == 2

    def test_empty_list(self):
        result = batch_validate([])
        assert result == []

    def test_each_validation_report(self):
        m = _square_mask(8)
        assemblies = [([m, m], [(0, 0), (0, 0)], (16, 16))]
        result = batch_validate(assemblies)
        assert all(isinstance(r, ValidationReport) for r in result)
