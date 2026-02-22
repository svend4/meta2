"""Extra tests for puzzle_reconstruction.verification.overlap_validator."""
import pytest
import numpy as np

from puzzle_reconstruction.verification.overlap_validator import (
    OverlapRecord,
    ValidationReport,
    batch_validate,
    bbox_overlap,
    check_pair_overlap,
    mask_iou,
    overlap_area_matrix,
    validate_assembly,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mask(h=16, w=16, fill=True):
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[:] = 255
    return m


def _half_mask(h=8, w=8, top=True):
    m = np.zeros((h, w), dtype=np.uint8)
    if top:
        m[: h // 2, :] = 255
    else:
        m[h // 2 :, :] = 255
    return m


# ─── OverlapRecord extras ─────────────────────────────────────────────────────

class TestOverlapRecordExtra:
    def test_repr_is_string(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=5.0, iou=0.3)
        assert isinstance(repr(r), str)

    def test_both_idx_zero_valid(self):
        r = OverlapRecord(idx1=0, idx2=0, overlap_area=0.0, iou=0.0)
        assert r.pair == (0, 0)

    def test_large_overlap_area_valid(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=1e6, iou=1.0)
        assert r.overlap_area == pytest.approx(1e6)

    def test_pair_preserves_order(self):
        r = OverlapRecord(idx1=5, idx2=3, overlap_area=0.0, iou=0.0)
        assert r.pair == (5, 3)

    def test_iou_zero_valid(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=0.0)
        assert r.iou == pytest.approx(0.0)

    def test_iou_one_valid(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=100.0, iou=1.0)
        assert r.iou == pytest.approx(1.0)

    def test_overlap_area_zero_valid(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=0.0)
        assert r.overlap_area == pytest.approx(0.0)


# ─── ValidationReport extras ──────────────────────────────────────────────────

class TestValidationReportExtra:
    def test_len_zero_when_no_overlaps(self):
        rpt = ValidationReport(overlaps=[], n_overlaps=0,
                               total_area=0.0, max_iou=0.0, is_valid=True)
        assert len(rpt) == 0

    def test_is_valid_false_when_overlaps_present(self):
        rec = OverlapRecord(idx1=0, idx2=1, overlap_area=10.0, iou=0.5)
        rpt = ValidationReport(overlaps=[rec], n_overlaps=1,
                               total_area=10.0, max_iou=0.5, is_valid=False)
        assert rpt.is_valid is False

    def test_total_area_zero_valid(self):
        rpt = ValidationReport(overlaps=[], n_overlaps=0,
                               total_area=0.0, max_iou=0.0, is_valid=True)
        assert rpt.total_area == pytest.approx(0.0)

    def test_max_iou_stored(self):
        rpt = ValidationReport(overlaps=[], n_overlaps=0,
                               total_area=0.0, max_iou=0.75, is_valid=True)
        assert rpt.max_iou == pytest.approx(0.75)

    def test_n_overlaps_matches_len(self):
        rec = OverlapRecord(idx1=0, idx2=1, overlap_area=5.0, iou=0.2)
        rpt = ValidationReport(overlaps=[rec], n_overlaps=1,
                               total_area=5.0, max_iou=0.2, is_valid=False)
        assert len(rpt) == rpt.n_overlaps


# ─── bbox_overlap extras ──────────────────────────────────────────────────────

class TestBboxOverlapExtra:
    def test_contained_inside(self):
        # Inner box fully inside outer
        area = bbox_overlap((0, 0, 20, 20), (5, 5, 5, 5))
        assert area == pytest.approx(25.0)

    def test_adjacent_no_overlap(self):
        area = bbox_overlap((0, 0, 10, 10), (10, 0, 10, 10))
        assert area == pytest.approx(0.0)

    def test_narrow_strip_overlap(self):
        # (0,0,10,10) ∩ (9,0,10,10) = 1×10 = 10
        area = bbox_overlap((0, 0, 10, 10), (9, 0, 10, 10))
        assert area == pytest.approx(10.0)

    def test_zero_width_box(self):
        area = bbox_overlap((0, 0, 0, 10), (0, 0, 10, 10))
        assert area == pytest.approx(0.0)

    def test_zero_height_box(self):
        area = bbox_overlap((0, 0, 10, 0), (0, 0, 10, 10))
        assert area == pytest.approx(0.0)

    def test_symmetry(self):
        a = bbox_overlap((0, 0, 10, 10), (5, 5, 10, 10))
        b = bbox_overlap((5, 5, 10, 10), (0, 0, 10, 10))
        assert a == pytest.approx(b)

    def test_large_boxes(self):
        area = bbox_overlap((0, 0, 1000, 1000), (0, 0, 1000, 1000))
        assert area == pytest.approx(1_000_000.0)


# ─── mask_iou extras ──────────────────────────────────────────────────────────

class TestMaskIouExtra:
    def test_single_pixel_overlap(self):
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m2 = np.zeros((4, 4), dtype=np.uint8)
        m1[0, 0] = 255
        m2[0, 0] = 255
        assert mask_iou(m1, m2) == pytest.approx(1.0)

    def test_one_empty_mask(self):
        m1 = _mask(8, 8, fill=True)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_half_overlap(self):
        m1 = _half_mask(top=True)
        m2 = _half_mask(top=False)
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_three_quarter_overlap(self):
        m1 = _mask(4, 4)      # 16 px
        m2 = np.zeros((4, 4), dtype=np.uint8)
        m2[:3, :] = 255        # 12 px; union=16, inter=12
        iou = mask_iou(m1, m2)
        assert iou == pytest.approx(12 / 16, abs=1e-6)

    def test_non_square_identical(self):
        m = _mask(8, 16)
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_result_in_range(self):
        m1 = _mask(10, 10)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[:7, :7] = 255
        iou = mask_iou(m1, m2)
        assert 0.0 <= iou <= 1.0

    def test_non_2d_mask2_raises(self):
        with pytest.raises(ValueError):
            mask_iou(np.ones((4, 4), dtype=np.uint8),
                     np.ones((4, 4, 3), dtype=np.uint8))


# ─── check_pair_overlap extras ────────────────────────────────────────────────

class TestCheckPairOverlapExtra:
    def test_same_position_full_iou(self):
        m = _mask(8, 8)
        rec = check_pair_overlap(0, 1, m, m, canvas_size=(16, 16))
        assert rec.iou == pytest.approx(1.0)

    def test_far_apart_zero_iou(self):
        m = _mask(8, 8)
        rec = check_pair_overlap(0, 1, m, m,
                                  canvas_size=(100, 100),
                                  pos1=(0, 0), pos2=(90, 90))
        assert rec.iou == pytest.approx(0.0)

    def test_ids_stored(self):
        m = _mask(8, 8)
        rec = check_pair_overlap(4, 9, m, m, canvas_size=(16, 16))
        assert rec.idx1 == 4
        assert rec.idx2 == 9

    def test_overlap_area_nonneg(self):
        m = _mask(8, 8)
        rec = check_pair_overlap(0, 1, m, m, canvas_size=(32, 32),
                                  pos1=(0, 0), pos2=(4, 4))
        assert rec.overlap_area >= 0.0

    def test_iou_in_range(self):
        m = _mask(8, 8)
        rec = check_pair_overlap(0, 1, m, m, canvas_size=(32, 32),
                                  pos1=(0, 0), pos2=(4, 4))
        assert 0.0 <= rec.iou <= 1.0


# ─── validate_assembly extras ─────────────────────────────────────────────────

class TestValidateAssemblyExtra:
    def test_three_non_overlapping(self):
        m = _mask(8, 8)
        masks = [m, m, m]
        positions = [(0, 0), (0, 100), (0, 200)]
        rpt = validate_assembly(masks, positions, canvas_size=(300, 300))
        assert rpt.is_valid is True
        assert rpt.n_overlaps == 0

    def test_three_all_overlapping(self):
        m = _mask(8, 8)
        rpt = validate_assembly([m, m, m], [(0, 0)] * 3, canvas_size=(16, 16))
        assert rpt.n_overlaps > 0
        assert rpt.is_valid is False

    def test_single_mask_always_valid(self):
        m = _mask(8, 8)
        rpt = validate_assembly([m], [(0, 0)], canvas_size=(16, 16))
        assert rpt.is_valid is True

    def test_total_area_nonneg_nonoverlap(self):
        m = _mask(8, 8)
        rpt = validate_assembly([m, m], [(0, 0), (0, 100)],
                                 canvas_size=(200, 200))
        assert rpt.total_area >= 0.0

    def test_iou_threshold_high_marks_valid(self):
        # High threshold → small overlap ignored
        m = _mask(8, 8)
        rpt = validate_assembly([m, m], [(0, 0), (4, 4)],
                                 canvas_size=(32, 32),
                                 iou_threshold=0.999)
        assert rpt.is_valid is True

    def test_empty_masks_raises_or_valid(self):
        # An empty assembly with 0 masks could be valid (no overlaps)
        try:
            rpt = validate_assembly([], [], canvas_size=(16, 16))
            assert rpt.is_valid is True
        except (ValueError, IndexError):
            pass  # Acceptable to raise on empty input


# ─── overlap_area_matrix extras ───────────────────────────────────────────────

class TestOverlapAreaMatrixExtra:
    def test_single_mask_1x1(self):
        m = _mask(8, 8)
        mat = overlap_area_matrix([m], [(0, 0)], canvas_size=(16, 16))
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_all_diag_zero(self):
        masks = [_mask(8, 8) for _ in range(4)]
        positions = [(i * 10, 0) for i in range(4)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(50, 50))
        for i in range(4):
            assert mat[i, i] == pytest.approx(0.0)

    def test_values_nonneg(self):
        masks = [_mask(8, 8) for _ in range(3)]
        positions = [(0, 0), (4, 4), (0, 20)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(32, 32))
        assert np.all(mat >= 0.0)

    def test_no_overlap_zero_off_diagonal(self):
        masks = [_mask(8, 8), _mask(8, 8)]
        positions = [(0, 0), (0, 100)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(200, 200))
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[1, 0] == pytest.approx(0.0)

    def test_symmetric_3x3(self):
        masks = [_mask(8, 8) for _ in range(3)]
        positions = [(0, 0), (4, 0), (8, 0)]
        mat = overlap_area_matrix(masks, positions, canvas_size=(32, 32))
        np.testing.assert_array_almost_equal(mat, mat.T)


# ─── batch_validate extras ────────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_three_assemblies(self):
        m = _mask(8, 8)
        assemblies = [
            ([m, m], [(0, 0), (0, 100)], (200, 200)),
            ([m], [(0, 0)], (16, 16)),
            ([m, m], [(0, 0), (0, 0)], (16, 16)),
        ]
        result = batch_validate(assemblies)
        assert len(result) == 3

    def test_one_valid_one_invalid(self):
        m = _mask(8, 8)
        assemblies = [
            ([m, m], [(0, 0), (0, 100)], (200, 200)),  # valid
            ([m, m], [(0, 0), (0, 0)], (16, 16)),       # invalid
        ]
        result = batch_validate(assemblies)
        assert result[0].is_valid is True
        assert result[1].is_valid is False

    def test_all_validation_reports(self):
        m = _mask(8, 8)
        assemblies = [([m], [(0, 0)], (16, 16)) for _ in range(4)]
        result = batch_validate(assemblies)
        for r in result:
            assert isinstance(r, ValidationReport)

    def test_single_assembly(self):
        m = _mask(8, 8)
        result = batch_validate([([m], [(0, 0)], (16, 16))])
        assert len(result) == 1
        assert result[0].is_valid is True
