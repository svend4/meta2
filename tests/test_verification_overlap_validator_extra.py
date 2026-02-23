"""Extra tests for puzzle_reconstruction/verification/overlap_validator.py."""
from __future__ import annotations

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
        return np.full((h, w), 255, dtype=np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _partial(h=32, w=32, frac=0.5):
    m = np.zeros((h, w), dtype=np.uint8)
    m[:int(h * frac), :] = 255
    return m


# ─── OverlapRecord (extra) ────────────────────────────────────────────────────

class TestOverlapRecordExtra:
    def test_large_overlap_area_ok(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=1e6, iou=0.9)
        assert r.overlap_area == pytest.approx(1e6)

    def test_pair_reversed_different(self):
        r1 = OverlapRecord(idx1=0, idx2=1, overlap_area=10.0, iou=0.5)
        r2 = OverlapRecord(idx1=1, idx2=0, overlap_area=10.0, iou=0.5)
        # pair tuples are different
        assert r1.pair != r2.pair

    def test_iou_at_boundaries_valid(self):
        for iou in (0.0, 0.01, 0.5, 0.99, 1.0):
            r = OverlapRecord(idx1=0, idx2=1, overlap_area=0.0, iou=iou)
            assert r.iou == pytest.approx(iou)

    def test_large_indices_ok(self):
        r = OverlapRecord(idx1=999, idx2=1000, overlap_area=0.0, iou=0.0)
        assert r.idx1 == 999
        assert r.idx2 == 1000

    def test_params_extra_keys(self):
        r = OverlapRecord(idx1=0, idx2=1, overlap_area=5.0, iou=0.3,
                          params={"method": "pixel"})
        assert r.params["method"] == "pixel"

    def test_same_idx_allowed(self):
        r = OverlapRecord(idx1=3, idx2=3, overlap_area=50.0, iou=1.0)
        assert r.pair == (3, 3)


# ─── ValidationReport (extra) ────────────────────────────────────────────────

class TestValidationReportExtra:
    def _rec(self):
        return OverlapRecord(idx1=0, idx2=1, overlap_area=5.0, iou=0.3)

    def test_max_iou_zero_for_no_overlaps(self):
        r = ValidationReport(overlaps=[], n_overlaps=0,
                              total_area=0.0, max_iou=0.0, is_valid=True)
        assert r.max_iou == pytest.approx(0.0)

    def test_max_iou_large_ok(self):
        r = ValidationReport(overlaps=[], n_overlaps=0,
                              total_area=0.0, max_iou=1.0, is_valid=True)
        assert r.max_iou == pytest.approx(1.0)

    def test_overlaps_list_len(self):
        recs = [self._rec(), self._rec()]
        r = ValidationReport(overlaps=recs, n_overlaps=2,
                              total_area=10.0, max_iou=0.3, is_valid=False)
        assert len(r.overlaps) == 2

    def test_is_valid_true(self):
        r = ValidationReport(overlaps=[], n_overlaps=0,
                              total_area=0.0, max_iou=0.0, is_valid=True)
        assert r.is_valid is True

    def test_max_iou_zero_ok(self):
        r = ValidationReport(overlaps=[], n_overlaps=0,
                              total_area=0.0, max_iou=0.0, is_valid=True)
        assert r.max_iou == pytest.approx(0.0)

    def test_total_area_large_ok(self):
        r = ValidationReport(overlaps=[], n_overlaps=0,
                              total_area=1e8, max_iou=0.0, is_valid=True)
        assert r.total_area == pytest.approx(1e8)


# ─── bbox_overlap (extra) ─────────────────────────────────────────────────────

class TestBboxOverlapExtra:
    def test_one_inside_other(self):
        # inner box (2,2,4,4) inside outer (0,0,10,10) → overlap = 16
        result = bbox_overlap((0, 0, 10, 10), (2, 2, 4, 4))
        assert result == pytest.approx(16.0)

    def test_large_boxes_full_overlap(self):
        result = bbox_overlap((0, 0, 1000, 1000), (0, 0, 1000, 1000))
        assert result == pytest.approx(1e6)

    def test_zero_height_both_boxes(self):
        result = bbox_overlap((0, 0, 10, 0), (0, 0, 10, 0))
        assert result == pytest.approx(0.0)

    def test_symmetry_in_position(self):
        r1 = bbox_overlap((0, 0, 10, 10), (5, 0, 10, 10))
        r2 = bbox_overlap((5, 0, 10, 10), (0, 0, 10, 10))
        assert r1 == pytest.approx(r2)

    def test_partial_overlap_exact(self):
        # (0,0,8,8) and (4,4,8,8) → overlap = 4×4 = 16
        result = bbox_overlap((0, 0, 8, 8), (4, 4, 8, 8))
        assert result == pytest.approx(16.0)

    def test_different_box_sizes(self):
        # Large vs small: (0,0,20,20) and (5,5,5,5) → overlap = 25
        result = bbox_overlap((0, 0, 20, 20), (5, 5, 5, 5))
        assert result == pytest.approx(25.0)


# ─── mask_iou (extra) ─────────────────────────────────────────────────────────

class TestMaskIouExtra:
    def test_half_overlap_approx_third(self):
        # m1: full 8x8 (64), m2: left half 8x4 (32)
        # intersection=32, union=64 → 0.5
        m1 = _mask(8, 8, filled=True)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[:, :4] = 255
        result = mask_iou(m1, m2)
        assert result == pytest.approx(0.5)

    def test_nonoverlapping_disjoint(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m1[:4, :] = 255  # top half
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[4:, :] = 255  # bottom half
        result = mask_iou(m1, m2)
        assert result == pytest.approx(0.0)

    def test_binary_values_accepted(self):
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m1[2:6, 2:6] = 1  # binary (not 255)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[2:6, 2:6] = 1
        result = mask_iou(m1, m2)
        # Any non-zero treated as foreground
        assert result > 0.0

    def test_result_float(self):
        m = _mask(8, 8)
        assert isinstance(mask_iou(m, m), float)

    def test_large_masks(self):
        m1 = _mask(256, 256, filled=True)
        m2 = _mask(256, 256, filled=True)
        result = mask_iou(m1, m2)
        assert result == pytest.approx(1.0)

    def test_three_quarter_overlap(self):
        # m1: 8x8 (64), m2: 8x6 left portion (48)
        # intersection=48, union=64 → 0.75
        m1 = _mask(8, 8, filled=True)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[:, :6] = 255
        result = mask_iou(m1, m2)
        assert result == pytest.approx(48 / 64, abs=1e-5)


# ─── check_pair_overlap (extra) ───────────────────────────────────────────────

class TestCheckPairOverlapExtra:
    def test_iou_equals_one_same_mask_same_pos(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (0, 0))
        assert r.iou == pytest.approx(1.0)

    def test_partial_shift_iou_between_0_and_1(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (8, 0))
        assert 0.0 < r.iou < 1.0

    def test_overlap_area_positive_when_overlapping(self):
        m = _mask(16, 16)
        r = check_pair_overlap(0, 1, m, m, (64, 64), (0, 0), (0, 0))
        assert r.overlap_area > 0.0

    def test_idx1_idx2_stored(self):
        m = _mask(8, 8)
        r = check_pair_overlap(4, 9, m, m, (32, 32), (0, 0), (0, 0))
        assert r.idx1 == 4
        assert r.idx2 == 9

    def test_different_mask_sizes(self):
        m1 = _mask(8, 8)
        m2 = _mask(16, 16)
        # Different sizes are acceptable if valid call
        r = check_pair_overlap(0, 1, m1, m2, (64, 64))
        assert isinstance(r, OverlapRecord)

    def test_zero_overlap_iou_zero(self):
        m = _mask(8, 8)
        # Place them far apart
        r = check_pair_overlap(0, 1, m, m, (128, 128), (0, 0), (64, 64))
        assert r.iou == pytest.approx(0.0)


# ─── validate_assembly (extra) ────────────────────────────────────────────────

class TestValidateAssemblyExtra:
    def test_three_non_overlapping_valid(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m, m], [(0, 0), (20, 0), (40, 0)], (64, 64))
        assert r.is_valid is True
        assert r.n_overlaps == 0

    def test_three_all_overlapping_detected(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m, m], [(0, 0), (0, 0), (0, 0)], (64, 64))
        assert r.n_overlaps > 0
        assert r.is_valid is False

    def test_max_iou_le_one(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (0, 0)], (64, 64))
        assert r.max_iou <= 1.0

    def test_total_area_le_sum_of_mask_areas(self):
        m = _mask(16, 16)
        r = validate_assembly([m, m], [(0, 0), (0, 0)], (64, 64))
        max_possible = 16 * 16 * 2
        assert r.total_area <= max_possible

    def test_iou_threshold_1_never_invalid_unless_perfect_overlap(self):
        m = _mask(8, 8)
        r = validate_assembly([m, m], [(0, 0), (4, 0)], (64, 64),
                               iou_threshold=1.0)
        # IoU < 1.0 so should be valid at threshold=1.0
        assert r.is_valid is True

    def test_report_len_equals_n_overlaps(self):
        m = _mask(8, 8)
        r = validate_assembly([m, m, m], [(0, 0), (0, 0), (0, 0)], (64, 64))
        assert len(r) == r.n_overlaps

    def test_large_canvas(self):
        m = _mask(32, 32)
        r = validate_assembly([m, m], [(0, 0), (500, 500)], (1000, 1000))
        assert r.is_valid is True


# ─── overlap_area_matrix (extra) ──────────────────────────────────────────────

class TestOverlapAreaMatrixExtra:
    def test_zero_diagonal(self):
        m = _mask(16, 16)
        mat = overlap_area_matrix([m, m, m], [(0, 0), (32, 0), (64, 0)], (128, 32))
        for i in range(3):
            assert mat[i, i] == pytest.approx(0.0)

    def test_three_fragments_non_overlapping_zero(self):
        m = _mask(16, 16)
        mat = overlap_area_matrix([m, m, m], [(0, 0), (32, 0), (64, 0)], (128, 32))
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[0, 2] == pytest.approx(0.0)

    def test_all_overlapping_positive_off_diag(self):
        m = _mask(16, 16)
        mat = overlap_area_matrix([m, m, m], [(0, 0), (0, 0), (0, 0)], (64, 64))
        assert mat[0, 1] > 0.0
        assert mat[0, 2] > 0.0
        assert mat[1, 2] > 0.0

    def test_values_nonneg(self):
        m = _mask(8, 8)
        mat = overlap_area_matrix([m, m], [(0, 0), (4, 4)], (32, 32))
        assert (mat >= 0.0).all()

    def test_single_fragment_1x1_zero(self):
        m = _mask(8, 8)
        mat = overlap_area_matrix([m], [(0, 0)], (32, 32))
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.0)


# ─── batch_validate (extra) ───────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_single_assembly_valid(self):
        m = _mask(8, 8)
        result = batch_validate([([m], [(0, 0)], (32, 32))])
        assert result[0].is_valid is True

    def test_multiple_assemblies_independent(self):
        m = _mask(8, 8)
        assemblies = [
            ([m], [(0, 0)], (32, 32)),
            ([m, m], [(0, 0), (0, 0)], (32, 32)),
        ]
        result = batch_validate(assemblies)
        assert result[0].is_valid is True
        assert result[1].is_valid is False

    def test_large_batch(self):
        m = _mask(8, 8)
        assemblies = [([m], [(0, 0)], (32, 32))] * 20
        result = batch_validate(assemblies)
        assert len(result) == 20

    def test_all_valid_assemblies(self):
        m = _mask(8, 8)
        assemblies = [([m, m], [(0, 0), (16, 0)], (32, 32))] * 5
        result = batch_validate(assemblies)
        assert all(r.is_valid for r in result)

    def test_iou_threshold_applied_uniformly(self):
        m = _mask(8, 8)
        assemblies = [([m, m], [(0, 0), (4, 0)], (32, 32))] * 3
        results_low = batch_validate(assemblies, iou_threshold=0.0)
        results_high = batch_validate(assemblies, iou_threshold=1.0)
        # At threshold=0 any overlap is a violation; at 1.0 only perfect overlap
        for r_low, r_high in zip(results_low, results_high):
            assert r_low.n_overlaps >= r_high.n_overlaps

    def test_each_result_is_validation_report(self):
        m = _mask(8, 8)
        for r in batch_validate([([m], [(0, 0)], (32, 32))]):
            assert isinstance(r, ValidationReport)
