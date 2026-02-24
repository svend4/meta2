"""Extra tests for puzzle_reconstruction/utils/mask_layout_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.mask_layout_utils import (
    MaskOpRecord,
    MaskCoverageRecord,
    FragmentPlacementRecord,
    LayoutDiffRecord,
    LayoutScoreRecord,
    FeatureSelectionRecord,
    PcaRecord,
    make_mask_coverage_record,
    make_layout_diff_record,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mask(h=10, w=10, fill=True) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[:] = 255
    return m


# ─── MaskOpRecord ─────────────────────────────────────────────────────────────

class TestMaskOpRecordExtra:
    def test_stores_operation(self):
        r = MaskOpRecord(operation="erode", input_shape=(10, 10),
                          n_nonzero_before=80, n_nonzero_after=60)
        assert r.operation == "erode"

    def test_invalid_operation_raises(self):
        with pytest.raises(ValueError):
            MaskOpRecord(operation="blur", input_shape=(10, 10),
                          n_nonzero_before=50, n_nonzero_after=40)

    def test_negative_before_raises(self):
        with pytest.raises(ValueError):
            MaskOpRecord(operation="erode", input_shape=(10, 10),
                          n_nonzero_before=-1, n_nonzero_after=0)

    def test_area_change(self):
        r = MaskOpRecord(operation="dilate", input_shape=(10, 10),
                          n_nonzero_before=50, n_nonzero_after=70)
        assert r.area_change == 20

    def test_coverage_ratio(self):
        r = MaskOpRecord(operation="erode", input_shape=(10, 10),
                          n_nonzero_before=100, n_nonzero_after=50)
        assert r.coverage_ratio == pytest.approx(0.5)

    def test_coverage_zero_shape(self):
        r = MaskOpRecord(operation="invert", input_shape=(0, 0),
                          n_nonzero_before=0, n_nonzero_after=0)
        assert r.coverage_ratio == pytest.approx(0.0)


# ─── MaskCoverageRecord ───────────────────────────────────────────────────────

class TestMaskCoverageRecordExtra:
    def test_stores_n_masks(self):
        r = MaskCoverageRecord(n_masks=3, canvas_shape=(10, 10),
                                n_covered_pixels=50, n_total_pixels=100)
        assert r.n_masks == 3

    def test_coverage_ratio(self):
        r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                                n_covered_pixels=50, n_total_pixels=100)
        assert r.coverage_ratio == pytest.approx(0.5)

    def test_is_fully_covered_true(self):
        r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                                n_covered_pixels=100, n_total_pixels=100)
        assert r.is_fully_covered is True

    def test_is_fully_covered_false(self):
        r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                                n_covered_pixels=50, n_total_pixels=100)
        assert r.is_fully_covered is False

    def test_zero_total_pixels(self):
        r = MaskCoverageRecord(n_masks=0, canvas_shape=(0, 0),
                                n_covered_pixels=0, n_total_pixels=0)
        assert r.coverage_ratio == pytest.approx(0.0)


# ─── FragmentPlacementRecord ──────────────────────────────────────────────────

class TestFragmentPlacementRecordExtra:
    def test_stores_n_total(self):
        r = FragmentPlacementRecord(n_total=10, n_placed=7)
        assert r.n_total == 10

    def test_negative_n_total_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacementRecord(n_total=-1, n_placed=0)

    def test_n_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacementRecord(n_total=5, n_placed=6)

    def test_coverage(self):
        r = FragmentPlacementRecord(n_total=10, n_placed=7)
        assert r.coverage == pytest.approx(0.7)

    def test_n_missing(self):
        r = FragmentPlacementRecord(n_total=10, n_placed=7)
        assert r.n_missing == 3

    def test_zero_total_coverage_one(self):
        r = FragmentPlacementRecord(n_total=0, n_placed=0)
        assert r.coverage == pytest.approx(1.0)


# ─── LayoutDiffRecord ─────────────────────────────────────────────────────────

class TestLayoutDiffRecordExtra:
    def test_stores_n_fragments(self):
        r = LayoutDiffRecord(n_fragments=5, mean_shift=0.1, max_shift=0.5, n_moved=2)
        assert r.n_fragments == 5

    def test_negative_mean_shift_raises(self):
        with pytest.raises(ValueError):
            LayoutDiffRecord(n_fragments=3, mean_shift=-0.1, max_shift=0.0, n_moved=0)

    def test_is_stable_true(self):
        r = LayoutDiffRecord(n_fragments=3, mean_shift=0.0, max_shift=0.0, n_moved=0)
        assert r.is_stable is True

    def test_is_stable_false(self):
        r = LayoutDiffRecord(n_fragments=3, mean_shift=0.1, max_shift=0.5, n_moved=1)
        assert r.is_stable is False


# ─── LayoutScoreRecord ────────────────────────────────────────────────────────

class TestLayoutScoreRecordExtra:
    def test_score_improvement(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.5, final_score=0.8)
        assert r.score_improvement == pytest.approx(0.3)

    def test_converged_true(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.5, final_score=0.5)
        assert r.converged is True

    def test_converged_false(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.5, final_score=0.8)
        assert r.converged is False


# ─── FeatureSelectionRecord ───────────────────────────────────────────────────

class TestFeatureSelectionRecordExtra:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FeatureSelectionRecord(method="unknown", n_input_features=10,
                                    n_selected_features=5)

    def test_selected_exceeds_input_raises(self):
        with pytest.raises(ValueError):
            FeatureSelectionRecord(method="variance", n_input_features=5,
                                    n_selected_features=10)

    def test_selection_ratio(self):
        r = FeatureSelectionRecord(method="variance", n_input_features=10,
                                    n_selected_features=4)
        assert r.selection_ratio == pytest.approx(0.4)

    def test_zero_input_ratio_zero(self):
        r = FeatureSelectionRecord(method="pca", n_input_features=0,
                                    n_selected_features=0)
        assert r.selection_ratio == pytest.approx(0.0)


# ─── PcaRecord ────────────────────────────────────────────────────────────────

class TestPcaRecordExtra:
    def test_total_variance_explained(self):
        r = PcaRecord(n_input_features=10, n_components=3,
                       explained_variance_ratio=[0.5, 0.3, 0.1])
        assert r.total_variance_explained == pytest.approx(0.9)

    def test_dominant_component_ratio(self):
        r = PcaRecord(n_input_features=10, n_components=3,
                       explained_variance_ratio=[0.6, 0.2, 0.1])
        assert r.dominant_component_ratio == pytest.approx(0.6)

    def test_empty_variance_dominant_zero(self):
        r = PcaRecord(n_input_features=5, n_components=0)
        assert r.dominant_component_ratio == pytest.approx(0.0)


# ─── make_mask_coverage_record ────────────────────────────────────────────────

class TestMakeMaskCoverageRecordExtra:
    def test_returns_record(self):
        masks = [_mask(10, 10)]
        r = make_mask_coverage_record(masks, (10, 10))
        assert isinstance(r, MaskCoverageRecord)

    def test_empty_masks(self):
        r = make_mask_coverage_record([], (10, 10))
        assert r.n_covered_pixels == 0

    def test_full_mask_coverage(self):
        masks = [_mask(10, 10)]
        r = make_mask_coverage_record(masks, (10, 10))
        assert r.coverage_ratio == pytest.approx(1.0)


# ─── make_layout_diff_record ──────────────────────────────────────────────────

class TestMakeLayoutDiffRecordExtra:
    def test_returns_record(self):
        d = {"n_fragments": 5, "mean_shift": 0.2, "max_shift": 0.8, "n_moved": 2}
        r = make_layout_diff_record(d)
        assert isinstance(r, LayoutDiffRecord)

    def test_values_stored(self):
        d = {"n_fragments": 3, "mean_shift": 0.1, "max_shift": 0.5, "n_moved": 1}
        r = make_layout_diff_record(d)
        assert r.n_fragments == 3 and r.mean_shift == pytest.approx(0.1)

    def test_empty_dict_defaults(self):
        r = make_layout_diff_record({})
        assert r.n_fragments == 0 and r.mean_shift == pytest.approx(0.0)
