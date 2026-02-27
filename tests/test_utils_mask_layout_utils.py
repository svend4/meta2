"""Tests for puzzle_reconstruction.utils.mask_layout_utils"""
import numpy as np
import pytest
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

np.random.seed(42)


# ─── MaskOpRecord ─────────────────────────────────────────────────────────────

def test_mask_op_record_valid():
    r = MaskOpRecord(operation="erode", input_shape=(100, 100),
                     n_nonzero_before=5000, n_nonzero_after=4000)
    assert r.area_change == -1000
    assert r.coverage_ratio == pytest.approx(0.4, abs=1e-9)


def test_mask_op_record_invalid_operation():
    with pytest.raises(ValueError):
        MaskOpRecord(operation="unknown_op", input_shape=(100, 100),
                     n_nonzero_before=0, n_nonzero_after=0)


def test_mask_op_record_invalid_nonzero():
    with pytest.raises(ValueError):
        MaskOpRecord(operation="dilate", input_shape=(100, 100),
                     n_nonzero_before=-1, n_nonzero_after=0)


def test_mask_op_record_all_valid_ops():
    for op in ["erode", "dilate", "invert", "and", "or", "xor", "crop"]:
        r = MaskOpRecord(operation=op, input_shape=(10, 10),
                         n_nonzero_before=10, n_nonzero_after=10)
        assert r.operation == op


def test_mask_op_record_area_change_positive():
    r = MaskOpRecord(operation="dilate", input_shape=(50, 50),
                     n_nonzero_before=100, n_nonzero_after=200)
    assert r.area_change == 100


# ─── MaskCoverageRecord ───────────────────────────────────────────────────────

def test_mask_coverage_record_ratio():
    r = MaskCoverageRecord(n_masks=2, canvas_shape=(100, 100),
                           n_covered_pixels=7000, n_total_pixels=10000)
    assert r.coverage_ratio == pytest.approx(0.7, abs=1e-9)


def test_mask_coverage_record_fully_covered():
    r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                           n_covered_pixels=100, n_total_pixels=100)
    assert r.is_fully_covered is True


def test_mask_coverage_record_not_fully_covered():
    r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                           n_covered_pixels=50, n_total_pixels=100)
    assert r.is_fully_covered is False


def test_mask_coverage_record_zero_total():
    r = MaskCoverageRecord(n_masks=0, canvas_shape=(0, 0),
                           n_covered_pixels=0, n_total_pixels=0)
    assert r.coverage_ratio == 0.0


# ─── FragmentPlacementRecord ──────────────────────────────────────────────────

def test_fragment_placement_record_coverage():
    r = FragmentPlacementRecord(n_total=10, n_placed=7)
    assert r.coverage == pytest.approx(0.7, abs=1e-9)
    assert r.n_missing == 3


def test_fragment_placement_record_all_placed():
    r = FragmentPlacementRecord(n_total=5, n_placed=5)
    assert r.coverage == 1.0
    assert r.n_missing == 0


def test_fragment_placement_record_zero_total():
    r = FragmentPlacementRecord(n_total=0, n_placed=0)
    assert r.coverage == 1.0


def test_fragment_placement_record_invalid_negative():
    with pytest.raises(ValueError):
        FragmentPlacementRecord(n_total=-1, n_placed=0)


def test_fragment_placement_record_exceeds_total():
    with pytest.raises(ValueError):
        FragmentPlacementRecord(n_total=5, n_placed=6)


# ─── LayoutDiffRecord ─────────────────────────────────────────────────────────

def test_layout_diff_record_is_stable():
    r = LayoutDiffRecord(n_fragments=5, mean_shift=0.0, max_shift=0.0, n_moved=0)
    assert r.is_stable is True


def test_layout_diff_record_not_stable():
    r = LayoutDiffRecord(n_fragments=5, mean_shift=1.5, max_shift=3.0, n_moved=2)
    assert r.is_stable is False


def test_layout_diff_record_invalid_mean_shift():
    with pytest.raises(ValueError):
        LayoutDiffRecord(n_fragments=3, mean_shift=-1.0, max_shift=0.0, n_moved=0)


# ─── LayoutScoreRecord ────────────────────────────────────────────────────────

def test_layout_score_record_improvement():
    r = LayoutScoreRecord(n_pairs=10, initial_score=0.5, final_score=0.8)
    assert abs(r.score_improvement - 0.3) < 1e-9


def test_layout_score_record_converged():
    r = LayoutScoreRecord(n_pairs=5, initial_score=0.7, final_score=0.7)
    assert r.converged is True


def test_layout_score_record_not_converged():
    r = LayoutScoreRecord(n_pairs=5, initial_score=0.5, final_score=0.8)
    assert r.converged is False


# ─── FeatureSelectionRecord ───────────────────────────────────────────────────

def test_feature_selection_record_ratio():
    r = FeatureSelectionRecord(method="variance", n_input_features=100, n_selected_features=20)
    assert r.selection_ratio == pytest.approx(0.2, abs=1e-9)


def test_feature_selection_record_invalid_method():
    with pytest.raises(ValueError):
        FeatureSelectionRecord(method="unknown", n_input_features=10, n_selected_features=5)


def test_feature_selection_record_exceeds_input():
    with pytest.raises(ValueError):
        FeatureSelectionRecord(method="pca", n_input_features=10, n_selected_features=15)


def test_feature_selection_record_zero_input():
    r = FeatureSelectionRecord(method="rank", n_input_features=0, n_selected_features=0)
    assert r.selection_ratio == 0.0


# ─── PcaRecord ────────────────────────────────────────────────────────────────

def test_pca_record_total_variance():
    r = PcaRecord(n_input_features=50, n_components=3,
                  explained_variance_ratio=[0.5, 0.3, 0.1])
    assert abs(r.total_variance_explained - 0.9) < 1e-9


def test_pca_record_dominant_component():
    r = PcaRecord(n_input_features=50, n_components=2,
                  explained_variance_ratio=[0.6, 0.3])
    assert r.dominant_component_ratio == pytest.approx(0.6, abs=1e-9)


def test_pca_record_empty_variance():
    r = PcaRecord(n_input_features=50, n_components=0)
    assert r.total_variance_explained == 0.0
    assert r.dominant_component_ratio == 0.0


# ─── make_mask_coverage_record ────────────────────────────────────────────────

def test_make_mask_coverage_record_single():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    r = make_mask_coverage_record([mask], canvas_shape=(50, 50))
    assert r.n_masks == 1
    assert r.n_covered_pixels == 400  # 20x20
    assert r.n_total_pixels == 2500


def test_make_mask_coverage_record_empty():
    r = make_mask_coverage_record([], canvas_shape=(10, 10))
    assert r.n_covered_pixels == 0


def test_make_mask_coverage_record_union():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m1[0:5, 0:5] = 255  # 25 pixels
    m2[0:5, 0:5] = 255  # same 25 pixels → union = 25
    r = make_mask_coverage_record([m1, m2], canvas_shape=(10, 10))
    assert r.n_covered_pixels == 25


# ─── make_layout_diff_record ──────────────────────────────────────────────────

def test_make_layout_diff_record():
    d = {"n_fragments": 5, "mean_shift": 1.5, "max_shift": 3.0, "n_moved": 2}
    r = make_layout_diff_record(d, label="test")
    assert r.n_fragments == 5
    assert r.mean_shift == pytest.approx(1.5)
    assert r.label == "test"


def test_make_layout_diff_record_defaults():
    r = make_layout_diff_record({})
    assert r.n_fragments == 0
    assert r.mean_shift == 0.0
    assert r.is_stable is True
