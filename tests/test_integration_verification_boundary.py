"""
Integration tests for puzzle_reconstruction.verification modules:
  - boundary_validator
  - overlap_validator
  - seam_analyzer
"""
import pytest
import numpy as np

from puzzle_reconstruction.verification.boundary_validator import (
    BoundaryViolation,
    BoundaryReport,
    validate_edge_gap,
    validate_alignment,
    validate_pair,
    validate_all_pairs,
    boundary_quality_score,
)
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
from puzzle_reconstruction.verification.seam_analyzer import (
    SeamAnalysis,
    extract_seam_profiles,
    analyze_seam,
    score_seam_quality,
    batch_analyze_seams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_mask(h: int = 50, w: int = 50, val: int = 255) -> np.ndarray:
    """Return a filled binary mask of shape (h, w)."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[:, :] = val
    return m


def _patch_mask(h: int = 50, w: int = 50, r0=10, r1=30, c0=10, c1=30) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1, c0:c1] = 255
    return m


def _solid_image(h: int = 50, w: int = 50, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _random_image(h: int = 50, w: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ===========================================================================
# 1. TestBoundaryValidator
# ===========================================================================

class TestBoundaryValidator:
    """Tests for validate_edge_gap, validate_alignment, validate_pair,
    validate_all_pairs, and boundary_quality_score."""

    # --- validate_edge_gap ---

    def test_edge_gap_returns_none_when_valid(self):
        """Small gap within tolerance → None."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=52.0)
        assert result is None

    def test_edge_gap_returns_boundary_violation_on_large_gap(self):
        """Gap exceeding max_gap → BoundaryViolation with type 'gap'."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=70.0)
        assert isinstance(result, BoundaryViolation)
        assert result.violation_type == "gap"

    def test_edge_gap_returns_boundary_violation_on_overlap(self):
        """Overlap exceeding max_overlap → BoundaryViolation with type 'overlap'."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=30.0)
        assert isinstance(result, BoundaryViolation)
        assert result.violation_type == "overlap"

    def test_edge_gap_exact_boundary_is_valid(self):
        """Pieces touching exactly (gap=0) → no violation."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=50.0)
        assert result is None

    def test_boundary_violation_has_severity(self):
        """BoundaryViolation must expose a numeric severity attribute."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=30.0)
        assert result is not None
        assert isinstance(result.severity, (int, float))
        assert result.severity > 0

    # --- validate_alignment ---

    def test_alignment_returns_none_within_tolerance(self):
        """Small tilt within max_tilt_deg → None."""
        result = validate_alignment(angle_deg=1.0)
        assert result is None

    def test_alignment_returns_violation_for_large_tilt(self):
        """Tilt exceeding default 2° → BoundaryViolation with type 'tilt'."""
        result = validate_alignment(angle_deg=5.0)
        assert isinstance(result, BoundaryViolation)
        assert result.violation_type == "tilt"

    def test_alignment_zero_tilt_no_violation(self):
        """Zero tilt → no violation."""
        result = validate_alignment(angle_deg=0.0)
        assert result is None

    # --- validate_pair ---

    def test_validate_pair_returns_list(self):
        """validate_pair must always return a list."""
        result = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=52.0)
        assert isinstance(result, list)

    def test_validate_pair_empty_list_when_valid(self):
        """Well-placed, aligned pair → empty violations list."""
        result = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=52.0, angle_deg=0.0)
        assert result == []

    def test_validate_pair_detects_overlap_violation(self):
        """Overlapping pair → at least one violation."""
        result = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=30.0)
        assert len(result) >= 1
        types = [v.violation_type for v in result]
        assert "overlap" in types

    def test_validate_pair_detects_tilt_violation(self):
        """Tilted pair → tilt violation present."""
        result = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=52.0, angle_deg=5.0)
        types = [v.violation_type for v in result]
        assert "tilt" in types

    # --- validate_all_pairs ---

    def test_validate_all_pairs_returns_boundary_report(self):
        """validate_all_pairs must return a BoundaryReport."""
        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        assert isinstance(report, BoundaryReport)

    def test_boundary_report_has_required_attrs(self):
        """BoundaryReport exposes violations, n_pairs_checked, is_valid, overall_score."""
        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        assert hasattr(report, "violations")
        assert hasattr(report, "n_pairs_checked")
        assert hasattr(report, "is_valid")
        assert hasattr(report, "overall_score")

    def test_validate_all_pairs_aligned_pairs_no_violations(self):
        """Properly spaced pairs → no violations, is_valid=True."""
        report = validate_all_pairs(
            [(0, 1), (1, 2)],
            positions=[0.0, 52.0, 104.0],
            sizes=[50.0, 50.0, 50.0],
        )
        assert report.is_valid is True
        assert len(report.violations) == 0
        assert report.n_pairs_checked == 2

    def test_validate_all_pairs_with_tilt_in_angles(self):
        """Passing angles list with large tilt → not valid."""
        report = validate_all_pairs(
            [(0, 1)],
            positions=[0.0, 52.0],
            sizes=[50.0, 50.0],
            angles=[5.0, 0.0],
        )
        assert report.is_valid is False

    # --- boundary_quality_score ---

    def test_quality_score_returns_float(self):
        """boundary_quality_score must return a float."""
        qs = boundary_quality_score([], n_pairs=1)
        assert isinstance(qs, float)

    def test_quality_score_is_one_when_no_violations(self):
        """No violations → quality score = 1.0."""
        qs = boundary_quality_score([], n_pairs=1)
        assert qs == pytest.approx(1.0)

    def test_quality_score_decreases_with_violations(self):
        """Violations → quality score < 1.0."""
        violations = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=30.0, angle_deg=5.0)
        qs = boundary_quality_score(violations, n_pairs=1)
        assert qs < 1.0

    def test_quality_score_in_zero_one_range(self):
        """Quality score must be in [0, 1]."""
        violations = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=30.0, angle_deg=10.0)
        qs = boundary_quality_score(violations, n_pairs=1)
        assert 0.0 <= qs <= 1.0


# ===========================================================================
# 2. TestOverlapValidator
# ===========================================================================

class TestOverlapValidator:
    """Tests for bbox_overlap, mask_iou, check_pair_overlap,
    validate_assembly, overlap_area_matrix, and batch_validate."""

    # --- bbox_overlap ---

    def test_bbox_overlap_returns_float(self):
        result = bbox_overlap((0, 0, 30, 30), (20, 20, 50, 50))
        assert isinstance(result, float)

    def test_bbox_overlap_non_overlapping_is_zero(self):
        result = bbox_overlap((0, 0, 10, 10), (20, 20, 30, 30))
        assert result == pytest.approx(0.0)

    def test_bbox_overlap_partial_overlap(self):
        result = bbox_overlap((0, 0, 20, 20), (10, 10, 30, 30))
        assert result > 0.0

    # --- mask_iou ---

    def test_mask_iou_returns_float_in_unit_interval(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=15, r1=35)
        iou = mask_iou(m1, m2)
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_mask_iou_no_overlap_is_zero(self):
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m1[0:20, 0:20] = 255
        m2[30:50, 30:50] = 255
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_mask_iou_full_overlap_is_one(self):
        m = _patch_mask()
        assert mask_iou(m, m.copy()) == pytest.approx(1.0)

    def test_mask_iou_zero_masks_returns_zero(self):
        m = np.zeros((50, 50), dtype=np.uint8)
        assert mask_iou(m, m) == pytest.approx(0.0)

    # --- check_pair_overlap ---

    def test_check_pair_overlap_returns_overlap_record(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=15, r1=35)
        rec = check_pair_overlap(0, 1, m1, m2, canvas_size=(50, 50))
        assert isinstance(rec, OverlapRecord)

    def test_overlap_record_has_iou_attr(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=15, r1=35)
        rec = check_pair_overlap(0, 1, m1, m2, canvas_size=(50, 50))
        assert hasattr(rec, "iou")
        assert 0.0 <= rec.iou <= 1.0

    def test_check_pair_overlap_non_overlapping_positions(self):
        """Masks placed far apart on canvas → iou=0."""
        m = _solid_mask(50, 50)
        rec = check_pair_overlap(0, 1, m, m.copy(), canvas_size=(200, 200),
                                 pos1=(0, 0), pos2=(100, 0))
        assert rec.iou == pytest.approx(0.0)
        assert rec.overlap_area == pytest.approx(0.0)

    # --- validate_assembly ---

    def test_validate_assembly_returns_validation_report(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        report = validate_assembly([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert isinstance(report, ValidationReport)

    def test_validation_report_has_n_overlaps(self):
        m = _patch_mask()
        report = validate_assembly([m, m.copy()], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert hasattr(report, "n_overlaps")

    def test_validate_assembly_identical_masks_conflict(self):
        """Two identical masks at same position → overlap detected."""
        m = _patch_mask()
        report = validate_assembly([m, m.copy()], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert report.n_overlaps >= 1
        assert report.is_valid is False

    def test_validate_assembly_single_mask_no_conflict(self):
        """Single mask → no conflicts."""
        m = _patch_mask()
        report = validate_assembly([m], [(0, 0)], canvas_size=(100, 100))
        assert report.n_overlaps == 0
        assert report.is_valid is True

    # --- overlap_area_matrix ---

    def test_overlap_area_matrix_returns_2d_array(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        mat = overlap_area_matrix([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert isinstance(mat, np.ndarray)
        assert mat.ndim == 2

    def test_overlap_area_matrix_diagonal_is_zero(self):
        """Diagonal entries must be zero (self-overlap excluded)."""
        m1 = _patch_mask()
        m2 = _patch_mask(r0=5, r1=25)
        mat = overlap_area_matrix([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_overlap_area_matrix_2x2_non_overlapping(self):
        m_a = np.zeros((100, 100), dtype=np.uint8)
        m_b = np.zeros((100, 100), dtype=np.uint8)
        m_a[0:40, 0:40] = 255
        m_b[60:100, 60:100] = 255
        mat = overlap_area_matrix([m_a, m_b], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert mat.shape == (2, 2)
        assert np.all(mat == pytest.approx(0.0))

    # --- batch_validate ---

    def test_batch_validate_returns_list(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=5, r1=25)
        result = batch_validate([([m1, m2], [(0, 0), (0, 0)], (100, 100))])
        assert isinstance(result, list)

    def test_batch_validate_empty_input_returns_empty_list(self):
        assert batch_validate([]) == []


# ===========================================================================
# 3. TestSeamAnalyzer
# ===========================================================================

class TestSeamAnalyzer:
    """Tests for extract_seam_profiles, analyze_seam, score_seam_quality,
    and batch_analyze_seams."""

    # --- extract_seam_profiles ---

    def test_extract_seam_profiles_returns_tuple_of_two(self):
        img1 = _random_image(seed=1)
        img2 = _random_image(seed=2)
        result = extract_seam_profiles(img1, img2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_seam_profiles_are_numpy_arrays(self):
        img1 = _random_image(seed=3)
        img2 = _random_image(seed=4)
        p1, p2 = extract_seam_profiles(img1, img2)
        assert isinstance(p1, np.ndarray)
        assert isinstance(p2, np.ndarray)

    def test_seam_profile_length_matches_image_width(self):
        """For side=2 (right) and side=0 (left), profile length = image width."""
        img1 = _random_image(h=60, w=80, seed=5)
        img2 = _random_image(h=60, w=80, seed=6)
        p1, p2 = extract_seam_profiles(img1, img2, side1=2, side2=0)
        assert p1.shape[0] == img1.shape[1]
        assert p2.shape[0] == img2.shape[1]

    # --- analyze_seam ---

    def test_analyze_seam_returns_seam_analysis(self):
        img1 = _random_image(seed=7)
        img2 = _random_image(seed=8)
        result = analyze_seam(img1, img2)
        assert isinstance(result, SeamAnalysis)

    def test_seam_analysis_has_quality_score(self):
        img1 = _random_image(seed=9)
        img2 = _random_image(seed=10)
        analysis = analyze_seam(img1, img2)
        assert hasattr(analysis, "quality_score")
        assert isinstance(analysis.quality_score, float)

    def test_seam_analysis_has_component_scores(self):
        img1 = _random_image(seed=11)
        img2 = _random_image(seed=12)
        analysis = analyze_seam(img1, img2)
        assert hasattr(analysis, "brightness_score")
        assert hasattr(analysis, "gradient_score")
        assert hasattr(analysis, "texture_score")

    def test_seam_analysis_has_profile_length(self):
        img1 = _random_image(h=50, w=40, seed=13)
        img2 = _random_image(h=50, w=40, seed=14)
        analysis = analyze_seam(img1, img2)
        assert hasattr(analysis, "profile_length")
        assert analysis.profile_length == img1.shape[1]

    def test_seam_analysis_identical_images_high_quality(self):
        """Identical images at seam → quality_score == 1.0."""
        img = _solid_image(val=128)
        analysis = analyze_seam(img, img.copy())
        assert analysis.quality_score == pytest.approx(1.0, abs=1e-6)

    def test_seam_analysis_very_different_images_lower_quality(self):
        """Dark vs bright images → quality_score < 1.0."""
        img_dark = _solid_image(val=0)
        img_bright = _solid_image(val=255)
        analysis = analyze_seam(img_dark, img_bright)
        assert analysis.quality_score < 1.0

    # --- score_seam_quality ---

    def test_score_seam_quality_returns_float(self):
        img1 = _random_image(seed=15)
        img2 = _random_image(seed=16)
        analysis = analyze_seam(img1, img2)
        score = score_seam_quality(analysis)
        assert isinstance(score, float)

    def test_score_seam_quality_in_unit_interval(self):
        img1 = _random_image(seed=17)
        img2 = _random_image(seed=18)
        analysis = analyze_seam(img1, img2)
        score = score_seam_quality(analysis)
        assert 0.0 <= score <= 1.0

    def test_score_seam_quality_matches_analysis_quality_score(self):
        img1 = _random_image(seed=19)
        img2 = _random_image(seed=20)
        analysis = analyze_seam(img1, img2)
        assert score_seam_quality(analysis) == pytest.approx(analysis.quality_score)

    # --- batch_analyze_seams ---

    def test_batch_analyze_seams_returns_list(self):
        img1 = _random_image(seed=21)
        img2 = _random_image(seed=22)
        result = batch_analyze_seams([img1, img2], [(0, 1)])
        assert isinstance(result, list)

    def test_batch_analyze_seams_length_matches_pairs(self):
        imgs = [_random_image(seed=i) for i in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_analyze_seams(imgs, pairs)
        assert len(result) == len(pairs)

    def test_batch_analyze_seams_contains_seam_analysis(self):
        img1 = _random_image(seed=23)
        img2 = _random_image(seed=24)
        result = batch_analyze_seams([img1, img2], [(0, 1)])
        assert isinstance(result[0], SeamAnalysis)

    def test_batch_analyze_seams_empty_returns_empty(self):
        result = batch_analyze_seams([], [])
        assert result == []


# ===========================================================================
# 4. TestVerificationBoundaryIntegration
# ===========================================================================

class TestVerificationBoundaryIntegration:
    """Integration tests combining multiple verification components."""

    def test_boundary_then_overlap_pipeline(self):
        """Run boundary validation followed by overlap validation without error."""
        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        assert report.is_valid is True

        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        ov_report = validate_assembly([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert isinstance(ov_report, ValidationReport)

    def test_quality_score_from_report_violations(self):
        """boundary_quality_score applied to violations from validate_all_pairs."""
        report = validate_all_pairs(
            [(0, 1)],
            positions=[0.0, 30.0],   # overlapping
            sizes=[50.0, 50.0],
        )
        qs = boundary_quality_score(report.violations, n_pairs=report.n_pairs_checked)
        assert 0.0 <= qs <= 1.0

    def test_seam_then_boundary_combined(self):
        """Seam score and boundary score both numeric; combined average in [0,1]."""
        img1 = _random_image(seed=30)
        img2 = _random_image(seed=31)
        seam_score = score_seam_quality(analyze_seam(img1, img2))

        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        bnd_score = report.overall_score

        combined = (seam_score + bnd_score) / 2.0
        assert 0.0 <= combined <= 1.0

    def test_multi_pair_validation_no_crash(self):
        """Three-piece assembly: boundary + overlap both run cleanly."""
        report = validate_all_pairs(
            [(0, 1), (1, 2)],
            positions=[0.0, 52.0, 104.0],
            sizes=[50.0, 50.0, 50.0],
        )
        assert report.n_pairs_checked == 2

        masks = [_patch_mask(c0=0, c1=20), _patch_mask(c0=25, c1=45), _patch_mask(c0=0, c1=20)]
        ov_mat = overlap_area_matrix(masks, [(0, 0), (0, 0), (0, 0)], canvas_size=(100, 100))
        assert ov_mat.shape == (3, 3)

    def test_full_pipeline_valid_assembly(self):
        """Full pipeline: boundary valid, no overlap, good seam."""
        b_report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        assert b_report.is_valid is True

        m1 = _patch_mask(c0=0, c1=20)
        m2 = _patch_mask(c0=30, c1=50)
        ov_report = validate_assembly([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert ov_report.is_valid is True

        img = _solid_image()
        seam = analyze_seam(img, img.copy())
        assert seam.quality_score == pytest.approx(1.0, abs=1e-6)

    def test_batch_seams_with_multiple_pairs(self):
        """batch_analyze_seams returns one result per pair."""
        imgs = [_random_image(seed=i) for i in range(3)]
        results = batch_analyze_seams(imgs, [(0, 1), (1, 2)])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, SeamAnalysis)
            assert 0.0 <= r.quality_score <= 1.0

    def test_overlap_matrix_symmetric_property(self):
        """overlap_area_matrix[i,j] == overlap_area_matrix[j,i]."""
        m1 = _patch_mask()
        m2 = _patch_mask(r0=15, r1=35)
        mat = overlap_area_matrix([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert mat[0, 1] == pytest.approx(mat[1, 0])

    def test_boundary_quality_score_with_multi_pair_violations(self):
        """Quality score degrades with violations across multiple pairs."""
        v1 = validate_pair(0, 1, pos1=0.0, size1=50.0, pos2=30.0)
        v2 = validate_pair(1, 2, pos1=30.0, size1=50.0, pos2=60.0, angle_deg=5.0)
        all_violations = v1 + v2
        qs = boundary_quality_score(all_violations, n_pairs=2)
        assert qs < 1.0
        assert 0.0 <= qs <= 1.0

    def test_validate_all_pairs_score_consistent_with_quality_score(self):
        """overall_score from BoundaryReport matches boundary_quality_score result."""
        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        manual_qs = boundary_quality_score(report.violations, n_pairs=report.n_pairs_checked)
        assert report.overall_score == pytest.approx(manual_qs, rel=0.01)

    def test_batch_validate_multiple_assemblies(self):
        """batch_validate processes multiple assemblies independently."""
        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        m3 = _patch_mask()
        assemblies = [
            ([m1, m2], [(0, 0), (0, 0)], (100, 100)),
            ([m3, m3.copy()], [(0, 0), (0, 0)], (100, 100)),
        ]
        results = batch_validate(assemblies)
        assert len(results) == 2
        assert isinstance(results[0], ValidationReport)
        assert isinstance(results[1], ValidationReport)


# ===========================================================================
# 5. TestVerificationEdgeCases
# ===========================================================================

class TestVerificationEdgeCases:
    """Edge case tests: single pairs, zero sizes, extreme values, empty inputs."""

    def test_single_pair_boundary_report(self):
        """validate_all_pairs with one pair → n_pairs_checked=1."""
        report = validate_all_pairs([(0, 1)], positions=[0.0, 52.0], sizes=[50.0, 50.0])
        assert report.n_pairs_checked == 1

    def test_large_gap_produces_violation(self):
        """Gap much larger than max_gap → violation with type 'gap'."""
        v = validate_edge_gap(pos1=0.0, size1=50.0, pos2=200.0)
        assert v is not None
        assert v.violation_type == "gap"

    def test_zero_iou_for_non_overlapping_masks(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[0:40, 0:40] = 255
        m2[60:100, 60:100] = 255
        assert mask_iou(m1, m2) == pytest.approx(0.0)

    def test_boundary_with_exact_max_gap(self):
        """Gap == max_gap exactly → should not produce violation."""
        result = validate_edge_gap(pos1=0.0, size1=50.0, pos2=55.0, max_gap=5.0)
        assert result is None

    def test_batch_analyze_seams_empty_list(self):
        result = batch_analyze_seams([], [])
        assert result == []
        assert isinstance(result, list)

    def test_validate_assembly_single_mask(self):
        """Assembly with a single mask always valid."""
        m = _patch_mask()
        report = validate_assembly([m], [(0, 0)], canvas_size=(100, 100))
        assert report.is_valid is True
        assert report.n_overlaps == 0

    def test_overlap_area_matrix_2x2_shape(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        mat = overlap_area_matrix([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        assert mat.shape == (2, 2)

    def test_overlap_area_matrix_2x2_diagonal_zero(self):
        m1 = _patch_mask()
        m2 = _patch_mask(r0=30, r1=50)
        mat = overlap_area_matrix([m1, m2], [(0, 0), (0, 0)], canvas_size=(100, 100))
        np.testing.assert_array_almost_equal(mat.diagonal(), [0.0, 0.0])

    def test_zero_size_mask_iou_is_zero(self):
        m_empty = np.zeros((50, 50), dtype=np.uint8)
        assert mask_iou(m_empty, m_empty) == pytest.approx(0.0)
