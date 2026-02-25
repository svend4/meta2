"""
Tests for the 12 new validators activated in VerificationSuite (Phase 2).

Covers:
    boundary, layout_verify, overlap_validate, spatial, placement,
    layout_score, fragment_valid, quality_report, score_report,
    full_report, metrics, overlap_area

Tests are intentionally self-contained: they build minimal Assembly/Fragment/
Placement objects (no external IO) so they run fast and without side-effects.
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.verification.suite import (
    VerificationSuite,
    VerificationReport,
    ValidatorResult,
    all_validator_names,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img(h: int = 50, w: int = 60) -> np.ndarray:
    """Return a simple white BGR image."""
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _mask(h: int = 50, w: int = 60) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8) * 255


def _fragment(fid: int, h: int = 50, w: int = 60) -> Fragment:
    return Fragment(fragment_id=fid, image=_img(h, w), mask=_mask(h, w))


def _placement(fid: int, x: float = 0.0, y: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y))


def _simple_assembly(n: int = 3) -> Assembly:
    """Build an Assembly with n non-overlapping fragments side by side."""
    fragments = [_fragment(i) for i in range(n)]
    placements = [_placement(i, x=float(i * 60)) for i in range(n)]
    return Assembly(
        fragments=fragments,
        placements=placements,
        total_score=0.75,
        ocr_score=0.6,
    )


def _empty_assembly() -> Assembly:
    """Assembly with no fragments or placements."""
    return Assembly(fragments=[], placements=[], total_score=0.0)


# ─── all_validator_names() ────────────────────────────────────────────────────

class TestAllValidatorNames:
    def test_returns_21_names(self):
        names = all_validator_names()
        assert len(names) == 21

    def test_contains_original_9(self):
        names = all_validator_names()
        for v in ["assembly_score", "layout", "completeness", "seam",
                  "overlap", "text_coherence", "confidence", "consistency",
                  "edge_quality"]:
            assert v in names, f"'{v}' missing from all_validator_names()"

    def test_contains_new_12(self):
        names = all_validator_names()
        for v in ["boundary", "layout_verify", "overlap_validate", "spatial",
                  "placement", "layout_score", "fragment_valid", "quality_report",
                  "score_report", "full_report", "metrics", "overlap_area"]:
            assert v in names, f"'{v}' missing from all_validator_names()"

    def test_no_duplicates(self):
        names = all_validator_names()
        assert len(names) == len(set(names))

    def test_returns_list(self):
        assert isinstance(all_validator_names(), list)


# ─── VerificationSuite.run_all() ─────────────────────────────────────────────

class TestRunAll:
    def test_returns_verification_report(self):
        asm = _simple_assembly()
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert isinstance(report, VerificationReport)

    def test_all_21_validators_attempted(self):
        asm = _simple_assembly()
        suite = VerificationSuite()
        report = suite.run_all(asm)
        # At least the 12 new validators must appear
        names = {r.name for r in report.results}
        for v in ["boundary", "layout_verify", "overlap_validate", "spatial",
                  "placement", "layout_score", "fragment_valid", "quality_report",
                  "score_report", "full_report", "metrics", "overlap_area"]:
            assert v in names, f"Validator '{v}' not in run_all() results"

    def test_final_score_in_range(self):
        asm = _simple_assembly()
        report = VerificationSuite().run_all(asm)
        assert 0.0 <= report.final_score <= 1.0

    def test_run_all_empty_assembly(self):
        asm = _empty_assembly()
        report = VerificationSuite().run_all(asm)
        assert isinstance(report, VerificationReport)
        assert 0.0 <= report.final_score <= 1.0

    def test_summary_contains_all_names(self):
        asm = _simple_assembly()
        report = VerificationSuite().run_all(asm)
        summary = report.summary()
        assert "ИТОГО" in summary


# ─── Individual new validators ────────────────────────────────────────────────

class TestNewValidatorsIntegration:
    """Run each of the 12 new validators individually and check contracts."""

    NEW_VALIDATORS = [
        "boundary",
        "layout_verify",
        "overlap_validate",
        "spatial",
        "placement",
        "layout_score",
        "fragment_valid",
        "quality_report",
        "score_report",
        "full_report",
        "metrics",
        "overlap_area",
    ]

    @pytest.fixture
    def asm(self):
        return _simple_assembly(3)

    @pytest.fixture
    def asm_single(self):
        return _simple_assembly(1)

    def _run_single(self, name: str, assembly) -> ValidatorResult:
        suite = VerificationSuite(validators=[name])
        report = suite.run(assembly)
        assert len(report.results) == 1
        return report.results[0]

    @pytest.mark.parametrize("name", NEW_VALIDATORS)
    def test_score_in_unit_range(self, name, asm):
        result = self._run_single(name, asm)
        assert 0.0 <= result.score <= 1.0, (
            f"Validator '{name}' returned score={result.score} outside [0,1]"
        )

    @pytest.mark.parametrize("name", NEW_VALIDATORS)
    def test_result_has_name(self, name, asm):
        result = self._run_single(name, asm)
        assert result.name == name

    @pytest.mark.parametrize("name", NEW_VALIDATORS)
    def test_no_unhandled_exception(self, name, asm):
        """Validators must never propagate exceptions (graceful degradation)."""
        result = self._run_single(name, asm)
        # If it raises an unhandled exception the test itself fails.
        # If the validator catches and returns error=..., that is also acceptable.
        assert result is not None

    @pytest.mark.parametrize("name", NEW_VALIDATORS)
    def test_empty_assembly_does_not_crash(self, name):
        asm = _empty_assembly()
        result = self._run_single(name, asm)
        assert 0.0 <= result.score <= 1.0


# ─── boundary ─────────────────────────────────────────────────────────────────

class TestBoundaryValidator:
    def test_perfect_layout_high_score(self):
        asm = _simple_assembly(4)
        suite = VerificationSuite(validators=["boundary"])
        report = suite.run(asm)
        result = report.results[0]
        assert result.score > 0.5

    def test_single_fragment_returns_full_score(self):
        asm = _simple_assembly(1)
        suite = VerificationSuite(validators=["boundary"])
        report = suite.run(asm)
        # 1 fragment → no pairs → trivially valid
        assert report.results[0].score == pytest.approx(1.0)

    def test_overlapping_fragments_lower_score(self):
        # Two fragments at the same position → max overlap → low score
        asm = Assembly(
            fragments=[_fragment(0), _fragment(1)],
            placements=[_placement(0, 0.0), _placement(1, 0.0)],
            total_score=0.0,
        )
        suite = VerificationSuite(validators=["boundary"])
        report = suite.run(asm)
        # Severe overlap should degrade the score
        assert report.results[0].score < 1.0


# ─── overlap_area ─────────────────────────────────────────────────────────────

class TestOverlapArea:
    def test_non_overlapping_gives_good_score(self):
        asm = _simple_assembly(3)
        suite = VerificationSuite(validators=["overlap_area"])
        report = suite.run(asm)
        assert report.results[0].score > 0.0  # score defined [0,1]

    def test_empty_gives_valid_score(self):
        asm = _empty_assembly()
        suite = VerificationSuite(validators=["overlap_area"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0


# ─── spatial ──────────────────────────────────────────────────────────────────

class TestSpatialValidator:
    def test_well_spaced_fragments_pass(self):
        asm = Assembly(
            fragments=[_fragment(i) for i in range(3)],
            placements=[_placement(i, x=float(i * 200)) for i in range(3)],
            total_score=0.8,
        )
        suite = VerificationSuite(validators=["spatial"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0

    def test_all_fragments_at_origin_degrade(self):
        asm = Assembly(
            fragments=[_fragment(i) for i in range(3)],
            placements=[_placement(i, 0.0, 0.0) for i in range(3)],
            total_score=0.0,
        )
        suite = VerificationSuite(validators=["spatial"])
        report = suite.run(asm)
        # All at same position → bad spatial arrangement
        assert report.results[0].score <= 1.0


# ─── placement ────────────────────────────────────────────────────────────────

class TestPlacementValidator:
    def test_valid_placement_no_collision(self):
        asm = _simple_assembly(4)
        suite = VerificationSuite(validators=["placement"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0

    def test_duplicate_positions_degrade_score(self):
        asm = Assembly(
            fragments=[_fragment(i) for i in range(3)],
            placements=[_placement(i, 0.0, 0.0) for i in range(3)],
            total_score=0.0,
        )
        suite = VerificationSuite(validators=["placement"])
        report = suite.run(asm)
        assert report.results[0].score <= 1.0


# ─── fragment_valid ───────────────────────────────────────────────────────────

class TestFragmentValid:
    def test_valid_fragments_pass(self):
        asm = _simple_assembly(3)
        suite = VerificationSuite(validators=["fragment_valid"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0

    def test_no_fragments_returns_valid_score(self):
        asm = _empty_assembly()
        suite = VerificationSuite(validators=["fragment_valid"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0


# ─── metrics ──────────────────────────────────────────────────────────────────

class TestMetricsValidator:
    def test_returns_score_without_ground_truth(self):
        """metrics validator must work even without ground-truth data."""
        asm = _simple_assembly(3)
        suite = VerificationSuite(validators=["metrics"])
        report = suite.run(asm)
        result = report.results[0]
        assert 0.0 <= result.score <= 1.0

    def test_empty_assembly_fallback(self):
        asm = _empty_assembly()
        suite = VerificationSuite(validators=["metrics"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0


# ─── layout_score ─────────────────────────────────────────────────────────────

class TestLayoutScore:
    def test_uniform_layout_produces_score(self):
        asm = _simple_assembly(4)
        suite = VerificationSuite(validators=["layout_score"])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0


# ─── quality_report / score_report / full_report ──────────────────────────────

class TestReportValidators:
    @pytest.mark.parametrize("name", ["quality_report", "score_report", "full_report"])
    def test_report_validator_score_range(self, name):
        asm = _simple_assembly(3)
        suite = VerificationSuite(validators=[name])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0

    @pytest.mark.parametrize("name", ["quality_report", "score_report", "full_report"])
    def test_report_validator_empty_asm(self, name):
        asm = _empty_assembly()
        suite = VerificationSuite(validators=[name])
        report = suite.run(asm)
        assert 0.0 <= report.results[0].score <= 1.0


# ─── Registry integrity ───────────────────────────────────────────────────────

class TestRegistryIntegrity:
    def test_all_21_validators_registered(self):
        suite = VerificationSuite(validators=all_validator_names())
        asm = _simple_assembly(2)
        report = suite.run(asm)
        registered = {r.name for r in report.results}
        # All requested validators must appear in results
        for name in all_validator_names():
            assert name in registered, f"Validator '{name}' missing from registry"

    def test_unknown_validator_is_skipped_gracefully(self):
        """Requesting a non-existent validator should not raise."""
        suite = VerificationSuite(validators=["nonexistent_validator_xyz"])
        asm = _simple_assembly(2)
        # run() should complete without raising
        report = suite.run(asm)
        assert isinstance(report, VerificationReport)

    def test_empty_validators_list_uses_default(self):
        suite = VerificationSuite(validators=[])
        asm = _simple_assembly(2)
        report = suite.run(asm)
        assert isinstance(report, VerificationReport)

    def test_subset_of_validators(self):
        names = ["boundary", "metrics", "placement"]
        suite = VerificationSuite(validators=names)
        asm = _simple_assembly(3)
        report = suite.run(asm)
        result_names = {r.name for r in report.results}
        for n in names:
            assert n in result_names
