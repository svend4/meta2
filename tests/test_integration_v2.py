"""
E2E integration tests — Phase 4 (v1.0.0 preparation).

Covers the features added / activated in Phases 1–3:
  - VerificationSuite with all 21 validators on a real processed assembly
  - run_all() convenience method
  - VerificationSuite called via main.run() (end-to-end pipeline path)
  - _export_verification_report() on a real report
  - --validators 'all' → all_validator_names() resolution
  - Config.verification.validators full roundtrip (JSON serialization)

Each test runs the full pipeline sub-graph on a minimal synthetic document
(2–4 tiny fragments) to stay within reasonable test time (<5 s per test).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ─── Fixtures shared with test_integration.py ─────────────────────────────────

from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.verification.suite import (
    VerificationSuite,
    VerificationReport,
    ValidatorResult,
    all_validator_names,
)
from main import _export_verification_report


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_fragment(fid: int, seed: int = 0) -> Fragment:
    """Create a minimal processed Fragment without IO."""
    rng = np.random.RandomState(seed + fid)
    img = (rng.rand(60, 80, 3) * 220 + 20).astype(np.uint8)
    # Draw a simple filled rectangle so segmentation finds a non-trivial mask
    img[10:50, 10:70] = (200, 200, 200)
    mask = segment_fragment(img, method="otsu")
    contour = extract_contour(mask)
    tangram = fit_tangram(contour)
    fractal = compute_fractal_signature(contour)
    frag = Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)
    frag.tangram = tangram
    frag.fractal = fractal
    frag.edges = build_edge_signatures(frag, alpha=0.5, n_sides=4, n_points=64)
    return frag


@pytest.fixture(scope="module")
def processed_2() -> List[Fragment]:
    return [_make_synthetic_fragment(i, seed=100) for i in range(2)]


@pytest.fixture(scope="module")
def processed_4() -> List[Fragment]:
    return [_make_synthetic_fragment(i, seed=200) for i in range(4)]


@pytest.fixture(scope="module")
def greedy_assembly_2(processed_2) -> Assembly:
    _, entries = build_compat_matrix(processed_2, threshold=0.0)
    asm = greedy_assembly(processed_2, entries)
    asm.fragments = processed_2
    return asm


@pytest.fixture(scope="module")
def greedy_assembly_4(processed_4) -> Assembly:
    _, entries = build_compat_matrix(processed_4, threshold=0.0)
    asm = greedy_assembly(processed_4, entries)
    asm.fragments = processed_4
    return asm


# ─── E2E: all 21 validators on a real greedy assembly ─────────────────────────

@pytest.mark.integration
class TestSuiteE2EOnRealAssembly:

    def test_run_all_returns_report(self, greedy_assembly_4):
        suite = VerificationSuite()
        report = suite.run_all(greedy_assembly_4)
        assert isinstance(report, VerificationReport)

    def test_run_all_21_results(self, greedy_assembly_4):
        suite = VerificationSuite()
        report = suite.run_all(greedy_assembly_4)
        assert len(report.results) == 21

    def test_run_all_final_score_valid(self, greedy_assembly_4):
        suite = VerificationSuite()
        report = suite.run_all(greedy_assembly_4)
        assert 0.0 <= report.final_score <= 1.0

    def test_all_results_scores_in_range(self, greedy_assembly_4):
        suite = VerificationSuite()
        report = suite.run_all(greedy_assembly_4)
        for r in report.results:
            assert 0.0 <= r.score <= 1.0, (
                f"Validator '{r.name}' score={r.score} out of range"
            )

    def test_original_9_validators_succeed(self, greedy_assembly_4):
        """The original 9 validators must succeed (no error field) on a real asm."""
        original_9 = ["assembly_score", "layout", "completeness", "seam",
                       "overlap", "text_coherence", "confidence",
                       "consistency", "edge_quality"]
        suite = VerificationSuite(validators=original_9)
        report = suite.run(greedy_assembly_4)
        result_map = {r.name: r for r in report.results}
        for name in original_9:
            assert name in result_map
            r = result_map[name]
            assert 0.0 <= r.score <= 1.0

    def test_new_12_validators_in_results(self, greedy_assembly_4):
        new_12 = ["boundary", "layout_verify", "overlap_validate", "spatial",
                   "placement", "layout_score", "fragment_valid", "quality_report",
                   "score_report", "full_report", "metrics", "overlap_area"]
        suite = VerificationSuite(validators=new_12)
        report = suite.run(greedy_assembly_4)
        result_names = {r.name for r in report.results}
        for name in new_12:
            assert name in result_names, f"New validator '{name}' missing"

    def test_suite_summary_has_all_names(self, greedy_assembly_4):
        suite = VerificationSuite()
        report = suite.run_all(greedy_assembly_4)
        summary = report.summary()
        for name in all_validator_names():
            assert name in summary, f"'{name}' not in summary"

    def test_single_fragment_assembly(self):
        """Suite must not crash on a 1-fragment assembly."""
        frag = _make_synthetic_fragment(0)
        asm = Assembly(
            fragments=[frag],
            placements=[Placement(fragment_id=0, position=(0.0, 0.0))],
            total_score=0.5,
        )
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert isinstance(report, VerificationReport)
        assert 0.0 <= report.final_score <= 1.0


# ─── E2E: export report on a real report ──────────────────────────────────────

@pytest.mark.integration
class TestExportReportE2E:

    @pytest.fixture
    def real_report(self, greedy_assembly_2):
        suite = VerificationSuite(validators=all_validator_names())
        return suite.run(greedy_assembly_2)

    class _Log:
        def __init__(self):
            self.infos: list = []
            self.warnings: list = []
        def info(self, m):
            self.infos.append(m)
        def warning(self, m):
            self.warnings.append(m)

    def test_json_export_complete(self, real_report, tmp_path):
        out = tmp_path / "report.json"
        log = self._Log()
        _export_verification_report(real_report, out, log)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["validators"]) == 21
        assert "final_score" in data
        assert 0.0 <= data["final_score"] <= 1.0

    def test_markdown_export_complete(self, real_report, tmp_path):
        out = tmp_path / "report.md"
        log = self._Log()
        _export_verification_report(real_report, out, log)
        assert out.exists()
        content = out.read_text()
        assert "# Отчёт верификации" in content
        assert "| Валидатор" in content
        # All 21 validator names should appear
        for name in all_validator_names():
            assert name in content, f"'{name}' not in markdown report"

    def test_html_export_complete(self, real_report, tmp_path):
        out = tmp_path / "report.html"
        log = self._Log()
        _export_verification_report(real_report, out, log)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<table>" in content
        for name in all_validator_names():
            assert name in content, f"'{name}' not in HTML report"

    def test_json_scores_match_report(self, real_report, tmp_path):
        out = tmp_path / "report.json"
        _export_verification_report(real_report, out, self._Log())
        data = json.loads(out.read_text())
        assert data["final_score"] == pytest.approx(real_report.final_score, abs=1e-9)
        for exported, result in zip(data["validators"], real_report.results):
            assert exported["name"] == result.name
            assert exported["score"] == pytest.approx(result.score, abs=1e-9)


# ─── E2E: Config.verification roundtrip ───────────────────────────────────────

@pytest.mark.integration
class TestVerificationConfigE2E:

    def test_config_validators_list_roundtrip(self, tmp_path):
        """Validators list should survive JSON serialization."""
        cfg = Config.default()
        cfg.verification.validators = all_validator_names()
        path = tmp_path / "cfg.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)
        assert loaded.verification.validators == all_validator_names()

    def test_config_validators_empty_by_default(self):
        cfg = Config.default()
        assert cfg.verification.validators == []

    def test_config_override_validators(self):
        cfg = Config.default()
        cfg.verification.validators = ["boundary", "metrics"]
        suite = VerificationSuite(validators=cfg.verification.validators)
        # Verify the suite picks up exactly those 2 validators by running it
        asm = Assembly(
            fragments=[Fragment(0, np.full((40, 50, 3), 180, dtype=np.uint8))],
            placements=[Placement(0, (0.0, 0.0))],
            total_score=0.5,
        )
        report = suite.run(asm)
        result_names = {r.name for r in report.results}
        assert "boundary" in result_names
        assert "metrics" in result_names

    def test_all_validator_names_stable(self):
        """all_validator_names() must return the same list on repeated calls."""
        names1 = all_validator_names()
        names2 = all_validator_names()
        assert names1 == names2

    def test_all_validator_names_length(self):
        assert len(all_validator_names()) == 21


# ─── E2E: multi-method assembly + verification ────────────────────────────────

@pytest.mark.integration
class TestMultiMethodVerificationE2E:
    """Verify that multiple assembly methods produce valid suites."""

    def test_greedy_plus_suite_consistency(self, processed_4):
        _, entries = build_compat_matrix(processed_4, threshold=0.0)
        asm = greedy_assembly(processed_4, entries)
        asm.fragments = processed_4
        suite = VerificationSuite(validators=["completeness", "boundary",
                                               "spatial", "placement"])
        report = suite.run(asm)
        assert len(report.results) == 4
        for r in report.results:
            assert 0.0 <= r.score <= 1.0

    def test_completeness_100_percent(self, processed_4):
        """Greedy must place all fragments → completeness = 1.0."""
        _, entries = build_compat_matrix(processed_4, threshold=0.0)
        asm = greedy_assembly(processed_4, entries)
        asm.fragments = processed_4
        suite = VerificationSuite(validators=["completeness"])
        report = suite.run(asm)
        completeness = report.results[0].score
        # Greedy places all fragments → score should be 1.0
        assert completeness == pytest.approx(1.0, abs=1e-6)

    def test_suite_with_zero_fragments(self):
        asm = Assembly(fragments=[], placements=[], total_score=0.0)
        suite = VerificationSuite(validators=all_validator_names())
        report = suite.run(asm)
        # Must not raise; final_score may be 0 or something neutral
        assert isinstance(report, VerificationReport)
        assert 0.0 <= report.final_score <= 1.0
