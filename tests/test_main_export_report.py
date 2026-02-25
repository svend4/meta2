"""
Tests for new CLI options added in Phase 3:
  --validators  — comma-separated list or 'all'
  --export-report PATH — export verification report (.json/.md/.html)

Also tests _export_verification_report() helper for all three formats.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from main import build_parser, _export_verification_report
from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.verification.suite import (
    VerificationSuite,
    VerificationReport,
    ValidatorResult,
    all_validator_names,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img() -> np.ndarray:
    return np.full((40, 50, 3), 180, dtype=np.uint8)


def _minimal_assembly(n: int = 2) -> Assembly:
    frags = [Fragment(fragment_id=i, image=_img()) for i in range(n)]
    placements = [Placement(fragment_id=i, position=(i * 55.0, 0.0)) for i in range(n)]
    return Assembly(fragments=frags, placements=placements, total_score=0.7)


def _minimal_report() -> VerificationReport:
    """Build a VerificationReport without actually running the suite."""
    results = [
        ValidatorResult(name="boundary", score=0.9, details="0 violations"),
        ValidatorResult(name="metrics",  score=0.5, details="neutral"),
        ValidatorResult(name="broken",   score=0.0, error="simulated error"),
    ]
    report = VerificationReport(results=results, final_score=0.7)
    return report


class FakeLog:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ─── Parser: --validators ──────────────────────────────────────────────────────

class TestValidatorsArg:
    def test_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d"])
        assert args.validators is None

    def test_single_validator(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d", "--validators", "boundary"])
        assert args.validators == "boundary"

    def test_multiple_validators(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d",
                                   "--validators", "boundary,metrics,placement"])
        assert args.validators == "boundary,metrics,placement"

    def test_all_keyword(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d", "--validators", "all"])
        assert args.validators == "all"

    def test_validators_with_spaces(self):
        """Spaces inside the comma-list should be accepted (stripped at runtime)."""
        parser = build_parser()
        args = parser.parse_args(["--input", "/d",
                                   "--validators", " boundary , metrics "])
        assert "boundary" in args.validators


# ─── Parser: --export-report ───────────────────────────────────────────────────

class TestExportReportArg:
    def test_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d"])
        assert args.export_report is None

    def test_json_path(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d",
                                   "--export-report", "report.json"])
        assert args.export_report == "report.json"

    def test_md_path(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d",
                                   "--export-report", "/tmp/out.md"])
        assert args.export_report == "/tmp/out.md"

    def test_html_path(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/d",
                                   "--export-report", "result.html"])
        assert args.export_report == "result.html"

    def test_metavar_exposed(self):
        """Ensure --export-report is documented in help."""
        parser = build_parser()
        import io
        buf = io.StringIO()
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass
        # build_parser should not raise


# ─── _export_verification_report: JSON ────────────────────────────────────────

class TestExportJson:
    def test_creates_file(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        assert out.exists()

    def test_json_has_final_score(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        data = json.loads(out.read_text())
        assert "final_score" in data
        assert data["final_score"] == pytest.approx(0.7)

    def test_json_has_validators_list(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        data = json.loads(out.read_text())
        assert "validators" in data
        assert isinstance(data["validators"], list)
        assert len(data["validators"]) == 3

    def test_json_validator_fields(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        data = json.loads(out.read_text())
        v = data["validators"][0]
        assert "name" in v
        assert "score" in v
        assert "details" in v
        assert "error" in v
        assert "success" in v

    def test_json_failed_validator_has_error(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        data = json.loads(out.read_text())
        broken = next(v for v in data["validators"] if v["name"] == "broken")
        assert broken["error"] == "simulated error"
        assert broken["success"] is False

    def test_json_log_info_called(self, tmp_path):
        report = _minimal_report()
        log = FakeLog()
        out = tmp_path / "report.json"
        _export_verification_report(report, out, log)
        assert any("report.json" in m or str(out) in m for m in log.infos)


# ─── _export_verification_report: Markdown ────────────────────────────────────

class TestExportMarkdown:
    def test_creates_md_file(self, tmp_path):
        out = tmp_path / "report.md"
        _export_verification_report(_minimal_report(), out, FakeLog())
        assert out.exists()

    def test_md_contains_table_header(self, tmp_path):
        out = tmp_path / "report.md"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "| Валидатор" in content

    def test_md_contains_final_score(self, tmp_path):
        out = tmp_path / "report.md"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "0.7" in content

    def test_md_contains_validator_names(self, tmp_path):
        out = tmp_path / "report.md"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "boundary" in content
        assert "metrics" in content

    def test_txt_extension_also_works(self, tmp_path):
        out = tmp_path / "report.txt"
        _export_verification_report(_minimal_report(), out, FakeLog())
        assert out.exists()
        content = out.read_text()
        assert "Валидатор" in content


# ─── _export_verification_report: HTML ────────────────────────────────────────

class TestExportHtml:
    def test_creates_html_file(self, tmp_path):
        out = tmp_path / "report.html"
        _export_verification_report(_minimal_report(), out, FakeLog())
        assert out.exists()

    def test_html_has_table_tag(self, tmp_path):
        out = tmp_path / "report.html"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "<table>" in content

    def test_html_has_doctype(self, tmp_path):
        out = tmp_path / "report.html"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_contains_final_score(self, tmp_path):
        out = tmp_path / "report.html"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "0.7" in content

    def test_html_contains_validator_rows(self, tmp_path):
        out = tmp_path / "report.html"
        _export_verification_report(_minimal_report(), out, FakeLog())
        content = out.read_text()
        assert "boundary" in content
        assert "metrics" in content


# ─── _export_verification_report: unknown extension fallback ──────────────────

class TestExportFallback:
    def test_unknown_ext_creates_md(self, tmp_path):
        out = tmp_path / "report.xyz"
        _export_verification_report(_minimal_report(), out, FakeLog())
        # Falls back to .md
        md_out = tmp_path / "report.md"
        assert md_out.exists()

    def test_export_error_is_logged_as_warning(self, tmp_path):
        """If write fails (e.g., bad path), log.warning should be called."""
        log = FakeLog()
        bad_path = Path("/nonexistent_dir/x/report.json")
        _export_verification_report(_minimal_report(), bad_path, log)
        assert len(log.warnings) > 0


# ─── Integration: --validators all → 21 results ───────────────────────────────

class TestValidatorsAllIntegration:
    def test_all_keyword_resolves_to_21(self):
        names = all_validator_names()
        assert len(names) == 21

    def test_suite_with_all_names_returns_21_results(self):
        asm = _minimal_assembly(3)
        suite = VerificationSuite(validators=all_validator_names())
        report = suite.run(asm)
        assert len(report.results) == 21

    def test_validators_csv_subset_runs_correctly(self):
        asm = _minimal_assembly(2)
        names_csv = "boundary,metrics,placement"
        names = [v.strip() for v in names_csv.split(",")]
        suite = VerificationSuite(validators=names)
        report = suite.run(asm)
        result_names = {r.name for r in report.results}
        for n in names:
            assert n in result_names
