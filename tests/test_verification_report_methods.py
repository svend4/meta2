"""
Tests for VerificationReport serialisation methods added in Phase 6:
    as_dict(), to_json(), to_markdown(), to_html()

Also tests that Pipeline.verify_suite() and PipelineResult.verification_report
work end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.verification.suite import (
    VerificationReport,
    ValidatorResult,
    VerificationSuite,
    all_validator_names,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _minimal_report() -> VerificationReport:
    results = [
        ValidatorResult(name="boundary", score=0.9, details="OK"),
        ValidatorResult(name="metrics",  score=0.5, details="neutral"),
        ValidatorResult(name="broken",   score=0.0, error="simulated error"),
    ]
    return VerificationReport(results=results, final_score=0.7)


def _empty_report() -> VerificationReport:
    return VerificationReport(results=[], final_score=0.0)


# ─── VerificationReport.as_dict() ─────────────────────────────────────────────

class TestAsDictMethod:

    def test_returns_dict(self):
        d = _minimal_report().as_dict()
        assert isinstance(d, dict)

    def test_has_final_score(self):
        d = _minimal_report().as_dict()
        assert "final_score" in d
        assert d["final_score"] == pytest.approx(0.7)

    def test_has_validators_list(self):
        d = _minimal_report().as_dict()
        assert "validators" in d
        assert isinstance(d["validators"], list)
        assert len(d["validators"]) == 3

    def test_validator_has_required_fields(self):
        v = _minimal_report().as_dict()["validators"][0]
        for field in ("name", "score", "details", "error", "success"):
            assert field in v, f"Field '{field}' missing from validator dict"

    def test_successful_validator_success_true(self):
        d = _minimal_report().as_dict()
        ok = next(v for v in d["validators"] if v["name"] == "boundary")
        assert ok["success"] is True
        assert ok["error"] is None

    def test_failed_validator_success_false(self):
        d = _minimal_report().as_dict()
        broken = next(v for v in d["validators"] if v["name"] == "broken")
        assert broken["success"] is False
        assert broken["error"] == "simulated error"

    def test_scores_match(self):
        report = _minimal_report()
        d = report.as_dict()
        for result, entry in zip(report.results, d["validators"]):
            assert entry["score"] == pytest.approx(result.score)

    def test_empty_report(self):
        d = _empty_report().as_dict()
        assert d["final_score"] == 0.0
        assert d["validators"] == []


# ─── VerificationReport.to_json() ─────────────────────────────────────────────

class TestToJsonMethod:

    def test_returns_string(self):
        assert isinstance(_minimal_report().to_json(), str)

    def test_valid_json(self):
        j = _minimal_report().to_json()
        parsed = json.loads(j)  # must not raise
        assert isinstance(parsed, dict)

    def test_roundtrip_final_score(self):
        report = _minimal_report()
        parsed = json.loads(report.to_json())
        assert parsed["final_score"] == pytest.approx(report.final_score, abs=1e-9)

    def test_roundtrip_scores(self):
        report = _minimal_report()
        parsed = json.loads(report.to_json())
        for r, v in zip(report.results, parsed["validators"]):
            assert v["name"] == r.name
            assert v["score"] == pytest.approx(r.score, abs=1e-9)

    def test_indent_2_by_default(self):
        j = _minimal_report().to_json()
        assert "\n" in j  # indented → multiline

    def test_indent_0_valid_json(self):
        j = _minimal_report().to_json(indent=0)
        parsed = json.loads(j)  # must not raise
        assert parsed["final_score"] == pytest.approx(0.7)

    def test_as_dict_consistency(self):
        report = _minimal_report()
        assert json.loads(report.to_json()) == report.as_dict()


# ─── VerificationReport.to_markdown() ────────────────────────────────────────

class TestToMarkdownMethod:

    def test_returns_string(self):
        assert isinstance(_minimal_report().to_markdown(), str)

    def test_contains_h1(self):
        assert "# Отчёт" in _minimal_report().to_markdown()

    def test_contains_final_score(self):
        md = _minimal_report().to_markdown()
        assert "0.7000" in md

    def test_contains_table_header(self):
        md = _minimal_report().to_markdown()
        assert "| Валидатор" in md

    def test_contains_separator(self):
        md = _minimal_report().to_markdown()
        assert "|--------" in md

    def test_contains_validator_names(self):
        md = _minimal_report().to_markdown()
        assert "boundary" in md
        assert "metrics" in md
        assert "broken" in md

    def test_error_validator_shown(self):
        md = _minimal_report().to_markdown()
        assert "✗" in md or "simulated error" in md

    def test_empty_report_no_crash(self):
        md = _empty_report().to_markdown()
        assert "# Отчёт" in md


# ─── VerificationReport.to_html() ────────────────────────────────────────────

class TestToHtmlMethod:

    def test_returns_string(self):
        assert isinstance(_minimal_report().to_html(), str)

    def test_has_doctype(self):
        assert "<!DOCTYPE html>" in _minimal_report().to_html()

    def test_has_table(self):
        assert "<table>" in _minimal_report().to_html()

    def test_has_tbody(self):
        html = _minimal_report().to_html()
        assert "<tbody>" in html
        assert "</tbody>" in html

    def test_has_final_score(self):
        html = _minimal_report().to_html()
        assert "0.7000" in html

    def test_contains_validator_names(self):
        html = _minimal_report().to_html()
        assert "boundary" in html
        assert "metrics" in html
        assert "broken" in html

    def test_error_class_present(self):
        html = _minimal_report().to_html()
        assert "class='err'" in html or 'class="err"' in html

    def test_ok_class_present(self):
        html = _minimal_report().to_html()
        assert "class='ok'" in html or 'class="ok"' in html

    def test_utf8_meta(self):
        assert "utf-8" in _minimal_report().to_html()

    def test_empty_report_no_crash(self):
        html = _empty_report().to_html()
        assert "<!DOCTYPE html>" in html


# ─── Pipeline.verify_suite() integration ─────────────────────────────────────

class TestPipelineVerifySuite:

    def _make_assembly(self, n: int = 2) -> Assembly:
        frags = [Fragment(fragment_id=i,
                          image=np.full((40, 50, 3), 180, dtype=np.uint8))
                 for i in range(n)]
        placements = [Placement(fragment_id=i, position=(i * 55.0, 0.0))
                      for i in range(n)]
        return Assembly(fragments=frags, placements=placements, total_score=0.7)

    def test_verify_suite_returns_report(self):
        from puzzle_reconstruction.pipeline import Pipeline
        pl  = Pipeline()
        asm = self._make_assembly(2)
        report = pl.verify_suite(asm, validators=["boundary", "metrics"])
        from puzzle_reconstruction.verification.suite import VerificationReport
        assert isinstance(report, VerificationReport)

    def test_verify_suite_correct_validators(self):
        from puzzle_reconstruction.pipeline import Pipeline
        pl     = Pipeline()
        asm    = self._make_assembly(2)
        report = pl.verify_suite(asm, validators=["boundary", "metrics"])
        names  = {r.name for r in report.results}
        assert "boundary" in names
        assert "metrics" in names

    def test_verify_suite_all_when_none_specified(self):
        from puzzle_reconstruction.pipeline import Pipeline
        pl     = Pipeline()
        asm    = self._make_assembly(2)
        report = pl.verify_suite(asm, validators=None)
        assert len(report.results) == 21

    def test_verify_suite_from_config(self):
        """Validates that cfg.verification.validators is picked up."""
        from puzzle_reconstruction.pipeline import Pipeline
        from puzzle_reconstruction.config import Config
        cfg = Config.default()
        cfg.verification.validators = ["completeness", "overlap"]
        pl     = Pipeline(cfg=cfg)
        asm    = self._make_assembly(2)
        report = pl.verify_suite(asm)
        names  = {r.name for r in report.results}
        assert "completeness" in names
        assert "overlap" in names

    def test_verify_suite_score_in_range(self):
        from puzzle_reconstruction.pipeline import Pipeline
        pl     = Pipeline()
        asm    = self._make_assembly(3)
        report = pl.verify_suite(asm, validators=all_validator_names())
        assert 0.0 <= report.final_score <= 1.0
        for r in report.results:
            assert 0.0 <= r.score <= 1.0


# ─── PipelineResult.verification_report field ─────────────────────────────────

class TestPipelineResultVerificationReport:

    def _make_minimal_pipeline_result(self, vr=None):
        from puzzle_reconstruction.pipeline import PipelineResult, Pipeline
        from puzzle_reconstruction.utils.logger import PipelineTimer
        from puzzle_reconstruction.config import Config
        asm = Assembly(
            fragments=[],
            placements=[],
            total_score=0.5,
        )
        timer = PipelineTimer()
        cfg   = Config.default()
        return PipelineResult(asm, timer, cfg, n_input=0,
                              verification_report=vr)

    def test_verification_report_none_by_default(self):
        pr = self._make_minimal_pipeline_result()
        assert pr.verification_report is None

    def test_verification_report_stored(self):
        vr = _minimal_report()
        pr = self._make_minimal_pipeline_result(vr=vr)
        assert pr.verification_report is vr

    def test_summary_without_verification_report(self):
        pr = self._make_minimal_pipeline_result()
        s  = pr.summary()
        assert "Верификация" not in s

    def test_summary_with_verification_report(self):
        vr = _minimal_report()
        pr = self._make_minimal_pipeline_result(vr=vr)
        s  = pr.summary()
        assert "Верификация" in s or "верификация" in s.lower()
        assert "3" in s  # 3 validators

    def test_summary_shows_final_score(self):
        vr = VerificationReport(
            results=[ValidatorResult("a", 0.8)],
            final_score=0.8,
        )
        pr = self._make_minimal_pipeline_result(vr=vr)
        s  = pr.summary()
        assert "80" in s or "0.8" in s
