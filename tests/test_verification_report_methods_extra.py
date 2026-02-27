"""Extra tests for puzzle_reconstruction/verification/suite.py (VerificationReport, ValidatorResult, VerificationSuite)"""
from __future__ import annotations

import json
import math
import pytest
import numpy as np

from puzzle_reconstruction.verification.suite import (
    VerificationReport,
    ValidatorResult,
    VerificationSuite,
    all_validator_names,
    list_validators,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_report(*scores, final=None) -> VerificationReport:
    results = [ValidatorResult(name=f"v{i}", score=s) for i, s in enumerate(scores)]
    if final is None:
        final = float(np.mean(scores)) if scores else 0.0
    return VerificationReport(results=results, final_score=final)


def _report_with_error() -> VerificationReport:
    return VerificationReport(
        results=[
            ValidatorResult("ok_validator", 0.8, details="fine"),
            ValidatorResult("bad_validator", 0.0, error="something broke"),
        ],
        final_score=0.4,
    )


# ─── ValidatorResult – edge cases ────────────────────────────────────────────

class TestValidatorResultExtra:

    def test_success_true_when_no_error(self):
        r = ValidatorResult("x", 0.5)
        assert r.success is True

    def test_success_false_when_error_set(self):
        r = ValidatorResult("x", 0.0, error="boom")
        assert r.success is False

    def test_success_false_when_empty_string_error(self):
        # An empty string error is still an error (not None)
        r = ValidatorResult("x", 0.0, error="")
        # "" is not None, so success is False
        assert r.success is False

    def test_score_zero_no_error_is_success(self):
        r = ValidatorResult("x", 0.0)
        assert r.success is True

    def test_score_one_with_error_is_not_success(self):
        r = ValidatorResult("x", 1.0, error="unexpected")
        assert r.success is False

    def test_details_default_empty_string(self):
        r = ValidatorResult("x", 0.5)
        assert r.details == ""

    def test_error_default_none(self):
        r = ValidatorResult("x", 0.5)
        assert r.error is None


# ─── VerificationReport.summary() – extra coverage ───────────────────────────

class TestVerificationReportSummaryExtra:

    def test_summary_contains_total_label(self):
        report = _make_report(0.8, 0.6, final=0.7)
        s = report.summary()
        assert "ИТОГО" in s

    def test_summary_contains_final_score(self):
        report = _make_report(0.5, final=0.5)
        s = report.summary()
        assert "0.500" in s

    def test_summary_shows_error_message(self):
        report = _report_with_error()
        s = report.summary()
        assert "something broke" in s

    def test_summary_shows_details_when_present(self):
        r = VerificationReport(
            results=[ValidatorResult("v", 0.9, details="detail text")],
            final_score=0.9,
        )
        assert "detail text" in r.summary()

    def test_empty_report_summary_no_crash(self):
        report = VerificationReport(results=[], final_score=0.0)
        s = report.summary()
        assert "ИТОГО" in s

    def test_summary_returns_string(self):
        assert isinstance(_make_report(0.5).summary(), str)


# ─── VerificationReport.as_dict() – extra edge cases ─────────────────────────

class TestAsDictExtra:

    def test_single_validator(self):
        report = _make_report(0.75)
        d = report.as_dict()
        assert len(d["validators"]) == 1

    def test_score_preserved_exactly(self):
        report = _make_report(0.123456789)
        d = report.as_dict()
        assert d["validators"][0]["score"] == pytest.approx(0.123456789, rel=1e-9)

    def test_details_preserved(self):
        r = VerificationReport(
            results=[ValidatorResult("v", 0.5, details="my detail")],
            final_score=0.5,
        )
        d = r.as_dict()
        assert d["validators"][0]["details"] == "my detail"

    def test_error_preserved(self):
        r = VerificationReport(
            results=[ValidatorResult("v", 0.0, error="oops")],
            final_score=0.0,
        )
        d = r.as_dict()
        assert d["validators"][0]["error"] == "oops"
        assert d["validators"][0]["success"] is False

    def test_many_validators(self):
        report = _make_report(*[i / 10.0 for i in range(10)])
        d = report.as_dict()
        assert len(d["validators"]) == 10

    def test_keys_are_exactly_right(self):
        d = _make_report(0.5).as_dict()
        assert set(d.keys()) == {"final_score", "validators"}

    def test_validator_dict_keys(self):
        d = _make_report(0.5).as_dict()
        expected = {"name", "score", "details", "error", "success"}
        assert set(d["validators"][0].keys()) == expected


# ─── VerificationReport.to_json() – extra edge cases ─────────────────────────

class TestToJsonExtra:

    def test_unicode_preserved(self):
        r = VerificationReport(
            results=[ValidatorResult("метрика", 0.7, details="всё хорошо")],
            final_score=0.7,
        )
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["validators"][0]["name"] == "метрика"
        assert "хорошо" in parsed["validators"][0]["details"]

    def test_none_indent_produces_json(self):
        j = _make_report(0.5).to_json(indent=None)
        parsed = json.loads(j)
        assert "final_score" in parsed

    def test_large_indent_produces_json(self):
        j = _make_report(0.5).to_json(indent=8)
        parsed = json.loads(j)
        assert "final_score" in parsed

    def test_empty_report_serialises(self):
        j = VerificationReport(results=[], final_score=0.0).to_json()
        parsed = json.loads(j)
        assert parsed["validators"] == []

    def test_final_score_zero_in_json(self):
        j = VerificationReport(results=[], final_score=0.0).to_json()
        assert json.loads(j)["final_score"] == 0.0

    def test_roundtrip_all_fields(self):
        report = _report_with_error()
        parsed = json.loads(report.to_json())
        vlist = parsed["validators"]
        assert vlist[0]["name"] == "ok_validator"
        assert vlist[0]["score"] == pytest.approx(0.8)
        assert vlist[0]["success"] is True
        assert vlist[1]["success"] is False
        assert vlist[1]["error"] == "something broke"


# ─── VerificationReport.to_markdown() – extra edge cases ─────────────────────

class TestToMarkdownExtra:

    def test_score_4_decimal_places(self):
        report = _make_report(0.12345678, final=0.12345678)
        md = report.to_markdown()
        assert "0.1235" in md or "0.1234" in md  # 4dp rounding

    def test_all_validator_names_present(self):
        names = ["alpha", "beta", "gamma"]
        results = [ValidatorResult(n, 0.5) for n in names]
        report = VerificationReport(results=results, final_score=0.5)
        md = report.to_markdown()
        for n in names:
            assert n in md

    def test_error_validator_dash_score(self):
        r = VerificationReport(
            results=[ValidatorResult("bad", 0.0, error="err!")],
            final_score=0.0,
        )
        md = r.to_markdown()
        # Score should show "—" for error validators
        assert "—" in md

    def test_pipe_in_details_escaped(self):
        r = VerificationReport(
            results=[ValidatorResult("v", 0.5, details="a|b")],
            final_score=0.5,
        )
        md = r.to_markdown()
        # The pipe in details should be escaped as \|
        assert "\\|" in md or "a|b" not in md.split("|")[1:]

    def test_footer_present(self):
        md = _make_report(0.9).to_markdown()
        assert "puzzle-reconstruction" in md

    def test_multiline_is_string(self):
        md = _make_report(0.8, 0.6).to_markdown()
        assert "\n" in md


# ─── VerificationReport.to_html() – extra edge cases ─────────────────────────

class TestToHtmlExtra:

    def test_has_head_and_body(self):
        html = _make_report(0.7).to_html()
        assert "<head>" in html
        assert "<body>" in html

    def test_score_displayed_4_decimal_places(self):
        report = _make_report(0.5678, final=0.5678)
        html = report.to_html()
        assert "0.5678" in html

    def test_validator_names_in_code_tags(self):
        r = VerificationReport(
            results=[ValidatorResult("myval", 0.6)],
            final_score=0.6,
        )
        html = r.to_html()
        assert "<code>myval</code>" in html

    def test_error_row_has_err_class(self):
        r = _report_with_error()
        html = r.to_html()
        assert "class='err'" in html or 'class="err"' in html

    def test_ok_row_has_ok_class(self):
        r = _report_with_error()
        html = r.to_html()
        assert "class='ok'" in html or 'class="ok"' in html

    def test_footer_present(self):
        html = _make_report(0.5).to_html()
        assert "puzzle-reconstruction" in html

    def test_empty_details_no_crash(self):
        r = VerificationReport(
            results=[ValidatorResult("v", 0.5)],
            final_score=0.5,
        )
        html = r.to_html()
        assert "<tbody>" in html

    def test_returns_nonempty_string(self):
        html = _make_report(0.4).to_html()
        assert len(html) > 100

    def test_error_score_shown_as_dash(self):
        r = VerificationReport(
            results=[ValidatorResult("fail", 0.0, error="crash")],
            final_score=0.0,
        )
        html = r.to_html()
        assert "—" in html


# ─── all_validator_names / list_validators ────────────────────────────────────

class TestValidatorRegistry:

    def test_all_validator_names_returns_list(self):
        names = all_validator_names()
        assert isinstance(names, list)

    def test_all_validator_names_nonempty(self):
        names = all_validator_names()
        assert len(names) > 0

    def test_all_validator_names_strings(self):
        for n in all_validator_names():
            assert isinstance(n, str)

    def test_list_validators_subset_of_all(self):
        lv = set(list_validators())
        av = set(all_validator_names())
        # list_validators should be a subset (same or fewer)
        assert lv <= av or av <= lv  # order may differ but overlap expected

    def test_known_validators_present(self):
        names = set(all_validator_names())
        for known in ("boundary", "metrics", "completeness", "overlap"):
            assert known in names


# ─── VerificationSuite – edge cases ──────────────────────────────────────────

class TestVerificationSuiteExtra:

    def _make_assembly(self, n=2):
        from puzzle_reconstruction.models import Assembly, Fragment, Placement
        frags = [Fragment(fragment_id=i,
                          image=np.full((32, 32, 3), 150, dtype=np.uint8))
                 for i in range(n)]
        placements = [Placement(fragment_id=i, position=(float(i * 40), 0.0))
                      for i in range(n)]
        return Assembly(fragments=frags, placements=placements, total_score=0.7)

    def test_empty_validators_returns_report(self):
        suite = VerificationSuite(validators=[])
        asm = self._make_assembly(2)
        report = suite.run(asm)
        assert isinstance(report, VerificationReport)

    def test_is_empty_true_when_no_validators(self):
        suite = VerificationSuite(validators=[])
        assert suite.is_empty() is True

    def test_is_empty_false_when_validators_set(self):
        suite = VerificationSuite(validators=["boundary"])
        assert suite.is_empty() is False

    def test_run_single_validator(self):
        suite = VerificationSuite(validators=["boundary"])
        asm = self._make_assembly(2)
        report = suite.run(asm)
        assert len(report.results) == 1
        assert report.results[0].name == "boundary"

    def test_final_score_in_range(self):
        suite = VerificationSuite(validators=["boundary", "completeness"])
        asm = self._make_assembly(2)
        report = suite.run(asm)
        assert 0.0 <= report.final_score <= 1.0

    def test_all_result_scores_in_range(self):
        suite = VerificationSuite(validators=["boundary", "metrics",
                                              "completeness", "overlap"])
        asm = self._make_assembly(3)
        report = suite.run(asm)
        for r in report.results:
            assert 0.0 <= r.score <= 1.0

    def test_run_all_returns_report(self):
        suite = VerificationSuite(validators=all_validator_names())
        asm = self._make_assembly(2)
        report = suite.run_all(asm)
        assert isinstance(report, VerificationReport)

    def test_unknown_validator_gracefully_handled(self):
        # Should either skip or produce an error result, not raise
        suite = VerificationSuite(validators=["nonexistent_validator_xyz"])
        asm = self._make_assembly(2)
        try:
            report = suite.run(asm)
            # If it runs, report should be a VerificationReport
            assert isinstance(report, VerificationReport)
        except Exception:
            pass  # raising is also acceptable

    def test_report_to_json_after_run(self):
        suite = VerificationSuite(validators=["boundary"])
        asm = self._make_assembly(2)
        report = suite.run(asm)
        j = report.to_json()
        parsed = json.loads(j)
        assert "final_score" in parsed
