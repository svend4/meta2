"""Additional tests for puzzle_reconstruction.verification.suite."""
from __future__ import annotations

import json
import pytest

from puzzle_reconstruction.verification.suite import (
    ValidatorResult,
    VerificationReport,
    VerificationSuite,
    _safe_run,
    all_validator_names,
    list_validators,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

class _FakeAssembly:
    """Minimal Assembly-like object for suite tests."""

    def __init__(self, total_score: float = 0.75,
                 n_placements: int = 3, n_fragments: int = 3):
        self.total_score = total_score
        self.placements = [_FakePlacement(i) for i in range(n_placements)]
        self.fragments = [_FakeFragment(i) for i in range(n_fragments)]
        self.ocr_score = 0.8
        self.method = "greedy"


class _FakePlacement:
    def __init__(self, i: int):
        self.fragment_id = i
        self.position = (float(i * 120), 0.0)
        self.rotation = 0.0
        self.score = 0.9


class _FakeFragment:
    import numpy as _np

    def __init__(self, i: int):
        import numpy as np
        self.fragment_id = i
        self.image = np.ones((80, 80, 3), dtype="uint8") * 200
        self.mask = (np.ones((80, 80), dtype="uint8") * 255)
        self.contour = None
        self.edges = []


# ─── ValidatorResult ──────────────────────────────────────────────────────────

class TestValidatorResultExtra:
    def test_success_no_error(self):
        r = ValidatorResult(name="test", score=0.8)
        assert r.success is True

    def test_success_with_error(self):
        r = ValidatorResult(name="test", score=0.0, error="fail")
        assert r.success is False

    def test_score_stored(self):
        r = ValidatorResult(name="x", score=0.42)
        assert r.score == pytest.approx(0.42)

    def test_name_stored(self):
        r = ValidatorResult(name="my_validator", score=0.5)
        assert r.name == "my_validator"

    def test_details_default_empty(self):
        r = ValidatorResult(name="x", score=0.5)
        assert r.details == ""

    def test_details_stored(self):
        r = ValidatorResult(name="x", score=0.5, details="ok")
        assert r.details == "ok"

    def test_error_default_none(self):
        r = ValidatorResult(name="x", score=0.5)
        assert r.error is None

    def test_error_stored(self):
        r = ValidatorResult(name="x", score=0.0, error="boom")
        assert r.error == "boom"

    def test_score_zero_no_error_is_success(self):
        r = ValidatorResult(name="x", score=0.0)
        assert r.success is True


# ─── VerificationReport ───────────────────────────────────────────────────────

class TestVerificationReportExtra:
    def _make_report(self) -> VerificationReport:
        results = [
            ValidatorResult(name="a", score=0.9, details="good"),
            ValidatorResult(name="b", score=0.5),
            ValidatorResult(name="c", score=0.0, error="timeout"),
        ]
        return VerificationReport(results=results, final_score=0.7)

    def test_final_score_stored(self):
        r = self._make_report()
        assert r.final_score == pytest.approx(0.7)

    def test_results_length(self):
        r = self._make_report()
        assert len(r.results) == 3

    def test_summary_returns_str(self):
        r = self._make_report()
        assert isinstance(r.summary(), str)

    def test_summary_contains_final_score(self):
        r = self._make_report()
        text = r.summary()
        assert "0.700" in text

    def test_summary_contains_validator_names(self):
        r = self._make_report()
        text = r.summary()
        assert "a" in text
        assert "b" in text

    def test_summary_shows_error(self):
        r = self._make_report()
        text = r.summary()
        assert "ERROR" in text or "timeout" in text

    def test_as_dict_structure(self):
        r = self._make_report()
        d = r.as_dict()
        assert "final_score" in d
        assert "validators" in d
        assert isinstance(d["validators"], list)

    def test_as_dict_final_score_value(self):
        r = self._make_report()
        assert r.as_dict()["final_score"] == pytest.approx(0.7)

    def test_as_dict_validator_keys(self):
        r = self._make_report()
        for v in r.as_dict()["validators"]:
            assert "name" in v
            assert "score" in v
            assert "details" in v
            assert "error" in v
            assert "success" in v

    def test_to_json_returns_str(self):
        r = self._make_report()
        j = r.to_json()
        assert isinstance(j, str)

    def test_to_json_is_valid_json(self):
        r = self._make_report()
        obj = json.loads(r.to_json())
        assert "final_score" in obj

    def test_to_json_indent(self):
        r = self._make_report()
        # Default indent=2 → multi-line
        j = r.to_json(indent=2)
        assert "\n" in j

    def test_to_markdown_returns_str(self):
        r = self._make_report()
        md = r.to_markdown()
        assert isinstance(md, str)

    def test_to_markdown_has_table(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "|" in md

    def test_to_markdown_contains_final_score(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "0.7" in md

    def test_to_html_returns_str(self):
        r = self._make_report()
        html = r.to_html()
        assert isinstance(html, str)

    def test_to_html_starts_doctype(self):
        r = self._make_report()
        assert r.to_html().startswith("<!DOCTYPE")

    def test_to_html_contains_table(self):
        r = self._make_report()
        assert "<table" in r.to_html()

    def test_to_html_contains_score(self):
        r = self._make_report()
        assert "0.7" in r.to_html()

    def test_empty_results_report(self):
        r = VerificationReport(results=[], final_score=0.5)
        assert r.summary() != ""
        assert r.as_dict()["validators"] == []

    def test_error_validator_as_dict(self):
        r = VerificationReport(
            results=[ValidatorResult("x", 0.0, error="failed")],
            final_score=0.0,
        )
        v = r.as_dict()["validators"][0]
        assert v["error"] == "failed"
        assert v["success"] is False


# ─── _safe_run ────────────────────────────────────────────────────────────────

class TestSafeRunExtra:
    def test_good_function(self):
        def fn(asm): return 0.8, "ok"
        r = _safe_run("test", fn, None)
        assert r.score == pytest.approx(0.8)
        assert r.details == "ok"
        assert r.success is True

    def test_function_raises_exception(self):
        def fn(asm): raise RuntimeError("boom")
        r = _safe_run("test", fn, None)
        assert r.success is False
        assert "boom" in (r.error or "")

    def test_score_clamped_above(self):
        def fn(asm): return 2.5, ""
        r = _safe_run("x", fn, None)
        assert r.score == pytest.approx(1.0)

    def test_score_clamped_below(self):
        def fn(asm): return -0.5, ""
        r = _safe_run("x", fn, None)
        assert r.score == pytest.approx(0.0)

    def test_score_boundary_zero(self):
        def fn(asm): return 0.0, ""
        r = _safe_run("x", fn, None)
        assert r.score == pytest.approx(0.0)
        assert r.success is True

    def test_score_boundary_one(self):
        def fn(asm): return 1.0, ""
        r = _safe_run("x", fn, None)
        assert r.score == pytest.approx(1.0)


# ─── list_validators / all_validator_names ────────────────────────────────────

class TestValidatorListsExtra:
    def test_list_validators_returns_list(self):
        result = list_validators()
        assert isinstance(result, list)

    def test_list_validators_nonempty(self):
        assert len(list_validators()) > 0

    def test_list_validators_sorted(self):
        v = list_validators()
        assert v == sorted(v)

    def test_list_validators_str_elements(self):
        for name in list_validators():
            assert isinstance(name, str)

    def test_all_validator_names_returns_list(self):
        names = all_validator_names()
        assert isinstance(names, list)

    def test_all_validator_names_has_21(self):
        names = all_validator_names()
        assert len(names) == 21

    def test_all_validator_names_contains_base_nine(self):
        names = all_validator_names()
        for v in ["assembly_score", "layout", "completeness", "seam",
                  "overlap", "text_coherence", "confidence",
                  "consistency", "edge_quality"]:
            assert v in names, f"Missing base validator: {v}"

    def test_all_validator_names_contains_extended(self):
        names = all_validator_names()
        for v in ["boundary", "layout_verify", "overlap_validate", "spatial",
                  "placement", "layout_score", "fragment_valid", "quality_report",
                  "score_report", "full_report", "metrics", "overlap_area"]:
            assert v in names, f"Missing extended validator: {v}"

    def test_all_validator_names_no_duplicates(self):
        names = all_validator_names()
        assert len(names) == len(set(names))


# ─── VerificationSuite ────────────────────────────────────────────────────────

class TestVerificationSuiteExtra:
    def test_is_empty_no_validators(self):
        suite = VerificationSuite(validators=[])
        assert suite.is_empty() is True

    def test_is_empty_with_validators(self):
        suite = VerificationSuite(validators=["confidence"])
        assert suite.is_empty() is False

    def test_run_empty_validators_returns_report(self):
        asm = _FakeAssembly(total_score=0.6)
        suite = VerificationSuite(validators=[])
        report = suite.run(asm)
        assert isinstance(report, VerificationReport)

    def test_run_empty_uses_total_score(self):
        asm = _FakeAssembly(total_score=0.65)
        suite = VerificationSuite(validators=[])
        report = suite.run(asm)
        assert report.final_score == pytest.approx(0.65)

    def test_run_empty_has_no_results(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=[])
        report = suite.run(asm)
        assert report.results == []

    def test_run_confidence_validator(self):
        asm = _FakeAssembly(total_score=0.8)
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        assert len(report.results) == 1
        assert report.results[0].name == "confidence"
        assert 0.0 <= report.final_score <= 1.0

    def test_run_unavailable_validator(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["__nonexistent_validator__"])
        report = suite.run(asm)
        assert len(report.results) == 1
        r = report.results[0]
        assert r.success is False  # error recorded

    def test_run_multiple_validators(self):
        asm = _FakeAssembly(total_score=0.7)
        suite = VerificationSuite(validators=["confidence", "metrics"])
        report = suite.run(asm)
        assert len(report.results) == 2

    def test_run_final_score_is_float(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        assert isinstance(report.final_score, float)

    def test_run_all_returns_report(self):
        asm = _FakeAssembly(total_score=0.5)
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert isinstance(report, VerificationReport)

    def test_run_all_has_results(self):
        asm = _FakeAssembly(total_score=0.5)
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert len(report.results) > 0

    def test_run_all_final_score_in_range(self):
        asm = _FakeAssembly(total_score=0.5)
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert 0.0 <= report.final_score <= 1.0

    def test_run_report_summary_callable(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_run_json_serializable(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence", "metrics"])
        report = suite.run(asm)
        obj = json.loads(report.to_json())
        assert "final_score" in obj
        assert len(obj["validators"]) == 2

    def test_validator_names_preserved(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence", "metrics"])
        report = suite.run(asm)
        names = [r.name for r in report.results]
        assert names == ["confidence", "metrics"]

    def test_run_assembly_score_validator(self):
        asm = _FakeAssembly(n_placements=3, n_fragments=3, total_score=0.9)
        suite = VerificationSuite(validators=["assembly_score"])
        report = suite.run(asm)
        assert 0.0 <= report.final_score <= 1.0

    def test_run_empty_assembly(self):
        asm = _FakeAssembly(total_score=0.0, n_placements=0, n_fragments=0)
        suite = VerificationSuite(validators=["confidence", "metrics"])
        report = suite.run(asm)
        assert isinstance(report.final_score, float)

    def test_report_markdown_has_validators(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        md = report.to_markdown()
        assert "confidence" in md

    def test_report_html_has_validators(self):
        asm = _FakeAssembly()
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        html = report.to_html()
        assert "confidence" in html
