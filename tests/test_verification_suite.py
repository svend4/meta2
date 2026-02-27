"""Tests for puzzle_reconstruction.verification.suite"""
import pytest
import json
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.suite import (
    ValidatorResult,
    VerificationReport,
    list_validators,
    all_validator_names,
    VerificationSuite,
    _safe_run,
)
from puzzle_reconstruction.models import Assembly, Fragment


def make_assembly(total_score=0.5, ocr_score=0.6, n_frags=2):
    frags = [
        Fragment(fragment_id=i, image=np.zeros((50, 50, 3), dtype=np.uint8))
        for i in range(n_frags)
    ]
    asm = Assembly()
    asm.total_score = total_score
    asm.ocr_score = ocr_score
    asm.placements = []
    asm.fragments = frags
    return asm


# ─── ValidatorResult ──────────────────────────────────────────────────────────

def test_validator_result_success_true():
    r = ValidatorResult(name="test", score=0.8)
    assert r.success is True


def test_validator_result_success_false():
    r = ValidatorResult(name="test", score=0.0, error="Some error")
    assert r.success is False


def test_validator_result_with_details():
    r = ValidatorResult(name="layout", score=0.7, details="3 violations")
    assert r.details == "3 violations"


# ─── VerificationReport ───────────────────────────────────────────────────────

def test_verification_report_summary():
    results = [
        ValidatorResult(name="layout", score=0.8),
        ValidatorResult(name="seam", score=0.7),
    ]
    report = VerificationReport(results=results, final_score=0.75)
    summary = report.summary()
    assert "layout" in summary
    assert "seam" in summary
    assert "0.750" in summary or "ИТОГО" in summary


def test_verification_report_as_dict():
    results = [ValidatorResult(name="test", score=0.5)]
    report = VerificationReport(results=results, final_score=0.5)
    d = report.as_dict()
    assert "final_score" in d
    assert "validators" in d
    assert d["final_score"] == pytest.approx(0.5)


def test_verification_report_to_json():
    results = [ValidatorResult(name="test", score=0.5)]
    report = VerificationReport(results=results, final_score=0.5)
    js = report.to_json()
    parsed = json.loads(js)
    assert "final_score" in parsed


def test_verification_report_to_markdown():
    results = [ValidatorResult(name="layout", score=0.8)]
    report = VerificationReport(results=results, final_score=0.8)
    md = report.to_markdown()
    assert "layout" in md
    assert "0.8" in md or "0.80" in md


def test_verification_report_to_html():
    results = [ValidatorResult(name="layout", score=0.8)]
    report = VerificationReport(results=results, final_score=0.8)
    html = report.to_html()
    assert "<!DOCTYPE html>" in html
    assert "layout" in html


def test_verification_report_to_html_error_result():
    results = [ValidatorResult(name="bad_validator", score=0.0, error="failed")]
    report = VerificationReport(results=results, final_score=0.0)
    html = report.to_html()
    assert "failed" in html or "err" in html


def test_verification_report_empty():
    report = VerificationReport()
    assert report.results == []
    assert report.final_score == 0.0


# ─── list_validators ──────────────────────────────────────────────────────────

def test_list_validators_returns_list():
    validators = list_validators()
    assert isinstance(validators, list)
    assert len(validators) > 0


def test_list_validators_sorted():
    validators = list_validators()
    assert validators == sorted(validators)


def test_list_validators_known_names():
    validators = list_validators()
    # At least some common validators should be present
    assert any(v in validators for v in ["layout", "seam", "overlap"])


# ─── all_validator_names ──────────────────────────────────────────────────────

def test_all_validator_names_count():
    names = all_validator_names()
    assert len(names) >= 20


def test_all_validator_names_contains_expected():
    names = all_validator_names()
    expected = ["assembly_score", "layout", "completeness", "seam"]
    for name in expected:
        assert name in names


def test_all_validator_names_no_duplicates():
    names = all_validator_names()
    assert len(names) == len(set(names))


# ─── _safe_run ────────────────────────────────────────────────────────────────

def test_safe_run_success():
    def fn(asm):
        return 0.7, "details here"

    result = _safe_run("test", fn, None)
    assert result.success
    assert result.score == pytest.approx(0.7)
    assert result.details == "details here"


def test_safe_run_exception():
    def fn(asm):
        raise ValueError("Something went wrong")

    result = _safe_run("test", fn, None)
    assert not result.success
    assert "Something went wrong" in (result.error or "")


def test_safe_run_score_clamped_above():
    def fn(asm):
        return 5.0, ""

    result = _safe_run("test", fn, None)
    assert result.score <= 1.0


def test_safe_run_score_clamped_below():
    def fn(asm):
        return -1.0, ""

    result = _safe_run("test", fn, None)
    assert result.score >= 0.0


# ─── VerificationSuite ────────────────────────────────────────────────────────

def test_verification_suite_is_empty_true():
    suite = VerificationSuite()
    assert suite.is_empty()


def test_verification_suite_is_empty_false():
    suite = VerificationSuite(validators=["layout"])
    assert not suite.is_empty()


def test_verification_suite_run_empty_validators():
    suite = VerificationSuite()
    asm = make_assembly(total_score=0.7)
    report = suite.run(asm)
    assert report.final_score == pytest.approx(0.7)
    assert report.results == []


def test_verification_suite_run_unknown_validator():
    suite = VerificationSuite(validators=["nonexistent_validator_xyz"])
    asm = make_assembly()
    report = suite.run(asm)
    assert len(report.results) == 1
    assert report.results[0].success is False


def test_verification_suite_run_layout():
    suite = VerificationSuite(validators=["layout"])
    asm = make_assembly()
    report = suite.run(asm)
    assert len(report.results) >= 1


def test_verification_suite_run_multiple():
    suite = VerificationSuite(validators=["layout", "seam"])
    asm = make_assembly(n_frags=3)
    report = suite.run(asm)
    assert len(report.results) == 2


def test_verification_suite_final_score_range():
    suite = VerificationSuite(validators=["layout"])
    asm = make_assembly()
    report = suite.run(asm)
    assert 0.0 <= report.final_score <= 1.0


def test_verification_suite_run_all():
    suite = VerificationSuite()
    asm = make_assembly(total_score=0.6)
    report = suite.run_all(asm)
    assert isinstance(report, VerificationReport)
    assert len(report.results) >= 20


def test_verification_suite_run_all_final_score_range():
    suite = VerificationSuite()
    asm = make_assembly()
    report = suite.run_all(asm)
    assert 0.0 <= report.final_score <= 1.0


def test_verification_suite_run_completeness():
    suite = VerificationSuite(validators=["completeness"])
    asm = make_assembly(n_frags=4)
    report = suite.run(asm)
    assert isinstance(report, VerificationReport)


def test_verification_suite_results_names():
    suite = VerificationSuite(validators=["layout", "seam"])
    asm = make_assembly()
    report = suite.run(asm)
    names = [r.name for r in report.results]
    assert "layout" in names
    assert "seam" in names
