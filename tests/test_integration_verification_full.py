"""
Интеграционные тесты: полный прогон всех 21 валидаторов VerificationSuite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.verification.suite import (
    VerificationSuite,
    VerificationReport,
    ValidatorResult,
    all_validator_names,
    list_validators,
)
from puzzle_reconstruction.verification.metrics import evaluate_reconstruction

pytestmark = pytest.mark.integration


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_processed_fragment(fid: int, img: np.ndarray) -> Fragment | None:
    try:
        mask    = segment_fragment(img, method="otsu")
        contour = extract_contour(mask)
        if len(contour) < 4:
            return None
        tangram = fit_tangram(contour)
        fractal = compute_fractal_signature(contour)
        frag    = Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)
        frag.tangram = tangram
        frag.fractal = fractal
        frag.edges   = build_edge_signatures(frag, alpha=0.5, n_sides=4)
        return frag
    except Exception:
        return None


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def doc():
    return generate_test_document(width=280, height=360, seed=55)


@pytest.fixture(scope="module")
def torn_4(doc):
    return tear_document(doc, n_pieces=4, noise_level=0.3, seed=21)


@pytest.fixture(scope="module")
def fragments_4(torn_4):
    frags = []
    for idx, img in enumerate(torn_4):
        f = _make_processed_fragment(idx, img)
        if f is not None:
            frags.append(f)
    return frags


@pytest.fixture(scope="module")
def assembly_4(fragments_4):
    if len(fragments_4) < 2:
        pytest.skip("not enough fragments")
    matrix, entries = build_compat_matrix(fragments_4, threshold=0.0)
    asm = greedy_assembly(fragments_4, entries)
    asm.compat_matrix = matrix
    asm.fragments     = fragments_4
    return asm


@pytest.fixture(scope="module")
def suite_all():
    return VerificationSuite(validators=all_validator_names())


@pytest.fixture(scope="module")
def report_all(suite_all, assembly_4):
    return suite_all.run(assembly_4)


# ─── TestAllValidatorNames ────────────────────────────────────────────────────

class TestAllValidatorNames:
    def test_returns_list(self):
        names = all_validator_names()
        assert isinstance(names, list)

    def test_not_empty(self):
        assert len(all_validator_names()) > 0

    def test_at_least_9_validators(self):
        assert len(all_validator_names()) >= 9

    def test_all_strings(self):
        for n in all_validator_names():
            assert isinstance(n, str) and len(n) > 0

    def test_core_validators_present(self):
        names = all_validator_names()
        for v in ("assembly_score", "layout", "completeness", "seam", "overlap"):
            assert v in names, f"missing validator '{v}'"

    def test_list_validators_subset(self):
        all_n  = set(all_validator_names())
        listed = set(list_validators())
        assert listed <= all_n


# ─── TestVerificationSuiteInit ────────────────────────────────────────────────

class TestVerificationSuiteInit:
    def test_empty_validators_list(self):
        s = VerificationSuite(validators=[])
        assert s.is_empty()

    def test_nonempty_validators(self):
        s = VerificationSuite(validators=["assembly_score"])
        assert not s.is_empty()

    def test_validators_stored(self):
        s = VerificationSuite(validators=["layout", "completeness"])
        assert "layout" in s.validators
        assert "completeness" in s.validators


# ─── TestVerificationSuiteRun ─────────────────────────────────────────────────

class TestVerificationSuiteRun:
    def test_returns_report(self, suite_all, assembly_4):
        report = suite_all.run(assembly_4)
        assert isinstance(report, VerificationReport)

    def test_report_has_results(self, report_all):
        assert isinstance(report_all.results, list)
        assert len(report_all.results) > 0

    def test_report_results_are_validator_results(self, report_all):
        for r in report_all.results:
            assert isinstance(r, ValidatorResult)

    def test_each_result_has_name(self, report_all):
        for r in report_all.results:
            assert isinstance(r.name, str) and len(r.name) > 0

    def test_each_result_score_in_range(self, report_all):
        for r in report_all.results:
            if r.success:
                assert 0.0 <= r.score <= 1.0, f"{r.name}: score={r.score}"

    def test_final_score_in_range(self, report_all):
        assert 0.0 <= report_all.final_score <= 1.0

    def test_final_score_is_finite(self, report_all):
        assert np.isfinite(report_all.final_score)

    def test_run_all_method(self, suite_all, assembly_4):
        report = suite_all.run_all(assembly_4)
        assert isinstance(report, VerificationReport)
        assert report.final_score >= 0.0

    def test_empty_suite_uses_assembly_score(self, assembly_4):
        s = VerificationSuite(validators=[])
        report = s.run(assembly_4)
        assert report.final_score == pytest.approx(
            assembly_4.total_score, abs=1e-6
        )


# ─── TestVerificationSubsets ─────────────────────────────────────────────────

class TestVerificationSubsets:
    @pytest.mark.parametrize("validator", [
        "assembly_score", "layout", "completeness",
        "seam", "overlap", "confidence",
    ])
    def test_single_validator_runs(self, validator, assembly_4):
        s = VerificationSuite(validators=[validator])
        r = s.run(assembly_4)
        assert isinstance(r, VerificationReport)
        assert len(r.results) == 1
        assert r.results[0].name == validator

    def test_subset_two_validators(self, assembly_4):
        s = VerificationSuite(validators=["layout", "completeness"])
        r = s.run(assembly_4)
        assert len(r.results) == 2
        names = {x.name for x in r.results}
        assert names == {"layout", "completeness"}

    def test_unknown_validator_has_error(self, assembly_4):
        s = VerificationSuite(validators=["__no_such_validator__"])
        r = s.run(assembly_4)
        assert len(r.results) == 1
        assert r.results[0].error is not None

    def test_mixed_known_unknown(self, assembly_4):
        s = VerificationSuite(validators=["layout", "__bad__"])
        r = s.run(assembly_4)
        assert len(r.results) == 2
        results_by_name = {x.name: x for x in r.results}
        assert results_by_name["layout"].success
        assert results_by_name["__bad__"].error is not None


# ─── TestVerificationReport ───────────────────────────────────────────────────

class TestVerificationReport:
    def test_summary_is_string(self, report_all):
        s = report_all.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_summary_contains_score(self, report_all):
        s = report_all.summary()
        assert any(c.isdigit() for c in s)

    def test_as_dict_is_dict(self, report_all):
        d = report_all.as_dict()
        assert isinstance(d, dict)

    def test_as_dict_has_final_score(self, report_all):
        d = report_all.as_dict()
        assert "final_score" in d

    def test_as_dict_has_validators(self, report_all):
        d = report_all.as_dict()
        assert "validators" in d
        assert isinstance(d["validators"], list)

    def test_to_json_is_valid_json(self, report_all):
        j = report_all.to_json()
        parsed = json.loads(j)
        assert "final_score" in parsed

    def test_to_markdown_is_string(self, report_all):
        md = report_all.to_markdown()
        assert isinstance(md, str)
        assert "final" in md.lower() or "итог" in md.lower()

    def test_to_html_is_string(self, report_all):
        html = report_all.to_html()
        assert "<html" in html.lower() or "<!DOCTYPE" in html

    def test_json_roundtrip_preserves_score(self, report_all):
        j = report_all.to_json()
        d = json.loads(j)
        assert d["final_score"] == pytest.approx(report_all.final_score, abs=1e-6)


# ─── TestValidatorResultProperties ───────────────────────────────────────────

class TestValidatorResultProperties:
    def test_success_when_no_error(self):
        r = ValidatorResult(name="test", score=0.8)
        assert r.success is True

    def test_failure_when_error_set(self):
        r = ValidatorResult(name="test", score=0.0, error="fail")
        assert r.success is False

    def test_score_stored(self):
        r = ValidatorResult(name="x", score=0.42)
        assert r.score == pytest.approx(0.42)


# ─── TestReconstructionMetrics ────────────────────────────────────────────────

class TestReconstructionMetrics:
    def _placements_to_dict(self, assembly) -> dict:
        """Convert Assembly.placements to Dict[int, Tuple[np.ndarray, float]]."""
        result = {}
        for fid, (pos, angle) in assembly.placements.items():
            result[fid] = (np.asarray(pos, dtype=float), float(angle))
        return result

    def test_evaluate_reconstruction_runs(self, assembly_4, fragments_4):
        if len(fragments_4) < 2:
            pytest.skip("not enough fragments")
        predicted = self._placements_to_dict(assembly_4)
        if len(predicted) < 1:
            pytest.skip("no placements in assembly")
        result = evaluate_reconstruction(predicted, predicted)
        assert result is not None

    def test_evaluate_returns_object(self, assembly_4):
        predicted = self._placements_to_dict(assembly_4)
        if len(predicted) < 1:
            pytest.skip("no placements in assembly")
        result = evaluate_reconstruction(predicted, predicted)
        assert hasattr(result, "neighbor_accuracy")

    def test_evaluate_perfect_reconstruction(self, assembly_4):
        predicted = self._placements_to_dict(assembly_4)
        if len(predicted) < 1:
            pytest.skip("no placements in assembly")
        result = evaluate_reconstruction(predicted, predicted)
        # Perfect reconstruction (same as ground truth) → NA=1.0
        assert result.neighbor_accuracy == pytest.approx(1.0, abs=0.05)

    def test_evaluate_empty_returns_metrics(self):
        result = evaluate_reconstruction({}, {})
        assert result.n_fragments == 0
