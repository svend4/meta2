"""Extra tests for puzzle_reconstruction/verification/assembly_scorer.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.verification.assembly_scorer import (
    ScoreComponent,
    AssemblyScoreReport,
    AssemblyScorerParams,
    score_geometry,
    score_coverage,
    score_seam_quality,
    score_uniqueness,
    compute_assembly_score,
    compare_assemblies,
    rank_assemblies,
)


# ─── ScoreComponent (extra) ─────────────────────────────────────────────────

class TestScoreComponentExtra:
    def test_default_weight_one(self):
        sc = ScoreComponent(name="x", value=0.5)
        assert sc.weight == pytest.approx(1.0)

    def test_weighted_value_with_weight(self):
        sc = ScoreComponent(name="x", value=0.4, weight=2.0)
        assert sc.weighted_value() == pytest.approx(0.8)

    def test_weighted_value_zero_weight(self):
        sc = ScoreComponent(name="x", value=0.9, weight=0.0)
        assert sc.weighted_value() == pytest.approx(0.0)

    def test_value_boundary_zero(self):
        sc = ScoreComponent(name="x", value=0.0)
        assert sc.weighted_value() == pytest.approx(0.0)

    def test_value_boundary_one(self):
        sc = ScoreComponent(name="x", value=1.0)
        assert sc.weighted_value() == pytest.approx(1.0)

    def test_name_stored(self):
        sc = ScoreComponent(name="coverage", value=0.5)
        assert sc.name == "coverage"

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=-0.1)

    def test_value_above_one_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=1.1)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=0.5, weight=-0.1)


# ─── AssemblyScorerParams (extra) ────────────────────────────────────────────

class TestAssemblyScorerParamsExtra:
    def test_total_weight_defaults(self):
        params = AssemblyScorerParams()
        total = params.total_weight()
        assert total > 0.0

    def test_total_weight_custom(self):
        params = AssemblyScorerParams(
            w_geometry=1.0, w_coverage=1.0, w_seam=1.0, w_uniqueness=1.0
        )
        assert params.total_weight() == pytest.approx(4.0)

    def test_all_equal_weights(self):
        params = AssemblyScorerParams(
            w_geometry=1.0, w_coverage=1.0, w_seam=1.0, w_uniqueness=1.0
        )
        assert params.total_weight() == pytest.approx(4.0)

    def test_attributes_stored(self):
        params = AssemblyScorerParams(
            w_geometry=0.3, w_coverage=0.4, w_seam=0.2, w_uniqueness=0.1
        )
        assert params.w_geometry == pytest.approx(0.3)
        assert params.w_coverage == pytest.approx(0.4)
        assert params.w_seam == pytest.approx(0.2)
        assert params.w_uniqueness == pytest.approx(0.1)


# ─── AssemblyScoreReport (extra) ─────────────────────────────────────────────

class TestAssemblyScoreReportExtra:
    def _make_report(self, total=0.7, n=4):
        comps = [
            ScoreComponent("geometry", 0.8),
            ScoreComponent("coverage", 0.6),
        ]
        return AssemblyScoreReport(
            total_score=total,
            components=comps,
            n_fragments=n,
        )

    def test_component_by_name_found(self):
        report = self._make_report()
        comp = report.component_by_name("geometry")
        assert comp is not None
        assert comp.value == pytest.approx(0.8)

    def test_component_by_name_missing_returns_none(self):
        report = self._make_report()
        comp = report.component_by_name("nonexistent")
        assert comp is None

    def test_to_dict_contains_total(self):
        report = self._make_report(total=0.65)
        d = report.to_dict()
        assert "total_score" in d

    def test_to_dict_contains_n_fragments(self):
        report = self._make_report(n=6)
        d = report.to_dict()
        assert d.get("n_fragments") == 6

    def test_n_fragments_stored(self):
        report = self._make_report(n=8)
        assert report.n_fragments == 8

    def test_total_score_boundary(self):
        comps = [ScoreComponent("x", 0.0)]
        r = AssemblyScoreReport(total_score=0.0, components=comps, n_fragments=1)
        assert r.total_score == pytest.approx(0.0)


# ─── score_geometry (extra) ──────────────────────────────────────────────────

class TestScoreGeometryExtra:
    def test_perfect_returns_high(self):
        result = score_geometry(overlap_ratio=0.0, gap_ratio=0.0, alignment_score=1.0)
        assert result >= 0.9

    def test_high_overlap_penalized(self):
        r1 = score_geometry(0.0, 0.0)
        r2 = score_geometry(0.5, 0.0)
        assert r1 > r2

    def test_high_gap_penalized(self):
        r1 = score_geometry(0.0, 0.0)
        r2 = score_geometry(0.0, 0.5)
        assert r1 > r2

    def test_result_in_range(self):
        result = score_geometry(0.3, 0.3, 0.5)
        assert 0.0 <= result <= 1.0

    def test_alignment_affects_score(self):
        r1 = score_geometry(0.0, 0.0, alignment_score=1.0)
        r2 = score_geometry(0.0, 0.0, alignment_score=0.5)
        assert r1 >= r2


# ─── score_coverage (extra) ──────────────────────────────────────────────────

class TestScoreCoverageExtra:
    def test_full_coverage(self):
        result = score_coverage(n_placed=5, n_total=5)
        assert result == pytest.approx(1.0)

    def test_zero_placed(self):
        result = score_coverage(n_placed=0, n_total=5)
        assert result == pytest.approx(0.0)

    def test_partial_coverage(self):
        result = score_coverage(n_placed=2, n_total=4)
        assert 0.0 < result < 1.0

    def test_result_in_range(self):
        result = score_coverage(n_placed=3, n_total=10)
        assert 0.0 <= result <= 1.0

    def test_min_coverage_zero_default(self):
        result = score_coverage(n_placed=1, n_total=4, min_coverage=0.0)
        assert result >= 0.0


# ─── score_seam_quality (extra) ──────────────────────────────────────────────

class TestScoreSeamQualityExtra:
    def test_empty_list_returns_neutral(self):
        result = score_seam_quality([])
        assert 0.0 <= result <= 1.0

    def test_all_high_scores(self):
        result = score_seam_quality([0.9, 0.95, 1.0])
        assert result >= 0.8

    def test_all_low_scores(self):
        result = score_seam_quality([0.0, 0.05, 0.1])
        assert result <= 0.2

    def test_result_in_range(self):
        result = score_seam_quality([0.3, 0.7, 0.5])
        assert 0.0 <= result <= 1.0

    def test_single_score(self):
        result = score_seam_quality([0.6])
        assert 0.0 <= result <= 1.0


# ─── score_uniqueness (extra) ────────────────────────────────────────────────

class TestScoreUniquenessExtra:
    def test_no_duplicates_high(self):
        result = score_uniqueness(n_fragments=5, n_duplicates=0)
        assert result >= 0.9

    def test_all_duplicates_low(self):
        result = score_uniqueness(n_fragments=5, n_duplicates=5)
        assert result <= 0.1

    def test_result_in_range(self):
        result = score_uniqueness(n_fragments=10, n_duplicates=3)
        assert 0.0 <= result <= 1.0

    def test_one_fragment_no_duplicate(self):
        result = score_uniqueness(n_fragments=1, n_duplicates=0)
        assert result >= 0.0


# ─── compute_assembly_score (extra) ──────────────────────────────────────────

class TestComputeAssemblyScoreExtra:
    def test_returns_report(self):
        report = compute_assembly_score(n_placed=3, n_total=4)
        assert isinstance(report, AssemblyScoreReport)

    def test_n_fragments_stored(self):
        report = compute_assembly_score(n_placed=3, n_total=5)
        assert report.n_fragments >= 0

    def test_total_score_in_range(self):
        report = compute_assembly_score(
            n_placed=4, n_total=4,
            overlap_ratio=0.0, gap_ratio=0.0,
            seam_scores=[0.8, 0.9], n_duplicates=0,
        )
        assert 0.0 <= report.total_score <= 1.0

    def test_has_components(self):
        report = compute_assembly_score(n_placed=2, n_total=4)
        assert len(report.components) > 0

    def test_perfect_assembly_high_score(self):
        report = compute_assembly_score(
            n_placed=4, n_total=4,
            overlap_ratio=0.0, gap_ratio=0.0,
            seam_scores=[1.0, 1.0, 1.0], n_duplicates=0,
        )
        assert report.total_score >= 0.7


# ─── compare_assemblies (extra) ──────────────────────────────────────────────

class TestCompareAssembliesExtra:
    def _make_report(self, total):
        comps = [ScoreComponent("x", min(total, 1.0))]
        return AssemblyScoreReport(total_score=total, components=comps, n_fragments=4)

    def test_higher_wins(self):
        r1 = self._make_report(0.8)
        r2 = self._make_report(0.4)
        assert compare_assemblies(r1, r2) > 0

    def test_lower_loses(self):
        r1 = self._make_report(0.3)
        r2 = self._make_report(0.9)
        assert compare_assemblies(r1, r2) < 0

    def test_equal_returns_zero(self):
        r = self._make_report(0.5)
        assert compare_assemblies(r, r) == 0

    def test_returns_int(self):
        r1 = self._make_report(0.6)
        r2 = self._make_report(0.4)
        result = compare_assemblies(r1, r2)
        assert isinstance(result, int)


# ─── rank_assemblies (extra) ─────────────────────────────────────────────────

class TestRankAssembliesExtra:
    def _make_report(self, total):
        comps = [ScoreComponent("x", min(total, 1.0))]
        return AssemblyScoreReport(total_score=total, components=comps, n_fragments=4)

    def test_length_matches(self):
        reports = [self._make_report(v) for v in [0.3, 0.7, 0.5]]
        ranked = rank_assemblies(reports)
        assert len(ranked) == 3

    def test_rank_starts_at_one(self):
        reports = [self._make_report(0.5)]
        ranked = rank_assemblies(reports)
        assert ranked[0][0] == 1

    def test_sorted_by_score_descending(self):
        reports = [self._make_report(v) for v in [0.2, 0.8, 0.5]]
        ranked = rank_assemblies(reports)
        scores = [r.total_score for _, r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_returns_empty(self):
        assert rank_assemblies([]) == []

    def test_single_report_rank_one(self):
        r = self._make_report(0.6)
        ranked = rank_assemblies([r])
        assert ranked[0][0] == 1
        assert ranked[0][1] is r
