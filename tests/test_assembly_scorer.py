"""Tests for puzzle_reconstruction.verification.assembly_scorer."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.verification.assembly_scorer import (
    AssemblyScoreReport,
    AssemblyScorerParams,
    ScoreComponent,
    compare_assemblies,
    compute_assembly_score,
    rank_assemblies,
    score_coverage,
    score_geometry,
    score_seam_quality,
    score_uniqueness,
)


# ─── ScoreComponent ───────────────────────────────────────────────────────────

class TestScoreComponent:
    def test_fields_stored(self):
        c = ScoreComponent(name="geometry", value=0.8, weight=0.3)
        assert c.name == "geometry"
        assert c.value == pytest.approx(0.8)
        assert c.weight == pytest.approx(0.3)

    def test_default_weight_one(self):
        c = ScoreComponent(name="x", value=0.5)
        assert c.weight == pytest.approx(1.0)

    def test_default_details_empty(self):
        c = ScoreComponent(name="x", value=0.5)
        assert c.details == {}

    def test_value_below_zero_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=-0.1)

    def test_value_above_one_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=1.001)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=0.5, weight=-0.1)

    def test_zero_weight_allowed(self):
        c = ScoreComponent(name="x", value=0.5, weight=0.0)
        assert c.weight == pytest.approx(0.0)

    def test_weighted_value(self):
        c = ScoreComponent(name="x", value=0.4, weight=0.5)
        assert c.weighted_value() == pytest.approx(0.2)

    def test_boundary_values_allowed(self):
        ScoreComponent(name="a", value=0.0)
        ScoreComponent(name="b", value=1.0)


# ─── AssemblyScoreReport ──────────────────────────────────────────────────────

class TestAssemblyScoreReport:
    def _report(self) -> AssemblyScoreReport:
        comps = [
            ScoreComponent("geometry", 0.9, 0.3),
            ScoreComponent("coverage", 0.7, 0.3),
        ]
        return AssemblyScoreReport(
            total_score=0.8,
            components=comps,
            n_fragments=5,
            params={"w_geometry": 0.3},
        )

    def test_fields_stored(self):
        r = self._report()
        assert r.total_score == pytest.approx(0.8)
        assert r.n_fragments == 5
        assert len(r.components) == 2

    def test_component_by_name_found(self):
        r = self._report()
        c = r.component_by_name("geometry")
        assert c is not None
        assert c.name == "geometry"

    def test_component_by_name_not_found(self):
        r = self._report()
        assert r.component_by_name("nonexistent") is None

    def test_to_dict_keys(self):
        r = self._report()
        d = r.to_dict()
        assert "total_score" in d
        assert "n_fragments" in d
        assert "components" in d
        assert "params" in d

    def test_to_dict_total_score(self):
        r = self._report()
        d = r.to_dict()
        assert d["total_score"] == pytest.approx(0.8)

    def test_to_dict_components_list(self):
        r = self._report()
        d = r.to_dict()
        assert isinstance(d["components"], list)
        assert len(d["components"]) == 2

    def test_to_dict_component_has_keys(self):
        r = self._report()
        d = r.to_dict()
        c0 = d["components"][0]
        assert "name" in c0
        assert "value" in c0
        assert "weight" in c0

    def test_default_components_empty(self):
        r = AssemblyScoreReport(total_score=0.5)
        assert r.components == []

    def test_default_n_fragments_zero(self):
        r = AssemblyScoreReport(total_score=0.5)
        assert r.n_fragments == 0


# ─── AssemblyScorerParams ─────────────────────────────────────────────────────

class TestAssemblyScorerParams:
    def test_defaults(self):
        p = AssemblyScorerParams()
        assert p.w_geometry == pytest.approx(0.30)
        assert p.w_coverage == pytest.approx(0.30)
        assert p.w_seam == pytest.approx(0.25)
        assert p.w_uniqueness == pytest.approx(0.15)

    def test_total_weight(self):
        p = AssemblyScorerParams()
        assert p.total_weight() == pytest.approx(1.0)

    def test_custom_weights(self):
        p = AssemblyScorerParams(w_geometry=0.5, w_coverage=0.5, w_seam=0.0, w_uniqueness=0.0)
        assert p.total_weight() == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AssemblyScorerParams(w_geometry=-0.1)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyScorerParams(
                w_geometry=0.0, w_coverage=0.0,
                w_seam=0.0, w_uniqueness=0.0,
            )

    def test_zero_individual_weight_ok(self):
        p = AssemblyScorerParams(w_seam=0.0, w_uniqueness=0.0,
                                 w_geometry=0.5, w_coverage=0.5)
        assert p.total_weight() == pytest.approx(1.0)


# ─── score_geometry ───────────────────────────────────────────────────────────

class TestScoreGeometry:
    def test_perfect_score(self):
        assert score_geometry(0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_full_overlap_zero(self):
        assert score_geometry(1.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_full_gap_reduces_score(self):
        s = score_geometry(0.0, 1.0, 1.0)
        assert s == pytest.approx(0.5)

    def test_alignment_multiplier(self):
        s = score_geometry(0.0, 0.0, 0.5)
        assert s == pytest.approx(0.5)

    def test_result_in_unit_interval(self):
        s = score_geometry(0.3, 0.4, 0.8)
        assert 0.0 <= s <= 1.0

    def test_overlap_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(-0.1, 0.0)
        with pytest.raises(ValueError):
            score_geometry(1.1, 0.0)

    def test_gap_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(0.0, -0.1)
        with pytest.raises(ValueError):
            score_geometry(0.0, 1.1)

    def test_alignment_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(0.0, 0.0, alignment_score=1.5)
        with pytest.raises(ValueError):
            score_geometry(0.0, 0.0, alignment_score=-0.1)

    def test_non_negative(self):
        assert score_geometry(1.0, 1.0, 0.0) >= 0.0


# ─── score_coverage ───────────────────────────────────────────────────────────

class TestScoreCoverage:
    def test_full_coverage(self):
        assert score_coverage(10, 10) == pytest.approx(1.0)

    def test_zero_placed(self):
        assert score_coverage(0, 10) == pytest.approx(0.0)

    def test_half_placed(self):
        assert score_coverage(5, 10) == pytest.approx(0.5)

    def test_n_total_zero_raises(self):
        with pytest.raises(ValueError):
            score_coverage(0, 0)

    def test_n_placed_negative_raises(self):
        with pytest.raises(ValueError):
            score_coverage(-1, 10)

    def test_n_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            score_coverage(11, 10)

    def test_min_coverage_threshold(self):
        # 3/10 = 0.3 < min_coverage=0.5 → 0.0
        assert score_coverage(3, 10, min_coverage=0.5) == pytest.approx(0.0)

    def test_above_min_coverage(self):
        assert score_coverage(6, 10, min_coverage=0.5) == pytest.approx(0.6)

    def test_result_in_unit_interval(self):
        assert 0.0 <= score_coverage(7, 10) <= 1.0


# ─── score_seam_quality ───────────────────────────────────────────────────────

class TestScoreSeamQuality:
    def test_empty_returns_one(self):
        assert score_seam_quality([]) == pytest.approx(1.0)

    def test_all_ones(self):
        assert score_seam_quality([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert score_seam_quality([0.0, 0.0]) == pytest.approx(0.0)

    def test_average_value(self):
        assert score_seam_quality([0.4, 0.6]) == pytest.approx(0.5)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_seam_quality([0.5, 1.5])
        with pytest.raises(ValueError):
            score_seam_quality([-0.1, 0.5])

    def test_single_score(self):
        assert score_seam_quality([0.7]) == pytest.approx(0.7)

    def test_result_in_unit_interval(self):
        s = score_seam_quality([0.3, 0.7, 0.5])
        assert 0.0 <= s <= 1.0


# ─── score_uniqueness ─────────────────────────────────────────────────────────

class TestScoreUniqueness:
    def test_no_duplicates(self):
        assert score_uniqueness(10, 0) == pytest.approx(1.0)

    def test_all_duplicates(self):
        assert score_uniqueness(10, 10) == pytest.approx(0.0)

    def test_half_duplicates(self):
        assert score_uniqueness(10, 5) == pytest.approx(0.5)

    def test_zero_fragments_returns_one(self):
        assert score_uniqueness(0, 0) == pytest.approx(1.0)

    def test_negative_fragments_raises(self):
        with pytest.raises(ValueError):
            score_uniqueness(-1, 0)

    def test_negative_duplicates_raises(self):
        with pytest.raises(ValueError):
            score_uniqueness(10, -1)

    def test_duplicates_exceed_fragments_capped(self):
        s = score_uniqueness(5, 10)
        assert s == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        s = score_uniqueness(8, 3)
        assert 0.0 <= s <= 1.0


# ─── compute_assembly_score ───────────────────────────────────────────────────

class TestComputeAssemblyScore:
    def test_returns_report(self):
        r = compute_assembly_score(n_placed=5, n_total=10)
        assert isinstance(r, AssemblyScoreReport)

    def test_total_score_in_unit_interval(self):
        r = compute_assembly_score(n_placed=5, n_total=10)
        assert 0.0 <= r.total_score <= 1.0

    def test_n_fragments_matches_placed(self):
        r = compute_assembly_score(n_placed=7, n_total=10)
        assert r.n_fragments == 7

    def test_four_components(self):
        r = compute_assembly_score(n_placed=5, n_total=10)
        assert len(r.components) == 4

    def test_component_names(self):
        r = compute_assembly_score(n_placed=5, n_total=10)
        names = {c.name for c in r.components}
        assert names == {"geometry", "coverage", "seam", "uniqueness"}

    def test_perfect_assembly_high_score(self):
        r = compute_assembly_score(
            n_placed=10, n_total=10,
            overlap_ratio=0.0, gap_ratio=0.0,
            alignment_score=1.0,
            seam_scores=[1.0] * 5,
            n_duplicates=0,
        )
        assert r.total_score == pytest.approx(1.0)

    def test_custom_params_stored(self):
        p = AssemblyScorerParams(w_geometry=0.5, w_coverage=0.5,
                                 w_seam=0.0, w_uniqueness=0.0)
        r = compute_assembly_score(n_placed=5, n_total=10, params=p)
        assert r.params["w_geometry"] == pytest.approx(0.5)

    def test_seam_scores_none_treated_as_empty(self):
        r = compute_assembly_score(n_placed=5, n_total=10, seam_scores=None)
        seam_comp = r.component_by_name("seam")
        assert seam_comp is not None
        assert seam_comp.value == pytest.approx(1.0)

    def test_duplicates_reduce_uniqueness(self):
        r_no_dup = compute_assembly_score(n_placed=10, n_total=10, n_duplicates=0)
        r_dup = compute_assembly_score(n_placed=10, n_total=10, n_duplicates=5)
        uniq_no_dup = r_no_dup.component_by_name("uniqueness").value
        uniq_dup = r_dup.component_by_name("uniqueness").value
        assert uniq_dup < uniq_no_dup


# ─── compare_assemblies ───────────────────────────────────────────────────────

class TestCompareAssemblies:
    def _report(self, score: float) -> AssemblyScoreReport:
        return AssemblyScoreReport(total_score=score)

    def test_a_better(self):
        assert compare_assemblies(self._report(0.8), self._report(0.6)) == 1

    def test_b_better(self):
        assert compare_assemblies(self._report(0.5), self._report(0.9)) == -1

    def test_equal(self):
        assert compare_assemblies(self._report(0.7), self._report(0.7)) == 0

    def test_returns_int(self):
        result = compare_assemblies(self._report(0.8), self._report(0.5))
        assert isinstance(result, int)

    def test_tiny_difference_detected(self):
        assert compare_assemblies(
            self._report(0.500001), self._report(0.5)
        ) in (0, 1)


# ─── rank_assemblies ──────────────────────────────────────────────────────────

class TestRankAssemblies:
    def _report(self, score: float, n: int = 0) -> AssemblyScoreReport:
        return AssemblyScoreReport(total_score=score, n_fragments=n)

    def test_empty_returns_empty(self):
        assert rank_assemblies([]) == []

    def test_single_report_rank_one(self):
        result = rank_assemblies([self._report(0.5)])
        assert len(result) == 1
        assert result[0][0] == 1

    def test_sorted_descending(self):
        reports = [self._report(0.3), self._report(0.9), self._report(0.6)]
        ranked = rank_assemblies(reports)
        scores = [r.total_score for _, r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_one_based(self):
        reports = [self._report(s) for s in [0.8, 0.5, 0.3]]
        ranked = rank_assemblies(reports)
        ranks = [rank for rank, _ in ranked]
        assert ranks == [1, 2, 3]

    def test_returns_list_of_tuples(self):
        result = rank_assemblies([self._report(0.5)])
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_best_is_rank_one(self):
        reports = [self._report(0.2), self._report(0.9), self._report(0.5)]
        ranked = rank_assemblies(reports)
        rank1_report = ranked[0][1]
        assert rank1_report.total_score == pytest.approx(0.9)
