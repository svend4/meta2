"""Tests for puzzle_reconstruction/verification/assembly_scorer.py"""
import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _perfect_report(total: float = 0.95) -> AssemblyScoreReport:
    return AssemblyScoreReport(
        total_score=total,
        components=[],
        n_fragments=5,
    )


def _make_report(n_placed=5, n_total=5, overlap=0.0, gap=0.0) -> AssemblyScoreReport:
    return compute_assembly_score(n_placed, n_total,
                                  overlap_ratio=overlap, gap_ratio=gap)


# ─── TestScoreComponent ───────────────────────────────────────────────────────

class TestScoreComponent:
    def test_construction(self):
        c = ScoreComponent(name="geometry", value=0.8, weight=0.3)
        assert c.name == "geometry"
        assert c.value == pytest.approx(0.8)
        assert c.weight == pytest.approx(0.3)

    def test_value_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=1.1)

    def test_value_negative_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=-0.1)

    def test_weight_negative_raises(self):
        with pytest.raises(ValueError):
            ScoreComponent(name="x", value=0.5, weight=-0.1)

    def test_weight_zero_ok(self):
        c = ScoreComponent(name="x", value=0.5, weight=0.0)
        assert c.weighted_value() == pytest.approx(0.0)

    def test_weighted_value(self):
        c = ScoreComponent(name="x", value=0.8, weight=0.5)
        assert c.weighted_value() == pytest.approx(0.4)

    def test_boundary_value_zero(self):
        c = ScoreComponent(name="x", value=0.0)
        assert c.value == 0.0

    def test_boundary_value_one(self):
        c = ScoreComponent(name="x", value=1.0)
        assert c.value == 1.0


# ─── TestAssemblyScoreReport ──────────────────────────────────────────────────

class TestAssemblyScoreReport:
    def test_construction(self):
        r = AssemblyScoreReport(total_score=0.75, n_fragments=4)
        assert r.total_score == pytest.approx(0.75)
        assert r.n_fragments == 4

    def test_component_by_name_found(self):
        c = ScoreComponent(name="geometry", value=0.9)
        r = AssemblyScoreReport(total_score=0.9, components=[c])
        assert r.component_by_name("geometry") is c

    def test_component_by_name_missing(self):
        r = AssemblyScoreReport(total_score=0.9)
        assert r.component_by_name("missing") is None

    def test_to_dict_keys(self):
        r = _make_report()
        d = r.to_dict()
        assert "total_score" in d
        assert "n_fragments" in d
        assert "components" in d
        assert "params" in d

    def test_to_dict_component_list(self):
        r = _make_report()
        d = r.to_dict()
        assert isinstance(d["components"], list)
        for c in d["components"]:
            assert "name" in c
            assert "value" in c
            assert "weight" in c


# ─── TestAssemblyScorerParams ─────────────────────────────────────────────────

class TestAssemblyScorerParams:
    def test_defaults(self):
        p = AssemblyScorerParams()
        assert p.w_geometry == pytest.approx(0.30)
        assert p.w_coverage == pytest.approx(0.30)
        assert p.w_seam == pytest.approx(0.25)
        assert p.w_uniqueness == pytest.approx(0.15)
        assert p.min_coverage == pytest.approx(0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AssemblyScorerParams(w_geometry=-0.1)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            AssemblyScorerParams(w_geometry=0, w_coverage=0,
                                 w_seam=0, w_uniqueness=0)

    def test_total_weight(self):
        p = AssemblyScorerParams()
        assert p.total_weight() == pytest.approx(1.0)

    def test_custom_weights(self):
        p = AssemblyScorerParams(w_geometry=1.0, w_coverage=0,
                                  w_seam=0, w_uniqueness=0)
        assert p.total_weight() == pytest.approx(1.0)


# ─── TestScoreGeometry ────────────────────────────────────────────────────────

class TestScoreGeometry:
    def test_perfect(self):
        assert score_geometry(0.0, 0.0) == pytest.approx(1.0)

    def test_full_overlap(self):
        assert score_geometry(1.0, 0.0) == pytest.approx(0.5)

    def test_full_gap(self):
        assert score_geometry(0.0, 1.0) == pytest.approx(0.5)

    def test_both_penalties(self):
        # penalty = (0.5 + 0.5) / 2 = 0.5 → 1 - 0.5 = 0.5
        assert score_geometry(0.5, 0.5) == pytest.approx(0.5)

    def test_alignment_multiplier(self):
        v = score_geometry(0.0, 0.0, alignment_score=0.5)
        assert v == pytest.approx(0.5)

    def test_overlap_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(1.1, 0.0)

    def test_gap_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(0.0, -0.1)

    def test_alignment_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_geometry(0.0, 0.0, alignment_score=1.5)

    def test_returns_float(self):
        assert isinstance(score_geometry(0.0, 0.0), float)

    def test_result_in_0_1(self):
        for overlap in [0.0, 0.3, 0.7, 1.0]:
            for gap in [0.0, 0.2, 0.5]:
                v = score_geometry(overlap, gap)
                assert 0.0 <= v <= 1.0


# ─── TestScoreCoverage ────────────────────────────────────────────────────────

class TestScoreCoverage:
    def test_full_coverage(self):
        assert score_coverage(10, 10) == pytest.approx(1.0)

    def test_half_coverage(self):
        assert score_coverage(5, 10) == pytest.approx(0.5)

    def test_zero_placed(self):
        assert score_coverage(0, 10) == pytest.approx(0.0)

    def test_n_total_zero_raises(self):
        with pytest.raises(ValueError):
            score_coverage(0, 0)

    def test_n_placed_negative_raises(self):
        with pytest.raises(ValueError):
            score_coverage(-1, 5)

    def test_n_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            score_coverage(6, 5)

    def test_min_coverage_threshold(self):
        assert score_coverage(5, 10, min_coverage=0.6) == pytest.approx(0.0)

    def test_min_coverage_met(self):
        assert score_coverage(7, 10, min_coverage=0.6) == pytest.approx(0.7)

    def test_returns_float(self):
        assert isinstance(score_coverage(5, 10), float)


# ─── TestScoreSeamQuality ─────────────────────────────────────────────────────

class TestScoreSeamQuality:
    def test_empty_returns_one(self):
        assert score_seam_quality([]) == pytest.approx(1.0)

    def test_single_seam(self):
        assert score_seam_quality([0.8]) == pytest.approx(0.8)

    def test_average(self):
        assert score_seam_quality([0.4, 0.6, 1.0]) == pytest.approx(
            (0.4 + 0.6 + 1.0) / 3
        )

    def test_all_ones(self):
        assert score_seam_quality([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert score_seam_quality([0.0, 0.0]) == pytest.approx(0.0)

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            score_seam_quality([0.5, 1.1])

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            score_seam_quality([-0.1, 0.5])

    def test_returns_float(self):
        assert isinstance(score_seam_quality([0.5]), float)


# ─── TestScoreUniqueness ──────────────────────────────────────────────────────

class TestScoreUniqueness:
    def test_no_duplicates(self):
        assert score_uniqueness(10, 0) == pytest.approx(1.0)

    def test_all_duplicates(self):
        assert score_uniqueness(10, 10) == pytest.approx(0.0)

    def test_half_duplicates(self):
        assert score_uniqueness(10, 5) == pytest.approx(0.5)

    def test_zero_fragments(self):
        assert score_uniqueness(0, 0) == pytest.approx(1.0)

    def test_negative_fragments_raises(self):
        with pytest.raises(ValueError):
            score_uniqueness(-1, 0)

    def test_negative_duplicates_raises(self):
        with pytest.raises(ValueError):
            score_uniqueness(10, -1)

    def test_excess_duplicates_capped_at_zero(self):
        # More duplicates than fragments → capped at 0
        assert score_uniqueness(3, 10) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(score_uniqueness(5, 1), float)


# ─── TestComputeAssemblyScore ─────────────────────────────────────────────────

class TestComputeAssemblyScore:
    def test_returns_report(self):
        r = compute_assembly_score(5, 5)
        assert isinstance(r, AssemblyScoreReport)

    def test_perfect_score(self):
        r = compute_assembly_score(5, 5, overlap_ratio=0.0, gap_ratio=0.0,
                                   alignment_score=1.0, seam_scores=[1.0],
                                   n_duplicates=0)
        assert r.total_score == pytest.approx(1.0)

    def test_score_in_0_1(self):
        r = compute_assembly_score(3, 10, overlap_ratio=0.5, gap_ratio=0.3,
                                   seam_scores=[0.4, 0.6])
        assert 0.0 <= r.total_score <= 1.0

    def test_partial_coverage_reduces_score(self):
        full = compute_assembly_score(10, 10)
        partial = compute_assembly_score(5, 10)
        assert partial.total_score < full.total_score

    def test_four_components_present(self):
        r = compute_assembly_score(5, 5)
        names = {c.name for c in r.components}
        assert names == {"geometry", "coverage", "seam", "uniqueness"}

    def test_n_fragments_set(self):
        r = compute_assembly_score(7, 10)
        assert r.n_fragments == 7

    def test_custom_params(self):
        params = AssemblyScorerParams(w_geometry=1.0, w_coverage=0,
                                      w_seam=0, w_uniqueness=0)
        r = compute_assembly_score(5, 5, overlap_ratio=0.5, params=params)
        geom = r.component_by_name("geometry")
        assert r.total_score == pytest.approx(geom.value)

    def test_none_seam_scores_defaults(self):
        r = compute_assembly_score(5, 5, seam_scores=None)
        seam = r.component_by_name("seam")
        assert seam.value == pytest.approx(1.0)


# ─── TestCompareAssemblies ────────────────────────────────────────────────────

class TestCompareAssemblies:
    def test_a_better(self):
        a = _perfect_report(0.9)
        b = _perfect_report(0.5)
        assert compare_assemblies(a, b) == 1

    def test_b_better(self):
        a = _perfect_report(0.3)
        b = _perfect_report(0.8)
        assert compare_assemblies(a, b) == -1

    def test_equal(self):
        a = _perfect_report(0.7)
        b = _perfect_report(0.7)
        assert compare_assemblies(a, b) == 0

    def test_returns_int(self):
        assert isinstance(compare_assemblies(_perfect_report(), _perfect_report()), int)


# ─── TestRankAssemblies ───────────────────────────────────────────────────────

class TestRankAssemblies:
    def test_empty_returns_empty(self):
        assert rank_assemblies([]) == []

    def test_single_report(self):
        r = _perfect_report(0.8)
        ranked = rank_assemblies([r])
        assert len(ranked) == 1
        assert ranked[0][0] == 1
        assert ranked[0][1] is r

    def test_sorted_descending(self):
        r1 = _perfect_report(0.5)
        r2 = _perfect_report(0.9)
        r3 = _perfect_report(0.3)
        ranked = rank_assemblies([r1, r2, r3])
        scores = [r.total_score for _, r in ranked]
        assert scores[0] >= scores[1] >= scores[2]

    def test_rank_starts_at_one(self):
        r1 = _perfect_report(0.8)
        r2 = _perfect_report(0.4)
        ranked = rank_assemblies([r1, r2])
        assert ranked[0][0] == 1
        assert ranked[1][0] == 2

    def test_length_preserved(self):
        reports = [_perfect_report(i / 10) for i in range(5)]
        ranked = rank_assemblies(reports)
        assert len(ranked) == 5
