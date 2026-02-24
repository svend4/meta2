"""Extra tests for puzzle_reconstruction/verification/assembly_scorer.py"""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _report(total=0.7):
    return AssemblyScoreReport(total_score=total, n_fragments=5)


def _make(n_placed=5, n_total=5, **kw):
    return compute_assembly_score(n_placed, n_total, **kw)


# ─── TestScoreComponentExtra ──────────────────────────────────────────────────

class TestScoreComponentExtra:
    def test_weight_1_valid(self):
        c = ScoreComponent(name="x", value=0.5, weight=1.0)
        assert c.weight == pytest.approx(1.0)

    def test_large_weight_valid(self):
        c = ScoreComponent(name="x", value=0.5, weight=100.0)
        assert c.weight == pytest.approx(100.0)

    def test_default_weight_stored(self):
        c = ScoreComponent(name="x", value=0.5)
        assert c.weight >= 0.0

    def test_weighted_value_1_times_1(self):
        c = ScoreComponent(name="x", value=1.0, weight=1.0)
        assert c.weighted_value() == pytest.approx(1.0)

    def test_weighted_value_half(self):
        c = ScoreComponent(name="x", value=0.6, weight=0.5)
        assert c.weighted_value() == pytest.approx(0.3)

    def test_value_0_5_stored(self):
        c = ScoreComponent(name="geom", value=0.5)
        assert c.value == pytest.approx(0.5)

    def test_name_stored(self):
        c = ScoreComponent(name="coverage", value=0.9)
        assert c.name == "coverage"


# ─── TestAssemblyScoreReportExtra ─────────────────────────────────────────────

class TestAssemblyScoreReportExtra:
    def test_empty_components_ok(self):
        r = AssemblyScoreReport(total_score=0.5, components=[], n_fragments=5)
        assert r.components == []

    def test_total_score_0(self):
        r = AssemblyScoreReport(total_score=0.0, n_fragments=5)
        assert r.total_score == pytest.approx(0.0)

    def test_total_score_1(self):
        r = AssemblyScoreReport(total_score=1.0, n_fragments=5)
        assert r.total_score == pytest.approx(1.0)

    def test_multiple_components(self):
        comps = [ScoreComponent(name=f"c{i}", value=0.5) for i in range(4)]
        r = AssemblyScoreReport(total_score=0.5, components=comps)
        assert len(r.components) == 4

    def test_to_dict_total_score_value(self):
        r = _make()
        d = r.to_dict()
        assert d["total_score"] == pytest.approx(r.total_score)

    def test_component_by_name_returns_component(self):
        c = ScoreComponent(name="seam", value=0.75)
        r = AssemblyScoreReport(total_score=0.75, components=[c])
        found = r.component_by_name("seam")
        assert found is c


# ─── TestAssemblyScorerParamsExtra ────────────────────────────────────────────

class TestAssemblyScorerParamsExtra:
    def test_w_coverage_zero_ok_if_others_nonzero(self):
        p = AssemblyScorerParams(w_geometry=1.0, w_coverage=0.0,
                                  w_seam=0.0, w_uniqueness=0.0)
        assert p.total_weight() == pytest.approx(1.0)

    def test_min_coverage_0_5(self):
        p = AssemblyScorerParams(min_coverage=0.5)
        assert p.min_coverage == pytest.approx(0.5)

    def test_default_min_coverage_zero(self):
        p = AssemblyScorerParams()
        assert p.min_coverage == pytest.approx(0.0)

    def test_all_equal_weights(self):
        p = AssemblyScorerParams(w_geometry=0.25, w_coverage=0.25,
                                  w_seam=0.25, w_uniqueness=0.25)
        assert p.total_weight() == pytest.approx(1.0)

    def test_custom_large_weights(self):
        p = AssemblyScorerParams(w_geometry=2.0, w_coverage=2.0,
                                  w_seam=0.0, w_uniqueness=0.0)
        assert p.total_weight() == pytest.approx(4.0)


# ─── TestScoreGeometryExtra ───────────────────────────────────────────────────

class TestScoreGeometryExtra:
    def test_small_overlap_close_to_1(self):
        v = score_geometry(0.01, 0.0)
        assert v > 0.9

    def test_small_gap_close_to_1(self):
        v = score_geometry(0.0, 0.01)
        assert v > 0.9

    def test_alignment_1_ok(self):
        v = score_geometry(0.0, 0.0, alignment_score=1.0)
        assert v == pytest.approx(1.0)

    def test_alignment_0_zero(self):
        v = score_geometry(0.0, 0.0, alignment_score=0.0)
        assert v == pytest.approx(0.0)

    def test_all_combos_in_range(self):
        for overlap in (0.0, 0.2, 0.5, 1.0):
            for gap in (0.0, 0.2, 0.5):
                v = score_geometry(overlap, gap, alignment_score=0.5)
                assert 0.0 <= v <= 1.0

    def test_alignment_neg_raises(self):
        with pytest.raises(ValueError):
            score_geometry(0.0, 0.0, alignment_score=-0.1)


# ─── TestScoreCoverageExtra ───────────────────────────────────────────────────

class TestScoreCoverageExtra:
    def test_one_of_one_full(self):
        assert score_coverage(1, 1) == pytest.approx(1.0)

    def test_one_of_ten(self):
        assert score_coverage(1, 10) == pytest.approx(0.1)

    def test_min_coverage_1_zero_for_all(self):
        # min_coverage=1.0 means only 100% placement counts
        v = score_coverage(9, 10, min_coverage=1.0)
        assert v == pytest.approx(0.0)

    def test_min_coverage_met_exact(self):
        # 7/10 = 0.7 exactly meets 0.7 threshold
        v = score_coverage(7, 10, min_coverage=0.7)
        assert v == pytest.approx(0.7)

    def test_n_total_1_and_placed_1(self):
        assert score_coverage(1, 1) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(score_coverage(3, 5), float)


# ─── TestScoreSeamQualityExtra ────────────────────────────────────────────────

class TestScoreSeamQualityExtra:
    def test_two_seams(self):
        v = score_seam_quality([0.4, 0.8])
        assert v == pytest.approx(0.6)

    def test_ten_seams(self):
        seams = [0.5] * 10
        assert score_seam_quality(seams) == pytest.approx(0.5)

    def test_boundary_0_and_1(self):
        assert score_seam_quality([0.0, 1.0]) == pytest.approx(0.5)

    def test_five_random_seams_in_range(self):
        seams = [0.1, 0.3, 0.5, 0.7, 0.9]
        v = score_seam_quality(seams)
        assert 0.0 <= v <= 1.0

    def test_single_zero(self):
        assert score_seam_quality([0.0]) == pytest.approx(0.0)

    def test_single_one(self):
        assert score_seam_quality([1.0]) == pytest.approx(1.0)


# ─── TestScoreUniquenessExtra ─────────────────────────────────────────────────

class TestScoreUniquenessExtra:
    def test_one_frag_zero_dups(self):
        assert score_uniqueness(1, 0) == pytest.approx(1.0)

    def test_five_frags_two_dups(self):
        assert score_uniqueness(5, 2) == pytest.approx(0.6)

    def test_large_count_no_dups(self):
        assert score_uniqueness(100, 0) == pytest.approx(1.0)

    def test_large_count_all_dups(self):
        assert score_uniqueness(100, 100) == pytest.approx(0.0)

    def test_excess_dups_zero(self):
        assert score_uniqueness(5, 10) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(score_uniqueness(5, 2), float)


# ─── TestComputeAssemblyScoreExtra ────────────────────────────────────────────

class TestComputeAssemblyScoreExtra:
    def test_seam_scores_half(self):
        r = _make(seam_scores=[0.5, 0.5])
        seam = r.component_by_name("seam")
        assert seam.value == pytest.approx(0.5)

    def test_n_duplicates_3(self):
        r = _make(n_duplicates=3)
        uniq = r.component_by_name("uniqueness")
        assert uniq.value == pytest.approx(0.4)

    def test_partial_coverage_coverage_component(self):
        r = _make(n_placed=3, n_total=10)
        cov = r.component_by_name("coverage")
        assert cov.value == pytest.approx(0.3)

    def test_zero_coverage_score(self):
        r = _make(n_placed=0, n_total=10)
        cov = r.component_by_name("coverage")
        assert cov.value == pytest.approx(0.0)

    def test_params_stored_in_report(self):
        r = _make()
        assert "params" in r.to_dict()

    def test_geometry_component_present(self):
        r = _make(overlap_ratio=0.2, gap_ratio=0.1)
        geom = r.component_by_name("geometry")
        assert geom is not None
        assert 0.0 <= geom.value <= 1.0


# ─── TestCompareAssembliesExtra ───────────────────────────────────────────────

class TestCompareAssembliesExtra:
    def test_five_comparisons_consistent(self):
        reports = [_report(v / 10.0) for v in range(1, 6)]
        for i in range(len(reports) - 1):
            assert compare_assemblies(reports[i], reports[i + 1]) == -1

    def test_reverse_consistent(self):
        a = _report(0.8)
        b = _report(0.3)
        assert compare_assemblies(a, b) == -compare_assemblies(b, a)

    def test_equal_edge_zero(self):
        a = _report(0.0)
        b = _report(0.0)
        assert compare_assemblies(a, b) == 0

    def test_equal_edge_one(self):
        a = _report(1.0)
        b = _report(1.0)
        assert compare_assemblies(a, b) == 0

    def test_result_in_neg1_0_1(self):
        for x, y in [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]:
            result = compare_assemblies(_report(x), _report(y))
            assert result in (-1, 0, 1)


# ─── TestRankAssembliesExtra ──────────────────────────────────────────────────

class TestRankAssembliesExtra:
    def test_all_equal_scores(self):
        reports = [_report(0.5) for _ in range(4)]
        ranked = rank_assemblies(reports)
        assert len(ranked) == 4

    def test_ten_reports_length(self):
        reports = [_report(i / 10.0) for i in range(10)]
        ranked = rank_assemblies(reports)
        assert len(ranked) == 10

    def test_rank_1_is_best(self):
        r1 = _report(0.9)
        r2 = _report(0.5)
        r3 = _report(0.1)
        ranked = rank_assemblies([r1, r2, r3])
        assert ranked[0][1].total_score == pytest.approx(0.9)

    def test_ranks_are_unique_integers(self):
        reports = [_report(i / 5.0) for i in range(1, 6)]
        ranked = rank_assemblies(reports)
        ranks = [rnk for rnk, _ in ranked]
        assert sorted(ranks) == list(range(1, 6))

    def test_returns_list_of_tuples(self):
        ranked = rank_assemblies([_report(0.5)])
        assert isinstance(ranked, list)
        assert isinstance(ranked[0], tuple)
        assert len(ranked[0]) == 2
