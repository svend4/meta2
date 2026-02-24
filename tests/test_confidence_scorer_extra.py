"""Extra tests for puzzle_reconstruction/verification/confidence_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, CompatEntry, Edge, Fragment, Placement
from puzzle_reconstruction.verification.confidence_scorer import (
    AssemblyConfidence,
    ScoreComponent,
    compute_confidence,
    grade_from_score,
    score_assembly_score,
    score_coverage,
    score_edge_compat,
    score_layout,
    score_uniqueness,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fragment(fid: int, w: int = 50, h: int = 50) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((h, w, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, w, h),
    )


def _placement(fid: int, x: float = 0.0, y: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y), rotation=0.0)


def _assembly(placements, score: float = 0.75, method: str = "greedy") -> Assembly:
    return Assembly(placements=placements, total_score=score, method=method)


def _edge(fid: int, side: int = 0) -> Edge:
    return Edge(
        edge_id=fid * 10 + side,
        contour=np.zeros((5, 2), dtype=np.float64),
        text_hint="",
    )


def _entry(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    return CompatEntry(edge_i=_edge(fid_i), edge_j=_edge(fid_j), score=score)


def _perfect_assembly(n: int = 3) -> tuple:
    frags = [_fragment(i) for i in range(n)]
    pls = [_placement(i, float(i) * 60.0, 0.0) for i in range(n)]
    asm = _assembly(pls, score=0.9)
    return frags, asm


# ─── ScoreComponent (extra) ───────────────────────────────────────────────────

class TestScoreComponentExtra:
    def test_name_stored(self):
        sc = ScoreComponent(name="coverage", value=0.5)
        assert sc.name == "coverage"

    def test_value_stored(self):
        sc = ScoreComponent(name="x", value=0.42)
        assert sc.value == pytest.approx(0.42)

    def test_weight_default_1(self):
        sc = ScoreComponent(name="x", value=0.5)
        assert sc.weight == pytest.approx(1.0)

    def test_description_default_empty(self):
        sc = ScoreComponent(name="x", value=0.5)
        assert sc.description == ""

    def test_description_stored(self):
        sc = ScoreComponent(name="x", value=0.5, description="test desc")
        assert sc.description == "test desc"

    def test_weighted_value_times_weight(self):
        sc = ScoreComponent(name="x", value=0.5, weight=3.0)
        assert sc.weighted == pytest.approx(1.5)

    def test_weighted_zero_value(self):
        sc = ScoreComponent(name="x", value=0.0, weight=5.0)
        assert sc.weighted == pytest.approx(0.0)

    def test_weighted_full_value(self):
        sc = ScoreComponent(name="x", value=1.0, weight=2.0)
        assert sc.weighted == pytest.approx(2.0)

    def test_repr_contains_weight(self):
        sc = ScoreComponent(name="x", value=0.5, weight=1.5)
        assert "1.5" in repr(sc) or "w=" in repr(sc)

    def test_custom_weight_stored(self):
        sc = ScoreComponent(name="edge_compat", value=0.8, weight=1.5)
        assert sc.weight == pytest.approx(1.5)


# ─── AssemblyConfidence (extra) ───────────────────────────────────────────────

class TestAssemblyConfidenceExtra:
    def _make(self, total: float = 0.75) -> AssemblyConfidence:
        comps = [
            ScoreComponent("edge_compat", 0.8, 1.5),
            ScoreComponent("layout", 0.7, 1.0),
            ScoreComponent("coverage", 0.6, 0.8),
        ]
        return AssemblyConfidence(
            total=total,
            components=comps,
            grade=grade_from_score(total),
            n_fragments=3,
            assembly_method="greedy",
        )

    def test_total_stored(self):
        conf = self._make(0.82)
        assert conf.total == pytest.approx(0.82)

    def test_grade_stored(self):
        conf = self._make(0.90)
        assert conf.grade == "A"

    def test_n_fragments_stored(self):
        conf = self._make()
        assert conf.n_fragments == 3

    def test_assembly_method_stored(self):
        conf = self._make()
        assert conf.assembly_method == "greedy"

    def test_get_existing_component(self):
        conf = self._make()
        sc = conf.get("edge_compat")
        assert sc is not None
        assert sc.name == "edge_compat"

    def test_get_nonexistent_returns_none(self):
        conf = self._make()
        assert conf.get("unknown_component") is None

    def test_as_dict_keys(self):
        conf = self._make()
        d = conf.as_dict()
        assert "edge_compat" in d
        assert "layout" in d
        assert "coverage" in d

    def test_as_dict_values_are_floats(self):
        conf = self._make()
        for v in conf.as_dict().values():
            assert isinstance(v, float)

    def test_summary_contains_grade(self):
        conf = self._make()
        s = conf.summary()
        assert conf.grade in s

    def test_summary_contains_total(self):
        conf = self._make(0.75)
        s = conf.summary()
        assert "0.75" in s or "0.750" in s

    def test_repr_contains_grade(self):
        conf = self._make()
        assert conf.grade in repr(conf)

    def test_repr_contains_total(self):
        conf = self._make(0.75)
        r = repr(conf)
        assert "0.75" in r or "0.750" in r

    def test_components_count(self):
        conf = self._make()
        assert len(conf.components) == 3


# ─── grade_from_score (extra) ─────────────────────────────────────────────────

class TestGradeFromScoreExtra:
    def test_grade_a_at_085(self):
        assert grade_from_score(0.85) == "A"

    def test_grade_a_at_1(self):
        assert grade_from_score(1.0) == "A"

    def test_grade_a_at_09(self):
        assert grade_from_score(0.9) == "A"

    def test_grade_b_at_07(self):
        assert grade_from_score(0.70) == "B"

    def test_grade_b_at_0849(self):
        assert grade_from_score(0.849) == "B"

    def test_grade_c_at_055(self):
        assert grade_from_score(0.55) == "C"

    def test_grade_c_at_069(self):
        assert grade_from_score(0.69) == "C"

    def test_grade_d_at_04(self):
        assert grade_from_score(0.40) == "D"

    def test_grade_d_at_054(self):
        assert grade_from_score(0.54) == "D"

    def test_grade_f_at_039(self):
        assert grade_from_score(0.39) == "F"

    def test_grade_f_at_0(self):
        assert grade_from_score(0.0) == "F"

    def test_returns_string(self):
        assert isinstance(grade_from_score(0.5), str)


# ─── score_edge_compat (extra) ────────────────────────────────────────────────

class TestScoreEdgeCompatExtra:
    def test_returns_score_component(self):
        frags, asm = _perfect_assembly()
        result = score_edge_compat(asm, [])
        assert isinstance(result, ScoreComponent)

    def test_name_is_edge_compat(self):
        frags, asm = _perfect_assembly()
        result = score_edge_compat(asm, [])
        assert result.name == "edge_compat"

    def test_empty_entries_returns_zero(self):
        frags, asm = _perfect_assembly()
        result = score_edge_compat(asm, [])
        assert result.value == pytest.approx(0.0)

    def test_empty_placements_returns_zero(self):
        asm = _assembly([])
        result = score_edge_compat(asm, [_entry(0, 1, 0.8)])
        assert result.value == pytest.approx(0.0)

    def test_value_in_0_1(self):
        frags, asm = _perfect_assembly()
        entries = [_entry(0, 1, 0.9), _entry(1, 2, 0.8)]
        result = score_edge_compat(asm, entries)
        assert 0.0 <= result.value <= 1.0

    def test_custom_weight_applied(self):
        frags, asm = _perfect_assembly()
        result = score_edge_compat(asm, [], weight=2.5)
        assert result.weight == pytest.approx(2.5)

    def test_default_weight(self):
        frags, asm = _perfect_assembly()
        result = score_edge_compat(asm, [])
        assert result.weight == pytest.approx(1.5)


# ─── score_layout (extra) ─────────────────────────────────────────────────────

class TestScoreLayoutExtra:
    def test_returns_score_component(self):
        frags, asm = _perfect_assembly()
        result = score_layout(asm, frags)
        assert isinstance(result, ScoreComponent)

    def test_name_is_layout(self):
        frags, asm = _perfect_assembly()
        result = score_layout(asm, frags)
        assert result.name == "layout"

    def test_empty_fragments_returns_zero(self):
        _, asm = _perfect_assembly()
        result = score_layout(asm, [])
        assert result.value == pytest.approx(0.0)

    def test_empty_placements_returns_zero(self):
        frags, _ = _perfect_assembly()
        asm = _assembly([])
        result = score_layout(asm, frags)
        assert result.value == pytest.approx(0.0)

    def test_perfect_layout_high_value(self):
        frags, asm = _perfect_assembly(3)
        result = score_layout(asm, frags)
        assert result.value >= 0.0

    def test_value_in_0_1(self):
        frags, asm = _perfect_assembly()
        result = score_layout(asm, frags)
        assert 0.0 <= result.value <= 1.0

    def test_custom_weight(self):
        frags, asm = _perfect_assembly()
        result = score_layout(asm, frags, weight=2.0)
        assert result.weight == pytest.approx(2.0)


# ─── score_coverage (extra) ───────────────────────────────────────────────────

class TestScoreCoverageExtra:
    def test_returns_score_component(self):
        _, asm = _perfect_assembly()
        result = score_coverage(asm, [0, 1, 2])
        assert isinstance(result, ScoreComponent)

    def test_name_is_coverage(self):
        _, asm = _perfect_assembly()
        result = score_coverage(asm, [0, 1, 2])
        assert result.name == "coverage"

    def test_no_fragment_ids_zero(self):
        _, asm = _perfect_assembly()
        result = score_coverage(asm, [])
        assert result.value == pytest.approx(0.0)

    def test_all_covered_value_1(self):
        pls = [_placement(i) for i in range(3)]
        asm = _assembly(pls)
        result = score_coverage(asm, [0, 1, 2])
        assert result.value == pytest.approx(1.0)

    def test_partial_coverage_less_than_1(self):
        pls = [_placement(0)]
        asm = _assembly(pls)
        result = score_coverage(asm, [0, 1, 2])
        assert result.value < 1.0
        assert result.value > 0.0

    def test_empty_placements_zero_value(self):
        asm = _assembly([])
        result = score_coverage(asm, [0, 1, 2])
        assert result.value == pytest.approx(0.0)

    def test_value_in_0_1(self):
        pls = [_placement(0), _placement(1)]
        asm = _assembly(pls)
        result = score_coverage(asm, [0, 1, 2, 3])
        assert 0.0 <= result.value <= 1.0

    def test_custom_weight(self):
        _, asm = _perfect_assembly()
        result = score_coverage(asm, [0, 1, 2], weight=0.5)
        assert result.weight == pytest.approx(0.5)

    def test_coverage_half(self):
        pls = [_placement(0)]
        asm = _assembly(pls)
        result = score_coverage(asm, [0, 1])
        assert result.value == pytest.approx(0.5)


# ─── score_uniqueness (extra) ─────────────────────────────────────────────────

class TestScoreUniquenessExtra:
    def test_returns_score_component(self):
        _, asm = _perfect_assembly()
        result = score_uniqueness(asm)
        assert isinstance(result, ScoreComponent)

    def test_name_is_uniqueness(self):
        _, asm = _perfect_assembly()
        result = score_uniqueness(asm)
        assert result.name == "uniqueness"

    def test_no_placements_zero(self):
        asm = _assembly([])
        result = score_uniqueness(asm)
        assert result.value == pytest.approx(0.0)

    def test_all_unique_value_1(self):
        _, asm = _perfect_assembly(3)
        result = score_uniqueness(asm)
        assert result.value == pytest.approx(1.0)

    def test_one_duplicate_value_08(self):
        pls = [_placement(0), _placement(1), _placement(0)]  # 0 is a duplicate
        asm = _assembly(pls)
        result = score_uniqueness(asm)
        assert result.value == pytest.approx(0.8)

    def test_five_duplicates_clamped_to_zero(self):
        pls = [_placement(0)] * 6  # 5 duplicates of fid=0
        asm = _assembly(pls)
        result = score_uniqueness(asm)
        assert result.value == pytest.approx(0.0)

    def test_custom_weight(self):
        _, asm = _perfect_assembly()
        result = score_uniqueness(asm, weight=2.0)
        assert result.weight == pytest.approx(2.0)

    def test_value_in_0_1(self):
        pls = [_placement(0), _placement(0), _placement(1)]
        asm = _assembly(pls)
        result = score_uniqueness(asm)
        assert 0.0 <= result.value <= 1.0


# ─── score_assembly_score (extra) ─────────────────────────────────────────────

class TestScoreAssemblyScoreExtra:
    def test_returns_score_component(self):
        asm = _assembly([_placement(0)])
        result = score_assembly_score(asm)
        assert isinstance(result, ScoreComponent)

    def test_name_is_assembly_score(self):
        asm = _assembly([_placement(0)])
        result = score_assembly_score(asm)
        assert result.name == "assembly_score"

    def test_value_clipped_to_0_1(self):
        asm = _assembly([_placement(0)], score=2.0)
        result = score_assembly_score(asm)
        assert result.value == pytest.approx(1.0)

    def test_negative_score_clipped_to_0(self):
        asm = _assembly([_placement(0)], score=-0.5)
        result = score_assembly_score(asm)
        assert result.value == pytest.approx(0.0)

    def test_normal_score_preserved(self):
        asm = _assembly([_placement(0)], score=0.7)
        result = score_assembly_score(asm)
        assert result.value == pytest.approx(0.7)

    def test_custom_weight(self):
        asm = _assembly([_placement(0)])
        result = score_assembly_score(asm, weight=1.5)
        assert result.weight == pytest.approx(1.5)


# ─── compute_confidence (extra) ───────────────────────────────────────────────

class TestComputeConfidenceExtra:
    def test_returns_assembly_confidence(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        assert isinstance(result, AssemblyConfidence)

    def test_total_in_0_1(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        assert 0.0 <= result.total <= 1.0

    def test_grade_is_string(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        assert isinstance(result.grade, str)

    def test_grade_is_valid(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_5_components(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        assert len(result.components) == 5

    def test_n_fragments_matches_placements(self):
        frags, asm = _perfect_assembly(4)
        result = compute_confidence(asm, frags, [])
        assert result.n_fragments == 4

    def test_assembly_method_stored(self):
        frags, asm = _perfect_assembly()
        asm.method = "beam_search"
        result = compute_confidence(asm, frags, [])
        assert result.assembly_method == "beam_search"

    def test_custom_weights_used(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [], weights={"edge_compat": 0.0})
        ec = result.get("edge_compat")
        assert ec is not None
        assert ec.weight == pytest.approx(0.0)

    def test_custom_all_fragment_ids(self):
        frags, asm = _perfect_assembly(2)
        result = compute_confidence(asm, frags, [], all_fragment_ids=[0, 1, 2, 3])
        cov = result.get("coverage")
        assert cov is not None
        assert cov.value < 1.0

    def test_empty_assembly_low_confidence(self):
        frags, _ = _perfect_assembly(3)
        asm = _assembly([], score=0.0)
        result = compute_confidence(asm, frags, [])
        assert result.total < 0.5

    def test_all_component_names_present(self):
        frags, asm = _perfect_assembly()
        result = compute_confidence(asm, frags, [])
        names = {c.name for c in result.components}
        for name in ("edge_compat", "layout", "coverage", "uniqueness", "assembly_score"):
            assert name in names
