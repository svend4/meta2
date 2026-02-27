"""Tests for puzzle_reconstruction/verification/confidence_scorer.py."""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, Placement, EdgeSignature, EdgeSide
)
from puzzle_reconstruction.verification.confidence_scorer import (
    ScoreComponent,
    AssemblyConfidence,
    grade_from_score,
    score_edge_compat,
    score_layout,
    score_coverage,
    score_uniqueness,
    score_assembly_score,
    compute_confidence,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int, h=32, w=32) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((h, w, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(2)]
    return frag


def _make_entry(ei, ej, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_placement(fid: int, x: float = 0.0, y: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y), rotation=0.0)


def _make_assembly(frags, score=0.5):
    """Build Assembly with Placement list objects."""
    placements = [
        _make_placement(f.fragment_id, x=float(i * 100), y=0.0)
        for i, f in enumerate(frags)
    ]
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((len(frags), len(frags))),
        total_score=score,
    )


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=0.5))
    return entries


# ─── ScoreComponent ───────────────────────────────────────────────────────────

class TestScoreComponent:
    def test_basic_creation(self):
        comp = ScoreComponent(name="edge_compat", value=0.75, weight=1.5)
        assert comp.name == "edge_compat"
        assert abs(comp.value - 0.75) < 1e-9
        assert abs(comp.weight - 1.5) < 1e-9

    def test_weighted_value(self):
        comp = ScoreComponent(name="test", value=0.8, weight=2.0)
        assert abs(comp.weighted - 1.6) < 1e-9

    def test_default_description_empty(self):
        comp = ScoreComponent(name="x", value=0.5)
        assert comp.description == ""

    def test_repr_contains_name(self):
        comp = ScoreComponent(name="coverage", value=0.9)
        assert "coverage" in repr(comp)

    def test_default_weight_is_one(self):
        comp = ScoreComponent(name="x", value=0.5)
        assert comp.weight == 1.0


# ─── AssemblyConfidence ───────────────────────────────────────────────────────

class TestAssemblyConfidence:
    def _make_confidence(self):
        components = [
            ScoreComponent(name="edge_compat", value=0.8, weight=1.5),
            ScoreComponent(name="layout", value=0.6, weight=1.0),
            ScoreComponent(name="coverage", value=1.0, weight=0.8),
        ]
        return AssemblyConfidence(
            total=0.79,
            components=components,
            grade="B",
            n_fragments=3,
            assembly_method="greedy",
        )

    def test_basic_fields(self):
        conf = self._make_confidence()
        assert abs(conf.total - 0.79) < 1e-9
        assert conf.grade == "B"
        assert conf.n_fragments == 3
        assert conf.assembly_method == "greedy"

    def test_get_existing_component(self):
        conf = self._make_confidence()
        comp = conf.get("layout")
        assert comp is not None
        assert comp.name == "layout"

    def test_get_nonexistent_returns_none(self):
        conf = self._make_confidence()
        assert conf.get("nonexistent") is None

    def test_as_dict_contains_all_names(self):
        conf = self._make_confidence()
        d = conf.as_dict()
        assert "edge_compat" in d
        assert "layout" in d
        assert "coverage" in d

    def test_as_dict_values_are_floats(self):
        conf = self._make_confidence()
        d = conf.as_dict()
        for v in d.values():
            assert isinstance(v, float)

    def test_summary_is_string(self):
        conf = self._make_confidence()
        s = conf.summary()
        assert isinstance(s, str)
        assert "B" in s

    def test_repr_contains_grade(self):
        conf = self._make_confidence()
        assert "B" in repr(conf)


# ─── grade_from_score ─────────────────────────────────────────────────────────

class TestGradeFromScore:
    def test_grade_a(self):
        assert grade_from_score(1.0) == "A"
        assert grade_from_score(0.85) == "A"

    def test_grade_b(self):
        assert grade_from_score(0.70) == "B"
        assert grade_from_score(0.84) == "B"

    def test_grade_c(self):
        assert grade_from_score(0.55) == "C"
        assert grade_from_score(0.69) == "C"

    def test_grade_d(self):
        assert grade_from_score(0.40) == "D"
        assert grade_from_score(0.54) == "D"

    def test_grade_f(self):
        assert grade_from_score(0.0) == "F"
        assert grade_from_score(0.39) == "F"

    def test_boundary_0_85_is_a(self):
        assert grade_from_score(0.85) == "A"

    def test_just_below_0_85_is_b(self):
        assert grade_from_score(0.849) == "B"


# ─── score_edge_compat ────────────────────────────────────────────────────────

class TestScoreEdgeCompat:
    def test_returns_score_component(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.7)
        entries = _make_entries(frags)
        comp = score_edge_compat(asm, entries)
        assert isinstance(comp, ScoreComponent)
        assert comp.name == "edge_compat"

    def test_value_in_0_1(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.8)
        entries = _make_entries(frags)
        comp = score_edge_compat(asm, entries)
        assert 0.0 <= comp.value <= 1.0

    def test_empty_entries_returns_zero(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        comp = score_edge_compat(asm, [])
        assert comp.value == 0.0

    def test_empty_placements_returns_zero(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = Assembly(
            fragments=frags,
            placements=[],
            compat_matrix=np.array([]),
            total_score=0.5,
        )
        entries = _make_entries(frags)
        comp = score_edge_compat(asm, entries)
        assert comp.value == 0.0

    def test_weight_applied(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        entries = _make_entries(frags)
        comp = score_edge_compat(asm, entries, weight=2.0)
        assert abs(comp.weight - 2.0) < 1e-9


# ─── score_layout ─────────────────────────────────────────────────────────────

class TestScoreLayout:
    def test_returns_score_component(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.7)
        comp = score_layout(asm, frags)
        assert isinstance(comp, ScoreComponent)
        assert comp.name == "layout"

    def test_value_in_0_1(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.5)
        comp = score_layout(asm, frags)
        assert 0.0 <= comp.value <= 1.0

    def test_empty_fragments_returns_zero(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        comp = score_layout(asm, [])
        assert comp.value == 0.0

    def test_empty_placements_returns_zero(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = Assembly(
            fragments=frags,
            placements=[],
            compat_matrix=np.array([]),
            total_score=0.5,
        )
        comp = score_layout(asm, frags)
        assert comp.value == 0.0


# ─── score_coverage ───────────────────────────────────────────────────────────

class TestScoreCoverage:
    def test_full_coverage(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.5)
        all_ids = [f.fragment_id for f in frags]
        comp = score_coverage(asm, all_ids)
        assert abs(comp.value - 1.0) < 1e-9

    def test_partial_coverage(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        all_ids = [0, 1, 2, 3]  # 4 total, 2 placed
        comp = score_coverage(asm, all_ids)
        assert abs(comp.value - 0.5) < 1e-9

    def test_zero_coverage_on_empty_ids(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        comp = score_coverage(asm, [])
        assert comp.value == 0.0

    def test_returns_score_component(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        comp = score_coverage(asm, [0, 1])
        assert isinstance(comp, ScoreComponent)
        assert comp.name == "coverage"

    def test_value_in_0_1(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        comp = score_coverage(asm, [0, 1, 2, 3, 4])
        assert 0.0 <= comp.value <= 1.0

    def test_weight_applied(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        comp = score_coverage(asm, [0, 1], weight=0.5)
        assert abs(comp.weight - 0.5) < 1e-9


# ─── score_uniqueness ─────────────────────────────────────────────────────────

class TestScoreUniqueness:
    def test_all_unique_gives_one(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        comp = score_uniqueness(asm)
        assert abs(comp.value - 1.0) < 1e-9

    def test_duplicate_reduces_score(self):
        frags = [_make_fragment(0)]
        # Create assembly with duplicate placement
        asm = Assembly(
            fragments=frags,
            placements=[
                _make_placement(0, x=0.0),
                _make_placement(0, x=100.0),  # duplicate
            ],
            compat_matrix=np.array([]),
            total_score=0.5,
        )
        comp = score_uniqueness(asm)
        assert comp.value < 1.0

    def test_empty_placements_zero_score(self):
        asm = Assembly(
            fragments=[],
            placements=[],
            compat_matrix=np.array([]),
            total_score=0.0,
        )
        comp = score_uniqueness(asm)
        assert comp.value == 0.0

    def test_returns_score_component(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        comp = score_uniqueness(asm)
        assert isinstance(comp, ScoreComponent)
        assert comp.name == "uniqueness"

    def test_value_nonnegative(self):
        frags = [_make_fragment(0)]
        asm = Assembly(
            fragments=frags,
            placements=[_make_placement(0)] * 10,  # lots of duplicates
            compat_matrix=np.array([]),
            total_score=0.0,
        )
        comp = score_uniqueness(asm)
        assert comp.value >= 0.0


# ─── score_assembly_score ─────────────────────────────────────────────────────

class TestScoreAssemblyScore:
    def test_high_total_score(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.9)
        comp = score_assembly_score(asm)
        assert abs(comp.value - 0.9) < 1e-9

    def test_zero_total_score(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.0)
        comp = score_assembly_score(asm)
        assert abs(comp.value) < 1e-9

    def test_score_clipped_to_1(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=2.0)
        comp = score_assembly_score(asm)
        assert comp.value <= 1.0

    def test_returns_score_component(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        comp = score_assembly_score(asm)
        assert isinstance(comp, ScoreComponent)
        assert comp.name == "assembly_score"


# ─── compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_returns_assembly_confidence(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.7)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert isinstance(conf, AssemblyConfidence)

    def test_total_in_0_1(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.7)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert 0.0 <= conf.total <= 1.0

    def test_grade_is_string(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags, score=0.7)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert isinstance(conf.grade, str)
        assert conf.grade in ("A", "B", "C", "D", "F")

    def test_n_fragments_matches_placements(self):
        frags = [_make_fragment(i) for i in range(4)]
        asm = _make_assembly(frags, score=0.6)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert conf.n_fragments == 4

    def test_components_list_not_empty(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert len(conf.components) > 0

    def test_all_fragment_ids_provided(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags[:2], score=0.5)  # Only 2 placed
        entries = _make_entries(frags)
        all_ids = [f.fragment_id for f in frags]
        conf = compute_confidence(asm, frags[:2], entries, all_fragment_ids=all_ids)
        coverage_comp = conf.get("coverage")
        assert coverage_comp is not None
        assert coverage_comp.value < 1.0

    def test_custom_weights_applied(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        entries = _make_entries(frags)
        custom_weights = {"coverage": 5.0}
        conf = compute_confidence(asm, frags, entries, weights=custom_weights)
        coverage_comp = conf.get("coverage")
        assert coverage_comp is not None
        assert abs(coverage_comp.weight - 5.0) < 1e-9

    def test_assembly_method_stored(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.5)
        asm.method = "genetic"
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        assert conf.assembly_method == "genetic"

    def test_grade_consistent_with_total(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags, score=0.95)
        entries = _make_entries(frags)
        conf = compute_confidence(asm, frags, entries)
        expected_grade = grade_from_score(conf.total)
        assert conf.grade == expected_grade
