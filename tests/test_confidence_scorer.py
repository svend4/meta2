"""
Тесты для puzzle_reconstruction/verification/confidence_scorer.py

Покрытие:
    ScoreComponent    — weighted, repr, defaults
    AssemblyConfidence — get (найден / не найден), as_dict, summary, repr,
                         n_fragments, assembly_method
    grade_from_score  — границы A/B/C/D/F, граничные значения
    score_edge_compat — пустые entries → value=0, пустые placements → value=0,
                        score ∈ [0,1], вес применяется
    score_layout      — пустые фрагменты → value=0, идеальное → высокое,
                        перекрытие → пониженное
    score_coverage    — нет fid → value=0, все fid → 1.0,
                        частичное покрытие < 1.0
    score_uniqueness  — нет дубликатов → 1.0, 1 дубликат → 0.8,
                        много дубликатов → max(0, ...)
    score_assembly_score — clip в [0,1]
    compute_confidence — тип AssemblyConfidence, total ∈ [0,1],
                         grade строка, 5 компонент, custom weights,
                         n_fragments = len(placements), assembly_method = method
"""
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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _fragment(fid: int, w: int = 50, h: int = 50) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((h, w, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, w, h),
    )


def _placement(fid: int, x: float = 0.0, y: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y), rotation=0.0)


def _assembly(placements, score=0.75, method="greedy"):
    return Assembly(placements=placements, total_score=score, method=method)


def _edge(fid: int, side: int = 0) -> Edge:
    return Edge(
        edge_id=fid * 10 + side,
        contour=np.zeros((5, 2), dtype=np.float64),
        text_hint="",
    )


def _entry(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=_edge(fid_i),
        edge_j=_edge(fid_j),
        score=score,
    )


def _perfect_assembly(n: int = 3) -> tuple:
    """N фрагментов в ряд без перекрытий."""
    frags = [_fragment(i) for i in range(n)]
    pls   = [_placement(i, i * 60.0, 0.0) for i in range(n)]
    asm   = _assembly(pls, score=0.9)
    return frags, asm


# ─── ScoreComponent ───────────────────────────────────────────────────────────

class TestScoreComponent:
    def test_weighted(self):
        sc = ScoreComponent(name="x", value=0.8, weight=2.0)
        assert sc.weighted == pytest.approx(1.6)

    def test_weighted_zero_weight(self):
        sc = ScoreComponent(name="x", value=1.0, weight=0.0)
        assert sc.weighted == pytest.approx(0.0)

    def test_repr_contains_name(self):
        sc = ScoreComponent(name="edge_compat", value=0.7, weight=1.5)
        assert "edge_compat" in repr(sc)

    def test_repr_contains_value(self):
        sc = ScoreComponent(name="x", value=0.654, weight=1.0)
        assert "0.654" in repr(sc)

    def test_defaults(self):
        sc = ScoreComponent(name="y", value=0.5)
        assert sc.weight       == pytest.approx(1.0)
        assert sc.description  == ""


# ─── AssemblyConfidence ───────────────────────────────────────────────────────

class TestAssemblyConfidence:
    def _make_conf(self, total=0.75):
        components = [
            ScoreComponent("a", 0.8, 1.0),
            ScoreComponent("b", 0.7, 1.0),
        ]
        return AssemblyConfidence(
            total=total,
            components=components,
            grade=grade_from_score(total),
            n_fragments=3,
            assembly_method="beam",
        )

    def test_get_existing(self):
        c = self._make_conf()
        assert c.get("a") is not None
        assert c.get("a").value == pytest.approx(0.8)

    def test_get_missing(self):
        c = self._make_conf()
        assert c.get("nonexistent") is None

    def test_as_dict(self):
        c = self._make_conf()
        d = c.as_dict()
        assert "a" in d and "b" in d
        assert d["a"] == pytest.approx(0.8)

    def test_summary_contains_grade(self):
        c = self._make_conf(0.8)
        assert c.grade in c.summary()

    def test_summary_contains_total(self):
        c = self._make_conf(0.75)
        assert "0.750" in c.summary()

    def test_repr_contains_grade(self):
        c = self._make_conf(0.9)
        assert "A" in repr(c)

    def test_repr_contains_n_fragments(self):
        c = self._make_conf()
        assert "3" in repr(c)

    def test_assembly_method_stored(self):
        c = self._make_conf()
        assert c.assembly_method == "beam"

    def test_n_fragments_stored(self):
        c = self._make_conf()
        assert c.n_fragments == 3


# ─── grade_from_score ─────────────────────────────────────────────────────────

class TestGradeFromScore:
    def test_grade_a(self):
        assert grade_from_score(0.85) == "A"
        assert grade_from_score(1.00) == "A"

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
        assert grade_from_score(0.39) == "F"
        assert grade_from_score(0.00) == "F"

    def test_boundary_85(self):
        assert grade_from_score(0.85) == "A"
        assert grade_from_score(0.849) == "B"

    def test_boundary_70(self):
        assert grade_from_score(0.70) == "B"
        assert grade_from_score(0.699) == "C"

    def test_boundary_40(self):
        assert grade_from_score(0.40) == "D"
        assert grade_from_score(0.399) == "F"


# ─── score_edge_compat ────────────────────────────────────────────────────────

class TestScoreEdgeCompat:
    def test_empty_entries_zero(self):
        frags, asm = _perfect_assembly(3)
        sc = score_edge_compat(asm, entries=[], weight=1.0)
        assert sc.value == pytest.approx(0.0)

    def test_empty_placements_zero(self):
        asm = _assembly(placements=[], score=0.0)
        sc  = score_edge_compat(asm, entries=[_entry(0, 1, 0.8)])
        assert sc.value == pytest.approx(0.0)

    def test_value_in_range(self):
        frags, asm = _perfect_assembly(3)
        entries    = [_entry(0, 1, 0.8), _entry(1, 2, 0.7)]
        sc         = score_edge_compat(asm, entries)
        assert 0.0 <= sc.value <= 1.0

    def test_weight_stored(self):
        frags, asm = _perfect_assembly(2)
        sc = score_edge_compat(asm, [_entry(0, 1, 0.6)], weight=1.5)
        assert sc.weight == pytest.approx(1.5)

    def test_name(self):
        frags, asm = _perfect_assembly(2)
        sc = score_edge_compat(asm, [])
        assert sc.name == "edge_compat"

    def test_high_entries_high_value(self):
        frags, asm = _perfect_assembly(3)
        entries    = [_entry(0, 1, 0.99), _entry(1, 2, 0.99)]
        sc         = score_edge_compat(asm, entries)
        assert sc.value > 0.6

    def test_irrelevant_entries_ignored(self):
        """Entries с чужими fid не влияют на оценку."""
        frags, asm = _perfect_assembly(2)
        entries    = [_entry(10, 11, 0.5)]  # fid 10/11 не в сборке
        sc         = score_edge_compat(asm, entries)
        assert sc.value == pytest.approx(float(np.clip(asm.total_score, 0, 1)))


# ─── score_layout ─────────────────────────────────────────────────────────────

class TestScoreLayout:
    def test_empty_fragments_zero(self):
        asm = _assembly([_placement(0)])
        sc  = score_layout(asm, fragments=[])
        assert sc.value == pytest.approx(0.0)

    def test_empty_placements_zero(self):
        frags = [_fragment(0)]
        asm   = _assembly(placements=[])
        sc    = score_layout(asm, frags)
        assert sc.value == pytest.approx(0.0)

    def test_perfect_layout_high_value(self):
        frags, asm = _perfect_assembly(3)
        sc = score_layout(asm, frags)
        assert sc.value >= 0.5

    def test_overlap_lowers_value(self):
        """Перекрывающиеся фрагменты → score ниже, чем идеальные."""
        frags = [_fragment(0), _fragment(1)]
        asm_ok  = _assembly([_placement(0, 0, 0), _placement(1, 60, 0)])
        asm_bad = _assembly([_placement(0, 0, 0), _placement(1, 5, 0)])
        sc_ok   = score_layout(asm_ok,  frags)
        sc_bad  = score_layout(asm_bad, frags)
        assert sc_ok.value >= sc_bad.value

    def test_value_in_range(self):
        frags, asm = _perfect_assembly(4)
        sc = score_layout(asm, frags)
        assert 0.0 <= sc.value <= 1.0

    def test_name(self):
        frags, asm = _perfect_assembly(2)
        sc = score_layout(asm, frags)
        assert sc.name == "layout"

    def test_weight_stored(self):
        frags, asm = _perfect_assembly(2)
        sc = score_layout(asm, frags, weight=2.0)
        assert sc.weight == pytest.approx(2.0)


# ─── score_coverage ───────────────────────────────────────────────────────────

class TestScoreCoverage:
    def test_empty_fids_zero(self):
        asm = _assembly([_placement(0)])
        sc  = score_coverage(asm, all_fragment_ids=[])
        assert sc.value == pytest.approx(0.0)

    def test_full_coverage_one(self):
        asm = _assembly([_placement(i) for i in range(4)])
        sc  = score_coverage(asm, all_fragment_ids=list(range(4)))
        assert sc.value == pytest.approx(1.0)

    def test_partial_coverage(self):
        asm = _assembly([_placement(0), _placement(1)])
        sc  = score_coverage(asm, all_fragment_ids=[0, 1, 2, 3])
        assert sc.value == pytest.approx(0.5)

    def test_value_in_range(self):
        asm = _assembly([_placement(0)])
        sc  = score_coverage(asm, all_fragment_ids=[0, 1, 2])
        assert 0.0 <= sc.value <= 1.0

    def test_name(self):
        asm = _assembly([_placement(0)])
        sc  = score_coverage(asm, [0])
        assert sc.name == "coverage"

    def test_weight_stored(self):
        asm = _assembly([_placement(0)])
        sc  = score_coverage(asm, [0], weight=0.5)
        assert sc.weight == pytest.approx(0.5)

    def test_duplicate_placements_capped(self):
        """Дубликат fid не должен давать > 1.0."""
        asm = _assembly([_placement(0), _placement(0)])
        sc  = score_coverage(asm, all_fragment_ids=[0])
        assert sc.value <= 1.0


# ─── score_uniqueness ─────────────────────────────────────────────────────────

class TestScoreUniqueness:
    def test_no_duplicates_one(self):
        asm = _assembly([_placement(i) for i in range(4)])
        sc  = score_uniqueness(asm)
        assert sc.value == pytest.approx(1.0)

    def test_one_duplicate(self):
        asm = _assembly([_placement(0), _placement(0), _placement(1)])
        sc  = score_uniqueness(asm)
        assert sc.value == pytest.approx(0.8)

    def test_two_duplicates(self):
        asm = _assembly([_placement(0), _placement(0),
                          _placement(1), _placement(1)])
        sc  = score_uniqueness(asm)
        assert sc.value == pytest.approx(0.6)

    def test_many_duplicates_floor_zero(self):
        """5+ дубликатов → 0.0 (не отрицательное)."""
        pls = [_placement(0)] * 8
        sc  = score_uniqueness(_assembly(pls))
        assert sc.value >= 0.0

    def test_empty_placements_zero(self):
        asm = _assembly(placements=[])
        sc  = score_uniqueness(asm)
        assert sc.value == pytest.approx(0.0)

    def test_name(self):
        asm = _assembly([_placement(0)])
        sc  = score_uniqueness(asm)
        assert sc.name == "uniqueness"

    def test_value_in_range(self):
        asm = _assembly([_placement(0), _placement(0), _placement(1)])
        sc  = score_uniqueness(asm)
        assert 0.0 <= sc.value <= 1.0


# ─── score_assembly_score ─────────────────────────────────────────────────────

class TestScoreAssemblyScore:
    def test_normal_score(self):
        asm = _assembly([], score=0.75)
        sc  = score_assembly_score(asm)
        assert sc.value == pytest.approx(0.75)

    def test_clip_above_one(self):
        asm = _assembly([], score=1.5)
        sc  = score_assembly_score(asm)
        assert sc.value == pytest.approx(1.0)

    def test_clip_below_zero(self):
        asm = _assembly([], score=-0.3)
        sc  = score_assembly_score(asm)
        assert sc.value == pytest.approx(0.0)

    def test_name(self):
        sc = score_assembly_score(_assembly([]))
        assert sc.name == "assembly_score"

    def test_weight_stored(self):
        sc = score_assembly_score(_assembly([]), weight=2.0)
        assert sc.weight == pytest.approx(2.0)


# ─── compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_returns_assembly_confidence(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        assert isinstance(r, AssemblyConfidence)

    def test_total_in_range(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [_entry(0, 1, 0.8)])
        assert 0.0 <= r.total <= 1.0

    def test_grade_is_string(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        assert isinstance(r.grade, str)
        assert r.grade in ("A", "B", "C", "D", "F")

    def test_five_components(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        assert len(r.components) == 5

    def test_n_fragments_set(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        assert r.n_fragments == 3

    def test_assembly_method_set(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        assert r.assembly_method == asm.method

    def test_custom_weights(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [],
                                weights={"edge_compat": 0.0, "layout": 10.0})
        # layout имеет очень большой вес → итог близок к layout value
        layout_sc = r.get("layout")
        assert layout_sc is not None

    def test_all_fragment_ids_param(self):
        frags, asm = _perfect_assembly(2)
        r = compute_confidence(asm, frags, [],
                                all_fragment_ids=[0, 1, 2, 3])
        cov = r.get("coverage")
        assert cov is not None
        assert cov.value == pytest.approx(0.5)

    def test_perfect_assembly_high_grade(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [_entry(0, 1, 0.95),
                                             _entry(1, 2, 0.95)])
        # Идеальная сборка → оценка ≥ C
        assert r.grade in ("A", "B", "C")

    def test_all_components_have_weights(self):
        frags, asm = _perfect_assembly(3)
        r = compute_confidence(asm, frags, [])
        for c in r.components:
            assert c.weight > 0.0
