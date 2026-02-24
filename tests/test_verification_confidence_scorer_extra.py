"""Extra tests for puzzle_reconstruction/verification/confidence_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.confidence_scorer import (
    ScoreComponent,
    AssemblyConfidence,
    grade_from_score,
    score_edge_compat,
    score_coverage,
    score_uniqueness,
    score_assembly_score,
)
from puzzle_reconstruction.models import Assembly


# ─── helpers ──────────────────────────────────────────────────────────────────

def _asm(score=0.5):
    return Assembly(
        fragments=[], placements={},
        compat_matrix=np.array([]), total_score=score,
    )


# ─── ScoreComponent ─────────────────────────────────────────────────────────

class TestScoreComponentExtra:
    def test_valid(self):
        sc = ScoreComponent(name="test", value=0.8, weight=1.5)
        assert sc.weighted == pytest.approx(1.2)

    def test_repr(self):
        sc = ScoreComponent(name="edge", value=0.5, weight=1.0)
        s = repr(sc)
        assert "edge" in s
        assert "0.500" in s


# ─── AssemblyConfidence ─────────────────────────────────────────────────────

class TestAssemblyConfidenceExtra:
    def test_valid(self):
        sc = ScoreComponent(name="x", value=0.8, weight=1.0)
        ac = AssemblyConfidence(total=0.8, components=[sc], grade="B")
        assert ac.total == pytest.approx(0.8)
        assert ac.grade == "B"

    def test_get(self):
        sc = ScoreComponent(name="edge", value=0.9, weight=1.0)
        ac = AssemblyConfidence(total=0.9, components=[sc], grade="A")
        assert ac.get("edge") is sc
        assert ac.get("unknown") is None

    def test_as_dict(self):
        sc = ScoreComponent(name="edge", value=0.9, weight=1.0)
        ac = AssemblyConfidence(total=0.9, components=[sc], grade="A")
        d = ac.as_dict()
        assert d["edge"] == pytest.approx(0.9)

    def test_summary(self):
        sc = ScoreComponent(name="edge", value=0.9, weight=1.0)
        ac = AssemblyConfidence(total=0.9, components=[sc], grade="A")
        s = ac.summary()
        assert "edge" in s
        assert "A" in s

    def test_repr(self):
        ac = AssemblyConfidence(total=0.5, components=[], grade="C",
                                n_fragments=3)
        s = repr(ac)
        assert "C" in s
        assert "3" in s


# ─── grade_from_score ───────────────────────────────────────────────────────

class TestGradeFromScoreExtra:
    def test_A(self):
        assert grade_from_score(0.85) == "A"
        assert grade_from_score(1.0) == "A"

    def test_B(self):
        assert grade_from_score(0.70) == "B"
        assert grade_from_score(0.84) == "B"

    def test_C(self):
        assert grade_from_score(0.55) == "C"

    def test_D(self):
        assert grade_from_score(0.40) == "D"

    def test_F(self):
        assert grade_from_score(0.0) == "F"
        assert grade_from_score(0.39) == "F"


# ─── score_edge_compat ──────────────────────────────────────────────────────

class TestScoreEdgeCompatExtra:
    def test_no_entries(self):
        sc = score_edge_compat(_asm(), [])
        assert sc.name == "edge_compat"
        assert sc.value == pytest.approx(0.0)

    def test_no_placements(self):
        sc = score_edge_compat(_asm(), [])
        assert sc.value == pytest.approx(0.0)


# ─── score_coverage ─────────────────────────────────────────────────────────

class TestScoreCoverageExtra:
    def test_no_fragments(self):
        sc = score_coverage(_asm(), [])
        assert sc.value == pytest.approx(0.0)

    def test_all_placed(self):
        sc = score_coverage(_asm(), [0, 1])
        # Assembly has no placements so coverage = 0
        assert sc.value == pytest.approx(0.0)


# ─── score_uniqueness ───────────────────────────────────────────────────────

class TestScoreUniquenessExtra:
    def test_no_placements(self):
        sc = score_uniqueness(_asm())
        assert sc.value == pytest.approx(0.0)


# ─── score_assembly_score ───────────────────────────────────────────────────

class TestScoreAssemblyScoreExtra:
    def test_basic(self):
        sc = score_assembly_score(_asm(0.7))
        assert sc.name == "assembly_score"
        assert sc.value == pytest.approx(0.7)

    def test_clipped(self):
        sc = score_assembly_score(_asm(1.5))
        assert sc.value <= 1.0

    def test_zero(self):
        sc = score_assembly_score(_asm(0.0))
        assert sc.value == pytest.approx(0.0)
