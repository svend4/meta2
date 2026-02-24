"""Extra tests for puzzle_reconstruction/assembly/fragment_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
    add_adjacency,
)
from puzzle_reconstruction.assembly.cost_matrix import CostMatrix
from puzzle_reconstruction.assembly.fragment_scorer import (
    ScoreConfig,
    FragmentScore,
    AssemblyScore,
    score_fragment,
    score_assembly,
    top_k_placed,
    bottom_k_placed,
    batch_score,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cm(n, fill=0.5):
    m = np.full((n, n), fill, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return CostMatrix(matrix=m, n_fragments=n, method="test")


def _state(*ids, n=4):
    s = create_state(n)
    for i in ids:
        s = place_fragment(s, i, position=(float(i), 0.0))
    return s


def _adj_state(n=4, edges=()):
    s = _state(*range(n), n=n)
    for i, j in edges:
        s = add_adjacency(s, i, j)
    return s


# ─── ScoreConfig (extra) ─────────────────────────────────────────────────────

class TestScoreConfigExtra:
    def test_large_weights_ok(self):
        cfg = ScoreConfig(neighbor_weight=10.0, coverage_weight=5.0)
        assert cfg.total_weight == pytest.approx(15.0)

    def test_one_weight_zero(self):
        cfg = ScoreConfig(neighbor_weight=1.0, coverage_weight=0.0)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_min_neighbors_large(self):
        cfg = ScoreConfig(min_neighbors=10)
        assert cfg.min_neighbors == 10

    def test_independent_instances(self):
        c1 = ScoreConfig(neighbor_weight=0.3)
        c2 = ScoreConfig(neighbor_weight=0.9)
        assert c1.neighbor_weight != c2.neighbor_weight

    def test_default_min_neighbors(self):
        cfg = ScoreConfig()
        assert cfg.min_neighbors == 1


# ─── FragmentScore (extra) ───────────────────────────────────────────────────

class TestFragmentScoreExtra:
    def test_boundary_score_zero(self):
        fs = FragmentScore(fragment_idx=0, local_score=0.0, n_neighbors=0)
        assert fs.local_score == pytest.approx(0.0)

    def test_boundary_score_one(self):
        fs = FragmentScore(fragment_idx=0, local_score=1.0, n_neighbors=0)
        assert fs.local_score == pytest.approx(1.0)

    def test_large_idx_ok(self):
        fs = FragmentScore(fragment_idx=9999, local_score=0.5, n_neighbors=0)
        assert fs.fragment_idx == 9999

    def test_many_neighbors(self):
        fs = FragmentScore(fragment_idx=0, local_score=0.5, n_neighbors=100)
        assert fs.n_neighbors == 100

    def test_is_reliable_default_false(self):
        fs = FragmentScore(fragment_idx=0, local_score=0.5, n_neighbors=0)
        assert fs.is_reliable is False


# ─── AssemblyScore (extra) ───────────────────────────────────────────────────

class TestAssemblyScoreExtra:
    def test_score_zero(self):
        asm = AssemblyScore(global_score=0.0, coverage=0.0, mean_local=0.0)
        assert asm.global_score == pytest.approx(0.0)

    def test_score_one(self):
        asm = AssemblyScore(global_score=1.0, coverage=1.0, mean_local=1.0)
        assert asm.global_score == pytest.approx(1.0)

    def test_n_reliable_default_zero(self):
        asm = AssemblyScore(global_score=0.5, coverage=0.5, mean_local=0.5)
        assert asm.n_reliable == 0

    def test_fragment_scores_stored(self):
        fscores = {0: FragmentScore(0, 0.5, 1)}
        asm = AssemblyScore(global_score=0.5, coverage=0.5, mean_local=0.5,
                            fragment_scores=fscores)
        assert 0 in asm.fragment_scores

    def test_summary_string(self):
        asm = AssemblyScore(global_score=0.7, coverage=0.8, mean_local=0.6)
        s = asm.summary()
        assert isinstance(s, str)
        assert len(s) > 0


# ─── score_fragment (extra) ──────────────────────────────────────────────────

class TestScoreFragmentExtra:
    def test_multiple_neighbors_average(self):
        s = _adj_state(n=4, edges=[(0, 1), (0, 2)])
        m = np.full((4, 4), 0.0, dtype=np.float32)
        m[0, 1] = 0.6
        m[1, 0] = 0.6
        m[0, 2] = 0.4
        m[2, 0] = 0.4
        cm = CostMatrix(matrix=m, n_fragments=4, method="test")
        result = score_fragment(s, 0, cm)
        assert result.n_neighbors == 2
        assert result.local_score == pytest.approx(0.5, abs=0.01)

    def test_fragment_idx_correct(self):
        s = _state(3, n=5)
        cm = _cm(5)
        result = score_fragment(s, 3, cm)
        assert result.fragment_idx == 3

    def test_result_type(self):
        s = _state(0, n=3)
        cm = _cm(3)
        result = score_fragment(s, 0, cm)
        assert isinstance(result, FragmentScore)

    def test_score_in_range(self):
        s = _adj_state(n=3, edges=[(0, 1)])
        cm = _cm(3, fill=0.3)
        result = score_fragment(s, 0, cm)
        assert 0.0 <= result.local_score <= 1.0


# ─── score_assembly (extra) ──────────────────────────────────────────────────

class TestScoreAssemblyExtra:
    def test_empty_state(self):
        s = create_state(3)
        cm = _cm(3)
        result = score_assembly(s, cm)
        assert result.coverage == pytest.approx(0.0)

    def test_full_state_coverage_one(self):
        s = _state(*range(5), n=5)
        cm = _cm(5)
        result = score_assembly(s, cm)
        assert result.coverage == pytest.approx(1.0)

    def test_fragment_scores_keys_match_placed(self):
        s = _state(0, 2, n=4)
        cm = _cm(4)
        result = score_assembly(s, cm)
        assert set(result.fragment_scores.keys()) == {0, 2}

    def test_global_score_in_range(self):
        s = _adj_state(n=3, edges=[(0, 1)])
        cm = _cm(3)
        result = score_assembly(s, cm)
        assert 0.0 <= result.global_score <= 1.0

    def test_result_type(self):
        s = _state(0, n=3)
        cm = _cm(3)
        result = score_assembly(s, cm)
        assert isinstance(result, AssemblyScore)


# ─── top_k_placed / bottom_k_placed (extra) ──────────────────────────────────

class TestTopKBottomKExtra:
    def _asm(self):
        fscores = {
            0: FragmentScore(0, 0.1, 1),
            1: FragmentScore(1, 0.5, 1),
            2: FragmentScore(2, 0.9, 1),
        }
        return AssemblyScore(global_score=0.5, coverage=1.0, mean_local=0.5,
                             fragment_scores=fscores)

    def test_top_2_ascending(self):
        result = top_k_placed(self._asm(), k=2)
        assert len(result) == 2
        scores = [r[1] for r in result]
        assert scores == sorted(scores)

    def test_bottom_2_descending(self):
        result = bottom_k_placed(self._asm(), k=2)
        assert len(result) == 2
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_all(self):
        result = top_k_placed(self._asm(), k=3)
        assert len(result) == 3

    def test_bottom_k_all(self):
        result = bottom_k_placed(self._asm(), k=3)
        assert len(result) == 3

    def test_top_1_lowest(self):
        result = top_k_placed(self._asm(), k=1)
        assert result[0][0] == 0  # lowest local_score

    def test_bottom_1_highest(self):
        result = bottom_k_placed(self._asm(), k=1)
        assert result[0][0] == 2  # highest local_score


# ─── batch_score (extra) ─────────────────────────────────────────────────────

class TestBatchScoreExtra:
    def test_single_state(self):
        cm = _cm(3)
        result = batch_score([_state(0, n=3)], cm)
        assert len(result) == 1

    def test_multiple_states(self):
        cm = _cm(3)
        states = [_state(0, n=3), _state(1, n=3), _state(0, 1, n=3)]
        result = batch_score(states, cm)
        assert len(result) == 3

    def test_all_assembly_scores(self):
        cm = _cm(3)
        result = batch_score([_state(0, n=3)], cm)
        assert isinstance(result[0], AssemblyScore)

    def test_empty_returns_empty(self):
        cm = _cm(3)
        assert batch_score([], cm) == []

    def test_scores_in_range(self):
        cm = _cm(4)
        states = [_state(0, 1, n=4)]
        for asm in batch_score(states, cm):
            assert 0.0 <= asm.global_score <= 1.0
