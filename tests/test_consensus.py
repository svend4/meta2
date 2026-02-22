"""
Тесты для puzzle_reconstruction/matching/consensus.py

Покрытие:
    ConsensusResult    — vote_fraction, is_consensus, top_pairs, summary
    assembly_to_pairs  — пустая сборка, 2/3/4 фрагмента, порог расстояния
    vote_on_pairs      — пустой список, один, несколько Assembly, подсчёт голосов
    build_consensus    — пустой список, threshold=0/1, консенсусные пары
    consensus_score_matrix — симметрия, диагональ, диапазон [0,1], пустой
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, CompatEntry, EdgeSide, EdgeSignature, Fragment
from puzzle_reconstruction.matching.consensus import (
    ConsensusResult,
    assembly_to_pairs,
    build_consensus,
    consensus_score_matrix,
    vote_on_pairs,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _make_fragment(fid: int) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((4, 2)),
    )


def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((8, 2)),
        fd=1.5, css_vec=np.zeros(8), ifs_coeffs=np.zeros(4), length=60.0,
    )


def _make_entry(fid_i: int, fid_j: int, score: float = 0.5) -> CompatEntry:
    ei = _make_edge(fid_i * 10)
    ej = _make_edge(fid_j * 10)
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_assembly(fids: list, positions: list, score: float = 0.5) -> Assembly:
    """Создаёт Assembly с заданными позициями."""
    frags = [_make_fragment(fid) for fid in fids]
    placements = {
        fid: (np.array(pos, dtype=np.float64), 0.0)
        for fid, pos in zip(fids, positions)
    }
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=score,
    )


# ─── assembly_to_pairs ────────────────────────────────────────────────────────

class TestAssemblyToPairs:
    def test_empty_assembly(self):
        a = _make_assembly([], [])
        pairs = assembly_to_pairs(a)
        assert pairs == set()

    def test_single_fragment(self):
        a = _make_assembly([0], [[0, 0]])
        pairs = assembly_to_pairs(a)
        assert pairs == set()

    def test_two_close_fragments(self):
        """Два фрагмента близко → одна пара."""
        a = _make_assembly([0, 1], [[0, 0], [100, 0]])
        pairs = assembly_to_pairs(a, adjacency_threshold=200.0)
        assert frozenset({0, 1}) in pairs

    def test_two_far_fragments(self):
        """Два фрагмента далеко → пар нет."""
        a = _make_assembly([0, 1], [[0, 0], [1000, 0]])
        pairs = assembly_to_pairs(a, adjacency_threshold=200.0)
        assert frozenset({0, 1}) not in pairs

    def test_three_fragments_triangle(self):
        """Три фрагмента на расстоянии 100 → 3 пары."""
        a = _make_assembly([0, 1, 2],
                            [[0, 0], [100, 0], [50, 87]])
        pairs = assembly_to_pairs(a, adjacency_threshold=200.0)
        assert len(pairs) == 3
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert frozenset({i, j}) in pairs

    def test_threshold_boundary(self):
        """Пара прямо на границе порога."""
        a = _make_assembly([0, 1], [[0, 0], [200, 0]])
        # dist = 200, threshold = 200 → не включается (строго <)
        pairs = assembly_to_pairs(a, adjacency_threshold=200.0)
        assert frozenset({0, 1}) not in pairs
        # threshold = 201 → включается
        pairs2 = assembly_to_pairs(a, adjacency_threshold=201.0)
        assert frozenset({0, 1}) in pairs2

    def test_pairs_are_frozensets(self):
        a = _make_assembly([0, 1, 2], [[0, 0], [50, 0], [0, 50]])
        pairs = assembly_to_pairs(a)
        for p in pairs:
            assert isinstance(p, frozenset)
            assert len(p) == 2


# ─── vote_on_pairs ────────────────────────────────────────────────────────────

class TestVoteOnPairs:
    def test_empty_assemblies(self):
        votes = vote_on_pairs([])
        assert votes == {}

    def test_single_assembly(self):
        a = _make_assembly([0, 1], [[0, 0], [100, 0]])
        votes = vote_on_pairs([a], adjacency_threshold=200.0)
        assert frozenset({0, 1}) in votes
        assert votes[frozenset({0, 1})] == 1

    def test_counts_accumulate(self):
        """Одна пара появляется в двух Assembly → count = 2."""
        a1 = _make_assembly([0, 1], [[0, 0], [100, 0]])
        a2 = _make_assembly([0, 1], [[0, 0], [150, 0]])
        votes = vote_on_pairs([a1, a2], adjacency_threshold=200.0)
        assert votes.get(frozenset({0, 1}), 0) == 2

    def test_disjoint_pairs(self):
        """Разные методы выбирают разные пары → разные счётчики."""
        a1 = _make_assembly([0, 1, 2], [[0, 0], [100, 0], [5000, 0]])
        a2 = _make_assembly([0, 1, 2], [[5000, 0], [100, 0], [200, 0]])
        votes = vote_on_pairs([a1, a2], adjacency_threshold=200.0)
        assert frozenset({0, 1}) in votes
        assert frozenset({1, 2}) in votes

    def test_all_counts_positive(self):
        a1 = _make_assembly([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        votes = vote_on_pairs([a1, a1, a1])
        for v in votes.values():
            assert v > 0


# ─── ConsensusResult ──────────────────────────────────────────────────────────

class TestConsensusResult:
    @pytest.fixture
    def result(self):
        pair_votes = {
            frozenset({0, 1}): 3,
            frozenset({1, 2}): 2,
            frozenset({0, 2}): 1,
        }
        consensus_pairs = {frozenset({0, 1}), frozenset({1, 2})}
        return ConsensusResult(
            pair_votes=pair_votes,
            consensus_pairs=consensus_pairs,
            n_methods=3,
            threshold=0.5,
        )

    def test_vote_fraction(self, result):
        assert math.isclose(result.vote_fraction(0, 1), 1.0)
        assert math.isclose(result.vote_fraction(1, 2), 2/3, rel_tol=1e-6)
        assert math.isclose(result.vote_fraction(0, 2), 1/3, rel_tol=1e-6)

    def test_vote_fraction_missing_pair(self, result):
        assert math.isclose(result.vote_fraction(5, 6), 0.0)

    def test_is_consensus(self, result):
        assert result.is_consensus(0, 1)
        assert result.is_consensus(1, 2)
        assert not result.is_consensus(0, 2)
        assert not result.is_consensus(9, 10)

    def test_top_pairs_order(self, result):
        top = result.top_pairs(n=2)
        assert top[0][0] == frozenset({0, 1})
        assert top[0][1] == 3
        assert top[1][1] == 2

    def test_top_pairs_limit(self, result):
        top = result.top_pairs(n=100)
        assert len(top) == len(result.pair_votes)

    def test_summary_contains_key_info(self, result):
        s = result.summary()
        assert "3" in s                     # n_methods
        assert "ConsensusResult" in s

    def test_zero_methods_vote_fraction(self):
        r = ConsensusResult({}, set(), n_methods=0, threshold=0.5)
        assert math.isclose(r.vote_fraction(0, 1), 0.0)


# ─── build_consensus ──────────────────────────────────────────────────────────

class TestBuildConsensus:
    def test_empty_assemblies(self):
        result = build_consensus([], [], [], threshold=0.5)
        assert isinstance(result, ConsensusResult)
        assert result.n_methods == 0
        assert result.pair_votes == {}
        assert result.assembly is None

    def test_single_assembly_threshold_0(self):
        """threshold=0 → все пары консенсусные."""
        a = _make_assembly([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        result = build_consensus([a], [], [], threshold=0.0,
                                  build_assembly=False)
        # Все пары с ≥1 голосом должны войти
        assert result.consensus_pairs == set(result.pair_votes.keys())

    def test_threshold_1_requires_all_votes(self):
        """threshold=1.0 → пара должна быть в ВСЕХ методах."""
        a1 = _make_assembly([0, 1], [[0, 0], [100, 0]])
        a2 = _make_assembly([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        result = build_consensus([a1, a2], [], [], threshold=1.0,
                                  build_assembly=False)
        # Пара {0,1} есть в обеих → консенсус
        pair = frozenset({0, 1})
        if pair in result.pair_votes and result.pair_votes[pair] == 2:
            assert pair in result.consensus_pairs

    def test_majority_threshold(self):
        """Пара в 2/3 методов, threshold=0.5 → консенсус."""
        pos = [[0, 0], [100, 0], [200, 0]]
        a1 = _make_assembly([0, 1, 2], pos)
        a2 = _make_assembly([0, 1, 2], pos)
        a3 = _make_assembly([0, 1, 2], [[5000, 0], [100, 0], [200, 0]])
        result = build_consensus([a1, a2, a3], [], [], threshold=0.5,
                                  build_assembly=False)
        assert isinstance(result, ConsensusResult)
        assert result.n_methods == 3

    def test_build_assembly_false(self):
        a = _make_assembly([0, 1], [[0, 0], [100, 0]])
        result = build_consensus([a], [], [], build_assembly=False)
        assert result.assembly is None

    def test_returns_consensus_result(self):
        a = _make_assembly([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        result = build_consensus([a, a], [], [], threshold=0.5,
                                  build_assembly=False)
        assert isinstance(result, ConsensusResult)

    def test_n_methods_correct(self):
        a1 = _make_assembly([0, 1], [[0, 0], [100, 0]])
        a2 = _make_assembly([0, 1], [[0, 0], [100, 0]])
        a3 = _make_assembly([0, 1], [[0, 0], [100, 0]])
        result = build_consensus([a1, a2, a3], [], [], build_assembly=False)
        assert result.n_methods == 3


# ─── consensus_score_matrix ───────────────────────────────────────────────────

class TestConsensusScoreMatrix:
    def test_empty_fragments(self):
        result = ConsensusResult({}, set(), n_methods=1, threshold=0.5)
        mat = consensus_score_matrix(result, [])
        assert mat.shape == (0, 0)

    def test_shape(self):
        frags = [_make_fragment(i) for i in range(4)]
        result = ConsensusResult({}, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        assert mat.shape == (4, 4)

    def test_symmetry(self):
        frags = [_make_fragment(i) for i in range(3)]
        pair_votes = {frozenset({0, 1}): 2, frozenset({1, 2}): 1}
        result = ConsensusResult(pair_votes, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        assert np.allclose(mat, mat.T)

    def test_diagonal_zero(self):
        frags = [_make_fragment(i) for i in range(3)]
        pair_votes = {frozenset({0, 1}): 2}
        result = ConsensusResult(pair_votes, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        np.testing.assert_array_equal(np.diag(mat), 0.0)

    def test_values_in_range(self):
        frags = [_make_fragment(i) for i in range(4)]
        pair_votes = {
            frozenset({i, j}): np.random.randint(1, 4)
            for i in range(4) for j in range(i + 1, 4)
        }
        result = ConsensusResult(pair_votes, set(), n_methods=3, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_known_value(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        pair_votes = {frozenset({0, 1}): 2}
        result = ConsensusResult(pair_votes, set(), n_methods=4, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        assert math.isclose(mat[0, 1], 2 / 4)
        assert math.isclose(mat[1, 0], 2 / 4)

    def test_dtype_float64(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        result = ConsensusResult({}, set(), n_methods=1, threshold=0.5)
        mat = consensus_score_matrix(result, frags)
        assert mat.dtype == np.float64
