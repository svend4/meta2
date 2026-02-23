"""Extra tests for puzzle_reconstruction.matching.consensus."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frag(fid):
    return Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((4, 2)),
    )


def _asm(fids, positions, score=0.5):
    frags = [_frag(fid) for fid in fids]
    placements = {
        fid: (np.array(pos, dtype=np.float64), 0.0)
        for fid, pos in zip(fids, positions)
    }
    return Assembly(
        fragments=frags, placements=placements,
        compat_matrix=np.array([]), total_score=score,
    )


# ─── TestAssemblyToPairsExtra ─────────────────────────────────────────────────

class TestAssemblyToPairsExtra:
    def test_empty_assembly(self):
        assert assembly_to_pairs(_asm([], [])) == set()

    def test_single_fragment(self):
        assert assembly_to_pairs(_asm([0], [[0, 0]])) == set()

    def test_two_close(self):
        pairs = assembly_to_pairs(_asm([0, 1], [[0, 0], [50, 0]]),
                                   adjacency_threshold=200.0)
        assert frozenset({0, 1}) in pairs

    def test_two_far(self):
        pairs = assembly_to_pairs(_asm([0, 1], [[0, 0], [1000, 0]]),
                                   adjacency_threshold=200.0)
        assert frozenset({0, 1}) not in pairs

    def test_three_close_three_pairs(self):
        pairs = assembly_to_pairs(
            _asm([0, 1, 2], [[0, 0], [100, 0], [50, 87]]),
            adjacency_threshold=200.0,
        )
        assert len(pairs) == 3

    def test_threshold_strict_less(self):
        # dist = 200, threshold = 200 → NOT included (strict <)
        pairs = assembly_to_pairs(
            _asm([0, 1], [[0, 0], [200, 0]]),
            adjacency_threshold=200.0,
        )
        assert frozenset({0, 1}) not in pairs

    def test_threshold_just_above(self):
        pairs = assembly_to_pairs(
            _asm([0, 1], [[0, 0], [200, 0]]),
            adjacency_threshold=201.0,
        )
        assert frozenset({0, 1}) in pairs

    def test_pairs_are_frozensets(self):
        pairs = assembly_to_pairs(
            _asm([0, 1, 2], [[0, 0], [50, 0], [0, 50]]))
        for p in pairs:
            assert isinstance(p, frozenset)
            assert len(p) == 2

    def test_four_fragments_line(self):
        # 4 fragments on a line, adjacent pairs only
        a = _asm([0, 1, 2, 3], [[0, 0], [100, 0], [200, 0], [300, 0]])
        pairs = assembly_to_pairs(a, adjacency_threshold=150.0)
        assert frozenset({0, 1}) in pairs
        assert frozenset({1, 2}) in pairs
        assert frozenset({2, 3}) in pairs
        assert frozenset({0, 3}) not in pairs


# ─── TestVoteOnPairsExtra ─────────────────────────────────────────────────────

class TestVoteOnPairsExtra:
    def test_empty(self):
        assert vote_on_pairs([]) == {}

    def test_single_assembly_one_pair(self):
        a = _asm([0, 1], [[0, 0], [100, 0]])
        votes = vote_on_pairs([a], adjacency_threshold=200.0)
        assert frozenset({0, 1}) in votes
        assert votes[frozenset({0, 1})] == 1

    def test_accumulate(self):
        a1 = _asm([0, 1], [[0, 0], [100, 0]])
        a2 = _asm([0, 1], [[0, 0], [150, 0]])
        votes = vote_on_pairs([a1, a2], adjacency_threshold=200.0)
        assert votes.get(frozenset({0, 1}), 0) == 2

    def test_three_assemblies(self):
        a = _asm([0, 1], [[0, 0], [100, 0]])
        votes = vote_on_pairs([a, a, a], adjacency_threshold=200.0)
        assert votes[frozenset({0, 1})] == 3

    def test_all_counts_positive(self):
        a = _asm([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        votes = vote_on_pairs([a, a])
        for v in votes.values():
            assert v > 0


# ─── TestConsensusResultExtra ─────────────────────────────────────────────────

class TestConsensusResultExtra:
    def _make(self):
        pair_votes = {
            frozenset({0, 1}): 3,
            frozenset({1, 2}): 2,
            frozenset({0, 2}): 1,
        }
        consensus_pairs = {frozenset({0, 1}), frozenset({1, 2})}
        return ConsensusResult(
            pair_votes=pair_votes, consensus_pairs=consensus_pairs,
            n_methods=3, threshold=0.5,
        )

    def test_vote_fraction_full(self):
        r = self._make()
        assert math.isclose(r.vote_fraction(0, 1), 1.0)

    def test_vote_fraction_partial(self):
        r = self._make()
        assert math.isclose(r.vote_fraction(1, 2), 2 / 3, rel_tol=1e-6)

    def test_vote_fraction_low(self):
        r = self._make()
        assert math.isclose(r.vote_fraction(0, 2), 1 / 3, rel_tol=1e-6)

    def test_vote_fraction_missing(self):
        r = self._make()
        assert math.isclose(r.vote_fraction(5, 6), 0.0)

    def test_is_consensus_true(self):
        r = self._make()
        assert r.is_consensus(0, 1)

    def test_is_consensus_false(self):
        r = self._make()
        assert not r.is_consensus(0, 2)

    def test_top_pairs_sorted(self):
        r = self._make()
        top = r.top_pairs(n=3)
        votes = [v for _, v in top]
        assert votes == sorted(votes, reverse=True)

    def test_top_pairs_limit(self):
        r = self._make()
        top = r.top_pairs(n=1)
        assert len(top) == 1

    def test_summary_string(self):
        s = self._make().summary()
        assert isinstance(s, str)
        assert "ConsensusResult" in s

    def test_zero_methods(self):
        r = ConsensusResult({}, set(), n_methods=0, threshold=0.5)
        assert math.isclose(r.vote_fraction(0, 1), 0.0)


# ─── TestBuildConsensusExtra ──────────────────────────────────────────────────

class TestBuildConsensusExtra:
    def test_empty_assemblies(self):
        r = build_consensus([], [], [], threshold=0.5)
        assert isinstance(r, ConsensusResult)
        assert r.n_methods == 0

    def test_single_assembly_threshold_0(self):
        a = _asm([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        r = build_consensus([a], [], [], threshold=0.0, build_assembly=False)
        assert r.consensus_pairs == set(r.pair_votes.keys())

    def test_n_methods_correct(self):
        a1 = _asm([0, 1], [[0, 0], [100, 0]])
        a2 = _asm([0, 1], [[0, 0], [100, 0]])
        r = build_consensus([a1, a2], [], [], build_assembly=False)
        assert r.n_methods == 2

    def test_build_assembly_false_none(self):
        a = _asm([0, 1], [[0, 0], [100, 0]])
        r = build_consensus([a], [], [], build_assembly=False)
        assert r.assembly is None

    def test_threshold_1_all_must_agree(self):
        a1 = _asm([0, 1], [[0, 0], [100, 0]])
        a2 = _asm([0, 1], [[0, 0], [100, 0]])
        r = build_consensus([a1, a2], [], [], threshold=1.0,
                             build_assembly=False)
        pair = frozenset({0, 1})
        if pair in r.pair_votes and r.pair_votes[pair] == 2:
            assert pair in r.consensus_pairs

    def test_returns_consensus_result_type(self):
        a = _asm([0, 1, 2], [[0, 0], [100, 0], [200, 0]])
        r = build_consensus([a], [], [], build_assembly=False)
        assert isinstance(r, ConsensusResult)


# ─── TestConsensusScoreMatrixExtra ────────────────────────────────────────────

class TestConsensusScoreMatrixExtra:
    def test_empty_shape(self):
        r = ConsensusResult({}, set(), n_methods=1, threshold=0.5)
        assert consensus_score_matrix(r, []).shape == (0, 0)

    def test_shape_4x4(self):
        frags = [_frag(i) for i in range(4)]
        r = ConsensusResult({}, set(), n_methods=2, threshold=0.5)
        assert consensus_score_matrix(r, frags).shape == (4, 4)

    def test_symmetry(self):
        frags = [_frag(i) for i in range(3)]
        pv = {frozenset({0, 1}): 2, frozenset({1, 2}): 1}
        r = ConsensusResult(pv, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(r, frags)
        assert np.allclose(mat, mat.T)

    def test_diagonal_zero(self):
        frags = [_frag(i) for i in range(3)]
        pv = {frozenset({0, 1}): 2}
        r = ConsensusResult(pv, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(r, frags)
        np.testing.assert_array_equal(np.diag(mat), 0.0)

    def test_values_in_range(self):
        frags = [_frag(i) for i in range(3)]
        pv = {frozenset({0, 1}): 1, frozenset({1, 2}): 2}
        r = ConsensusResult(pv, set(), n_methods=2, threshold=0.5)
        mat = consensus_score_matrix(r, frags)
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_known_value(self):
        frags = [_frag(0), _frag(1)]
        pv = {frozenset({0, 1}): 3}
        r = ConsensusResult(pv, set(), n_methods=4, threshold=0.5)
        mat = consensus_score_matrix(r, frags)
        assert math.isclose(mat[0, 1], 3 / 4)
        assert math.isclose(mat[1, 0], 3 / 4)

    def test_dtype(self):
        frags = [_frag(0), _frag(1)]
        r = ConsensusResult({}, set(), n_methods=1, threshold=0.5)
        mat = consensus_score_matrix(r, frags)
        assert mat.dtype == np.float64
