"""Tests for puzzle_reconstruction/matching/consensus.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.consensus import (
    ConsensusResult,
    build_consensus,
    assembly_to_pairs,
    vote_on_pairs,
    consensus_score_matrix,
)
from puzzle_reconstruction.models import (
    Assembly,
    Fragment,
    CompatEntry,
    EdgeSignature,
)
from puzzle_reconstruction.models import EdgeSide


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_fragment(fid: int, pos=(0.0, 0.0)):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    frag = Fragment(
        fragment_id=fid,
        image=img,
        mask=mask,
        contour=contour,
    )
    frag.placed = True
    frag.position = np.array(list(pos), dtype=np.float64)
    return frag


def make_assembly(placements: dict) -> Assembly:
    """placements: {frag_id: (pos_array, angle)}"""
    frags = [make_fragment(fid, pos) for fid, (pos, _) in placements.items()]
    return Assembly(
        fragments=frags,
        placements={fid: (np.array(pos, dtype=np.float64), angle)
                    for fid, (pos, angle) in placements.items()},
        compat_matrix=np.zeros((1, 1)),
    )


def make_edge_sig(edge_id: int, side=None):
    if side is None:
        side = EdgeSide.TOP
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((10, 2)),
        fd=1.0,
        css_vec=np.zeros(64),
        ifs_coeffs=np.zeros(8),
        length=100.0,
    )


def make_compat_entry(edge_id_i: int, edge_id_j: int, score: float = 0.8):
    return CompatEntry(
        edge_i=make_edge_sig(edge_id_i),
        edge_j=make_edge_sig(edge_id_j),
        score=score,
        dtw_dist=0.1,
        css_sim=0.9,
        fd_diff=0.05,
        text_score=0.7,
    )


# ─── assembly_to_pairs ───────────────────────────────────────────────────────

class TestAssemblyToPairs:
    def test_empty_assembly(self):
        asm = make_assembly({})
        pairs = assembly_to_pairs(asm)
        assert pairs == set()

    def test_single_fragment(self):
        asm = make_assembly({0: ([0.0, 0.0], 0.0)})
        pairs = assembly_to_pairs(asm)
        assert pairs == set()

    def test_two_close_fragments(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([100.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=200.0)
        assert frozenset({0, 1}) in pairs

    def test_two_far_fragments(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([1000.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=250.0)
        assert len(pairs) == 0

    def test_three_fragments_adjacency(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([100.0, 0.0], 0.0),
            2: ([200.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=150.0)
        assert frozenset({0, 1}) in pairs
        assert frozenset({1, 2}) in pairs
        # 0 and 2 are 200 apart, threshold 150 → not adjacent
        assert frozenset({0, 2}) not in pairs

    def test_pairs_are_frozensets(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([50.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm)
        for p in pairs:
            assert isinstance(p, frozenset)
            assert len(p) == 2


# ─── vote_on_pairs ───────────────────────────────────────────────────────────

class TestVoteOnPairs:
    def test_empty_assemblies(self):
        votes = vote_on_pairs([])
        assert votes == {}

    def test_single_assembly(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([50.0, 0.0], 0.0),
        })
        votes = vote_on_pairs([asm], adjacency_threshold=100.0)
        assert frozenset({0, 1}) in votes
        assert votes[frozenset({0, 1})] == 1

    def test_two_assemblies_same_pair(self):
        asm1 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm2 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        votes = vote_on_pairs([asm1, asm2], adjacency_threshold=100.0)
        assert votes[frozenset({0, 1})] == 2

    def test_two_assemblies_different_pairs(self):
        asm1 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm2 = make_assembly({2: ([0.0, 0.0], 0.0), 3: ([50.0, 0.0], 0.0)})
        votes = vote_on_pairs([asm1, asm2], adjacency_threshold=100.0)
        assert frozenset({0, 1}) in votes
        assert frozenset({2, 3}) in votes
        assert votes[frozenset({0, 1})] == 1
        assert votes[frozenset({2, 3})] == 1

    def test_votes_are_non_negative(self):
        asm = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([30.0, 0.0], 0.0)})
        votes = vote_on_pairs([asm])
        for v in votes.values():
            assert v >= 1


# ─── ConsensusResult ─────────────────────────────────────────────────────────

class TestConsensusResult:
    def _make_result(self):
        votes = {
            frozenset({0, 1}): 3,
            frozenset({1, 2}): 2,
            frozenset({0, 2}): 1,
        }
        consensus_pairs = {frozenset({0, 1}), frozenset({1, 2})}
        return ConsensusResult(
            pair_votes=votes,
            consensus_pairs=consensus_pairs,
            n_methods=3,
            threshold=0.5,
        )

    def test_vote_fraction(self):
        r = self._make_result()
        assert abs(r.vote_fraction(0, 1) - 1.0) < 1e-9
        assert abs(r.vote_fraction(1, 2) - 2/3) < 1e-9
        assert abs(r.vote_fraction(0, 2) - 1/3) < 1e-9

    def test_vote_fraction_unknown_pair(self):
        r = self._make_result()
        assert r.vote_fraction(5, 6) == 0.0

    def test_is_consensus_true(self):
        r = self._make_result()
        assert r.is_consensus(0, 1) is True
        assert r.is_consensus(1, 2) is True

    def test_is_consensus_false(self):
        r = self._make_result()
        assert r.is_consensus(0, 2) is False

    def test_top_pairs(self):
        r = self._make_result()
        top = r.top_pairs(n=2)
        assert len(top) == 2
        # First should have highest vote count
        assert top[0][1] >= top[1][1]

    def test_top_pairs_all(self):
        r = self._make_result()
        top = r.top_pairs(n=10)
        assert len(top) == 3  # Only 3 pairs in votes

    def test_summary_contains_info(self):
        r = self._make_result()
        s = r.summary()
        assert "ConsensusResult" in s
        assert "methods=3" in s

    def test_no_methods_vote_fraction(self):
        r = ConsensusResult(
            pair_votes={},
            consensus_pairs=set(),
            n_methods=0,
            threshold=0.5,
        )
        assert r.vote_fraction(0, 1) == 0.0


# ─── build_consensus ─────────────────────────────────────────────────────────

class TestBuildConsensus:
    def test_empty_assemblies(self):
        result = build_consensus([], [], [], build_assembly=False)
        assert isinstance(result, ConsensusResult)
        assert result.n_methods == 0
        assert result.pair_votes == {}
        assert result.consensus_pairs == set()

    def test_single_assembly(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([50.0, 0.0], 0.0),
        })
        result = build_consensus([asm], [], [], threshold=0.5, build_assembly=False)
        assert result.n_methods == 1
        assert isinstance(result, ConsensusResult)

    def test_threshold_applied(self):
        asm1 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm2 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm3 = make_assembly({0: ([0.0, 0.0], 0.0), 2: ([50.0, 0.0], 0.0)})
        # Pair (0,1) in 2/3 assemblies, pair (0,2) in 1/3
        result = build_consensus(
            [asm1, asm2, asm3], [], [],
            threshold=0.6, build_assembly=False
        )
        pair_01 = frozenset({0, 1})
        pair_02 = frozenset({0, 2})
        assert pair_01 in result.consensus_pairs
        assert pair_02 not in result.consensus_pairs

    def test_n_methods_correct(self):
        asms = [
            make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
            for _ in range(5)
        ]
        result = build_consensus(asms, [], [], build_assembly=False)
        assert result.n_methods == 5

    def test_threshold_stored(self):
        result = build_consensus([], [], [], threshold=0.75, build_assembly=False)
        assert result.threshold == 0.75


# ─── consensus_score_matrix ──────────────────────────────────────────────────

class TestConsensusScoreMatrix:
    def _make_fragments(self, n):
        return [make_fragment(i) for i in range(n)]

    def test_shape(self):
        frags = self._make_fragments(4)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2},
            consensus_pairs={frozenset({0, 1})},
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat.shape == (4, 4)

    def test_dtype_float64(self):
        frags = self._make_fragments(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 1},
            consensus_pairs=set(),
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat.dtype == np.float64

    def test_symmetric(self):
        frags = self._make_fragments(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2, frozenset({1, 2}): 1},
            consensus_pairs=set(),
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        np.testing.assert_array_equal(mat, mat.T)

    def test_values_in_range(self):
        frags = self._make_fragments(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2, frozenset({1, 2}): 1},
            consensus_pairs=set(),
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_correct_values(self):
        frags = self._make_fragments(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2},
            consensus_pairs=set(),
            n_methods=4,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert abs(mat[0, 1] - 0.5) < 1e-9
        assert abs(mat[1, 0] - 0.5) < 1e-9
        assert mat[0, 2] == 0.0

    def test_empty_pair_votes(self):
        frags = self._make_fragments(3)
        result = ConsensusResult(
            pair_votes={},
            consensus_pairs=set(),
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert (mat == 0.0).all()

    def test_single_fragment(self):
        frags = self._make_fragments(1)
        result = ConsensusResult(
            pair_votes={},
            consensus_pairs=set(),
            n_methods=1,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat.shape == (1, 1)
