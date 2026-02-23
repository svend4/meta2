"""Extra tests for puzzle_reconstruction/matching/consensus.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.consensus import (
    ConsensusResult,
    assembly_to_pairs,
    vote_on_pairs,
    consensus_score_matrix,
    build_consensus,
)
from puzzle_reconstruction.models import (
    Assembly,
    Fragment,
)
from puzzle_reconstruction.models import EdgeSide


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_fragment(fid: int):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    frag = Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)
    frag.placed = True
    frag.position = np.array([0.0, 0.0])
    return frag


def make_assembly(placements: dict) -> Assembly:
    frags = [make_fragment(fid) for fid in placements]
    return Assembly(
        fragments=frags,
        placements={fid: (np.array(pos, dtype=np.float64), angle)
                    for fid, (pos, angle) in placements.items()},
        compat_matrix=np.zeros((1, 1)),
    )


# ─── TestConsensusResultExtra ─────────────────────────────────────────────────

class TestConsensusResultExtra:
    def _result(self, votes, n_methods=3, threshold=0.5):
        consensus = {p for p, v in votes.items()
                     if v / max(1, n_methods) >= threshold}
        return ConsensusResult(
            pair_votes=votes,
            consensus_pairs=consensus,
            n_methods=n_methods,
            threshold=threshold,
        )

    def test_top_pairs_1(self):
        votes = {frozenset({0, 1}): 3, frozenset({1, 2}): 1}
        r = self._result(votes)
        top = r.top_pairs(n=1)
        assert len(top) == 1
        assert top[0][1] == 3

    def test_top_pairs_empty_votes(self):
        r = ConsensusResult(pair_votes={}, consensus_pairs=set(),
                            n_methods=2, threshold=0.5)
        assert r.top_pairs(n=5) == []

    def test_summary_contains_threshold(self):
        r = ConsensusResult(pair_votes={}, consensus_pairs=set(),
                            n_methods=4, threshold=0.75)
        s = r.summary()
        assert "75%" in s

    def test_vote_fraction_n_methods_0(self):
        r = ConsensusResult(pair_votes={}, consensus_pairs=set(),
                            n_methods=0, threshold=0.5)
        assert r.vote_fraction(0, 1) == pytest.approx(0.0)

    def test_is_consensus_unknown_pair_false(self):
        r = ConsensusResult(
            pair_votes={frozenset({0, 1}): 3},
            consensus_pairs={frozenset({0, 1})},
            n_methods=3, threshold=0.5,
        )
        assert r.is_consensus(5, 6) is False

    def test_vote_fraction_full(self):
        r = ConsensusResult(
            pair_votes={frozenset({0, 1}): 5},
            consensus_pairs={frozenset({0, 1})},
            n_methods=5, threshold=0.5,
        )
        assert r.vote_fraction(0, 1) == pytest.approx(1.0)

    def test_threshold_stored(self):
        r = ConsensusResult(pair_votes={}, consensus_pairs=set(),
                            n_methods=2, threshold=0.66)
        assert r.threshold == pytest.approx(0.66)

    def test_assembly_default_none(self):
        r = ConsensusResult(pair_votes={}, consensus_pairs=set(),
                            n_methods=1, threshold=0.5)
        assert r.assembly is None


# ─── TestAssemblyToPairsExtra ─────────────────────────────────────────────────

class TestAssemblyToPairsExtra:
    def test_threshold_exactly_at_distance(self):
        # Fragments exactly 100 apart, threshold=100 → not adjacent (< not <=)
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([100.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=100.0)
        assert len(pairs) == 0

    def test_threshold_just_above_distance(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([100.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=100.1)
        assert frozenset({0, 1}) in pairs

    def test_four_fragments_all_close(self):
        asm = make_assembly({
            i: ([float(i) * 10, 0.0], 0.0) for i in range(4)
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=500.0)
        # C(4,2)=6 pairs all within threshold
        assert len(pairs) == 6

    def test_pairs_are_frozensets_of_size_2(self):
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([10.0, 0.0], 0.0),
            2: ([20.0, 0.0], 0.0),
        })
        pairs = assembly_to_pairs(asm, adjacency_threshold=50.0)
        for p in pairs:
            assert isinstance(p, frozenset)
            assert len(p) == 2

    def test_diagonal_fragments(self):
        # Placed diagonally; ensure distance is computed correctly
        asm = make_assembly({
            0: ([0.0, 0.0], 0.0),
            1: ([30.0, 40.0], 0.0),  # distance = 50
        })
        pairs_close = assembly_to_pairs(asm, adjacency_threshold=60.0)
        pairs_far = assembly_to_pairs(asm, adjacency_threshold=40.0)
        assert frozenset({0, 1}) in pairs_close
        assert frozenset({0, 1}) not in pairs_far


# ─── TestVoteOnPairsExtra ─────────────────────────────────────────────────────

class TestVoteOnPairsExtra:
    def test_three_assemblies_same_pair(self):
        asms = [
            make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
            for _ in range(3)
        ]
        votes = vote_on_pairs(asms, adjacency_threshold=100.0)
        assert votes[frozenset({0, 1})] == 3

    def test_empty_returns_empty(self):
        assert vote_on_pairs([]) == {}

    def test_no_adjacent_pairs(self):
        asms = [make_assembly({0: ([0.0, 0.0], 0.0), 1: ([999.0, 0.0], 0.0)})]
        votes = vote_on_pairs(asms, adjacency_threshold=100.0)
        assert len(votes) == 0

    def test_vote_counts_nonneg(self):
        asms = [
            make_assembly({0: ([0.0, 0.0], 0.0), 1: ([30.0, 0.0], 0.0)}),
            make_assembly({0: ([0.0, 0.0], 0.0), 2: ([30.0, 0.0], 0.0)}),
        ]
        votes = vote_on_pairs(asms, adjacency_threshold=50.0)
        for v in votes.values():
            assert v >= 1


# ─── TestBuildConsensusExtra ──────────────────────────────────────────────────

class TestBuildConsensusExtra:
    def test_threshold_1_0_requires_all(self):
        asm1 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm2 = make_assembly({0: ([0.0, 0.0], 0.0), 2: ([50.0, 0.0], 0.0)})
        result = build_consensus([asm1, asm2], [], [], threshold=1.0,
                                  build_assembly=False)
        # Pair {0,1} only in 1/2 → not consensus
        assert frozenset({0, 1}) not in result.consensus_pairs

    def test_low_threshold_includes_more(self):
        asm1 = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        asm2 = make_assembly({0: ([0.0, 0.0], 0.0), 2: ([50.0, 0.0], 0.0)})
        result = build_consensus([asm1, asm2], [], [], threshold=0.4,
                                  build_assembly=False)
        # Both pairs appear in at least 1/2 = 0.5 ≥ 0.4 methods
        assert frozenset({0, 1}) in result.consensus_pairs
        assert frozenset({0, 2}) in result.consensus_pairs

    def test_pair_votes_populated(self):
        asm = make_assembly({0: ([0.0, 0.0], 0.0), 1: ([50.0, 0.0], 0.0)})
        result = build_consensus([asm], [], [], threshold=0.5,
                                  build_assembly=False)
        assert frozenset({0, 1}) in result.pair_votes


# ─── TestConsensusScoreMatrixExtra ────────────────────────────────────────────

class TestConsensusScoreMatrixExtra:
    def _frags(self, n):
        return [make_fragment(i) for i in range(n)]

    def test_zero_diagonal(self):
        frags = self._frags(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2},
            consensus_pairs=set(),
            n_methods=4,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        import numpy as np
        assert np.allclose(np.diag(mat), 0.0)

    def test_unknown_pair_gives_zero(self):
        frags = self._frags(4)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2},
            consensus_pairs=set(),
            n_methods=4,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat[2, 3] == pytest.approx(0.0)
        assert mat[3, 2] == pytest.approx(0.0)

    def test_high_vote_fraction(self):
        frags = self._frags(2)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 4},
            consensus_pairs={frozenset({0, 1})},
            n_methods=4,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        assert mat[0, 1] == pytest.approx(1.0)
        assert mat[1, 0] == pytest.approx(1.0)

    def test_values_in_0_1(self):
        frags = self._frags(3)
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 1, frozenset({1, 2}): 3},
            consensus_pairs=set(),
            n_methods=3,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        import numpy as np
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)
