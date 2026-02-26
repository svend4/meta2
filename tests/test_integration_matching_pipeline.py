"""
Integration tests for the matching modules.

Covers:
    - puzzle_reconstruction.matching.compat_matrix
    - puzzle_reconstruction.matching.consensus
    - puzzle_reconstruction.matching.score_aggregator
    - puzzle_reconstruction.matching.global_matcher
    - Cross-module integration pipelines

Test classes:
    TestCompatMatrix        (11 tests)
    TestConsensus           (11 tests)
    TestScoreAggregator     (11 tests)
    TestGlobalMatcher       (11 tests)
    TestMatchingIntegration (11 tests)
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment,
    CompatEntry,
    Assembly,
    Placement,
    FractalSignature,
    TangramSignature,
    ShapeClass,
    EdgeSignature,
    EdgeSide,
)
from puzzle_reconstruction.matching.compat_matrix import (
    build_compat_matrix,
    match_score,
    top_candidates,
)
from puzzle_reconstruction.matching.consensus import (
    build_consensus,
    ConsensusResult,
    vote_on_pairs,
    consensus_score_matrix,
    assembly_to_pairs,
)
from puzzle_reconstruction.matching.score_aggregator import (
    AggregatedScore,
    AggregationConfig,
    AggregationReport,
    aggregate_scores,
    aggregate_score_matrix,
    batch_aggregate_scores,
    filter_aggregated,
)
from puzzle_reconstruction.matching.global_matcher import (
    GlobalMatch,
    GlobalMatchConfig,
    GlobalMatchResult,
    global_match,
    aggregate_pair_scores,
    filter_matches,
    merge_match_results,
    rank_candidates,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_edge(edge_id: int, side: EdgeSide, rng: np.random.Generator) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=rng.random((5, 2)),
        fd=float(rng.random()),
        css_vec=rng.random(10),
        ifs_coeffs=rng.random(6),
        length=60.0,
    )


def _make_fragment(fid: int, rng: np.random.Generator) -> Fragment:
    img = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.ones((60, 60), dtype=np.uint8) * 255
    contour = np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=np.int32)
    fractal = FractalSignature(
        fd_box=1.3,
        fd_divider=1.2,
        ifs_coeffs=np.zeros(6),
        css_image=[],
        chain_code="",
        curve=np.zeros((4, 2)),
    )
    tangram = TangramSignature(
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=0.5,
    )
    edges = [
        _make_edge(0, EdgeSide.TOP, rng),
        _make_edge(1, EdgeSide.BOTTOM, rng),
    ]
    return Fragment(
        fragment_id=fid,
        image=img,
        mask=mask,
        contour=contour,
        edges=edges,
        tangram=tangram,
        fractal=fractal,
    )


def _make_assembly(fragment_ids: List[int], score: float = 0.7, spacing: float = 70.0) -> Assembly:
    placements = [
        Placement(fragment_id=fid, position=(i * spacing, 0.0), rotation=0.0)
        for i, fid in enumerate(fragment_ids)
    ]
    return Assembly(placements=placements, total_score=score, method="greedy")


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def fragments(rng) -> List[Fragment]:
    return [_make_fragment(i, rng) for i in range(4)]


@pytest.fixture(scope="module")
def compat_result(fragments):
    return build_compat_matrix(fragments)


@pytest.fixture(scope="module")
def compat_matrix(compat_result):
    matrix, _ = compat_result
    return matrix


@pytest.fixture(scope="module")
def compat_entries(compat_result):
    _, entries = compat_result
    return entries


@pytest.fixture(scope="module")
def all_edges(fragments) -> List[EdgeSignature]:
    return [e for f in fragments for e in f.edges]


@pytest.fixture(scope="module")
def assemblies(fragments) -> List[Assembly]:
    a1 = _make_assembly([0, 1, 2])
    a2 = _make_assembly([0, 2, 3])
    a3 = _make_assembly([1, 3])
    return [a1, a2, a3]


# ---------------------------------------------------------------------------
# TestCompatMatrix
# ---------------------------------------------------------------------------

class TestCompatMatrix:
    """Tests for build_compat_matrix, match_score, top_candidates."""

    def test_build_compat_matrix_returns_tuple(self, compat_result):
        matrix, entries = compat_result
        assert isinstance(matrix, np.ndarray)
        assert isinstance(entries, list)

    def test_matrix_is_square(self, compat_matrix, fragments):
        n_edges = sum(len(f.edges) for f in fragments)
        assert compat_matrix.shape == (n_edges, n_edges)

    def test_matrix_diagonal_is_zero(self, compat_matrix):
        diag = np.diag(compat_matrix)
        assert np.all(diag == 0.0)

    def test_entries_are_compat_entry_objects(self, compat_entries):
        assert len(compat_entries) > 0
        for entry in compat_entries:
            assert isinstance(entry, CompatEntry)

    def test_entries_count_is_cross_fragment_edge_pairs(self, fragments, compat_entries):
        import math
        n_total_edges = sum(len(f.edges) for f in fragments)
        intra_pairs = sum(math.comb(len(f.edges), 2) for f in fragments)
        expected = math.comb(n_total_edges, 2) - intra_pairs
        assert len(compat_entries) == expected

    def test_matrix_values_are_finite(self, compat_matrix):
        off_diag = compat_matrix[compat_matrix != 0]
        assert np.all(np.isfinite(off_diag))

    def test_match_score_returns_compat_entry(self, all_edges):
        e_i = all_edges[0]
        e_j = all_edges[2]
        result = match_score(e_i, e_j)
        assert isinstance(result, CompatEntry)

    def test_top_candidates_returns_list(self, compat_matrix, all_edges):
        result = top_candidates(compat_matrix, all_edges, 0, k=3)
        assert isinstance(result, list)

    def test_top_candidates_indices_in_range(self, compat_matrix, all_edges):
        result = top_candidates(compat_matrix, all_edges, 0, k=5)
        n = len(all_edges)
        for idx, _score in result:
            assert 0 <= idx < n

    def test_top_candidates_at_most_k(self, compat_matrix, all_edges):
        k = 2
        result = top_candidates(compat_matrix, all_edges, 0, k=k)
        assert len(result) <= k

    def test_matrix_non_negative_off_diagonal(self, compat_matrix):
        n = compat_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert compat_matrix[i, j] >= 0.0


# ---------------------------------------------------------------------------
# TestConsensus
# ---------------------------------------------------------------------------

class TestConsensus:
    """Tests for build_consensus, vote_on_pairs, consensus_score_matrix, assembly_to_pairs."""

    def test_build_consensus_returns_consensus_result(self, fragments, assemblies, compat_entries):
        result = build_consensus(assemblies, fragments, compat_entries)
        assert isinstance(result, ConsensusResult)

    def test_consensus_result_has_pair_votes(self, fragments, assemblies, compat_entries):
        result = build_consensus(assemblies, fragments, compat_entries)
        assert hasattr(result, "pair_votes")
        assert isinstance(result.pair_votes, dict)

    def test_vote_on_pairs_returns_dict(self, assemblies):
        votes = vote_on_pairs(assemblies)
        assert isinstance(votes, dict)

    def test_vote_on_pairs_keys_are_pairs(self, assemblies):
        votes = vote_on_pairs(assemblies)
        for key in votes:
            assert isinstance(key, frozenset)
            assert len(key) == 2

    def test_consensus_score_matrix_returns_2d_array(self, fragments, assemblies, compat_entries):
        cr = build_consensus(assemblies, fragments, compat_entries)
        sm = consensus_score_matrix(cr, fragments)
        assert isinstance(sm, np.ndarray)
        assert sm.ndim == 2

    def test_assembly_to_pairs_returns_set_or_list(self):
        asm = _make_assembly([0, 1, 2])
        pairs = assembly_to_pairs(asm)
        assert isinstance(pairs, (set, list, frozenset))

    def test_consensus_result_has_consensus_pairs(self, fragments, assemblies, compat_entries):
        result = build_consensus(assemblies, fragments, compat_entries)
        assert hasattr(result, "consensus_pairs")

    def test_build_consensus_on_single_assembly(self, fragments, compat_entries):
        asm = _make_assembly([0, 1])
        result = build_consensus([asm], fragments, compat_entries)
        assert isinstance(result, ConsensusResult)

    def test_vote_on_pairs_multiple_assemblies_counts_correctly(self, assemblies):
        votes = vote_on_pairs(assemblies)
        # pair (0,2) appears in both a1 and a2 => count >= 2
        pair_02 = frozenset({0, 2})
        if pair_02 in votes:
            assert votes[pair_02] >= 2

    def test_consensus_score_matrix_shape_matches_fragments(self, fragments, assemblies, compat_entries):
        cr = build_consensus(assemblies, fragments, compat_entries)
        sm = consensus_score_matrix(cr, fragments)
        n = len(fragments)
        assert sm.shape == (n, n)

    def test_consensus_result_attributes_are_finite(self, fragments, assemblies, compat_entries):
        cr = build_consensus(assemblies, fragments, compat_entries)
        sm = consensus_score_matrix(cr, fragments)
        assert np.all(np.isfinite(sm))


# ---------------------------------------------------------------------------
# TestScoreAggregator
# ---------------------------------------------------------------------------

class TestScoreAggregator:
    """Tests for aggregate_scores, aggregate_score_matrix, batch_aggregate_scores, filter_aggregated."""

    def test_aggregate_scores_returns_aggregated_score(self):
        result = aggregate_scores([0.6, 0.8, 0.7], pair=(0, 1))
        assert isinstance(result, AggregatedScore)

    def test_aggregated_score_has_score_attribute(self):
        result = aggregate_scores([0.5, 0.9])
        assert hasattr(result, "score")
        assert isinstance(result.score, float)

    def test_aggregate_score_matrix_returns_ndarray(self):
        m1 = np.array([[0.0, 0.6], [0.6, 0.0]])
        m2 = np.array([[0.0, 0.8], [0.8, 0.0]])
        result = aggregate_score_matrix([m1, m2])
        assert isinstance(result, np.ndarray)

    def test_aggregate_score_matrix_shape_correct(self):
        m1 = np.zeros((4, 4))
        m2 = np.ones((4, 4)) * 0.5
        result = aggregate_score_matrix([m1, m2])
        assert result.shape == (4, 4)

    def test_aggregate_score_matrix_values_finite(self):
        m1 = np.array([[0.0, 0.5, 0.3], [0.5, 0.0, 0.7], [0.3, 0.7, 0.0]])
        m2 = np.array([[0.0, 0.4, 0.6], [0.4, 0.0, 0.2], [0.6, 0.2, 0.0]])
        result = aggregate_score_matrix([m1, m2])
        assert np.all(np.isfinite(result))

    def test_batch_aggregate_scores_returns_aggregation_report(self):
        pairs = [(0, 1), (1, 2)]
        score_lists = [[0.6, 0.7], [0.4, 0.5]]
        result = batch_aggregate_scores(pairs, score_lists)
        assert isinstance(result, AggregationReport)

    def test_batch_aggregate_scores_count_matches_input(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        score_lists = [[0.8, 0.6], [0.5, 0.4], [0.3, 0.7]]
        report = batch_aggregate_scores(pairs, score_lists)
        assert report.n_pairs == len(pairs)

    def test_filter_aggregated_returns_list(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        score_lists = [[0.8, 0.6], [0.5, 0.4], [0.3, 0.2]]
        report = batch_aggregate_scores(pairs, score_lists)
        result = filter_aggregated(report, threshold=0.5)
        assert isinstance(result, list)

    def test_aggregation_config_constructable(self):
        cfg = AggregationConfig()
        assert hasattr(cfg, "strategy")

    def test_aggregate_scores_with_single_score(self):
        result = aggregate_scores([0.75])
        assert abs(result.score - 0.75) < 1e-6

    def test_aggregate_score_matrix_diagonal_zero(self):
        m1 = np.array([[0.0, 0.6, 0.4], [0.6, 0.0, 0.7], [0.4, 0.7, 0.0]])
        m2 = np.array([[0.0, 0.5, 0.3], [0.5, 0.0, 0.8], [0.3, 0.8, 0.0]])
        result = aggregate_score_matrix([m1, m2])
        diag = np.diag(result)
        assert np.allclose(diag, 0.0)


# ---------------------------------------------------------------------------
# TestGlobalMatcher
# ---------------------------------------------------------------------------

class TestGlobalMatcher:
    """Tests for global_match, aggregate_pair_scores, filter_matches, rank_candidates, merge_match_results."""

    _scores_per_channel = {
        "shape": {(0, 1): 0.8, (0, 2): 0.6, (1, 2): 0.7},
        "color": {(0, 1): 0.7, (0, 2): 0.5, (1, 2): 0.9},
    }
    _fragment_ids = [0, 1, 2]

    def test_global_match_returns_global_match_result(self):
        result = global_match(self._fragment_ids, self._scores_per_channel)
        assert isinstance(result, GlobalMatchResult)

    def test_global_match_result_has_matches(self):
        result = global_match(self._fragment_ids, self._scores_per_channel)
        assert hasattr(result, "matches")
        assert isinstance(result.matches, dict)

    def test_aggregate_pair_scores_returns_dict(self):
        result = aggregate_pair_scores(self._scores_per_channel)
        assert isinstance(result, dict)

    def test_filter_matches_returns_global_match_result(self):
        result = global_match(self._fragment_ids, self._scores_per_channel)
        filtered = filter_matches(result, min_score=0.5)
        assert isinstance(filtered, GlobalMatchResult)

    def test_rank_candidates_returns_list(self):
        pair_scores = {(0, 1): 0.8, (0, 2): 0.6}
        result = rank_candidates(0, pair_scores)
        assert isinstance(result, list)

    def test_global_match_object_has_required_attributes(self):
        result = global_match(self._fragment_ids, self._scores_per_channel)
        first_fid = self._fragment_ids[0]
        match_list = result.matches[first_fid]
        assert len(match_list) > 0
        m = match_list[0]
        assert hasattr(m, "fragment_id")
        assert hasattr(m, "candidate_id")
        assert hasattr(m, "score")

    def test_filter_matches_with_high_threshold_removes_low_scores(self):
        result = global_match(self._fragment_ids, self._scores_per_channel)
        filtered = filter_matches(result, min_score=0.95)
        for fid in filtered.matches:
            for m in filtered.matches[fid]:
                assert m.score >= 0.95

    def test_merge_match_results_returns_global_match_result(self):
        r1 = global_match(self._fragment_ids, {"shape": {(0, 1): 0.8, (0, 2): 0.6, (1, 2): 0.7}})
        r2 = global_match(self._fragment_ids, {"color": {(0, 1): 0.7, (0, 2): 0.5, (1, 2): 0.9}})
        merged = merge_match_results([r1, r2])
        assert isinstance(merged, GlobalMatchResult)

    def test_global_match_config_constructable(self):
        cfg = GlobalMatchConfig()
        assert hasattr(cfg, "top_k")
        assert hasattr(cfg, "min_score")

    def test_global_match_on_single_fragment_trivial_case(self):
        result = global_match([0], {"shape": {}})
        assert isinstance(result, GlobalMatchResult)
        assert result.n_fragments == 1
        assert result.matches[0] == []

    def test_filter_matches_empty_input(self):
        result = global_match([], {})
        filtered = filter_matches(result, min_score=0.0)
        assert isinstance(filtered, GlobalMatchResult)
        assert filtered.matches == {}


# ---------------------------------------------------------------------------
# TestMatchingIntegration
# ---------------------------------------------------------------------------

class TestMatchingIntegration:
    """Cross-module integration tests combining multiple matching pipeline stages."""

    def test_build_compat_matrix_then_top_candidates_pipeline(self, fragments, compat_matrix, all_edges):
        tc = top_candidates(compat_matrix, all_edges, 0, k=3)
        assert isinstance(tc, list)
        assert len(tc) <= 3
        n_edges = len(all_edges)
        for idx, score in tc:
            assert 0 <= idx < n_edges
            assert 0.0 <= score <= 1.0

    def test_aggregate_score_matrix_then_filter_pipeline(self):
        m1 = np.array([[0.0, 0.8, 0.4], [0.8, 0.0, 0.6], [0.4, 0.6, 0.0]])
        m2 = np.array([[0.0, 0.9, 0.3], [0.9, 0.0, 0.5], [0.3, 0.5, 0.0]])
        agg_matrix = aggregate_score_matrix([m1, m2])
        # Build batch report from the aggregated matrix values
        pairs = [(0, 1), (0, 2), (1, 2)]
        score_lists = [
            [agg_matrix[0, 1]],
            [agg_matrix[0, 2]],
            [agg_matrix[1, 2]],
        ]
        report = batch_aggregate_scores(pairs, score_lists)
        kept = filter_aggregated(report, threshold=0.5)
        for agg in kept:
            assert agg.score >= 0.5

    def test_vote_on_pairs_then_build_consensus_pipeline(self, fragments, compat_entries):
        assemblies = [
            _make_assembly([0, 1, 2]),
            _make_assembly([0, 2, 3]),
        ]
        votes = vote_on_pairs(assemblies)
        assert isinstance(votes, dict)
        cr = build_consensus(assemblies, fragments, compat_entries)
        assert isinstance(cr, ConsensusResult)
        assert cr.n_methods > 0

    def test_full_matching_chain_no_crash(self, fragments, compat_matrix, compat_entries, all_edges):
        assemblies = [_make_assembly([0, 1, 2]), _make_assembly([1, 2, 3])]
        votes = vote_on_pairs(assemblies)
        cr = build_consensus(assemblies, fragments, compat_entries)
        sm = consensus_score_matrix(cr, fragments)
        assert sm.shape == (len(fragments), len(fragments))
        # Build global match from compat scores
        frag_ids = [f.fragment_id for f in fragments]
        edge_scores: Dict[Tuple[int, int], float] = {}
        for i, ei in enumerate(all_edges):
            for j, ej in enumerate(all_edges):
                if i != j:
                    fi = fragments[i // 2].fragment_id
                    fj = fragments[j // 2].fragment_id
                    if fi != fj:
                        key = (min(fi, fj), max(fi, fj))
                        if key not in edge_scores:
                            edge_scores[key] = float(compat_matrix[i, j])
        result = global_match(frag_ids, {"compat": edge_scores})
        assert isinstance(result, GlobalMatchResult)

    def test_aggregate_scores_consistent_with_aggregate_score_matrix(self):
        m1 = np.array([[0.0, 0.6], [0.6, 0.0]])
        m2 = np.array([[0.0, 0.8], [0.8, 0.0]])
        mat_result = aggregate_score_matrix([m1, m2])
        scalar_result = aggregate_scores([0.6, 0.8], pair=(0, 1))
        assert abs(mat_result[0, 1] - scalar_result.score) < 1e-6

    def test_top_candidates_from_compat_matrix_then_global_match(self, fragments, compat_matrix, compat_entries, all_edges):
        frag_ids = [f.fragment_id for f in fragments]
        edge_scores: Dict[Tuple[int, int], float] = {}
        for i, ei in enumerate(all_edges):
            for j, ej in enumerate(all_edges):
                if i != j:
                    fi = fragments[i // 2].fragment_id
                    fj = fragments[j // 2].fragment_id
                    if fi != fj:
                        key = (min(fi, fj), max(fi, fj))
                        if key not in edge_scores:
                            edge_scores[key] = float(compat_matrix[i, j])
        result = global_match(frag_ids, {"edge_compat": edge_scores})
        assert result.n_fragments == len(frag_ids)
        for fid in frag_ids:
            assert fid in result.matches

    def test_consensus_from_multiple_assemblies(self, fragments, compat_entries):
        assemblies = [
            _make_assembly([0, 1]),
            _make_assembly([1, 2]),
            _make_assembly([0, 2, 3]),
            _make_assembly([1, 3]),
        ]
        cr = build_consensus(assemblies, fragments, compat_entries)
        assert isinstance(cr, ConsensusResult)
        assert len(cr.pair_votes) > 0

    def test_filter_aggregated_threshold_zero_keeps_all(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        score_lists = [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        report = batch_aggregate_scores(pairs, score_lists)
        kept = filter_aggregated(report, threshold=0.0)
        assert len(kept) == len(pairs)

    def test_filter_aggregated_threshold_one_keeps_none(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        score_lists = [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        report = batch_aggregate_scores(pairs, score_lists)
        kept = filter_aggregated(report, threshold=1.0)
        assert len(kept) == 0

    def test_aggregate_score_matrix_then_normalize(self):
        m1 = np.array([[0.0, 0.6, 0.3], [0.6, 0.0, 0.9], [0.3, 0.9, 0.0]])
        m2 = np.array([[0.0, 0.4, 0.7], [0.4, 0.0, 0.5], [0.7, 0.5, 0.0]])
        result = aggregate_score_matrix([m1, m2])
        max_val = result.max()
        if max_val > 0:
            normalized = result / max_val
            assert normalized.max() <= 1.0 + 1e-9
        assert result.shape == (3, 3)

    def test_batch_aggregate_scores_same_as_single_aggregate(self):
        pairs = [(0, 1)]
        source_scores = [0.4, 0.6, 0.8]
        report = batch_aggregate_scores(pairs, [source_scores])
        single = aggregate_scores(source_scores, pair=(0, 1))
        assert abs(report.scores[0].score - single.score) < 1e-9
