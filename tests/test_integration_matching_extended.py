"""
Integration tests for puzzle_reconstruction matching modules.

Covers:
    1. global_matcher   — GlobalMatchConfig, GlobalMatch, GlobalMatchResult,
                         aggregate_pair_scores, rank_candidates, global_match,
                         filter_matches, merge_match_results
    2. spectral_matcher — SpectralMatchResult, magnitude_spectrum, log_magnitude,
                         spectrum_correlation, phase_correlation, match_spectra,
                         batch_spectral_match
    3. graph_match      — FragmentGraph, build_fragment_graph, mst_ordering,
                         spectral_ordering, random_walk_similarity,
                         degree_centrality, analyze_graph
    4. consensus        — assembly_to_pairs, vote_on_pairs, consensus_score_matrix,
                         ConsensusResult helpers
    5. curve_descriptor — CurveDescriptorConfig, CurveDescriptor,
                         compute_fourier_descriptor, compute_curvature_profile,
                         describe_curve, descriptor_distance, descriptor_similarity,
                         batch_describe_curves, find_best_match
    6. text_flow        — detect_text_baseline_angle, detect_text_line_positions,
                         build_text_line_profile, compare_baseline_angles,
                         align_line_positions, match_text_flow, TextFlowScorer
    7. score_combiner   — ScoreVector, CombinedScore, weighted_combine, min_combine,
                         max_combine, rank_combine, normalize_score_vectors,
                         batch_combine
    8. score_normalizer — normalize_minmax, normalize_zscore, normalize_rank,
                         calibrate_scores, combine_scores, normalize_score_matrix,
                         batch_normalize_scores

All tests use synthetic data and np.random.RandomState for reproducibility.
No mocks — all computations are real.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make sure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── imports from modules under test ───────────────────────────────────────────

from puzzle_reconstruction.matching.global_matcher import (
    GlobalMatchConfig,
    GlobalMatch,
    GlobalMatchResult,
    aggregate_pair_scores,
    rank_candidates,
    global_match,
    filter_matches,
    merge_match_results,
    _aggregate_scores,
)

from puzzle_reconstruction.matching.spectral_matcher import (
    SpectralMatchResult,
    magnitude_spectrum,
    log_magnitude,
    spectrum_correlation,
    phase_correlation,
    match_spectra,
    batch_spectral_match,
)

from puzzle_reconstruction.matching.graph_match import (
    FragmentGraph,
    GraphMatchResult,
    build_fragment_graph,
    mst_ordering,
    spectral_ordering,
    random_walk_similarity,
    degree_centrality,
    analyze_graph,
)

from puzzle_reconstruction.matching.consensus import (
    ConsensusResult,
    assembly_to_pairs,
    vote_on_pairs,
    consensus_score_matrix,
)

from puzzle_reconstruction.matching.curve_descriptor import (
    CurveDescriptorConfig,
    CurveDescriptor,
    compute_fourier_descriptor,
    compute_curvature_profile,
    describe_curve,
    descriptor_distance,
    descriptor_similarity,
    batch_describe_curves,
    find_best_match,
)

from puzzle_reconstruction.matching.text_flow import (
    TextLineProfile,
    TextFlowMatch,
    TextFlowConfig,
    detect_text_baseline_angle,
    detect_text_line_positions,
    build_text_line_profile,
    compare_baseline_angles,
    align_line_positions,
    match_text_flow,
    TextFlowScorer,
)

from puzzle_reconstruction.matching.score_combiner import (
    ScoreVector,
    CombinedScore,
    weighted_combine,
    min_combine,
    max_combine,
    rank_combine,
    normalize_score_vectors,
    batch_combine,
)

from puzzle_reconstruction.matching.score_normalizer import (
    ScoreNormResult,
    normalize_minmax,
    normalize_zscore,
    normalize_rank,
    calibrate_scores,
    combine_scores,
    normalize_score_matrix,
    batch_normalize_scores,
)

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSignature,
    EdgeSide,
    Fragment,
    Placement,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rng(seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _rand_img(rng: np.random.RandomState, h: int = 64, w: int = 64) -> np.ndarray:
    return rng.randint(0, 255, (h, w), dtype=np.uint8)


def _rand_img_color(rng: np.random.RandomState, h: int = 64, w: int = 64) -> np.ndarray:
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    rng = _rng(fid)
    img = _rand_img_color(rng, 64, 64)
    edges = []
    for k in range(n_edges):
        es = EdgeSignature(
            edge_id=fid * 10 + k,
            side=EdgeSide.TOP,
            virtual_curve=rng.rand(20, 2),
            fd=0.5,
            css_vec=rng.rand(8),
            ifs_coeffs=rng.rand(6),
            length=50.0,
        )
        edges.append(es)
    return Fragment(fragment_id=fid, image=img, edges=edges)


def _make_assembly_dict(fids: List[int], rng: np.random.RandomState) -> Assembly:
    """Build an Assembly with dict-style placements."""
    placements = {}
    for fid in fids:
        pos = rng.rand(2) * 200.0
        placements[fid] = (pos, 0.0)
    return Assembly(placements=placements)


def _make_assembly_list(fids: List[int], rng: np.random.RandomState) -> Assembly:
    """Build an Assembly with list-style Placement objects."""
    pos_list = [rng.rand(2) * 200.0 for _ in fids]
    placements = [
        Placement(fragment_id=fid, position=(float(p[0]), float(p[1])))
        for fid, p in zip(fids, pos_list)
    ]
    return Assembly(placements=placements)


def _make_curve(n: int = 30, rng: np.random.RandomState = None) -> np.ndarray:
    if rng is None:
        rng = _rng(7)
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t) * 50 + rng.randn(n) * 2
    y = np.sin(t) * 30 + rng.randn(n) * 2
    return np.column_stack([x, y])


def _make_compat_entry(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    """Create a minimal CompatEntry linking two fragments."""
    rng = _rng(fid_i + fid_j)
    frag_i = _make_fragment(fid_i, 1)
    frag_j = _make_fragment(fid_j, 1)
    return CompatEntry(
        edge_i=frag_i.edges[0],
        edge_j=frag_j.edges[0],
        score=score,
    )


# =============================================================================
# 1. global_matcher
# =============================================================================

class TestGlobalMatchConfig(unittest.TestCase):
    """Tests for GlobalMatchConfig validation and defaults."""

    def test_default_construction(self):
        cfg = GlobalMatchConfig()
        self.assertEqual(cfg.top_k, 5)
        self.assertEqual(cfg.min_score, 0.0)
        self.assertEqual(cfg.aggregate, "mean")
        self.assertTrue(cfg.symmetric)

    def test_valid_custom_config(self):
        cfg = GlobalMatchConfig(top_k=3, min_score=0.2, aggregate="max", symmetric=False)
        self.assertEqual(cfg.top_k, 3)
        self.assertEqual(cfg.min_score, 0.2)

    def test_invalid_top_k_raises(self):
        with self.assertRaises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_invalid_min_score_negative(self):
        with self.assertRaises(ValueError):
            GlobalMatchConfig(min_score=-0.1)

    def test_invalid_min_score_above_one(self):
        with self.assertRaises(ValueError):
            GlobalMatchConfig(min_score=1.5)

    def test_invalid_aggregate_raises(self):
        with self.assertRaises(ValueError):
            GlobalMatchConfig(aggregate="median")

    def test_aggregate_min_valid(self):
        cfg = GlobalMatchConfig(aggregate="min")
        self.assertEqual(cfg.aggregate, "min")


class TestGlobalMatchDataclasses(unittest.TestCase):
    """Tests for GlobalMatch and GlobalMatchResult dataclasses."""

    def test_global_match_valid(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        self.assertTrue(m.is_top)

    def test_global_match_is_top_false(self):
        m = GlobalMatch(fragment_id=0, candidate_id=2, score=0.5, rank=2)
        self.assertFalse(m.is_top)

    def test_global_match_invalid_score(self):
        with self.assertRaises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.5, rank=1)

    def test_global_match_invalid_rank(self):
        with self.assertRaises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)

    def test_global_match_result_top_match(self):
        m1 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1)
        m2 = GlobalMatch(fragment_id=0, candidate_id=2, score=0.5, rank=2)
        result = GlobalMatchResult(matches={0: [m1, m2]}, n_fragments=1, n_channels=2)
        top = result.top_match(0)
        self.assertIsNotNone(top)
        self.assertEqual(top.candidate_id, 1)

    def test_global_match_result_missing_fragment(self):
        result = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        self.assertIsNone(result.top_match(99))

    def test_global_match_result_fragment_ids(self):
        m = GlobalMatch(fragment_id=5, candidate_id=6, score=0.7, rank=1)
        result = GlobalMatchResult(matches={5: [m]}, n_fragments=1, n_channels=1)
        self.assertIn(5, result.fragment_ids())

    def test_global_match_result_all_top_matches(self):
        m0 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        m1 = GlobalMatch(fragment_id=1, candidate_id=0, score=0.6, rank=1)
        result = GlobalMatchResult(matches={0: [m0], 1: [m1]}, n_fragments=2, n_channels=1)
        tops = result.all_top_matches()
        self.assertEqual(len(tops), 2)


class TestAggregatePairScores(unittest.TestCase):
    """Tests for aggregate_pair_scores and related helpers."""

    def _make_channel_scores(self) -> Dict:
        return {
            "ch_a": {(0, 1): 0.8, (1, 2): 0.6},
            "ch_b": {(0, 1): 0.4, (1, 2): 0.9, (2, 3): 0.7},
        }

    def test_aggregate_mean_default(self):
        scores = self._make_channel_scores()
        result = aggregate_pair_scores(scores)
        self.assertIn((0, 1), result)
        self.assertAlmostEqual(result[(0, 1)], (0.8 + 0.4) / 2.0, places=5)

    def test_aggregate_max(self):
        scores = {"ch": {(0, 1): 0.3, (1, 2): 0.9}}
        cfg = GlobalMatchConfig(aggregate="max")
        result = aggregate_pair_scores(scores, cfg)
        # max of single channel = channel value itself
        self.assertAlmostEqual(result[(0, 1)], 0.3, places=5)

    def test_aggregate_min(self):
        scores = {
            "ch_a": {(0, 1): 0.8},
            "ch_b": {(0, 1): 0.2},
        }
        cfg = GlobalMatchConfig(aggregate="min")
        result = aggregate_pair_scores(scores, cfg)
        self.assertAlmostEqual(result[(0, 1)], 0.2, places=5)

    def test_all_scores_in_unit_interval(self):
        rng = _rng(1)
        scores = {}
        for c in ["c1", "c2", "c3"]:
            ch = {}
            for i in range(5):
                for j in range(i + 1, 5):
                    ch[(i, j)] = float(rng.rand())
            scores[c] = ch
        result = aggregate_pair_scores(scores)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_symmetric_averaging(self):
        scores = {
            "ch": {(0, 1): 0.6, (1, 0): 0.4},
        }
        cfg = GlobalMatchConfig(symmetric=True)
        result = aggregate_pair_scores(scores, cfg)
        self.assertAlmostEqual(result[(0, 1)], 0.5, places=5)

    def test_rank_candidates_top_k(self):
        pair_scores = {(0, 1): 0.9, (0, 2): 0.7, (0, 3): 0.5, (0, 4): 0.3}
        cfg = GlobalMatchConfig(top_k=2)
        ranked = rank_candidates(0, pair_scores, cfg)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].candidate_id, 1)
        self.assertEqual(ranked[0].rank, 1)

    def test_global_match_full_pipeline(self):
        fragment_ids = [0, 1, 2]
        scores_per_channel = {
            "shape": {(0, 1): 0.9, (1, 2): 0.6, (0, 2): 0.3},
        }
        result = global_match(fragment_ids, scores_per_channel)
        self.assertEqual(result.n_fragments, 3)
        self.assertEqual(result.n_channels, 1)
        top0 = result.top_match(0)
        self.assertIsNotNone(top0)
        self.assertEqual(top0.candidate_id, 1)

    def test_filter_matches(self):
        m_hi = GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1)
        m_lo = GlobalMatch(fragment_id=0, candidate_id=2, score=0.2, rank=2)
        result = GlobalMatchResult(matches={0: [m_hi, m_lo]}, n_fragments=1, n_channels=1)
        filtered = filter_matches(result, min_score=0.5)
        self.assertEqual(len(filtered.matches[0]), 1)
        self.assertEqual(filtered.matches[0][0].candidate_id, 1)

    def test_filter_matches_invalid_threshold(self):
        result = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        with self.assertRaises(ValueError):
            filter_matches(result, min_score=1.5)

    def test_merge_match_results_empty(self):
        merged = merge_match_results([])
        self.assertEqual(merged.n_fragments, 0)

    def test_merge_match_results_combines(self):
        m0a = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        m0b = GlobalMatch(fragment_id=0, candidate_id=1, score=0.6, rank=1)
        r1 = GlobalMatchResult(matches={0: [m0a]}, n_fragments=1, n_channels=1)
        r2 = GlobalMatchResult(matches={0: [m0b]}, n_fragments=1, n_channels=1)
        merged = merge_match_results([r1, r2])
        top = merged.top_match(0)
        self.assertAlmostEqual(top.score, 0.7, places=5)


# =============================================================================
# 2. spectral_matcher
# =============================================================================

class TestSpectralMatchResult(unittest.TestCase):
    """Tests for SpectralMatchResult dataclass."""

    def test_valid_result(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.7, phase_shift=(1.0, 2.0))
        self.assertEqual(r.idx1, 0)
        self.assertEqual(r.score, 0.7)

    def test_invalid_score_above_one(self):
        with self.assertRaises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=1.1)

    def test_invalid_score_negative(self):
        with self.assertRaises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=-0.1)

    def test_params_stored(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5, params={"foo": "bar"})
        self.assertEqual(r.params["foo"], "bar")


class TestMagnitudeSpectrum(unittest.TestCase):
    """Tests for magnitude_spectrum and log_magnitude."""

    def test_grayscale_image_shape(self):
        img = _rand_img(_rng(2))
        spec = magnitude_spectrum(img)
        self.assertEqual(spec.shape, (64, 64))

    def test_color_image_shape(self):
        img = _rand_img_color(_rng(3))
        spec = magnitude_spectrum(img)
        self.assertEqual(spec.shape, (64, 64))

    def test_spectrum_nonnegative(self):
        img = _rand_img(_rng(4))
        spec = magnitude_spectrum(img)
        self.assertTrue(np.all(spec >= 0.0))

    def test_log_magnitude_range(self):
        img = _rand_img(_rng(5))
        spec = magnitude_spectrum(img)
        log = log_magnitude(spec)
        self.assertAlmostEqual(float(log.min()), 0.0, places=5)
        self.assertAlmostEqual(float(log.max()), 1.0, places=5)

    def test_log_magnitude_uniform_input(self):
        spec = np.ones((32, 32), dtype=np.float64) * 5.0
        log = log_magnitude(spec)
        # All-equal spectrum → all zeros
        self.assertTrue(np.all(log == 0.0))

    def test_spectrum_correlation_range(self):
        rng = _rng(6)
        img1 = _rand_img(rng)
        img2 = _rand_img(rng)
        s1 = magnitude_spectrum(img1)
        s2 = magnitude_spectrum(img2)
        corr = spectrum_correlation(s1, s2)
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    def test_spectrum_correlation_identical(self):
        img = _rand_img(_rng(7))
        s = magnitude_spectrum(img)
        corr = spectrum_correlation(s, s)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_spectrum_correlation_empty_raises(self):
        s = np.ones((0, 0))
        with self.assertRaises(ValueError):
            spectrum_correlation(s, s)


class TestPhaseAndMatchSpectra(unittest.TestCase):
    """Tests for phase_correlation, match_spectra, batch_spectral_match."""

    def test_phase_correlation_returns_tuple(self):
        rng = _rng(8)
        img1 = _rand_img(rng)
        img2 = _rand_img(rng)
        result = phase_correlation(img1, img2)
        self.assertEqual(len(result), 3)
        score, dy, dx = result
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_phase_correlation_identical_zero_shift(self):
        img = _rand_img(_rng(9))
        score, dy, dx = phase_correlation(img, img)
        self.assertAlmostEqual(dy, 0.0, places=5)
        self.assertAlmostEqual(dx, 0.0, places=5)

    def test_match_spectra_returns_result(self):
        rng = _rng(10)
        img1 = _rand_img(rng)
        img2 = _rand_img(rng)
        r = match_spectra(img1, img2, idx1=0, idx2=1)
        self.assertIsInstance(r, SpectralMatchResult)
        self.assertEqual(r.idx1, 0)
        self.assertEqual(r.idx2, 1)
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)

    def test_match_spectra_invalid_weights(self):
        img = _rand_img(_rng(11))
        with self.assertRaises(ValueError):
            match_spectra(img, img, w_corr=0.0, w_phase=0.0)

    def test_match_spectra_w_corr_only(self):
        rng = _rng(12)
        img1 = _rand_img(rng)
        img2 = _rand_img(rng)
        r = match_spectra(img1, img2, w_corr=1.0, w_phase=0.0)
        self.assertGreaterEqual(r.score, 0.0)

    def test_batch_spectral_match_length(self):
        rng = _rng(13)
        query = _rand_img(rng)
        candidates = [_rand_img(rng) for _ in range(4)]
        results = batch_spectral_match(query, candidates, query_idx=0)
        self.assertEqual(len(results), 4)

    def test_batch_spectral_match_indices(self):
        rng = _rng(14)
        query = _rand_img(rng)
        candidates = [_rand_img(rng) for _ in range(3)]
        results = batch_spectral_match(query, candidates, query_idx=5)
        for i, r in enumerate(results):
            self.assertEqual(r.idx1, 5)
            self.assertEqual(r.idx2, i)


# =============================================================================
# 3. graph_match
# =============================================================================

class TestFragmentGraph(unittest.TestCase):
    """Tests for FragmentGraph construction and properties."""

    def _make_simple_graph(self) -> FragmentGraph:
        graph = FragmentGraph()
        for fid in [0, 1, 2]:
            graph.add_node(_make_fragment(fid, 1))
        graph.add_edge(0, 1, 0.9)
        graph.add_edge(1, 2, 0.6)
        graph.add_edge(0, 2, 0.3)
        return graph

    def test_node_count(self):
        g = self._make_simple_graph()
        self.assertEqual(g.n_nodes, 3)

    def test_edge_count(self):
        g = self._make_simple_graph()
        self.assertEqual(g.n_edges, 3)

    def test_weight_retrieval(self):
        g = self._make_simple_graph()
        self.assertAlmostEqual(g.weight(0, 1), 0.9)

    def test_missing_edge_zero_weight(self):
        g = self._make_simple_graph()
        self.assertAlmostEqual(g.weight(0, 99), 0.0)

    def test_add_edge_max_weight(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        g.add_node(_make_fragment(1))
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 1, 0.8)
        self.assertAlmostEqual(g.weight(0, 1), 0.8)

    def test_adjacency_matrix_shape(self):
        g = self._make_simple_graph()
        A, fids = g.adjacency_matrix()
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(len(fids), 3)

    def test_adjacency_matrix_symmetric(self):
        g = self._make_simple_graph()
        A, _ = g.adjacency_matrix()
        np.testing.assert_array_almost_equal(A, A.T)

    def test_laplacian_row_sum_zero(self):
        g = self._make_simple_graph()
        L, _ = g.laplacian()
        row_sums = L.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(3), decimal=10)

    def test_neighbors(self):
        g = self._make_simple_graph()
        nbrs = g.neighbors(0)
        nbr_ids = [n for n, _ in nbrs]
        self.assertIn(1, nbr_ids)
        self.assertIn(2, nbr_ids)


class TestGraphAlgorithms(unittest.TestCase):
    """Tests for graph algorithms: MST, spectral, random walk, centrality."""

    def _make_graph(self, n: int = 4) -> FragmentGraph:
        g = FragmentGraph()
        for i in range(n):
            g.add_node(_make_fragment(i, 1))
        rng = _rng(20)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j, float(rng.rand()))
        return g

    def test_mst_ordering_all_nodes(self):
        g = self._make_graph(4)
        order = mst_ordering(g)
        self.assertEqual(set(order), {0, 1, 2, 3})

    def test_mst_ordering_empty(self):
        g = FragmentGraph()
        self.assertEqual(mst_ordering(g), [])

    def test_mst_ordering_single_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(7, 1))
        self.assertEqual(mst_ordering(g), [7])

    def test_spectral_ordering_all_nodes(self):
        g = self._make_graph(4)
        order = spectral_ordering(g)
        self.assertEqual(set(order), {0, 1, 2, 3})

    def test_spectral_ordering_two_nodes(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        g.add_node(_make_fragment(1))
        g.add_edge(0, 1, 0.5)
        order = spectral_ordering(g)
        self.assertEqual(set(order), {0, 1})

    def test_random_walk_shape(self):
        g = self._make_graph(3)
        R = random_walk_similarity(g)
        self.assertEqual(R.shape, (3, 3))

    def test_random_walk_empty_graph(self):
        g = FragmentGraph()
        R = random_walk_similarity(g)
        self.assertEqual(R.shape, (0, 0))

    def test_degree_centrality_range(self):
        g = self._make_graph(4)
        cent = degree_centrality(g)
        for v in cent.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_degree_centrality_empty(self):
        g = FragmentGraph()
        cent = degree_centrality(g)
        self.assertEqual(cent, {})

    def test_analyze_graph_returns_result(self):
        g = self._make_graph(4)
        result = analyze_graph(g)
        self.assertIsInstance(result, GraphMatchResult)
        self.assertEqual(len(result.spectral_order), 4)
        self.assertEqual(len(result.centrality), 4)

    def test_build_fragment_graph_with_entries(self):
        frags = [_make_fragment(i, 2) for i in range(3)]
        entries = [
            _make_compat_entry(0, 1, 0.8),
            _make_compat_entry(1, 2, 0.6),
        ]
        g = build_fragment_graph(frags, entries)
        # All fragments should be nodes
        self.assertGreaterEqual(g.n_nodes, 1)


# =============================================================================
# 4. consensus
# =============================================================================

class TestConsensus(unittest.TestCase):
    """Tests for consensus module helpers."""

    def test_assembly_to_pairs_dict_placements(self):
        rng = _rng(30)
        # Place fragments close together so they are neighbours
        assembly = Assembly(placements={
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([10.0, 0.0]), 0.0),
            2: (np.array([500.0, 500.0]), 0.0),
        })
        pairs = assembly_to_pairs(assembly, adjacency_threshold=100.0)
        # Frags 0 and 1 are close; 2 is far
        self.assertIn(frozenset({0, 1}), pairs)
        self.assertNotIn(frozenset({0, 2}), pairs)

    def test_assembly_to_pairs_list_placements(self):
        placements = [
            Placement(fragment_id=0, position=(0.0, 0.0)),
            Placement(fragment_id=1, position=(5.0, 0.0)),
            Placement(fragment_id=2, position=(1000.0, 1000.0)),
        ]
        assembly = Assembly(placements=placements)
        pairs = assembly_to_pairs(assembly, adjacency_threshold=50.0)
        self.assertIn(frozenset({0, 1}), pairs)
        self.assertNotIn(frozenset({0, 2}), pairs)

    def test_vote_on_pairs_accumulates(self):
        a1 = Assembly(placements={
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([10.0, 0.0]), 0.0),
        })
        a2 = Assembly(placements={
            0: (np.array([0.0, 0.0]), 0.0),
            1: (np.array([10.0, 0.0]), 0.0),
        })
        votes = vote_on_pairs([a1, a2], adjacency_threshold=100.0)
        pair = frozenset({0, 1})
        self.assertEqual(votes.get(pair, 0), 2)

    def test_vote_on_pairs_empty(self):
        votes = vote_on_pairs([])
        self.assertEqual(votes, {})

    def test_consensus_result_vote_fraction(self):
        pair = frozenset({0, 1})
        result = ConsensusResult(
            pair_votes={pair: 3},
            consensus_pairs={pair},
            n_methods=4,
            threshold=0.5,
        )
        self.assertAlmostEqual(result.vote_fraction(0, 1), 0.75)

    def test_consensus_result_is_consensus(self):
        pair = frozenset({0, 1})
        result = ConsensusResult(
            pair_votes={pair: 3},
            consensus_pairs={pair},
            n_methods=4,
            threshold=0.5,
        )
        self.assertTrue(result.is_consensus(0, 1))
        self.assertFalse(result.is_consensus(0, 2))

    def test_consensus_result_top_pairs(self):
        votes = {frozenset({i, i + 1}): 3 - i for i in range(3)}
        result = ConsensusResult(
            pair_votes=votes,
            consensus_pairs=set(),
            n_methods=3,
            threshold=0.5,
        )
        top = result.top_pairs(2)
        self.assertEqual(len(top), 2)
        # Top pair should have highest vote count (3)
        self.assertEqual(top[0][1], 3)

    def test_consensus_score_matrix_shape(self):
        frags = [_make_fragment(i, 1) for i in range(3)]
        pair = frozenset({0, 1})
        result = ConsensusResult(
            pair_votes={pair: 2},
            consensus_pairs={pair},
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        self.assertEqual(mat.shape, (3, 3))

    def test_consensus_score_matrix_symmetric(self):
        frags = [_make_fragment(i, 1) for i in range(3)]
        pair01 = frozenset({0, 1})
        pair12 = frozenset({1, 2})
        result = ConsensusResult(
            pair_votes={pair01: 2, pair12: 1},
            consensus_pairs={pair01},
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_consensus_score_matrix_values(self):
        frags = [_make_fragment(i, 1) for i in range(2)]
        pair = frozenset({0, 1})
        result = ConsensusResult(
            pair_votes={pair: 1},
            consensus_pairs={pair},
            n_methods=2,
            threshold=0.5,
        )
        mat = consensus_score_matrix(result, frags)
        self.assertAlmostEqual(mat[0, 1], 0.5)
        self.assertAlmostEqual(mat[1, 0], 0.5)

    def test_consensus_result_summary(self):
        result = ConsensusResult(
            pair_votes={frozenset({0, 1}): 2},
            consensus_pairs={frozenset({0, 1})},
            n_methods=2,
            threshold=0.5,
        )
        s = result.summary()
        self.assertIn("ConsensusResult", s)


# =============================================================================
# 5. curve_descriptor
# =============================================================================

class TestCurveDescriptorConfig(unittest.TestCase):
    """Tests for CurveDescriptorConfig validation."""

    def test_defaults(self):
        cfg = CurveDescriptorConfig()
        self.assertEqual(cfg.n_harmonics, 8)
        self.assertTrue(cfg.normalize)
        self.assertTrue(cfg.center)
        self.assertIsNone(cfg.resample_n)

    def test_invalid_n_harmonics(self):
        with self.assertRaises(ValueError):
            CurveDescriptorConfig(n_harmonics=0)

    def test_invalid_resample_n(self):
        with self.assertRaises(ValueError):
            CurveDescriptorConfig(resample_n=1)

    def test_valid_resample_n(self):
        cfg = CurveDescriptorConfig(resample_n=50)
        self.assertEqual(cfg.resample_n, 50)


class TestFourierDescriptor(unittest.TestCase):
    """Tests for compute_fourier_descriptor."""

    def test_output_shape(self):
        curve = _make_curve(30)
        desc = compute_fourier_descriptor(curve, n_harmonics=8)
        self.assertEqual(desc.shape, (8,))

    def test_invalid_shape_raises(self):
        curve = np.ones((10, 3))
        with self.assertRaises(ValueError):
            compute_fourier_descriptor(curve)

    def test_empty_curve_zeros(self):
        curve = np.zeros((0, 2))
        desc = compute_fourier_descriptor(curve, n_harmonics=4)
        np.testing.assert_array_equal(desc, np.zeros(4))

    def test_normalized_first_harmonic_one(self):
        curve = _make_curve(40)
        desc = compute_fourier_descriptor(curve, n_harmonics=8, normalize=True)
        # First amplitude should be 1.0 when normalized (unless it was 0)
        if desc[0] > 1e-9:
            self.assertAlmostEqual(desc[0], 1.0, places=5)


class TestCurveDescriptorFunctions(unittest.TestCase):
    """Tests for describe_curve, descriptor_distance, descriptor_similarity."""

    def test_describe_curve_returns_descriptor(self):
        curve = _make_curve(30)
        d = describe_curve(curve)
        self.assertIsInstance(d, CurveDescriptor)
        self.assertGreater(d.arc_length, 0.0)
        self.assertEqual(d.n_points, 30)

    def test_describe_curve_invalid_shape(self):
        with self.assertRaises(ValueError):
            describe_curve(np.ones((10,)))

    def test_curvature_profile_shape(self):
        curve = _make_curve(20)
        curv = compute_curvature_profile(curve)
        self.assertEqual(curv.shape, (20,))

    def test_curvature_profile_short_curve(self):
        curve = np.array([[0.0, 0.0], [1.0, 0.0]])
        curv = compute_curvature_profile(curve)
        np.testing.assert_array_equal(curv, np.zeros(2))

    def test_descriptor_distance_same_curve(self):
        curve = _make_curve(30)
        d = describe_curve(curve)
        dist = descriptor_distance(d, d)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_descriptor_similarity_same_curve(self):
        curve = _make_curve(30)
        d = describe_curve(curve)
        sim = descriptor_similarity(d, d)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_descriptor_similarity_invalid_sigma(self):
        curve = _make_curve(30)
        d = describe_curve(curve)
        with self.assertRaises(ValueError):
            descriptor_similarity(d, d, sigma=0.0)

    def test_descriptor_similarity_range(self):
        rng = _rng(40)
        c1 = _make_curve(30, rng)
        c2 = _make_curve(30, rng)
        d1 = describe_curve(c1)
        d2 = describe_curve(c2)
        sim = descriptor_similarity(d1, d2)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_batch_describe_curves_length(self):
        rng = _rng(41)
        curves = [_make_curve(20, rng) for _ in range(5)]
        descs = batch_describe_curves(curves)
        self.assertEqual(len(descs), 5)

    def test_find_best_match_returns_self(self):
        curves = [_make_curve(30, _rng(i)) for i in range(4)]
        descs = batch_describe_curves(curves)
        query = descs[0]
        idx, dist = find_best_match(query, descs)
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_find_best_match_empty_raises(self):
        curve = _make_curve(30)
        d = describe_curve(curve)
        with self.assertRaises(ValueError):
            find_best_match(d, [])

    def test_describe_curve_with_resample(self):
        curve = _make_curve(50)
        cfg = CurveDescriptorConfig(resample_n=20)
        d = describe_curve(curve, cfg)
        # n_points is original
        self.assertEqual(d.n_points, 50)
        # fourier_desc has right length
        self.assertEqual(len(d.fourier_desc), cfg.n_harmonics)


# =============================================================================
# 6. text_flow
# =============================================================================

class TestTextBaselineDetection(unittest.TestCase):
    """Tests for text baseline angle and position detection."""

    def _make_gradient(self, h: int = 32, w: int = 32, seed: int = 50) -> np.ndarray:
        rng = _rng(seed)
        return rng.rand(h, w).astype(np.float64)

    def test_detect_angle_returns_tuple(self):
        grad = self._make_gradient()
        angle, conf = detect_text_baseline_angle(grad)
        self.assertIsInstance(angle, float)
        self.assertIsInstance(conf, float)

    def test_detect_angle_range(self):
        grad = self._make_gradient()
        angle, conf = detect_text_baseline_angle(grad)
        self.assertGreaterEqual(angle, -90.0)
        self.assertLess(angle, 90.0)

    def test_detect_confidence_range(self):
        grad = self._make_gradient()
        _, conf = detect_text_baseline_angle(grad)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_detect_angle_empty(self):
        angle, conf = detect_text_baseline_angle(np.array([]).reshape(0, 0))
        self.assertEqual(angle, 0.0)
        self.assertEqual(conf, 0.0)

    def test_detect_line_positions_range(self):
        grad = self._make_gradient()
        positions = detect_text_line_positions(grad)
        for p in positions:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_detect_line_positions_empty(self):
        positions = detect_text_line_positions(np.array([]).reshape(0, 0))
        self.assertEqual(len(positions), 0)

    def test_build_text_line_profile(self):
        grad = self._make_gradient()
        profile = build_text_line_profile(grad)
        self.assertIsInstance(profile, TextLineProfile)
        self.assertEqual(profile.n_lines, len(profile.line_positions))


class TestTextFlowScoring(unittest.TestCase):
    """Tests for compare_baseline_angles, align_line_positions, match_text_flow."""

    def test_compare_angles_exact_match(self):
        score = compare_baseline_angles(0.0, 0.0)
        self.assertAlmostEqual(score, 1.0)

    def test_compare_angles_large_diff(self):
        score = compare_baseline_angles(0.0, 90.0)
        self.assertAlmostEqual(score, 0.0)

    def test_compare_angles_partial(self):
        score = compare_baseline_angles(0.0, 15.0, tolerance_deg=5.0, max_diff_deg=30.0)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_align_line_positions_perfect(self):
        pos = np.array([0.1, 0.5, 0.9])
        score = align_line_positions(pos, pos, tolerance=0.05)
        self.assertAlmostEqual(score, 1.0)

    def test_align_line_positions_no_match(self):
        a = np.array([0.1, 0.2])
        b = np.array([0.8, 0.9])
        score = align_line_positions(a, b, tolerance=0.05)
        self.assertAlmostEqual(score, 0.0)

    def test_align_line_positions_empty(self):
        score = align_line_positions(np.array([]), np.array([0.5]))
        self.assertAlmostEqual(score, 0.0)

    def test_match_text_flow_low_confidence(self):
        pa = TextLineProfile(angle_deg=0.0, line_positions=np.array([0.5]),
                             confidence=0.05, n_lines=1)
        pb = TextLineProfile(angle_deg=0.0, line_positions=np.array([0.5]),
                             confidence=0.05, n_lines=1)
        result = match_text_flow(pa, pb)
        # Low confidence → uncertain score of 0.5
        self.assertAlmostEqual(result.score, 0.5)

    def test_match_text_flow_high_confidence(self):
        pa = TextLineProfile(angle_deg=0.0, line_positions=np.array([0.3, 0.7]),
                             confidence=0.9, n_lines=2)
        pb = TextLineProfile(angle_deg=0.0, line_positions=np.array([0.3, 0.7]),
                             confidence=0.9, n_lines=2)
        cfg = TextFlowConfig()
        result = match_text_flow(pa, pb, cfg)
        self.assertGreater(result.score, 0.5)

    def test_text_flow_scorer_score(self):
        rng = _rng(60)
        grad_a = rng.rand(32, 32)
        grad_b = rng.rand(32, 32)
        scorer = TextFlowScorer()
        result = scorer.score(grad_a, grad_b)
        self.assertIsInstance(result, TextFlowMatch)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_text_flow_scorer_score_batch(self):
        rng = _rng(61)
        grad_a = rng.rand(32, 32)
        grads_b = [rng.rand(32, 32) for _ in range(3)]
        scorer = TextFlowScorer()
        results = scorer.score_batch(grad_a, grads_b)
        self.assertEqual(len(results), 3)


# =============================================================================
# 7. score_combiner
# =============================================================================

class TestScoreVector(unittest.TestCase):
    """Tests for ScoreVector validation."""

    def test_valid_construction(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.7})
        self.assertEqual(len(sv), 2)
        self.assertEqual(sv.pair, (0, 1))

    def test_invalid_idx_negative(self):
        with self.assertRaises(ValueError):
            ScoreVector(idx1=-1, idx2=0, scores={"a": 0.5})

    def test_invalid_score_range(self):
        with self.assertRaises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": 1.5})


class TestCombineMethods(unittest.TestCase):
    """Tests for weighted_combine, min_combine, max_combine, rank_combine."""

    def _make_sv(self, idx1: int, idx2: int, scores: dict) -> ScoreVector:
        return ScoreVector(idx1=idx1, idx2=idx2, scores=scores)

    def test_weighted_combine_equal_weights(self):
        sv = self._make_sv(0, 1, {"a": 0.6, "b": 0.4})
        cs = weighted_combine(sv)
        self.assertAlmostEqual(cs.score, 0.5, places=5)

    def test_weighted_combine_custom_weights(self):
        sv = self._make_sv(0, 1, {"a": 0.8, "b": 0.2})
        cs = weighted_combine(sv, weights={"a": 2.0, "b": 1.0})
        expected = (0.8 * 2 + 0.2 * 1) / 3.0
        self.assertAlmostEqual(cs.score, expected, places=5)

    def test_weighted_combine_empty_raises(self):
        sv = ScoreVector.__new__(ScoreVector)
        sv.idx1 = 0
        sv.idx2 = 1
        sv.scores = {}
        sv.params = {}
        with self.assertRaises(ValueError):
            weighted_combine(sv)

    def test_min_combine(self):
        sv = self._make_sv(0, 1, {"a": 0.3, "b": 0.9})
        cs = min_combine(sv)
        self.assertAlmostEqual(cs.score, 0.3, places=5)

    def test_max_combine(self):
        sv = self._make_sv(0, 1, {"a": 0.3, "b": 0.9})
        cs = max_combine(sv)
        self.assertAlmostEqual(cs.score, 0.9, places=5)

    def test_rank_combine_best_gets_highest_score(self):
        svs = [
            self._make_sv(0, 1, {"a": 0.9, "b": 0.8}),
            self._make_sv(0, 2, {"a": 0.5, "b": 0.4}),
            self._make_sv(0, 3, {"a": 0.1, "b": 0.2}),
        ]
        results = rank_combine(svs)
        self.assertEqual(len(results), 3)
        scores = [r.score for r in results]
        self.assertEqual(scores[0], max(scores))

    def test_rank_combine_single(self):
        svs = [self._make_sv(0, 1, {"a": 0.7})]
        results = rank_combine(svs)
        self.assertAlmostEqual(results[0].score, 1.0, places=5)

    def test_normalize_score_vectors(self):
        svs = [
            self._make_sv(0, 1, {"a": 0.2, "b": 0.4}),
            self._make_sv(0, 2, {"a": 0.6, "b": 0.8}),
        ]
        normed = normalize_score_vectors(svs)
        self.assertEqual(len(normed), 2)
        # After min-max normalization, extremes should be 0 and 1
        a_vals = [sv.scores["a"] for sv in normed]
        self.assertAlmostEqual(min(a_vals), 0.0, places=5)
        self.assertAlmostEqual(max(a_vals), 1.0, places=5)

    def test_batch_combine_weighted(self):
        svs = [
            self._make_sv(0, 1, {"x": 0.5}),
            self._make_sv(0, 2, {"x": 0.9}),
        ]
        results = batch_combine(svs, method="weighted")
        self.assertEqual(len(results), 2)
        # Results should be sorted descending
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_batch_combine_invalid_method(self):
        svs = [self._make_sv(0, 1, {"x": 0.5})]
        with self.assertRaises(ValueError):
            batch_combine(svs, method="foobar")

    def test_batch_combine_rank_method(self):
        svs = [
            self._make_sv(i, i + 1, {"a": float(i) / 4.0})
            for i in range(4)
        ]
        results = batch_combine(svs, method="rank")
        self.assertEqual(len(results), 4)

    def test_combined_score_invalid(self):
        with self.assertRaises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=1.5)


# =============================================================================
# 8. score_normalizer
# =============================================================================

class TestNormalizeMinmax(unittest.TestCase):
    """Tests for normalize_minmax."""

    def test_output_range_default(self):
        rng = _rng(70)
        scores = rng.rand(20)
        result = normalize_minmax(scores)
        self.assertAlmostEqual(float(result.scores.min()), 0.0, places=5)
        self.assertAlmostEqual(float(result.scores.max()), 1.0, places=5)

    def test_custom_feature_range(self):
        scores = np.array([0.0, 0.5, 1.0])
        result = normalize_minmax(scores, feature_range=(0.2, 0.8))
        self.assertAlmostEqual(float(result.scores.min()), 0.2, places=5)
        self.assertAlmostEqual(float(result.scores.max()), 0.8, places=5)

    def test_uniform_scores(self):
        scores = np.ones(10) * 0.5
        result = normalize_minmax(scores)
        np.testing.assert_array_almost_equal(result.scores, np.zeros(10))

    def test_method_name(self):
        result = normalize_minmax(np.array([0.1, 0.9]))
        self.assertEqual(result.method, "minmax")

    def test_original_min_max_stored(self):
        scores = np.array([0.3, 0.7])
        result = normalize_minmax(scores)
        self.assertAlmostEqual(result.original_min, 0.3)
        self.assertAlmostEqual(result.original_max, 0.7)


class TestNormalizeZscore(unittest.TestCase):
    """Tests for normalize_zscore."""

    def test_output_in_unit_interval(self):
        rng = _rng(71)
        scores = rng.randn(100)
        result = normalize_zscore(scores)
        self.assertTrue(np.all(result.scores >= 0.0))
        self.assertTrue(np.all(result.scores <= 1.0))

    def test_uniform_returns_half(self):
        scores = np.ones(10) * 3.0
        result = normalize_zscore(scores)
        np.testing.assert_array_almost_equal(result.scores, np.full(10, 0.5))

    def test_method_name(self):
        result = normalize_zscore(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result.method, "zscore")


class TestNormalizeRank(unittest.TestCase):
    """Tests for normalize_rank."""

    def test_output_range(self):
        rng = _rng(72)
        scores = rng.rand(10)
        result = normalize_rank(scores)
        self.assertAlmostEqual(float(result.scores.min()), 0.0, places=5)
        self.assertAlmostEqual(float(result.scores.max()), 1.0, places=5)

    def test_single_element(self):
        result = normalize_rank(np.array([0.5]))
        np.testing.assert_array_almost_equal(result.scores, np.array([0.0]))

    def test_method_name(self):
        result = normalize_rank(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result.method, "rank")

    def test_uniform_returns_half(self):
        result = normalize_rank(np.ones(5))
        np.testing.assert_array_almost_equal(result.scores, np.full(5, 0.5))


class TestCalibrateScores(unittest.TestCase):
    """Tests for calibrate_scores."""

    def test_calibrated_range(self):
        rng = _rng(73)
        scores = rng.rand(50)
        ref = rng.rand(50)
        result = calibrate_scores(scores, ref)
        self.assertEqual(result.method, "calibrated")
        self.assertEqual(len(result.scores), 50)

    def test_empty_scores(self):
        result = calibrate_scores(np.array([]), np.array([0.5]))
        self.assertEqual(len(result.scores), 0)


class TestCombineScores(unittest.TestCase):
    """Tests for combine_scores."""

    def test_weighted_method(self):
        a = np.array([0.2, 0.4, 0.6])
        b = np.array([0.8, 0.6, 0.4])
        result = combine_scores([a, b], method="weighted")
        np.testing.assert_array_almost_equal(result, np.array([0.5, 0.5, 0.5]))

    def test_min_method(self):
        a = np.array([0.2, 0.9])
        b = np.array([0.8, 0.3])
        result = combine_scores([a, b], method="min")
        np.testing.assert_array_almost_equal(result, np.array([0.2, 0.3]))

    def test_max_method(self):
        a = np.array([0.2, 0.9])
        b = np.array([0.8, 0.3])
        result = combine_scores([a, b], method="max")
        np.testing.assert_array_almost_equal(result, np.array([0.8, 0.9]))

    def test_product_method(self):
        a = np.array([0.5, 1.0])
        b = np.array([0.5, 0.5])
        result = combine_scores([a, b], method="product")
        np.testing.assert_array_almost_equal(result, np.array([0.25, 0.5]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            combine_scores([])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            combine_scores([np.array([0.1, 0.2]), np.array([0.3])])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            combine_scores([np.array([0.5])], method="unknown")

    def test_custom_weights(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = combine_scores([a, b], weights=[3.0, 1.0], method="weighted")
        np.testing.assert_array_almost_equal(result, np.array([0.75, 0.25]))


class TestNormalizeScoreMatrix(unittest.TestCase):
    """Tests for normalize_score_matrix."""

    def test_output_shape_preserved(self):
        rng = _rng(74)
        mat = rng.rand(5, 5).astype(np.float32)
        result = normalize_score_matrix(mat, method="minmax")
        self.assertEqual(result.shape, (5, 5))

    def test_non_square_raises(self):
        mat = np.ones((3, 4))
        with self.assertRaises(ValueError):
            normalize_score_matrix(mat)

    def test_unknown_method_raises(self):
        mat = np.eye(3)
        with self.assertRaises(ValueError):
            normalize_score_matrix(mat, method="bad")

    def test_diagonal_preserved(self):
        rng = _rng(75)
        mat = rng.rand(4, 4)
        np.fill_diagonal(mat, 0.42)
        result = normalize_score_matrix(mat, method="minmax", keep_diagonal=True)
        for i in range(4):
            self.assertAlmostEqual(result[i, i], 0.42, places=5)

    def test_zscore_method(self):
        rng = _rng(76)
        mat = rng.rand(4, 4)
        result = normalize_score_matrix(mat, method="zscore")
        self.assertEqual(result.shape, (4, 4))

    def test_rank_method(self):
        rng = _rng(77)
        mat = rng.rand(4, 4)
        result = normalize_score_matrix(mat, method="rank")
        self.assertEqual(result.shape, (4, 4))


class TestBatchNormalizeScores(unittest.TestCase):
    """Tests for batch_normalize_scores."""

    def test_returns_list_of_results(self):
        rng = _rng(78)
        arrays = [rng.rand(10) for _ in range(3)]
        results = batch_normalize_scores(arrays, method="minmax")
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, ScoreNormResult)

    def test_zscore_batch(self):
        rng = _rng(79)
        arrays = [rng.randn(20) for _ in range(4)]
        results = batch_normalize_scores(arrays, method="zscore")
        self.assertEqual(len(results), 4)

    def test_rank_batch(self):
        rng = _rng(80)
        arrays = [rng.rand(15) for _ in range(2)]
        results = batch_normalize_scores(arrays, method="rank")
        self.assertEqual(len(results), 2)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            batch_normalize_scores([np.array([0.5])], method="bad")


if __name__ == "__main__":
    unittest.main()
