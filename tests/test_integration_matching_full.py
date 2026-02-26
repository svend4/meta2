"""
Интеграционные тесты: полный цикл матчинга.
compat_matrix → threshold → consensus → global_matcher
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
pytestmark = pytest.mark.integration

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.models import Fragment, CompatEntry
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import compute_fractal_signature, build_edge_signatures
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix, top_candidates
from puzzle_reconstruction.matching.consensus import (
    build_consensus, vote_on_pairs, assembly_to_pairs, consensus_score_matrix,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.config import Config, MatchingConfig
from puzzle_reconstruction.scoring.score_normalizer import normalize_score_matrix
from puzzle_reconstruction.scoring.threshold_selector import select_threshold, ThresholdConfig
from puzzle_reconstruction.scoring.consistency_checker import run_consistency_check


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc():
    return generate_test_document(width=300, height=400, seed=3)


@pytest.fixture(scope="module")
def images_4(doc):
    return tear_document(doc, n_pieces=4, noise_level=0.3, seed=7)


@pytest.fixture(scope="module")
def fragments_4(images_4):
    frags = []
    for i, img in enumerate(images_4):
        try:
            mask = segment_fragment(img)
            cont = extract_contour(mask)
            frag = Fragment(fragment_id=i, image=img, mask=mask, contour=cont)
            frag.tangram = fit_tangram(cont)
            frag.fractal = compute_fractal_signature(cont)
            frag.edges = build_edge_signatures(frag)
            frags.append(frag)
        except Exception:
            pass
    return frags


@pytest.fixture(scope="module")
def compat_4(fragments_4):
    matrix, entries = build_compat_matrix(fragments_4, threshold=0.0)
    return matrix, entries


# ---------------------------------------------------------------------------
# Helper: build edge-to-fragment map using object identity
# ---------------------------------------------------------------------------

def _edge_to_frag_map(fragments):
    mapping = {}
    for frag in fragments:
        for edge in frag.edges:
            mapping[id(edge)] = frag.fragment_id
    return mapping


# ---------------------------------------------------------------------------
# class TestCompatMatrixProperties
# ---------------------------------------------------------------------------

class TestCompatMatrixProperties:
    def test_returns_tuple_of_two(self, compat_4):
        assert isinstance(compat_4, tuple)
        assert len(compat_4) == 2

    def test_matrix_is_ndarray(self, compat_4):
        matrix, _ = compat_4
        assert isinstance(matrix, np.ndarray)

    def test_matrix_shape_equals_total_edges(self, compat_4, fragments_4):
        matrix, _ = compat_4
        total_edges = sum(len(f.edges) for f in fragments_4)
        assert matrix.shape == (total_edges, total_edges)

    def test_matrix_symmetric(self, compat_4):
        matrix, _ = compat_4
        assert np.allclose(matrix, matrix.T, atol=1e-6)

    def test_matrix_values_nonnegative(self, compat_4):
        matrix, _ = compat_4
        assert (matrix >= 0).all()

    def test_matrix_values_at_most_one(self, compat_4):
        matrix, _ = compat_4
        # Allow tiny floating-point overshoot
        assert (matrix <= 1.0 + 1e-6).all()

    def test_diagonal_is_zero(self, compat_4):
        matrix, _ = compat_4
        diag = np.diag(matrix)
        assert np.allclose(diag, 0.0, atol=1e-6)

    def test_entries_sorted_by_score_desc(self, compat_4):
        _, entries = compat_4
        if len(entries) < 2:
            pytest.skip("Too few entries to check ordering")
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_entries_have_compat_entry_type(self, compat_4):
        _, entries = compat_4
        assert entries, "Expected at least one CompatEntry"
        for e in entries:
            assert isinstance(e, CompatEntry)

    def test_no_nan_in_matrix(self, compat_4):
        matrix, _ = compat_4
        assert not np.any(np.isnan(matrix))

    def test_no_inf_in_matrix(self, compat_4):
        matrix, _ = compat_4
        assert not np.any(np.isinf(matrix))

    def test_threshold_filters_entries(self, fragments_4):
        _, entries_none = build_compat_matrix(fragments_4, threshold=0.0)
        _, entries_half = build_compat_matrix(fragments_4, threshold=0.5)
        assert len(entries_half) < len(entries_none)

    def test_top_candidates_returns_list(self, compat_4, fragments_4):
        matrix, _ = compat_4
        all_edges = [e for f in fragments_4 for e in f.edges]
        result = top_candidates(matrix, all_edges, edge_idx=0, k=3)
        assert isinstance(result, list)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# class TestScoreNormalization
# ---------------------------------------------------------------------------

class TestScoreNormalization:
    def test_normalize_returns_normalized_result(self, compat_4):
        matrix, _ = compat_4
        result = normalize_score_matrix(matrix)
        assert result is not None

    def test_normalized_values_in_range(self, compat_4):
        matrix, _ = compat_4
        result = normalize_score_matrix(matrix)
        data = result.data
        assert float(data.min()) >= -1e-6
        assert float(data.max()) <= 1.0 + 1e-6

    def test_normalization_preserves_shape(self, compat_4):
        matrix, _ = compat_4
        result = normalize_score_matrix(matrix)
        assert result.data.shape == matrix.shape


# ---------------------------------------------------------------------------
# class TestThresholdSelection
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    @pytest.fixture(scope="class")
    def scores(self, compat_4):
        matrix, _ = compat_4
        # Flatten upper triangle, excluding diagonal
        n = matrix.shape[0]
        idx = np.triu_indices(n, k=1)
        return matrix[idx].astype(np.float32)

    def test_select_threshold_returns_float(self, scores):
        result = select_threshold(scores)
        assert isinstance(result.threshold, float)

    def test_threshold_in_reasonable_range(self, scores):
        result = select_threshold(scores)
        assert 0.0 <= result.threshold <= 1.0

    def test_threshold_otsu_method(self, scores):
        cfg = ThresholdConfig(method="otsu")
        result = select_threshold(scores, cfg)
        assert result.method == "otsu"
        assert 0.0 <= result.threshold <= 1.0

    def test_threshold_percentile_method(self, scores):
        cfg = ThresholdConfig(method="percentile", percentile=75)
        result = select_threshold(scores, cfg)
        assert result.method == "percentile"
        assert 0.0 <= result.threshold <= 1.0


# ---------------------------------------------------------------------------
# class TestConsistencyCheck
# ---------------------------------------------------------------------------

class TestConsistencyCheck:
    @pytest.fixture(scope="class")
    def report(self, fragments_4):
        fids = [f.fragment_id for f in fragments_4]
        positions = [(i * 150, 0) for i in range(len(fids))]
        sizes = [(140, 190)] * len(fids)
        return run_consistency_check(
            fragment_ids=fids,
            expected_ids=fids,
            positions=positions,
            sizes=sizes,
            canvas_w=600,
            canvas_h=400,
        )

    def test_returns_consistency_report(self, report):
        assert report is not None

    def test_report_has_is_consistent_attribute(self, report):
        assert hasattr(report, "is_consistent")
        assert isinstance(report.is_consistent, bool)

    def test_report_has_n_errors(self, report):
        assert hasattr(report, "n_errors")
        assert isinstance(report.n_errors, int)
        assert report.n_errors >= 0

    def test_report_has_n_warnings(self, report):
        assert hasattr(report, "n_warnings")
        assert isinstance(report.n_warnings, int)
        assert report.n_warnings >= 0

    def test_empty_assembly_not_crash(self):
        from puzzle_reconstruction.models import Assembly
        report = run_consistency_check(
            fragment_ids=[],
            expected_ids=[],
            positions=[],
            sizes=[],
            canvas_w=300,
            canvas_h=400,
        )
        assert report is not None
        assert hasattr(report, "is_consistent")


# ---------------------------------------------------------------------------
# class TestConsensusVoting
# ---------------------------------------------------------------------------

class TestConsensusVoting:
    @pytest.fixture(scope="class")
    def assemblies(self, fragments_4, compat_4):
        _, entries = compat_4
        a1 = greedy_assembly(fragments_4, entries)
        a2 = beam_search(fragments_4, entries, beam_width=3)
        return [a1, a2]

    @pytest.fixture(scope="class")
    def consensus_result(self, assemblies, fragments_4, compat_4):
        _, entries = compat_4
        return build_consensus(
            assemblies=assemblies,
            fragments=fragments_4,
            entries=entries,
            threshold=0.5,
        )

    def test_build_consensus_returns_result(self, consensus_result):
        from puzzle_reconstruction.matching.consensus import ConsensusResult
        assert isinstance(consensus_result, ConsensusResult)

    def test_consensus_result_has_pair_votes(self, consensus_result):
        assert hasattr(consensus_result, "pair_votes")
        assert isinstance(consensus_result.pair_votes, dict)

    def test_consensus_result_has_n_methods(self, consensus_result):
        assert hasattr(consensus_result, "n_methods")
        assert consensus_result.n_methods == 2

    def test_vote_fraction_in_range(self, consensus_result, fragments_4):
        if len(fragments_4) < 2:
            pytest.skip("Need at least 2 fragments")
        fid_a = fragments_4[0].fragment_id
        fid_b = fragments_4[1].fragment_id
        frac = consensus_result.vote_fraction(fid_a, fid_b)
        assert 0.0 <= frac <= 1.0

    def test_consensus_score_matrix_shape(self, consensus_result, fragments_4):
        csm = consensus_score_matrix(consensus_result, fragments_4)
        n = len(fragments_4)
        assert csm.shape == (n, n)

    def test_consensus_score_matrix_symmetric(self, consensus_result, fragments_4):
        csm = consensus_score_matrix(consensus_result, fragments_4)
        assert np.allclose(csm, csm.T, atol=1e-9)

    def test_consensus_score_matrix_values_in_01(self, consensus_result, fragments_4):
        csm = consensus_score_matrix(consensus_result, fragments_4)
        assert (csm >= 0.0).all()
        assert (csm <= 1.0 + 1e-9).all()

    def test_assembly_to_pairs_returns_set(self, assemblies):
        pairs = assembly_to_pairs(assemblies[0])
        assert isinstance(pairs, set)

    def test_vote_on_pairs_returns_dict(self, assemblies):
        votes = vote_on_pairs(assemblies)
        assert isinstance(votes, dict)
        for pair, count in votes.items():
            assert isinstance(count, int)
            assert count >= 1

    def test_consensus_top_pairs_sorted_desc(self, consensus_result):
        top = consensus_result.top_pairs(n=10)
        if len(top) < 2:
            pytest.skip("Too few pairs for ordering check")
        vote_counts = [v for _, v in top]
        assert vote_counts == sorted(vote_counts, reverse=True)


# ---------------------------------------------------------------------------
# class TestMatcherCombination
# ---------------------------------------------------------------------------

class TestMatcherCombination:
    def test_all_default_matchers_produce_scores(self, compat_4):
        _, entries = compat_4
        assert entries, "Expected at least one CompatEntry"
        for e in entries:
            assert math.isfinite(e.css_sim), f"css_sim not finite: {e.css_sim}"
            assert math.isfinite(e.dtw_dist), f"dtw_dist not finite: {e.dtw_dist}"
            assert math.isfinite(e.fd_diff), f"fd_diff not finite: {e.fd_diff}"

    def test_weighted_combination_in_range(self, compat_4):
        _, entries = compat_4
        for e in entries:
            assert 0.0 <= e.score <= 1.0, (
                f"CompatEntry score out of range [0,1]: {e.score}"
            )

    def test_no_same_fragment_pairs(self, compat_4, fragments_4):
        _, entries = compat_4
        edge_to_frag = _edge_to_frag_map(fragments_4)
        same_frag_entries = [
            e for e in entries
            if edge_to_frag.get(id(e.edge_i)) == edge_to_frag.get(id(e.edge_j))
        ]
        assert len(same_frag_entries) == 0, (
            f"Found {len(same_frag_entries)} entries where both edges belong "
            "to the same fragment"
        )
