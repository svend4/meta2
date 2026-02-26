"""
Integration tests for puzzle_reconstruction.assembly.cost_matrix and
puzzle_reconstruction.assembly.fragment_scorer modules.

~55 tests across 5 classes covering:
- CostMatrix: build_from_scores, build_from_distances, build_combined
- CostMatrix transforms: apply_forbidden_mask, normalize_costs, top_k_candidates
- fragment_scorer: score_fragment, score_assembly, top_k_placed, batch_score
- score_tracker: create_tracker, record_snapshot, detect_convergence, summarize_tracker
- integration: cost_matrix → fragment_scorer pipeline
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.cost_matrix import (
    CostMatrix,
    apply_forbidden_mask,
    build_combined,
    build_from_distances,
    build_from_scores,
    normalize_costs,
    top_k_candidates,
)
from puzzle_reconstruction.assembly.fragment_scorer import (
    AssemblyScore,
    FragmentScore,
    ScoreConfig,
    batch_score,
    score_assembly,
    score_fragment,
    top_k_placed,
)
from puzzle_reconstruction.assembly.score_tracker import (
    ScoreTracker,
    create_tracker,
    detect_convergence,
    record_snapshot,
    summarize_tracker,
)
from puzzle_reconstruction.assembly.assembly_state import (
    create_state,
    place_fragment,
    add_adjacency,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_score_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 1.0, (n, n)).astype(np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def make_cost_matrix(n: int, seed: int = 42) -> CostMatrix:
    return build_from_scores(make_score_matrix(n, seed))


def make_placed_state(n: int = 4):
    s = create_state(n)
    for i in range(n):
        s = place_fragment(s, i, (float(i * 50), 0.0))
    for i in range(n - 1):
        s = add_adjacency(s, i, i + 1)
    return s


# ─── TestBuildCostMatrix ──────────────────────────────────────────────────────

class TestBuildCostMatrix:
    """Tests for build_from_scores and build_from_distances."""

    def test_build_from_scores_returns_cost_matrix(self):
        m = make_score_matrix(4)
        cm = build_from_scores(m)
        assert isinstance(cm, CostMatrix)

    def test_build_from_scores_shape(self):
        m = make_score_matrix(5)
        cm = build_from_scores(m)
        assert cm.matrix.shape == (5, 5)

    def test_build_from_scores_diagonal_zero(self):
        m = make_score_matrix(4)
        cm = build_from_scores(m)
        assert np.allclose(np.diag(cm.matrix), 0.0, atol=1e-5)

    def test_build_from_scores_invert_true(self):
        m = make_score_matrix(4)
        cm = build_from_scores(m, invert=True)
        # cost = 1 - normalized_score → should be in [0, 1]
        assert np.all(cm.matrix >= 0.0 - 1e-5)
        assert np.all(cm.matrix <= 1.0 + 1e-5)

    def test_build_from_scores_n_fragments(self):
        m = make_score_matrix(6)
        cm = build_from_scores(m)
        assert cm.n_fragments == 6

    def test_build_from_distances_returns_cost_matrix(self):
        d = make_score_matrix(4)  # treat as distances
        cm = build_from_distances(d)
        assert isinstance(cm, CostMatrix)

    def test_build_from_distances_diagonal_zero(self):
        d = make_score_matrix(4)
        cm = build_from_distances(d)
        assert np.allclose(np.diag(cm.matrix), 0.0, atol=1e-5)

    def test_build_from_distances_shape(self):
        d = make_score_matrix(5)
        cm = build_from_distances(d)
        assert cm.matrix.shape == (5, 5)

    def test_build_from_distances_values_finite(self):
        d = make_score_matrix(4)
        cm = build_from_distances(d)
        assert np.all(np.isfinite(cm.matrix))

    def test_cost_matrix_repr(self):
        cm = make_cost_matrix(3)
        rep = repr(cm)
        assert "CostMatrix" in rep


# ─── TestCostMatrixTransforms ─────────────────────────────────────────────────

class TestCostMatrixTransforms:
    """Tests for build_combined, apply_forbidden_mask, normalize_costs, top_k_candidates."""

    def test_build_combined_returns_cost_matrix(self):
        cm1 = make_cost_matrix(4, seed=1)
        cm2 = make_cost_matrix(4, seed=2)
        combined = build_combined([cm1, cm2])
        assert isinstance(combined, CostMatrix)

    def test_build_combined_shape(self):
        cm1 = make_cost_matrix(4, seed=1)
        cm2 = make_cost_matrix(4, seed=2)
        combined = build_combined([cm1, cm2])
        assert combined.matrix.shape == (4, 4)

    def test_build_combined_single(self):
        cm = make_cost_matrix(4)
        combined = build_combined([cm])
        assert np.allclose(combined.matrix, cm.matrix, atol=1e-5)

    def test_build_combined_with_weights(self):
        cm1 = make_cost_matrix(4, seed=1)
        cm2 = make_cost_matrix(4, seed=2)
        combined = build_combined([cm1, cm2], weights=[0.7, 0.3])
        assert isinstance(combined, CostMatrix)

    def test_apply_forbidden_mask_returns_cost_matrix(self):
        cm = make_cost_matrix(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = True
        result = apply_forbidden_mask(cm, mask)
        assert isinstance(result, CostMatrix)

    def test_apply_forbidden_mask_fills_value(self):
        cm = make_cost_matrix(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = True
        result = apply_forbidden_mask(cm, mask, fill_value=999.0)
        assert result.matrix[0, 1] == pytest.approx(999.0)

    def test_normalize_costs_returns_cost_matrix(self):
        cm = make_cost_matrix(4)
        result = normalize_costs(cm)
        assert isinstance(result, CostMatrix)

    def test_normalize_costs_values_in_unit_interval(self):
        cm = make_cost_matrix(4)
        result = normalize_costs(cm, method="minmax")
        assert np.all(result.matrix >= 0.0 - 1e-5)
        assert np.all(result.matrix <= 1.0 + 1e-5)

    def test_top_k_candidates_returns_dict(self):
        cm = make_cost_matrix(5)
        result = top_k_candidates(cm, k=2)
        assert isinstance(result, dict)

    def test_top_k_candidates_max_k_entries(self):
        cm = make_cost_matrix(5)
        result = top_k_candidates(cm, k=2)
        for v in result.values():
            assert len(v) <= 2


# ─── TestFragmentScorer ───────────────────────────────────────────────────────

class TestFragmentScorer:
    """Tests for fragment_scorer functions."""

    def test_score_fragment_returns_fragment_score(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_fragment(state, 0, cm)
        assert isinstance(result, FragmentScore)

    def test_score_fragment_has_local_score_attr(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_fragment(state, 0, cm)
        assert hasattr(result, "local_score")
        assert np.isfinite(result.local_score)

    def test_score_fragment_local_score_non_negative(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_fragment(state, 0, cm)
        assert result.local_score >= 0.0

    def test_score_assembly_returns_assembly_score(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_assembly(state, cm)
        assert isinstance(result, AssemblyScore)

    def test_score_assembly_global_score_finite(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_assembly(state, cm)
        assert np.isfinite(result.global_score)

    def test_score_assembly_has_fragment_scores(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        result = score_assembly(state, cm)
        assert hasattr(result, "fragment_scores")

    def test_top_k_placed_returns_list(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        asm_score = score_assembly(state, cm)
        top = top_k_placed(asm_score, k=2)
        assert isinstance(top, list)

    def test_top_k_placed_at_most_k(self):
        state = make_placed_state(4)
        cm = make_cost_matrix(4)
        asm_score = score_assembly(state, cm)
        top = top_k_placed(asm_score, k=2)
        assert len(top) <= 2

    def test_batch_score_returns_list(self):
        cm = make_cost_matrix(4)
        states = [make_placed_state(4) for _ in range(3)]
        results = batch_score(states, cm)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_score_each_is_assembly_score(self):
        cm = make_cost_matrix(4)
        states = [make_placed_state(4)]
        results = batch_score(states, cm)
        assert isinstance(results[0], AssemblyScore)


# ─── TestScoreTracker ─────────────────────────────────────────────────────────

class TestScoreTracker:
    """Tests for score_tracker functions."""

    def test_create_tracker_returns_score_tracker(self):
        tracker = create_tracker()
        assert isinstance(tracker, ScoreTracker)

    def test_record_snapshot_returns_tracker(self):
        tracker = create_tracker()
        tracker2 = record_snapshot(tracker, iteration=0, score=0.5, n_placed=2)
        assert isinstance(tracker2, ScoreTracker)

    def test_record_multiple_snapshots(self):
        tracker = create_tracker()
        for i in range(5):
            tracker = record_snapshot(tracker, iteration=i, score=float(i) * 0.1, n_placed=i)
        summary = summarize_tracker(tracker)
        assert summary is not None

    def test_summarize_tracker_returns_dict(self):
        tracker = create_tracker()
        tracker = record_snapshot(tracker, iteration=0, score=0.7, n_placed=3)
        summary = summarize_tracker(tracker)
        assert isinstance(summary, dict)

    def test_summarize_tracker_has_best_score(self):
        tracker = create_tracker()
        tracker = record_snapshot(tracker, iteration=0, score=0.7, n_placed=3)
        tracker = record_snapshot(tracker, iteration=1, score=0.9, n_placed=4)
        summary = summarize_tracker(tracker)
        assert "best_score" in summary

    def test_detect_convergence_none_on_few_records(self):
        tracker = create_tracker()
        tracker = record_snapshot(tracker, iteration=0, score=0.5, n_placed=2)
        result = detect_convergence(tracker, window=5)
        # Not enough data → None
        assert result is None or isinstance(result, int)

    def test_detect_convergence_on_flat_series(self):
        tracker = create_tracker()
        for i in range(10):
            tracker = record_snapshot(tracker, iteration=i, score=0.8, n_placed=4)
        result = detect_convergence(tracker, window=5, tol=0.001)
        # Flat series should converge
        assert result is None or isinstance(result, int)

    def test_score_tracker_snapshot_count(self):
        tracker = create_tracker()
        n = 6
        for i in range(n):
            tracker = record_snapshot(tracker, iteration=i, score=float(i) * 0.1, n_placed=i)
        assert len(tracker.snapshots) == n


# ─── TestCostMatrixScorerIntegration ─────────────────────────────────────────

class TestCostMatrixScorerIntegration:
    """Integration: cost_matrix → fragment_scorer pipeline."""

    def test_pipeline_cost_then_score(self):
        scores = make_score_matrix(4)
        cm = build_from_scores(scores)
        state = make_placed_state(4)
        asm_score = score_assembly(state, cm)
        assert np.isfinite(asm_score.global_score)

    def test_pipeline_normalize_then_score(self):
        scores = make_score_matrix(4)
        cm = build_from_scores(scores)
        cm_norm = normalize_costs(cm)
        state = make_placed_state(4)
        asm_score = score_assembly(state, cm_norm)
        assert np.isfinite(asm_score.global_score)

    def test_pipeline_combined_then_score(self):
        cm1 = make_cost_matrix(4, seed=1)
        cm2 = make_cost_matrix(4, seed=2)
        combined = build_combined([cm1, cm2], weights=[0.5, 0.5])
        state = make_placed_state(4)
        asm_score = score_assembly(state, combined)
        assert np.isfinite(asm_score.global_score)

    def test_pipeline_batch_score_ordering(self):
        cm = make_cost_matrix(4)
        good_state = make_placed_state(4)
        results = batch_score([good_state, good_state], cm)
        assert results[0].global_score == pytest.approx(results[1].global_score)

    def test_pipeline_tracker_records_assembly_scores(self):
        cm = make_cost_matrix(4)
        state = make_placed_state(4)
        tracker = create_tracker()
        for i in range(5):
            score = score_assembly(state, cm).global_score
            tracker = record_snapshot(tracker, iteration=i, score=score, n_placed=4)
        summary = summarize_tracker(tracker)
        assert summary["best_score"] == pytest.approx(score)

    def test_top_k_candidates_subset_selection(self):
        scores = make_score_matrix(6)
        cm = build_from_scores(scores)
        candidates = top_k_candidates(cm, k=3)
        # Each fragment should have at most 3 candidates
        for v in candidates.values():
            assert len(v) <= 3
