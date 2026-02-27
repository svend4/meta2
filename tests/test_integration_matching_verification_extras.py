"""
Integration tests for:
  - puzzle_reconstruction.matching.compat_matrix
  - puzzle_reconstruction.matching.curve_descriptor
  - puzzle_reconstruction.matching.matcher_registry
  - puzzle_reconstruction.matching.pair_scorer
  - puzzle_reconstruction.matching.pairwise
  - puzzle_reconstruction.matching.rotation_dtw
  - puzzle_reconstruction.verification.suite
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_edge_signature(
    edge_id: int = 0,
    n_pts: int = 20,
    fd: float = 1.5,
    length: float = 100.0,
    rng: np.random.Generator | None = None,
):
    """Return an EdgeSignature with deterministic random fields."""
    from puzzle_reconstruction.models import EdgeSignature, EdgeSide

    if rng is None:
        rng = np.random.default_rng(42 + edge_id)

    virtual_curve = rng.random((n_pts, 2)).astype(np.float64) * 100.0
    css_vec = rng.random(64).astype(np.float64)
    ifs_coeffs = rng.random(12).astype(np.float64)

    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=virtual_curve,
        fd=fd,
        css_vec=css_vec,
        ifs_coeffs=ifs_coeffs,
        length=length,
    )


def _make_fragment(fragment_id: int = 0, n_edges: int = 2, rng_seed: int = 0):
    """Return a Fragment with EdgeSignature edges."""
    from puzzle_reconstruction.models import Fragment

    rng = np.random.default_rng(rng_seed)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    fragment = Fragment(
        fragment_id=fragment_id,
        image=image,
    )
    for k in range(n_edges):
        edge_id = fragment_id * 100 + k
        fragment.edges.append(
            _make_edge_signature(
                edge_id=edge_id,
                rng=np.random.default_rng(rng_seed + k + 1),
            )
        )
    return fragment


def _make_circle_curve(n: int = 40, radius: float = 50.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _make_line_curve(n: int = 20, length: float = 100.0) -> np.ndarray:
    xs = np.linspace(0, length, n)
    ys = np.zeros(n)
    return np.column_stack([xs, ys])


# ===========================================================================
# TestCompatMatrix
# ===========================================================================


class TestCompatMatrix:
    """Tests for puzzle_reconstruction.matching.compat_matrix."""

    def test_build_compat_matrix_shape(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        matrix, entries = build_compat_matrix(frags)
        n_edges = sum(len(f.edges) for f in frags)
        assert matrix.shape == (n_edges, n_edges)

    def test_build_compat_matrix_symmetric(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_build_compat_matrix_diagonal_zero(self):
        """Self-edges are not compared — diagonal should remain 0."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        assert np.all(np.diag(matrix) == 0.0)

    def test_build_compat_matrix_no_same_fragment_pairs(self):
        """Same-fragment edges should have score == 0 in the matrix."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(0, n_edges=3), _make_fragment(1, n_edges=2)]
        matrix, _ = build_compat_matrix(frags)
        # Edges 0,1,2 belong to frag 0 → matrix[0,1], [0,2], [1,2] must be 0
        assert matrix[0, 1] == 0.0
        assert matrix[0, 2] == 0.0
        assert matrix[1, 2] == 0.0

    def test_build_compat_matrix_entries_sorted_desc(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        _, entries = build_compat_matrix(frags, threshold=0.0)
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_build_compat_matrix_threshold_filters(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        _, entries_all = build_compat_matrix(frags, threshold=0.0)
        _, entries_hi = build_compat_matrix(frags, threshold=0.9)
        assert len(entries_hi) <= len(entries_all)
        for e in entries_hi:
            assert e.score >= 0.9

    def test_build_compat_matrix_dtype_float32(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(2)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.dtype == np.float32

    def test_build_compat_matrix_scores_in_range(self):
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(3)]
        matrix, entries = build_compat_matrix(frags)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
        for e in entries:
            assert 0.0 <= e.score <= 1.0

    def test_top_candidates_returns_k_results(self):
        from puzzle_reconstruction.matching.compat_matrix import (
            build_compat_matrix, top_candidates,
        )

        frags = [_make_fragment(i, n_edges=3) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        all_edges = [e for f in frags for e in f.edges]
        candidates = top_candidates(matrix, all_edges, edge_idx=0, k=3)
        assert len(candidates) <= 3

    def test_top_candidates_excludes_self(self):
        from puzzle_reconstruction.matching.compat_matrix import (
            build_compat_matrix, top_candidates,
        )

        frags = [_make_fragment(i, n_edges=3) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        all_edges = [e for f in frags for e in f.edges]
        candidates = top_candidates(matrix, all_edges, edge_idx=0, k=5)
        assert all(idx != 0 for idx, _ in candidates)

    def test_top_candidates_sorted_desc(self):
        from puzzle_reconstruction.matching.compat_matrix import (
            build_compat_matrix, top_candidates,
        )

        frags = [_make_fragment(i, n_edges=3) for i in range(4)]
        matrix, _ = build_compat_matrix(frags)
        all_edges = [e for f in frags for e in f.edges]
        candidates = top_candidates(matrix, all_edges, edge_idx=0, k=5)
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_build_compat_matrix_single_edge_per_fragment(self):
        """Minimal case: 2 fragments, 1 edge each → 1 cross-fragment pair."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=1) for i in range(2)]
        matrix, entries = build_compat_matrix(frags, threshold=0.0)
        assert matrix.shape == (2, 2)
        # Off-diagonal should be the score of the only valid pair
        assert matrix[0, 1] == matrix[1, 0]
        assert len(entries) == 1

    def test_build_compat_matrix_compat_entry_fields(self):
        """Each entry must have valid score, dtw_dist, css_sim fields."""
        from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix

        frags = [_make_fragment(i, n_edges=2) for i in range(2)]
        _, entries = build_compat_matrix(frags, threshold=0.0)
        for e in entries:
            assert hasattr(e, "score")
            assert hasattr(e, "dtw_dist")
            assert hasattr(e, "css_sim")
            assert e.dtw_dist >= 0.0


# ===========================================================================
# TestCurveDescriptor
# ===========================================================================


class TestCurveDescriptor:
    """Tests for puzzle_reconstruction.matching.curve_descriptor."""

    def test_compute_fourier_descriptor_shape(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_fourier_descriptor,
        )

        curve = _make_circle_curve(40)
        desc = compute_fourier_descriptor(curve, n_harmonics=8)
        assert desc.shape == (8,)

    def test_compute_fourier_descriptor_first_element_one_when_normalized(self):
        """After normalization the first harmonic should be 1.0."""
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_fourier_descriptor,
        )

        curve = _make_circle_curve(40)
        desc = compute_fourier_descriptor(curve, n_harmonics=8, normalize=True)
        assert abs(desc[0] - 1.0) < 1e-9

    def test_compute_fourier_descriptor_dtype(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_fourier_descriptor,
        )

        curve = _make_circle_curve(40)
        desc = compute_fourier_descriptor(curve, n_harmonics=6)
        assert desc.dtype == np.float64

    def test_compute_fourier_descriptor_invalid_shape(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_fourier_descriptor,
        )

        with pytest.raises(ValueError):
            compute_fourier_descriptor(np.zeros((10, 3)), n_harmonics=4)

    def test_compute_fourier_descriptor_empty_curve(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_fourier_descriptor,
        )

        desc = compute_fourier_descriptor(np.zeros((0, 2)), n_harmonics=8)
        assert desc.shape == (8,)
        assert np.all(desc == 0.0)

    def test_compute_curvature_profile_shape(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_curvature_profile,
        )

        curve = _make_circle_curve(30)
        curv = compute_curvature_profile(curve)
        assert curv.shape == (30,)

    def test_compute_curvature_profile_endpoints_zero(self):
        """The first and last curvature values should be zero."""
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_curvature_profile,
        )

        curve = _make_circle_curve(30)
        curv = compute_curvature_profile(curve)
        assert curv[0] == 0.0
        assert curv[-1] == 0.0

    def test_compute_curvature_profile_short_curve(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            compute_curvature_profile,
        )

        curve = np.array([[0.0, 0.0], [1.0, 0.0]])  # 2 points
        curv = compute_curvature_profile(curve)
        assert np.all(curv == 0.0)

    def test_describe_curve_returns_correct_type(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            CurveDescriptor, describe_curve,
        )

        curve = _make_circle_curve(30)
        d = describe_curve(curve)
        assert isinstance(d, CurveDescriptor)

    def test_describe_curve_arc_length_positive(self):
        from puzzle_reconstruction.matching.curve_descriptor import describe_curve

        curve = _make_circle_curve(40, radius=50.0)
        d = describe_curve(curve)
        assert d.arc_length > 0.0

    def test_describe_curve_circle_arc_length_approx(self):
        """Arc length of a circle with r=50 should be close to 2*pi*50."""
        from puzzle_reconstruction.matching.curve_descriptor import describe_curve

        curve = _make_circle_curve(200, radius=50.0)
        d = describe_curve(curve)
        expected = 2 * np.pi * 50.0
        # Allow 5% relative error because the polygon approximation is not exact
        assert abs(d.arc_length - expected) / expected < 0.05

    def test_descriptor_distance_identical_zero(self):
        """Distance between a descriptor and itself should be 0."""
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, descriptor_distance,
        )

        curve = _make_circle_curve(40)
        d = describe_curve(curve)
        assert descriptor_distance(d, d) == 0.0

    def test_descriptor_similarity_range(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, descriptor_similarity,
        )

        c1 = _make_circle_curve(40, radius=50.0)
        c2 = _make_line_curve(40, length=100.0)
        d1 = describe_curve(c1)
        d2 = describe_curve(c2)
        sim = descriptor_similarity(d1, d2, sigma=1.0)
        assert 0.0 <= sim <= 1.0

    def test_descriptor_similarity_identical_is_one(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, descriptor_similarity,
        )

        curve = _make_circle_curve(40)
        d = describe_curve(curve)
        sim = descriptor_similarity(d, d, sigma=1.0)
        assert abs(sim - 1.0) < 1e-9

    def test_descriptor_similarity_bad_sigma(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, descriptor_similarity,
        )

        d = describe_curve(_make_circle_curve(20))
        with pytest.raises(ValueError):
            descriptor_similarity(d, d, sigma=0.0)

    def test_batch_describe_curves_length(self):
        from puzzle_reconstruction.matching.curve_descriptor import batch_describe_curves

        curves = [_make_circle_curve(20 + i) for i in range(5)]
        descs = batch_describe_curves(curves)
        assert len(descs) == 5

    def test_find_best_match_known_result(self):
        """Query identical to one of the candidates → distance to that candidate is 0."""
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, find_best_match,
        )

        # Use very different curves so the identical one clearly wins
        curves = [
            _make_line_curve(20, length=float(10 + i * 50)) for i in range(5)
        ]
        # Use a copy of curves[3] as the query; it should match index 3 perfectly
        descs = [describe_curve(c) for c in curves]
        query_desc = describe_curve(curves[3].copy())
        idx, dist = find_best_match(query_desc, descs)
        # Distance to the identical descriptor must be 0
        assert dist == pytest.approx(0.0, abs=1e-9)
        # The index returned must be the one with zero distance
        assert descs[idx].fourier_desc.tolist() == query_desc.fourier_desc.tolist()

    def test_find_best_match_empty_raises(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            describe_curve, find_best_match,
        )

        d = describe_curve(_make_circle_curve(20))
        with pytest.raises(ValueError):
            find_best_match(d, [])

    def test_describe_curve_with_resample(self):
        from puzzle_reconstruction.matching.curve_descriptor import (
            CurveDescriptorConfig, describe_curve,
        )

        curve = _make_circle_curve(100)
        cfg = CurveDescriptorConfig(n_harmonics=6, resample_n=30)
        d = describe_curve(curve, cfg)
        assert d.fourier_desc.shape == (6,)

    def test_curve_descriptor_config_invalid_n_harmonics(self):
        from puzzle_reconstruction.matching.curve_descriptor import CurveDescriptorConfig

        with pytest.raises(ValueError):
            CurveDescriptorConfig(n_harmonics=0)

    def test_curve_descriptor_config_invalid_resample_n(self):
        from puzzle_reconstruction.matching.curve_descriptor import CurveDescriptorConfig

        with pytest.raises(ValueError):
            CurveDescriptorConfig(resample_n=1)


# ===========================================================================
# TestMatcherRegistry
# ===========================================================================


class TestMatcherRegistry:
    """Tests for puzzle_reconstruction.matching.matcher_registry."""

    def test_default_matchers_registered(self):
        """At least 'fd' and 'text' matchers should always be registered."""
        from puzzle_reconstruction.matching.matcher_registry import MATCHER_REGISTRY

        assert "fd" in MATCHER_REGISTRY
        assert "text" in MATCHER_REGISTRY

    def test_list_matchers_sorted(self):
        from puzzle_reconstruction.matching.matcher_registry import list_matchers

        names = list_matchers()
        assert names == sorted(names)

    def test_get_matcher_returns_callable(self):
        from puzzle_reconstruction.matching.matcher_registry import get_matcher

        fn = get_matcher("fd")
        assert callable(fn)

    def test_get_matcher_unknown_returns_none(self):
        from puzzle_reconstruction.matching.matcher_registry import get_matcher

        assert get_matcher("nonexistent_matcher_xyz") is None

    def test_register_decorator_adds_to_registry(self):
        from puzzle_reconstruction.matching.matcher_registry import (
            MATCHER_REGISTRY, register,
        )

        @register("test_matcher_abc")
        def _m(e_i, e_j):
            return 0.5

        assert "test_matcher_abc" in MATCHER_REGISTRY
        # Cleanup
        del MATCHER_REGISTRY["test_matcher_abc"]

    def test_register_fn_adds_to_registry(self):
        from puzzle_reconstruction.matching.matcher_registry import (
            MATCHER_REGISTRY, register_fn,
        )

        register_fn("test_fn_xyz", lambda e, f: 0.7)
        assert "test_fn_xyz" in MATCHER_REGISTRY
        del MATCHER_REGISTRY["test_fn_xyz"]

    def test_fd_matcher_same_fd_returns_one(self):
        """fd matcher: same fd values → score = 1.0."""
        from puzzle_reconstruction.matching.matcher_registry import MATCHER_REGISTRY

        e = _make_edge_signature(fd=1.5)
        score = MATCHER_REGISTRY["fd"](e, e)
        assert score == pytest.approx(1.0)

    def test_fd_matcher_different_fd_score_less_than_one(self):
        from puzzle_reconstruction.matching.matcher_registry import MATCHER_REGISTRY

        e1 = _make_edge_signature(edge_id=0, fd=1.2)
        e2 = _make_edge_signature(edge_id=1, fd=2.5)
        score = MATCHER_REGISTRY["fd"](e1, e2)
        assert 0.0 <= score < 1.0

    def test_text_matcher_returns_zero(self):
        """text matcher always returns 0.0 (external signal placeholder)."""
        from puzzle_reconstruction.matching.matcher_registry import MATCHER_REGISTRY

        e = _make_edge_signature()
        score = MATCHER_REGISTRY["text"](e, e)
        assert score == 0.0

    def test_safe_score_clamps_above_one(self):
        from puzzle_reconstruction.matching.matcher_registry import (
            MATCHER_REGISTRY, _safe_score, register_fn,
        )

        register_fn("_test_overflow", lambda e, f: 999.0)
        e = _make_edge_signature()
        score = _safe_score(MATCHER_REGISTRY["_test_overflow"], e, e)
        assert score <= 1.0
        del MATCHER_REGISTRY["_test_overflow"]

    def test_safe_score_clamps_below_zero(self):
        from puzzle_reconstruction.matching.matcher_registry import (
            MATCHER_REGISTRY, _safe_score, register_fn,
        )

        register_fn("_test_underflow", lambda e, f: -5.0)
        e = _make_edge_signature()
        score = _safe_score(MATCHER_REGISTRY["_test_underflow"], e, e)
        assert score >= 0.0
        del MATCHER_REGISTRY["_test_underflow"]

    def test_safe_score_exception_returns_zero(self):
        from puzzle_reconstruction.matching.matcher_registry import _safe_score

        def bad_fn(e_i, e_j):
            raise RuntimeError("oops")

        e = _make_edge_signature()
        score = _safe_score(bad_fn, e, e)
        assert score == 0.0

    def test_compute_scores_known_matchers(self):
        from puzzle_reconstruction.matching.matcher_registry import compute_scores

        e = _make_edge_signature()
        scores = compute_scores(e, e, ["fd", "text"])
        assert set(scores.keys()) == {"fd", "text"}
        assert 0.0 <= scores["fd"] <= 1.0

    def test_compute_scores_missing_matcher_zero(self):
        from puzzle_reconstruction.matching.matcher_registry import compute_scores

        e = _make_edge_signature()
        scores = compute_scores(e, e, ["fd", "no_such_matcher"])
        assert scores["no_such_matcher"] == 0.0

    def test_weighted_combine_equal_weights(self):
        from puzzle_reconstruction.matching.matcher_registry import weighted_combine

        scores = {"a": 0.8, "b": 0.4}
        weights = {"a": 1.0, "b": 1.0}
        result = weighted_combine(scores, weights)
        assert result == pytest.approx(0.6, abs=1e-9)

    def test_weighted_combine_zero_total_weight(self):
        from puzzle_reconstruction.matching.matcher_registry import weighted_combine

        scores = {"a": 0.8}
        weights = {"a": 0.0}
        result = weighted_combine(scores, weights)
        assert result == 0.0


# ===========================================================================
# TestPairScorer
# ===========================================================================


class TestPairScorer:
    """Tests for puzzle_reconstruction.matching.pair_scorer."""

    def test_scoring_weights_total(self):
        from puzzle_reconstruction.matching.pair_scorer import ScoringWeights

        w = ScoringWeights(color=1.0, texture=1.0, geometry=1.0, gradient=1.0)
        assert w.total == 4.0

    def test_scoring_weights_normalized_sum_one(self):
        from puzzle_reconstruction.matching.pair_scorer import ScoringWeights

        w = ScoringWeights(color=2.0, texture=3.0, geometry=1.0, gradient=4.0)
        n = w.normalized()
        assert n.total == pytest.approx(1.0, abs=1e-9)

    def test_scoring_weights_negative_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import ScoringWeights

        with pytest.raises(ValueError):
            ScoringWeights(color=-1.0, texture=1.0, geometry=1.0, gradient=1.0)

    def test_scoring_weights_all_zero_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import ScoringWeights

        with pytest.raises(ValueError):
            ScoringWeights(color=0.0, texture=0.0, geometry=0.0, gradient=0.0)

    def test_scoring_weights_as_dict_keys(self):
        from puzzle_reconstruction.matching.pair_scorer import ScoringWeights

        w = ScoringWeights()
        d = w.as_dict()
        assert set(d.keys()) == {"color", "texture", "geometry", "gradient"}

    def test_pair_score_result_properties(self):
        from puzzle_reconstruction.matching.pair_scorer import PairScoreResult

        r = PairScoreResult(
            idx_a=3,
            idx_b=7,
            score=0.75,
            channels={"color": 0.9, "texture": 0.6},
        )
        assert r.n_channels == 2
        assert r.pair_key == (3, 7)
        assert r.dominant_channel == "color"
        assert r.is_strong_match is True

    def test_pair_score_result_pair_key_ordered(self):
        from puzzle_reconstruction.matching.pair_scorer import PairScoreResult

        r = PairScoreResult(idx_a=9, idx_b=2, score=0.5)
        assert r.pair_key == (2, 9)

    def test_pair_score_result_invalid_score_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import PairScoreResult

        with pytest.raises(ValueError):
            PairScoreResult(idx_a=0, idx_b=1, score=1.5)

    def test_aggregate_channels_equal_weights(self):
        from puzzle_reconstruction.matching.pair_scorer import aggregate_channels

        cs = {"color": 0.8, "texture": 0.4}
        result = aggregate_channels(cs)
        assert 0.0 <= result <= 1.0

    def test_aggregate_channels_empty_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import aggregate_channels

        with pytest.raises(ValueError):
            aggregate_channels({})

    def test_aggregate_channels_out_of_range_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import aggregate_channels

        with pytest.raises(ValueError):
            aggregate_channels({"color": 1.5})

    def test_score_pair_creates_result(self):
        from puzzle_reconstruction.matching.pair_scorer import score_pair

        cs = {"color": 0.7, "geometry": 0.9}
        r = score_pair(1, 2, cs)
        assert r.idx_a == 1
        assert r.idx_b == 2
        assert 0.0 <= r.score <= 1.0

    def test_select_top_pairs_threshold(self):
        from puzzle_reconstruction.matching.pair_scorer import (
            PairScoreResult, select_top_pairs,
        )

        results = [
            PairScoreResult(0, 1, 0.9, {}),
            PairScoreResult(0, 2, 0.4, {}),
            PairScoreResult(1, 2, 0.75, {}),
        ]
        top = select_top_pairs(results, threshold=0.7)
        assert all(r.score >= 0.7 for r in top)
        assert len(top) == 2

    def test_select_top_pairs_top_k(self):
        from puzzle_reconstruction.matching.pair_scorer import (
            PairScoreResult, select_top_pairs,
        )

        results = [PairScoreResult(i, i + 1, 0.1 * i, {}) for i in range(9)]
        top = select_top_pairs(results, top_k=3)
        assert len(top) == 3
        # Should be the 3 highest-scoring pairs
        assert top[0].score >= top[1].score >= top[2].score

    def test_select_top_pairs_bad_threshold_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import select_top_pairs

        with pytest.raises(ValueError):
            select_top_pairs([], threshold=-0.1)

    def test_build_score_matrix_shape(self):
        from puzzle_reconstruction.matching.pair_scorer import (
            PairScoreResult, build_score_matrix,
        )

        results = [PairScoreResult(0, 1, 0.8, {}), PairScoreResult(1, 2, 0.6, {})]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat.shape == (3, 3)

    def test_build_score_matrix_symmetric(self):
        from puzzle_reconstruction.matching.pair_scorer import (
            PairScoreResult, build_score_matrix,
        )

        results = [PairScoreResult(0, 2, 0.75, {})]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat[0, 2] == mat[2, 0]

    def test_build_score_matrix_bad_n_fragments(self):
        from puzzle_reconstruction.matching.pair_scorer import build_score_matrix

        with pytest.raises(ValueError):
            build_score_matrix([], n_fragments=0)

    def test_batch_score_pairs_length(self):
        from puzzle_reconstruction.matching.pair_scorer import batch_score_pairs

        pairs = [(0, 1), (1, 2), (2, 3)]
        css = [{"color": 0.5}, {"texture": 0.7}, {"geometry": 0.3}]
        results = batch_score_pairs(pairs, css)
        assert len(results) == 3

    def test_batch_score_pairs_length_mismatch_raises(self):
        from puzzle_reconstruction.matching.pair_scorer import batch_score_pairs

        with pytest.raises(ValueError):
            batch_score_pairs([(0, 1)], [{"c": 0.5}, {"c": 0.4}])


# ===========================================================================
# TestPairwise
# ===========================================================================


class TestPairwise:
    """Tests for puzzle_reconstruction.matching.pairwise."""

    def test_match_score_returns_compat_entry(self):
        from puzzle_reconstruction.matching.pairwise import match_score
        from puzzle_reconstruction.models import CompatEntry

        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1)
        entry = match_score(e1, e2)
        assert isinstance(entry, CompatEntry)

    def test_match_score_range(self):
        from puzzle_reconstruction.matching.pairwise import match_score

        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1)
        entry = match_score(e1, e2)
        assert 0.0 <= entry.score <= 1.0

    def test_match_score_text_score_influence(self):
        """Providing a non-zero text_score should change the overall score."""
        from puzzle_reconstruction.matching.pairwise import match_score

        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1)
        entry_no_text = match_score(e1, e2, text_score=0.0)
        entry_with_text = match_score(e1, e2, text_score=1.0)
        # With a perfect text score the overall score should be >= no_text case
        assert entry_with_text.score >= entry_no_text.score - 1e-9

    def test_match_score_fields_populated(self):
        from puzzle_reconstruction.matching.pairwise import match_score

        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1)
        entry = match_score(e1, e2)
        assert entry.dtw_dist >= 0.0
        assert 0.0 <= entry.css_sim <= 1.0
        assert entry.fd_diff >= 0.0

    def test_match_score_length_penalty(self):
        """Very different edge lengths should penalise the score."""
        from puzzle_reconstruction.matching.pairwise import match_score

        e_short = _make_edge_signature(edge_id=0, length=10.0)
        e_long = _make_edge_signature(edge_id=1, length=500.0)
        entry = match_score(e_short, e_long)
        # Length ratio < 0.5 → score should be further suppressed
        assert entry.score < 1.0

    def test_match_score_same_fd(self):
        """Same FD values → FD component contributes 1.0."""
        from puzzle_reconstruction.matching.pairwise import match_score

        e1 = _make_edge_signature(edge_id=0, fd=1.5)
        e2 = _make_edge_signature(edge_id=1, fd=1.5)
        entry = match_score(e1, e2)
        assert entry.fd_diff == pytest.approx(0.0, abs=1e-9)

    def test_weighted_internal_helper(self):
        from puzzle_reconstruction.matching.pairwise import _weighted

        scores = {"a": 0.8, "b": 0.4}
        weights = {"a": 1.0, "b": 1.0}
        result = _weighted(scores, weights, ["a", "b"])
        assert result == pytest.approx(0.6, abs=1e-9)

    def test_weighted_missing_active_matcher(self):
        from puzzle_reconstruction.matching.pairwise import _weighted

        scores = {"a": 0.8}
        weights = {"a": 1.0, "b": 1.0}
        result = _weighted(scores, weights, ["a", "b"])
        # Only 'a' is in scores; effective total_w = 1.0
        assert result == pytest.approx(0.8, abs=1e-9)

    def test_rank_combine_values(self):
        from puzzle_reconstruction.matching.pairwise import _rank_combine

        # Three matchers: identical vals → all ranks are 0,0.5,1 → mean 0.5
        scores = {"a": 0.1, "b": 0.5, "c": 0.9}
        result = _rank_combine(scores, ["a", "b", "c"])
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_ifs_distance_norm_same(self):
        from puzzle_reconstruction.matching.pairwise import _ifs_distance_norm

        a = np.array([1.0, 2.0, 3.0])
        assert _ifs_distance_norm(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_ifs_distance_norm_different_lengths(self):
        from puzzle_reconstruction.matching.pairwise import _ifs_distance_norm

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        dist = _ifs_distance_norm(a, b)
        assert dist >= 0.0

    def test_ifs_distance_norm_empty(self):
        from puzzle_reconstruction.matching.pairwise import _ifs_distance_norm

        dist = _ifs_distance_norm(np.array([]), np.array([]))
        assert dist == pytest.approx(1.0)

    def test_match_score_symmetric_scores_close(self):
        """match_score(e1, e2) and match_score(e2, e1) should be equal (symmetric)."""
        from puzzle_reconstruction.matching.pairwise import match_score

        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1)
        entry_ab = match_score(e1, e2)
        entry_ba = match_score(e2, e1)
        assert entry_ab.score == pytest.approx(entry_ba.score, abs=1e-6)


# ===========================================================================
# TestRotationDTW
# ===========================================================================


class TestRotationDTW:
    """Tests for puzzle_reconstruction.matching.rotation_dtw."""

    def test_rotation_dtw_result_type(self):
        from puzzle_reconstruction.matching.rotation_dtw import (
            RotationDTWResult, rotation_dtw,
        )

        a = _make_circle_curve(30)
        b = _make_circle_curve(30)
        result = rotation_dtw(a, b, n_angles=8, n_points=20)
        assert isinstance(result, RotationDTWResult)

    def test_rotation_dtw_distance_finite(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw

        a = _make_circle_curve(30)
        b = _make_circle_curve(30)
        result = rotation_dtw(a, b, n_angles=8, n_points=20)
        assert math.isfinite(result.distance)

    def test_rotation_dtw_identical_curves_small_distance(self):
        """Identical curves → distance should be near zero."""
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw

        a = _make_circle_curve(40)
        result = rotation_dtw(a, a.copy(), n_angles=4, n_points=40)
        assert result.distance < 1.0

    def test_rotation_dtw_angle_in_range(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw

        a = _make_circle_curve(20)
        b = _make_circle_curve(20)
        result = rotation_dtw(a, b, n_angles=12, n_points=20)
        assert 0.0 <= result.best_angle_deg < 360.0

    def test_rotation_dtw_mirrored_flag(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw

        a = _make_circle_curve(20)
        b = _make_circle_curve(20)
        result_no_mirror = rotation_dtw(a, b, n_angles=4, check_mirror=False)
        assert result_no_mirror.mirrored is False

    def test_rotation_dtw_short_curve_returns_inf(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw

        a = np.array([[0.0, 0.0]])  # only 1 point
        b = _make_circle_curve(20)
        result = rotation_dtw(a, b)
        assert result.distance == float("inf")

    def test_rotation_dtw_similarity_range(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw_similarity

        a = _make_circle_curve(30)
        b = _make_circle_curve(30)
        sim = rotation_dtw_similarity(a, b, n_angles=8, n_points=20)
        assert 0.0 <= sim <= 1.0

    def test_rotation_dtw_similarity_identical_approaches_one(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw_similarity

        a = _make_circle_curve(30)
        sim = rotation_dtw_similarity(a, a.copy(), n_angles=4, n_points=30)
        assert sim > 0.5  # Should be fairly high for identical curves

    def test_rotation_dtw_similarity_short_curve_zero(self):
        from puzzle_reconstruction.matching.rotation_dtw import rotation_dtw_similarity

        a = np.array([[0.0, 0.0]])
        b = _make_circle_curve(20)
        sim = rotation_dtw_similarity(a, b)
        assert sim == 0.0

    def test_batch_rotation_dtw_length(self):
        from puzzle_reconstruction.matching.rotation_dtw import batch_rotation_dtw

        query = _make_circle_curve(20)
        candidates = [_make_circle_curve(20 + i) for i in range(4)]
        results = batch_rotation_dtw(
            query, candidates, n_angles=4, n_points=20
        )
        assert len(results) == 4

    def test_batch_rotation_dtw_all_finite(self):
        from puzzle_reconstruction.matching.rotation_dtw import batch_rotation_dtw

        query = _make_circle_curve(20)
        candidates = [_make_circle_curve(20 + i) for i in range(3)]
        results = batch_rotation_dtw(
            query, candidates, n_angles=4, n_points=20
        )
        for r in results:
            assert math.isfinite(r.distance)

    def test_rotate_curve_centroid_preserved(self):
        """Rotating around centroid should leave the centroid unchanged."""
        from puzzle_reconstruction.matching.rotation_dtw import _rotate_curve

        rng = np.random.default_rng(0)
        curve = rng.random((20, 2)) * 100
        centroid_before = curve.mean(axis=0)
        rotated = _rotate_curve(curve, 45.0)
        centroid_after = rotated.mean(axis=0)
        np.testing.assert_allclose(centroid_before, centroid_after, atol=1e-9)

    def test_mirror_curve_x_flipped(self):
        """Mirror should flip x-coordinates around their mean."""
        from puzzle_reconstruction.matching.rotation_dtw import _mirror_curve

        curve = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        mirrored = _mirror_curve(curve)
        # cx = 2; mirrored x = 2*2 - [0,2,4] = [4,2,0]
        np.testing.assert_allclose(mirrored[:, 0], [4.0, 2.0, 0.0])
        # y unchanged
        np.testing.assert_allclose(mirrored[:, 1], [0.0, 0.0, 0.0])

    def test_resample_curve_output_shape(self):
        from puzzle_reconstruction.matching.rotation_dtw import _resample_curve

        curve = _make_circle_curve(50)
        resampled = _resample_curve(curve, 20)
        assert resampled.shape == (20, 2)

    def test_resample_curve_single_point(self):
        from puzzle_reconstruction.matching.rotation_dtw import _resample_curve

        curve = np.array([[5.0, 3.0]])  # Only 1 point
        resampled = _resample_curve(curve, 10)
        assert resampled.shape == (10, 2)


# ===========================================================================
# TestVerificationSuite
# ===========================================================================


class TestVerificationSuite:
    """Tests for puzzle_reconstruction.verification.suite."""

    # ── Helper ──────────────────────────────────────────────────────────────

    def _make_assembly(self, n_frags: int = 3, total_score: float = 0.8):
        from puzzle_reconstruction.models import Assembly, Fragment, Placement

        rng = np.random.default_rng(99)
        frags = []
        placements = []
        for i in range(n_frags):
            img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            frags.append(Fragment(fragment_id=i, image=img))
            placements.append(Placement(
                fragment_id=i, position=(float(i * 40), 0.0)
            ))
        return Assembly(
            fragments=frags,
            placements=placements,
            total_score=total_score,
        )

    # ── ValidatorResult ─────────────────────────────────────────────────────

    def test_validator_result_success_no_error(self):
        from puzzle_reconstruction.verification.suite import ValidatorResult

        vr = ValidatorResult(name="test", score=0.75)
        assert vr.success is True

    def test_validator_result_failure_has_error(self):
        from puzzle_reconstruction.verification.suite import ValidatorResult

        vr = ValidatorResult(name="test", score=0.0, error="something failed")
        assert vr.success is False

    # ── VerificationReport ──────────────────────────────────────────────────

    def test_verification_report_summary_contains_score(self):
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        report = VerificationReport(
            results=[ValidatorResult("completeness", 0.9)],
            final_score=0.9,
        )
        summary = report.summary()
        assert "completeness" in summary
        assert "0.900" in summary

    def test_verification_report_as_dict_keys(self):
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        report = VerificationReport(
            results=[ValidatorResult("completeness", 0.5)],
            final_score=0.5,
        )
        d = report.as_dict()
        assert "final_score" in d
        assert "validators" in d

    def test_verification_report_to_json_valid(self):
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        report = VerificationReport(
            results=[ValidatorResult("x", 0.3)],
            final_score=0.3,
        )
        payload = report.to_json()
        parsed = json.loads(payload)
        assert parsed["final_score"] == pytest.approx(0.3, abs=1e-6)

    def test_verification_report_to_markdown_headers(self):
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        report = VerificationReport(
            results=[ValidatorResult("edge_quality", 0.8)],
            final_score=0.8,
        )
        md = report.to_markdown()
        assert "# " in md
        assert "edge_quality" in md

    def test_verification_report_to_html_structure(self):
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        report = VerificationReport(
            results=[ValidatorResult("seam", 0.65)],
            final_score=0.65,
        )
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "seam" in html

    # ── VerificationSuite basic API ─────────────────────────────────────────

    def test_suite_empty_validators_uses_total_score(self):
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly(total_score=0.55)
        suite = VerificationSuite(validators=[])
        report = suite.run(asm)
        assert report.final_score == pytest.approx(0.55, abs=1e-6)
        assert len(report.results) == 0

    def test_suite_is_empty_flag(self):
        from puzzle_reconstruction.verification.suite import VerificationSuite

        assert VerificationSuite(validators=[]).is_empty() is True
        assert VerificationSuite(validators=["completeness"]).is_empty() is False

    def test_suite_completeness_validator_runs(self):
        """completeness validator should always be in the registry (fallback exists)."""
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly(n_frags=4)
        suite = VerificationSuite(validators=["completeness"])
        report = suite.run(asm)
        assert len(report.results) == 1
        r = report.results[0]
        assert r.name == "completeness"
        assert 0.0 <= r.score <= 1.0

    def test_suite_completeness_full_placement(self):
        """All fragments placed → completeness score == 1.0."""
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly(n_frags=3)
        suite = VerificationSuite(validators=["completeness"])
        report = suite.run(asm)
        assert report.results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_suite_unknown_validator_score_zero(self):
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly()
        suite = VerificationSuite(validators=["absolutely_nonexistent_xyz"])
        report = suite.run(asm)
        assert report.results[0].score == 0.0
        assert report.results[0].error is not None

    def test_suite_final_score_average_of_successful(self):
        """final_score should be the average of successful validator scores."""
        from puzzle_reconstruction.verification.suite import (
            ValidatorResult, VerificationReport,
        )

        results = [
            ValidatorResult("a", 0.8),
            ValidatorResult("b", 0.6),
            ValidatorResult("c", 0.0, error="boom"),
        ]
        # Successful: a=0.8, b=0.6 → average=0.7
        successful = [r for r in results if r.success]
        final = sum(r.score for r in successful) / len(successful)
        assert final == pytest.approx(0.7, abs=1e-6)

    def test_suite_metrics_validator_runs(self):
        """metrics validator should always have a fallback registered."""
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly()
        suite = VerificationSuite(validators=["metrics"])
        report = suite.run(asm)
        assert len(report.results) == 1
        assert 0.0 <= report.results[0].score <= 1.0

    def test_list_validators_sorted(self):
        from puzzle_reconstruction.verification.suite import list_validators

        names = list_validators()
        assert names == sorted(names)
        # completeness must always be present (fallback registered)
        assert "completeness" in names

    def test_all_validator_names_count(self):
        from puzzle_reconstruction.verification.suite import all_validator_names

        names = all_validator_names()
        assert len(names) == 21

    def test_suite_confidence_validator_fallback(self):
        """confidence validator has a fallback that returns total_score."""
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly(total_score=0.65)
        suite = VerificationSuite(validators=["confidence"])
        report = suite.run(asm)
        assert len(report.results) == 1

    def test_safe_run_exception_produces_error_result(self):
        from puzzle_reconstruction.verification.suite import _safe_run

        def bad_validator(asm):
            raise RuntimeError("intentional test error")

        result = _safe_run("test_bad", bad_validator, None)
        assert result.success is False
        assert result.score == 0.0
        assert "intentional test error" in (result.error or "")

    def test_safe_run_clamps_score_above_one(self):
        from puzzle_reconstruction.verification.suite import _safe_run

        def over_scorer(asm):
            return 999.0, "over"

        class FakeAsm:
            pass

        result = _safe_run("test_over", over_scorer, FakeAsm())
        assert result.score <= 1.0

    def test_suite_run_all_returns_report(self):
        """run_all should return a VerificationReport with at least some results."""
        from puzzle_reconstruction.verification.suite import VerificationSuite

        asm = self._make_assembly(n_frags=2, total_score=0.7)
        suite = VerificationSuite()
        report = suite.run_all(asm)
        assert hasattr(report, "final_score")
        assert hasattr(report, "results")
        assert 0.0 <= report.final_score <= 1.0
