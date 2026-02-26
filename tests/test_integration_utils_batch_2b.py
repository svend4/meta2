"""Integration tests for utils batch 2b.

Modules covered:
    1. puzzle_reconstruction.utils.distance_shape_utils
    2. puzzle_reconstruction.utils.edge_profile_utils
    3. puzzle_reconstruction.utils.edge_profiler
    4. puzzle_reconstruction.utils.edge_scorer
    5. puzzle_reconstruction.utils.event_affine_utils
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ── imports ──────────────────────────────────────────────────────────────────

from puzzle_reconstruction.utils.distance_shape_utils import (
    DistanceMatrixRecord,
    SimilarityPair,
    DistanceBatchResult,
    ContourMatchRecord,
    ShapeContextBatchSummary,
    MetricsRunRecord,
    EvidenceAggregationRecord,
    make_distance_record,
    make_contour_match,
)
from puzzle_reconstruction.utils.edge_profile_utils import (
    EdgeProfileConfig,
    EdgeProfile as EPUEdgeProfile,
    build_edge_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    profile_correlation,
    resample_profile,
)
from puzzle_reconstruction.utils.edge_profiler import (
    ProfileConfig,
    EdgeProfile as EPREdgeProfile,
    compute_brightness_profile,
    compute_gradient_profile,
    compute_diff_profile,
    normalize_profile,
    aggregate_profiles,
    compare_profiles,
)
from puzzle_reconstruction.utils.edge_scorer import (
    EdgeScoreConfig,
    EdgeScoreResult,
    score_edge_overlap,
    score_edge_curvature,
    score_edge_length,
    score_edge_endpoints,
    aggregate_edge_scores,
)
from puzzle_reconstruction.utils.event_affine_utils import (
    EventRecordConfig,
    EventRecordEntry,
    EventRecordSummary,
    AffineMatchConfig,
    AffineMatchEntry,
    AffineMatchSummary,
    make_event_record_entry,
    summarise_event_record_entries,
    filter_error_events,
    filter_events_by_level,
    filter_events_by_name,
    filter_events_by_time_range,
    top_k_recent_events,
    latest_event_entry,
    event_record_stats,
    compare_event_summaries,
    make_affine_match_entry,
    summarise_affine_match_entries,
    filter_strong_affine_matches,
    filter_affine_by_inliers,
    top_k_affine_match_entries,
    best_affine_match_entry,
    affine_match_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. distance_shape_utils  (11 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanceShapeUtils:
    def _square_matrix(self, n=4):
        m = rng.random((n, n))
        return (m + m.T) / 2

    def test_distance_matrix_record_basic(self):
        mat = self._square_matrix()
        rec = make_distance_record("label", "euclidean", mat)
        assert rec.n == 4
        assert rec.metric == "euclidean"

    def test_distance_matrix_record_non_square_raises(self):
        with pytest.raises(ValueError):
            DistanceMatrixRecord("x", "l2", np.zeros((3, 4)))

    def test_distance_matrix_record_max_value(self):
        mat = np.array([[0.0, 3.0], [3.0, 0.0]])
        rec = make_distance_record("a", "l2", mat)
        assert rec.max_value == pytest.approx(3.0)

    def test_distance_matrix_record_min_offdiag(self):
        mat = np.array([[0.0, 2.0, 5.0],
                        [2.0, 0.0, 1.0],
                        [5.0, 1.0, 0.0]])
        rec = make_distance_record("b", "l2", mat)
        assert rec.min_offdiag == pytest.approx(1.0)

    def test_similarity_pair_valid(self):
        sp = SimilarityPair(0, 1, 0.8)
        assert sp.is_high is True

    def test_similarity_pair_low(self):
        sp = SimilarityPair(2, 3, 0.3)
        assert sp.is_high is False

    def test_similarity_pair_invalid_similarity(self):
        with pytest.raises(ValueError):
            SimilarityPair(0, 1, 1.5)

    def test_distance_batch_result_best_pair(self):
        res = DistanceBatchResult(n_queries=3, metric="l2",
                                  top_pairs=[(0, 1, 0.9), (1, 2, 0.7)])
        assert res.best_pair == (0, 1, 0.9)

    def test_distance_batch_result_empty(self):
        res = DistanceBatchResult(n_queries=0, metric="l2")
        assert res.best_pair is None

    def test_contour_match_record_is_match(self):
        cm = make_contour_match(0, 1, 0.1, 10, similarity=0.7)
        assert cm.is_match is True

    def test_metrics_run_record_is_perfect(self):
        mrr = MetricsRunRecord("run1", precision=1.0, recall=1.0, f1=1.0, n_fragments=5)
        assert mrr.is_perfect is True

    def test_evidence_aggregation_record_is_confident(self):
        ear = EvidenceAggregationRecord(step=1, pair_id=(0, 1),
                                        n_channels=3, confidence=0.9)
        assert ear.is_confident is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. edge_profile_utils  (11 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeProfileUtils:
    def _make_profile(self, n=64, side="top"):
        vals = rng.random(n).astype(np.float32)
        return EPUEdgeProfile(values=vals, side=side)

    def test_build_edge_profile_shape(self):
        pts = rng.random((20, 2)) * 100
        ep = build_edge_profile(pts, side="top")
        assert len(ep) == 64

    def test_build_edge_profile_normalized_range(self):
        pts = rng.random((30, 2)) * 50
        ep = build_edge_profile(pts, side="bottom", cfg=EdgeProfileConfig(normalize=True))
        assert ep.values.min() >= -1e-6
        assert ep.values.max() <= 1.0 + 1e-6

    def test_build_edge_profile_custom_n_samples(self):
        pts = rng.random((15, 2)) * 10
        ep = build_edge_profile(pts, side="left", cfg=EdgeProfileConfig(n_samples=32))
        assert ep.n_samples == 32

    def test_build_edge_profile_invalid_side(self):
        pts = rng.random((10, 2))
        with pytest.raises(ValueError):
            build_edge_profile(pts, side="diagonal")

    def test_edge_profile_repr_contains_side(self):
        ep = self._make_profile(side="right")
        assert "right" in repr(ep)

    def test_profile_l2_distance_identical(self):
        ep = self._make_profile()
        assert profile_l2_distance(ep, ep) == pytest.approx(0.0)

    def test_profile_l2_distance_different(self):
        a = EPUEdgeProfile(values=np.zeros(64, dtype=np.float32), side="top")
        b = EPUEdgeProfile(values=np.ones(64, dtype=np.float32), side="top")
        assert profile_l2_distance(a, b) == pytest.approx(8.0, rel=1e-4)

    def test_profile_l2_distance_length_mismatch(self):
        a = self._make_profile(n=32)
        b = self._make_profile(n=64)
        with pytest.raises(ValueError):
            profile_l2_distance(a, b)

    def test_profile_cosine_similarity_identical(self):
        ep = self._make_profile()
        assert profile_cosine_similarity(ep, ep) == pytest.approx(1.0, rel=1e-5)

    def test_profile_correlation_identical(self):
        ep = self._make_profile()
        assert profile_correlation(ep, ep) == pytest.approx(1.0, rel=1e-5)

    def test_resample_profile_changes_length(self):
        ep = self._make_profile(n=64)
        ep32 = resample_profile(ep, 32)
        assert ep32.n_samples == 32


# ─────────────────────────────────────────────────────────────────────────────
# 3. edge_profiler  (11 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeProfiler:
    def _strip(self, rows=8, cols=32):
        return rng.random((rows, cols))

    def test_profile_config_defaults(self):
        cfg = ProfileConfig()
        assert cfg.n_samples == 32
        assert cfg.profile_type == "brightness"

    def test_profile_config_invalid_n_samples(self):
        with pytest.raises(ValueError):
            ProfileConfig(n_samples=1)

    def test_profile_config_invalid_type(self):
        with pytest.raises(ValueError):
            ProfileConfig(profile_type="unknown")

    def test_compute_brightness_profile_shape(self):
        strip = self._strip()
        prof = compute_brightness_profile(strip, n_samples=32)
        assert prof.shape == (32,)

    def test_compute_brightness_profile_values_in_range(self):
        strip = rng.random((6, 40))
        prof = compute_brightness_profile(strip, n_samples=16)
        assert prof.min() >= 0.0
        assert prof.max() <= 1.0 + 1e-9

    def test_compute_gradient_profile_shape(self):
        strip = self._strip()
        prof = compute_gradient_profile(strip, n_samples=20)
        assert len(prof) == 20

    def test_compute_gradient_profile_nonneg(self):
        strip = self._strip()
        prof = compute_gradient_profile(strip, n_samples=16)
        assert np.all(prof >= -1e-9)

    def test_compute_diff_profile_shape(self):
        strip = self._strip()
        prof = compute_diff_profile(strip, n_samples=16)
        assert len(prof) == 16

    def test_normalize_profile_range(self):
        raw = rng.random(50) * 10
        normed = normalize_profile(raw)
        assert normed.min() >= 0.0 - 1e-9
        assert normed.max() <= 1.0 + 1e-9

    def test_aggregate_profiles_equal_weights(self):
        p1 = np.ones(32)
        p2 = np.zeros(32)
        combined = aggregate_profiles([p1, p2])
        np.testing.assert_allclose(combined, np.full(32, 0.5))

    def test_compare_profiles_identical(self):
        p = rng.random(32)
        assert compare_profiles(p, p) == pytest.approx(1.0, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. edge_scorer  (11 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeScorer:
    def _line(self, n=20, dx=1.0, noise=0.0):
        xs = np.linspace(0, dx * (n - 1), n)
        ys = rng.random(n) * noise
        return np.column_stack([xs, ys])

    def test_score_config_defaults(self):
        cfg = EdgeScoreConfig()
        assert cfg.n_samples == 64
        assert cfg.endpoint_sigma == pytest.approx(10.0)

    def test_score_config_invalid_n_samples(self):
        with pytest.raises(ValueError):
            EdgeScoreConfig(n_samples=1)

    def test_score_config_normalized_weights_sum_to_one(self):
        cfg = EdgeScoreConfig()
        total = sum(cfg.normalized_weights.values())
        assert total == pytest.approx(1.0)

    def test_score_edge_overlap_identical_curves(self):
        curve = self._line(20, noise=0.0)
        score = score_edge_overlap(curve, curve)
        assert 0.0 <= score <= 1.0

    def test_score_edge_overlap_range(self):
        a = self._line(15, noise=0.5)
        b = self._line(15, noise=2.0)
        score = score_edge_overlap(a, b)
        assert 0.0 <= score <= 1.0

    def test_score_edge_curvature_short_curves(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[2.0, 0.0], [3.0, 1.0]])
        score = score_edge_curvature(a, b)
        assert score == pytest.approx(0.5)

    def test_score_edge_length_equal(self):
        a = self._line(10, dx=1.0, noise=0.0)
        score = score_edge_length(a, a)
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_score_edge_length_range(self):
        a = self._line(10, dx=1.0, noise=0.0)
        b = self._line(10, dx=2.0, noise=0.0)
        score = score_edge_length(a, b)
        assert 0.0 <= score <= 1.0

    def test_score_edge_endpoints_same_curve(self):
        curve = self._line(10, dx=1.0, noise=0.0)
        score = score_edge_endpoints(curve, curve)
        assert 0.0 <= score <= 1.0

    def test_aggregate_edge_scores_range(self):
        score = aggregate_edge_scores(0.8, 0.7, 0.9, 0.6)
        assert 0.0 <= score <= 1.0

    def test_edge_score_result_to_dict(self):
        res = EdgeScoreResult(overlap=0.9, curvature=0.8, length=0.7,
                              endpoints=0.6, total=0.75)
        d = res.to_dict()
        assert set(d.keys()) == {"overlap", "curvature", "length", "endpoints", "total"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. event_affine_utils  (12 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEventAffineUtils:
    def _entries(self, n=6):
        levels = ["debug", "info", "warning", "error"]
        entries = []
        for i in range(n):
            level = levels[i % len(levels)]
            entries.append(make_event_record_entry(
                event_id=i,
                name=f"evt_{i % 3}",
                level=level,
                timestamp=float(i * 10),
                is_error=(level == "error"),
            ))
        return entries

    def _affine_entries(self):
        return [
            make_affine_match_entry(0, 1, score=0.9, n_inliers=50, reprojection_error=1.2),
            make_affine_match_entry(1, 2, score=0.4, n_inliers=10, reprojection_error=3.5),
            make_affine_match_entry(2, 3, score=0.7, n_inliers=30, reprojection_error=2.0,
                                    has_transform=False),
        ]

    def test_summarise_event_records_count(self):
        entries = self._entries(6)
        summary = summarise_event_record_entries(entries)
        assert summary.n_entries == 6

    def test_summarise_event_records_error_rate(self):
        entries = self._entries(4)
        summary = summarise_event_record_entries(entries)
        assert 0.0 <= summary.error_rate <= 1.0

    def test_summarise_empty_events(self):
        summary = summarise_event_record_entries([])
        assert summary.n_entries == 0
        assert summary.error_rate == 0.0

    def test_filter_error_events(self):
        entries = self._entries(8)
        errors = filter_error_events(entries)
        assert all(e.is_error for e in errors)

    def test_filter_events_by_level(self):
        entries = self._entries(8)
        warnings = filter_events_by_level(entries, "warning")
        assert all(e.level == "warning" for e in warnings)

    def test_filter_events_by_name(self):
        entries = self._entries(6)
        named = filter_events_by_name(entries, "evt_0")
        assert all(e.name == "evt_0" for e in named)

    def test_filter_events_by_time_range(self):
        entries = self._entries(6)
        in_range = filter_events_by_time_range(entries, 10.0, 30.0)
        assert all(10.0 <= e.timestamp <= 30.0 for e in in_range)

    def test_latest_event_entry(self):
        entries = self._entries(5)
        latest = latest_event_entry(entries)
        assert latest is not None
        assert latest.timestamp == max(e.timestamp for e in entries)

    def test_top_k_recent_events(self):
        entries = self._entries(6)
        top3 = top_k_recent_events(entries, 3)
        assert len(top3) == 3
        assert top3[0].timestamp >= top3[1].timestamp

    def test_summarise_affine_entries(self):
        entries = self._affine_entries()
        summary = summarise_affine_match_entries(entries)
        assert summary.n_entries == 3
        assert summary.n_no_transform == 1

    def test_filter_strong_affine_matches(self):
        entries = self._affine_entries()
        strong = filter_strong_affine_matches(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in strong)

    def test_best_affine_match_entry(self):
        entries = self._affine_entries()
        best = best_affine_match_entry(entries)
        assert best is not None
        assert best.score == max(e.score for e in entries)
