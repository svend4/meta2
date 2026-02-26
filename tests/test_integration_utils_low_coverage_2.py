"""Integration tests for under-tested utility modules (batch 2)."""
import pytest
import numpy as np

# ── contour_sampler ──────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.contour_sampler import (
    SamplerConfig, SampledContour, sample_uniform, sample_curvature,
    sample_random, sample_corners, sample_contour, normalize_contour, batch_sample,
)

def _make_contour(rng, n=50):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1)

def test_sampler_config_defaults():
    cfg = SamplerConfig()
    assert cfg.n_points == 32
    assert cfg.strategy == "uniform"

def test_sampler_config_invalid_n_points():
    with pytest.raises(ValueError):
        SamplerConfig(n_points=1)

def test_sampler_config_invalid_strategy():
    with pytest.raises(ValueError):
        SamplerConfig(strategy="invalid")

def test_sampler_config_invalid_corner_threshold():
    with pytest.raises(ValueError):
        SamplerConfig(corner_threshold=-0.1)

def test_sample_uniform_basic():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_uniform(pts, n_points=16)
    assert result.points.shape == (16, 2)
    assert result.strategy == "uniform"

def test_sample_uniform_arc_lengths_increasing():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_uniform(pts, n_points=20)
    assert np.all(np.diff(result.arc_lengths) >= 0)

def test_sample_uniform_closed():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_uniform(pts, n_points=16, closed=True)
    assert result.n_points == 16

def test_sample_uniform_invalid_n_points():
    with pytest.raises(ValueError):
        sample_uniform(np.zeros((10, 2)), n_points=1)

def test_sample_curvature_returns_correct_shape():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_curvature(pts, n_points=16)
    assert result.points.shape == (16, 2)
    assert result.strategy == "curvature"

def test_sample_random_basic():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_random(pts, n_points=16, seed=7)
    assert result.points.shape == (16, 2)
    assert result.strategy == "random"

def test_sample_random_reproducible():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    r1 = sample_random(pts, n_points=16, seed=99)
    r2 = sample_random(pts, n_points=16, seed=99)
    np.testing.assert_array_equal(r1.points, r2.points)

def test_sample_corners_basic():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_corners(pts, n_points=16)
    assert result.points.shape == (16, 2)
    assert result.strategy == "corners"

def test_sample_contour_uniform():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    cfg = SamplerConfig(n_points=12, strategy="uniform")
    result = sample_contour(pts, cfg)
    assert result.n_points == 12

def test_sample_contour_curvature():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    cfg = SamplerConfig(n_points=12, strategy="curvature")
    result = sample_contour(pts, cfg)
    assert result.n_points == 12

def test_sample_contour_random():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    cfg = SamplerConfig(n_points=12, strategy="random", seed=5)
    result = sample_contour(pts, cfg)
    assert result.n_points == 12

def test_sample_contour_corners():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    cfg = SamplerConfig(n_points=12, strategy="corners")
    result = sample_contour(pts, cfg)
    assert result.n_points == 12

def test_normalize_contour_range():
    rng = np.random.default_rng(42)
    pts = rng.uniform(10, 50, (30, 2))
    normed = normalize_contour(pts)
    assert normed.max() <= 1.0 + 1e-9
    assert normed.min() >= -1.0 - 1e-9

def test_normalize_contour_degenerate():
    pts = np.ones((10, 2)) * 5.0
    normed = normalize_contour(pts)
    assert normed.shape == (10, 2)

def test_batch_sample_length():
    rng = np.random.default_rng(42)
    contours = [_make_contour(rng) for _ in range(4)]
    results = batch_sample(contours)
    assert len(results) == 4

def test_sampled_contour_properties():
    rng = np.random.default_rng(42)
    pts = _make_contour(rng)
    result = sample_uniform(pts, n_points=20)
    assert result.n_points == 20
    assert result.n_source == len(pts)
    assert result.total_arc_length > 0

# ── curvature_utils ──────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.curvature_utils import (
    CurvatureConfig, compute_curvature, compute_total_curvature,
    find_inflection_points, compute_turning_angle, smooth_curvature,
    corner_score, find_corners, batch_curvature,
)

def _make_curve(rng, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([t, np.sin(t)], axis=1)

def test_curvature_config_defaults():
    cfg = CurvatureConfig()
    assert cfg.smooth_sigma == 1.0
    assert cfg.corner_threshold == 0.1

def test_curvature_config_invalid_threshold():
    with pytest.raises(ValueError):
        CurvatureConfig(corner_threshold=-0.1)

def test_curvature_config_invalid_min_distance():
    with pytest.raises(ValueError):
        CurvatureConfig(min_distance=0)

def test_compute_curvature_shape():
    rng = np.random.default_rng(42)
    pts = _make_curve(rng)
    kappa = compute_curvature(pts)
    assert kappa.shape == (len(pts),)

def test_compute_curvature_requires_3_points():
    with pytest.raises(ValueError):
        compute_curvature(np.zeros((2, 2)))

def test_compute_curvature_requires_correct_shape():
    with pytest.raises(ValueError):
        compute_curvature(np.zeros((10, 3)))

def test_compute_total_curvature_positive():
    rng = np.random.default_rng(42)
    pts = _make_curve(rng)
    total = compute_total_curvature(pts)
    assert total >= 0.0

def test_find_inflection_points_returns_array():
    rng = np.random.default_rng(42)
    pts = _make_curve(rng)
    infl = find_inflection_points(pts)
    assert isinstance(infl, np.ndarray)

def test_compute_turning_angle_circle():
    n = 100
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    angle = compute_turning_angle(circle)
    assert abs(abs(angle) - 2 * np.pi) < 0.5

def test_compute_turning_angle_requires_2_points():
    with pytest.raises(ValueError):
        compute_turning_angle(np.zeros((1, 2)))

def test_smooth_curvature_basic():
    kappa = np.array([0.0, 1.0, 0.5, 2.0, 0.1])
    smoothed = smooth_curvature(kappa, sigma=1.0)
    assert smoothed.shape == kappa.shape

def test_smooth_curvature_invalid_sigma():
    with pytest.raises(ValueError):
        smooth_curvature(np.ones(5), sigma=0.0)

def test_smooth_curvature_invalid_ndim():
    with pytest.raises(ValueError):
        smooth_curvature(np.ones((5, 2)), sigma=1.0)

def test_corner_score_range():
    rng = np.random.default_rng(42)
    pts = _make_curve(rng)
    scores = corner_score(pts)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-9

def test_find_corners_returns_int64():
    rng = np.random.default_rng(42)
    pts = _make_curve(rng)
    corners = find_corners(pts)
    assert corners.dtype == np.int64

def test_batch_curvature_length():
    rng = np.random.default_rng(42)
    curves = [_make_curve(rng) for _ in range(3)]
    results = batch_curvature(curves)
    assert len(results) == 3

def test_batch_curvature_empty_raises():
    with pytest.raises(ValueError):
        batch_curvature([])

# ── curve_metrics ────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.curve_metrics import (
    CurveMetricConfig, curve_l2, curve_l2_mirror, hausdorff_distance,
    frechet_distance_approx, curve_length, length_ratio,
    CurveComparisonResult, compare_curves, batch_compare_curves,
)

def _make_line(rng, n=20):
    x = np.linspace(0, 1, n)
    y = rng.uniform(0, 0.1, n)
    return np.stack([x, y], axis=1)

def test_curve_metric_config_defaults():
    cfg = CurveMetricConfig()
    assert cfg.n_samples == 64

def test_curve_metric_config_invalid_n_samples():
    with pytest.raises(ValueError):
        CurveMetricConfig(n_samples=1)

def test_curve_metric_config_invalid_eps():
    with pytest.raises(ValueError):
        CurveMetricConfig(eps=0.0)

def test_curve_l2_identical():
    rng = np.random.default_rng(42)
    pts = _make_line(rng)
    assert curve_l2(pts, pts) < 1e-9

def test_curve_l2_different():
    rng = np.random.default_rng(42)
    a = _make_line(rng)
    b = a + 1.0
    assert curve_l2(a, b) > 0.5

def test_curve_l2_mirror_reversed():
    rng = np.random.default_rng(42)
    a = _make_line(rng)
    assert curve_l2_mirror(a, a[::-1]) < curve_l2(a, a[::-1]) + 1e-9

def test_hausdorff_distance_identical():
    rng = np.random.default_rng(42)
    pts = _make_line(rng)
    assert hausdorff_distance(pts, pts) < 1e-9

def test_hausdorff_distance_positive():
    rng = np.random.default_rng(42)
    a = _make_line(rng)
    b = a + 2.0
    assert hausdorff_distance(a, b) > 1.0

def test_frechet_distance_identical():
    rng = np.random.default_rng(42)
    pts = _make_line(rng)
    cfg = CurveMetricConfig(n_samples=16)
    assert frechet_distance_approx(pts, pts, cfg) < 1e-9

def test_curve_length_basic():
    pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    assert abs(curve_length(pts) - 2.0) < 1e-9

def test_curve_length_single_point():
    pts = np.array([[5, 5]], dtype=float)
    assert curve_length(pts) == 0.0

def test_length_ratio_identical():
    rng = np.random.default_rng(42)
    a = _make_line(rng)
    assert abs(length_ratio(a, a) - 1.0) < 1e-9

def test_length_ratio_zero():
    a = np.array([[0, 0], [1, 0]], dtype=float)
    b = np.array([[0, 0], [0, 0]], dtype=float)
    assert length_ratio(a, b) == 0.0

def test_compare_curves_returns_result():
    rng = np.random.default_rng(42)
    a = _make_line(rng)
    b = _make_line(rng)
    cfg = CurveMetricConfig(n_samples=16)
    result = compare_curves(a, b, cfg)
    assert isinstance(result, CurveComparisonResult)

def test_curve_comparison_result_similarity():
    result = CurveComparisonResult(l2=0.1, hausdorff=0.2, frechet=0.3, length_ratio=0.9)
    sim = result.similarity(sigma=1.0)
    assert 0.0 <= sim <= 1.0

def test_curve_comparison_result_to_dict():
    result = CurveComparisonResult(l2=0.1, hausdorff=0.2, frechet=0.3, length_ratio=0.9)
    d = result.to_dict()
    assert "l2" in d

def test_batch_compare_curves_length():
    rng = np.random.default_rng(42)
    cfg = CurveMetricConfig(n_samples=16)
    pairs = [(_make_line(rng), _make_line(rng)) for _ in range(3)]
    results = batch_compare_curves(pairs, cfg)
    assert len(results) == 3

# ── descriptor_utils ─────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.descriptor_utils import (
    DescriptorConfig, l2_normalize, l1_normalize, batch_l2_normalize,
    l2_distance, cosine_distance, chi2_distance, l1_distance,
    descriptor_distance, pairwise_l2, pairwise_cosine,
    DescriptorMatch, nn_match, ratio_test, mean_pool, max_pool,
    vlad_encode, batch_nn_match, top_k_matches, filter_matches_by_distance,
)

def test_l2_normalize_unit_norm():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(10)
    norm = np.linalg.norm(l2_normalize(v))
    assert abs(norm - 1.0) < 1e-9

def test_l2_normalize_zero_vector():
    v = np.zeros(5)
    result = l2_normalize(v)
    assert np.allclose(result, 0)

def test_l1_normalize_sum_one():
    rng = np.random.default_rng(42)
    v = np.abs(rng.standard_normal(8))
    normed = l1_normalize(v)
    assert abs(normed.sum() - 1.0) < 1e-9

def test_batch_l2_normalize_shape():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((5, 8))
    normed = batch_l2_normalize(mat)
    assert normed.shape == mat.shape

def test_l2_distance_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert l2_distance(v, v) == 0.0

def test_cosine_distance_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(v, v) < 1e-9

def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 0.5) < 1e-9

def test_chi2_distance_identical():
    v = np.array([0.25, 0.25, 0.25, 0.25])
    assert chi2_distance(v, v) < 1e-9

def test_l1_distance_basic():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert abs(l1_distance(a, b) - 7.0) < 1e-9

def test_descriptor_distance_dispatch_l2():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert descriptor_distance(a, b, "l2") == pytest.approx(np.sqrt(2))

def test_descriptor_distance_invalid_metric():
    a = np.array([1.0])
    with pytest.raises(ValueError):
        descriptor_distance(a, a, "invalid")

def test_pairwise_l2_shape():
    rng = np.random.default_rng(42)
    mat_a = rng.standard_normal((4, 6))
    mat_b = rng.standard_normal((5, 6))
    result = pairwise_l2(mat_a, mat_b)
    assert result.shape == (4, 5)

def test_pairwise_cosine_shape():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((4, 6))
    result = pairwise_cosine(mat, mat)
    assert result.shape == (4, 4)

def test_nn_match_returns_correct_length():
    rng = np.random.default_rng(42)
    q = rng.standard_normal((3, 8))
    t = rng.standard_normal((5, 8))
    matches = nn_match(q, t)
    assert len(matches) == 3

def test_nn_match_empty_query():
    q = np.zeros((0, 8))
    t = np.ones((5, 8))
    assert nn_match(q, t) == []

def test_ratio_test_returns_list():
    rng = np.random.default_rng(42)
    q = rng.standard_normal((4, 8))
    t = rng.standard_normal((6, 8))
    matches = ratio_test(q, t, ratio=0.75)
    assert isinstance(matches, list)

def test_mean_pool_shape():
    rng = np.random.default_rng(42)
    descs = rng.standard_normal((5, 8))
    pooled = mean_pool(descs)
    assert pooled.shape == (8,)

def test_max_pool_shape():
    rng = np.random.default_rng(42)
    descs = rng.standard_normal((5, 8))
    pooled = max_pool(descs)
    assert pooled.shape == (8,)

def test_vlad_encode_shape():
    rng = np.random.default_rng(42)
    descs = rng.standard_normal((10, 4)).astype(np.float32)
    codebook = rng.standard_normal((3, 4)).astype(np.float32)
    vlad = vlad_encode(descs, codebook)
    assert vlad.shape == (12,)

def test_top_k_matches():
    matches = [DescriptorMatch(0, 1, 0.5), DescriptorMatch(1, 2, 0.1), DescriptorMatch(2, 3, 0.9)]
    top = top_k_matches(matches, 2)
    assert len(top) == 2
    assert top[0].distance <= top[1].distance

def test_filter_matches_by_distance():
    matches = [DescriptorMatch(0, 1, 0.3), DescriptorMatch(1, 2, 0.8)]
    filtered = filter_matches_by_distance(matches, 0.5)
    assert len(filtered) == 1

# ── distance_matrix ──────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.distance_matrix import (
    DistanceConfig, euclidean_distance_matrix, cosine_distance_matrix,
    manhattan_distance_matrix, build_distance_matrix, normalize_distance_matrix,
    to_similarity_matrix, threshold_distance_matrix, top_k_distance_pairs,
)

def test_distance_config_defaults():
    cfg = DistanceConfig()
    assert cfg.metric == "euclidean"

def test_distance_config_invalid_metric():
    with pytest.raises(ValueError):
        DistanceConfig(metric="minkowski")

def test_euclidean_distance_matrix_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 4))
    mat = euclidean_distance_matrix(X)
    assert mat.shape == (5, 5)

def test_euclidean_distance_matrix_diagonal_zero():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 4))
    mat = euclidean_distance_matrix(X)
    np.testing.assert_allclose(np.diag(mat), 0.0)

def test_euclidean_distance_matrix_symmetric():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 4))
    mat = euclidean_distance_matrix(X)
    np.testing.assert_allclose(mat, mat.T)

def test_cosine_distance_matrix_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 6))
    mat = cosine_distance_matrix(X)
    assert mat.shape == (4, 4)

def test_manhattan_distance_matrix_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 6))
    mat = manhattan_distance_matrix(X)
    assert mat.shape == (4, 4)

def test_build_distance_matrix_euclidean():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 4))
    cfg = DistanceConfig(metric="euclidean", normalize=False)
    mat = build_distance_matrix(X, cfg)
    assert mat.shape == (5, 5)

def test_build_distance_matrix_normalized_range():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 4))
    cfg = DistanceConfig(metric="euclidean", normalize=True)
    mat = build_distance_matrix(X, cfg)
    assert mat.max() <= 1.0 + 1e-9

def test_normalize_distance_matrix_range():
    mat = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]], dtype=float)
    normed = normalize_distance_matrix(mat)
    assert normed.max() <= 1.0 + 1e-9
    assert np.diag(normed).sum() == 0.0

def test_to_similarity_matrix_inverse():
    mat = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    sim = to_similarity_matrix(mat, method="inverse")
    np.testing.assert_allclose(np.diag(sim), 1.0)
    assert sim.min() > 0.0

def test_to_similarity_matrix_gaussian():
    mat = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    sim = to_similarity_matrix(mat, method="gaussian", sigma=1.0)
    np.testing.assert_allclose(np.diag(sim), 1.0)

def test_threshold_distance_matrix():
    mat = np.array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=float)
    result = threshold_distance_matrix(mat, threshold=2.0)
    assert result[0, 2] == 0.0
    assert result[0, 1] == 1.0

def test_top_k_distance_pairs():
    mat = np.array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=float)
    pairs = top_k_distance_pairs(mat, k=2)
    assert len(pairs) == 2
    assert pairs[0][2] <= pairs[1][2]

# ── distance_shape_utils ─────────────────────────────────────────────────────
from puzzle_reconstruction.utils.distance_shape_utils import (
    DistanceMatrixRecord, SimilarityPair, DistanceBatchResult,
    ContourMatchRecord, ShapeContextBatchSummary, MetricsRunRecord,
    EvidenceAggregationRecord, make_distance_record, make_contour_match,
)

def test_distance_matrix_record_n():
    mat = np.eye(4)
    rec = DistanceMatrixRecord(label="test", metric="l2", matrix=mat)
    assert rec.n == 4

def test_distance_matrix_record_max_value():
    mat = np.array([[0, 2], [2, 0]], dtype=float)
    rec = DistanceMatrixRecord(label="t", metric="l2", matrix=mat)
    assert rec.max_value == 2.0

def test_distance_matrix_record_invalid_shape():
    with pytest.raises(ValueError):
        DistanceMatrixRecord(label="t", metric="l2", matrix=np.ones((3, 4)))

def test_similarity_pair_is_high():
    sp = SimilarityPair(i=0, j=1, similarity=0.7)
    assert sp.is_high

def test_similarity_pair_invalid_index():
    with pytest.raises(ValueError):
        SimilarityPair(i=-1, j=0, similarity=0.5)

def test_similarity_pair_invalid_similarity():
    with pytest.raises(ValueError):
        SimilarityPair(i=0, j=1, similarity=1.5)

def test_distance_batch_result_best_pair():
    r = DistanceBatchResult(n_queries=3, metric="l2", top_pairs=[(0, 1, 0.5)])
    assert r.best_pair == (0, 1, 0.5)

def test_distance_batch_result_no_pairs():
    r = DistanceBatchResult(n_queries=3, metric="l2")
    assert r.best_pair is None

def test_contour_match_record_is_match():
    rec = ContourMatchRecord(contour_id_a=0, contour_id_b=1, cost=0.2, n_correspondences=10, similarity=0.8)
    assert rec.is_match

def test_contour_match_record_invalid_cost():
    with pytest.raises(ValueError):
        ContourMatchRecord(contour_id_a=0, contour_id_b=1, cost=-1.0, n_correspondences=5)

def test_shape_context_batch_summary_is_valid():
    s = ShapeContextBatchSummary(n_contours=5, mean_similarity=0.7)
    assert s.is_valid

def test_metrics_run_record_is_perfect():
    r = MetricsRunRecord(run_id="r1", precision=1.0, recall=1.0, f1=1.0, n_fragments=10)
    assert r.is_perfect

def test_metrics_run_record_invalid_precision():
    with pytest.raises(ValueError):
        MetricsRunRecord(run_id="r1", precision=1.5, recall=0.5, f1=0.5, n_fragments=10)

def test_evidence_aggregation_record_is_confident():
    r = EvidenceAggregationRecord(step=1, pair_id=(0, 1), n_channels=3, confidence=0.8)
    assert r.is_confident

def test_make_distance_record():
    mat = np.eye(3)
    rec = make_distance_record("test", "euclidean", mat)
    assert rec.label == "test"

def test_make_contour_match():
    rec = make_contour_match(0, 1, 0.3, 5, 0.7)
    assert rec.contour_id_a == 0

# ── edge_profile_utils ───────────────────────────────────────────────────────
from puzzle_reconstruction.utils.edge_profile_utils import (
    EdgeProfileConfig, EdgeProfile as EP_EdgeProfile,
    build_edge_profile, profile_l2_distance, profile_cosine_similarity,
    profile_correlation, resample_profile, flip_profile, mean_profile,
    batch_build_profiles, pairwise_l2_matrix, best_matching_profile,
)

def _make_edge_pts(rng, n=20):
    x = np.linspace(0, 10, n)
    y = rng.uniform(0, 1, n)
    return np.stack([x, y], axis=1)

def test_edge_profile_config_defaults():
    cfg = EdgeProfileConfig()
    assert cfg.n_samples == 64

def test_edge_profile_basic_construction():
    ep = EP_EdgeProfile(values=np.array([0.1, 0.5, 0.9]))
    assert ep.n_samples == 3

def test_edge_profile_invalid_side():
    with pytest.raises(ValueError):
        EP_EdgeProfile(values=np.zeros(5), side="diagonal")

def test_build_edge_profile_top():
    rng = np.random.default_rng(42)
    pts = _make_edge_pts(rng)
    ep = build_edge_profile(pts, side="top", cfg=EdgeProfileConfig(n_samples=16))
    assert ep.n_samples == 16

def test_build_edge_profile_left():
    rng = np.random.default_rng(42)
    pts = _make_edge_pts(rng)
    ep = build_edge_profile(pts, side="left", cfg=EdgeProfileConfig(n_samples=16))
    assert ep.n_samples == 16

def test_build_edge_profile_normalized_range():
    rng = np.random.default_rng(42)
    pts = _make_edge_pts(rng)
    ep = build_edge_profile(pts, cfg=EdgeProfileConfig(n_samples=16, normalize=True))
    assert ep.values.max() <= 1.0 + 1e-6

def test_profile_l2_distance_identical():
    ep = EP_EdgeProfile(values=np.array([0.1, 0.5, 0.9]))
    assert profile_l2_distance(ep, ep) == 0.0

def test_profile_l2_distance_mismatched():
    ep1 = EP_EdgeProfile(values=np.zeros(5))
    ep2 = EP_EdgeProfile(values=np.zeros(6))
    with pytest.raises(ValueError):
        profile_l2_distance(ep1, ep2)

def test_profile_cosine_similarity_identical():
    rng = np.random.default_rng(42)
    v = rng.uniform(0.1, 1.0, 10).astype(np.float32)
    ep = EP_EdgeProfile(values=v)
    assert abs(profile_cosine_similarity(ep, ep) - 1.0) < 1e-5

def test_profile_correlation_identical():
    rng = np.random.default_rng(42)
    v = rng.uniform(0, 1, 20).astype(np.float32)
    ep = EP_EdgeProfile(values=v)
    assert abs(profile_correlation(ep, ep) - 1.0) < 1e-5

def test_resample_profile_shape():
    ep = EP_EdgeProfile(values=np.linspace(0, 1, 16).astype(np.float32))
    resampled = resample_profile(ep, n_samples=32)
    assert resampled.n_samples == 32

def test_flip_profile():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ep = EP_EdgeProfile(values=v)
    flipped = flip_profile(ep)
    np.testing.assert_array_equal(flipped.values, v[::-1])

def test_mean_profile_shape():
    eps = [EP_EdgeProfile(values=np.random.default_rng(i).uniform(0, 1, 10).astype(np.float32)) for i in range(4)]
    mp = mean_profile(eps)
    assert mp.n_samples == 10

def test_batch_build_profiles_length():
    rng = np.random.default_rng(42)
    point_sets = [_make_edge_pts(rng) for _ in range(3)]
    profiles = batch_build_profiles(point_sets, cfg=EdgeProfileConfig(n_samples=16))
    assert len(profiles) == 3

def test_pairwise_l2_matrix_shape():
    eps = [EP_EdgeProfile(values=np.random.default_rng(i).uniform(0, 1, 10).astype(np.float32)) for i in range(4)]
    mat = pairwise_l2_matrix(eps)
    assert mat.shape == (4, 4)

def test_best_matching_profile():
    rng = np.random.default_rng(42)
    query = EP_EdgeProfile(values=np.ones(10, dtype=np.float32))
    candidates = [EP_EdgeProfile(values=(np.ones(10) * i).astype(np.float32)) for i in range(5)]
    idx, dist = best_matching_profile(query, candidates)
    assert idx == 1  # closest to ones is ones (i=1)
    assert dist >= 0.0

# ── edge_profiler ────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.edge_profiler import (
    ProfileConfig, EdgeProfile as EP2_EdgeProfile,
    compute_brightness_profile, compute_gradient_profile, compute_diff_profile,
    normalize_profile, aggregate_profiles, compare_profiles, batch_profile_edges,
)

def _make_strip(rng, h=8, w=32):
    return rng.uniform(0, 255, (h, w))

def test_profile_config_defaults():
    cfg = ProfileConfig()
    assert cfg.n_samples == 32

def test_profile_config_invalid_n_samples():
    with pytest.raises(ValueError):
        ProfileConfig(n_samples=1)

def test_profile_config_invalid_type():
    with pytest.raises(ValueError):
        ProfileConfig(profile_type="invalid")

def test_compute_brightness_profile_shape():
    rng = np.random.default_rng(42)
    strip = _make_strip(rng)
    profile = compute_brightness_profile(strip, n_samples=16)
    assert profile.shape == (16,)

def test_compute_brightness_profile_invalid_ndim():
    with pytest.raises(ValueError):
        compute_brightness_profile(np.zeros(10), n_samples=4)

def test_compute_gradient_profile_shape():
    rng = np.random.default_rng(42)
    strip = _make_strip(rng)
    profile = compute_gradient_profile(strip, n_samples=16)
    assert profile.shape == (16,)

def test_compute_diff_profile_shape():
    rng = np.random.default_rng(42)
    strip = _make_strip(rng)
    profile = compute_diff_profile(strip, n_samples=16)
    assert profile.shape == (16,)

def test_normalize_profile_range():
    rng = np.random.default_rng(42)
    p = rng.uniform(5, 20, 20)
    normed = normalize_profile(p)
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0 + 1e-9

def test_normalize_profile_constant():
    p = np.ones(10) * 5.0
    normed = normalize_profile(p)
    assert np.all(normed == 0.0)

def test_aggregate_profiles_equal_weights():
    p1 = np.array([0.0, 0.5, 1.0])
    p2 = np.array([1.0, 0.5, 0.0])
    result = aggregate_profiles([p1, p2])
    np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

def test_aggregate_profiles_custom_weights():
    p1 = np.ones(4)
    p2 = np.zeros(4)
    result = aggregate_profiles([p1, p2], weights=[1.0, 0.0])
    np.testing.assert_allclose(result, 1.0)

def test_aggregate_profiles_empty_raises():
    with pytest.raises(ValueError):
        aggregate_profiles([])

def test_compare_profiles_identical():
    p = np.array([0.1, 0.5, 0.9])
    assert compare_profiles(p, p) == pytest.approx(1.0)

def test_compare_profiles_different():
    p1 = np.zeros(10)
    p2 = np.ones(10)
    sim = compare_profiles(p1, p2)
    assert 0.0 <= sim <= 1.0

def test_batch_profile_edges_length():
    rng = np.random.default_rng(42)
    strips = [_make_strip(rng) for _ in range(3)]
    cfg = ProfileConfig(n_samples=16)
    results = batch_profile_edges(strips, cfg)
    assert len(results) == 3

# ── edge_scorer ──────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.edge_scorer import (
    EdgeScoreConfig, EdgeScoreResult, score_edge_overlap, score_edge_curvature,
    score_edge_length, score_edge_endpoints, aggregate_edge_scores,
    rank_edge_pairs, batch_score_edges,
)

def _make_edge_curve(rng, n=20):
    t = np.linspace(0, 2, n)
    y = rng.uniform(-0.1, 0.1, n)
    return np.stack([t, y], axis=1)

def test_edge_score_config_defaults():
    cfg = EdgeScoreConfig()
    assert cfg.n_samples == 64

def test_edge_score_config_invalid_n_samples():
    with pytest.raises(ValueError):
        EdgeScoreConfig(n_samples=1)

def test_edge_score_config_normalized_weights_sum_to_one():
    cfg = EdgeScoreConfig()
    assert abs(sum(cfg.normalized_weights.values()) - 1.0) < 1e-9

def test_score_edge_overlap_identical():
    rng = np.random.default_rng(42)
    curve = _make_edge_curve(rng)
    score = score_edge_overlap(curve, curve)
    assert 0.0 <= score <= 1.0

def test_score_edge_overlap_different():
    rng = np.random.default_rng(42)
    a = _make_edge_curve(rng)
    b = a + np.array([0, 100])
    score = score_edge_overlap(a, b)
    assert score < 0.5

def test_score_edge_curvature_returns_float():
    rng = np.random.default_rng(42)
    a = _make_edge_curve(rng)
    b = _make_edge_curve(rng)
    score = score_edge_curvature(a, b)
    assert 0.0 <= score <= 1.0

def test_score_edge_length_identical():
    rng = np.random.default_rng(42)
    curve = _make_edge_curve(rng)
    score = score_edge_length(curve, curve)
    assert abs(score - 1.0) < 1e-6

def test_score_edge_endpoints_returns_float():
    rng = np.random.default_rng(42)
    a = _make_edge_curve(rng)
    b = _make_edge_curve(rng)
    score = score_edge_endpoints(a, b)
    assert 0.0 <= score <= 1.0

def test_aggregate_edge_scores_range():
    cfg = EdgeScoreConfig()
    total = aggregate_edge_scores(0.8, 0.7, 0.6, 0.5, cfg)
    assert 0.0 <= total <= 1.0

def test_edge_score_result_to_dict():
    result = EdgeScoreResult(overlap=0.8, curvature=0.7, length=0.6, endpoints=0.5, total=0.7)
    d = result.to_dict()
    assert "total" in d

def test_rank_edge_pairs_sorted():
    r1 = EdgeScoreResult(total=0.3)
    r2 = EdgeScoreResult(total=0.9)
    r3 = EdgeScoreResult(total=0.5)
    ranked = rank_edge_pairs([(0, 1, r1), (1, 2, r2), (2, 3, r3)])
    assert ranked[0][2].total >= ranked[1][2].total

def test_batch_score_edges_length():
    rng = np.random.default_rng(42)
    curves_a = [_make_edge_curve(rng) for _ in range(3)]
    curves_b = [_make_edge_curve(rng) for _ in range(3)]
    results = batch_score_edges(curves_a, curves_b)
    assert len(results) == 3

def test_batch_score_edges_mismatched_raises():
    rng = np.random.default_rng(42)
    curves_a = [_make_edge_curve(rng) for _ in range(3)]
    curves_b = [_make_edge_curve(rng) for _ in range(2)]
    with pytest.raises(ValueError):
        batch_score_edges(curves_a, curves_b)

# ── event_affine_utils ───────────────────────────────────────────────────────
from puzzle_reconstruction.utils.event_affine_utils import (
    EventRecordConfig, EventRecordEntry, EventRecordSummary,
    make_event_record_entry, summarise_event_record_entries,
    filter_error_events, filter_events_by_level, filter_events_by_name,
    filter_events_by_time_range, top_k_recent_events,
    latest_event_entry, event_record_stats, compare_event_summaries,
    batch_summarise_event_record_entries,
    AffineMatchConfig, AffineMatchEntry, AffineMatchSummary,
    make_affine_match_entry, summarise_affine_match_entries,
    filter_strong_affine_matches, filter_weak_affine_matches,
    filter_affine_by_inliers, filter_affine_with_transform,
    top_k_affine_match_entries, best_affine_match_entry,
    affine_match_stats, compare_affine_summaries,
    batch_summarise_affine_match_entries,
)

def _make_events():
    return [
        make_event_record_entry(i, f"evt_{i}", "info", float(i)) for i in range(5)
    ] + [make_event_record_entry(5, "err", "error", 5.0, is_error=True)]

def test_make_event_record_entry():
    e = make_event_record_entry(0, "test", "info", 1.0)
    assert e.event_id == 0

def test_summarise_event_record_entries():
    entries = _make_events()
    summary = summarise_event_record_entries(entries)
    assert summary.n_entries == 6
    assert summary.n_errors == 1

def test_summarise_event_record_entries_empty():
    summary = summarise_event_record_entries([])
    assert summary.n_entries == 0

def test_filter_error_events():
    entries = _make_events()
    errors = filter_error_events(entries)
    assert len(errors) == 1

def test_filter_events_by_level():
    entries = _make_events()
    info = filter_events_by_level(entries, "info")
    assert all(e.level == "info" for e in info)

def test_filter_events_by_name():
    entries = _make_events()
    filtered = filter_events_by_name(entries, "evt_0")
    assert len(filtered) == 1

def test_filter_events_by_time_range():
    entries = _make_events()
    filtered = filter_events_by_time_range(entries, 1.0, 3.0)
    assert all(1.0 <= e.timestamp <= 3.0 for e in filtered)

def test_top_k_recent_events():
    entries = _make_events()
    top = top_k_recent_events(entries, k=2)
    assert len(top) == 2

def test_latest_event_entry():
    entries = _make_events()
    latest = latest_event_entry(entries)
    assert latest.timestamp == 5.0

def test_latest_event_entry_empty():
    assert latest_event_entry([]) is None

def test_event_record_stats():
    entries = _make_events()
    stats = event_record_stats(entries)
    assert "count" in stats
    assert stats["count"] == 6.0

def test_compare_event_summaries():
    entries = _make_events()
    s1 = summarise_event_record_entries(entries)
    s2 = summarise_event_record_entries(entries[:3])
    diff = compare_event_summaries(s1, s2)
    assert "error_rate_delta" in diff

def test_batch_summarise_event_record_entries():
    entries = _make_events()
    results = batch_summarise_event_record_entries([entries, entries[:2]])
    assert len(results) == 2

def _make_affine_entries():
    return [make_affine_match_entry(i, i+1, 0.5 + 0.1*i, 10 + i, 0.5 - 0.05*i) for i in range(5)]

def test_make_affine_match_entry():
    e = make_affine_match_entry(0, 1, 0.8, 12, 0.3)
    assert e.score == 0.8

def test_summarise_affine_match_entries():
    entries = _make_affine_entries()
    summary = summarise_affine_match_entries(entries)
    assert summary.n_entries == 5

def test_summarise_affine_match_entries_empty():
    summary = summarise_affine_match_entries([])
    assert summary.n_entries == 0

def test_filter_strong_affine_matches():
    entries = _make_affine_entries()
    strong = filter_strong_affine_matches(entries, threshold=0.7)
    assert all(e.score >= 0.7 for e in strong)

def test_filter_weak_affine_matches():
    entries = _make_affine_entries()
    weak = filter_weak_affine_matches(entries, threshold=0.6)
    assert all(e.score < 0.6 for e in weak)

def test_filter_affine_by_inliers():
    entries = _make_affine_entries()
    filtered = filter_affine_by_inliers(entries, min_inliers=12)
    assert all(e.n_inliers >= 12 for e in filtered)

def test_filter_affine_with_transform():
    entries = _make_affine_entries()
    with_t = filter_affine_with_transform(entries)
    assert all(e.has_transform for e in with_t)

def test_top_k_affine_match_entries():
    entries = _make_affine_entries()
    top = top_k_affine_match_entries(entries, k=2)
    assert len(top) == 2
    assert top[0].score >= top[1].score

def test_best_affine_match_entry():
    entries = _make_affine_entries()
    best = best_affine_match_entry(entries)
    assert best is not None
    assert all(best.score >= e.score for e in entries)

def test_affine_match_stats():
    entries = _make_affine_entries()
    stats = affine_match_stats(entries)
    assert "mean" in stats

def test_compare_affine_summaries():
    entries = _make_affine_entries()
    s1 = summarise_affine_match_entries(entries)
    s2 = summarise_affine_match_entries(entries[:2])
    diff = compare_affine_summaries(s1, s2)
    assert "mean_score_delta" in diff

def test_batch_summarise_affine_match_entries():
    entries = _make_affine_entries()
    results = batch_summarise_affine_match_entries([entries, entries[:2]])
    assert len(results) == 2

# ── filter_pipeline_utils ────────────────────────────────────────────────────
from puzzle_reconstruction.utils.filter_pipeline_utils import (
    FilterStepConfig, FilterStepResult, FilterPipelineSummary,
    make_filter_step, steps_from_log, summarise_pipeline,
    filter_effective_steps, filter_by_removal_rate,
    most_aggressive_step, least_aggressive_step,
    pipeline_stats, compare_pipelines, batch_summarise_pipelines,
)

def _make_steps():
    return [
        make_filter_step("threshold", 100, 70),
        make_filter_step("nms", 70, 50),
        make_filter_step("size", 50, 45),
    ]

def test_filter_step_config_defaults():
    cfg = FilterStepConfig()
    assert cfg.name == "threshold"

def test_filter_step_config_invalid_name():
    with pytest.raises(ValueError):
        FilterStepConfig(name="")

def test_filter_step_config_invalid_threshold():
    with pytest.raises(ValueError):
        FilterStepConfig(threshold=1.5)

def test_make_filter_step_removal():
    step = make_filter_step("test", 100, 60)
    assert step.n_removed == 40

def test_filter_step_removal_rate():
    step = make_filter_step("test", 100, 60)
    assert abs(step.removal_rate - 0.4) < 1e-9

def test_steps_from_log():
    log = [{"step_name": "s1", "n_input": 100, "n_output": 80}]
    steps = steps_from_log(log)
    assert len(steps) == 1
    assert steps[0].step_name == "s1"

def test_summarise_pipeline_basic():
    steps = _make_steps()
    summary = summarise_pipeline(steps)
    assert summary.n_initial == 100
    assert summary.n_final == 45

def test_summarise_pipeline_empty():
    summary = summarise_pipeline([])
    assert summary.n_initial == 0

def test_filter_effective_steps():
    steps = _make_steps()
    effective = filter_effective_steps(steps)
    assert all(s.n_removed > 0 for s in effective)

def test_filter_by_removal_rate():
    steps = _make_steps()
    filtered = filter_by_removal_rate(steps, min_rate=0.1)
    assert all(s.removal_rate >= 0.1 for s in filtered)

def test_most_aggressive_step():
    steps = _make_steps()
    aggressive = most_aggressive_step(steps)
    assert aggressive.n_removed == max(s.n_removed for s in steps)

def test_least_aggressive_step():
    steps = _make_steps()
    mild = least_aggressive_step(steps)
    assert mild.n_removed == min(s.n_removed for s in steps)

def test_pipeline_stats():
    steps = _make_steps()
    stats = pipeline_stats(steps)
    assert stats["n_steps"] == 3

def test_compare_pipelines():
    steps = _make_steps()
    s1 = summarise_pipeline(steps)
    s2 = summarise_pipeline(steps[:2])
    diff = compare_pipelines(s1, s2)
    assert "n_steps_delta" in diff

def test_batch_summarise_pipelines():
    log1 = [{"step_name": "s1", "n_input": 100, "n_output": 80}]
    log2 = [{"step_name": "s2", "n_input": 50, "n_output": 30}]
    results = batch_summarise_pipelines([log1, log2])
    assert len(results) == 2

# ── fragment_filter_utils ────────────────────────────────────────────────────
from puzzle_reconstruction.utils.fragment_filter_utils import (
    FragmentFilterConfig, FragmentQuality, compute_fragment_area,
    compute_aspect_ratio, compute_fill_ratio, evaluate_fragment,
    deduplicate_fragments, filter_fragments, sort_by_area,
    top_k_fragments, fragment_quality_summary,
)

def _make_mask(rng, h=20, w=30):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:15, 5:25] = 1
    return mask

def test_fragment_filter_config_defaults():
    cfg = FragmentFilterConfig()
    assert cfg.min_area == 0.0

def test_fragment_filter_config_invalid_min_area():
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_area=-1.0)

def test_fragment_filter_config_invalid_fill_ratio():
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_fill_ratio=1.5)

def test_compute_fragment_area():
    rng = np.random.default_rng(42)
    mask = _make_mask(rng)
    area = compute_fragment_area(mask)
    assert area == float(np.count_nonzero(mask))

def test_compute_aspect_ratio_range():
    rng = np.random.default_rng(42)
    mask = _make_mask(rng)
    ar = compute_aspect_ratio(mask)
    assert 0.0 < ar <= 1.0

def test_compute_fill_ratio_range():
    rng = np.random.default_rng(42)
    mask = _make_mask(rng)
    fill = compute_fill_ratio(mask)
    assert 0.0 < fill <= 1.0

def test_compute_fill_ratio_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert compute_fill_ratio(mask) == 1.0

def test_evaluate_fragment_passes():
    rng = np.random.default_rng(42)
    mask = _make_mask(rng)
    cfg = FragmentFilterConfig(min_area=10.0)
    q = evaluate_fragment(0, mask, cfg)
    assert q.passed

def test_evaluate_fragment_area_too_small():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    cfg = FragmentFilterConfig(min_area=100.0)
    q = evaluate_fragment(0, mask, cfg)
    assert not q.passed
    assert q.reject_reason == "area_too_small"

def test_deduplicate_fragments_removes_duplicates():
    img = np.zeros((5, 5), dtype=np.uint8)
    frags = [(0, img), (1, img), (2, img.copy())]
    result = deduplicate_fragments(frags)
    assert len(result) <= 2

def test_filter_fragments_basic():
    rng = np.random.default_rng(42)
    mask = _make_mask(rng)
    img = np.zeros((20, 30), dtype=np.uint8)
    frags = [(i, img, mask) for i in range(4)]
    cfg = FragmentFilterConfig(min_area=5.0, deduplicate=False)
    kept, qualities = filter_fragments(frags, cfg)
    assert len(qualities) == 4

def test_sort_by_area():
    rng = np.random.default_rng(42)
    mask1 = _make_mask(rng, h=20, w=30)
    mask2 = np.ones((5, 5), dtype=np.uint8)
    img = np.zeros((20, 30), dtype=np.uint8)
    frags = [(0, img, mask1), (1, img[:5, :5], mask2)]
    sorted_frags = sort_by_area(frags, descending=True)
    a0 = compute_fragment_area(sorted_frags[0][2])
    a1 = compute_fragment_area(sorted_frags[1][2])
    assert a0 >= a1

def test_top_k_fragments():
    rng = np.random.default_rng(42)
    frags = []
    for i in range(5):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:i+1, :i+1] = 1
        frags.append((i, np.zeros((10, 10), dtype=np.uint8), mask))
    top = top_k_fragments(frags, k=2)
    assert len(top) == 2

def test_fragment_quality_summary():
    qualities = [
        FragmentQuality(0, 100.0, 0.8, 0.9, passed=True),
        FragmentQuality(1, 5.0, 0.5, 0.7, passed=False, reject_reason="area_too_small"),
    ]
    summary = fragment_quality_summary(qualities)
    assert summary["total"] == 2
    assert summary["passed"] == 1

# ── fragment_stats ───────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.fragment_stats import (
    FragmentMetrics, CollectionStats, compute_fragment_metrics,
    compute_collection_stats, area_histogram, compare_collections,
    outlier_indices,
)

def _make_fm_mask(h=30, w=40):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:25, 5:35] = 1
    return mask

def test_fragment_metrics_construction():
    fm = FragmentMetrics(fragment_id=0, area=100.0, aspect=1.5, density=0.8, n_edges=4, perimeter=50.0)
    assert fm.fragment_id == 0

def test_fragment_metrics_invalid_id():
    with pytest.raises(ValueError):
        FragmentMetrics(fragment_id=-1, area=100.0, aspect=1.5, density=0.8, n_edges=4, perimeter=50.0)

def test_fragment_metrics_invalid_density():
    with pytest.raises(ValueError):
        FragmentMetrics(fragment_id=0, area=100.0, aspect=1.5, density=1.5, n_edges=4, perimeter=50.0)

def test_compute_fragment_metrics_basic():
    mask = _make_fm_mask()
    fm = compute_fragment_metrics(0, mask, n_edges=4)
    assert fm.area > 0
    assert fm.aspect > 0
    assert 0.0 < fm.density <= 1.0

def test_compute_fragment_metrics_invalid_ndim():
    with pytest.raises(ValueError):
        compute_fragment_metrics(0, np.zeros(10))

def test_compute_collection_stats_basic():
    masks = [_make_fm_mask() for _ in range(5)]
    metrics = [compute_fragment_metrics(i, m) for i, m in enumerate(masks)]
    stats = compute_collection_stats(metrics)
    assert stats.n_fragments == 5

def test_compute_collection_stats_empty_raises():
    with pytest.raises(ValueError):
        compute_collection_stats([])

def test_collection_stats_to_dict():
    masks = [_make_fm_mask() for _ in range(3)]
    metrics = [compute_fragment_metrics(i, m) for i, m in enumerate(masks)]
    stats = compute_collection_stats(metrics)
    d = stats.to_dict()
    assert "n_fragments" in d

def test_area_histogram_basic():
    masks = [_make_fm_mask() for _ in range(5)]
    metrics = [compute_fragment_metrics(i, m) for i, m in enumerate(masks)]
    counts, edges = area_histogram(metrics, n_bins=5)
    assert len(counts) == 5
    assert len(edges) == 6

def test_area_histogram_normalized():
    masks = [_make_fm_mask() for _ in range(5)]
    metrics = [compute_fragment_metrics(i, m) for i, m in enumerate(masks)]
    counts, _ = area_histogram(metrics, n_bins=5, normalize=True)
    assert abs(counts.sum() - 1.0) < 1e-9

def test_compare_collections():
    mask = _make_fm_mask()
    metrics1 = [compute_fragment_metrics(i, mask) for i in range(3)]
    mask2 = np.ones((10, 10), dtype=np.uint8)
    metrics2 = [compute_fragment_metrics(i, mask2) for i in range(3)]
    s1 = compute_collection_stats(metrics1)
    s2 = compute_collection_stats(metrics2)
    diff = compare_collections(s1, s2)
    assert "delta_total_area" in diff

def test_outlier_indices_by_area():
    masks = [_make_fm_mask() for _ in range(8)]
    # Add one outlier with huge area
    huge = np.ones((100, 100), dtype=np.uint8)
    metrics = [compute_fragment_metrics(i, m) for i, m in enumerate(masks)]
    metrics.append(compute_fragment_metrics(8, huge))
    outliers = outlier_indices(metrics, z_threshold=2.0, by="area")
    assert len(outliers) > 0

def test_outlier_indices_invalid_threshold():
    mask = _make_fm_mask()
    metrics = [compute_fragment_metrics(i, mask) for i in range(3)]
    with pytest.raises(ValueError):
        outlier_indices(metrics, z_threshold=0.0)

def test_outlier_indices_invalid_by():
    mask = _make_fm_mask()
    metrics = [compute_fragment_metrics(i, mask) for i in range(3)]
    with pytest.raises(ValueError):
        outlier_indices(metrics, by="invalid_field")

def test_outlier_indices_single_fragment():
    mask = _make_fm_mask()
    metrics = [compute_fragment_metrics(0, mask)]
    result = outlier_indices(metrics)
    assert result == []
