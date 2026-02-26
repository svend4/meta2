"""Integration tests for puzzle_reconstruction/algorithms descriptor modules.

Covers:
    - boundary_descriptor
    - edge_comparator
    - edge_extractor
    - edge_scorer
    - fourier_descriptor
    - fragment_aligner
    - fragment_classifier
    - shape_context
    - wavelet_descriptor
    - zernike_descriptor
    - descriptor_aggregator
    - descriptor_combiner
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle(n=50, r=40.0):
    """Return (n, 2) array of circle points."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)]).astype(np.float64)


def _make_line_pts(n=20):
    """Return (n, 2) array of roughly straight points with slight noise."""
    x = np.linspace(0, 100, n)
    y = np.random.RandomState(0).randn(n) * 0.5
    return np.column_stack([x, y]).astype(np.float64)


def _make_gray_image(h=64, w=64, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.rand(h, w) * 200 + 20).astype(np.uint8)


def _make_bgr_image(h=64, w=64, seed=7):
    rng = np.random.RandomState(seed)
    return (rng.rand(h, w, 3) * 200 + 20).astype(np.uint8)


# ===========================================================================
# 1. boundary_descriptor
# ===========================================================================

from puzzle_reconstruction.algorithms.boundary_descriptor import (
    DescriptorConfig,
    BoundaryDescriptor,
    compute_curvature,
    curvature_histogram,
    direction_histogram,
    chord_distribution,
    extract_descriptor,
    descriptor_similarity,
    batch_extract_descriptors,
)


class TestBoundaryDescriptor:
    def test_descriptor_config_defaults(self):
        cfg = DescriptorConfig()
        assert cfg.n_bins == 32
        assert cfg.normalize is True

    def test_descriptor_config_invalid_bins(self):
        with pytest.raises(ValueError):
            DescriptorConfig(n_bins=2)

    def test_descriptor_config_invalid_sigma(self):
        with pytest.raises(ValueError):
            DescriptorConfig(smooth_sigma=-1.0)

    def test_compute_curvature_basic(self):
        pts = _make_circle(n=60)
        curv = compute_curvature(pts, smooth_sigma=0.0)
        assert curv.shape == (60,)
        assert curv.dtype == np.float32

    def test_compute_curvature_too_few_points(self):
        with pytest.raises(ValueError):
            compute_curvature(np.array([[0, 0], [1, 1]]))

    def test_curvature_histogram_shape(self):
        curv = compute_curvature(_make_circle(n=60))
        h = curvature_histogram(curv, n_bins=16)
        assert h.shape == (16,)
        assert abs(h.sum() - 1.0) < 1e-5  # normalized

    def test_direction_histogram_basic(self):
        pts = _make_circle(n=40)
        h = direction_histogram(pts, n_bins=8)
        assert h.shape == (8,)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_chord_distribution_basic(self):
        pts = _make_circle(n=30)
        h = chord_distribution(pts, n_bins=10)
        assert h.shape == (10,)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_extract_descriptor_returns_boundary_descriptor(self):
        pts = _make_circle(n=50)
        desc = extract_descriptor(pts, fragment_id=1, edge_id=2)
        assert isinstance(desc, BoundaryDescriptor)
        assert desc.fragment_id == 1
        assert desc.edge_id == 2
        assert desc.length > 0

    def test_extract_descriptor_feature_vector(self):
        pts = _make_circle(n=50)
        desc = extract_descriptor(pts)
        fv = desc.feature_vector
        assert fv.ndim == 1
        assert len(fv) == 3 * desc.n_bins

    def test_descriptor_similarity_same(self):
        pts = _make_circle(n=50)
        d = extract_descriptor(pts)
        sim = descriptor_similarity(d, d)
        assert 0.0 <= sim <= 1.0
        assert sim > 0.9  # identical should be near 1

    def test_descriptor_similarity_different(self):
        d1 = extract_descriptor(_make_circle(n=50))
        d2 = extract_descriptor(_make_line_pts(n=20))
        sim = descriptor_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_batch_extract_descriptors(self):
        pts_list = [_make_circle(n=30), _make_circle(n=40)]
        descs = batch_extract_descriptors(pts_list, fragment_id=5)
        assert len(descs) == 2
        assert descs[0].edge_id == 0
        assert descs[1].edge_id == 1


# ===========================================================================
# 2. edge_comparator
# ===========================================================================

from puzzle_reconstruction.algorithms.edge_comparator import (
    CompareConfig,
    EdgeCompareResult,
    dtw_distance,
    css_similarity,
    fd_score,
    ifs_similarity,
    compare_edges,
    build_compat_matrix,
    top_k_matches,
)
from puzzle_reconstruction.models import EdgeSignature, EdgeSide


def _make_edge_signature(edge_id=0, n=10, seed=0):
    rng = np.random.RandomState(seed)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=rng.rand(n, 2).astype(np.float64),
        fd=1.2 + rng.rand() * 0.3,
        css_vec=rng.rand(20).astype(np.float32),
        ifs_coeffs=rng.rand(8).astype(np.float32),
        length=float(rng.rand() * 100 + 50),
    )


class TestEdgeComparator:
    def test_compare_config_defaults(self):
        cfg = CompareConfig()
        assert cfg.w_dtw >= 0
        assert cfg.fd_sigma > 0

    def test_dtw_distance_basic(self):
        a = np.random.rand(8, 2)
        b = np.random.rand(8, 2)
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_dtw_distance_identical(self):
        a = np.random.rand(6, 2)
        assert dtw_distance(a, a) == 0.0

    def test_css_similarity_identical(self):
        v = np.random.rand(20).astype(np.float32)
        assert css_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_css_similarity_range(self):
        v1 = np.random.rand(15)
        v2 = np.random.rand(15)
        s = css_similarity(v1, v2)
        assert 0.0 <= s <= 1.0

    def test_fd_score_identical(self):
        assert fd_score(1.5, 1.5) == pytest.approx(1.0)

    def test_fd_score_far(self):
        assert fd_score(1.0, 5.0) < 0.01

    def test_ifs_similarity_range(self):
        a = np.random.rand(8)
        b = np.random.rand(8)
        s = ifs_similarity(a, b)
        assert 0.0 <= s <= 1.0

    def test_compare_edges_result(self):
        e1 = _make_edge_signature(edge_id=0)
        e2 = _make_edge_signature(edge_id=1, seed=1)
        result = compare_edges(e1, e2)
        assert isinstance(result, EdgeCompareResult)
        assert 0.0 <= result.score <= 1.0

    def test_build_compat_matrix_shape(self):
        edges = [_make_edge_signature(i, seed=i) for i in range(3)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (3, 3)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_top_k_matches(self):
        query = _make_edge_signature(0, seed=10)
        candidates = [_make_edge_signature(i, seed=i) for i in range(5)]
        results = top_k_matches(query, candidates, k=3)
        assert len(results) == 3
        # Sorted descending by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 3. edge_extractor
# ===========================================================================

from puzzle_reconstruction.algorithms.edge_extractor import (
    EdgeSegment,
    FragmentEdges,
    detect_boundary,
    extract_edge_points,
    split_edge_by_side,
    compute_edge_length,
    simplify_edge,
    extract_fragment_edges,
    batch_extract_edges,
)


class TestEdgeExtractor:
    def test_detect_boundary_returns_mask(self):
        img = _make_gray_image()
        mask = detect_boundary(img)
        assert mask.shape == img.shape
        assert mask.dtype == np.uint8

    def test_detect_boundary_bgr(self):
        img = _make_bgr_image()
        mask = detect_boundary(img)
        assert mask.ndim == 2

    def test_extract_edge_points_shape(self):
        img = _make_gray_image()
        mask = detect_boundary(img)
        pts = extract_edge_points(mask)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_compute_edge_length_basic(self):
        pts = np.array([[0, 0], [3, 4]], dtype=np.float32)
        length = compute_edge_length(pts)
        assert length == pytest.approx(5.0)

    def test_compute_edge_length_single_point(self):
        pts = np.array([[1, 2]], dtype=np.float32)
        assert compute_edge_length(pts) == 0.0

    def test_split_edge_by_side_keys(self):
        pts = np.array([[5, 0], [5, 63], [0, 5], [63, 5]], dtype=np.float32)
        result = split_edge_by_side(pts, (64, 64))
        assert set(result.keys()) == {"top", "bottom", "left", "right"}

    def test_simplify_edge_reduces_points(self):
        pts = np.random.rand(100, 2).astype(np.float32) * 50
        simplified = simplify_edge(pts, epsilon=2.0)
        assert simplified.shape[1] == 2
        assert len(simplified) <= len(pts)

    def test_extract_fragment_edges_returns_fragment_edges(self):
        img = _make_gray_image(h=80, w=80)
        fe = extract_fragment_edges(img)
        assert isinstance(fe, FragmentEdges)
        assert fe.n_segments == 4

    def test_batch_extract_edges(self):
        images = [_make_gray_image(seed=i) for i in range(3)]
        result = batch_extract_edges(images)
        assert len(result) == 3


# ===========================================================================
# 4. edge_scorer
# ===========================================================================

from puzzle_reconstruction.algorithms.edge_scorer import (
    EdgeScore,
    score_color_compat,
    score_gradient_compat,
    score_texture_compat,
    score_edge_pair,
    batch_score_edges,
)


class TestEdgeScorer:
    def test_score_color_compat_range(self):
        img1 = _make_gray_image()
        img2 = _make_gray_image(seed=99)
        s = score_color_compat(img1, img2)
        assert 0.0 <= s <= 1.0

    def test_score_color_compat_same_strip(self):
        # Same strip on same side should give score close to 1.0
        img = _make_gray_image()
        s = score_color_compat(img, img, side1=0, side2=0)
        assert s > 0.99

    def test_score_gradient_compat_range(self):
        img1 = _make_gray_image()
        img2 = _make_gray_image(seed=13)
        s = score_gradient_compat(img1, img2)
        assert 0.0 <= s <= 1.0

    def test_score_texture_compat_same(self):
        img = _make_gray_image()
        s = score_texture_compat(img, img)
        assert s > 0.9

    def test_score_edge_pair_returns_edge_score(self):
        img1 = _make_gray_image()
        img2 = _make_gray_image(seed=2)
        es = score_edge_pair(img1, img2)
        assert isinstance(es, EdgeScore)
        assert 0.0 <= es.total_score <= 1.0

    def test_score_edge_pair_channels(self):
        img1 = _make_bgr_image()
        img2 = _make_bgr_image(seed=3)
        es = score_edge_pair(img1, img2, idx1=0, idx2=1, side1=2, side2=0)
        assert 0.0 <= es.color_score <= 1.0
        assert 0.0 <= es.gradient_score <= 1.0
        assert 0.0 <= es.texture_score <= 1.0

    def test_batch_score_edges_length(self):
        images = [_make_gray_image(seed=i) for i in range(3)]
        pairs = [(0, 1), (1, 2)]
        results = batch_score_edges(images, pairs)
        assert len(results) == 2


# ===========================================================================
# 5. fourier_descriptor
# ===========================================================================

from puzzle_reconstruction.algorithms.fourier_descriptor import (
    FourierConfig,
    FourierDescriptor,
    complex_representation,
    compute_contour_centroid,
    compute_fd,
    fd_similarity,
    batch_compute_fd,
    rank_by_fd,
)


class TestFourierDescriptor:
    def test_fourier_config_defaults(self):
        cfg = FourierConfig()
        assert cfg.n_coeffs >= 4

    def test_fourier_config_invalid(self):
        with pytest.raises(ValueError):
            FourierConfig(n_coeffs=2)

    def test_complex_representation(self):
        pts = _make_circle(n=20)
        z = complex_representation(pts)
        assert z.shape == (20,)
        assert np.iscomplexobj(z)

    def test_compute_contour_centroid(self):
        pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        cx, cy = compute_contour_centroid(pts)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(1.0)

    def test_compute_fd_basic(self):
        pts = _make_circle(n=64)
        desc = compute_fd(pts)
        assert isinstance(desc, FourierDescriptor)
        assert desc.coefficients.ndim == 1
        assert len(desc.coefficients) == 2 * desc.n_coeffs

    def test_compute_fd_too_few_points(self):
        with pytest.raises(ValueError):
            compute_fd(np.random.rand(3, 2))

    def test_fd_similarity_same(self):
        pts = _make_circle(n=64)
        d = compute_fd(pts)
        assert fd_similarity(d, d) == pytest.approx(1.0, abs=1e-4)

    def test_fd_magnitude_property(self):
        pts = _make_circle(n=64)
        d = compute_fd(pts)
        mag = d.magnitude
        assert mag.shape == (d.n_coeffs,)
        assert (mag >= 0).all()

    def test_batch_compute_fd(self):
        pts_list = [_make_circle(n=40), _make_circle(n=50)]
        descs = batch_compute_fd(pts_list)
        assert len(descs) == 2

    def test_rank_by_fd(self):
        query = compute_fd(_make_circle(n=64))
        candidates = [compute_fd(_make_circle(n=64, r=r)) for r in [30, 40, 50]]
        ranked = rank_by_fd(query, candidates)
        assert len(ranked) == 3
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 6. fragment_aligner
# ===========================================================================

from puzzle_reconstruction.algorithms.fragment_aligner import (
    AlignmentResult,
    estimate_shift,
    phase_correlation_align,
    template_match_align,
    apply_shift,
    batch_align,
)


class TestFragmentAligner:
    def test_estimate_shift_basic(self):
        rng = np.random.RandomState(0)
        strip = rng.rand(64).astype(np.float32)
        shift, conf = estimate_shift(strip, strip)
        assert isinstance(shift, float)
        assert 0.0 <= conf <= 1.0

    def test_estimate_shift_returns_near_zero_for_identical(self):
        rng = np.random.RandomState(1)
        strip = rng.rand(64).astype(np.float32)
        shift, conf = estimate_shift(strip, strip)
        assert abs(shift) < 2.0  # should be near 0

    def test_phase_correlation_align(self):
        img1 = _make_gray_image(h=80, w=80)
        img2 = _make_gray_image(h=80, w=80, seed=5)
        result = phase_correlation_align(img1, img2)
        assert isinstance(result, AlignmentResult)
        assert result.method == "phase"
        assert 0.0 <= result.confidence <= 1.0

    def test_template_match_align(self):
        img1 = _make_gray_image(h=80, w=80)
        img2 = _make_gray_image(h=80, w=80, seed=6)
        result = template_match_align(img1, img2)
        assert isinstance(result, AlignmentResult)
        assert result.method == "template"

    def test_apply_shift_shape_preserved(self):
        img = _make_gray_image()
        shifted = apply_shift(img, dx=5.0, dy=3.0)
        assert shifted.shape == img.shape

    def test_apply_shift_zero_unchanged(self):
        img = _make_gray_image()
        shifted = apply_shift(img, dx=0.0, dy=0.0)
        assert np.allclose(shifted, img)

    def test_batch_align_phase(self):
        images = [_make_gray_image(seed=i, h=80, w=80) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 2, 2, 0)]
        results = batch_align(images, pairs, method="phase")
        assert len(results) == 2
        assert all(r.method == "phase" for r in results)

    def test_batch_align_invalid_method(self):
        with pytest.raises(ValueError):
            batch_align([], [], method="unknown")


# ===========================================================================
# 7. fragment_classifier
# ===========================================================================

from puzzle_reconstruction.algorithms.fragment_classifier import (
    FragmentType,
    FragmentFeatures,
    ClassifyResult,
    compute_texture_features,
    compute_edge_features,
    compute_shape_features,
    detect_text_presence,
    classify_fragment_type,
    classify_fragment,
    batch_classify,
)


class TestFragmentClassifier:
    def test_fragment_type_values(self):
        assert FragmentType.CORNER.value == "corner"
        assert FragmentType.INNER.value == "inner"

    def test_compute_texture_features(self):
        gray = _make_gray_image()
        tv, lbp = compute_texture_features(gray)
        assert tv >= 0.0
        assert 0.0 <= lbp <= 1.0

    def test_compute_edge_features(self):
        gray = _make_gray_image()
        densities, straightness = compute_edge_features(gray)
        assert len(densities) == 4
        assert len(straightness) == 4
        assert all(d >= 0 for d in densities)

    def test_compute_shape_features(self):
        gray = _make_gray_image()
        asp, fill, angle = compute_shape_features(gray)
        assert asp > 0
        assert 0.0 <= fill <= 1.0

    def test_detect_text_presence(self):
        gray = _make_gray_image()
        has_text, density, n_rows = detect_text_presence(gray)
        assert isinstance(has_text, bool)
        assert 0.0 <= density <= 1.0
        assert n_rows >= 0

    def test_classify_fragment_type_inner(self):
        # Near-zero densities → inner fragment
        ftype, conf, sides = classify_fragment_type(
            (0.001, 0.001, 0.001, 0.001),
            (0.001, 0.001, 0.001, 0.001),
            1.0,
        )
        assert ftype == FragmentType.INNER

    def test_classify_fragment(self):
        img = _make_gray_image(h=80, w=80)
        result = classify_fragment(img)
        assert isinstance(result, ClassifyResult)
        assert isinstance(result.fragment_type, FragmentType)
        assert 0.0 <= result.confidence <= 1.0

    def test_fragment_features_as_vector(self):
        ff = FragmentFeatures()
        v = ff.as_vector()
        # 4 edge_densities + 4 edge_straightness + 6 scalars = 14 elements
        assert v.ndim == 1
        assert len(v) > 0
        assert v.dtype == np.float32

    def test_batch_classify(self):
        images = [_make_gray_image(seed=i) for i in range(3)]
        results = batch_classify(images)
        assert len(results) == 3


# ===========================================================================
# 8. shape_context
# ===========================================================================

from puzzle_reconstruction.algorithms.shape_context import (
    ShapeContextResult,
    compute_shape_context,
    shape_context_distance,
    normalize_shape_context,
    match_shape_contexts,
    contour_similarity,
    log_polar_histogram,
)


class TestShapeContext:
    def test_compute_shape_context_basic(self):
        pts = _make_circle(n=20)
        result = compute_shape_context(pts)
        assert isinstance(result, ShapeContextResult)
        assert result.descriptors.shape == (20, result.n_bins_r * result.n_bins_theta)

    def test_shape_context_distance_zero_for_same(self):
        desc = np.array([0.1, 0.2, 0.3, 0.4])
        d = shape_context_distance(desc, desc)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_shape_context_distance_range(self):
        d1 = np.random.rand(20)
        d2 = np.random.rand(20)
        d1 /= d1.sum()
        d2 /= d2.sum()
        dist = shape_context_distance(d1, d2)
        assert 0.0 <= dist <= 0.5 + 1e-6

    def test_normalize_shape_context(self):
        sc = np.array([1.0, 2.0, 3.0, 4.0])
        n = normalize_shape_context(sc)
        assert n.sum() == pytest.approx(1.0)

    def test_normalize_shape_context_zeros(self):
        sc = np.zeros(5)
        n = normalize_shape_context(sc)
        assert (n == 0).all()

    def test_match_shape_contexts(self):
        pts_a = _make_circle(n=15)
        pts_b = _make_circle(n=15)
        sc_a = compute_shape_context(pts_a)
        sc_b = compute_shape_context(pts_b)
        cost, corr = match_shape_contexts(sc_a, sc_b)
        assert cost >= 0.0
        assert corr.shape[1] == 2

    def test_contour_similarity_same(self):
        pts = _make_circle(n=30)
        sim = contour_similarity(pts, pts)
        assert 0.0 <= sim <= 1.0

    def test_contour_similarity_different(self):
        pts_a = _make_circle(n=30)
        pts_b = _make_line_pts(n=30)
        sim = contour_similarity(pts_a, pts_b)
        assert 0.0 <= sim <= 1.0


# ===========================================================================
# 9. wavelet_descriptor
# ===========================================================================

from puzzle_reconstruction.algorithms.wavelet_descriptor import (
    WaveletDescriptor,
    compute_wavelet_descriptor,
    wavelet_similarity,
    wavelet_similarity_mirror,
    batch_wavelet_similarity,
)


class TestWaveletDescriptor:
    def test_compute_wavelet_descriptor_basic(self):
        pts = _make_circle(n=64)
        wd = compute_wavelet_descriptor(pts)
        assert isinstance(wd, WaveletDescriptor)
        assert wd.coeffs.ndim == 1
        assert wd.n_levels > 0

    def test_compute_wavelet_descriptor_short(self):
        pts = np.array([[0, 0]], dtype=float)
        wd = compute_wavelet_descriptor(pts)
        assert isinstance(wd, WaveletDescriptor)

    def test_energy_per_level_sums_to_one(self):
        pts = _make_circle(n=64)
        wd = compute_wavelet_descriptor(pts)
        assert wd.energy_per_level.sum() == pytest.approx(1.0, abs=1e-5)

    def test_wavelet_similarity_same(self):
        pts = _make_circle(n=64)
        wd = compute_wavelet_descriptor(pts)
        sim = wavelet_similarity(wd, wd)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_wavelet_similarity_range(self):
        wd1 = compute_wavelet_descriptor(_make_circle(n=64))
        wd2 = compute_wavelet_descriptor(_make_line_pts(n=64))
        sim = wavelet_similarity(wd1, wd2)
        assert 0.0 <= sim <= 1.0

    def test_wavelet_similarity_mirror(self):
        wd1 = compute_wavelet_descriptor(_make_circle(n=64))
        wd2 = compute_wavelet_descriptor(_make_circle(n=64))
        sim = wavelet_similarity_mirror(wd1, wd2)
        assert 0.0 <= sim <= 1.0

    def test_batch_wavelet_similarity(self):
        query = compute_wavelet_descriptor(_make_circle(n=64))
        candidates = [compute_wavelet_descriptor(_make_circle(n=64, r=r))
                      for r in [20, 30, 40]]
        scores = batch_wavelet_similarity(query, candidates)
        assert scores.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in scores)


# ===========================================================================
# 10. zernike_descriptor
# ===========================================================================

from puzzle_reconstruction.algorithms.zernike_descriptor import (
    ZernikeDescriptor,
    zernike_moments,
    zernike_similarity,
    zernike_to_feature_vector,
)


class TestZernikeDescriptor:
    def test_zernike_moments_basic(self):
        pts = _make_circle(n=64)
        zd = zernike_moments(pts, order=6)
        assert isinstance(zd, ZernikeDescriptor)
        assert zd.moments.dtype == complex
        assert len(zd.magnitudes) == len(zd.moments)

    def test_zernike_moments_short_contour(self):
        pts = np.array([[0.0, 0.0]], dtype=float)
        zd = zernike_moments(pts, order=4)
        assert isinstance(zd, ZernikeDescriptor)

    def test_zernike_similarity_same(self):
        pts = _make_circle(n=64)
        zd = zernike_moments(pts, order=6)
        sim = zernike_similarity(zd, zd)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_zernike_similarity_range(self):
        zd1 = zernike_moments(_make_circle(n=64), order=6)
        zd2 = zernike_moments(_make_line_pts(n=64), order=6)
        sim = zernike_similarity(zd1, zd2)
        assert 0.0 <= sim <= 1.0

    def test_zernike_to_feature_vector_normalized(self):
        pts = _make_circle(n=64)
        zd = zernike_moments(pts, order=6)
        fv = zernike_to_feature_vector(zd)
        assert fv.ndim == 1
        norm = np.linalg.norm(fv)
        assert norm == pytest.approx(1.0, abs=1e-5) or norm == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# 11. descriptor_aggregator
# ===========================================================================

from puzzle_reconstruction.algorithms.descriptor_aggregator import (
    AggregatorConfig,
    AggregatedDescriptor,
    l2_normalize,
    concatenate_descriptors,
    weighted_average_descriptors,
    pca_reduce,
    elementwise_aggregate,
    aggregate,
    distance_matrix,
    batch_aggregate,
)


class TestDescriptorAggregator:
    def test_aggregator_config_defaults(self):
        cfg = AggregatorConfig()
        assert cfg.mode == "concat"
        assert cfg.normalize is True

    def test_aggregator_config_invalid_mode(self):
        with pytest.raises(ValueError):
            AggregatorConfig(mode="invalid_mode")

    def test_l2_normalize_vector(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = l2_normalize(v)
        assert np.linalg.norm(n) == pytest.approx(1.0)

    def test_l2_normalize_matrix(self):
        M = np.random.rand(4, 8).astype(np.float32)
        N = l2_normalize(M)
        norms = np.linalg.norm(N, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_concatenate_descriptors(self):
        descs = {"a": np.ones(4), "b": np.ones(6)}
        out = concatenate_descriptors(descs, normalize=False)
        assert len(out) == 10

    def test_concatenate_descriptors_empty_raises(self):
        with pytest.raises(ValueError):
            concatenate_descriptors({})

    def test_weighted_average_descriptors(self):
        descs = {"a": np.ones(4), "b": np.zeros(4)}
        out = weighted_average_descriptors(descs, weights={"a": 1.0, "b": 0.0}, normalize=False)
        assert len(out) == 4

    def test_pca_reduce_shape(self):
        M = np.random.rand(10, 20).astype(np.float32)
        reduced = pca_reduce(M, n_components=5)
        assert reduced.shape == (10, 5)

    def test_elementwise_aggregate_max(self):
        descs = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 1.0, 2.0])}
        out = elementwise_aggregate(descs, mode="max", normalize=False)
        assert list(out) == [3.0, 2.0, 3.0]

    def test_aggregate_concat_mode(self):
        descs = {"a": np.random.rand(8), "b": np.random.rand(8)}
        cfg = AggregatorConfig(mode="concat", normalize=False)
        out = aggregate(descs, cfg)
        assert len(out) == 16

    def test_distance_matrix_cosine(self):
        V = np.random.rand(5, 10).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert D.shape == (5, 5)
        assert np.allclose(np.diag(D), 0.0, atol=1e-5)

    def test_batch_aggregate(self):
        groups = [{"a": np.random.rand(8)} for _ in range(4)]
        results = batch_aggregate(groups)
        assert len(results) == 4


# ===========================================================================
# 12. descriptor_combiner
# ===========================================================================

from puzzle_reconstruction.algorithms.descriptor_combiner import (
    CombineConfig,
    DescriptorSet,
    CombineResult,
    combine_descriptors,
    combine_selected,
    batch_combine,
    descriptor_distance,
    build_distance_matrix,
    find_nearest,
)


class TestDescriptorCombiner:
    def _make_desc_set(self, fragment_id=0):
        return DescriptorSet(
            fragment_id=fragment_id,
            descriptors={
                "shape": np.random.rand(8).astype(np.float32),
                "texture": np.random.rand(6).astype(np.float32),
                "color": np.random.rand(4).astype(np.float32),
            },
        )

    def test_combine_config_defaults(self):
        cfg = CombineConfig()
        assert cfg.normalize is True
        assert cfg.l2_final is True

    def test_combine_config_invalid_weight(self):
        with pytest.raises(ValueError):
            CombineConfig(weights={"shape": -1.0})

    def test_descriptor_set_properties(self):
        ds = self._make_desc_set()
        assert "shape" in ds.names
        assert ds.total_dim == 18
        assert ds.has("texture")
        assert not ds.has("missing")

    def test_combine_descriptors_basic(self):
        ds = self._make_desc_set()
        result = combine_descriptors(ds)
        assert isinstance(result, CombineResult)
        assert result.vector.ndim == 1

    def test_combine_descriptors_empty_raises(self):
        ds = DescriptorSet(fragment_id=0, descriptors={})
        with pytest.raises(ValueError):
            combine_descriptors(ds)

    def test_combine_selected_basic(self):
        ds = self._make_desc_set()
        result = combine_selected(ds, names=["shape", "color"])
        assert isinstance(result, CombineResult)
        assert set(result.used_names) == {"shape", "color"}

    def test_combine_selected_missing_all_raises(self):
        ds = self._make_desc_set()
        with pytest.raises(ValueError):
            combine_selected(ds, names=["nonexistent"])

    def test_combine_result_properties(self):
        ds = self._make_desc_set()
        result = combine_descriptors(ds)
        assert result.dim == len(result.vector)
        norm = result.norm
        assert norm >= 0.0

    def test_descriptor_distance_cosine(self):
        ds = self._make_desc_set(0)
        r1 = combine_descriptors(ds)
        r2 = combine_descriptors(self._make_desc_set(1))
        d = descriptor_distance(r1, r2, metric="cosine")
        assert d >= 0.0

    def test_descriptor_distance_same_zero(self):
        ds = self._make_desc_set(0)
        r = combine_descriptors(ds)
        d = descriptor_distance(r, r, metric="cosine")
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_batch_combine(self):
        dsets = [self._make_desc_set(i) for i in range(4)]
        results = batch_combine(dsets)
        assert len(results) == 4

    def test_build_distance_matrix_shape(self):
        dsets = [self._make_desc_set(i) for i in range(3)]
        results = batch_combine(dsets)
        mat = build_distance_matrix(results)
        assert mat.shape == (3, 3)
        assert np.allclose(np.diag(mat), 0.0, atol=1e-5)

    def test_find_nearest(self):
        query = combine_descriptors(self._make_desc_set(0))
        candidates = [combine_descriptors(self._make_desc_set(i)) for i in range(5)]
        nearest = find_nearest(query, candidates, top_k=3)
        assert len(nearest) == 3
        dists = [d for _, d in nearest]
        assert dists == sorted(dists)
