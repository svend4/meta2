"""Extra tests for puzzle_reconstruction/algorithms/boundary_descriptor.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.boundary_descriptor import (
    BoundaryDescriptor,
    DescriptorConfig,
    batch_extract_descriptors,
    chord_distribution,
    compute_curvature,
    curvature_histogram,
    descriptor_similarity,
    direction_histogram,
    extract_descriptor,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=32, r=5.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(n=40, s=5.0):
    side = n // 4
    pts = []
    for i in range(side):
        pts.append([i * s / side, 0])
    for i in range(side):
        pts.append([s, i * s / side])
    for i in range(side):
        pts.append([s - i * s / side, s])
    for i in range(side):
        pts.append([0, s - i * s / side])
    return np.array(pts, dtype=np.float64)


def _line(n=20):
    x = np.linspace(0.0, 10.0, n)
    return np.stack([x, np.zeros(n)], axis=1)


def _make_desc(n_bins=8, fid=0, eid=0):
    h = np.ones(n_bins, dtype=np.float32) / n_bins
    return BoundaryDescriptor(
        fragment_id=fid,
        edge_id=eid,
        curvature_hist=h,
        direction_hist=h,
        chord_hist=h,
        length=10.0,
    )


# ─── TestDescriptorConfigExtra ────────────────────────────────────────────────

class TestDescriptorConfigExtra:
    def test_n_bins_large(self):
        cfg = DescriptorConfig(n_bins=128)
        assert cfg.n_bins == 128

    def test_smooth_sigma_large(self):
        cfg = DescriptorConfig(smooth_sigma=5.0)
        assert cfg.smooth_sigma == pytest.approx(5.0)

    def test_normalize_false(self):
        cfg = DescriptorConfig(normalize=False)
        assert cfg.normalize is False

    def test_max_chord_large(self):
        cfg = DescriptorConfig(max_chord=100.0)
        assert cfg.max_chord == pytest.approx(100.0)

    def test_n_bins_exactly_4_ok(self):
        cfg = DescriptorConfig(n_bins=4)
        assert cfg.n_bins == 4

    def test_n_bins_3_raises(self):
        with pytest.raises(ValueError):
            DescriptorConfig(n_bins=3)

    def test_smooth_sigma_exactly_zero_ok(self):
        cfg = DescriptorConfig(smooth_sigma=0.0)
        assert cfg.smooth_sigma == pytest.approx(0.0)


# ─── TestBoundaryDescriptorExtra ─────────────────────────────────────────────

class TestBoundaryDescriptorExtra:
    def test_valid_zero_length(self):
        h = np.zeros(8, dtype=np.float32)
        desc = BoundaryDescriptor(
            fragment_id=0, edge_id=0,
            curvature_hist=h, direction_hist=h, chord_hist=h,
            length=0.0,
        )
        assert desc.length == pytest.approx(0.0)

    def test_n_bins_32(self):
        desc = _make_desc(n_bins=32)
        assert desc.n_bins == 32

    def test_feature_vector_n16(self):
        desc = _make_desc(n_bins=16)
        assert desc.feature_vector.shape == (48,)

    def test_feature_vector_uniform(self):
        desc = _make_desc(n_bins=8)
        fv = desc.feature_vector
        # uniform histograms → all values equal to 1/8
        assert np.allclose(fv, 1.0 / 8, atol=1e-6)

    def test_large_fragment_id(self):
        h = np.zeros(8, dtype=np.float32)
        desc = BoundaryDescriptor(
            fragment_id=10000, edge_id=999,
            curvature_hist=h, direction_hist=h, chord_hist=h,
            length=5.0,
        )
        assert desc.fragment_id == 10000
        assert desc.edge_id == 999

    def test_different_hists_allowed(self):
        h1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        h2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        h3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        desc = BoundaryDescriptor(
            fragment_id=0, edge_id=0,
            curvature_hist=h1, direction_hist=h2, chord_hist=h3,
            length=3.0,
        )
        assert desc.n_bins == 4


# ─── TestComputeCurvatureExtra ────────────────────────────────────────────────

class TestComputeCurvatureExtra:
    def test_circle_curvature_approximately_uniform(self):
        pts = _circle(n=64, r=10.0)
        out = compute_curvature(pts, smooth_sigma=0.0)
        # all curvatures should have similar magnitude
        assert out.std() < out.mean() * 2

    def test_large_n_points(self):
        pts = _circle(n=200)
        out = compute_curvature(pts)
        assert out.shape == (200,)

    def test_output_finite(self):
        pts = _circle(n=32)
        out = compute_curvature(pts, smooth_sigma=1.0)
        assert np.all(np.isfinite(out))

    def test_sigma_effect_finite(self):
        pts = _circle(n=32)
        out_smooth = compute_curvature(pts, smooth_sigma=3.0)
        assert np.all(np.isfinite(out_smooth))
        assert out_smooth.shape == (32,)

    def test_square_contour(self):
        pts = _square(n=40)
        out = compute_curvature(pts)
        assert out.shape == (40,)
        assert np.all(np.isfinite(out))

    def test_minimum_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        out = compute_curvature(pts)
        assert out.shape == (3,)


# ─── TestCurvatureHistogramExtra ──────────────────────────────────────────────

class TestCurvatureHistogramExtra:
    def test_n_bins_32(self):
        curv = np.linspace(-2, 2, 50)
        out = curvature_histogram(curv, n_bins=32)
        assert len(out) == 32

    def test_all_same_curvature(self):
        curv = np.full(20, 0.5)
        out = curvature_histogram(curv, n_bins=8, normalize=False)
        assert out.sum() == pytest.approx(20.0)

    def test_not_normalized_nonneg(self):
        curv = np.random.default_rng(0).random(30) - 0.5
        out = curvature_histogram(curv, n_bins=8, normalize=False)
        assert float(out.min()) >= 0.0

    def test_normalized_nonneg(self):
        curv = np.linspace(-1, 1, 50)
        out = curvature_histogram(curv, n_bins=8, normalize=True)
        assert float(out.min()) >= 0.0

    def test_large_n_bins(self):
        curv = np.linspace(-2, 2, 200)
        out = curvature_histogram(curv, n_bins=64)
        assert len(out) == 64

    def test_dtype_float(self):
        curv = np.zeros(10, dtype=np.float32)
        out = curvature_histogram(curv, n_bins=8)
        assert out.dtype in (np.float32, np.float64)


# ─── TestDirectionHistogramExtra ──────────────────────────────────────────────

class TestDirectionHistogramExtra:
    def test_n_bins_32(self):
        pts = _circle(n=64)
        out = direction_histogram(pts, n_bins=32)
        assert len(out) == 32

    def test_circle_approximately_uniform(self):
        pts = _circle(n=256)
        out = direction_histogram(pts, n_bins=8, normalize=True)
        # Circle should have roughly uniform direction distribution
        assert out.std() < 0.1

    def test_nonneg_values(self):
        pts = _circle(n=32)
        out = direction_histogram(pts, n_bins=8)
        assert float(out.min()) >= 0.0

    def test_square_contour(self):
        pts = _square(n=40)
        out = direction_histogram(pts, n_bins=8, normalize=True)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_two_points_zeros(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = direction_histogram(pts, n_bins=8)
        # Should not crash and return 8 values
        assert len(out) == 8

    def test_not_normalized_sum_is_n_segments(self):
        pts = _circle(n=16)
        out = direction_histogram(pts, n_bins=8, normalize=False)
        assert out.sum() == pytest.approx(15.0, abs=2.0)


# ─── TestChordDistributionExtra ───────────────────────────────────────────────

class TestChordDistributionExtra:
    def test_n_bins_32(self):
        pts = _circle(n=32)
        out = chord_distribution(pts, n_bins=32)
        assert len(out) == 32

    def test_nonneg_values(self):
        pts = _circle(n=20)
        out = chord_distribution(pts, n_bins=8, normalize=False)
        assert float(out.min()) >= 0.0

    def test_max_chord_small(self):
        pts = _circle(n=20, r=5.0)
        # max chord < radius means many chords are clipped
        out = chord_distribution(pts, n_bins=8, max_chord=1.0, normalize=False)
        assert len(out) == 8

    def test_square_contour(self):
        pts = _square(n=40)
        out = chord_distribution(pts, n_bins=8, normalize=True)
        assert len(out) == 8

    def test_large_n_bins(self):
        pts = _circle(n=64)
        out = chord_distribution(pts, n_bins=64)
        assert len(out) == 64

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0]])
        out = chord_distribution(pts, n_bins=8)
        assert len(out) == 8


# ─── TestExtractDescriptorExtra ───────────────────────────────────────────────

class TestExtractDescriptorExtra:
    def test_circle_length_positive(self):
        pts = _circle(n=64, r=10.0)
        desc = extract_descriptor(pts)
        assert desc.length > 0.0

    def test_square_contour(self):
        pts = _square(n=40)
        desc = extract_descriptor(pts)
        assert isinstance(desc, BoundaryDescriptor)
        assert desc.length > 0.0

    def test_large_n_bins(self):
        pts = _circle(n=32)
        cfg = DescriptorConfig(n_bins=16)
        desc = extract_descriptor(pts, cfg=cfg)
        assert desc.feature_vector.shape == (48,)

    def test_normalize_false_no_crash(self):
        pts = _circle(n=20)
        cfg = DescriptorConfig(n_bins=8, normalize=False)
        desc = extract_descriptor(pts, cfg=cfg)
        assert isinstance(desc, BoundaryDescriptor)

    def test_with_max_chord(self):
        pts = _circle(n=32, r=5.0)
        cfg = DescriptorConfig(n_bins=8, max_chord=3.0)
        desc = extract_descriptor(pts, cfg=cfg)
        assert isinstance(desc, BoundaryDescriptor)

    def test_feature_vector_dtype(self):
        pts = _circle(n=20)
        desc = extract_descriptor(pts)
        assert desc.feature_vector.dtype == np.float32

    def test_custom_fragment_edge_ids(self):
        pts = _circle(n=20)
        desc = extract_descriptor(pts, fragment_id=42, edge_id=7)
        assert desc.fragment_id == 42
        assert desc.edge_id == 7


# ─── TestDescriptorSimilarityExtra ────────────────────────────────────────────

class TestDescriptorSimilarityExtra:
    def test_same_descriptor_different_ids(self):
        d1 = _make_desc(n_bins=8, fid=0, eid=0)
        d2 = _make_desc(n_bins=8, fid=1, eid=1)
        sim = descriptor_similarity(d1, d2)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_histograms_low_sim(self):
        h1 = np.zeros(8, dtype=np.float32)
        h1[0] = 1.0
        h2 = np.zeros(8, dtype=np.float32)
        h2[4] = 1.0
        d1 = BoundaryDescriptor(
            fragment_id=0, edge_id=0,
            curvature_hist=h1, direction_hist=h1, chord_hist=h1, length=1.0,
        )
        d2 = BoundaryDescriptor(
            fragment_id=1, edge_id=0,
            curvature_hist=h2, direction_hist=h2, chord_hist=h2, length=1.0,
        )
        sim = descriptor_similarity(d1, d2)
        assert sim < 0.5

    def test_weight_curvature_only(self):
        d1 = _make_desc(n_bins=8)
        h2 = np.zeros(8, dtype=np.float32)
        h2[0] = 1.0
        d2 = BoundaryDescriptor(
            fragment_id=1, edge_id=0,
            curvature_hist=h2,
            direction_hist=d1.direction_hist,
            chord_hist=d1.chord_hist,
            length=5.0,
        )
        sim = descriptor_similarity(d1, d2, w_curvature=1.0, w_direction=0.0, w_chord=0.0)
        assert 0.0 <= sim <= 1.0

    def test_n_bins_mismatch_raises(self):
        d1 = _make_desc(n_bins=8)
        d2 = _make_desc(n_bins=16)
        with pytest.raises(ValueError):
            descriptor_similarity(d1, d2)

    def test_result_is_float(self):
        d1 = _make_desc(n_bins=8)
        assert isinstance(descriptor_similarity(d1, d1), float)


# ─── TestBatchExtractDescriptorsExtra ────────────────────────────────────────

class TestBatchExtractDescriptorsExtra:
    def test_ten_contours(self):
        pts_list = [_circle(n=20, r=float(i + 2)) for i in range(10)]
        result = batch_extract_descriptors(pts_list)
        assert len(result) == 10

    def test_empty_list(self):
        result = batch_extract_descriptors([])
        assert result == []

    def test_mixed_shapes(self):
        pts_list = [_circle(n=16), _square(n=40), _line(n=20)]
        result = batch_extract_descriptors(pts_list)
        assert len(result) == 3

    def test_edge_ids_sequential(self):
        pts_list = [_circle(n=16) for _ in range(5)]
        result = batch_extract_descriptors(pts_list)
        for i, desc in enumerate(result):
            assert desc.edge_id == i

    def test_custom_config(self):
        pts_list = [_circle(n=16)]
        cfg = DescriptorConfig(n_bins=16)
        result = batch_extract_descriptors(pts_list, cfg=cfg)
        assert result[0].n_bins == 16

    def test_all_lengths_positive(self):
        pts_list = [_circle(n=20, r=float(i + 1)) for i in range(5)]
        result = batch_extract_descriptors(pts_list)
        assert all(desc.length > 0.0 for desc in result)
