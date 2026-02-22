"""Тесты для puzzle_reconstruction.algorithms.boundary_descriptor."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle_pts(n=32, r=5.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line_pts(n=20) -> np.ndarray:
    x = np.linspace(0.0, 10.0, n)
    return np.stack([x, np.zeros(n)], axis=1)


def _make_desc(n_bins=8) -> BoundaryDescriptor:
    h = np.ones(n_bins, dtype=np.float32) / n_bins
    return BoundaryDescriptor(
        fragment_id=0,
        edge_id=0,
        curvature_hist=h,
        direction_hist=h,
        chord_hist=h,
        length=10.0,
    )


# ─── TestDescriptorConfig ─────────────────────────────────────────────────────

class TestDescriptorConfig:
    def test_defaults_ok(self):
        cfg = DescriptorConfig()
        assert cfg.n_bins == 32
        assert cfg.smooth_sigma == 1.0
        assert cfg.normalize is True
        assert cfg.max_chord is None

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            DescriptorConfig(n_bins=3)

    def test_n_bins_4_ok(self):
        cfg = DescriptorConfig(n_bins=4)
        assert cfg.n_bins == 4

    def test_smooth_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            DescriptorConfig(smooth_sigma=-0.1)

    def test_smooth_sigma_zero_ok(self):
        cfg = DescriptorConfig(smooth_sigma=0.0)
        assert cfg.smooth_sigma == 0.0

    def test_max_chord_zero_raises(self):
        with pytest.raises(ValueError):
            DescriptorConfig(max_chord=0.0)

    def test_max_chord_negative_raises(self):
        with pytest.raises(ValueError):
            DescriptorConfig(max_chord=-1.0)

    def test_max_chord_positive_ok(self):
        cfg = DescriptorConfig(max_chord=5.0)
        assert cfg.max_chord == pytest.approx(5.0)


# ─── TestBoundaryDescriptor ───────────────────────────────────────────────────

class TestBoundaryDescriptor:
    def test_fragment_id_negative_raises(self):
        h = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError):
            BoundaryDescriptor(
                fragment_id=-1, edge_id=0,
                curvature_hist=h, direction_hist=h, chord_hist=h,
                length=0.0,
            )

    def test_edge_id_negative_raises(self):
        h = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError):
            BoundaryDescriptor(
                fragment_id=0, edge_id=-1,
                curvature_hist=h, direction_hist=h, chord_hist=h,
                length=0.0,
            )

    def test_length_negative_raises(self):
        h = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError):
            BoundaryDescriptor(
                fragment_id=0, edge_id=0,
                curvature_hist=h, direction_hist=h, chord_hist=h,
                length=-1.0,
            )

    def test_2d_curvature_hist_raises(self):
        h1d = np.zeros(8, dtype=np.float32)
        h2d = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            BoundaryDescriptor(
                fragment_id=0, edge_id=0,
                curvature_hist=h2d, direction_hist=h1d, chord_hist=h1d,
                length=0.0,
            )

    def test_n_bins_property(self):
        desc = _make_desc(n_bins=16)
        assert desc.n_bins == 16

    def test_feature_vector_length(self):
        desc = _make_desc(n_bins=8)
        assert desc.feature_vector.shape == (24,)

    def test_feature_vector_dtype_float32(self):
        desc = _make_desc(n_bins=8)
        assert desc.feature_vector.dtype == np.float32


# ─── TestComputeCurvature ─────────────────────────────────────────────────────

class TestComputeCurvature:
    def test_returns_ndarray(self):
        pts = _circle_pts(16)
        out = compute_curvature(pts)
        assert isinstance(out, np.ndarray)

    def test_shape_matches_input(self):
        pts = _circle_pts(20)
        out = compute_curvature(pts)
        assert out.shape == (20,)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            compute_curvature(np.zeros((10, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_curvature(np.zeros((2, 2)))

    def test_sigma_zero_ok(self):
        pts = _circle_pts(10)
        out = compute_curvature(pts, smooth_sigma=0.0)
        assert out.shape == (10,)

    def test_straight_line_near_zero(self):
        pts = _line_pts(20)
        out = compute_curvature(pts, smooth_sigma=0.0)
        assert np.abs(out[2:-2]).max() < 0.1


# ─── TestCurvatureHistogram ───────────────────────────────────────────────────

class TestCurvatureHistogram:
    def test_returns_ndarray(self):
        curv = np.zeros(20, dtype=np.float32)
        out = curvature_histogram(curv)
        assert isinstance(out, np.ndarray)

    def test_output_length_n_bins(self):
        curv = np.linspace(-1, 1, 20)
        out = curvature_histogram(curv, n_bins=16)
        assert len(out) == 16

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            curvature_histogram(np.zeros(10), n_bins=3)

    def test_empty_curvature_zeros(self):
        out = curvature_histogram(np.array([]), n_bins=8)
        assert (out == 0.0).all()
        assert len(out) == 8

    def test_normalized_sums_to_one(self):
        curv = np.linspace(-2, 2, 50)
        out = curvature_histogram(curv, n_bins=16, normalize=True)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_not_normalized_raw_counts(self):
        curv = np.zeros(10, dtype=np.float32)
        out = curvature_histogram(curv, n_bins=8, normalize=False)
        assert out.sum() == pytest.approx(10.0, abs=1e-6)


# ─── TestDirectionHistogram ───────────────────────────────────────────────────

class TestDirectionHistogram:
    def test_shape_n_bins(self):
        pts = _circle_pts(32)
        out = direction_histogram(pts, n_bins=16)
        assert out.shape == (16,)

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            direction_histogram(_circle_pts(10), n_bins=3)

    def test_single_point_zeros(self):
        pts = np.array([[1.0, 0.0]])
        out = direction_histogram(pts, n_bins=8)
        assert (out == 0.0).all()

    def test_normalized_sums_to_one(self):
        pts = _circle_pts(32)
        out = direction_histogram(pts, n_bins=16, normalize=True)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_not_normalized_raw_counts(self):
        pts = _circle_pts(32)
        out = direction_histogram(pts, n_bins=8, normalize=False)
        assert out.sum() == pytest.approx(31.0, abs=1.0)  # 31 segments


# ─── TestChordDistribution ────────────────────────────────────────────────────

class TestChordDistribution:
    def test_shape_n_bins(self):
        pts = _circle_pts(10)
        out = chord_distribution(pts, n_bins=8)
        assert out.shape == (8,)

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            chord_distribution(_circle_pts(10), n_bins=3)

    def test_single_point_zeros(self):
        pts = np.array([[1.0, 0.0]])
        out = chord_distribution(pts, n_bins=8)
        assert (out == 0.0).all()

    def test_normalized_sums_to_one(self):
        pts = _circle_pts(16)
        out = chord_distribution(pts, n_bins=8, normalize=True)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_custom_max_chord(self):
        pts = _circle_pts(16)
        out = chord_distribution(pts, n_bins=8, max_chord=20.0, normalize=False)
        assert len(out) == 8

    def test_large_contour_sampled(self):
        pts = _circle_pts(300)
        out = chord_distribution(pts, n_bins=8)
        assert len(out) == 8


# ─── TestExtractDescriptor ────────────────────────────────────────────────────

class TestExtractDescriptor:
    def test_returns_boundary_descriptor(self):
        pts = _circle_pts(16)
        desc = extract_descriptor(pts)
        assert isinstance(desc, BoundaryDescriptor)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            extract_descriptor(np.zeros((10, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            extract_descriptor(np.zeros((2, 2)))

    def test_fragment_edge_ids(self):
        pts = _circle_pts(16)
        desc = extract_descriptor(pts, fragment_id=3, edge_id=7)
        assert desc.fragment_id == 3
        assert desc.edge_id == 7

    def test_length_positive(self):
        pts = _circle_pts(16)
        desc = extract_descriptor(pts)
        assert desc.length > 0.0

    def test_feature_vector_length(self):
        pts = _circle_pts(16)
        cfg = DescriptorConfig(n_bins=8)
        desc = extract_descriptor(pts, cfg=cfg)
        assert desc.feature_vector.shape == (24,)

    def test_custom_config(self):
        pts = _circle_pts(16)
        cfg = DescriptorConfig(n_bins=16, smooth_sigma=0.0, normalize=True)
        desc = extract_descriptor(pts, cfg=cfg)
        assert desc.n_bins == 16


# ─── TestDescriptorSimilarity ─────────────────────────────────────────────────

class TestDescriptorSimilarity:
    def test_identical_one(self):
        desc = _make_desc(n_bins=8)
        sim = descriptor_similarity(desc, desc)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_result_in_range(self):
        d1 = _make_desc(n_bins=8)
        h2 = np.zeros(8, dtype=np.float32)
        h2[0] = 1.0
        d2 = BoundaryDescriptor(
            fragment_id=1, edge_id=1,
            curvature_hist=h2, direction_hist=h2, chord_hist=h2,
            length=5.0,
        )
        sim = descriptor_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_mismatched_n_bins_raises(self):
        d1 = _make_desc(n_bins=8)
        d2 = _make_desc(n_bins=16)
        with pytest.raises(ValueError):
            descriptor_similarity(d1, d2)

    def test_returns_float(self):
        desc = _make_desc(n_bins=8)
        sim = descriptor_similarity(desc, desc)
        assert isinstance(sim, float)

    def test_custom_weights(self):
        desc = _make_desc(n_bins=8)
        sim = descriptor_similarity(desc, desc, w_curvature=1.0,
                                    w_direction=0.0, w_chord=0.0)
        assert sim == pytest.approx(1.0, abs=1e-6)


# ─── TestBatchExtractDescriptors ──────────────────────────────────────────────

class TestBatchExtractDescriptors:
    def test_returns_list(self):
        pts_list = [_circle_pts(10), _line_pts(10)]
        result = batch_extract_descriptors(pts_list)
        assert isinstance(result, list)

    def test_length_matches(self):
        pts_list = [_circle_pts(10), _circle_pts(12), _circle_pts(8)]
        result = batch_extract_descriptors(pts_list)
        assert len(result) == 3

    def test_edge_ids_are_indices(self):
        pts_list = [_circle_pts(10), _circle_pts(12)]
        result = batch_extract_descriptors(pts_list)
        for i, desc in enumerate(result):
            assert desc.edge_id == i

    def test_fragment_id_passed(self):
        pts_list = [_circle_pts(10)]
        result = batch_extract_descriptors(pts_list, fragment_id=5)
        assert result[0].fragment_id == 5

    def test_all_boundary_descriptors(self):
        pts_list = [_circle_pts(10), _line_pts(10)]
        for desc in batch_extract_descriptors(pts_list):
            assert isinstance(desc, BoundaryDescriptor)
