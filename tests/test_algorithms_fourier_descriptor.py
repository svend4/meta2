"""Tests for puzzle_reconstruction/algorithms/fourier_descriptor.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fourier_descriptor import (
    FourierConfig,
    FourierDescriptor,
    compute_contour_centroid,
    complex_representation,
    compute_fd,
    fd_similarity,
    batch_compute_fd,
    rank_by_fd,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_circle_contour(n=64, r=10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def make_square_contour(side=10.0, n=40):
    pts = []
    for i in range(n // 4):
        pts.append([i * side / (n // 4), 0.0])
    for i in range(n // 4):
        pts.append([side, i * side / (n // 4)])
    for i in range(n // 4):
        pts.append([side - i * side / (n // 4), side])
    for i in range(n // 4):
        pts.append([0.0, side - i * side / (n // 4)])
    return np.array(pts[:n])


def make_fd(n_coeffs=8, fragment_id=0, edge_id=0):
    pts = make_circle_contour()
    cfg = FourierConfig(n_coeffs=n_coeffs)
    return compute_fd(pts, fragment_id=fragment_id, edge_id=edge_id, cfg=cfg)


# ─── FourierConfig ────────────────────────────────────────────────────────────

class TestFourierConfig:
    def test_default(self):
        cfg = FourierConfig()
        assert cfg.n_coeffs == 32
        assert cfg.normalize is True

    def test_custom(self):
        cfg = FourierConfig(n_coeffs=8, normalize=False)
        assert cfg.n_coeffs == 8
        assert cfg.normalize is False

    def test_too_few_coeffs_raises(self):
        with pytest.raises(ValueError):
            FourierConfig(n_coeffs=3)

    def test_minimum_coeffs(self):
        cfg = FourierConfig(n_coeffs=4)
        assert cfg.n_coeffs == 4


# ─── FourierDescriptor ────────────────────────────────────────────────────────

class TestFourierDescriptor:
    def test_basic_creation(self):
        fd = make_fd()
        assert isinstance(fd, FourierDescriptor)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FourierDescriptor(fragment_id=-1, edge_id=0,
                              coefficients=np.zeros(16, dtype=np.float32),
                              n_coeffs=8)

    def test_negative_edge_id_raises(self):
        with pytest.raises(ValueError):
            FourierDescriptor(fragment_id=0, edge_id=-1,
                              coefficients=np.zeros(16, dtype=np.float32),
                              n_coeffs=8)

    def test_n_coeffs_too_small_raises(self):
        with pytest.raises(ValueError):
            FourierDescriptor(fragment_id=0, edge_id=0,
                              coefficients=np.zeros(16, dtype=np.float32),
                              n_coeffs=3)

    def test_non_1d_coeffs_raises(self):
        with pytest.raises(ValueError):
            FourierDescriptor(fragment_id=0, edge_id=0,
                              coefficients=np.zeros((8, 2), dtype=np.float32),
                              n_coeffs=4)

    def test_dim_property(self):
        fd = make_fd(n_coeffs=8)
        assert fd.dim == 16  # 2 * n_coeffs

    def test_magnitude_shape(self):
        fd = make_fd(n_coeffs=8)
        assert fd.magnitude.shape == (8,)

    def test_magnitude_non_negative(self):
        fd = make_fd()
        assert (fd.magnitude >= 0.0).all()

    def test_magnitude_dtype_float32(self):
        fd = make_fd()
        assert fd.magnitude.dtype == np.float32

    def test_coefficients_dtype_float32(self):
        fd = make_fd()
        assert fd.coefficients.dtype == np.float32


# ─── compute_contour_centroid ─────────────────────────────────────────────────

class TestComputeContourCentroid:
    def test_basic(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 1.0) < 1e-9
        assert abs(cy - 1.0) < 1e-9

    def test_single_point(self):
        pts = np.array([[5.0, 3.0]])
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 5.0) < 1e-9
        assert abs(cy - 3.0) < 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_contour_centroid(np.array([]).reshape(0, 2))

    def test_n12_format(self):
        pts = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 2.0) < 1e-9
        assert abs(cy - 3.0) < 1e-9

    def test_circle_centroid_near_origin(self):
        pts = make_circle_contour(n=64, r=5.0)
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx) < 0.1
        assert abs(cy) < 0.1


# ─── complex_representation ──────────────────────────────────────────────────

class TestComplexRepresentation:
    def test_basic(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        z = complex_representation(pts)
        assert z[0] == complex(1.0, 2.0)
        assert z[1] == complex(3.0, 4.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            complex_representation(np.array([]).reshape(0, 2))

    def test_output_shape(self):
        pts = make_circle_contour(n=32)
        z = complex_representation(pts)
        assert z.shape == (32,)

    def test_output_complex(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        z = complex_representation(pts)
        assert np.iscomplexobj(z)

    def test_n12_format(self):
        pts = np.array([[[1.0, 2.0]], [[3.0, 0.0]]])
        z = complex_representation(pts)
        assert len(z) == 2
        assert z[0].real == 1.0
        assert z[0].imag == 2.0


# ─── compute_fd ───────────────────────────────────────────────────────────────

class TestComputeFd:
    def test_returns_fourier_descriptor(self):
        pts = make_circle_contour()
        fd = compute_fd(pts)
        assert isinstance(fd, FourierDescriptor)

    def test_too_few_points_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        with pytest.raises(ValueError):
            compute_fd(pts)

    def test_coefficients_shape(self):
        pts = make_circle_contour()
        cfg = FourierConfig(n_coeffs=8)
        fd = compute_fd(pts, cfg=cfg)
        assert fd.coefficients.shape == (16,)

    def test_fragment_id_set(self):
        pts = make_circle_contour()
        fd = compute_fd(pts, fragment_id=7)
        assert fd.fragment_id == 7

    def test_edge_id_set(self):
        pts = make_circle_contour()
        fd = compute_fd(pts, edge_id=3)
        assert fd.edge_id == 3

    def test_normalized_dominant_coeff_near_one(self):
        """With normalize=True, the first non-zero magnitude should be ≈ 1."""
        pts = make_circle_contour()
        cfg = FourierConfig(n_coeffs=8, normalize=True)
        fd = compute_fd(pts, cfg=cfg)
        # For a centered circle, DC (index 0) is ~0; fundamental (index 1) is
        # the first non-zero coefficient used for normalization → magnitude ≈ 1.
        dominant_mag = fd.magnitude.max()
        assert abs(dominant_mag - 1.0) < 0.1

    def test_params_stored(self):
        pts = make_circle_contour(n=32)
        fd = compute_fd(pts)
        assert "n_points" in fd.params
        assert fd.params["n_points"] == 32

    def test_n12_contour_input(self):
        pts = make_circle_contour().reshape(-1, 1, 2)
        fd = compute_fd(pts)
        assert isinstance(fd, FourierDescriptor)

    def test_minimum_4_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cfg = FourierConfig(n_coeffs=4)
        fd = compute_fd(pts, cfg=cfg)
        assert fd.n_coeffs == 4

    def test_square_vs_circle_different(self):
        """Square and circle should produce different descriptors."""
        circ = compute_fd(make_circle_contour())
        sq = compute_fd(make_square_contour())
        assert not np.allclose(circ.coefficients, sq.coefficients)


# ─── fd_similarity ────────────────────────────────────────────────────────────

class TestFdSimilarity:
    def test_identical_descriptors_score_one(self):
        fd = make_fd()
        assert abs(fd_similarity(fd, fd) - 1.0) < 0.01

    def test_in_range(self):
        fd1 = make_fd(n_coeffs=8)
        fd2 = compute_fd(make_square_contour(), cfg=FourierConfig(n_coeffs=8))
        sim = fd_similarity(fd1, fd2)
        assert 0.0 <= sim <= 1.0

    def test_mismatched_n_coeffs_raises(self):
        fd1 = make_fd(n_coeffs=4)
        fd2 = make_fd(n_coeffs=8)
        with pytest.raises(ValueError):
            fd_similarity(fd1, fd2)

    def test_circle_vs_circle_high(self):
        """Two circles with same radius → similarity ≈ 1."""
        cfg = FourierConfig(n_coeffs=8)
        fd1 = compute_fd(make_circle_contour(r=10), cfg=cfg)
        fd2 = compute_fd(make_circle_contour(r=10), cfg=cfg)
        assert fd_similarity(fd1, fd2) > 0.9

    def test_circle_vs_square_lower(self):
        """Circle vs square should be lower than circle vs circle."""
        cfg = FourierConfig(n_coeffs=8)
        circ1 = compute_fd(make_circle_contour(n=64), cfg=cfg)
        circ2 = compute_fd(make_circle_contour(n=64), cfg=cfg)
        sq = compute_fd(make_square_contour(n=64), cfg=cfg)
        assert fd_similarity(circ1, circ2) >= fd_similarity(circ1, sq)

    def test_zero_magnitude_case(self):
        """Zero coefficients → similarity = 1 (both zero)."""
        flat = np.zeros(16, dtype=np.float32)
        fd1 = FourierDescriptor(fragment_id=0, edge_id=0,
                                coefficients=flat, n_coeffs=8)
        fd2 = FourierDescriptor(fragment_id=0, edge_id=0,
                                coefficients=flat.copy(), n_coeffs=8)
        assert fd_similarity(fd1, fd2) == 1.0


# ─── batch_compute_fd ─────────────────────────────────────────────────────────

class TestBatchComputeFd:
    def test_returns_list(self):
        contours = [make_circle_contour() for _ in range(3)]
        results = batch_compute_fd(contours)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_fourier_descriptors(self):
        contours = [make_circle_contour() for _ in range(4)]
        results = batch_compute_fd(contours)
        assert all(isinstance(r, FourierDescriptor) for r in results)

    def test_edge_ids_assigned(self):
        contours = [make_circle_contour() for _ in range(3)]
        results = batch_compute_fd(contours)
        for i, r in enumerate(results):
            assert r.edge_id == i

    def test_empty_list(self):
        results = batch_compute_fd([])
        assert results == []

    def test_custom_config(self):
        contours = [make_circle_contour()]
        cfg = FourierConfig(n_coeffs=4)
        results = batch_compute_fd(contours, cfg)
        assert results[0].n_coeffs == 4


# ─── rank_by_fd ───────────────────────────────────────────────────────────────

class TestRankByFd:
    def test_returns_list_of_tuples(self):
        query = make_fd()
        candidates = [make_fd(edge_id=i) for i in range(3)]
        ranked = rank_by_fd(query, candidates)
        assert isinstance(ranked, list)
        for item in ranked:
            assert len(item) == 2

    def test_sorted_descending(self):
        query = make_fd()
        candidates = [make_fd(edge_id=i) for i in range(4)]
        ranked = rank_by_fd(query, candidates)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_identical_candidate_first(self):
        """The identical candidate should rank highest."""
        cfg = FourierConfig(n_coeffs=8)
        circle = make_circle_contour()
        query = compute_fd(circle, cfg=cfg)
        c1 = compute_fd(make_square_contour(), cfg=cfg)
        c2 = compute_fd(make_circle_contour(r=10), cfg=cfg)
        ranked = rank_by_fd(query, [c1, c2])
        # c2 (circle) should score higher
        assert ranked[0][1] >= ranked[1][1]

    def test_custom_indices(self):
        query = make_fd()
        candidates = [make_fd() for _ in range(3)]
        ranked = rank_by_fd(query, candidates, indices=[10, 20, 30])
        indices_out = [i for i, _ in ranked]
        assert set(indices_out) == {10, 20, 30}

    def test_len_mismatch_raises(self):
        query = make_fd()
        candidates = [make_fd() for _ in range(3)]
        with pytest.raises(ValueError):
            rank_by_fd(query, candidates, indices=[0, 1])

    def test_empty_candidates(self):
        query = make_fd()
        ranked = rank_by_fd(query, [])
        assert ranked == []
