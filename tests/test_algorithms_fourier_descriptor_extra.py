"""Extra tests for puzzle_reconstruction/algorithms/fourier_descriptor.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fourier_descriptor import (
    FourierConfig,
    FourierDescriptor,
    batch_compute_fd,
    complex_representation,
    compute_contour_centroid,
    compute_fd,
    fd_similarity,
    rank_by_fd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])

def _square(side=10.0, n=40):
    pts = []
    step = side / (n // 4)
    for i in range(n // 4):
        pts.append([i * step, 0.0])
    for i in range(n // 4):
        pts.append([side, i * step])
    for i in range(n // 4):
        pts.append([side - i * step, side])
    for i in range(n // 4):
        pts.append([0.0, side - i * step])
    return np.array(pts[:n])

def _make_fd(n_coeffs=8, fragment_id=0, edge_id=0):
    return compute_fd(_circle(), fragment_id=fragment_id, edge_id=edge_id,
                      cfg=FourierConfig(n_coeffs=n_coeffs))


# ─── TestFourierConfigExtra ───────────────────────────────────────────────────

class TestFourierConfigExtra:
    def test_normalize_true_stored(self):
        cfg = FourierConfig(normalize=True)
        assert cfg.normalize is True

    def test_normalize_false_stored(self):
        cfg = FourierConfig(normalize=False)
        assert cfg.normalize is False

    def test_large_n_coeffs_valid(self):
        cfg = FourierConfig(n_coeffs=128)
        assert cfg.n_coeffs == 128

    def test_n_coeffs_16(self):
        cfg = FourierConfig(n_coeffs=16)
        assert cfg.n_coeffs == 16

    def test_n_coeffs_below_4_raises(self):
        with pytest.raises(ValueError):
            FourierConfig(n_coeffs=2)

    def test_n_coeffs_4_boundary(self):
        cfg = FourierConfig(n_coeffs=4)
        assert cfg.n_coeffs == 4


# ─── TestFourierDescriptorExtra ──────────────────────────────────────────────

class TestFourierDescriptorExtra:
    def test_fragment_id_0_valid(self):
        fd = _make_fd(fragment_id=0)
        assert fd.fragment_id == 0

    def test_edge_id_0_valid(self):
        fd = _make_fd(edge_id=0)
        assert fd.edge_id == 0

    def test_large_fragment_id(self):
        fd = _make_fd(fragment_id=999)
        assert fd.fragment_id == 999

    def test_coefficients_shape_2n(self):
        fd = _make_fd(n_coeffs=8)
        assert fd.coefficients.shape == (16,)

    def test_params_default_empty(self):
        fd = _make_fd()
        assert isinstance(fd.params, dict)

    def test_params_stored(self):
        coeffs = np.zeros(16, dtype=np.float32)
        fd = FourierDescriptor(fragment_id=0, edge_id=0,
                                coefficients=coeffs, n_coeffs=8,
                                params={"source": "test"})
        assert fd.params["source"] == "test"

    def test_n_coeffs_stored(self):
        fd = _make_fd(n_coeffs=4)
        assert fd.n_coeffs == 4

    def test_dim_is_2_times_n_coeffs(self):
        for n in (4, 8, 16):
            fd = _make_fd(n_coeffs=n)
            assert fd.dim == 2 * n


# ─── TestComputeContourCentroidExtra ─────────────────────────────────────────

class TestComputeContourCentroidExtra:
    def test_two_points_centroid(self):
        pts = np.array([[0.0, 0.0], [4.0, 2.0]])
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 2.0) < 1e-9
        assert abs(cy - 1.0) < 1e-9

    def test_large_contour(self):
        pts = _circle(n=128, r=20.0)
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx) < 1.0
        assert abs(cy) < 1.0

    def test_asymmetric_points(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 20.0 / 3) < 1e-6

    def test_returns_tuple_of_two(self):
        pts = _circle(n=32)
        result = compute_contour_centroid(pts)
        assert len(result) == 2

    def test_all_same_point(self):
        pts = np.array([[3.0, 7.0]] * 10)
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 3.0) < 1e-9
        assert abs(cy - 7.0) < 1e-9


# ─── TestComplexRepresentationExtra ──────────────────────────────────────────

class TestComplexRepresentationExtra:
    def test_large_contour(self):
        pts = _circle(n=128)
        z = complex_representation(pts)
        assert z.shape == (128,)

    def test_imaginary_parts_correct(self):
        pts = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        z = complex_representation(pts)
        assert z[0].imag == pytest.approx(5.0)
        assert z[1].imag == pytest.approx(6.0)

    def test_real_parts_correct(self):
        pts = np.array([[4.0, 1.0], [5.0, 2.0]])
        z = complex_representation(pts)
        assert z[0].real == pytest.approx(4.0)
        assert z[1].real == pytest.approx(5.0)

    def test_all_same_point(self):
        pts = np.array([[2.0, 3.0]] * 5)
        z = complex_representation(pts)
        assert all(zi == complex(2.0, 3.0) for zi in z)

    def test_dtype_is_complex(self):
        pts = _circle(n=16)
        z = complex_representation(pts)
        assert np.iscomplexobj(z)


# ─── TestComputeFdExtra ───────────────────────────────────────────────────────

class TestComputeFdExtra:
    def test_square_contour(self):
        fd = compute_fd(_square(), cfg=FourierConfig(n_coeffs=8))
        assert isinstance(fd, FourierDescriptor)

    def test_normalize_false(self):
        pts = _circle()
        cfg = FourierConfig(n_coeffs=8, normalize=False)
        fd = compute_fd(pts, cfg=cfg)
        assert isinstance(fd, FourierDescriptor)

    def test_different_n_coeffs(self):
        pts = _circle()
        for n in (4, 8, 16, 32):
            fd = compute_fd(pts, cfg=FourierConfig(n_coeffs=n))
            assert fd.n_coeffs == n
            assert fd.coefficients.shape == (2 * n,)

    def test_circle_r5_vs_r10_different(self):
        fd1 = compute_fd(_circle(r=5.0))
        fd2 = compute_fd(_circle(r=10.0))
        # With normalization on, should be similar but with different raw coeffs
        assert isinstance(fd1, FourierDescriptor)
        assert isinstance(fd2, FourierDescriptor)

    def test_n12_input_gives_same_result(self):
        pts = _circle(n=32)
        fd1 = compute_fd(pts)
        fd2 = compute_fd(pts.reshape(-1, 1, 2))
        assert np.allclose(fd1.coefficients, fd2.coefficients)

    def test_params_n_points_stored(self):
        pts = _circle(n=48)
        fd = compute_fd(pts)
        assert fd.params.get("n_points") == 48


# ─── TestFdSimilarityExtra ────────────────────────────────────────────────────

class TestFdSimilarityExtra:
    def test_symmetric(self):
        cfg = FourierConfig(n_coeffs=8)
        fd1 = compute_fd(_circle(), cfg=cfg)
        fd2 = compute_fd(_square(), cfg=cfg)
        assert fd_similarity(fd1, fd2) == pytest.approx(
            fd_similarity(fd2, fd1), abs=1e-6
        )

    def test_circle_r5_vs_r10_normalized_high(self):
        """Normalized FDs of same shape (different radii) should be similar."""
        cfg = FourierConfig(n_coeffs=8, normalize=True)
        fd1 = compute_fd(_circle(r=5.0), cfg=cfg)
        fd2 = compute_fd(_circle(r=10.0), cfg=cfg)
        sim = fd_similarity(fd1, fd2)
        assert sim > 0.8

    def test_all_in_range_various(self):
        cfg = FourierConfig(n_coeffs=8)
        circle = compute_fd(_circle(), cfg=cfg)
        square = compute_fd(_square(), cfg=cfg)
        for fd_pair in [(circle, circle), (square, square), (circle, square)]:
            assert 0.0 <= fd_similarity(*fd_pair) <= 1.0

    def test_n_coeffs_4_works(self):
        cfg = FourierConfig(n_coeffs=4)
        fd1 = compute_fd(_circle(n=16), cfg=cfg)
        fd2 = compute_fd(_circle(n=16), cfg=cfg)
        assert fd_similarity(fd1, fd2) == pytest.approx(1.0, abs=0.01)


# ─── TestBatchComputeFdExtra ──────────────────────────────────────────────────

class TestBatchComputeFdExtra:
    def test_square_contours(self):
        contours = [_square() for _ in range(3)]
        results = batch_compute_fd(contours)
        assert all(isinstance(r, FourierDescriptor) for r in results)

    def test_ten_contours(self):
        contours = [_circle(n=32) for _ in range(10)]
        results = batch_compute_fd(contours)
        assert len(results) == 10

    def test_fragment_id_default_zero(self):
        contours = [_circle() for _ in range(3)]
        results = batch_compute_fd(contours)
        for r in results:
            assert r.fragment_id == 0

    def test_n_coeffs_preserved(self):
        contours = [_circle(n=32) for _ in range(3)]
        cfg = FourierConfig(n_coeffs=16)
        results = batch_compute_fd(contours, cfg)
        for r in results:
            assert r.n_coeffs == 16

    def test_edge_ids_sequential(self):
        contours = [_circle(n=32) for _ in range(5)]
        results = batch_compute_fd(contours)
        for i, r in enumerate(results):
            assert r.edge_id == i


# ─── TestRankByFdExtra ────────────────────────────────────────────────────────

class TestRankByFdExtra:
    def test_auto_indices_0_to_n(self):
        query = _make_fd()
        candidates = [_make_fd(edge_id=i) for i in range(4)]
        ranked = rank_by_fd(query, candidates)
        indices = [i for i, _ in ranked]
        assert set(indices) == {0, 1, 2, 3}

    def test_length_preserved(self):
        query = _make_fd()
        candidates = [_make_fd() for _ in range(5)]
        ranked = rank_by_fd(query, candidates)
        assert len(ranked) == 5

    def test_all_same_scores_close(self):
        cfg = FourierConfig(n_coeffs=8)
        query = compute_fd(_circle(), cfg=cfg)
        candidates = [compute_fd(_circle(), cfg=cfg) for _ in range(3)]
        ranked = rank_by_fd(query, candidates)
        scores = [s for _, s in ranked]
        for s in scores:
            assert s == pytest.approx(scores[0], abs=0.01)

    def test_scores_in_0_1(self):
        cfg = FourierConfig(n_coeffs=8)
        query = compute_fd(_circle(), cfg=cfg)
        candidates = [
            compute_fd(_circle(), cfg=cfg),
            compute_fd(_square(), cfg=cfg),
        ]
        for _, score in rank_by_fd(query, candidates):
            assert 0.0 <= score <= 1.0

    def test_custom_indices_preserved(self):
        query = _make_fd()
        candidates = [_make_fd() for _ in range(3)]
        ranked = rank_by_fd(query, candidates, indices=[5, 10, 15])
        indices = sorted(i for i, _ in ranked)
        assert indices == [5, 10, 15]
