"""
Тесты для puzzle_reconstruction.matching.boundary_matcher.
"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.boundary_matcher import (
    BoundaryMatch,
    extract_boundary_points,
    hausdorff_distance,
    chamfer_distance,
    frechet_approx,
    score_boundary_pair,
    match_boundary_pair,
    batch_match_boundaries,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _square_contour(x0: int, y0: int, size: int) -> np.ndarray:
    """Прямоугольный контур (N, 2)."""
    pts = [
        (x0, y0), (x0 + size, y0),
        (x0 + size, y0 + size), (x0, y0 + size),
    ]
    return np.array(pts, dtype=np.float32)


def _circle_contour(cx: float, cy: float, r: float, n: int = 64) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles),
                    cy + r * np.sin(angles)], axis=1).astype(np.float32)
    return pts


def _line_contour(n: int = 20, axis: str = "x") -> np.ndarray:
    if axis == "x":
        return np.column_stack([np.linspace(0, 100, n),
                                 np.zeros(n)]).astype(np.float64)
    return np.column_stack([np.zeros(n),
                             np.linspace(0, 100, n)]).astype(np.float64)


# ─── BoundaryMatch ────────────────────────────────────────────────────────────

class TestBoundaryMatch:
    def test_fields(self):
        bm = BoundaryMatch(0, 1, 2, 0, 0.9, 0.85, 0.8, 0.85)
        assert bm.idx1 == 0
        assert bm.idx2 == 1
        assert bm.side1 == 2
        assert bm.side2 == 0
        assert bm.hausdorff  == pytest.approx(0.9)
        assert bm.chamfer    == pytest.approx(0.85)
        assert bm.frechet    == pytest.approx(0.8)
        assert bm.total_score == pytest.approx(0.85)

    def test_default_params_empty_dict(self):
        bm = BoundaryMatch(0, 1, 2, 0, 0.5, 0.5, 0.5, 0.5)
        assert bm.params == {}

    def test_repr_contains_total(self):
        bm = BoundaryMatch(3, 7, 0, 2, 0.6, 0.6, 0.6, 0.6)
        r  = repr(bm)
        assert "total" in r.lower() or "0.6" in r


# ─── extract_boundary_points ──────────────────────────────────────────────────

class TestExtractBoundaryPoints:
    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_output_shape(self, side):
        cnt = _square_contour(0, 0, 100)
        pts = extract_boundary_points(cnt, side=side, n_points=20)
        assert pts.shape == (20, 2)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_output_dtype_float64(self, side):
        cnt = _square_contour(0, 0, 100)
        pts = extract_boundary_points(cnt, side=side)
        assert pts.dtype == np.float64

    def test_invalid_side_raises(self):
        cnt = _square_contour(0, 0, 50)
        with pytest.raises(ValueError):
            extract_boundary_points(cnt, side=4)

    def test_invalid_side_negative_raises(self):
        cnt = _square_contour(0, 0, 50)
        with pytest.raises(ValueError):
            extract_boundary_points(cnt, side=-1)

    def test_empty_contour_raises(self):
        cnt = np.empty((0, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            extract_boundary_points(cnt, side=0)

    def test_n1_contour_single_point_returns_repeated(self):
        cnt = np.array([[5.0, 10.0]], dtype=np.float32)
        pts = extract_boundary_points(cnt, side=0, n_points=5)
        assert pts.shape == (5, 2)
        assert np.all(pts[:, 0] == pytest.approx(5.0))

    def test_contour_n1_2_format(self):
        cnt = _square_contour(0, 0, 50).reshape(-1, 1, 2)
        pts = extract_boundary_points(cnt, side=2, n_points=10)
        assert pts.shape == (10, 2)

    def test_side0_points_near_min_y(self):
        cnt = _square_contour(0, 0, 100)  # y ∈ {0, 100}
        pts = extract_boundary_points(cnt, side=0, n_points=2)
        # Верх → минимальный y
        assert np.all(pts[:, 1] <= 50.0)

    def test_side2_points_near_max_y(self):
        cnt = _square_contour(0, 0, 100)
        pts = extract_boundary_points(cnt, side=2, n_points=2)
        # Низ → максимальный y
        assert np.all(pts[:, 1] >= 50.0)

    def test_side1_points_near_max_x(self):
        cnt = _square_contour(0, 0, 100)
        pts = extract_boundary_points(cnt, side=1, n_points=2)
        assert np.all(pts[:, 0] >= 50.0)

    def test_side3_points_near_min_x(self):
        cnt = _square_contour(0, 0, 100)
        pts = extract_boundary_points(cnt, side=3, n_points=2)
        assert np.all(pts[:, 0] <= 50.0)

    def test_large_n_points_with_repeat(self):
        cnt = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pts = extract_boundary_points(cnt, side=0, n_points=100)
        assert pts.shape == (100, 2)

    def test_circle_contour_all_sides(self):
        cnt = _circle_contour(50, 50, 30, n=100)
        for side in range(4):
            pts = extract_boundary_points(cnt, side=side, n_points=15)
            assert pts.shape == (15, 2)


# ─── hausdorff_distance ───────────────────────────────────────────────────────

class TestHausdorffDistance:
    def test_identical_sets_zero(self):
        pts = _line_contour(10)
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0)

    def test_symmetric(self):
        pts1 = _line_contour(10)
        pts2 = _line_contour(10) + np.array([[5.0, 5.0]])
        d12 = hausdorff_distance(pts1, pts2)
        d21 = hausdorff_distance(pts2, pts1)
        assert d12 == pytest.approx(d21, abs=1e-6)

    def test_known_value(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        assert hausdorff_distance(pts1, pts2) == pytest.approx(5.0)

    def test_non_negative(self):
        pts1 = np.random.rand(20, 2).astype(np.float64) * 100
        pts2 = np.random.rand(15, 2).astype(np.float64) * 100
        assert hausdorff_distance(pts1, pts2) >= 0.0

    def test_empty_returns_zero(self):
        pts  = _line_contour(5)
        emp  = np.empty((0, 2), dtype=np.float64)
        assert hausdorff_distance(emp, pts) == pytest.approx(0.0)
        assert hausdorff_distance(pts, emp) == pytest.approx(0.0)

    def test_larger_shift_larger_distance(self):
        base = np.array([[0.0, 0.0], [1.0, 0.0]])
        near = base + np.array([[1.0, 0.0]])
        far  = base + np.array([[10.0, 0.0]])
        assert hausdorff_distance(base, near) < hausdorff_distance(base, far)


# ─── chamfer_distance ─────────────────────────────────────────────────────────

class TestChamferDistance:
    def test_identical_sets_zero(self):
        pts = _line_contour(10)
        assert chamfer_distance(pts, pts) == pytest.approx(0.0)

    def test_symmetric(self):
        pts1 = _line_contour(10)
        pts2 = _line_contour(10) + np.array([[3.0, 4.0]])
        d12  = chamfer_distance(pts1, pts2)
        d21  = chamfer_distance(pts2, pts1)
        assert d12 == pytest.approx(d21, abs=1e-5)

    def test_known_value(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        assert chamfer_distance(pts1, pts2) == pytest.approx(5.0)

    def test_non_negative(self):
        pts1 = np.random.rand(20, 2).astype(np.float64) * 100
        pts2 = np.random.rand(15, 2).astype(np.float64) * 100
        assert chamfer_distance(pts1, pts2) >= 0.0

    def test_empty_returns_zero(self):
        pts = _line_contour(5)
        emp = np.empty((0, 2), dtype=np.float64)
        assert chamfer_distance(emp, pts) == pytest.approx(0.0)

    def test_chamfer_le_hausdorff(self):
        pts1 = np.random.rand(30, 2).astype(np.float64) * 50
        pts2 = np.random.rand(30, 2).astype(np.float64) * 50
        # Chamfer ≤ Hausdorff всегда
        assert chamfer_distance(pts1, pts2) <= hausdorff_distance(pts1, pts2) + 1e-6


# ─── frechet_approx ───────────────────────────────────────────────────────────

class TestFrechetApprox:
    def test_identical_curves_zero(self):
        pts = _line_contour(10)
        assert frechet_approx(pts, pts) == pytest.approx(0.0)

    def test_non_negative(self):
        pts1 = _line_contour(8)
        pts2 = _line_contour(8) + np.array([[2.0, 0.0]])
        assert frechet_approx(pts1, pts2) >= 0.0

    def test_empty_returns_zero(self):
        pts = _line_contour(5)
        emp = np.empty((0, 2), dtype=np.float64)
        assert frechet_approx(emp, pts) == pytest.approx(0.0)

    def test_single_point_curves(self):
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[3.0, 4.0]])
        assert frechet_approx(p1, p2) == pytest.approx(5.0)

    def test_larger_shift_larger_frechet(self):
        base = _line_contour(10)
        near = base + np.array([[1.0, 0.0]])
        far  = base + np.array([[10.0, 0.0]])
        assert frechet_approx(base, near) < frechet_approx(base, far)

    def test_large_arrays_no_error(self):
        pts1 = np.random.rand(200, 2).astype(np.float64)
        pts2 = np.random.rand(200, 2).astype(np.float64)
        result = frechet_approx(pts1, pts2)
        assert isinstance(result, float)


# ─── score_boundary_pair ──────────────────────────────────────────────────────

class TestScoreBoundaryPair:
    def test_returns_4_tuple(self):
        pts = _line_contour(10)
        result = score_boundary_pair(pts, pts)
        assert len(result) == 4

    def test_identical_all_ones(self):
        pts = _line_contour(10)
        h, c, f, total = score_boundary_pair(pts, pts)
        assert h == pytest.approx(1.0)
        assert c == pytest.approx(1.0)
        assert f == pytest.approx(1.0)
        assert total == pytest.approx(1.0)

    def test_all_scores_in_zero_one(self):
        pts1 = np.random.rand(15, 2).astype(np.float64) * 50
        pts2 = np.random.rand(15, 2).astype(np.float64) * 50
        for s in score_boundary_pair(pts1, pts2):
            assert 0.0 <= s <= 1.0

    def test_custom_max_dist_affects_score(self):
        pts1 = _line_contour(10)
        pts2 = pts1 + np.array([[10.0, 0.0]])
        _, _, _, total_small = score_boundary_pair(pts1, pts2, max_dist=5.0)
        _, _, _, total_large = score_boundary_pair(pts1, pts2, max_dist=100.0)
        assert total_large > total_small

    def test_custom_weights(self):
        pts = _line_contour(10)
        # Только hausdorff: weight (1,0,0)
        h, c, f, total = score_boundary_pair(pts, pts, weights=(1.0, 0.0, 0.0))
        assert total == pytest.approx(1.0)

    def test_zero_weight_sum_ignored(self):
        pts = _line_contour(10)
        # weights (0,0,0) — граничный случай
        h, c, f, total = score_boundary_pair(pts, pts, weights=(0.0, 0.0, 0.0))
        # Должен вернуть что-то, не упасть
        assert isinstance(total, float)


# ─── match_boundary_pair ──────────────────────────────────────────────────────

class TestMatchBoundaryPair:
    def test_returns_boundary_match(self):
        cnt = _square_contour(0, 0, 100)
        bm  = match_boundary_pair(cnt, cnt)
        assert isinstance(bm, BoundaryMatch)

    def test_indices_stored(self):
        cnt = _square_contour(0, 0, 100)
        bm  = match_boundary_pair(cnt, cnt, idx1=4, idx2=9)
        assert bm.idx1 == 4
        assert bm.idx2 == 9

    def test_sides_stored(self):
        cnt = _square_contour(0, 0, 100)
        bm  = match_boundary_pair(cnt, cnt, side1=1, side2=3)
        assert bm.side1 == 1
        assert bm.side2 == 3

    def test_identical_contours_high_score(self):
        cnt = _circle_contour(50, 50, 30, 64)
        bm  = match_boundary_pair(cnt, cnt, side1=2, side2=2, n_points=20)
        assert bm.total_score > 0.8

    def test_total_score_in_zero_one(self):
        cnt1 = _square_contour(0, 0, 100)
        cnt2 = _square_contour(200, 200, 100)
        bm   = match_boundary_pair(cnt1, cnt2)
        assert 0.0 <= bm.total_score <= 1.0

    def test_n_points_in_params(self):
        cnt = _square_contour(0, 0, 100)
        bm  = match_boundary_pair(cnt, cnt, n_points=30)
        assert bm.params["n_points"] == 30

    def test_max_dist_in_params(self):
        cnt = _square_contour(0, 0, 100)
        bm  = match_boundary_pair(cnt, cnt, max_dist=75.0)
        assert bm.params["max_dist"] == pytest.approx(75.0)

    def test_weights_in_params(self):
        cnt = _square_contour(0, 0, 100)
        w   = (2.0, 1.0, 0.0)
        bm  = match_boundary_pair(cnt, cnt, weights=w)
        assert bm.params["weights"] == w

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        cnt = _circle_contour(50, 50, 30, 64)
        bm  = match_boundary_pair(cnt, cnt, side1=side, side2=side)
        assert isinstance(bm.total_score, float)


# ─── batch_match_boundaries ───────────────────────────────────────────────────

class TestBatchMatchBoundaries:
    def test_empty_pairs(self):
        cnts   = [_square_contour(0, 0, 50)]
        result = batch_match_boundaries(cnts, [])
        assert result == []

    def test_length_matches_pairs(self):
        cnts  = [_square_contour(i * 10, 0, 50) for i in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_match_boundaries(cnts, pairs)
        assert len(result) == 3

    def test_all_results_boundary_match(self):
        cnts  = [_circle_contour(50, 50, 30, 64) for _ in range(3)]
        pairs = [(0, 1), (1, 2)]
        result = batch_match_boundaries(cnts, pairs)
        for r in result:
            assert isinstance(r, BoundaryMatch)

    def test_indices_correct(self):
        cnts  = [_square_contour(i * 20, 0, 50) for i in range(4)]
        pairs = [(0, 3)]
        result = batch_match_boundaries(cnts, pairs)
        assert result[0].idx1 == 0
        assert result[0].idx2 == 3

    def test_custom_side_pairs(self):
        cnts       = [_square_contour(0, 0, 100), _square_contour(0, 0, 100)]
        pairs      = [(0, 1)]
        side_pairs = [(1, 3)]
        result = batch_match_boundaries(cnts, pairs, side_pairs=side_pairs)
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_default_side_pair_2_0(self):
        cnts  = [_square_contour(0, 0, 100), _square_contour(0, 0, 100)]
        pairs = [(0, 1)]
        result = batch_match_boundaries(cnts, pairs)
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_kwargs_forwarded(self):
        cnts  = [_square_contour(0, 0, 100), _square_contour(0, 0, 100)]
        pairs = [(0, 1)]
        result = batch_match_boundaries(cnts, pairs, n_points=25, max_dist=80.0)
        assert result[0].params["n_points"] == 25
        assert result[0].params["max_dist"] == pytest.approx(80.0)
