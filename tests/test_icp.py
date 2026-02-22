"""
Тесты для puzzle_reconstruction/matching/icp.py

Покрытие:
    ICPResult.transform    — применение трансформации (R, t)
    icp_align              — пустой вход, тождественная трансформация,
                             сходимость на совпадающих облаках, смещённые облака,
                             track_history, init_R/init_t
    contour_icp            — передискретизация, try_mirror, улучшение RMSE
    align_fragment_edge    — возвращает (translation, rmse), rmse ≥ 0
    _nearest_neighbors     — корректность (все индексы валидны)
    _best_fit_transform    — идентичная пара, известное смещение, det(R)=+1
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.matching.icp import (
    icp_align,
    contour_icp,
    align_fragment_edge,
    ICPResult,
    _nearest_neighbors,
    _best_fit_transform,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def circle_pts():
    """40 точек единичной окружности."""
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=-1)


@pytest.fixture
def square_pts():
    """Квадрат 1×1, 40 точек."""
    n = 10
    sides = [
        np.stack([np.linspace(0, 1, n), np.zeros(n)], axis=-1),
        np.stack([np.ones(n), np.linspace(0, 1, n)], axis=-1),
        np.stack([np.linspace(1, 0, n), np.ones(n)], axis=-1),
        np.stack([np.zeros(n), np.linspace(1, 0, n)], axis=-1),
    ]
    return np.concatenate(sides, axis=0)


# ─── ICPResult ────────────────────────────────────────────────────────────────

class TestICPResult:
    def test_transform_identity(self, circle_pts):
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=0.0,
                        n_iter=0, converged=True)
        out = res.transform(circle_pts)
        assert np.allclose(out, circle_pts, atol=1e-10)

    def test_transform_translation(self, circle_pts):
        t = np.array([3.0, -1.5])
        res = ICPResult(R=np.eye(2), t=t, rmse=0.0,
                        n_iter=0, converged=True)
        out = res.transform(circle_pts)
        assert np.allclose(out - circle_pts, t, atol=1e-10)

    def test_transform_rotation(self):
        angle = np.pi / 4
        c, s  = np.cos(angle), np.sin(angle)
        R     = np.array([[c, -s], [s, c]])
        pts   = np.array([[1.0, 0.0]])
        res   = ICPResult(R=R, t=np.zeros(2), rmse=0.0,
                           n_iter=0, converged=True)
        out   = res.transform(pts)
        expected = np.array([[c, s]])
        assert np.allclose(out, expected, atol=1e-10)

    def test_transform_output_shape(self, circle_pts):
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=0.0,
                        n_iter=0, converged=True)
        out = res.transform(circle_pts)
        assert out.shape == circle_pts.shape


# ─── icp_align: граничные случаи ──────────────────────────────────────────────

class TestICPAlignEdgeCases:
    def test_empty_source(self, circle_pts):
        res = icp_align(np.zeros((0, 2)), circle_pts)
        assert isinstance(res, ICPResult)
        assert not np.isfinite(res.rmse) or res.rmse >= 0

    def test_empty_target(self, circle_pts):
        res = icp_align(circle_pts, np.zeros((0, 2)))
        assert isinstance(res, ICPResult)

    def test_single_point(self):
        src = np.array([[1.0, 0.0]])
        tgt = np.array([[2.0, 0.0]])
        res = icp_align(src, tgt, max_iter=10)
        assert isinstance(res, ICPResult)
        assert res.n_iter >= 1

    def test_returns_icp_result(self, circle_pts):
        res = icp_align(circle_pts, circle_pts)
        assert isinstance(res, ICPResult)
        assert res.n_iter >= 1

    def test_identical_clouds_low_rmse(self, circle_pts):
        res = icp_align(circle_pts, circle_pts, max_iter=30)
        assert res.rmse < 0.1

    def test_converged_flag(self, circle_pts):
        res = icp_align(circle_pts, circle_pts, max_iter=50, tol=1e-4)
        assert res.converged


# ─── icp_align: трансформация восстанавливается ───────────────────────────────

class TestICPAlignTransform:
    def test_known_translation(self, circle_pts):
        """ICP должен найти сдвиг [2, 1] между облаками."""
        t_true = np.array([2.0, 1.0])
        target = circle_pts + t_true
        res    = icp_align(circle_pts, target, max_iter=50, tol=1e-6)
        recovered = res.transform(circle_pts)
        assert np.allclose(recovered, target, atol=0.2), \
            f"RMSE={res.rmse:.4f}, t={res.t}"

    def test_known_rotation(self):
        """ICP должен найти поворот на 30° для правильного замкнутого контура."""
        angle = np.pi / 6
        c, s  = np.cos(angle), np.sin(angle)
        R_true = np.array([[c, -s], [s, c]])
        t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        source = np.stack([np.cos(t), np.sin(t)], axis=-1)
        target = source @ R_true.T

        res = icp_align(source, target, max_iter=100, tol=1e-8)
        recovered = res.transform(source)
        rmse = float(np.sqrt(np.mean(np.sum((recovered - target) ** 2, axis=1))))
        assert rmse < 0.05, f"Rotation recovery RMSE={rmse:.4f}"

    def test_track_history(self, circle_pts):
        t_true = np.array([1.0, 0.5])
        target = circle_pts + t_true
        res = icp_align(circle_pts, target, max_iter=20, track_history=True)
        assert len(res.rmse_history) > 0
        assert len(res.rmse_history) == res.n_iter

    def test_history_decreasing(self, circle_pts):
        """RMSE должен монотонно убывать или оставаться стабильным."""
        t_true = np.array([3.0, 2.0])
        target = circle_pts + t_true
        res = icp_align(circle_pts, target, max_iter=30, track_history=True)
        h = res.rmse_history
        if len(h) > 1:
            # Допускаем небольшой временный рост (численная точность)
            violations = sum(1 for i in range(1, len(h)) if h[i] > h[i-1] + 1e-3)
            assert violations < len(h) // 4, "RMSE должен в основном убывать"

    def test_init_t_used(self, circle_pts):
        """init_t = истинный сдвиг → сходимость быстрее чем без него."""
        t_true = np.array([5.0, 5.0])
        target = circle_pts + t_true

        res_cold = icp_align(circle_pts, target, max_iter=100)
        res_warm  = icp_align(circle_pts, target, max_iter=100, init_t=t_true)

        # С правильной инициализацией RMSE не хуже
        assert res_warm.rmse <= res_cold.rmse + 0.5


# ─── contour_icp ──────────────────────────────────────────────────────────────

class TestContourICP:
    def test_returns_icp_result(self, square_pts):
        res = contour_icp(square_pts, square_pts, n_points=20)
        assert isinstance(res, ICPResult)

    def test_identical_low_rmse(self, circle_pts):
        res = contour_icp(circle_pts, circle_pts, n_points=30)
        assert res.rmse < 0.15

    def test_small_shift(self, circle_pts):
        target = circle_pts + np.array([0.1, 0.1])
        res = contour_icp(circle_pts, target, n_points=40, max_iter=50)
        assert res.rmse < 0.5

    def test_try_mirror_no_crash(self, square_pts):
        res = contour_icp(square_pts, square_pts, try_mirror=True, n_points=20)
        assert isinstance(res, ICPResult)

    def test_try_mirror_false(self, square_pts):
        res = contour_icp(square_pts, square_pts, try_mirror=False, n_points=20)
        assert isinstance(res, ICPResult)

    def test_n_points_resampling(self, circle_pts):
        """Разные n_points → оба работают без ошибок."""
        for n in (10, 50, 100):
            res = contour_icp(circle_pts, circle_pts, n_points=n)
            assert isinstance(res, ICPResult)


# ─── align_fragment_edge ──────────────────────────────────────────────────────

class TestAlignFragmentEdge:
    def test_returns_tuple(self, circle_pts):
        t, rmse = align_fragment_edge(circle_pts, circle_pts)
        assert isinstance(t, np.ndarray)
        assert t.shape == (2,)
        assert isinstance(rmse, float)

    def test_rmse_nonneg(self, circle_pts):
        _, rmse = align_fragment_edge(circle_pts, circle_pts)
        assert rmse >= 0.0

    def test_identical_curves(self, square_pts):
        t, rmse = align_fragment_edge(square_pts, square_pts, n_points=30)
        assert rmse < 0.2


# ─── _nearest_neighbors ───────────────────────────────────────────────────────

class TestNearestNeighbors:
    def test_indices_in_range(self, circle_pts):
        n_tgt = 15
        tgt   = circle_pts[:n_tgt]
        idx   = _nearest_neighbors(circle_pts, tgt)
        assert idx.shape == (len(circle_pts),)
        assert idx.min() >= 0
        assert idx.max() < n_tgt

    def test_self_match(self, circle_pts):
        """Каждая точка должна ближе всего к себе самой."""
        idx = _nearest_neighbors(circle_pts, circle_pts)
        assert np.all(idx == np.arange(len(circle_pts)))

    def test_small_arrays(self):
        src = np.array([[0.0, 0.0], [1.0, 0.0]])
        tgt = np.array([[0.1, 0.0], [1.1, 0.0], [2.0, 2.0]])
        idx = _nearest_neighbors(src, tgt)
        assert idx[0] == 0
        assert idx[1] == 1


# ─── _best_fit_transform ──────────────────────────────────────────────────────

class TestBestFitTransform:
    def test_identity_on_identical(self, circle_pts):
        R, t = _best_fit_transform(circle_pts, circle_pts)
        assert np.allclose(R, np.eye(2), atol=1e-8)
        assert np.allclose(t, np.zeros(2), atol=1e-8)

    def test_known_translation(self, circle_pts):
        t_true = np.array([3.0, -2.0])
        target = circle_pts + t_true
        R, t   = _best_fit_transform(circle_pts, target)
        assert np.allclose(R, np.eye(2), atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)

    def test_det_R_positive(self, circle_pts):
        """det(R) должен быть +1 (поворот, не отражение)."""
        # Создаём случайную целевую облако
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(circle_pts))
        R, _ = _best_fit_transform(circle_pts, circle_pts[perm])
        det = np.linalg.det(R)
        assert math.isclose(det, 1.0, abs_tol=1e-6), f"det(R)={det}"

    def test_output_shapes(self, circle_pts):
        R, t = _best_fit_transform(circle_pts, circle_pts)
        assert R.shape == (2, 2)
        assert t.shape == (2,)
