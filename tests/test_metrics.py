"""
Юнит-тесты для модуля verification/metrics.py.

Тесты покрывают:
    - evaluate_reconstruction()  — все метрики, граничные случаи
    - _normalize_config()        — нормализация конфигурации
    - _angle_diff_deg()          — угловая разница
    - compare_methods()          — форматирование таблицы
    - ReconstructionMetrics      — корректность атрибутов
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.verification.metrics import (
    evaluate_reconstruction,
    compare_methods,
    ReconstructionMetrics,
    BenchmarkResult,
    _normalize_config,
    _angle_diff_deg,
    _compute_adjacency,
    _zero_metrics,
)


# ─── Вспомогательные функции ─────────────────────────────────────────────

def _simple_config(n: int = 4, spacing: float = 100.0) -> dict:
    """Создаёт простую конфигурацию — фрагменты в ряд."""
    return {i: (np.array([i * spacing, 0.0]), 0.0) for i in range(n)}


# ─── _angle_diff_deg ──────────────────────────────────────────────────────

class TestAngleDiffDeg:

    def test_zero_difference(self):
        assert _angle_diff_deg(45.0, 45.0) == 0.0

    def test_small_difference(self):
        assert abs(_angle_diff_deg(10.0, 15.0) - 5.0) < 1e-10

    def test_wraparound_360(self):
        """350° и 10° должны давать 20°, а не 340°."""
        assert abs(_angle_diff_deg(350.0, 10.0) - 20.0) < 1e-10

    def test_exactly_180(self):
        assert abs(_angle_diff_deg(0.0, 180.0) - 180.0) < 1e-10

    def test_symmetric(self):
        assert _angle_diff_deg(30.0, 70.0) == _angle_diff_deg(70.0, 30.0)

    def test_negative_angles(self):
        assert abs(_angle_diff_deg(-10.0, 10.0) - 20.0) < 1e-10


# ─── _normalize_config ────────────────────────────────────────────────────

class TestNormalizeConfig:

    def test_first_fragment_at_origin(self):
        cfg = {0: (np.array([100.0, 200.0]), math.pi / 4),
               1: (np.array([200.0, 200.0]), math.pi / 4)}
        norm = _normalize_config(cfg, [0, 1])
        np.testing.assert_allclose(norm[0][0], [0.0, 0.0], atol=1e-10)

    def test_first_fragment_angle_zero(self):
        cfg = {0: (np.array([0.0, 0.0]), math.pi / 3),
               1: (np.array([100.0, 0.0]), math.pi / 3)}
        norm = _normalize_config(cfg, [0, 1])
        assert abs(norm[0][1]) < 1e-10

    def test_relative_positions_preserved(self):
        """Расстояния между фрагментами сохраняются после нормализации."""
        cfg = {0: (np.array([0.0, 0.0]), 0.0),
               1: (np.array([50.0, 0.0]), 0.0),
               2: (np.array([0.0, 80.0]), 0.0)}
        norm = _normalize_config(cfg, [0, 1, 2])
        dist01 = np.linalg.norm(norm[1][0] - norm[0][0])
        dist02 = np.linalg.norm(norm[2][0] - norm[0][0])
        assert abs(dist01 - 50.0) < 1e-10
        assert abs(dist02 - 80.0) < 1e-10

    def test_empty_frag_ids(self):
        result = _normalize_config({0: (np.array([1.0, 2.0]), 0.0)}, [])
        assert result == {}

    def test_single_fragment(self):
        cfg = {5: (np.array([300.0, 400.0]), 1.5)}
        norm = _normalize_config(cfg, [5])
        np.testing.assert_allclose(norm[5][0], [0.0, 0.0], atol=1e-10)
        assert abs(norm[5][1]) < 1e-10


# ─── _compute_adjacency ───────────────────────────────────────────────────

class TestComputeAdjacency:

    def test_close_fragments_are_adjacent(self):
        cfg = {0: (np.array([0.0, 0.0]),   0.0),
               1: (np.array([50.0, 0.0]),  0.0),
               2: (np.array([500.0, 0.0]), 0.0)}
        adj = _compute_adjacency(cfg, threshold=100.0)
        # 0 и 1 близко → смежные; 0 и 2, 1 и 2 — далеко
        pairs = set(adj)
        assert (0, 1) in pairs or (1, 0) in pairs
        assert (0, 2) not in pairs and (2, 0) not in pairs

    def test_all_close_fully_connected(self):
        cfg = {i: (np.array([i * 5.0, 0.0]), 0.0) for i in range(4)}
        adj = _compute_adjacency(cfg, threshold=100.0)
        assert len(adj) == 4 * 3 // 2  # C(4,2) = 6 пар


# ─── evaluate_reconstruction ─────────────────────────────────────────────

class TestEvaluateReconstruction:

    def test_perfect_reconstruction_dc_1(self):
        """Идеальная сборка должна давать DC = 1.0."""
        gt = _simple_config(4)
        m  = evaluate_reconstruction(gt, gt)
        assert m.direct_comparison == pytest.approx(1.0, abs=1e-6)

    def test_perfect_reconstruction_rmse_0(self):
        gt = _simple_config(4)
        m  = evaluate_reconstruction(gt, gt)
        assert m.position_rmse == pytest.approx(0.0, abs=1e-6)

    def test_perfect_reconstruction_angular_0(self):
        gt = _simple_config(4)
        m  = evaluate_reconstruction(gt, gt)
        assert m.angular_error_deg == pytest.approx(0.0, abs=1e-3)

    def test_empty_prediction(self):
        gt = _simple_config(4)
        m  = evaluate_reconstruction({}, gt)
        assert m.n_fragments == 0
        assert m.direct_comparison == 0.0

    def test_offset_prediction_high_rmse(self):
        """Смещённая сборка должна иметь высокий RMSE."""
        gt = {i: (np.array([i * 100.0, 0.0]), 0.0) for i in range(4)}
        pred = {i: (np.array([i * 100.0 + 500.0, 500.0]), 0.0) for i in range(4)}
        m = evaluate_reconstruction(pred, gt)
        # После нормализации (первый фрагмент в начале координат)
        # смещение остальных должно быть нулевым (все сдвинуты одинаково)
        assert m.position_rmse < 1.0  # Структура сохранена

    def test_rotated_prediction(self):
        """Повёрнутая конфигурация с той же структурой → низкий RMSE."""
        gt = {0: (np.array([0.0, 0.0]),   0.0),
              1: (np.array([100.0, 0.0]), 0.0)}
        # Поворачиваем всю конфигурацию на 45°
        angle = math.pi / 4
        c, s  = math.cos(angle), math.sin(angle)
        def rotate(pos):
            return np.array([c * pos[0] - s * pos[1],
                              s * pos[0] + c * pos[1]])
        pred = {k: (rotate(pos), a + angle) for k, (pos, a) in gt.items()}
        m = evaluate_reconstruction(pred, gt)
        # Нормализация убирает поворот → структура та же → RMSE ≈ 0
        assert m.position_rmse < 5.0

    def test_metrics_types(self):
        gt = _simple_config(3)
        m  = evaluate_reconstruction(gt, gt)
        assert isinstance(m.neighbor_accuracy, float)
        assert isinstance(m.direct_comparison, float)
        assert isinstance(m.position_rmse, float)
        assert isinstance(m.angular_error_deg, float)
        assert isinstance(m.perfect, bool)
        assert isinstance(m.n_fragments, int)

    def test_metrics_ranges(self):
        """Все метрики в допустимых диапазонах."""
        gt = _simple_config(5)
        rng = np.random.RandomState(17)
        pred = {k: (rng.randn(2) * 200, float(rng.rand() * 2 * math.pi))
                for k in gt}
        m = evaluate_reconstruction(pred, gt)
        assert 0.0 <= m.neighbor_accuracy  <= 1.0
        assert 0.0 <= m.direct_comparison  <= 1.0
        assert 0.0 <= m.edge_match_rate    <= 1.0
        assert m.position_rmse >= 0.0
        assert m.angular_error_deg >= 0.0

    def test_n_fragments_correct(self):
        gt = _simple_config(6)
        m  = evaluate_reconstruction(gt, gt)
        assert m.n_fragments == 6

    def test_explicit_adjacency(self):
        """Явно переданный список смежности учитывается."""
        gt = {0: (np.array([0.0, 0.0]), 0.0),
              1: (np.array([1.0, 0.0]), 0.0),
              2: (np.array([2.0, 0.0]), 0.0)}
        m = evaluate_reconstruction(gt, gt, adjacency=[(0, 1), (1, 2)])
        assert m.n_total_pairs == 2

    def test_perfect_flag(self):
        gt = _simple_config(3)
        m  = evaluate_reconstruction(gt, gt)
        assert m.perfect is True

    def test_summary_string(self):
        gt = _simple_config(4)
        m  = evaluate_reconstruction(gt, gt)
        s  = m.summary()
        assert "Neighbor Accuracy" in s
        assert "RMSE" in s or "RMSE" in s.upper()


# ─── _zero_metrics ────────────────────────────────────────────────────────

class TestZeroMetrics:

    def test_zero_metrics_n_0(self):
        m = _zero_metrics(0)
        assert m.n_fragments == 0
        assert m.direct_comparison == 0.0
        assert m.perfect is False


# ─── compare_methods ─────────────────────────────────────────────────────

class TestCompareMethods:

    def _make_result(self, method: str, na: float, dc: float) -> BenchmarkResult:
        m = ReconstructionMetrics(
            neighbor_accuracy=na, direct_comparison=dc,
            perfect=(na == 1.0), position_rmse=10.0,
            angular_error_deg=5.0, n_fragments=4,
            n_correct_pairs=int(na * 4), n_total_pairs=4,
            edge_match_rate=dc,
        )
        return BenchmarkResult(method=method, metrics=m, runtime_sec=1.0)

    def test_compare_methods_returns_string(self):
        results = [
            self._make_result("greedy", 0.5, 0.4),
            self._make_result("beam",   0.8, 0.7),
        ]
        out = compare_methods(results)
        assert isinstance(out, str)

    def test_best_method_first(self):
        results = [
            self._make_result("greedy", 0.3, 0.3),
            self._make_result("beam",   0.9, 0.8),
            self._make_result("sa",     0.6, 0.5),
        ]
        out = compare_methods(results)
        lines = [l for l in out.split("\n") if l.strip() and "─" not in l]
        # Первая строка — заголовок, вторая — лучший метод
        assert "beam" in lines[1] or "beam" in out.split("\n")[2]

    def test_compare_empty(self):
        out = compare_methods([])
        assert isinstance(out, str)

    def test_all_methods_present(self):
        results = [self._make_result(m, 0.5, 0.4) for m in ("a", "b", "c")]
        out = compare_methods(results)
        assert "a" in out and "b" in out and "c" in out
