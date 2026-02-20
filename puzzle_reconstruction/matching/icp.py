"""
Iterative Closest Point (ICP) для точного выравнивания контуров фрагментов.

Назначение:
    После того как матрица совместимости выявила перспективные пары краёв,
    ICP уточняет пространственное выравнивание — находит оптимальный поворот
    и сдвиг, минимизирующий среднеквадратическое расстояние между контурами.

Алгоритм:
    1. Инициализация: совместить центроиды (или использовать начальное R, t).
    2. На каждой итерации:
       a. Для каждой точки source найти ближайшую точку в target (k-d поиск).
       b. Вычислить оптимальное R, t методом SVD (Umeyama 1991).
       c. Применить трансформацию к source.
       d. Проверить сходимость (изменение RMSE < tol).
    3. Вернуть ICPResult с матрицей поворота, вектором сдвига и итоговым RMSE.

Функции:
    icp_align          — основная функция ICP
    contour_icp        — обёртка над icp_align для пары контуров
    align_fragment_edge — финальное выравнивание двух EdgeSignature
    ICPResult          — результат выравнивания
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..utils.geometry import resample_curve, rotate_points, align_centroids


# ─── Результат ICP ────────────────────────────────────────────────────────────

@dataclass
class ICPResult:
    """
    Результат выравнивания ICP.

    Attributes:
        R:           Матрица поворота 2×2 (применить к source).
        t:           Вектор сдвига (2,) (применить после поворота).
        rmse:        Итоговое RMSE (после сходимости или max_iter).
        n_iter:      Число выполненных итераций.
        converged:   True если алгоритм сошёлся (улучшение < tol).
        rmse_history: RMSE на каждой итерации (только если track_history=True).
    """
    R:            np.ndarray          # (2, 2)
    t:            np.ndarray          # (2,)
    rmse:         float
    n_iter:       int
    converged:    bool
    rmse_history: List[float] = field(default_factory=list)

    def transform(self, pts: np.ndarray) -> np.ndarray:
        """
        Применяет трансформацию (R, t) к набору точек.

        Args:
            pts: (N, 2).

        Returns:
            (N, 2) — трансформированные точки.
        """
        pts = np.asarray(pts, dtype=np.float64)
        return pts @ self.R.T + self.t


# ─── Алгоритм ICP ────────────────────────────────────────────────────────────

def icp_align(source:       np.ndarray,
              target:       np.ndarray,
              max_iter:     int   = 50,
              tol:          float = 1e-5,
              init_R:       Optional[np.ndarray] = None,
              init_t:       Optional[np.ndarray] = None,
              track_history: bool = False) -> ICPResult:
    """
    Выравнивает *source* к *target* методом ICP.

    Args:
        source:        (N, 2) — исходные точки (трансформируются).
        target:        (M, 2) — целевые точки (фиксированы).
        max_iter:      Максимальное число итераций.
        tol:           Порог сходимости (∆RMSE < tol → стоп).
        init_R:        Начальная матрица поворота 2×2. None → единичная.
        init_t:        Начальный сдвиг (2,). None → совмещение центроидов.
        track_history: Записывать ли RMSE на каждой итерации.

    Returns:
        ICPResult с финальными R, t, rmse, n_iter, converged.
    """
    src = np.asarray(source, dtype=np.float64).copy()
    tgt = np.asarray(target, dtype=np.float64)

    if len(src) == 0 or len(tgt) == 0:
        return ICPResult(
            R=np.eye(2), t=np.zeros(2),
            rmse=float("inf"), n_iter=0, converged=False,
        )

    # Накопленная трансформация
    R_total = np.eye(2, dtype=np.float64) if init_R is None else np.array(init_R, dtype=np.float64)
    t_total = np.zeros(2, dtype=np.float64)

    # Начальная позиция: совмещаем центроиды если init_t не задан
    if init_t is None:
        t_init  = tgt.mean(axis=0) - src.mean(axis=0)
    else:
        t_init  = np.asarray(init_t, dtype=np.float64)

    src     = src @ R_total.T + t_init
    t_total = t_init.copy()

    prev_rmse  = float("inf")
    history    = []
    converged  = False

    for iteration in range(max_iter):
        # ── a. Ближайшие соответствия ─────────────────────────────────────
        indices = _nearest_neighbors(src, tgt)
        matched = tgt[indices]

        # ── b. Оптимальная трансформация (SVD / Умеяма) ───────────────────
        R_step, t_step = _best_fit_transform(src, matched)

        # ── c. Применяем трансформацию ────────────────────────────────────
        src = src @ R_step.T + t_step

        # Обновляем суммарную трансформацию
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        # ── d. RMSE и сходимость ──────────────────────────────────────────
        rmse = float(np.sqrt(np.mean(np.sum((src - matched) ** 2, axis=1))))
        if track_history:
            history.append(rmse)

        if abs(prev_rmse - rmse) < tol:
            converged = True
            prev_rmse = rmse
            break

        prev_rmse = rmse

    return ICPResult(
        R=R_total,
        t=t_total,
        rmse=prev_rmse,
        n_iter=iteration + 1,
        converged=converged,
        rmse_history=history,
    )


# ─── Обёртки высокого уровня ──────────────────────────────────────────────────

def contour_icp(contour_a:   np.ndarray,
                contour_b:   np.ndarray,
                n_points:    int  = 100,
                max_iter:    int  = 50,
                tol:         float = 1e-5,
                try_mirror:  bool = True) -> ICPResult:
    """
    Выравнивает контур *contour_a* к *contour_b* через ICP.

    Перед выравниванием оба контура передискретизируются до *n_points* точек.
    Если *try_mirror=True*, дополнительно проверяется зеркальная копия source
    и возвращается лучший результат по RMSE.

    Args:
        contour_a:  (N, 2) — источник (двигается к цели).
        contour_b:  (M, 2) — цель.
        n_points:   Число точек после передискретизации.
        max_iter:   Максимальное число итераций ICP.
        tol:        Порог сходимости.
        try_mirror: Проверять ли зеркало source (для разрыва).

    Returns:
        ICPResult с наилучшим выравниванием.
    """
    src = resample_curve(np.asarray(contour_a, dtype=np.float64), n_points)
    tgt = resample_curve(np.asarray(contour_b, dtype=np.float64), n_points)

    result_fwd = icp_align(src, tgt, max_iter=max_iter, tol=tol)

    if not try_mirror:
        return result_fwd

    # Зеркало по оси X относительно центроида
    src_m = src.copy()
    src_m[:, 0] = -src_m[:, 0] + 2 * src_m[:, 0].mean()
    result_mir  = icp_align(src_m, tgt, max_iter=max_iter, tol=tol)

    return result_fwd if result_fwd.rmse <= result_mir.rmse else result_mir


def align_fragment_edge(edge_curve_a: np.ndarray,
                        edge_curve_b: np.ndarray,
                        n_points: int = 80) -> Tuple[np.ndarray, float]:
    """
    Вычисляет оптимальный сдвиг для совмещения двух краёв фрагментов.

    Используется как постобработка после сборки: уточняет стык двух
    фрагментов, добавляя субпиксельную точность к position placement.

    Args:
        edge_curve_a: (N, 2) — кривая края фрагмента A.
        edge_curve_b: (M, 2) — кривая края фрагмента B.
        n_points:     Число точек для передискретизации.

    Returns:
        (translation: (2,), rmse: float) — вектор сдвига и итоговый RMSE.
    """
    result = contour_icp(edge_curve_a, edge_curve_b,
                          n_points=n_points, try_mirror=True)
    return result.t, result.rmse


# ─── Внутренние функции ───────────────────────────────────────────────────────

def _nearest_neighbors(source: np.ndarray,
                        target: np.ndarray) -> np.ndarray:
    """
    Для каждой точки source находит индекс ближайшей точки в target.

    Используется матричный подход O(N·M) — без k-d дерева для совместимости
    с малыми N (контуры 50–200 точек). При N·M > 50000 автоматически
    переключается на scipy.spatial.KDTree.

    Returns:
        (N,) int — индексы ближайших точек в target.
    """
    n, m = len(source), len(target)

    if n * m <= 50_000:
        # Матричный подход: O(N·M)
        diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]  # (N, M, 2)
        dist = np.sum(diff ** 2, axis=-1)                            # (N, M)
        return np.argmin(dist, axis=1)                               # (N,)

    # При большом числе точек используем KDTree
    try:
        from scipy.spatial import KDTree
        tree = KDTree(target)
        _, idx = tree.query(source)
        return idx
    except ImportError:
        # Резерв: матричный подход блоками
        block = 512
        indices = np.empty(n, dtype=int)
        for i in range(0, n, block):
            chunk = source[i:i + block]
            diff  = chunk[:, np.newaxis, :] - target[np.newaxis, :, :]
            dist  = np.sum(diff ** 2, axis=-1)
            indices[i:i + block] = np.argmin(dist, axis=1)
        return indices


def _best_fit_transform(source: np.ndarray,
                         target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Оптимальный поворот R и сдвиг t методом SVD (Умеяма 1991).

    Минимизирует sum_i ||R·p_i + t - q_i||^2

    Args:
        source, target: (N, 2) — соответствующие пары точек.

    Returns:
        (R: (2, 2), t: (2,))
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    S = source - mu_s
    T = target - mu_t

    H   = S.T @ T                        # (2, 2)
    U, _, Vt = np.linalg.svd(H)

    # Обеспечиваем det(R) = +1 (не отражение)
    d   = np.linalg.det(Vt.T @ U.T)
    D   = np.diag([1.0, d])

    R   = (Vt.T @ D @ U.T)
    t   = mu_t - R @ mu_s

    return R, t
