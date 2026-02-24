"""
Дескриптор Shape Context (контекст формы) для сопоставления контуров.

Shape Context (Belongie et al., 2002) описывает форму вокруг каждой точки
контура через лог-полярную гистограмму распределения остальных точек.
Это позволяет находить соответствия между точками двух контуров даже при
частичных деформациях и шуме.

Применение в пазлах:
    - Описание «профиля» края фрагмента.
    - Поиск соответствий между краями разных фрагментов.
    - Дополнение DTW и CSS дескрипторов для более устойчивого сравнения.

Функции:
    compute_shape_context    — вычисляет SC-дескрипторы для набора точек
    shape_context_distance   — χ²-расстояние между двумя SC-дескрипторами
    match_shape_contexts     — Hungarian-match двух наборов SC-дескрипторов
    normalize_shape_context  — L1-нормализация
    log_polar_histogram      — лог-полярная гистограмма для одной точки

Классы:
    ShapeContextResult       — результат вычисления SC
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ─── Константы по умолчанию ───────────────────────────────────────────────────

DEFAULT_N_BINS_R     = 5    # Логарифмических колец
DEFAULT_N_BINS_THETA = 12   # Угловых секторов (30° каждый)
DEFAULT_R_INNER      = 0.1  # Внутренний радиус (в долях max_dist)
DEFAULT_R_OUTER      = 2.0  # Внешний радиус  (в долях mean_dist)


# ─── ShapeContextResult ───────────────────────────────────────────────────────

@dataclass
class ShapeContextResult:
    """
    Результат вычисления Shape Context.

    Attributes:
        descriptors: (N, n_bins_r × n_bins_theta) — SC-дескрипторы для N точек.
        points:      (N, 2) — координаты точек.
        mean_dist:   Среднее попарное расстояние (использ. для нормализации).
        n_bins_r:    Число колец.
        n_bins_theta: Число угловых секторов.
    """
    descriptors:  np.ndarray
    points:       np.ndarray
    mean_dist:    float
    n_bins_r:     int
    n_bins_theta: int

    @property
    def descriptor_dim(self) -> int:
        return self.n_bins_r * self.n_bins_theta

    def __repr__(self) -> str:
        n = len(self.points)
        return (f"ShapeContextResult(N={n}, "
                f"bins=({self.n_bins_r},{self.n_bins_theta}), "
                f"mean_dist={self.mean_dist:.2f})")


# ─── Вычисление Shape Context ─────────────────────────────────────────────────

def compute_shape_context(points:      np.ndarray,
                            n_bins_r:    int   = DEFAULT_N_BINS_R,
                            n_bins_theta: int  = DEFAULT_N_BINS_THETA,
                            r_inner:     float = DEFAULT_R_INNER,
                            r_outer:     float = DEFAULT_R_OUTER,
                            normalize:   bool  = True) -> ShapeContextResult:
    """
    Вычисляет Shape Context дескриптор для каждой точки набора.

    Args:
        points:       (N, 2) массив точек контура (float).
        n_bins_r:     Число логарифмических радиальных колец.
        n_bins_theta: Число угловых секторов.
        r_inner:      Минимальный радиус (в долях mean_dist).
        r_outer:      Максимальный радиус (в долях mean_dist).
        normalize:    True → L1-нормализовать каждый дескриптор.

    Returns:
        ShapeContextResult с полем descriptors формы (N, n_bins_r × n_bins_theta).

    Raises:
        ValueError: Если N < 2 или points не 2D.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points должен быть (N, 2), получено: {pts.shape}")

    n = len(pts)
    if n < 2:
        # Вырожденный случай: один дескриптор нулей
        desc = np.zeros((max(n, 1), n_bins_r * n_bins_theta), dtype=np.float64)
        return ShapeContextResult(
            descriptors=desc,
            points=pts,
            mean_dist=0.0,
            n_bins_r=n_bins_r,
            n_bins_theta=n_bins_theta,
        )

    # ── Вычисление всех попарных расстояний и углов ───────────────────────
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    dists = np.linalg.norm(diff, axis=2)                   # (N, N)
    angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])      # (N, N) ∈ [-π, π]

    # Среднее попарное расстояние (нормализатор)
    mean_dist = float(dists[dists > 0].mean()) if (dists > 0).any() else 1.0

    # ── Логарифмические кольца ────────────────────────────────────────────
    r_min  = r_inner * mean_dist
    r_max  = r_outer * mean_dist
    # Логарифмически равные кольца от r_min до r_max
    r_bins = np.logspace(np.log10(max(r_min, 1e-6)),
                          np.log10(max(r_max, 1e-6 + 1)),
                          n_bins_r + 1)

    # ── Угловые секторы ───────────────────────────────────────────────────
    theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta + 1)

    # ── Дескрипторы ───────────────────────────────────────────────────────
    descriptors = np.zeros((n, n_bins_r * n_bins_theta), dtype=np.float64)

    for i in range(n):
        dist_i  = dists[i]   # (N,)
        angle_i = angles[i]  # (N,)

        # Исключаем точку саму с собой
        mask = np.arange(n) != i

        h = log_polar_histogram(
            dist_i[mask], angle_i[mask],
            r_bins=r_bins, theta_bins=theta_bins,
            n_bins_r=n_bins_r, n_bins_theta=n_bins_theta,
        )

        if normalize and h.sum() > 0:
            h = h / h.sum()

        descriptors[i] = h

    return ShapeContextResult(
        descriptors=descriptors,
        points=pts,
        mean_dist=mean_dist,
        n_bins_r=n_bins_r,
        n_bins_theta=n_bins_theta,
    )


def log_polar_histogram(dists:       np.ndarray,
                          angles:      np.ndarray,
                          r_bins:      np.ndarray,
                          theta_bins:  np.ndarray,
                          n_bins_r:    int,
                          n_bins_theta: int) -> np.ndarray:
    """
    Строит лог-полярную гистограмму для набора расстояний и углов.

    Args:
        dists:       (M,) расстояния от референсной точки.
        angles:      (M,) углы ∈ [-π, π].
        r_bins:      Границы радиальных колец (len = n_bins_r + 1).
        theta_bins:  Границы угловых секторов (len = n_bins_theta + 1).
        n_bins_r:    Число радиальных колец.
        n_bins_theta: Число угловых секторов.

    Returns:
        (n_bins_r × n_bins_theta,) гистограмма.
    """
    h = np.zeros((n_bins_r, n_bins_theta), dtype=np.float64)

    r_idx     = np.digitize(dists,  r_bins)     - 1   # 0..n_bins_r
    theta_idx = np.digitize(angles, theta_bins) - 1   # 0..n_bins_theta

    # Ограничиваем индексы
    r_idx     = np.clip(r_idx,     0, n_bins_r     - 1)
    theta_idx = np.clip(theta_idx, 0, n_bins_theta - 1)

    # Фильтруем точки вне диапазона r
    in_range = (dists >= r_bins[0]) & (dists < r_bins[-1])
    r_idx_f     = r_idx[in_range]
    theta_idx_f = theta_idx[in_range]

    np.add.at(h, (r_idx_f, theta_idx_f), 1)
    return h.ravel()


# ─── Расстояние ───────────────────────────────────────────────────────────────

def shape_context_distance(sc1: np.ndarray,
                             sc2: np.ndarray,
                             eps: float = 1e-10) -> float:
    """
    χ²-расстояние (Chi-squared distance) между двумя SC-дескрипторами.

    χ²(h₁, h₂) = 0.5 × Σ (h₁ᵢ - h₂ᵢ)² / (h₁ᵢ + h₂ᵢ + ε)

    Args:
        sc1: (D,) — первый дескриптор (L1-нормированный).
        sc2: (D,) — второй дескриптор (L1-нормированный).
        eps: Защита от деления на ноль.

    Returns:
        Расстояние ∈ [0, 0.5] (0 = идентичные дескрипторы).
    """
    a, b = np.asarray(sc1, dtype=np.float64), np.asarray(sc2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Формы дескрипторов не совпадают: {a.shape} vs {b.shape}")
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + eps)))


def normalize_shape_context(sc: np.ndarray) -> np.ndarray:
    """L1-нормализация: делит на сумму (или возвращает нули)."""
    total = sc.sum()
    return sc / total if total > 0 else sc.copy()


# ─── Сопоставление ────────────────────────────────────────────────────────────

def match_shape_contexts(result_a:   ShapeContextResult,
                          result_b:   ShapeContextResult) -> Tuple[float, np.ndarray]:
    """
    Находит оптимальное соответствие точек двух контуров через Hungarian method.

    Строит матрицу χ²-расстояний (N_a × N_b) и минимизирует суммарное
    расстояние назначения.

    Args:
        result_a: ShapeContextResult первого контура.
        result_b: ShapeContextResult второго контура.

    Returns:
        (cost, correspondence) — суммарная стоимость и массив (min(N_a,N_b), 2)
        пар индексов (idx_a, idx_b).

    Raises:
        ImportError: Если scipy недоступен.
    """
    da = result_a.descriptors  # (N_a, D)
    db = result_b.descriptors  # (N_b, D)

    # Матрица попарных χ²-расстояний (N_a, N_b)
    cost_matrix = _chi2_cost_matrix(da, db)

    if _SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Простой жадный fallback (не оптимальный, но без scipy)
        row_ind, col_ind = _greedy_assignment(cost_matrix)

    cost = float(cost_matrix[row_ind, col_ind].sum())
    correspondence = np.stack([row_ind, col_ind], axis=1)
    return cost, correspondence


def _chi2_cost_matrix(da: np.ndarray, db: np.ndarray,
                        eps: float = 1e-10) -> np.ndarray:
    """Вычисляет матрицу χ²-расстояний (N_a, N_b) векторизованно."""
    na, nb = len(da), len(db)
    # (N_a, N_b, D)
    a_exp = da[:, np.newaxis, :]
    b_exp = db[np.newaxis, :, :]
    cost  = 0.5 * np.sum((a_exp - b_exp) ** 2 / (a_exp + b_exp + eps), axis=2)
    return cost  # (N_a, N_b)


def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Жадный алгоритм назначения (O(min(N,M)²)). Fallback без scipy."""
    na, nb = cost.shape
    n      = min(na, nb)
    used_a = set()
    used_b = set()
    row_idx, col_idx = [], []

    flat_order = np.argsort(cost.ravel())
    for idx in flat_order:
        r = idx // nb
        c = idx % nb
        if r not in used_a and c not in used_b:
            row_idx.append(r)
            col_idx.append(c)
            used_a.add(r)
            used_b.add(c)
            if len(row_idx) == n:
                break

    return np.array(row_idx, dtype=np.intp), np.array(col_idx, dtype=np.intp)


# ─── Удобные обёртки ──────────────────────────────────────────────────────────

def contour_similarity(contour_a: np.ndarray,
                         contour_b: np.ndarray,
                         n_sample:  int = 40,
                         **sc_kwargs) -> float:
    """
    Вычисляет нормированное сходство двух контуров по Shape Context.

    Args:
        contour_a: (N, 2) или (N, 1, 2) первый контур.
        contour_b: (M, 2) или (M, 1, 2) второй контур.
        n_sample:  Число точек, на которые передискретизируются контуры.
        **sc_kwargs: Дополнительные параметры compute_shape_context.

    Returns:
        Сходство ∈ [0, 1] (1 = идентичные контуры).
    """
    # Guard: degenerate contours with fewer than 2 unique points
    raw_a = np.unique(np.asarray(contour_a, dtype=np.float64).reshape(-1, 2), axis=0)
    raw_b = np.unique(np.asarray(contour_b, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(raw_a) < 2 or len(raw_b) < 2:
        return 0.0

    pts_a = _preprocess_contour(contour_a, n_sample)
    pts_b = _preprocess_contour(contour_b, n_sample)

    if len(pts_a) < 2 or len(pts_b) < 2:
        return 0.0

    sc_a = compute_shape_context(pts_a, **sc_kwargs)
    sc_b = compute_shape_context(pts_b, **sc_kwargs)

    cost, _ = match_shape_contexts(sc_a, sc_b)

    # Нормируем: max возможная стоимость ≈ 0.5 × n_sample
    max_cost = 0.5 * n_sample
    similarity = max(0.0, 1.0 - cost / (max_cost + 1e-8))
    return float(np.clip(similarity, 0.0, 1.0))


def _preprocess_contour(contour: np.ndarray, n_sample: int) -> np.ndarray:
    """Выравнивает форму и передискретизирует контур до n_sample точек."""
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.squeeze(1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float64)
    if len(pts) == 0:
        return pts

    # Передискретизация через равномерное прореживание / интерполяцию
    n = len(pts)
    if n == n_sample:
        return pts
    if n < n_sample:
        indices = np.round(np.linspace(0, n - 1, n_sample)).astype(int)
    else:
        indices = np.round(np.linspace(0, n - 1, n_sample)).astype(int)
    return pts[indices]
