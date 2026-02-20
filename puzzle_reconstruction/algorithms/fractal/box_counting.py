"""
Фрактальная размерность методом Box-counting.

Алгоритм Ричардсона-Мандельброта: при уменьшении размера ячейки сетки r
количество занятых ячеек N(r) растёт как степенной закон: N ~ r^(-FD).
"""
import numpy as np


def box_counting_fd(contour: np.ndarray,
                    n_scales: int = 8) -> float:
    """
    Вычисляет фрактальную размерность контура методом Box-counting.

    Args:
        contour:  (N, 2) координаты точек контура.
        n_scales: Число масштабов (степени 2).

    Returns:
        FD ∈ [1.0, 2.0] — фрактальная размерность.
        1.0 = гладкая линия, 2.0 = заполняет плоскость.
    """
    pts = contour.astype(np.float64)
    if len(pts) < 4:
        return 1.0

    # Нормируем к [0, 1] для масштабо-независимости
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = (maxs - mins).max()
    if span == 0:
        return 1.0
    pts_norm = (pts - mins) / span

    log_r_inv = []
    log_N     = []

    for k in range(1, n_scales + 1):
        # Размер ячейки: r = 1/2^k
        n_bins = 2 ** k
        # Дискретизируем точки в сетку
        ix = np.floor(pts_norm[:, 0] * n_bins).astype(np.int32)
        iy = np.floor(pts_norm[:, 1] * n_bins).astype(np.int32)
        # Граничный случай: точки ровно в 1.0
        ix = np.clip(ix, 0, n_bins - 1)
        iy = np.clip(iy, 0, n_bins - 1)

        N = len(set(zip(ix.tolist(), iy.tolist())))

        log_r_inv.append(np.log2(n_bins))   # log(1/r) = log(n_bins)
        log_N.append(np.log2(N))

    log_r_inv = np.array(log_r_inv)
    log_N     = np.array(log_N)

    # Линейная регрессия: FD = slope(log N / log(1/r))
    fd = float(np.polyfit(log_r_inv, log_N, 1)[0])
    return np.clip(fd, 1.0, 2.0)


def box_counting_curve(contour: np.ndarray,
                       n_scales: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает полную кривую log(N) vs log(1/r) для визуализации.

    Returns:
        (log_r_inv, log_N) — массивы для построения графика.
    """
    pts = contour.astype(np.float64)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = (maxs - mins).max()
    if span == 0:
        return np.zeros(n_scales), np.zeros(n_scales)
    pts_norm = (pts - mins) / span

    log_r_inv, log_N = [], []
    for k in range(1, n_scales + 1):
        n_bins = 2 ** k
        ix = np.clip(np.floor(pts_norm[:, 0] * n_bins).astype(np.int32), 0, n_bins - 1)
        iy = np.clip(np.floor(pts_norm[:, 1] * n_bins).astype(np.int32), 0, n_bins - 1)
        N = len(set(zip(ix.tolist(), iy.tolist())))
        log_r_inv.append(float(np.log2(n_bins)))
        log_N.append(float(np.log2(max(N, 1))))

    return np.array(log_r_inv), np.array(log_N)
