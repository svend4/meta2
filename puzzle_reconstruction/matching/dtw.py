"""
Dynamic Time Warping (DTW) для сравнения кривых краёв.

DTW позволяет сравнивать кривые разной длины и с нелинейными деформациями —
идеально для краёв, которые могут быть чуть растянуты при разрыве бумаги.

Используется окно Сакое-Чибы для ускорения O(n·w) вместо O(n²).
"""
import numpy as np


def dtw_distance(a: np.ndarray,
                 b: np.ndarray,
                 window: int = 20) -> float:
    """
    DTW-расстояние между двумя параметрическими кривыми.

    Args:
        a: (N, D) — кривая края A.
        b: (M, D) — кривая края B.
        window: Ширина окна Сакое-Чибы.

    Returns:
        dist: нормализованное DTW-расстояние.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    w = max(window, abs(n - m))  # Минимальная ширина окна

    # Матрица стоимости
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])

    raw = dtw[n, m]
    # Нормализуем на длину пути
    return raw / (n + m) if (n + m) > 0 else 0.0


def dtw_distance_mirror(a: np.ndarray, b: np.ndarray, window: int = 20) -> float:
    """
    DTW для сопрягаемых краёв (один из них перевёрнут).
    Два края совпадают, если они зеркальны: a(t) ≈ b(1-t).
    Возвращает минимум из прямого и зеркального расстояний.
    """
    d_direct   = dtw_distance(a, b, window)
    d_mirrored = dtw_distance(a, b[::-1], window)
    return min(d_direct, d_mirrored)
