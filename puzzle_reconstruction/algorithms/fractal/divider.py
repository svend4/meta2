"""
Фрактальная размерность методом Divider (Richardson / компасный метод).

Исторически первый метод: Льюис Фрай Ричардсон измерял длины береговых линий
шагами разного размера. Чем мельче шаг, тем длиннее «берег» — парадокс Ричардсона.

FD = -slope( log L(s) vs log s )
где L(s) — число шагов длиной s, умноженное на s.
"""
import numpy as np
from typing import Tuple


def divider_fd(contour: np.ndarray,
               n_scales: int = 8) -> float:
    """
    Вычисляет фрактальную размерность методом Divider.

    Args:
        contour:  (N, 2) упорядоченные точки контура.
        n_scales: Число масштабов.

    Returns:
        FD ∈ [1.0, 2.0].
    """
    log_s, log_L = divider_curve(contour, n_scales)
    if len(log_s) < 2:
        return 1.0
    slope = float(np.polyfit(log_s, log_L, 1)[0])
    # FD = 1 - slope (так как L(s) ~ s^(1-FD), то slope = 1-FD)
    fd = 1.0 - slope
    return float(np.clip(fd, 1.0, 2.0))


def divider_curve(contour: np.ndarray,
                  n_scales: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит кривую log(L) vs log(s) методом компаса.

    Для каждого размера шага s идём вдоль контура «шагами» длиной s,
    подсчитываем число шагов count(s). Суммарная длина L(s) = count(s) × s.

    Returns:
        (log_s, log_L)
    """
    pts = contour.astype(np.float64)

    # Определяем диапазон шагов: от 1% до 10% длины контура
    seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum()
    if seg_len == 0:
        return np.array([]), np.array([])

    s_min = seg_len * 0.005
    s_max = seg_len * 0.15
    steps = np.geomspace(s_min, s_max, n_scales)

    log_s = []
    log_L = []

    for s in steps:
        count = _walk_with_step(pts, s)
        if count > 1:
            log_s.append(np.log(s))
            log_L.append(np.log(count * s))

    return np.array(log_s), np.array(log_L)


def _walk_with_step(pts: np.ndarray, step: float) -> int:
    """
    Идём вдоль контура «линейкой» длиной step.
    Возвращает число шагов (измерений).
    """
    if len(pts) < 2:
        return 0

    count = 0
    current = pts[0].copy()
    idx = 0

    while idx < len(pts) - 1:
        # Ищем следующую точку контура, отстоящую ровно на step
        remaining = step
        while idx < len(pts) - 1 and remaining > 0:
            d = np.hypot(pts[idx + 1, 0] - current[0],
                         pts[idx + 1, 1] - current[1])
            if d <= remaining:
                remaining -= d
                current = pts[idx + 1].copy()
                idx += 1
            else:
                # Интерполируем точку на отрезке
                t = remaining / d
                current = current + t * (pts[idx + 1] - current)
                remaining = 0
        count += 1

    return count
