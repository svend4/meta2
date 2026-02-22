"""
Фрактальная интерполяционная функция (FIF) по методу Барнсли.

Позволяет точно воспроизвести форму разорванного края через итерируемую
функциональную систему (IFS). Коэффициенты {d_n} — компактный «отпечаток» края.

Барнсли М.: "Fractals Everywhere", 1988.
"""
import numpy as np
from typing import Tuple


def fit_ifs_coefficients(curve: np.ndarray,
                         n_transforms: int = 8,
                         max_iter: int = 100,
                         tol: float = 1e-5) -> np.ndarray:
    """
    Подбирает коэффициенты IFS (вертикальные сжатия d_n) так,
    чтобы аттрактор IFS приближал исходную кривую.

    Args:
        curve:         (N, 2) точки кривой края, нормализованные к [0,1].
        n_transforms:  Число аффинных преобразований IFS.
        max_iter:      Максимум итераций оптимизации.
        tol:           Порог сходимости.

    Returns:
        d_coeffs: (n_transforms,) — вертикальные коэффициенты сжатия.
                  |d_n| < 1 обеспечивает сходимость IFS.
    """
    # Разбиваем кривую на n_transforms равных интервалов
    # Для каждого интервала [t_k, t_{k+1}] одно аффинное преобразование w_k
    y = _extract_height_profile(curve)  # 1D профиль высот края
    N = len(y)
    if N < n_transforms + 2:
        n_transforms = max(2, N // 4)

    # Опорные точки: разбиваем [0, N-1] на n_transforms сегментов
    breakpoints = np.linspace(0, N - 1, n_transforms + 1).astype(int)

    d_coeffs = np.zeros(n_transforms)

    for k in range(n_transforms):
        i0 = breakpoints[k]
        i1 = breakpoints[k + 1]
        i_mid_prev = breakpoints[0]       # начало предыдущего интервала
        i_mid_next = breakpoints[-1]      # конец

        # Локальная подстройка d_k:
        # Цель: w_k([0, N-1]) → [i0, i1]
        # x-часть: линейная, a_k = (i1-i0)/(N-1), e_k = i0
        # y-часть: c_k·x + d_k·y + f_k
        # Из условия интерполяции: w_k(0) = y[i0], w_k(N-1) = y[i1]

        y_sub = y[i0:i1 + 1]
        if len(y_sub) < 2:
            d_coeffs[k] = 0.0
            continue

        # Простая оценка: d_k = correlation(y_sub, y_full_resampled)
        y_full = _resample_1d(y, len(y_sub))
        # Коэффициент регрессии
        cov = np.cov(y_sub, y_full)
        if cov[1, 1] == 0:
            d_k = 0.0
        else:
            d_k = cov[0, 1] / cov[1, 1]
        # Ограничиваем: |d_k| < 1 для сходимости IFS
        d_coeffs[k] = np.clip(d_k, -0.95, 0.95)

    return d_coeffs


def reconstruct_from_ifs(d_coeffs: np.ndarray,
                         n_points: int = 256,
                         n_iter: int = 8) -> np.ndarray:
    """
    Восстанавливает 1D профиль края из IFS-коэффициентов.

    Args:
        d_coeffs: Коэффициенты, возвращённые fit_ifs_coefficients.
        n_points: Длина выходного профиля.
        n_iter:   Число итераций IFS (больше = точнее).

    Returns:
        profile: (n_points,) — восстановленный профиль.
    """
    n_transforms = len(d_coeffs)
    # Начинаем с нулевой функции
    y = np.zeros(n_points)

    for _ in range(n_iter):
        y_new = np.zeros(n_points)
        seg_len = n_points // n_transforms
        for k in range(n_transforms):
            i0 = k * seg_len
            i1 = (k + 1) * seg_len if k < n_transforms - 1 else n_points
            # Отображаем весь y на подотрезок [i0, i1]
            y_sub = _resample_1d(y, i1 - i0)
            y_new[i0:i1] = d_coeffs[k] * y_sub
        y = y_new

    return y


def ifs_distance(coeffs_a: np.ndarray, coeffs_b: np.ndarray) -> float:
    """
    Расстояние между двумя IFS-подписями (L2-норма разности коэффициентов).
    Используется для быстрого предварительного отбора совместимых краёв.
    """
    if len(coeffs_a) != len(coeffs_b):
        # Ресэмплируем к меньшей длине
        n = min(len(coeffs_a), len(coeffs_b))
        coeffs_a = _resample_1d(coeffs_a, n)
        coeffs_b = _resample_1d(coeffs_b, n)
    return float(np.linalg.norm(coeffs_a - coeffs_b))


# ---------------------------------------------------------------------------

def _extract_height_profile(curve: np.ndarray) -> np.ndarray:
    """
    Переводит 2D кривую в 1D профиль «высот» относительно хорды.
    Это позволяет применить 1D-версию IFS.
    """
    if len(curve) < 2:
        return np.zeros(len(curve))
    # Хорда: прямая между первой и последней точкой
    start, end = curve[0], curve[-1]
    chord_vec = end - start
    chord_len = np.linalg.norm(chord_vec)
    if chord_len == 0:
        return np.zeros(len(curve))
    chord_dir = chord_vec / chord_len
    perp_dir  = np.array([-chord_dir[1], chord_dir[0]])
    # Высота = проекция на перпендикуляр к хорде
    offsets = curve - start
    heights = offsets @ perp_dir
    return heights


def _resample_1d(arr: np.ndarray, n: int) -> np.ndarray:
    """Ресэмплирует 1D массив до n элементов через линейную интерполяцию."""
    if len(arr) == n:
        return arr.copy()
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, arr)
