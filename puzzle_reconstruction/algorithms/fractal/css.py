"""
Curvature Scale Space (CSS) — стандарт MPEG-7 для описания формы контура.

Введён Asada & Brady (1986), расширен Mokhtarian et al. (1992–1997).
Инвариантен к масштабу, переносу и повороту; устойчив к шуму.

Алгоритм:
  1. Параметризуем контур по длине дуги.
  2. При каждом масштабе σ сглаживаем x(t), y(t) гауссовым ядром.
  3. Вычисляем кривизну κ(t, σ).
  4. Находим нули кривизны (точки перегиба) → CSS-изображение.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple


def curvature_scale_space(contour: np.ndarray,
                          sigma_range: List[float] = None,
                          n_sigmas: int = 7) -> List[Tuple[float, np.ndarray]]:
    """
    Строит CSS-представление контура.

    Args:
        contour:    (N, 2) точки контура (замкнутый).
        sigma_range: Список значений σ. Если None — логарифмическая сетка от 1 до 64.
        n_sigmas:   Число масштабов (если sigma_range=None).

    Returns:
        css: [(sigma, zero_crossings_t)] — список пар (масштаб, позиции нулей кривизны).
             zero_crossings_t — нормализованные позиции ∈ [0, 1].
    """
    if sigma_range is None:
        sigma_range = np.geomspace(1, 64, n_sigmas).tolist()

    x = contour[:, 0].astype(float)
    y = contour[:, 1].astype(float)
    # Нормируем по длине дуги → t ∈ [0, 1]
    t = _arc_length_param(x, y)

    css = []
    for sigma in sigma_range:
        zc = _zero_crossings_at_sigma(x, y, t, float(sigma))
        css.append((float(sigma), zc))

    return css


def css_to_feature_vector(css: List[Tuple[float, np.ndarray]],
                          n_bins: int = 64) -> np.ndarray:
    """
    Преобразует CSS-изображение в плоский вектор признаков фиксированного размера.

    Каждый масштаб σ описывается гистограммой нулей кривизны по позиции t ∈ [0,1].
    Результат: конкатенация гистограмм по всем σ.

    Args:
        css:    Вывод curvature_scale_space().
        n_bins: Число бинов гистограммы для каждого масштаба.

    Returns:
        feature: (n_sigmas × n_bins,) нормализованный вектор.
    """
    parts = []
    for _, zc in css:
        hist, _ = np.histogram(zc, bins=n_bins, range=(0.0, 1.0))
        parts.append(hist.astype(float))
    if not parts:
        return np.zeros(n_bins)
    vec = np.concatenate(parts)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def css_similarity(css_a: np.ndarray, css_b: np.ndarray) -> float:
    """
    Косинусное сходство двух CSS-векторов.
    Возвращает значение ∈ [0, 1], где 1 = идентичные формы.
    """
    if css_a.shape != css_b.shape:
        n = min(len(css_a), len(css_b))
        css_a = css_a[:n]
        css_b = css_b[:n]
    dot = float(np.dot(css_a, css_b))
    norm_a = float(np.linalg.norm(css_a))
    norm_b = float(np.linalg.norm(css_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def css_similarity_mirror(css_a: np.ndarray, css_b: np.ndarray) -> float:
    """
    Сходство с учётом зеркальности (два сопрягаемых края — зеркальные).
    Проверяем прямое и перевёрнутое сходство, берём максимум.
    """
    direct   = css_similarity(css_a, css_b)
    mirrored = css_similarity(css_a, css_b[::-1])
    return max(direct, mirrored)


# ---------------------------------------------------------------------------

def _arc_length_param(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Возвращает параметр по длине дуги t ∈ [0, 1)."""
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    seg = np.hypot(dx, dy)
    cumlen = np.concatenate([[0], np.cumsum(seg[:-1])])
    total = cumlen[-1]
    if total == 0:
        return np.linspace(0, 1, len(x))
    return cumlen / total


def _zero_crossings_at_sigma(x: np.ndarray, y: np.ndarray,
                              t: np.ndarray, sigma: float) -> np.ndarray:
    """
    Вычисляет позиции нулей кривизны при масштабе sigma.
    """
    # Сглаживаем контур по периметру (mode='wrap' для замкнутой кривой)
    Xs = gaussian_filter1d(x, sigma, mode='wrap')
    Ys = gaussian_filter1d(y, sigma, mode='wrap')

    # Производные по дуге (аппроксимация конечными разностями)
    Xs1 = np.gradient(Xs)
    Xs2 = np.gradient(Xs1)
    Ys1 = np.gradient(Ys)
    Ys2 = np.gradient(Ys1)

    # Кривизна κ = (x'y'' - x''y') / (x'^2 + y'^2)^(3/2)
    denom = (Xs1 ** 2 + Ys1 ** 2) ** 1.5
    kappa = (Xs1 * Ys2 - Xs2 * Ys1) / (denom + 1e-10)

    # Нули кривизны: места смены знака
    sign_changes = np.where(np.diff(np.sign(kappa)))[0]

    # Интерполируем точную позицию нуля
    zero_t = []
    for idx in sign_changes:
        if abs(kappa[idx + 1] - kappa[idx]) < 1e-15:
            continue
        frac = -kappa[idx] / (kappa[idx + 1] - kappa[idx])
        t_zero = t[idx] + frac * (t[min(idx + 1, len(t) - 1)] - t[idx])
        zero_t.append(float(t_zero))

    return np.array(zero_t) if zero_t else np.array([])


def freeman_chain_code(contour: np.ndarray) -> str:
    """
    Цепной код Фримана (8 направлений) для контура.
    Используется как быстрый хэш формы.

    Направления: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    """
    if len(contour) < 2:
        return ""
    pts = np.round(contour).astype(int)
    code = []
    for i in range(len(pts) - 1):
        dx = int(pts[i + 1, 0] - pts[i, 0])
        dy = int(pts[i + 1, 1] - pts[i, 1])
        # Ограничиваем до {-1, 0, 1}
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        direction = _dx_dy_to_code(dx, dy)
        if direction is not None:
            code.append(str(direction))
    return "".join(code)


def _dx_dy_to_code(dx: int, dy: int) -> int | None:
    table = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    return table.get((dx, dy))
