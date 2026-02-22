"""
Утилиты пороговой обработки числовых массивов и матриц оценок.

Предоставляет функции для применения порогов к массивам,
бинаризации, нормализации и адаптивной пороговой обработки.
Используется при фильтрации матриц совместимости, бинаризации
оценок и сегментации сигналов.

Экспортирует:
    ThresholdConfig          — параметры пороговой обработки
    apply_threshold          — применение порога к массиву (→ bool маска)
    binarize                 — бинаризация массива в 0/1
    adaptive_threshold       — адаптивный порог через скользящее среднее
    soft_threshold           — мягкий порог (shrinkage)
    threshold_matrix         — применение порога к матрице (→ разреженная)
    hysteresis_threshold     — пороговая обработка с гистерезисом
    otsu_threshold           — метод Отцу для 1-D данных
    count_above              — число элементов выше порога
    fraction_above           — доля элементов выше порога
    batch_threshold          — пакетная пороговая обработка
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ─── ThresholdConfig ──────────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """Параметры пороговой обработки.

    Attributes:
        low:     Нижний порог (used for hysteresis and adaptive).
        high:    Верхний порог (low <= high).
        invert:  Инвертировать маску (True → ниже порога = True).
        mode:    Режим бинаризации: 'hard' | 'soft'.
    """
    low:    float = 0.3
    high:   float = 0.7
    invert: bool  = False
    mode:   str   = "hard"

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"low must be <= high, got {self.low} > {self.high}"
            )
        if self.mode not in ("hard", "soft"):
            raise ValueError(
                f"mode must be 'hard' or 'soft', got {self.mode!r}"
            )


# ─── apply_threshold ──────────────────────────────────────────────────────────

def apply_threshold(
    arr:   np.ndarray,
    value: float,
    invert: bool = False,
) -> np.ndarray:
    """Применяет пороговое значение к массиву.

    Args:
        arr:    Числовой массив (любой формы).
        value:  Порог.
        invert: Если True → маска = arr < value.

    Returns:
        Булев массив той же формы, что arr.

    Raises:
        ValueError: Если arr пуст.
    """
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("arr must not be empty")
    if invert:
        return a < value
    return a >= value


# ─── binarize ─────────────────────────────────────────────────────────────────

def binarize(
    arr:   np.ndarray,
    value: float,
    invert: bool = False,
) -> np.ndarray:
    """Бинаризует массив в значения 0 / 1 (float64).

    Args:
        arr:    Числовой массив.
        value:  Порог.
        invert: Если True → элементы ниже порога → 1.

    Returns:
        Массив float64 той же формы с значениями 0.0 или 1.0.

    Raises:
        ValueError: Если arr пуст.
    """
    mask = apply_threshold(arr, value, invert=invert)
    return mask.astype(np.float64)


# ─── adaptive_threshold ───────────────────────────────────────────────────────

def adaptive_threshold(
    arr:      np.ndarray,
    window:   int   = 8,
    offset:   float = 0.0,
    invert:   bool  = False,
) -> np.ndarray:
    """Адаптивный порог через скользящее среднее.

    Каждый элемент сравнивается с локальным средним в окне ± window//2.

    Args:
        arr:    1-D массив (не пустой).
        window: Размер окна (>= 1).
        offset: Смещение от локального среднего.
        invert: Инвертировать результат.

    Returns:
        Булев массив той же длины.

    Raises:
        ValueError: Если arr не 1-D, пуст или window < 1.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr must not be empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n   = len(a)
    out = np.empty(n, dtype=bool)
    hw  = window // 2
    for i in range(n):
        lo  = max(0, i - hw)
        hi  = min(n, i + hw + 1)
        local_mean = float(np.mean(a[lo:hi]))
        thresh = local_mean + offset
        if invert:
            out[i] = a[i] < thresh
        else:
            out[i] = a[i] >= thresh
    return out


# ─── soft_threshold ───────────────────────────────────────────────────────────

def soft_threshold(
    arr:   np.ndarray,
    value: float,
) -> np.ndarray:
    """Мягкий порог (shrinkage): sign(x) * max(|x| - value, 0).

    Args:
        arr:   1-D массив (не пустой).
        value: Порог (>= 0).

    Returns:
        Массив float64 той же длины.

    Raises:
        ValueError: Если arr не 1-D, пуст или value < 0.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr must not be empty")
    if value < 0.0:
        raise ValueError(f"value must be >= 0, got {value}")
    return np.sign(a) * np.maximum(np.abs(a) - value, 0.0)


# ─── threshold_matrix ─────────────────────────────────────────────────────────

def threshold_matrix(
    matrix: np.ndarray,
    value:  float,
    fill:   float = 0.0,
) -> np.ndarray:
    """Применяет порог к матрице: элементы ниже value заменяются на fill.

    Args:
        matrix: 2-D массив.
        value:  Порог.
        fill:   Значение для замены.

    Returns:
        Массив float64 той же формы.

    Raises:
        ValueError: Если matrix не 2-D.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={m.ndim}")
    result = m.copy()
    result[result < value] = fill
    return result


# ─── hysteresis_threshold ─────────────────────────────────────────────────────

def hysteresis_threshold(
    arr:  np.ndarray,
    low:  float,
    high: float,
) -> np.ndarray:
    """Пороговая обработка с гистерезисом (как в детекторе Canny).

    Элементы >= high → True сразу.
    Элементы в [low, high) → True, только если соседний элемент True.
    Элементы < low → False.

    Args:
        arr:  1-D массив (не пустой).
        low:  Нижний порог (low <= high).
        high: Верхний порог.

    Returns:
        Булев массив той же длины.

    Raises:
        ValueError: Если arr не 1-D, пуст или low > high.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr must not be empty")
    if low > high:
        raise ValueError(f"low must be <= high, got {low} > {high}")

    strong = a >= high
    weak   = (a >= low) & (a < high)
    result = strong.copy()

    # One pass: propagate strong through weak
    changed = True
    while changed:
        changed = False
        for i in range(len(a)):
            if weak[i] and not result[i]:
                if (i > 0 and result[i - 1]) or (i < len(a) - 1 and result[i + 1]):
                    result[i] = True
                    changed   = True
    return result


# ─── otsu_threshold ───────────────────────────────────────────────────────────

def otsu_threshold(arr: np.ndarray) -> float:
    """Вычисляет оптимальный порог методом Отцу для 1-D данных.

    Минимизирует взвешенную внутриклассовую дисперсию.

    Args:
        arr: 1-D массив (не менее 2 элементов, не пустой).

    Returns:
        Оптимальное пороговое значение (float).

    Raises:
        ValueError: Если arr не 1-D или содержит менее 2 элементов.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) < 2:
        raise ValueError(f"arr must have at least 2 elements, got {len(a)}")

    sorted_unique = np.unique(a)
    if len(sorted_unique) < 2:
        return float(sorted_unique[0])

    best_thresh = float(sorted_unique[0])
    best_var    = np.inf

    for thresh in sorted_unique[:-1]:
        fg = a[a >= thresh]
        bg = a[a < thresh]
        if len(fg) == 0 or len(bg) == 0:
            continue
        w_fg = len(fg) / len(a)
        w_bg = len(bg) / len(a)
        var  = w_fg * np.var(fg) + w_bg * np.var(bg)
        if var < best_var:
            best_var    = var
            best_thresh = float(thresh)

    return best_thresh


# ─── count_above ──────────────────────────────────────────────────────────────

def count_above(arr: np.ndarray, value: float) -> int:
    """Считает число элементов arr >= value.

    Args:
        arr:   Числовой массив (не пустой).
        value: Порог.

    Returns:
        int >= 0.

    Raises:
        ValueError: Если arr пуст.
    """
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("arr must not be empty")
    return int(np.sum(a >= value))


# ─── fraction_above ───────────────────────────────────────────────────────────

def fraction_above(arr: np.ndarray, value: float) -> float:
    """Вычисляет долю элементов arr >= value.

    Args:
        arr:   Числовой массив (не пустой).
        value: Порог.

    Returns:
        float в [0, 1].

    Raises:
        ValueError: Если arr пуст.
    """
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("arr must not be empty")
    return float(count_above(a, value)) / float(a.size)


# ─── batch_threshold ──────────────────────────────────────────────────────────

def batch_threshold(
    arrays: List[np.ndarray],
    value:  float,
    invert: bool = False,
) -> List[np.ndarray]:
    """Применяет apply_threshold ко всем массивам списка.

    Args:
        arrays: Список числовых массивов (не пустой список).
        value:  Порог.
        invert: Передаётся в apply_threshold.

    Returns:
        Список булевых массивов.

    Raises:
        ValueError: Если список arrays пуст.
    """
    if not arrays:
        raise ValueError("arrays must not be empty")
    return [apply_threshold(a, value, invert=invert) for a in arrays]
