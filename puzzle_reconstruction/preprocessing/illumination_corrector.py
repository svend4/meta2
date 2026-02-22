"""
Коррекция неравномерной засветки фрагментов документа.

Экспортирует:
    IlluminationParams  — параметры коррекции
    estimate_background — оценка фонового поля (Background Estimation)
    subtract_background — вычитание фона из изображения
    correct_by_homomorph — гомоморфная фильтрация (лог + НЧ-фильтр + экспонента)
    correct_by_retinex   — упрощённый Multi-Scale Retinex
    correct_illumination — унифицированный интерфейс (диспетчер методов)
    batch_correct        — коррекция списка изображений
    estimate_uniformity  — оценка равномерности освещённости [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class IlluminationParams:
    """Параметры коррекции освещённости.

    Attributes:
        method:        Метод коррекции: ``"background"``, ``"homomorph"``,
                       ``"retinex"``, ``"none"``.
        blur_ksize:    Размер ядра размытия для оценки фона (нечётное, ≥ 3).
        retinex_scales: Список масштабов для Multi-Scale Retinex (σ Гауссиана).
        homomorph_d0:  Радиус среза НЧ-фильтра в гомоморфном методе (пиксели).
        target_mean:   Целевое среднее яркости после коррекции (0–255).
        params:        Дополнительные пользовательские параметры.
    """
    method: str = "background"
    blur_ksize: int = 51
    retinex_scales: List[float] = field(default_factory=lambda: [15.0, 80.0, 250.0])
    homomorph_d0: float = 30.0
    target_mean: float = 128.0
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"background", "homomorph", "retinex", "none"}
        if self.method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}, got {self.method!r}")
        if self.blur_ksize < 3 or self.blur_ksize % 2 == 0:
            raise ValueError(
                f"blur_ksize must be an odd number >= 3, got {self.blur_ksize}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IlluminationParams(method={self.method!r}, "
            f"blur_ksize={self.blur_ksize}, target_mean={self.target_mean})"
        )


# ─── Приватные утилиты ────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _to_float32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


# ─── Публичные функции ────────────────────────────────────────────────────────

def estimate_background(
    img: np.ndarray,
    ksize: int = 51,
) -> np.ndarray:
    """Оценить поле фона методом сильного размытия (Background Estimation).

    Args:
        img:   Одноканальное или трёхканальное изображение uint8.
        ksize: Размер ядра медианного/Гауссова размытия (нечётное, ≥ 3).

    Returns:
        Фоновое поле — float32 того же размера, что и входное изображение.

    Raises:
        ValueError: Если ``ksize`` < 3 или чётное.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"ksize must be odd and >= 3, got {ksize}")
    gray = _to_gray(img).astype(np.float32)
    k = _ensure_odd(ksize)
    # Гауссово размытие имитирует плавный фон
    background = cv2.GaussianBlur(gray, (k, k), sigmaX=k / 3.0)
    return background.astype(np.float32)


def subtract_background(
    img: np.ndarray,
    background: Optional[np.ndarray] = None,
    ksize: int = 51,
    target_mean: float = 128.0,
) -> np.ndarray:
    """Вычесть фон из изображения и восстановить яркость.

    Args:
        img:         Входное изображение uint8 (1 или 3 канала).
        background:  Готовое фоновое поле float32; если ``None``,
                     вычисляется автоматически через :func:`estimate_background`.
        ksize:       Размер ядра для авто-оценки фона.
        target_mean: Целевое среднее яркости выходного изображения.

    Returns:
        Скорректированное изображение uint8 того же числа каналов.
    """
    gray = _to_gray(img).astype(np.float32)
    if background is None:
        background = estimate_background(img, ksize=ksize)

    bg = background.astype(np.float32)
    corrected = gray - bg + float(target_mean)
    corrected = np.clip(corrected, 0.0, 255.0).astype(np.uint8)

    if img.ndim == 3:
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    return corrected


def correct_by_homomorph(
    img: np.ndarray,
    d0: float = 30.0,
    target_mean: float = 128.0,
) -> np.ndarray:
    """Гомоморфная коррекция освещённости.

    Применяет логарифм → убирает низкочастотную компоненту (освещённость)
    Гауссовым фильтром → восстанавливает экспонентой.

    Args:
        img:         Входное изображение uint8 (1 или 3 канала).
        d0:          Стандартное отклонение Гауссова фильтра (> 0).
        target_mean: Целевое среднее яркости.

    Returns:
        Скорректированное изображение uint8.

    Raises:
        ValueError: Если ``d0`` ≤ 0.
    """
    if d0 <= 0:
        raise ValueError(f"d0 must be > 0, got {d0}")
    gray = _to_gray(img).astype(np.float32) + 1.0  # избегаем log(0)
    log_img = np.log(gray)

    ksize = _ensure_odd(int(d0 * 6) + 1)
    low_freq = cv2.GaussianBlur(log_img, (ksize, ksize), sigmaX=float(d0))
    high_freq = log_img - low_freq  # оставляем рефлектанс

    corrected = np.exp(high_freq)
    # Нормализуем к целевому среднему
    mean_val = corrected.mean()
    if mean_val > 1e-6:
        corrected = corrected * (float(target_mean) / mean_val)
    corrected = np.clip(corrected, 0.0, 255.0).astype(np.uint8)

    if img.ndim == 3:
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    return corrected


def correct_by_retinex(
    img: np.ndarray,
    scales: Optional[List[float]] = None,
    target_mean: float = 128.0,
) -> np.ndarray:
    """Упрощённый Multi-Scale Retinex (MSR).

    Retinex = log(I) - среднее log(GaussBlur(I)) по масштабам.

    Args:
        img:         Входное изображение uint8 (1 или 3 канала).
        scales:      Список σ Гауссиана; по умолчанию [15, 80, 250].
        target_mean: Целевое среднее яркости.

    Returns:
        Скорректированное изображение uint8.

    Raises:
        ValueError: Если список ``scales`` пуст.
    """
    if scales is None:
        scales = [15.0, 80.0, 250.0]
    if len(scales) == 0:
        raise ValueError("scales must not be empty")

    gray = _to_gray(img).astype(np.float32) + 1.0
    log_img = np.log(gray)

    retinex = np.zeros_like(log_img)
    for sigma in scales:
        k = _ensure_odd(int(sigma * 6) + 1)
        blurred = cv2.GaussianBlur(gray, (k, k), sigmaX=float(sigma))
        retinex += log_img - np.log(blurred.astype(np.float32) + 1.0)
    retinex /= len(scales)

    # Нормализуем к [0, 255] с целевым средним
    r_min, r_max = retinex.min(), retinex.max()
    if r_max - r_min > 1e-6:
        retinex = (retinex - r_min) / (r_max - r_min) * 255.0
    else:
        retinex = np.full_like(retinex, float(target_mean))

    # Смещаем к target_mean
    current_mean = retinex.mean()
    if current_mean > 1e-6:
        retinex = retinex * (float(target_mean) / current_mean)
    corrected = np.clip(retinex, 0.0, 255.0).astype(np.uint8)

    if img.ndim == 3:
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    return corrected


def correct_illumination(
    img: np.ndarray,
    params: Optional[IlluminationParams] = None,
) -> np.ndarray:
    """Унифицированный интерфейс коррекции освещённости.

    Args:
        img:    Входное изображение uint8 (1 или 3 канала).
        params: Параметры коррекции; если ``None``, используются значения
                по умолчанию (:class:`IlluminationParams`).

    Returns:
        Скорректированное изображение uint8.

    Raises:
        ValueError: Если ``params.method`` неизвестен (исключение
                    уже выбрасывается в ``__post_init__``).
    """
    if params is None:
        params = IlluminationParams()

    if params.method == "none":
        return img.copy()
    if params.method == "background":
        return subtract_background(
            img, ksize=params.blur_ksize, target_mean=params.target_mean
        )
    if params.method == "homomorph":
        return correct_by_homomorph(
            img, d0=params.homomorph_d0, target_mean=params.target_mean
        )
    if params.method == "retinex":
        return correct_by_retinex(
            img, scales=params.retinex_scales, target_mean=params.target_mean
        )
    # Недостижимо, но на случай добавления нового метода без обновления диспетчера
    raise ValueError(f"Unknown illumination method: {params.method!r}")


def batch_correct(
    images: List[np.ndarray],
    params: Optional[IlluminationParams] = None,
) -> List[np.ndarray]:
    """Применить коррекцию освещённости к списку изображений.

    Args:
        images: Список изображений uint8.
        params: Параметры коррекции; ``None`` → значения по умолчанию.

    Returns:
        Список скорректированных изображений той же длины.
        Для пустого списка — пустой список.
    """
    return [correct_illumination(img, params=params) for img in images]


def estimate_uniformity(img: np.ndarray, ksize: int = 51) -> float:
    """Оценить равномерность освещённости изображения.

    Равномерность = 1 - нормализованное стандартное отклонение фонового поля.

    Args:
        img:   Входное изображение uint8 (1 или 3 канала).
        ksize: Размер ядра для оценки фона.

    Returns:
        Значение [0, 1]; 1 — абсолютно равномерное освещение.
    """
    bg = estimate_background(img, ksize=ksize)
    mean_bg = bg.mean()
    if mean_bg < 1.0:
        return 1.0
    std_bg = bg.std()
    cv = std_bg / mean_bg  # коэффициент вариации
    return float(np.clip(1.0 - cv, 0.0, 1.0))
