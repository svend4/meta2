"""
Нормализация изображений-патчей перед сопоставлением.

Обеспечивает согласованное фотометрическое качество фрагментов за счёт
выравнивания гистограмм, растяжения контраста и стандартизации яркости.

Классы:
    NormalizationParams — параметры нормализации (метод + гиперпараметры)

Функции:
    equalize_histogram        — глобальное / CLAHE выравнивание гистограммы
    stretch_contrast          — линейное растяжение контраста по перцентилям
    standardize_patch         — сдвиг к заданным mean/std
    normalize_patch           — единый диспетчер нормализации
    batch_normalize           — пакетная нормализация списка изображений
    compute_normalization_stats — статистика яркости по набору изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── NormalizationParams ──────────────────────────────────────────────────────

@dataclass
class NormalizationParams:
    """
    Параметры нормализации патча.

    Attributes:
        method:         'equalize' | 'clahe' | 'stretch' | 'standardize' | 'none'.
        clip_limit:     Порог ограничения для CLAHE (метод 'clahe').
        tile_grid_size: Размер сетки тайлов для CLAHE.
        target_mean:    Целевое среднее значение яркости (метод 'standardize').
        target_std:     Целевое стандартное отклонение (метод 'standardize').
        low_pct:        Нижний перцентиль для растяжения (метод 'stretch').
        high_pct:       Верхний перцентиль для растяжения (метод 'stretch').
    """
    method:         str                 = "clahe"
    clip_limit:     float               = 2.0
    tile_grid_size: Tuple[int, int]     = (8, 8)
    target_mean:    float               = 128.0
    target_std:     float               = 50.0
    low_pct:        float               = 2.0
    high_pct:       float               = 98.0

    VALID_METHODS = ("equalize", "clahe", "stretch", "standardize", "none")

    def __post_init__(self) -> None:
        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown normalization method {self.method!r}. "
                f"Choose one of {self.VALID_METHODS}.")


# ─── _ensure_gray ─────────────────────────────────────────────────────────────

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Конвертирует BGR в grayscale; grayscale возвращает без изменений."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── _ensure_uint8 ────────────────────────────────────────────────────────────

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Клиппирует и конвертирует float в uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── equalize_histogram ───────────────────────────────────────────────────────

def equalize_histogram(
    img:    np.ndarray,
    method: str = "global",
    clip_limit:    float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Выравнивание гистограммы изображения.

    Args:
        img:            Grayscale или BGR изображение uint8.
        method:         'global' — глобальное выравнивание cv2.equalizeHist;
                        'clahe'  — адаптивное CLAHE.
        clip_limit:     Порог клиппирования для CLAHE.
        tile_grid_size: Размер сетки тайлов для CLAHE.

    Returns:
        Изображение uint8 с выровненной гистограммой (grayscale).

    Raises:
        ValueError: Неизвестный метод.
    """
    if method not in ("global", "clahe"):
        raise ValueError(f"method must be 'global' or 'clahe', got {method!r}.")

    gray = _ensure_gray(img)
    if gray.dtype != np.uint8:
        gray = _ensure_uint8(gray)

    if method == "global":
        return cv2.equalizeHist(gray)
    else:
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )
        return clahe.apply(gray)


# ─── stretch_contrast ─────────────────────────────────────────────────────────

def stretch_contrast(
    img:      np.ndarray,
    low_pct:  float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """
    Линейное растяжение контраста по перцентилям.

    Значения ниже low_pct-го перцентиля → 0,
    значения выше high_pct-го перцентиля → 255,
    остальные линейно интерполируются.

    Args:
        img:      Grayscale или BGR изображение.
        low_pct:  Нижний перцентиль отсечения.
        high_pct: Верхний перцентиль отсечения.

    Returns:
        Изображение uint8 с растянутым контрастом (grayscale).

    Raises:
        ValueError: Если low_pct >= high_pct.
    """
    if low_pct >= high_pct:
        raise ValueError(
            f"low_pct ({low_pct}) must be < high_pct ({high_pct}).")

    gray = _ensure_gray(img).astype(np.float64)
    lo   = np.percentile(gray, low_pct)
    hi   = np.percentile(gray, high_pct)

    if abs(hi - lo) < 1.0:
        return _ensure_uint8(gray)

    stretched = (gray - lo) / (hi - lo) * 255.0
    return _ensure_uint8(stretched)


# ─── standardize_patch ────────────────────────────────────────────────────────

def standardize_patch(
    img:         np.ndarray,
    target_mean: float = 128.0,
    target_std:  float = 50.0,
) -> np.ndarray:
    """
    Сдвигает яркость патча к заданным mean и std.

    out = (img - img.mean()) / max(img.std(), ε) * target_std + target_mean.
    Результат клиппируется в [0, 255].

    Args:
        img:         Grayscale или BGR изображение.
        target_mean: Целевое среднее (0–255).
        target_std:  Целевое стандартное отклонение.

    Returns:
        Изображение uint8 с новым mean/std (grayscale).
    """
    gray  = _ensure_gray(img).astype(np.float64)
    mu    = gray.mean()
    sigma = gray.std()
    if sigma < 1e-9:
        # Однородный патч → просто сдвигаем к target_mean
        result = np.full_like(gray, fill_value=target_mean, dtype=np.float64)
    else:
        result = (gray - mu) / sigma * target_std + target_mean
    return _ensure_uint8(result)


# ─── normalize_patch ──────────────────────────────────────────────────────────

def normalize_patch(
    img:    np.ndarray,
    params: Optional[NormalizationParams] = None,
) -> np.ndarray:
    """
    Нормализует изображение согласно заданным параметрам.

    Args:
        img:    Grayscale или BGR изображение.
        params: NormalizationParams. None → CLAHE с параметрами по умолчанию.

    Returns:
        Нормализованное изображение uint8 (grayscale).
    """
    if params is None:
        params = NormalizationParams()

    m = params.method

    if m == "none":
        return _ensure_uint8(_ensure_gray(img))
    elif m == "equalize":
        return equalize_histogram(img, method="global")
    elif m == "clahe":
        return equalize_histogram(
            img,
            method="clahe",
            clip_limit=params.clip_limit,
            tile_grid_size=params.tile_grid_size,
        )
    elif m == "stretch":
        return stretch_contrast(img, params.low_pct, params.high_pct)
    else:  # standardize
        return standardize_patch(img, params.target_mean, params.target_std)


# ─── batch_normalize ──────────────────────────────────────────────────────────

def batch_normalize(
    images: List[np.ndarray],
    params: Optional[NormalizationParams] = None,
) -> List[np.ndarray]:
    """
    Применяет нормализацию ко всем изображениям в списке.

    Args:
        images: Список изображений (BGR или grayscale).
        params: Параметры нормализации. None → CLAHE по умолчанию.

    Returns:
        Список нормализованных изображений uint8 той же длины.
    """
    if params is None:
        params = NormalizationParams()
    return [normalize_patch(img, params) for img in images]


# ─── compute_normalization_stats ──────────────────────────────────────────────

def compute_normalization_stats(
    images: List[np.ndarray],
) -> Dict:
    """
    Вычисляет статистику яркости по набору изображений.

    Args:
        images: Список изображений.

    Returns:
        Словарь:
          'mean'       — среднее по всем пикселям всех изображений,
          'std'        — стандартное отклонение,
          'min'        — минимум,
          'max'        — максимум,
          'p2'         — 2-й перцентиль,
          'p98'        — 98-й перцентиль,
          'n_images'   — число изображений.

    Raises:
        ValueError: Если список пустой.
    """
    if not images:
        raise ValueError("images list must not be empty.")

    all_pixels: List[np.ndarray] = []
    for img in images:
        gray = _ensure_gray(img).astype(np.float64)
        all_pixels.append(gray.ravel())

    flat = np.concatenate(all_pixels)
    return {
        "mean":     float(flat.mean()),
        "std":      float(flat.std()),
        "min":      float(flat.min()),
        "max":      float(flat.max()),
        "p2":       float(np.percentile(flat, 2)),
        "p98":      float(np.percentile(flat, 98)),
        "n_images": len(images),
    }
