"""
Утилиты извлечения и сравнения патчей изображений.

Предоставляет функции для вырезания прямоугольных и произвольных патчей
из изображений фрагментов, их нормализации и сравнения (SSD, NCC, MSE).

Экспортирует:
    PatchConfig      — параметры извлечения патча
    extract_patch    — вырезать прямоугольный патч с паддингом
    extract_patches  — вырезать несколько патчей за раз
    normalize_patch  — нормализовать патч в [0, 1] или z-score
    compare_patches  — сравнить два патча одним из методов
    patch_ssd        — сумма квадратов разностей (меньше = лучше)
    patch_ncc        — нормализованная взаимная корреляция ∈ [-1, 1]
    patch_mse        — среднеквадратичная ошибка (меньше = лучше)
    batch_compare    — сравнить список пар патчей
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─── PatchConfig ──────────────────────────────────────────────────────────────

@dataclass
class PatchConfig:
    """Параметры извлечения патча.

    Attributes:
        patch_h:    Высота патча (пикс., > 0).
        patch_w:    Ширина патча (пикс., > 0).
        pad_value:  Значение паддинга при выходе за границы [0, 255].
        normalize:  Нормализовать патч после извлечения.
        norm_mode:  Режим нормализации: ``'minmax'`` или ``'zscore'``.
    """
    patch_h:   int   = 32
    patch_w:   int   = 32
    pad_value: int   = 0
    normalize: bool  = False
    norm_mode: str   = "minmax"

    _NORM_MODES = frozenset({"minmax", "zscore"})

    def __post_init__(self) -> None:
        if self.patch_h <= 0:
            raise ValueError(f"patch_h must be > 0, got {self.patch_h}")
        if self.patch_w <= 0:
            raise ValueError(f"patch_w must be > 0, got {self.patch_w}")
        if not (0 <= self.pad_value <= 255):
            raise ValueError(
                f"pad_value must be in [0, 255], got {self.pad_value}"
            )
        if self.norm_mode not in self._NORM_MODES:
            raise ValueError(
                f"norm_mode must be one of {sorted(self._NORM_MODES)}, "
                f"got {self.norm_mode!r}"
            )


# ─── extract_patch ────────────────────────────────────────────────────────────

def extract_patch(
    img: np.ndarray,
    center_y: int,
    center_x: int,
    cfg: Optional[PatchConfig] = None,
) -> np.ndarray:
    """Вырезать прямоугольный патч с центром в (center_y, center_x).

    Пиксели за пределами изображения заполняются cfg.pad_value.
    Если cfg.normalize, патч нормализуется после извлечения.

    Args:
        img:      Изображение (H, W) или (H, W, C), uint8 или float.
        center_y: Центр патча по вертикали (пикс.).
        center_x: Центр патча по горизонтали (пикс.).
        cfg:      Конфигурация (None → PatchConfig()).

    Returns:
        Патч формы (patch_h, patch_w) или (patch_h, patch_w, C).

    Raises:
        ValueError: Если img не 2D/3D.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got ndim={img.ndim}")
    if cfg is None:
        cfg = PatchConfig()

    img_h, img_w = img.shape[:2]
    ph, pw = cfg.patch_h, cfg.patch_w
    is_color = img.ndim == 3

    y0 = center_y - ph // 2
    x0 = center_x - pw // 2
    y1 = y0 + ph
    x1 = x0 + pw

    # Region within image bounds
    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(img_h, y1)
    ix1 = min(img_w, x1)

    # Create padded patch
    if is_color:
        n_ch = img.shape[2]
        patch = np.full((ph, pw, n_ch), cfg.pad_value, dtype=img.dtype)
    else:
        patch = np.full((ph, pw), cfg.pad_value, dtype=img.dtype)

    # Destination in patch
    dy0 = iy0 - y0
    dx0 = ix0 - x0
    dy1 = dy0 + (iy1 - iy0)
    dx1 = dx0 + (ix1 - ix0)

    if iy1 > iy0 and ix1 > ix0:
        patch[dy0:dy1, dx0:dx1] = img[iy0:iy1, ix0:ix1]

    if cfg.normalize:
        patch = normalize_patch(patch, mode=cfg.norm_mode)

    return patch


# ─── extract_patches ──────────────────────────────────────────────────────────

def extract_patches(
    img: np.ndarray,
    centers: List[Tuple[int, int]],
    cfg: Optional[PatchConfig] = None,
) -> List[np.ndarray]:
    """Вырезать несколько патчей из одного изображения.

    Args:
        img:     Изображение (H, W) или (H, W, C).
        centers: Список пар (center_y, center_x).
        cfg:     Конфигурация (None → PatchConfig()).

    Returns:
        Список патчей в том же порядке, что и centers.
    """
    if cfg is None:
        cfg = PatchConfig()
    return [extract_patch(img, cy, cx, cfg) for cy, cx in centers]


# ─── normalize_patch ──────────────────────────────────────────────────────────

def normalize_patch(
    patch: np.ndarray,
    mode: str = "minmax",
) -> np.ndarray:
    """Нормализовать патч.

    Args:
        patch: Массив любой формы и dtype.
        mode:  ``'minmax'`` → [0, 1]; ``'zscore'`` → нулевое среднее, ед. σ.

    Returns:
        float32 массив той же формы.

    Raises:
        ValueError: Если mode не поддерживается.
    """
    valid = {"minmax", "zscore"}
    if mode not in valid:
        raise ValueError(f"mode must be one of {sorted(valid)}, got {mode!r}")

    p = patch.astype(np.float32)

    if mode == "minmax":
        mn, mx = float(p.min()), float(p.max())
        if mx - mn < 1e-8:
            return np.zeros_like(p)
        return (p - mn) / (mx - mn)

    # zscore
    mean = float(p.mean())
    std = float(p.std())
    if std < 1e-8:
        return np.zeros_like(p)
    return (p - mean) / std


# ─── patch_ssd ────────────────────────────────────────────────────────────────

def patch_ssd(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Сумма квадратов разностей двух патчей (меньше = лучше).

    Патчи автоматически преобразуются в float32 перед вычислением.

    Args:
        patch_a: Первый патч.
        patch_b: Второй патч.

    Returns:
        Неотрицательное число.

    Raises:
        ValueError: Если формы патчей не совпадают.
    """
    if patch_a.shape != patch_b.shape:
        raise ValueError(
            f"patch shapes must match: {patch_a.shape} vs {patch_b.shape}"
        )
    diff = patch_a.astype(np.float32) - patch_b.astype(np.float32)
    return float(np.sum(diff ** 2))


# ─── patch_ncc ────────────────────────────────────────────────────────────────

def patch_ncc(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Нормализованная взаимная корреляция ∈ [-1, 1].

    Возвращает 0.0 если один из патчей имеет нулевое стандартное отклонение.

    Args:
        patch_a: Первый патч.
        patch_b: Второй патч.

    Returns:
        float ∈ [-1, 1].

    Raises:
        ValueError: Если формы патчей не совпадают.
    """
    if patch_a.shape != patch_b.shape:
        raise ValueError(
            f"patch shapes must match: {patch_a.shape} vs {patch_b.shape}"
        )
    a = patch_a.astype(np.float64).ravel()
    b = patch_b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


# ─── patch_mse ────────────────────────────────────────────────────────────────

def patch_mse(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Среднеквадратичная ошибка двух патчей (меньше = лучше).

    Args:
        patch_a: Первый патч.
        patch_b: Второй патч.

    Returns:
        Неотрицательное число.

    Raises:
        ValueError: Если формы патчей не совпадают.
    """
    if patch_a.shape != patch_b.shape:
        raise ValueError(
            f"patch shapes must match: {patch_a.shape} vs {patch_b.shape}"
        )
    diff = patch_a.astype(np.float64) - patch_b.astype(np.float64)
    return float(np.mean(diff ** 2))


# ─── compare_patches ──────────────────────────────────────────────────────────

def compare_patches(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    method: str = "ncc",
) -> float:
    """Сравнить два патча выбранным методом.

    Args:
        patch_a: Первый патч.
        patch_b: Второй патч.
        method:  ``'ncc'``, ``'ssd'`` или ``'mse'``.

    Returns:
        Результат метрики.

    Raises:
        ValueError: Если формы патчей не совпадают или method неизвестен.
    """
    if method == "ncc":
        return patch_ncc(patch_a, patch_b)
    if method == "ssd":
        return patch_ssd(patch_a, patch_b)
    if method == "mse":
        return patch_mse(patch_a, patch_b)
    raise ValueError(
        f"method must be one of 'ncc', 'ssd', 'mse', got {method!r}"
    )


# ─── batch_compare ────────────────────────────────────────────────────────────

def batch_compare(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    method: str = "ncc",
) -> List[float]:
    """Сравнить список пар патчей.

    Args:
        pairs:  Список пар (patch_a, patch_b).
        method: Метод сравнения: ``'ncc'``, ``'ssd'`` или ``'mse'``.

    Returns:
        Список результатов той же длины, что и pairs.

    Raises:
        ValueError: Если method неизвестен.
    """
    if method not in {"ncc", "ssd", "mse"}:
        raise ValueError(
            f"method must be one of 'ncc', 'ssd', 'mse', got {method!r}"
        )
    return [compare_patches(a, b, method) for a, b in pairs]
