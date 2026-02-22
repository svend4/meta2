"""
Очистка изображений фрагментов документа от артефактов.

Устраняет типичные артефакты отсканированных/сфотографированных документов:
тени, неравномерное освещение, пограничные артефакты, случайные пятна.

Классы:
    CleanResult — результат очистки одного изображения

Функции:
    remove_shadow             — удаление теней (background subtraction)
    remove_border_artifacts   — зачистка пограничной полосы
    normalize_illumination    — выравнивание освещённости (Gaussian BG)
    remove_blobs              — удаление мелких тёмных пятен (CC-анализ)
    auto_clean                — автоматическая очистка (shadow + illumination)
    batch_clean               — пакетная обработка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


# ─── CleanResult ──────────────────────────────────────────────────────────────

@dataclass
class CleanResult:
    """
    Результат очистки изображения.

    Attributes:
        cleaned:           Очищенное изображение (uint8, тот же канальный формат).
        method:            Название метода очистки.
        params:            Словарь параметров.
        artifacts_removed: Число удалённых артефактов (блобов при blob removal).
    """
    cleaned:           np.ndarray
    method:            str
    params:            Dict = field(default_factory=dict)
    artifacts_removed: int = 0

    def __repr__(self) -> str:
        h, w = self.cleaned.shape[:2]
        return (f"CleanResult(method={self.method!r}, "
                f"shape=({h},{w}), artifacts={self.artifacts_removed})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


# ─── remove_shadow ────────────────────────────────────────────────────────────

def remove_shadow(img: np.ndarray,
                   block_size: int = 41,
                   c: float = 10.0) -> CleanResult:
    """
    Удаляет тени методом вычитания фона с большим окном.

    Оценка фона: размытие с окном ``block_size``×``block_size``.
    Результат: нормированное изображение без глобальных теней.

    Args:
        img:        BGR или grayscale изображение.
        block_size: Размер окна Gaussian для оценки фона (нечётный).
        c:          Смещение яркости после вычитания (добавляется к результату).

    Returns:
        CleanResult с method='shadow'.
    """
    ksize = _ensure_odd(max(block_size, 3))
    is_bgr = img.ndim == 3

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        bg  = cv2.GaussianBlur(ch.astype(np.float32), (ksize, ksize), 0)
        out = ch.astype(np.float32) - bg + c + 128.0
        return np.clip(out, 0, 255).astype(np.uint8)

    if is_bgr:
        channels = [_process_channel(img[:, :, i]) for i in range(img.shape[2])]
        cleaned  = cv2.merge(channels)
    else:
        cleaned  = _process_channel(img)

    return CleanResult(
        cleaned=cleaned,
        method="shadow",
        params={"block_size": ksize, "c": c},
    )


# ─── remove_border_artifacts ─────────────────────────────────────────────────

def remove_border_artifacts(img: np.ndarray,
                              border_px: int = 5,
                              fill: int = 255) -> CleanResult:
    """
    Зачищает пограничную полосу шириной ``border_px`` пикселей.

    Граничные пиксели заменяются значением ``fill`` (обычно 255 = белый).
    Полезно при наличии тёмных бордюров от сканера/рамки.

    Args:
        img:       BGR или grayscale изображение.
        border_px: Ширина зачищаемой полосы в пикселях.
        fill:      Значение заполнения [0, 255].

    Returns:
        CleanResult с method='border'.
    """
    cleaned = img.copy()
    h, w    = img.shape[:2]
    b       = max(0, int(border_px))

    fill_val: int | tuple
    if img.ndim == 3:
        fill_val = (fill, fill, fill)
    else:
        fill_val = fill

    if b > 0:
        cleaned[:b,  :]  = fill_val
        cleaned[h-b:, :] = fill_val
        cleaned[:,  :b]  = fill_val
        cleaned[:, w-b:] = fill_val

    return CleanResult(
        cleaned=cleaned,
        method="border",
        params={"border_px": b, "fill": fill},
    )


# ─── normalize_illumination ───────────────────────────────────────────────────

def normalize_illumination(img: np.ndarray,
                            sigma: float = 50.0) -> CleanResult:
    """
    Выравнивает освещённость методом Gaussian-background subtraction.

    Оценивает медленные вариации яркости (фон) через Gaussian blur,
    вычитает их и нормирует результат в [0, 255].

    Args:
        img:   BGR или grayscale изображение.
        sigma: σ фильтра Gaussian для оценки фона (пикселей).

    Returns:
        CleanResult с method='illumination'.
    """
    is_bgr = img.ndim == 3

    def _norm_channel(ch: np.ndarray) -> np.ndarray:
        f   = ch.astype(np.float64)
        bg  = cv2.GaussianBlur(f, (0, 0), sigma)
        diff = f - bg
        d_min, d_max = diff.min(), diff.max()
        rng = d_max - d_min
        if rng < 1e-6:
            return np.full_like(ch, 128, dtype=np.uint8)
        norm = ((diff - d_min) / rng * 255.0)
        return norm.astype(np.uint8)

    if is_bgr:
        channels = [_norm_channel(img[:, :, i]) for i in range(img.shape[2])]
        cleaned  = cv2.merge(channels)
    else:
        cleaned  = _norm_channel(img)

    return CleanResult(
        cleaned=cleaned,
        method="illumination",
        params={"sigma": sigma},
    )


# ─── remove_blobs ─────────────────────────────────────────────────────────────

def remove_blobs(img: np.ndarray,
                  min_area: int = 10,
                  max_area: int = 500,
                  fill: int = 255) -> CleanResult:
    """
    Удаляет мелкие тёмные пятна (блобы) с помощью анализа связных компонент.

    Алгоритм:
      1. Бинаризация (Otsu) в инвертированном виде (тёмное → переднее).
      2. Нахождение связных компонент.
      3. Компоненты с площадью в [min_area, max_area] заполняются fill.

    Args:
        img:      BGR или grayscale изображение.
        min_area: Нижняя граница площади блоба (пикс²).
        max_area: Верхняя граница площади блоба (пикс²).
        fill:     Значение заполнения [0, 255].

    Returns:
        CleanResult с method='blobs' и artifacts_removed=число удалённых.
    """
    gray   = _to_gray(img)
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    cleaned  = img.copy()
    removed  = 0

    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            mask = (labels == lab)
            if img.ndim == 3:
                cleaned[mask] = fill
            else:
                cleaned[mask] = fill
            removed += 1

    return CleanResult(
        cleaned=cleaned,
        method="blobs",
        params={"min_area": min_area, "max_area": max_area, "fill": fill},
        artifacts_removed=removed,
    )


# ─── auto_clean ───────────────────────────────────────────────────────────────

def auto_clean(img: np.ndarray,
               shadow_block: int = 41,
               illum_sigma: float = 30.0) -> CleanResult:
    """
    Автоматическая очистка: удаление теней + нормализация освещённости.

    Args:
        img:          BGR или grayscale изображение.
        shadow_block: Размер окна для remove_shadow.
        illum_sigma:  σ для normalize_illumination.

    Returns:
        CleanResult с method='auto'.
    """
    step1 = remove_shadow(img, block_size=shadow_block)
    step2 = normalize_illumination(step1.cleaned, sigma=illum_sigma)

    return CleanResult(
        cleaned=step2.cleaned,
        method="auto",
        params={
            "shadow_block": shadow_block,
            "illum_sigma":  illum_sigma,
        },
    )


# ─── batch_clean ──────────────────────────────────────────────────────────────

_DISPATCH = {
    "shadow":      remove_shadow,
    "border":      remove_border_artifacts,
    "illumination": normalize_illumination,
    "blobs":       remove_blobs,
    "auto":        auto_clean,
}


def batch_clean(images: List[np.ndarray],
                method: str = "auto",
                **kwargs) -> List[CleanResult]:
    """
    Пакетная очистка списка изображений.

    Args:
        images: Список BGR или grayscale изображений.
        method: 'shadow' | 'border' | 'illumination' | 'blobs' | 'auto'.
        **kwargs: Параметры, передаваемые в выбранный метод.

    Returns:
        Список CleanResult (по одному на изображение).

    Raises:
        ValueError: Если ``method`` неизвестен.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Available: {sorted(_DISPATCH.keys())}"
        )
    fn = _DISPATCH[method]
    return [fn(img, **kwargs) for img in images]
