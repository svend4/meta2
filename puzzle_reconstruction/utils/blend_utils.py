"""
Утилиты смешивания и наложения изображений для реконструкции документов.

Предоставляет функции для альфа-наложения, бесшовного смешивания
и создания масок перехода, необходимых при финальной сборке пазла.

Экспортирует:
    BlendConfig      — параметры смешивания
    alpha_blend      — альфа-наложение двух изображений
    weighted_blend   — взвешенное среднее пикселей двух изображений
    feather_mask     — создать маску с плавным переходом у краёв
    paste_with_mask  — вставить патч в холст через маску
    horizontal_blend — бесшовное горизонтальное смешивание двух полос
    vertical_blend   — бесшовное вертикальное смешивание двух полос
    batch_blend      — применить один режим смешивания к списку пар
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── BlendConfig ──────────────────────────────────────────────────────────────

@dataclass
class BlendConfig:
    """Параметры операций смешивания.

    Attributes:
        feather_px:   Ширина зоны плавного перехода (пикс., >= 0).
        gamma:        Гамма-коррекция при смешивании (> 0). 1.0 — без коррекции.
        clip_output:  Обрезать ли результат в [0, 255].
    """
    feather_px:  int   = 8
    gamma:       float = 1.0
    clip_output: bool  = True

    def __post_init__(self) -> None:
        if self.feather_px < 0:
            raise ValueError(
                f"feather_px must be >= 0, got {self.feather_px}"
            )
        if self.gamma <= 0.0:
            raise ValueError(
                f"gamma must be > 0, got {self.gamma}"
            )


# ─── alpha_blend ──────────────────────────────────────────────────────────────

def alpha_blend(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Альфа-наложение: result = alpha * src + (1 - alpha) * dst.

    Args:
        src:   Исходное изображение (H, W) или (H, W, C), uint8.
        dst:   Целевое изображение той же формы и dtype.
        alpha: Коэффициент смешивания ∈ [0, 1].

    Returns:
        Смешанное изображение uint8 той же формы.

    Raises:
        ValueError: Если формы не совпадают, alpha вне [0, 1], или ndim неверный.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if src.shape != dst.shape:
        raise ValueError(
            f"src and dst shapes must match: {src.shape} vs {dst.shape}"
        )
    if src.ndim not in (2, 3):
        raise ValueError(f"images must be 2-D or 3-D, got ndim={src.ndim}")

    src_f = src.astype(np.float32)
    dst_f = dst.astype(np.float32)
    result = alpha * src_f + (1.0 - alpha) * dst_f
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── weighted_blend ───────────────────────────────────────────────────────────

def weighted_blend(
    images: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Взвешенное среднее пикселей нескольких изображений.

    Args:
        images:  Список изображений одинаковой формы и dtype.
        weights: Веса (None → равные). Длина должна совпадать с len(images).

    Returns:
        Смешанное изображение uint8.

    Raises:
        ValueError: Если images пуст, формы не совпадают или веса неверны.
    """
    if not images:
        raise ValueError("images must not be empty")
    if weights is not None and len(weights) != len(images):
        raise ValueError(
            f"len(weights)={len(weights)} != len(images)={len(images)}"
        )
    ref_shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != ref_shape:
            raise ValueError(
                f"images[{i}] shape {img.shape} != reference {ref_shape}"
            )

    if weights is None:
        w_arr = np.ones(len(images), dtype=np.float32) / len(images)
    else:
        w_arr = np.array(weights, dtype=np.float32)
        w_sum = w_arr.sum()
        if w_sum > 0:
            w_arr = w_arr / w_sum
        else:
            w_arr = np.ones(len(images), dtype=np.float32) / len(images)

    acc = np.zeros(ref_shape, dtype=np.float32)
    for img, w in zip(images, w_arr):
        acc += img.astype(np.float32) * float(w)

    return np.clip(acc, 0, 255).astype(np.uint8)


# ─── feather_mask ─────────────────────────────────────────────────────────────

def feather_mask(
    h: int,
    w: int,
    feather_px: int = 8,
) -> np.ndarray:
    """Создать маску плавного перехода (feathering) у краёв холста.

    Центральная область = 1.0; к краям значения линейно убывают до 0.0.

    Args:
        h:          Высота маски (пикс., > 0).
        w:          Ширина маски (пикс., > 0).
        feather_px: Ширина зоны затухания у каждого края (>= 0).

    Returns:
        Маска float32 формы (H, W) со значениями ∈ [0, 1].

    Raises:
        ValueError: Если h <= 0, w <= 0, или feather_px < 0.
    """
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    if w <= 0:
        raise ValueError(f"w must be > 0, got {w}")
    if feather_px < 0:
        raise ValueError(f"feather_px must be >= 0, got {feather_px}")

    mask = np.ones((h, w), dtype=np.float32)
    fp = min(feather_px, h // 2, w // 2)
    if fp > 0:
        ramp = np.linspace(0.0, 1.0, fp, dtype=np.float32)
        # Top and bottom
        mask[:fp, :] *= ramp[:, np.newaxis]
        mask[-fp:, :] *= ramp[::-1, np.newaxis]
        # Left and right
        mask[:, :fp] *= ramp[np.newaxis, :]
        mask[:, -fp:] *= ramp[np.newaxis, ::-1]
    return mask


# ─── paste_with_mask ──────────────────────────────────────────────────────────

def paste_with_mask(
    canvas: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    y: int,
    x: int,
) -> np.ndarray:
    """Вставить патч на холст с учётом маски смешивания.

    result[y:y+h, x:x+w] = mask * patch + (1 - mask) * canvas

    Args:
        canvas: Целевое изображение (H, W) или (H, W, C), uint8.
        patch:  Патч того же числа каналов, что и canvas.
        mask:   Маска float32 (h, w) со значениями ∈ [0, 1].
        y:      Вертикальная позиция вставки.
        x:      Горизонтальная позиция вставки.

    Returns:
        Новый холст (копия) с вставленным патчем.

    Raises:
        ValueError: Если canvas не 2-D/3-D или patch не совместим с canvas.
    """
    if canvas.ndim not in (2, 3):
        raise ValueError(f"canvas must be 2-D or 3-D, got ndim={canvas.ndim}")
    if patch.ndim not in (2, 3):
        raise ValueError(f"patch must be 2-D or 3-D, got ndim={patch.ndim}")

    result = canvas.copy().astype(np.float32)
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]

    y2 = min(y + ph, ch)
    x2 = min(x + pw, cw)
    y1 = max(0, y)
    x1 = max(0, x)
    if y2 <= y1 or x2 <= x1:
        return canvas.copy()

    py1 = y1 - y
    px1 = x1 - x
    py2 = py1 + (y2 - y1)
    px2 = px1 + (x2 - x1)

    crop_mask = mask[py1:py2, px1:px2].astype(np.float32)
    crop_patch = patch[py1:py2, px1:px2].astype(np.float32)

    if canvas.ndim == 3:
        crop_mask = crop_mask[:, :, np.newaxis]

    result[y1:y2, x1:x2] = (
        crop_mask * crop_patch + (1.0 - crop_mask) * result[y1:y2, x1:x2]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── horizontal_blend ─────────────────────────────────────────────────────────

def horizontal_blend(
    left: np.ndarray,
    right: np.ndarray,
    overlap: int,
    cfg: Optional[BlendConfig] = None,
) -> np.ndarray:
    """Бесшовное горизонтальное смешивание двух полос одинаковой высоты.

    Args:
        left:    Левая полоса (H, W1) или (H, W1, C).
        right:   Правая полоса (H, W2) той же высоты и числа каналов.
        overlap: Ширина зоны перекрытия (пикс., >= 0).
        cfg:     Параметры. None → BlendConfig().

    Returns:
        Объединённое изображение (H, W1 + W2 - overlap).

    Raises:
        ValueError: Если высоты не совпадают или overlap отрицательный.
    """
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"left height {left.shape[0]} != right height {right.shape[0]}"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if cfg is None:
        cfg = BlendConfig()

    lh, lw = left.shape[:2]
    rw = right.shape[1]
    out_w = lw + rw - overlap
    is_color = left.ndim == 3
    n_ch = left.shape[2] if is_color else 1

    if is_color:
        canvas = np.zeros((lh, out_w, n_ch), dtype=np.uint8)
    else:
        canvas = np.zeros((lh, out_w), dtype=np.uint8)

    canvas[:, :lw] = left
    if overlap > 0:
        ramp = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        ov_left = left[:, lw - overlap:].astype(np.float32)
        ov_right = right[:, :overlap].astype(np.float32)
        if is_color:
            ramp = ramp[np.newaxis, :, np.newaxis]
        else:
            ramp = ramp[np.newaxis, :]
        blended = (ramp * ov_left + (1.0 - ramp) * ov_right)
        canvas[:, lw - overlap:lw] = np.clip(blended, 0, 255).astype(np.uint8)

    canvas[:, lw:] = right[:, overlap:]
    return canvas


# ─── vertical_blend ───────────────────────────────────────────────────────────

def vertical_blend(
    top: np.ndarray,
    bottom: np.ndarray,
    overlap: int,
    cfg: Optional[BlendConfig] = None,
) -> np.ndarray:
    """Бесшовное вертикальное смешивание двух полос одинаковой ширины.

    Args:
        top:     Верхняя полоса (H1, W) или (H1, W, C).
        bottom:  Нижняя полоса (H2, W) той же ширины и числа каналов.
        overlap: Высота зоны перекрытия (пикс., >= 0).
        cfg:     Параметры. None → BlendConfig().

    Returns:
        Объединённое изображение (H1 + H2 - overlap, W).

    Raises:
        ValueError: Если ширины не совпадают или overlap отрицательный.
    """
    if top.shape[1] != bottom.shape[1]:
        raise ValueError(
            f"top width {top.shape[1]} != bottom width {bottom.shape[1]}"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if cfg is None:
        cfg = BlendConfig()

    th, tw = top.shape[:2]
    bh = bottom.shape[0]
    out_h = th + bh - overlap
    is_color = top.ndim == 3
    n_ch = top.shape[2] if is_color else 1

    if is_color:
        canvas = np.zeros((out_h, tw, n_ch), dtype=np.uint8)
    else:
        canvas = np.zeros((out_h, tw), dtype=np.uint8)

    canvas[:th] = top
    if overlap > 0:
        ramp = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        ov_top = top[th - overlap:].astype(np.float32)
        ov_bot = bottom[:overlap].astype(np.float32)
        if is_color:
            ramp = ramp[:, np.newaxis, np.newaxis]
        else:
            ramp = ramp[:, np.newaxis]
        blended = (ramp * ov_top + (1.0 - ramp) * ov_bot)
        canvas[th - overlap:th] = np.clip(blended, 0, 255).astype(np.uint8)

    canvas[th:] = bottom[overlap:]
    return canvas


# ─── batch_blend ──────────────────────────────────────────────────────────────

def batch_blend(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Применить alpha_blend ко всем парам (src, dst).

    Args:
        pairs: Список пар (src, dst) одинаковой формы.
        alpha: Коэффициент смешивания ∈ [0, 1].

    Returns:
        Список смешанных изображений.
    """
    return [alpha_blend(src, dst, alpha) for src, dst in pairs]
