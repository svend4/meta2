"""
Разделение и обработка цветовых каналов изображений.

Предоставляет инструменты для разбиения изображений на отдельные
каналы, вычисления статистик по каналам и их рекомбинации.

Экспортирует:
    ChannelStats       — статистика одного канала изображения
    split_channels     — разбить изображение на список каналов
    merge_channels     — объединить каналы в многоканальное изображение
    channel_statistics — вычислить статистику канала
    equalize_channel   — гистограммное выравнивание канала
    normalize_channel  — нормализовать канал в [out_min, out_max]
    channel_difference — попиксельная разность двух каналов
    apply_per_channel  — применить функцию к каждому каналу
    batch_split        — пакетное разбиение на каналы
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ChannelStats:
    """Статистика одного канала изображения.

    Attributes:
        mean:   Среднее значение пикселей.
        std:    Стандартное отклонение.
        min_val: Минимум.
        max_val: Максимум.
        median: Медиана.
    """
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ChannelStats(mean={self.mean:.2f}, std={self.std:.2f}, "
            f"min={self.min_val}, max={self.max_val})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def split_channels(img: np.ndarray) -> List[np.ndarray]:
    """Разбить изображение на список каналов.

    Args:
        img: Изображение uint8 (H, W) или (H, W, C).

    Returns:
        Список 2D-массивов uint8 (H, W) — по одному на канал.
        Для одноканального изображения — список из одного элемента.

    Raises:
        ValueError: Если изображение не 2-D или 3-D.
    """
    if img.ndim == 2:
        return [img.copy()]
    if img.ndim == 3:
        return [img[:, :, c].copy() for c in range(img.shape[2])]
    raise ValueError(f"img must be 2-D or 3-D, got ndim={img.ndim}")


def merge_channels(channels: List[np.ndarray]) -> np.ndarray:
    """Объединить список каналов в многоканальное изображение.

    Args:
        channels: Список 2D-массивов (H, W) одинакового размера.
                  При одном канале возвращает 2D-массив.

    Returns:
        Массив uint8 (H, W) если один канал, иначе (H, W, C).

    Raises:
        ValueError: Если список пуст или каналы имеют разные размеры.
    """
    if not channels:
        raise ValueError("channels must not be empty")
    h, w = channels[0].shape[:2]
    for i, c in enumerate(channels):
        if c.shape[:2] != (h, w):
            raise ValueError(
                f"Channel {i} shape {c.shape[:2]} != first channel shape ({h}, {w})"
            )
    if len(channels) == 1:
        return channels[0].copy()
    return np.stack(channels, axis=-1)


def channel_statistics(channel: np.ndarray) -> ChannelStats:
    """Вычислить статистику пикселей канала.

    Args:
        channel: 2D-массив (H, W) числового типа.

    Returns:
        :class:`ChannelStats`.

    Raises:
        ValueError: Если ``channel`` не 2-D.
    """
    if channel.ndim != 2:
        raise ValueError(f"channel must be 2-D, got ndim={channel.ndim}")
    arr = channel.astype(np.float64)
    return ChannelStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min_val=float(arr.min()),
        max_val=float(arr.max()),
        median=float(np.median(arr)),
    )


def equalize_channel(channel: np.ndarray) -> np.ndarray:
    """Применить гистограммное выравнивание к каналу.

    Args:
        channel: 2D-массив uint8 (H, W).

    Returns:
        Выровненный канал uint8 (H, W).

    Raises:
        ValueError: Если ``channel`` не 2-D.
    """
    if channel.ndim != 2:
        raise ValueError(f"channel must be 2-D, got ndim={channel.ndim}")
    c = channel.astype(np.uint8)
    return cv2.equalizeHist(c)


def normalize_channel(
    channel: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """Нормализовать канал в диапазон [out_min, out_max].

    Args:
        channel: 2D-массив числового типа (H, W).
        out_min: Нижняя граница выходного диапазона.
        out_max: Верхняя граница (> out_min).

    Returns:
        Нормализованный массив float64 (H, W).

    Raises:
        ValueError: Если ``channel`` не 2-D или out_max ≤ out_min.
    """
    if channel.ndim != 2:
        raise ValueError(f"channel must be 2-D, got ndim={channel.ndim}")
    if out_max <= out_min:
        raise ValueError(
            f"out_max ({out_max}) must be > out_min ({out_min})"
        )
    arr = channel.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.full_like(arr, out_min, dtype=np.float64)
    return out_min + (arr - mn) / (mx - mn) * (out_max - out_min)


def channel_difference(
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    """Вычислить абсолютную разность двух каналов.

    Каналы зажимаются до общего минимального размера.

    Args:
        c1: Первый канал 2D (H1, W1).
        c2: Второй канал 2D (H2, W2).

    Returns:
        Абсолютная разность float64 (min_H, min_W).

    Raises:
        ValueError: Если любой канал не 2-D.
    """
    if c1.ndim != 2:
        raise ValueError(f"c1 must be 2-D, got ndim={c1.ndim}")
    if c2.ndim != 2:
        raise ValueError(f"c2 must be 2-D, got ndim={c2.ndim}")
    h = min(c1.shape[0], c2.shape[0])
    w = min(c1.shape[1], c2.shape[1])
    a = c1[:h, :w].astype(np.float64)
    b = c2[:h, :w].astype(np.float64)
    return np.abs(a - b)


def apply_per_channel(
    img: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Применить функцию ``func`` к каждому каналу изображения.

    Args:
        img:  Изображение (H, W) или (H, W, C).
        func: Функция (H, W) → (H, W).

    Returns:
        Изображение с обработанными каналами; форма совпадает с ``img``
        при условии, что ``func`` сохраняет размер.

    Raises:
        ValueError: Если ``img`` не 2-D или 3-D.
    """
    channels = split_channels(img)
    processed = [func(c) for c in channels]
    return merge_channels(processed)


def batch_split(images: List[np.ndarray]) -> List[List[np.ndarray]]:
    """Пакетно разбить список изображений на каналы.

    Args:
        images: Список изображений uint8.

    Returns:
        Список списков каналов (outer: по изображениям, inner: по каналам).
    """
    return [split_channels(img) for img in images]
