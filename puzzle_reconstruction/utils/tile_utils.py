"""
Разбивка изображений на тайлы и сборка обратно.

Предоставляет утилиты для структурированного разбиения изображений документов
на перекрывающиеся или неперекрывающиеся тайлы и их обратной сборки.

Экспортирует:
    TileConfig      — конфигурация тайлинга
    Tile            — один тайл с метаданными положения
    tile_image      — разбить изображение на тайлы
    reassemble_tiles — собрать тайлы обратно в изображение
    tile_overlap_ratio — доля перекрытия двух тайлов
    filter_tiles_by_content — отфильтровать тайлы с недостаточным содержимым
    compute_tile_grid  — вычислить координаты сетки тайлов
    batch_tile_images  — разбить список изображений на тайлы
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── TileConfig ───────────────────────────────────────────────────────────────

@dataclass
class TileConfig:
    """Конфигурация разбивки изображения на тайлы.

    Attributes:
        tile_h:    Высота тайла (пикс., > 0).
        tile_w:    Ширина тайла (пикс., > 0).
        stride_h:  Шаг по вертикали (пикс., > 0). По умолчанию = tile_h.
        stride_w:  Шаг по горизонтали (пикс., > 0). По умолчанию = tile_w.
        pad_value: Значение заполнения при паддинге (0–255).
    """
    tile_h:    int   = 64
    tile_w:    int   = 64
    stride_h:  int   = 0    # 0 → использовать tile_h (без перекрытия)
    stride_w:  int   = 0    # 0 → использовать tile_w (без перекрытия)
    pad_value: int   = 0

    def __post_init__(self) -> None:
        if self.tile_h <= 0:
            raise ValueError(f"tile_h must be > 0, got {self.tile_h}")
        if self.tile_w <= 0:
            raise ValueError(f"tile_w must be > 0, got {self.tile_w}")
        if self.stride_h < 0:
            raise ValueError(f"stride_h must be >= 0, got {self.stride_h}")
        if self.stride_w < 0:
            raise ValueError(f"stride_w must be >= 0, got {self.stride_w}")
        if not (0 <= self.pad_value <= 255):
            raise ValueError(
                f"pad_value must be in [0, 255], got {self.pad_value}"
            )

    @property
    def effective_stride_h(self) -> int:
        """Эффективный вертикальный шаг (tile_h если stride_h == 0)."""
        return self.stride_h if self.stride_h > 0 else self.tile_h

    @property
    def effective_stride_w(self) -> int:
        """Эффективный горизонтальный шаг (tile_w если stride_w == 0)."""
        return self.stride_w if self.stride_w > 0 else self.tile_w


# ─── Tile ─────────────────────────────────────────────────────────────────────

@dataclass
class Tile:
    """Один тайл изображения с метаданными положения.

    Attributes:
        data:    Пиксели тайла (H, W) или (H, W, C).
        row:     Индекс строки в сетке тайлов (0-based).
        col:     Индекс столбца в сетке тайлов (0-based).
        y:       Верхний край тайла в исходном изображении (пикс.).
        x:       Левый край тайла в исходном изображении (пикс.).
        source_h: Высота исходного изображения.
        source_w: Ширина исходного изображения.
    """
    data:     np.ndarray
    row:      int
    col:      int
    y:        int
    x:        int
    source_h: int
    source_w: int

    @property
    def h(self) -> int:
        """Высота тайла."""
        return self.data.shape[0]

    @property
    def w(self) -> int:
        """Ширина тайла."""
        return self.data.shape[1]


# ─── Публичные функции ────────────────────────────────────────────────────────

def compute_tile_grid(
    img_h: int,
    img_w: int,
    cfg: TileConfig,
) -> List[Tuple[int, int, int, int]]:
    """Вычислить координаты сетки тайлов (y, x, row, col).

    Args:
        img_h: Высота изображения (пикс.).
        img_w: Ширина изображения (пикс.).
        cfg:   Конфигурация тайлинга.

    Returns:
        Список кортежей (y, x, row, col) — верхний-левый угол каждого тайла
        и его индекс в сетке.

    Raises:
        ValueError: Если img_h или img_w <= 0.
    """
    if img_h <= 0:
        raise ValueError(f"img_h must be > 0, got {img_h}")
    if img_w <= 0:
        raise ValueError(f"img_w must be > 0, got {img_w}")

    sh = cfg.effective_stride_h
    sw = cfg.effective_stride_w

    coords: List[Tuple[int, int, int, int]] = []
    row = 0
    y = 0
    while y < img_h:
        col = 0
        x = 0
        while x < img_w:
            coords.append((y, x, row, col))
            x += sw
            col += 1
        y += sh
        row += 1
    return coords


def tile_image(
    img: np.ndarray,
    cfg: Optional[TileConfig] = None,
) -> List[Tile]:
    """Разбить изображение на тайлы.

    Тайлы у краёв изображения дополняются до cfg.tile_h × cfg.tile_w
    значением cfg.pad_value.

    Args:
        img: Изображение (H, W) или (H, W, C).
        cfg: Конфигурация тайлинга. None → TileConfig() с умолчаниями.

    Returns:
        Список :class:`Tile` в порядке строк-слева-направо.

    Raises:
        ValueError: Если img не 2-D или 3-D.
    """
    if cfg is None:
        cfg = TileConfig()
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2-D or 3-D, got ndim={img.ndim}")

    img_h, img_w = img.shape[:2]
    th, tw = cfg.tile_h, cfg.tile_w
    is_color = img.ndim == 3
    n_ch = img.shape[2] if is_color else 1

    grid = compute_tile_grid(img_h, img_w, cfg)
    tiles: List[Tile] = []

    for (y, x, row, col) in grid:
        # Размер реального фрагмента (может быть меньше тайла у края)
        y2 = min(y + th, img_h)
        x2 = min(x + tw, img_w)
        crop = img[y:y2, x:x2]

        # Паддинг если нужно
        need_pad = (y2 - y < th) or (x2 - x < tw)
        if need_pad:
            if is_color:
                pad = np.full((th, tw, n_ch), cfg.pad_value, dtype=img.dtype)
            else:
                pad = np.full((th, tw), cfg.pad_value, dtype=img.dtype)
            ch = y2 - y
            cw = x2 - x
            pad[:ch, :cw] = crop
            crop = pad

        tiles.append(Tile(
            data=crop,
            row=row,
            col=col,
            y=y,
            x=x,
            source_h=img_h,
            source_w=img_w,
        ))

    return tiles


def reassemble_tiles(
    tiles: List[Tile],
    out_shape: Tuple[int, int],
) -> np.ndarray:
    """Собрать тайлы обратно в изображение методом усреднения.

    Перекрывающиеся тайлы усредняются поэлементно.

    Args:
        tiles:     Список тайлов (результат :func:`tile_image`).
        out_shape: Форма выходного изображения (H, W).

    Returns:
        Изображение uint8 (H, W) или (H, W, C) — зависит от входных тайлов.

    Raises:
        ValueError: Если tiles пустой или out_shape некорректен.
    """
    if not tiles:
        raise ValueError("tiles must not be empty")
    th_out, tw_out = out_shape
    if th_out <= 0 or tw_out <= 0:
        raise ValueError(
            f"out_shape must have positive dimensions, got {out_shape}"
        )

    sample = tiles[0].data
    is_color = sample.ndim == 3
    if is_color:
        acc = np.zeros((th_out, tw_out, sample.shape[2]), dtype=np.float64)
        cnt = np.zeros((th_out, tw_out, 1), dtype=np.float64)
    else:
        acc = np.zeros((th_out, tw_out), dtype=np.float64)
        cnt = np.zeros((th_out, tw_out), dtype=np.float64)

    for tile in tiles:
        y, x = tile.y, tile.x
        th, tw = tile.h, tile.w
        y2 = min(y + th, th_out)
        x2 = min(x + tw, tw_out)
        crop_h = y2 - y
        crop_w = x2 - x
        if crop_h <= 0 or crop_w <= 0:
            continue
        acc[y:y2, x:x2] += tile.data[:crop_h, :crop_w].astype(np.float64)
        cnt[y:y2, x:x2] += 1.0

    cnt = np.maximum(cnt, 1.0)
    result = (acc / cnt).astype(np.uint8)
    return result


def tile_overlap_ratio(tile_a: Tile, tile_b: Tile) -> float:
    """Вычислить долю перекрытия двух тайлов (IoU).

    Args:
        tile_a: Первый тайл.
        tile_b: Второй тайл.

    Returns:
        IoU ∈ [0, 1].
    """
    ax1, ay1 = tile_a.x, tile_a.y
    ax2, ay2 = tile_a.x + tile_a.w, tile_a.y + tile_a.h
    bx1, by1 = tile_b.x, tile_b.y
    bx2, by2 = tile_b.x + tile_b.w, tile_b.y + tile_b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = tile_a.h * tile_a.w
    area_b = tile_b.h * tile_b.w
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def filter_tiles_by_content(
    tiles: List[Tile],
    min_foreground: float = 0.05,
) -> List[Tile]:
    """Отфильтровать тайлы с недостаточным количеством ненулевых пикселей.

    Args:
        tiles:           Список тайлов.
        min_foreground:  Минимальная доля ненулевых пикселей (0–1).

    Returns:
        Тайлы, у которых foreground_ratio >= min_foreground.

    Raises:
        ValueError: Если min_foreground не в [0, 1].
    """
    if not (0.0 <= min_foreground <= 1.0):
        raise ValueError(
            f"min_foreground must be in [0, 1], got {min_foreground}"
        )
    result: List[Tile] = []
    for tile in tiles:
        total = tile.data.size
        fg = float(np.count_nonzero(tile.data)) / max(1, total)
        if fg >= min_foreground:
            result.append(tile)
    return result


def batch_tile_images(
    images: List[np.ndarray],
    cfg: Optional[TileConfig] = None,
) -> List[List[Tile]]:
    """Разбить список изображений на тайлы.

    Args:
        images: Список изображений.
        cfg:    Конфигурация тайлинга.

    Returns:
        Список списков тайлов (по одному на изображение).
    """
    if cfg is None:
        cfg = TileConfig()
    return [tile_image(img, cfg) for img in images]
