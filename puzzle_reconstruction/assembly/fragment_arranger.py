"""Расстановка фрагментов на холсте сборки.

Модуль предоставляет функции для определения позиций фрагментов:
выравнивание по сетке, упаковка без перекрытий (greedy strip),
центрирование группы фрагментов, вычисление ограничивающего
прямоугольника для группы, сдвиг всех позиций и пакетная обработка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── ArrangementParams ────────────────────────────────────────────────────────

@dataclass
class ArrangementParams:
    """Параметры расстановки фрагментов.

    Атрибуты:
        strategy:   Стратегия расстановки ('grid', 'strip', 'center').
        cols:       Количество столбцов для grid (>= 1).
        gap:        Зазор между фрагментами в пикселях (>= 0).
        canvas_w:   Ширина холста (>= 1).
        canvas_h:   Высота холста (>= 1).
        params:     Дополнительные параметры.
    """

    strategy: str = "strip"
    cols: int = 4
    gap: int = 4
    canvas_w: int = 512
    canvas_h: int = 512
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"grid", "strip", "center"}
        if self.strategy not in valid:
            raise ValueError(
                f"Неизвестная стратегия '{self.strategy}'. Допустимые: {sorted(valid)}"
            )
        if self.cols < 1:
            raise ValueError(f"cols должен быть >= 1, получено {self.cols}")
        if self.gap < 0:
            raise ValueError(f"gap должен быть >= 0, получено {self.gap}")
        if self.canvas_w < 1:
            raise ValueError(f"canvas_w должен быть >= 1, получено {self.canvas_w}")
        if self.canvas_h < 1:
            raise ValueError(f"canvas_h должен быть >= 1, получено {self.canvas_h}")


# ─── FragmentPlacement ────────────────────────────────────────────────────────

@dataclass
class FragmentPlacement:
    """Позиция одного фрагмента на холсте.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        x:           Координата X верхнего левого угла (>= 0).
        y:           Координата Y верхнего левого угла (>= 0).
        width:       Ширина фрагмента (>= 1).
        height:      Высота фрагмента (>= 1).
        params:      Дополнительные параметры.
    """

    fragment_id: int
    x: int
    y: int
    width: int
    height: int
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.width < 1:
            raise ValueError(f"width должен быть >= 1, получено {self.width}")
        if self.height < 1:
            raise ValueError(f"height должен быть >= 1, получено {self.height}")

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """(x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """Центр фрагмента (cx, cy)."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


# ─── arrange_grid ─────────────────────────────────────────────────────────────

def arrange_grid(
    sizes: List[Tuple[int, int]],
    cols: int = 4,
    gap: int = 4,
) -> List[FragmentPlacement]:
    """Расставить фрагменты равномерной сеткой.

    Аргументы:
        sizes: Список (width, height) для каждого фрагмента.
        cols:  Количество столбцов (>= 1).
        gap:   Зазор между фрагментами в пикселях (>= 0).

    Возвращает:
        Список FragmentPlacement в порядке исходных фрагментов.

    Исключения:
        ValueError: Если sizes пуст или cols < 1 или gap < 0.
    """
    if cols < 1:
        raise ValueError(f"cols должен быть >= 1, получено {cols}")
    if gap < 0:
        raise ValueError(f"gap должен быть >= 0, получено {gap}")
    if not sizes:
        return []

    placements: List[FragmentPlacement] = []
    col_widths: List[int] = []
    row_heights: List[int] = []

    n = len(sizes)
    n_rows = (n + cols - 1) // cols

    # Вычисляем ширину каждого столбца и высоту каждой строки
    for row in range(n_rows):
        max_h = 0
        for col in range(cols):
            idx = row * cols + col
            if idx < n:
                max_h = max(max_h, sizes[idx][1])
        row_heights.append(max_h)

    for col in range(cols):
        max_w = 0
        for row in range(n_rows):
            idx = row * cols + col
            if idx < n:
                max_w = max(max_w, sizes[idx][0])
        col_widths.append(max_w)

    for i, (w, h) in enumerate(sizes):
        row = i // cols
        col = i % cols
        x = sum(col_widths[:col]) + col * gap
        y = sum(row_heights[:row]) + row * gap
        placements.append(
            FragmentPlacement(fragment_id=i, x=x, y=y, width=w, height=h)
        )

    return placements


# ─── arrange_strip ────────────────────────────────────────────────────────────

def arrange_strip(
    sizes: List[Tuple[int, int]],
    canvas_w: int,
    gap: int = 4,
) -> List[FragmentPlacement]:
    """Жадная упаковка фрагментов по строкам (strip packing).

    Переносит следующий фрагмент на новую строку, если не помещается
    в текущую.

    Аргументы:
        sizes:    Список (width, height) для каждого фрагмента.
        canvas_w: Ширина холста (>= 1).
        gap:      Зазор между фрагментами (>= 0).

    Возвращает:
        Список FragmentPlacement.

    Исключения:
        ValueError: Если canvas_w < 1 или gap < 0.
    """
    if canvas_w < 1:
        raise ValueError(f"canvas_w должен быть >= 1, получено {canvas_w}")
    if gap < 0:
        raise ValueError(f"gap должен быть >= 0, получено {gap}")
    if not sizes:
        return []

    placements: List[FragmentPlacement] = []
    x_cursor = 0
    y_cursor = 0
    row_height = 0

    for i, (w, h) in enumerate(sizes):
        if x_cursor + w > canvas_w and x_cursor > 0:
            # Перенос на новую строку
            y_cursor += row_height + gap
            x_cursor = 0
            row_height = 0
        placements.append(
            FragmentPlacement(fragment_id=i, x=x_cursor, y=y_cursor,
                              width=w, height=h)
        )
        x_cursor += w + gap
        row_height = max(row_height, h)

    return placements


# ─── center_placements ────────────────────────────────────────────────────────

def center_placements(
    placements: List[FragmentPlacement],
    canvas_w: int,
    canvas_h: int,
) -> List[FragmentPlacement]:
    """Сдвинуть группу размещённых фрагментов к центру холста.

    Аргументы:
        placements: Список FragmentPlacement.
        canvas_w:   Ширина холста (>= 1).
        canvas_h:   Высота холста (>= 1).

    Возвращает:
        Новый список FragmentPlacement со скорректированными позициями.

    Исключения:
        ValueError: Если canvas_w < 1 или canvas_h < 1.
    """
    if canvas_w < 1:
        raise ValueError(f"canvas_w должен быть >= 1, получено {canvas_w}")
    if canvas_h < 1:
        raise ValueError(f"canvas_h должен быть >= 1, получено {canvas_h}")
    if not placements:
        return []

    min_x = min(p.x for p in placements)
    min_y = min(p.y for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)
    group_w = max_x - min_x
    group_h = max_y - min_y

    off_x = max(0, (canvas_w - group_w) // 2) - min_x
    off_y = max(0, (canvas_h - group_h) // 2) - min_y

    result = []
    for p in placements:
        result.append(FragmentPlacement(
            fragment_id=p.fragment_id,
            x=max(0, p.x + off_x),
            y=max(0, p.y + off_y),
            width=p.width,
            height=p.height,
            params=dict(p.params),
        ))
    return result


# ─── group_bbox ───────────────────────────────────────────────────────────────

def group_bbox(
    placements: List[FragmentPlacement],
) -> Tuple[int, int, int, int]:
    """Вычислить ограничивающий прямоугольник группы фрагментов.

    Аргументы:
        placements: Список FragmentPlacement.

    Возвращает:
        (x, y, width, height) охватывающего прямоугольника.

    Исключения:
        ValueError: Если список пуст.
    """
    if not placements:
        raise ValueError("placements не может быть пустым")
    min_x = min(p.x for p in placements)
    min_y = min(p.y for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


# ─── shift_placements ─────────────────────────────────────────────────────────

def shift_placements(
    placements: List[FragmentPlacement],
    dx: int,
    dy: int,
) -> List[FragmentPlacement]:
    """Сдвинуть все фрагменты на (dx, dy).

    Аргументы:
        placements: Список FragmentPlacement.
        dx:         Сдвиг по X (может быть отрицательным, но x+dx >= 0).
        dy:         Сдвиг по Y (может быть отрицательным, но y+dy >= 0).

    Возвращает:
        Новый список FragmentPlacement со сдвинутыми позициями.

    Исключения:
        ValueError: Если после сдвига x или y стали бы отрицательными.
    """
    for p in placements:
        if p.x + dx < 0:
            raise ValueError(
                f"После сдвига x фрагмента {p.fragment_id} стало бы < 0: "
                f"{p.x} + {dx} = {p.x + dx}"
            )
        if p.y + dy < 0:
            raise ValueError(
                f"После сдвига y фрагмента {p.fragment_id} стало бы < 0: "
                f"{p.y} + {dy} = {p.y + dy}"
            )
    return [
        FragmentPlacement(
            fragment_id=p.fragment_id,
            x=p.x + dx,
            y=p.y + dy,
            width=p.width,
            height=p.height,
            params=dict(p.params),
        )
        for p in placements
    ]


# ─── arrange ──────────────────────────────────────────────────────────────────

def arrange(
    sizes: List[Tuple[int, int]],
    params: ArrangementParams,
) -> List[FragmentPlacement]:
    """Расставить фрагменты согласно параметрам.

    Аргументы:
        sizes:  Список (width, height) для каждого фрагмента.
        params: Параметры расстановки.

    Возвращает:
        Список FragmentPlacement.
    """
    if params.strategy == "grid":
        placements = arrange_grid(sizes, cols=params.cols, gap=params.gap)
    elif params.strategy == "strip":
        placements = arrange_strip(sizes, canvas_w=params.canvas_w, gap=params.gap)
    else:  # center
        placements = arrange_strip(sizes, canvas_w=params.canvas_w, gap=params.gap)
        placements = center_placements(placements, params.canvas_w, params.canvas_h)
    return placements


# ─── batch_arrange ────────────────────────────────────────────────────────────

def batch_arrange(
    size_lists: List[List[Tuple[int, int]]],
    params: ArrangementParams,
) -> List[List[FragmentPlacement]]:
    """Пакетная расстановка нескольких наборов фрагментов.

    Аргументы:
        size_lists: Список списков (width, height).
        params:     Параметры расстановки.

    Возвращает:
        Список списков FragmentPlacement.
    """
    return [arrange(sizes, params) for sizes in size_lists]
