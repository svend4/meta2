"""
Построение 2D-компоновки документа из размещённых фрагментов.

Экспортирует:
    LayoutCell          — ячейка компоновки (позиция, размер, индекс фрагмента)
    AssemblyLayout      — объект компоновки: сетка ячеек с метаданными
    create_layout       — создание пустой компоновки
    add_cell            — добавление ячейки в компоновку
    remove_cell         — удаление ячейки по индексу фрагмента
    compute_bounding_box — общий ограничивающий прямоугольник компоновки
    snap_to_grid        — выравнивание всех ячеек по регулярной сетке
    render_layout_image — растеризация компоновки в изображение-схему
    layout_to_dict      — сериализация в dict
    dict_to_layout      — десериализация из dict
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class LayoutCell:
    """Ячейка 2D-компоновки — один размещённый фрагмент.

    Attributes:
        fragment_idx: Индекс фрагмента.
        x:            Горизонтальное положение левого края (пиксели).
        y:            Вертикальное положение верхнего края (пиксели).
        width:        Ширина ячейки (> 0).
        height:       Высота ячейки (> 0).
        rotation:     Угол поворота фрагмента (градусы).
        meta:         Произвольные метаданные.
    """
    fragment_idx: int
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LayoutCell(idx={self.fragment_idx}, "
            f"pos=({self.x:.1f},{self.y:.1f}), "
            f"size={self.width:.1f}×{self.height:.1f})"
        )


@dataclass
class AssemblyLayout:
    """Компоновка всего документа.

    Attributes:
        cells:       Список ячеек компоновки.
        canvas_w:    Ширина холста (пиксели; 0 = авто).
        canvas_h:    Высота холста (пиксели; 0 = авто).
        params:      Параметры и метаданные компоновки.
    """
    cells: List[LayoutCell] = field(default_factory=list)
    canvas_w: float = 0.0
    canvas_h: float = 0.0
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AssemblyLayout(n_cells={len(self.cells)}, "
            f"canvas={self.canvas_w:.0f}×{self.canvas_h:.0f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def create_layout(
    canvas_w: float = 0.0,
    canvas_h: float = 0.0,
    **params,
) -> AssemblyLayout:
    """Создать пустую компоновку.

    Args:
        canvas_w:  Ширина холста (0 = авто).
        canvas_h:  Высота холста (0 = авто).
        **params:  Дополнительные параметры, сохраняемые в ``layout.params``.

    Returns:
        Пустой :class:`AssemblyLayout`.
    """
    return AssemblyLayout(
        cells=[],
        canvas_w=float(canvas_w),
        canvas_h=float(canvas_h),
        params=dict(params),
    )


def add_cell(
    layout: AssemblyLayout,
    fragment_idx: int,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float = 0.0,
    **meta,
) -> AssemblyLayout:
    """Добавить ячейку в компоновку.

    Если ячейка с данным ``fragment_idx`` уже существует, она заменяется.

    Args:
        layout:       Компоновка.
        fragment_idx: Индекс фрагмента.
        x, y:         Позиция левого верхнего угла.
        width, height: Размеры ячейки (> 0).
        rotation:     Угол поворота (градусы).
        **meta:       Произвольные метаданные ячейки.

    Returns:
        Тот же объект ``layout`` с добавленной ячейкой.

    Raises:
        ValueError: Если ``width`` ≤ 0 или ``height`` ≤ 0.
    """
    cell = LayoutCell(
        fragment_idx=int(fragment_idx),
        x=float(x), y=float(y),
        width=float(width), height=float(height),
        rotation=float(rotation),
        meta={k: v for k, v in meta.items()},
    )
    # Заменить существующую ячейку, если она уже есть
    layout.cells = [c for c in layout.cells if c.fragment_idx != fragment_idx]
    layout.cells.append(cell)
    return layout


def remove_cell(
    layout: AssemblyLayout,
    fragment_idx: int,
) -> AssemblyLayout:
    """Удалить ячейку с заданным индексом фрагмента.

    Args:
        layout:       Компоновка.
        fragment_idx: Индекс удаляемого фрагмента.

    Returns:
        Тот же объект ``layout`` без указанной ячейки.
        Если ячейка не найдена — объект не изменяется.
    """
    layout.cells = [c for c in layout.cells if c.fragment_idx != fragment_idx]
    return layout


def compute_bounding_box(
    layout: AssemblyLayout,
) -> Tuple[float, float, float, float]:
    """Вычислить общий ограничивающий прямоугольник всех ячеек.

    Args:
        layout: Компоновка с ячейками.

    Returns:
        Кортеж (x_min, y_min, width, height).
        Для пустой компоновки — (0, 0, 0, 0).
    """
    if not layout.cells:
        return (0.0, 0.0, 0.0, 0.0)
    x_min = min(c.x for c in layout.cells)
    y_min = min(c.y for c in layout.cells)
    x_max = max(c.x + c.width for c in layout.cells)
    y_max = max(c.y + c.height for c in layout.cells)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def snap_to_grid(
    layout: AssemblyLayout,
    grid_size: float = 1.0,
) -> AssemblyLayout:
    """Выровнять позиции всех ячеек по регулярной сетке.

    Args:
        layout:    Компоновка.
        grid_size: Шаг сетки (> 0).

    Returns:
        Тот же объект ``layout`` с округлёнными координатами.

    Raises:
        ValueError: Если ``grid_size`` ≤ 0.
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be > 0, got {grid_size}")
    for cell in layout.cells:
        cell.x = round(cell.x / grid_size) * grid_size
        cell.y = round(cell.y / grid_size) * grid_size
    return layout


def render_layout_image(
    layout: AssemblyLayout,
    padding: int = 2,
    bg_color: int = 240,
    cell_color: int = 180,
    border_color: int = 50,
) -> np.ndarray:
    """Растеризовать компоновку в схематичное изображение uint8.

    Каждая ячейка отображается как серый прямоугольник с тёмной рамкой
    и подписью (индекс фрагмента).

    Args:
        layout:       Компоновка.
        padding:      Отступ вокруг всех ячеек (пиксели).
        bg_color:     Яркость фона [0, 255].
        cell_color:   Яркость заливки ячейки [0, 255].
        border_color: Яркость рамки ячейки [0, 255].

    Returns:
        Изображение (H, W) uint8.
        Для пустой компоновки — изображение 1×1 цвета фона.
    """
    if not layout.cells:
        return np.full((1, 1), bg_color, dtype=np.uint8)

    x_min, y_min, w, h = compute_bounding_box(layout)
    canvas_w = max(int(np.ceil(w)) + 2 * padding, 1)
    canvas_h = max(int(np.ceil(h)) + 2 * padding, 1)
    img = np.full((canvas_h, canvas_w), bg_color, dtype=np.uint8)

    ox = -x_min + padding
    oy = -y_min + padding

    for cell in layout.cells:
        x0 = int(round(cell.x + ox))
        y0 = int(round(cell.y + oy))
        x1 = int(round(cell.x + cell.width + ox))
        y1 = int(round(cell.y + cell.height + oy))
        x0, x1 = max(0, x0), min(canvas_w - 1, x1)
        y0, y1 = max(0, y0), min(canvas_h - 1, y1)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cell_color
            img[y0, x0:x1] = border_color
            img[y1 - 1, x0:x1] = border_color
            img[y0:y1, x0] = border_color
            img[y0:y1, x1 - 1] = border_color

    return img


def layout_to_dict(layout: AssemblyLayout) -> dict:
    """Сериализовать компоновку в dict.

    Args:
        layout: Компоновка.

    Returns:
        Словарь, пригодный для JSON-сериализации.
    """
    return {
        "canvas_w": layout.canvas_w,
        "canvas_h": layout.canvas_h,
        "params": layout.params,
        "cells": [
            {
                "fragment_idx": c.fragment_idx,
                "x": c.x, "y": c.y,
                "width": c.width, "height": c.height,
                "rotation": c.rotation,
                "meta": c.meta,
            }
            for c in layout.cells
        ],
    }


def dict_to_layout(data: dict) -> AssemblyLayout:
    """Десериализовать компоновку из dict.

    Args:
        data: Словарь, ранее созданный :func:`layout_to_dict`.

    Returns:
        :class:`AssemblyLayout`.

    Raises:
        KeyError: Если в ``data`` отсутствуют обязательные поля.
    """
    layout = AssemblyLayout(
        canvas_w=float(data.get("canvas_w", 0.0)),
        canvas_h=float(data.get("canvas_h", 0.0)),
        params=dict(data.get("params", {})),
    )
    for cd in data.get("cells", []):
        cell = LayoutCell(
            fragment_idx=int(cd["fragment_idx"]),
            x=float(cd["x"]),
            y=float(cd["y"]),
            width=float(cd["width"]),
            height=float(cd["height"]),
            rotation=float(cd.get("rotation", 0.0)),
            meta=dict(cd.get("meta", {})),
        )
        layout.cells.append(cell)
    return layout
