"""
Основные структуры данных системы восстановления пазлов.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ShapeClass(str, Enum):
    TRIANGLE   = "triangle"
    RECTANGLE  = "rectangle"
    TRAPEZOID  = "trapezoid"
    PARALLELOGRAM = "parallelogram"
    PENTAGON   = "pentagon"
    HEXAGON    = "hexagon"
    POLYGON    = "polygon"


class EdgeSide(str, Enum):
    TOP    = "top"
    BOTTOM = "bottom"
    LEFT   = "left"
    RIGHT  = "right"
    UNKNOWN = "unknown"


@dataclass
class FractalSignature:
    """Фрактальное описание одного края фрагмента."""
    fd_box:      float           # Фракт. размерность (Box-counting)
    fd_divider:  float           # Фракт. размерность (Divider/Richardson)
    ifs_coeffs:  np.ndarray      # Коэффициенты IFS (shape: [M])
    css_image:   list            # Curvature Scale Space [(sigma, zero_crossings)]
    chain_code:  str             # Цепной код Фримана (8-направлений)
    curve:       np.ndarray      # Параметрическая кривая контура (N, 2)


@dataclass
class TangramSignature:
    """Геометрическое описание внутреннего многоугольника фрагмента."""
    polygon:     np.ndarray      # Вершины (K, 2), нормализованные
    shape_class: ShapeClass
    centroid:    np.ndarray      # (2,)
    angle:       float           # Угол главной оси, радианы
    scale:       float           # Масштаб (диагональ описанного прямоуголника = 1)
    area:        float           # Площадь нормализованного полигона


@dataclass
class EdgeSignature:
    """
    Уникальная подпись одного края фрагмента.

    Синтез внутреннего (Танграм) и внешнего (Фрактал) описания.
    """
    edge_id:    int
    side:       EdgeSide
    # Виртуальная кромка: alpha * tangram_edge + (1-alpha) * fractal_edge
    virtual_curve: np.ndarray   # (N, 2)
    fd:         float           # Средняя FD
    css_vec:    np.ndarray      # Сжатый CSS-дескриптор (плоский вектор)
    ifs_coeffs: np.ndarray      # IFS-коэффициенты (из фрактала)
    length:     float           # Физическая длина края в пикселях


@dataclass
class Edge:
    """
    Упрощённое описание одного края фрагмента.

    Используется в алгоритмах сопоставления и верификации.
    """
    edge_id:    int
    contour:    np.ndarray       # Параметрическая кривая контура (N, 2)
    text_hint:  str = ""         # Подсказка из OCR (если есть)


@dataclass
class Placement:
    """Размещение одного фрагмента в сборке."""
    fragment_id: int
    position:    Tuple[float, float]   # (x, y) — координаты верхнего левого угла
    rotation:    float = 0.0           # Угол поворота в градусах


@dataclass
class Fragment:
    """Один физический фрагмент документа."""
    fragment_id:  int
    image:        np.ndarray        # BGR-изображение (H, W, 3)
    mask:         Optional[np.ndarray] = None   # Бинарная маска (H, W), dtype=uint8
    contour:      Optional[np.ndarray] = None   # Внешний контур (N, 2)

    tangram:      Optional[TangramSignature] = None
    fractal:      Optional[FractalSignature] = None
    edges:        List[Any] = field(default_factory=list)

    # Позиция в собранном документе (заполняется при сборке)
    placed:       bool = False
    position:     Optional[np.ndarray] = None  # (2,) — сдвиг в пикселях
    rotation:     float = 0.0                  # Угол поворота, радианы

    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclass
class CompatEntry:
    """Запись о совместимости двух краёв."""
    edge_i:     Any             # EdgeSignature или Edge
    edge_j:     Any             # EdgeSignature или Edge
    score:      float           # [0, 1], выше — лучше
    dtw_dist:   float = 0.0
    css_sim:    float = 0.0
    fd_diff:    float = 0.0
    text_score: float = 0.0


@dataclass
class Assembly:
    """Итоговая сборка документа."""
    placements:    Any = field(default_factory=list)  # List[Placement] или Dict[int, tuple]
    fragments:     Optional[List[Fragment]] = None
    compat_matrix: Optional[np.ndarray] = None       # (N_edges, N_edges)
    total_score:   float = 0.0
    ocr_score:     float = 0.0
    method:        str = ""
