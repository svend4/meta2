"""
Основные структуры данных системы восстановления пазлов.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
class Fragment:
    """Один физический фрагмент документа."""
    fragment_id:  int
    image:        np.ndarray        # BGR-изображение (H, W, 3)
    mask:         np.ndarray        # Бинарная маска (H, W), dtype=uint8
    contour:      np.ndarray        # Внешний контур (N, 2)

    tangram:      Optional[TangramSignature] = None
    fractal:      Optional[FractalSignature] = None
    edges:        list[EdgeSignature] = field(default_factory=list)

    # Позиция в собранном документе (заполняется при сборке)
    placed:       bool = False
    position:     Optional[np.ndarray] = None  # (2,) — сдвиг в пикселях
    rotation:     float = 0.0                  # Угол поворота, радианы


@dataclass
class CompatEntry:
    """Запись о совместимости двух краёв."""
    edge_i:     EdgeSignature
    edge_j:     EdgeSignature
    score:      float           # [0, 1], выше — лучше
    dtw_dist:   float
    css_sim:    float
    fd_diff:    float
    text_score: float


@dataclass
class Assembly:
    """Итоговая сборка документа."""
    fragments:     list[Fragment]
    placements:    dict[int, tuple[np.ndarray, float]]  # frag_id → (pos, angle)
    compat_matrix: np.ndarray                            # (N_edges, N_edges)
    total_score:   float = 0.0
    ocr_score:     float = 0.0
