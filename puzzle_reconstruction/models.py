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


@dataclass
class MatchingState:
    """Full state of a matching run, suitable for pause/resume."""
    compat_matrix: np.ndarray          # N×N float32
    entries: list                       # List[CompatEntry] — filtered pairs
    threshold: float                    # Selected threshold
    n_fragments: int                    # Number of fragments
    timestamp: str                      # ISO timestamp when created
    config_dict: dict                   # Config as dict (for serialisation)
    method: str = "auto"               # Threshold method used

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict (matrix as list)."""
        return {
            "compat_matrix": self.compat_matrix.tolist(),
            "threshold": self.threshold,
            "n_fragments": self.n_fragments,
            "timestamp": self.timestamp,
            "config_dict": self.config_dict,
            "method": self.method,
            "n_entries": len(self.entries),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatchingState":
        """Reconstruct from dict (entries will be empty — matrix only)."""
        return cls(
            compat_matrix=np.array(d["compat_matrix"], dtype=np.float32),
            entries=[],
            threshold=d["threshold"],
            n_fragments=d["n_fragments"],
            timestamp=d["timestamp"],
            config_dict=d.get("config_dict", {}),
            method=d.get("method", "auto"),
        )

    def save(self, path: str) -> None:
        """Save to JSON file."""
        import json, pathlib
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "MatchingState":
        """Load from JSON file."""
        import json, pathlib
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))


@dataclass
class AssemblySession:
    """Saveable session for long-running assembly algorithms (SA, genetic)."""
    method: str                        # Algorithm name
    iteration: int                     # Current iteration number
    best_score: float                  # Best score achieved so far
    score_history: list                # List[float] — score per iteration
    best_placement: dict               # fragment_id → (x, y, angle) dict
    n_fragments: int
    config_dict: dict = field(default_factory=dict)
    random_seed: int = 42

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "iteration": self.iteration,
            "best_score": self.best_score,
            "score_history": self.score_history,
            "best_placement": self.best_placement,
            "n_fragments": self.n_fragments,
            "config_dict": self.config_dict,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblySession":
        return cls(**d)

    def checkpoint(self, path: str) -> None:
        """Save current state to JSON."""
        import json, pathlib
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def resume(cls, path: str) -> "AssemblySession":
        """Load from checkpoint file."""
        import json, pathlib
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))

    @property
    def is_converged(self) -> bool:
        """True if score hasn't improved in last 100 iterations."""
        if len(self.score_history) < 100:
            return False
        return max(self.score_history[-100:]) <= self.best_score + 1e-6
