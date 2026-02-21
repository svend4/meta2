"""Оценка и уточнение позиций фрагментов на холсте.

Модуль предоставляет методы для оценки начальных позиций фрагментов
по сетке, уточнения позиций через смещение, выравнивания по сетке
и генерации позиций-кандидатов для поиска.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── PositionConfig ───────────────────────────────────────────────────────────

@dataclass
class PositionConfig:
    """Параметры оценки позиций.

    Атрибуты:
        canvas_w:     Ширина холста (>= 1).
        canvas_h:     Высота холста (>= 1).
        grid_cols:    Число столбцов сетки (>= 1).
        grid_rows:    Число строк сетки (>= 1).
        padding:      Отступ между ячейками сетки (>= 0).
        snap_grid:    Привязать позиции к сетке.
        snap_size:    Размер ячейки привязки (>= 1).
    """

    canvas_w: int = 512
    canvas_h: int = 512
    grid_cols: int = 4
    grid_rows: int = 4
    padding: int = 0
    snap_grid: bool = False
    snap_size: int = 8

    def __post_init__(self) -> None:
        for name, val in (
            ("canvas_w", self.canvas_w),
            ("canvas_h", self.canvas_h),
        ):
            if val < 1:
                raise ValueError(f"{name} должен быть >= 1, получено {val}")
        for name, val in (
            ("grid_cols", self.grid_cols),
            ("grid_rows", self.grid_rows),
        ):
            if val < 1:
                raise ValueError(f"{name} должен быть >= 1, получено {val}")
        if self.padding < 0:
            raise ValueError(
                f"padding должен быть >= 0, получено {self.padding}"
            )
        if self.snap_size < 1:
            raise ValueError(
                f"snap_size должен быть >= 1, получено {self.snap_size}"
            )


# ─── FragmentPosition ─────────────────────────────────────────────────────────

@dataclass
class FragmentPosition:
    """Оценённая позиция фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        x:           Координата по горизонтали (>= 0).
        y:           Координата по вертикали (>= 0).
        confidence:  Уверенность в позиции [0, 1].
        method:      Метод, которым была получена позиция.
    """

    fragment_id: int
    x: int
    y: int
    confidence: float = 1.0
    method: str = "grid"

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence должен быть в [0, 1], получено {self.confidence}"
            )
        if not self.method:
            raise ValueError("method не должен быть пустым")

    @property
    def coords(self) -> Tuple[int, int]:
        """Координаты (x, y)."""
        return (self.x, self.y)


# ─── PositionEstimate ─────────────────────────────────────────────────────────

@dataclass
class PositionEstimate:
    """Результат оценки позиций для набора фрагментов.

    Атрибуты:
        positions:   Список FragmentPosition.
        n_fragments: Число фрагментов (>= 0).
        canvas_w:    Ширина холста (>= 1).
        canvas_h:    Высота холста (>= 1).
        mean_conf:   Средняя уверенность (>= 0).
    """

    positions: List[FragmentPosition]
    n_fragments: int
    canvas_w: int
    canvas_h: int
    mean_conf: float

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments должен быть >= 0, получено {self.n_fragments}"
            )
        for name, val in (("canvas_w", self.canvas_w), ("canvas_h", self.canvas_h)):
            if val < 1:
                raise ValueError(f"{name} должен быть >= 1, получено {val}")
        if self.mean_conf < 0.0:
            raise ValueError(
                f"mean_conf должен быть >= 0, получено {self.mean_conf}"
            )

    @property
    def by_id(self) -> dict:
        """Словарь {fragment_id: FragmentPosition}."""
        return {p.fragment_id: p for p in self.positions}


# ─── snap_to_grid ─────────────────────────────────────────────────────────────

def snap_to_grid(x: int, y: int, snap_size: int) -> Tuple[int, int]:
    """Привязать координаты к ближайшему кратному snap_size.

    Аргументы:
        x:         Координата X.
        y:         Координата Y.
        snap_size: Размер ячейки привязки (>= 1).

    Возвращает:
        (snapped_x, snapped_y).

    Исключения:
        ValueError: Если snap_size < 1.
    """
    if snap_size < 1:
        raise ValueError(f"snap_size должен быть >= 1, получено {snap_size}")
    sx = int(round(x / snap_size)) * snap_size
    sy = int(round(y / snap_size)) * snap_size
    return max(0, sx), max(0, sy)


# ─── estimate_grid_positions ──────────────────────────────────────────────────

def estimate_grid_positions(
    fragment_ids: List[int],
    frag_w: int,
    frag_h: int,
    cfg: Optional[PositionConfig] = None,
) -> PositionEstimate:
    """Разместить фрагменты на равномерной сетке.

    Аргументы:
        fragment_ids: Список ID фрагментов.
        frag_w:       Ширина фрагмента (>= 1).
        frag_h:       Высота фрагмента (>= 1).
        cfg:          Параметры (None → PositionConfig()).

    Возвращает:
        PositionEstimate.

    Исключения:
        ValueError: Если frag_w или frag_h < 1 или список пуст.
    """
    if cfg is None:
        cfg = PositionConfig()
    if frag_w < 1:
        raise ValueError(f"frag_w должен быть >= 1, получено {frag_w}")
    if frag_h < 1:
        raise ValueError(f"frag_h должен быть >= 1, получено {frag_h}")
    if not fragment_ids:
        return PositionEstimate(
            positions=[],
            n_fragments=0,
            canvas_w=cfg.canvas_w,
            canvas_h=cfg.canvas_h,
            mean_conf=0.0,
        )

    step_x = frag_w + cfg.padding
    step_y = frag_h + cfg.padding
    positions: List[FragmentPosition] = []

    for idx, fid in enumerate(fragment_ids):
        col = idx % cfg.grid_cols
        row = idx // cfg.grid_cols
        x = col * step_x
        y = row * step_y
        if cfg.snap_grid:
            x, y = snap_to_grid(x, y, cfg.snap_size)
        positions.append(FragmentPosition(
            fragment_id=fid,
            x=min(x, max(0, cfg.canvas_w - frag_w)),
            y=min(y, max(0, cfg.canvas_h - frag_h)),
            confidence=1.0,
            method="grid",
        ))

    return PositionEstimate(
        positions=positions,
        n_fragments=len(positions),
        canvas_w=cfg.canvas_w,
        canvas_h=cfg.canvas_h,
        mean_conf=1.0,
    )


# ─── refine_positions ─────────────────────────────────────────────────────────

def refine_positions(
    estimate: PositionEstimate,
    offsets: List[Tuple[int, int]],
    confidences: Optional[List[float]] = None,
) -> PositionEstimate:
    """Уточнить позиции фрагментов добавлением смещений.

    Аргументы:
        estimate:    Исходная оценка позиций.
        offsets:     Список (dx, dy) — по одному на фрагмент.
        confidences: Список новых уверенностей (None → сохранить текущие).

    Возвращает:
        Уточнённый PositionEstimate.

    Исключения:
        ValueError: Если len(offsets) != n_fragments.
    """
    if len(offsets) != estimate.n_fragments:
        raise ValueError(
            f"Длина offsets ({len(offsets)}) должна совпадать с "
            f"n_fragments ({estimate.n_fragments})"
        )
    if confidences is not None and len(confidences) != estimate.n_fragments:
        raise ValueError(
            f"Длина confidences ({len(confidences)}) должна совпадать с "
            f"n_fragments ({estimate.n_fragments})"
        )

    new_positions: List[FragmentPosition] = []
    for i, pos in enumerate(estimate.positions):
        dx, dy = offsets[i]
        new_x = max(0, pos.x + dx)
        new_y = max(0, pos.y + dy)
        conf = confidences[i] if confidences is not None else pos.confidence
        new_positions.append(FragmentPosition(
            fragment_id=pos.fragment_id,
            x=new_x,
            y=new_y,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            method="refined",
        ))

    mean_conf = float(np.mean([p.confidence for p in new_positions])) \
        if new_positions else 0.0

    return PositionEstimate(
        positions=new_positions,
        n_fragments=len(new_positions),
        canvas_w=estimate.canvas_w,
        canvas_h=estimate.canvas_h,
        mean_conf=mean_conf,
    )


# ─── generate_position_candidates ────────────────────────────────────────────

def generate_position_candidates(
    x: int,
    y: int,
    radius: int,
    step: int = 1,
) -> List[Tuple[int, int]]:
    """Сгенерировать кандидатов позиций вокруг центральной точки.

    Аргументы:
        x:      Центральная координата X (>= 0).
        y:      Центральная координата Y (>= 0).
        radius: Радиус поиска (>= 0).
        step:   Шаг между кандидатами (>= 1).

    Возвращает:
        Список (cx, cy) в радиусе radius с шагом step.

    Исключения:
        ValueError: Если radius < 0 или step < 1.
    """
    if radius < 0:
        raise ValueError(f"radius должен быть >= 0, получено {radius}")
    if step < 1:
        raise ValueError(f"step должен быть >= 1, получено {step}")

    candidates: List[Tuple[int, int]] = []
    for dx in range(-radius, radius + 1, step):
        for dy in range(-radius, radius + 1, step):
            cx = max(0, x + dx)
            cy = max(0, y + dy)
            candidates.append((cx, cy))

    return candidates


# ─── batch_estimate_positions ─────────────────────────────────────────────────

def batch_estimate_positions(
    id_lists: List[List[int]],
    frag_w: int,
    frag_h: int,
    cfg: Optional[PositionConfig] = None,
) -> List[PositionEstimate]:
    """Оценить позиции для нескольких наборов фрагментов.

    Аргументы:
        id_lists: Список наборов ID фрагментов.
        frag_w:   Ширина фрагмента (>= 1).
        frag_h:   Высота фрагмента (>= 1).
        cfg:      Параметры.

    Возвращает:
        Список PositionEstimate.
    """
    return [estimate_grid_positions(ids, frag_w, frag_h, cfg) for ids in id_lists]
