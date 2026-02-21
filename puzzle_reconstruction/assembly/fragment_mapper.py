"""Маппинг фрагментов к зонам холста.

Модуль предоставляет структуры и функции для разбивки холста на
прямоугольные зоны и назначения фрагментов в соответствующие зоны
на основе их позиций или оценок сходства.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── MapConfig ────────────────────────────────────────────────────────────────

@dataclass
class MapConfig:
    """Параметры маппинга фрагментов к зонам холста.

    Атрибуты:
        canvas_w:    Ширина холста в пикселях (>= 1).
        canvas_h:    Высота холста в пикселях (>= 1).
        n_zones_x:   Число зон по горизонтали (>= 1).
        n_zones_y:   Число зон по вертикали (>= 1).
        allow_multi: Разрешить одному фрагменту попадать в несколько зон.
    """

    canvas_w: int = 512
    canvas_h: int = 512
    n_zones_x: int = 4
    n_zones_y: int = 4
    allow_multi: bool = False

    def __post_init__(self) -> None:
        for name, val in (
            ("canvas_w", self.canvas_w),
            ("canvas_h", self.canvas_h),
            ("n_zones_x", self.n_zones_x),
            ("n_zones_y", self.n_zones_y),
        ):
            if val < 1:
                raise ValueError(f"{name} должен быть >= 1, получено {val}")


# ─── FragmentZone ──────────────────────────────────────────────────────────────

@dataclass
class FragmentZone:
    """Результат назначения одного фрагмента к зоне.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        zone_x:      Индекс зоны по горизонтали (>= 0).
        zone_y:      Индекс зоны по вертикали (>= 0).
        confidence:  Уверенность в назначении [0, 1].
    """

    fragment_id: int
    zone_x: int
    zone_y: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.zone_x < 0:
            raise ValueError(
                f"zone_x должен быть >= 0, получено {self.zone_x}"
            )
        if self.zone_y < 0:
            raise ValueError(
                f"zone_y должен быть >= 0, получено {self.zone_y}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence должен быть в [0, 1], получено {self.confidence}"
            )

    @property
    def zone_index(self) -> Tuple[int, int]:
        """Индекс зоны в виде (zone_x, zone_y)."""
        return (self.zone_x, self.zone_y)


# ─── MapResult ────────────────────────────────────────────────────────────────

@dataclass
class MapResult:
    """Итог маппинга набора фрагментов.

    Атрибуты:
        assignments:  Список FragmentZone.
        n_fragments:  Число фрагментов на входе (>= 0).
        n_zones:      Общее число зон (>= 1).
        n_assigned:   Число фрагментов, получивших зону (>= 0).
    """

    assignments: List[FragmentZone]
    n_fragments: int
    n_zones: int
    n_assigned: int

    def __post_init__(self) -> None:
        for name, val in (
            ("n_fragments", self.n_fragments),
            ("n_assigned", self.n_assigned),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")
        if self.n_zones < 1:
            raise ValueError(
                f"n_zones должен быть >= 1, получено {self.n_zones}"
            )

    @property
    def by_fragment(self) -> Dict[int, FragmentZone]:
        """Словарь {fragment_id: FragmentZone}."""
        return {fz.fragment_id: fz for fz in self.assignments}

    @property
    def by_zone(self) -> Dict[Tuple[int, int], List[int]]:
        """Словарь {(zone_x, zone_y): [fragment_id, ...]}."""
        result: Dict[Tuple[int, int], List[int]] = {}
        for fz in self.assignments:
            key = fz.zone_index
            result.setdefault(key, []).append(fz.fragment_id)
        return result

    @property
    def coverage_ratio(self) -> float:
        """Доля зон, содержащих хотя бы один фрагмент."""
        if self.n_zones == 0:
            return 0.0
        occupied = len(set(fz.zone_index for fz in self.assignments))
        return float(occupied) / float(self.n_zones)


# ─── compute_zone_grid ────────────────────────────────────────────────────────

def compute_zone_grid(
    cfg: Optional[MapConfig] = None,
) -> List[Tuple[int, int, int, int]]:
    """Вычислить прямоугольники зон на холсте.

    Аргументы:
        cfg: Параметры маппинга (None → MapConfig()).

    Возвращает:
        Список (x0, y0, x1, y1) для каждой зоны в порядке (zone_y, zone_x).
    """
    if cfg is None:
        cfg = MapConfig()
    zone_w = cfg.canvas_w / cfg.n_zones_x
    zone_h = cfg.canvas_h / cfg.n_zones_y
    zones: List[Tuple[int, int, int, int]] = []
    for zy in range(cfg.n_zones_y):
        for zx in range(cfg.n_zones_x):
            x0 = int(zx * zone_w)
            y0 = int(zy * zone_h)
            x1 = int((zx + 1) * zone_w)
            y1 = int((zy + 1) * zone_h)
            zones.append((x0, y0, x1, y1))
    return zones


# ─── assign_to_zone ───────────────────────────────────────────────────────────

def assign_to_zone(
    x: int,
    y: int,
    cfg: Optional[MapConfig] = None,
) -> Tuple[int, int]:
    """Определить индекс зоны для точки (x, y).

    Аргументы:
        x:   Координата X (пиксели).
        y:   Координата Y (пиксели).
        cfg: Параметры маппинга.

    Возвращает:
        (zone_x, zone_y) — индексы зоны, зажатые в допустимые границы.
    """
    if cfg is None:
        cfg = MapConfig()
    zone_w = cfg.canvas_w / cfg.n_zones_x
    zone_h = cfg.canvas_h / cfg.n_zones_y
    zx = int(x / zone_w)
    zy = int(y / zone_h)
    zx = min(zx, cfg.n_zones_x - 1)
    zy = min(zy, cfg.n_zones_y - 1)
    return (max(zx, 0), max(zy, 0))


# ─── build_fragment_map ───────────────────────────────────────────────────────

def build_fragment_map(
    fragment_ids: List[int],
    positions: List[Tuple[int, int]],
    cfg: Optional[MapConfig] = None,
) -> MapResult:
    """Построить маппинг фрагментов к зонам на основе позиций.

    Аргументы:
        fragment_ids: Список идентификаторов фрагментов.
        positions:    Список (x, y) для каждого фрагмента.
        cfg:          Параметры маппинга.

    Возвращает:
        MapResult.

    Исключения:
        ValueError: Если len(fragment_ids) != len(positions).
    """
    if cfg is None:
        cfg = MapConfig()
    if len(fragment_ids) != len(positions):
        raise ValueError(
            f"Длины fragment_ids ({len(fragment_ids)}) и positions "
            f"({len(positions)}) не совпадают"
        )

    n_zones = cfg.n_zones_x * cfg.n_zones_y
    assignments: List[FragmentZone] = []

    for fid, (x, y) in zip(fragment_ids, positions):
        zx, zy = assign_to_zone(x, y, cfg)
        assignments.append(FragmentZone(
            fragment_id=fid,
            zone_x=zx,
            zone_y=zy,
        ))

    return MapResult(
        assignments=assignments,
        n_fragments=len(fragment_ids),
        n_zones=n_zones,
        n_assigned=len(assignments),
    )


# ─── remap_fragments ──────────────────────────────────────────────────────────

def remap_fragments(
    result: MapResult,
    id_mapping: Dict[int, int],
) -> MapResult:
    """Переиндексировать fragment_id по словарю.

    Аргументы:
        result:     Исходный MapResult.
        id_mapping: Словарь {old_id: new_id}.

    Возвращает:
        Новый MapResult с обновлёнными идентификаторами (неизвестные пропускаются).
    """
    new_assignments: List[FragmentZone] = []
    for fz in result.assignments:
        new_id = id_mapping.get(fz.fragment_id)
        if new_id is None:
            continue
        new_assignments.append(FragmentZone(
            fragment_id=new_id,
            zone_x=fz.zone_x,
            zone_y=fz.zone_y,
            confidence=fz.confidence,
        ))
    return MapResult(
        assignments=new_assignments,
        n_fragments=result.n_fragments,
        n_zones=result.n_zones,
        n_assigned=len(new_assignments),
    )


# ─── score_mapping ────────────────────────────────────────────────────────────

def score_mapping(result: MapResult) -> float:
    """Вычислить оценку качества маппинга.

    Метрика: среднее confidence * coverage_ratio.

    Аргументы:
        result: MapResult.

    Возвращает:
        Оценка в [0, 1] (0 если нет назначений).
    """
    if not result.assignments:
        return 0.0
    mean_conf = float(np.mean([fz.confidence for fz in result.assignments]))
    return float(np.clip(mean_conf * result.coverage_ratio, 0.0, 1.0))


# ─── batch_build_fragment_maps ────────────────────────────────────────────────

def batch_build_fragment_maps(
    id_lists: List[List[int]],
    position_lists: List[List[Tuple[int, int]]],
    cfg: Optional[MapConfig] = None,
) -> List[MapResult]:
    """Построить маппинги для нескольких наборов фрагментов.

    Аргументы:
        id_lists:       Список списков идентификаторов.
        position_lists: Список списков позиций.
        cfg:            Общие параметры маппинга.

    Возвращает:
        Список MapResult.

    Исключения:
        ValueError: Если длины id_lists и position_lists не совпадают.
    """
    if len(id_lists) != len(position_lists):
        raise ValueError(
            f"Длины id_lists ({len(id_lists)}) и position_lists "
            f"({len(position_lists)}) не совпадают"
        )
    return [build_fragment_map(ids, pos, cfg)
            for ids, pos in zip(id_lists, position_lists)]
