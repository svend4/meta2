"""Обнаружение коллизий между размещёнными фрагментами.

Модуль проверяет, пересекаются ли прямоугольные области фрагментов
на холсте (AABB-тест), вычисляет глубину проникновения, строит граф
коллизий, предлагает вектор разрешения и выполняет пакетную проверку.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─── PlacedRect ───────────────────────────────────────────────────────────────

@dataclass
class PlacedRect:
    """Прямоугольная область фрагмента на холсте.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        x:           Координата X левого края (>= 0).
        y:           Координата Y верхнего края (>= 0).
        width:       Ширина (>= 1).
        height:      Высота (>= 1).
    """

    fragment_id: int
    x: int
    y: int
    width: int
    height: int

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
    def x2(self) -> int:
        """Правый край (x + width)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Нижний край (y + height)."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Центр прямоугольника."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> int:
        return self.width * self.height


# ─── CollisionInfo ────────────────────────────────────────────────────────────

@dataclass
class CollisionInfo:
    """Информация об одной коллизии между двумя фрагментами.

    Атрибуты:
        id1:          Идентификатор первого фрагмента.
        id2:          Идентификатор второго фрагмента.
        overlap_w:    Ширина области перекрытия (>= 0).
        overlap_h:    Высота области перекрытия (>= 0).
        overlap_area: Площадь перекрытия (>= 0).
        resolve_vec:  Вектор разрешения коллизии (dx, dy) для первого фрагмента.
    """

    id1: int
    id2: int
    overlap_w: int
    overlap_h: int
    overlap_area: int
    resolve_vec: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.overlap_w < 0:
            raise ValueError(
                f"overlap_w должен быть >= 0, получено {self.overlap_w}"
            )
        if self.overlap_h < 0:
            raise ValueError(
                f"overlap_h должен быть >= 0, получено {self.overlap_h}"
            )
        if self.overlap_area < 0:
            raise ValueError(
                f"overlap_area должен быть >= 0, получено {self.overlap_area}"
            )

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.id1, self.id2)


# ─── aabb_overlap ─────────────────────────────────────────────────────────────

def aabb_overlap(a: PlacedRect, b: PlacedRect) -> bool:
    """Проверить пересечение двух AABB (Axis-Aligned Bounding Box).

    Аргументы:
        a: Первый прямоугольник.
        b: Второй прямоугольник.

    Возвращает:
        True, если прямоугольники перекрываются (касание не считается).
    """
    return (a.x < b.x2 and a.x2 > b.x and
            a.y < b.y2 and a.y2 > b.y)


# ─── compute_overlap ──────────────────────────────────────────────────────────

def compute_overlap(a: PlacedRect, b: PlacedRect) -> Optional[CollisionInfo]:
    """Вычислить параметры перекрытия двух прямоугольников.

    Аргументы:
        a: Первый прямоугольник.
        b: Второй прямоугольник.

    Возвращает:
        CollisionInfo если есть перекрытие, иначе None.
    """
    if not aabb_overlap(a, b):
        return None

    ox = min(a.x2, b.x2) - max(a.x, b.x)
    oy = min(a.y2, b.y2) - max(a.y, b.y)
    ox = max(0, ox)
    oy = max(0, oy)

    # Вектор разрешения: толкаем `a` в сторону с наименьшим проникновением
    if ox <= oy:
        dx = ox if a.center[0] < b.center[0] else -ox
        dy = 0
    else:
        dx = 0
        dy = oy if a.center[1] < b.center[1] else -oy

    return CollisionInfo(
        id1=a.fragment_id,
        id2=b.fragment_id,
        overlap_w=ox,
        overlap_h=oy,
        overlap_area=ox * oy,
        resolve_vec=(dx, dy),
    )


# ─── detect_collisions ────────────────────────────────────────────────────────

def detect_collisions(rects: List[PlacedRect]) -> List[CollisionInfo]:
    """Обнаружить все коллизии между парами прямоугольников.

    Аргументы:
        rects: Список PlacedRect.

    Возвращает:
        Список CollisionInfo для всех коллизий (пары i < j).
    """
    collisions: List[CollisionInfo] = []
    n = len(rects)
    for i in range(n):
        for j in range(i + 1, n):
            info = compute_overlap(rects[i], rects[j])
            if info is not None:
                collisions.append(info)
    return collisions


# ─── collision_graph ──────────────────────────────────────────────────────────

def collision_graph(
    collisions: List[CollisionInfo],
) -> Dict[int, Set[int]]:
    """Построить граф коллизий (список смежности).

    Аргументы:
        collisions: Список CollisionInfo.

    Возвращает:
        Словарь {fragment_id: множество соседей с коллизией}.
    """
    graph: Dict[int, Set[int]] = {}
    for col in collisions:
        graph.setdefault(col.id1, set()).add(col.id2)
        graph.setdefault(col.id2, set()).add(col.id1)
    return graph


# ─── is_collision_free ────────────────────────────────────────────────────────

def is_collision_free(rects: List[PlacedRect]) -> bool:
    """Проверить отсутствие коллизий между всеми прямоугольниками.

    Аргументы:
        rects: Список PlacedRect.

    Возвращает:
        True если коллизий нет.
    """
    n = len(rects)
    for i in range(n):
        for j in range(i + 1, n):
            if aabb_overlap(rects[i], rects[j]):
                return False
    return True


# ─── total_overlap_area ───────────────────────────────────────────────────────

def total_overlap_area(collisions: List[CollisionInfo]) -> int:
    """Суммарная площадь всех перекрытий.

    Аргументы:
        collisions: Список CollisionInfo.

    Возвращает:
        Суммарная площадь перекрытий (int >= 0).
    """
    return sum(c.overlap_area for c in collisions)


# ─── resolve_greedy ───────────────────────────────────────────────────────────

def resolve_greedy(
    rects: List[PlacedRect],
    max_iter: int = 100,
) -> List[PlacedRect]:
    """Жадное разрешение коллизий (сдвиг первого фрагмента каждой пары).

    На каждой итерации применяет вектор разрешения к первому фрагменту
    каждой коллизии. Останавливается, когда коллизий нет или превышен
    лимит итераций.

    Аргументы:
        rects:    Список PlacedRect.
        max_iter: Максимальное число итераций (>= 1).

    Возвращает:
        Новый список PlacedRect (исходный не изменяется).

    Исключения:
        ValueError: Если max_iter < 1.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter должен быть >= 1, получено {max_iter}")

    # Создаём изменяемые копии
    result = [
        PlacedRect(r.fragment_id, r.x, r.y, r.width, r.height)
        for r in rects
    ]
    id_to_idx = {r.fragment_id: i for i, r in enumerate(result)}

    for _ in range(max_iter):
        collisions = detect_collisions(result)
        if not collisions:
            break
        for col in collisions:
            idx = id_to_idx[col.id1]
            dx, dy = col.resolve_vec
            new_x = max(0, result[idx].x + dx)
            new_y = max(0, result[idx].y + dy)
            result[idx] = PlacedRect(
                result[idx].fragment_id, new_x, new_y,
                result[idx].width, result[idx].height
            )

    return result


# ─── batch_detect ─────────────────────────────────────────────────────────────

def batch_detect(
    rect_groups: List[List[PlacedRect]],
) -> List[List[CollisionInfo]]:
    """Обнаружить коллизии в нескольких группах прямоугольников.

    Аргументы:
        rect_groups: Список групп (каждая группа — список PlacedRect).

    Возвращает:
        Список списков CollisionInfo, по одному на группу.
    """
    return [detect_collisions(group) for group in rect_groups]
