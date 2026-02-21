"""Сортировка фрагментов пазла для определения порядка сборки.

Модуль предоставляет стратегии упорядочивания фрагментов перед сборкой:
по идентификатору, площади, оценке совместимости или случайно.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_STRATEGIES = {"area", "score", "id", "random"}


# ─── SortConfig ───────────────────────────────────────────────────────────────

@dataclass
class SortConfig:
    """Параметры сортировки фрагментов.

    Атрибуты:
        strategy: Стратегия: 'area' | 'score' | 'id' | 'random'.
        reverse:  Сортировать по убыванию, если True.
        seed:     Зерно ГПСЧ для 'random' (>= 0).
    """

    strategy: str = "id"
    reverse: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy должна быть одной из {_STRATEGIES}, "
                f"получено '{self.strategy}'"
            )
        if self.seed < 0:
            raise ValueError(f"seed должен быть >= 0, получено {self.seed}")


# ─── FragmentSortInfo ──────────────────────────────────────────────────────────

@dataclass
class FragmentSortInfo:
    """Описание фрагмента для сортировки.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        area:        Площадь фрагмента (>= 0).
        score:       Оценка совместимости (>= 0).
        meta:        Произвольные метаданные.
    """

    fragment_id: int
    area: float = 0.0
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.area < 0.0:
            raise ValueError(
                f"area должна быть >= 0, получено {self.area}"
            )
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )


# ─── SortedFragment ───────────────────────────────────────────────────────────

@dataclass
class SortedFragment:
    """Фрагмент с назначенной позицией в порядке сборки.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        position:    Позиция в порядке сборки (>= 0).
        area:        Площадь фрагмента (>= 0).
        score:       Оценка совместимости (>= 0).
    """

    fragment_id: int
    position: int
    area: float = 0.0
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.position < 0:
            raise ValueError(
                f"position должна быть >= 0, получено {self.position}"
            )
        if self.area < 0.0:
            raise ValueError(
                f"area должна быть >= 0, получено {self.area}"
            )
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )

    @property
    def info(self) -> str:
        """Краткое описание фрагмента."""
        return (f"Fragment {self.fragment_id} @ pos {self.position} "
                f"[area={self.area:.1f}, score={self.score:.3f}]")


# ─── sort_by_id ───────────────────────────────────────────────────────────────

def sort_by_id(
    fragments: List[FragmentSortInfo],
    reverse: bool = False,
) -> List[FragmentSortInfo]:
    """Сортировать фрагменты по идентификатору.

    Аргументы:
        fragments: Список фрагментов.
        reverse:   По убыванию, если True.

    Возвращает:
        Отсортированный список.
    """
    return sorted(fragments, key=lambda f: f.fragment_id, reverse=reverse)


# ─── sort_by_area ─────────────────────────────────────────────────────────────

def sort_by_area(
    fragments: List[FragmentSortInfo],
    reverse: bool = False,
) -> List[FragmentSortInfo]:
    """Сортировать фрагменты по площади.

    Аргументы:
        fragments: Список фрагментов.
        reverse:   По убыванию, если True.

    Возвращает:
        Отсортированный список.
    """
    return sorted(fragments, key=lambda f: f.area, reverse=reverse)


# ─── sort_by_score ────────────────────────────────────────────────────────────

def sort_by_score(
    fragments: List[FragmentSortInfo],
    reverse: bool = False,
) -> List[FragmentSortInfo]:
    """Сортировать фрагменты по оценке совместимости.

    Аргументы:
        fragments: Список фрагментов.
        reverse:   По убыванию, если True.

    Возвращает:
        Отсортированный список.
    """
    return sorted(fragments, key=lambda f: f.score, reverse=reverse)


# ─── sort_random ──────────────────────────────────────────────────────────────

def sort_random(
    fragments: List[FragmentSortInfo],
    seed: int = 0,
) -> List[FragmentSortInfo]:
    """Перемешать фрагменты случайным образом.

    Аргументы:
        fragments: Список фрагментов.
        seed:      Зерно ГПСЧ (>= 0).

    Возвращает:
        Перемешанный список.

    Исключения:
        ValueError: Если seed < 0.
    """
    if seed < 0:
        raise ValueError(f"seed должен быть >= 0, получено {seed}")
    result = list(fragments)
    rng = _random.Random(seed)
    rng.shuffle(result)
    return result


# ─── sort_fragments ───────────────────────────────────────────────────────────

def sort_fragments(
    fragments: List[FragmentSortInfo],
    cfg: Optional[SortConfig] = None,
) -> List[FragmentSortInfo]:
    """Сортировать фрагменты согласно конфигурации.

    Аргументы:
        fragments: Список фрагментов.
        cfg:       Параметры (None → SortConfig()).

    Возвращает:
        Отсортированный список.
    """
    if cfg is None:
        cfg = SortConfig()

    if cfg.strategy == "id":
        return sort_by_id(fragments, cfg.reverse)
    if cfg.strategy == "area":
        return sort_by_area(fragments, cfg.reverse)
    if cfg.strategy == "score":
        return sort_by_score(fragments, cfg.reverse)
    # random
    return sort_random(fragments, cfg.seed)


# ─── assign_positions ─────────────────────────────────────────────────────────

def assign_positions(
    fragments: List[FragmentSortInfo],
) -> List[SortedFragment]:
    """Назначить позиции фрагментам (0-based).

    Аргументы:
        fragments: Список фрагментов в желаемом порядке.

    Возвращает:
        Список SortedFragment с position 0, 1, 2, ...
    """
    return [
        SortedFragment(
            fragment_id=f.fragment_id,
            position=i,
            area=f.area,
            score=f.score,
        )
        for i, f in enumerate(fragments)
    ]


# ─── reorder_by_positions ─────────────────────────────────────────────────────

def reorder_by_positions(
    sorted_frags: List[SortedFragment],
) -> List[SortedFragment]:
    """Упорядочить список SortedFragment по полю position.

    Аргументы:
        sorted_frags: Список SortedFragment.

    Возвращает:
        Список, упорядоченный по возрастанию position.
    """
    return sorted(sorted_frags, key=lambda sf: sf.position)


# ─── batch_sort ───────────────────────────────────────────────────────────────

def batch_sort(
    fragment_lists: List[List[FragmentSortInfo]],
    cfg: Optional[SortConfig] = None,
) -> List[List[SortedFragment]]:
    """Сортировать несколько списков фрагментов.

    Аргументы:
        fragment_lists: Список списков фрагментов.
        cfg:            Параметры.

    Возвращает:
        Список списков SortedFragment.
    """
    return [assign_positions(sort_fragments(fl, cfg)) for fl in fragment_lists]
