"""overlap_score_utils — утилиты оценки и анализа перекрытий фрагментов.

Содержит конфигурацию, структуры данных и вспомогательные функции
для работы с результатами проверки перекрытий в задаче сборки паззла.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class OverlapScoreConfig:
    """Параметры анализа перекрытий.

    Attributes
    ----------
    iou_threshold:
        Минимальный IoU для регистрации перекрытия как значимого [0, 1].
    area_threshold:
        Минимальная площадь перекрытия (пикс.) для учёта.
    penalise_self_overlap:
        Учитывать ли самоперекрытие (idx1 == idx2).
    """
    iou_threshold:         float = 0.05
    area_threshold:        float = 1.0
    penalise_self_overlap: bool  = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"iou_threshold должен быть в [0, 1]: {self.iou_threshold}"
            )
        if self.area_threshold < 0.0:
            raise ValueError(
                f"area_threshold не может быть отрицательным: {self.area_threshold}"
            )


# ─── Запись о перекрытии ──────────────────────────────────────────────────────

@dataclass
class OverlapScoreEntry:
    """Результат оценки одного перекрытия.

    Attributes
    ----------
    idx1, idx2:
        Индексы пары фрагментов.
    iou:
        Intersection-over-Union [0, 1].
    overlap_area:
        Площадь пересечения (пикс.).
    penalty:
        Штраф за перекрытие [0, 1] (выше → хуже).
    meta:
        Дополнительные данные.
    """
    idx1:         int
    idx2:         int
    iou:          float
    overlap_area: float
    penalty:      float = 0.0
    meta:         Dict  = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0 or self.idx2 < 0:
            raise ValueError(
                f"Индексы не могут быть отрицательными: idx1={self.idx1}, idx2={self.idx2}"
            )
        if not (0.0 <= self.iou <= 1.0):
            raise ValueError(f"iou должен быть в [0, 1]: {self.iou}")
        if self.overlap_area < 0.0:
            raise ValueError(f"overlap_area не может быть отрицательным: {self.overlap_area}")
        if not (0.0 <= self.penalty <= 1.0):
            raise ValueError(f"penalty должен быть в [0, 1]: {self.penalty}")

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.idx1, self.idx2)

    def __repr__(self) -> str:
        return (f"OverlapScoreEntry(pair=({self.idx1},{self.idx2}), "
                f"iou={self.iou:.3f}, area={self.overlap_area:.1f}, "
                f"penalty={self.penalty:.3f})")


# ─── Сводный результат оценки ─────────────────────────────────────────────────

@dataclass
class OverlapSummary:
    """Сводка результатов оценки перекрытий.

    Attributes
    ----------
    entries:
        Список всех обнаруженных перекрытий.
    n_overlaps:
        Количество значимых перекрытий.
    total_area:
        Суммарная площадь перекрытий.
    max_iou:
        Максимальный IoU среди всех пар.
    mean_penalty:
        Средний штраф.
    is_valid:
        True если перекрытий нет или все ниже порогов.
    """
    entries:     List[OverlapScoreEntry]
    n_overlaps:  int
    total_area:  float
    max_iou:     float
    mean_penalty: float
    is_valid:    bool

    def __post_init__(self) -> None:
        if self.n_overlaps < 0:
            raise ValueError(f"n_overlaps не может быть отрицательным: {self.n_overlaps}")
        if self.total_area < 0.0:
            raise ValueError(f"total_area не может быть отрицательным: {self.total_area}")

    def __repr__(self) -> str:
        status = "OK" if self.is_valid else "OVERLAPS"
        return (f"OverlapSummary({status}, n={self.n_overlaps}, "
                f"max_iou={self.max_iou:.3f}, total_area={self.total_area:.1f})")


# ─── Конструирование записей ──────────────────────────────────────────────────

def make_overlap_entry(
    idx1:         int,
    idx2:         int,
    iou:          float,
    overlap_area: float,
    cfg:          Optional[OverlapScoreConfig] = None,
    meta:         Optional[Dict] = None,
) -> OverlapScoreEntry:
    """Создать OverlapScoreEntry с автоматическим расчётом штрафа.

    Parameters
    ----------
    idx1, idx2:
        Индексы пары фрагментов (≥ 0).
    iou:
        IoU [0, 1].
    overlap_area:
        Площадь пересечения (пикс.).
    cfg:
        Конфигурация (используется для порогов).
    meta:
        Дополнительные данные.

    Returns
    -------
    OverlapScoreEntry
    """
    if cfg is None:
        cfg = OverlapScoreConfig()
    # Штраф = iou, если перекрытие значимо
    is_significant = (iou >= cfg.iou_threshold
                      and overlap_area >= cfg.area_threshold)
    penalty = float(iou) if is_significant else 0.0
    return OverlapScoreEntry(
        idx1=idx1,
        idx2=idx2,
        iou=float(iou),
        overlap_area=float(overlap_area),
        penalty=penalty,
        meta=dict(meta) if meta else {},
    )


# ─── Агрегация ────────────────────────────────────────────────────────────────

def summarise_overlaps(
    entries: List[OverlapScoreEntry],
    cfg:     Optional[OverlapScoreConfig] = None,
) -> OverlapSummary:
    """Создать сводку по списку OverlapScoreEntry.

    Parameters
    ----------
    entries:
        Список записей о перекрытиях.
    cfg:
        Конфигурация порогов.

    Returns
    -------
    OverlapSummary
    """
    if cfg is None:
        cfg = OverlapScoreConfig()
    significant = [
        e for e in entries
        if e.iou >= cfg.iou_threshold and e.overlap_area >= cfg.area_threshold
    ]
    n = len(significant)
    total_area = sum(e.overlap_area for e in significant)
    max_iou = max((e.iou for e in significant), default=0.0)
    mean_penalty = (sum(e.penalty for e in significant) / n if n > 0 else 0.0)
    return OverlapSummary(
        entries=significant,
        n_overlaps=n,
        total_area=total_area,
        max_iou=max_iou,
        mean_penalty=mean_penalty,
        is_valid=(n == 0),
    )


# ─── Фильтрация ───────────────────────────────────────────────────────────────

def filter_significant_overlaps(
    entries:       List[OverlapScoreEntry],
    iou_threshold: float = 0.05,
) -> List[OverlapScoreEntry]:
    """Отфильтровать записи с IoU ≥ порог.

    Parameters
    ----------
    entries:
        Исходный список.
    iou_threshold:
        Минимальный IoU [0, 1].

    Returns
    -------
    Отфильтрованный список.
    """
    return [e for e in entries if e.iou >= iou_threshold]


def filter_by_area(
    entries:        List[OverlapScoreEntry],
    min_area:       float = 0.0,
) -> List[OverlapScoreEntry]:
    """Оставить только записи с площадью ≥ min_area."""
    return [e for e in entries if e.overlap_area >= min_area]


def top_k_overlaps(
    entries: List[OverlapScoreEntry],
    k:       int,
) -> List[OverlapScoreEntry]:
    """Вернуть k записей с наибольшим IoU.

    Parameters
    ----------
    entries:
        Список записей.
    k:
        Количество записей.

    Returns
    -------
    Отсортированный по убыванию IoU список из ≤ k элементов.
    """
    return sorted(entries, key=lambda e: e.iou, reverse=True)[:k]


# ─── Статистика ───────────────────────────────────────────────────────────────

def overlap_stats(
    entries: List[OverlapScoreEntry],
) -> Dict:
    """Вычислить статистику по набору перекрытий.

    Returns
    -------
    dict с ключами: 'n', 'mean_iou', 'std_iou', 'max_iou', 'min_iou',
                    'total_area', 'mean_area'.
    """
    if not entries:
        return {
            "n": 0, "mean_iou": 0.0, "std_iou": 0.0,
            "max_iou": 0.0, "min_iou": 0.0,
            "total_area": 0.0, "mean_area": 0.0,
        }
    ious  = np.array([e.iou for e in entries], dtype=np.float64)
    areas = np.array([e.overlap_area for e in entries], dtype=np.float64)
    return {
        "n":          len(entries),
        "mean_iou":   float(ious.mean()),
        "std_iou":    float(ious.std()),
        "max_iou":    float(ious.max()),
        "min_iou":    float(ious.min()),
        "total_area": float(areas.sum()),
        "mean_area":  float(areas.mean()),
    }


def penalty_score(
    entries: List[OverlapScoreEntry],
) -> float:
    """Суммарный штраф за все перекрытия (среднее penalty).

    Returns 0.0 если нет перекрытий, иначе среднее penalty в [0, 1].
    """
    if not entries:
        return 0.0
    return float(np.mean([e.penalty for e in entries]))


# ─── Пакетная обработка ───────────────────────────────────────────────────────

def batch_make_overlap_entries(
    pairs:         List[Tuple[int, int]],
    ious:          List[float],
    overlap_areas: List[float],
    cfg:           Optional[OverlapScoreConfig] = None,
) -> List[OverlapScoreEntry]:
    """Пакетно создать список OverlapScoreEntry.

    Parameters
    ----------
    pairs:
        Список пар индексов.
    ious:
        Список значений IoU.
    overlap_areas:
        Список площадей.
    cfg:
        Конфигурация.

    Returns
    -------
    Список OverlapScoreEntry.
    """
    if not (len(pairs) == len(ious) == len(overlap_areas)):
        raise ValueError("Все входные списки должны иметь одинаковую длину")
    return [
        make_overlap_entry(idx1, idx2, iou, area, cfg)
        for (idx1, idx2), iou, area in zip(pairs, ious, overlap_areas)
    ]


def group_by_fragment(
    entries: List[OverlapScoreEntry],
) -> Dict[int, List[OverlapScoreEntry]]:
    """Сгруппировать записи по первому индексу фрагмента.

    Returns
    -------
    dict {idx1: [entries...]}
    """
    groups: Dict[int, List[OverlapScoreEntry]] = {}
    for e in entries:
        groups.setdefault(e.idx1, []).append(e)
    return groups
