"""rotation_score_utils — утилиты оценки качества оценок поворота.

Предоставляет функции для агрегации, ранжирования, фильтрации и
статистического анализа результатов оценки углов поворота фрагментов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class RotationScoreConfig:
    """Параметры агрегации и фильтрации оценок поворота.

    Attributes
    ----------
    min_confidence:
        Минимальный порог confidence (0..1) для учёта результата.
    angle_tolerance_deg:
        Допустимое отклонение угла (°) при сравнении.
    preferred_method:
        Предпочтительный метод ('pca', 'moments', 'gradient', 'refine', '').
    """
    min_confidence:    float = 0.0
    angle_tolerance_deg: float = 5.0
    preferred_method:  str   = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence должен быть в [0, 1], получено {self.min_confidence}"
            )
        if self.angle_tolerance_deg < 0.0:
            raise ValueError(
                f"angle_tolerance_deg не может быть отрицательным, "
                f"получено {self.angle_tolerance_deg}"
            )


# ─── Запись результата оценки ─────────────────────────────────────────────────

@dataclass
class RotationScoreEntry:
    """Результат одной оценки поворота.

    Attributes
    ----------
    image_id:
        Идентификатор изображения/фрагмента.
    angle_deg:
        Оцененный угол (°).
    confidence:
        Достоверность оценки [0, 1].
    method:
        Метод ('pca', 'moments', 'gradient', 'refine').
    meta:
        Дополнительные данные.
    """
    image_id:   int
    angle_deg:  float
    confidence: float
    method:     str
    meta:       Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_id < 0:
            raise ValueError(f"image_id не может быть отрицательным: {self.image_id}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence должен быть в [0, 1]: {self.confidence}"
            )

    def __repr__(self) -> str:
        return (f"RotationScoreEntry(id={self.image_id}, "
                f"angle={self.angle_deg:.2f}°, "
                f"conf={self.confidence:.3f}, method={self.method!r})")


# ─── Утилиты создания ─────────────────────────────────────────────────────────

def make_entry(
    image_id:   int,
    angle_deg:  float,
    confidence: float,
    method:     str,
    meta:       Optional[Dict] = None,
) -> RotationScoreEntry:
    """Создать RotationScoreEntry с проверкой.

    Parameters
    ----------
    image_id:
        ID изображения (≥ 0).
    angle_deg:
        Угол в градусах.
    confidence:
        Достоверность [0, 1].
    method:
        Метод оценки.
    meta:
        Дополнительные данные.

    Returns
    -------
    RotationScoreEntry
    """
    return RotationScoreEntry(
        image_id=image_id,
        angle_deg=float(angle_deg),
        confidence=float(confidence),
        method=method,
        meta=dict(meta) if meta else {},
    )


# ─── Фильтрация ───────────────────────────────────────────────────────────────

def filter_by_confidence(
    entries: List[RotationScoreEntry],
    min_confidence: float = 0.0,
) -> List[RotationScoreEntry]:
    """Оставить только записи с confidence ≥ min_confidence.

    Parameters
    ----------
    entries:
        Список RotationScoreEntry.
    min_confidence:
        Минимальный порог [0, 1].

    Returns
    -------
    Отфильтрованный список (порядок сохранён).
    """
    return [e for e in entries if e.confidence >= min_confidence]


def filter_by_method(
    entries: List[RotationScoreEntry],
    method:  str,
) -> List[RotationScoreEntry]:
    """Оставить только записи с указанным методом.

    Parameters
    ----------
    entries:
        Список RotationScoreEntry.
    method:
        Имя метода.

    Returns
    -------
    Отфильтрованный список.
    """
    return [e for e in entries if e.method == method]


def filter_by_angle_range(
    entries:    List[RotationScoreEntry],
    min_angle:  float = -90.0,
    max_angle:  float =  90.0,
) -> List[RotationScoreEntry]:
    """Оставить только записи с углом в заданном диапазоне.

    Parameters
    ----------
    entries:
        Список RotationScoreEntry.
    min_angle, max_angle:
        Границы диапазона углов (включительно).

    Returns
    -------
    Отфильтрованный список.
    """
    if min_angle > max_angle:
        raise ValueError(
            f"min_angle ({min_angle}) > max_angle ({max_angle})"
        )
    return [e for e in entries if min_angle <= e.angle_deg <= max_angle]


# ─── Ранжирование и агрегация ─────────────────────────────────────────────────

def rank_by_confidence(
    entries: List[RotationScoreEntry],
    reverse: bool = True,
) -> List[RotationScoreEntry]:
    """Отсортировать записи по убыванию (или возрастанию) confidence.

    Returns
    -------
    Новый отсортированный список.
    """
    return sorted(entries, key=lambda e: e.confidence, reverse=reverse)


def best_entry(
    entries: List[RotationScoreEntry],
    cfg:     Optional[RotationScoreConfig] = None,
) -> Optional[RotationScoreEntry]:
    """Вернуть запись с наибольшей confidence (с учётом фильтра).

    Parameters
    ----------
    entries:
        Список записей.
    cfg:
        Конфигурация (применяется min_confidence и preferred_method).

    Returns
    -------
    RotationScoreEntry или None, если список пуст / все отфильтрованы.
    """
    if cfg is None:
        cfg = RotationScoreConfig()
    candidates = filter_by_confidence(entries, cfg.min_confidence)
    if cfg.preferred_method:
        preferred = filter_by_method(candidates, cfg.preferred_method)
        if preferred:
            candidates = preferred
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.confidence)


def aggregate_angles(
    entries: List[RotationScoreEntry],
    weights: Optional[List[float]] = None,
) -> float:
    """Вычислить средневзвешенный угол по confidence (или weights).

    Parameters
    ----------
    entries:
        Список записей.
    weights:
        Явные веса; если None — используются confidence.

    Returns
    -------
    float — средневзвешенный угол (°); 0.0 если записей нет.
    """
    if not entries:
        return 0.0
    if weights is None:
        w = np.array([e.confidence for e in entries], dtype=np.float64)
    else:
        w = np.array(weights, dtype=np.float64)
    angles = np.array([e.angle_deg for e in entries], dtype=np.float64)
    total_w = w.sum()
    if total_w < 1e-10:
        return float(angles.mean())
    return float((w * angles).sum() / total_w)


# ─── Статистика ───────────────────────────────────────────────────────────────

def rotation_score_stats(
    entries: List[RotationScoreEntry],
) -> Dict:
    """Вычислить статистику по списку RotationScoreEntry.

    Returns
    -------
    dict с ключами:
        'n', 'mean_angle', 'std_angle', 'mean_confidence',
        'min_confidence', 'max_confidence', 'methods'.
    """
    if not entries:
        return {
            "n": 0, "mean_angle": 0.0, "std_angle": 0.0,
            "mean_confidence": 0.0, "min_confidence": 0.0,
            "max_confidence": 0.0, "methods": [],
        }
    angles = np.array([e.angle_deg for e in entries], dtype=np.float64)
    confs  = np.array([e.confidence for e in entries], dtype=np.float64)
    return {
        "n":               len(entries),
        "mean_angle":      float(angles.mean()),
        "std_angle":       float(angles.std()),
        "mean_confidence": float(confs.mean()),
        "min_confidence":  float(confs.min()),
        "max_confidence":  float(confs.max()),
        "methods":         list({e.method for e in entries}),
    }


def angle_agreement(
    entries:         List[RotationScoreEntry],
    tolerance_deg:   float = 5.0,
) -> float:
    """Доля пар записей, согласованных по углу (|Δangle| ≤ tolerance).

    Parameters
    ----------
    entries:
        Список записей.
    tolerance_deg:
        Порог согласования (°).

    Returns
    -------
    float в [0, 1]; 1.0 если все записи согласованы.
    """
    n = len(entries)
    if n <= 1:
        return 1.0
    angles = np.array([e.angle_deg for e in entries], dtype=np.float64)
    pairs = 0
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if abs(angles[i] - angles[j]) <= tolerance_deg:
                agree += 1
    return agree / pairs if pairs > 0 else 1.0


# ─── Пакетная обработка ───────────────────────────────────────────────────────

def batch_make_entries(
    image_ids:   List[int],
    angles_deg:  List[float],
    confidences: List[float],
    methods:     List[str],
) -> List[RotationScoreEntry]:
    """Пакетно создать список RotationScoreEntry.

    Parameters
    ----------
    image_ids, angles_deg, confidences, methods:
        Параллельные списки одинаковой длины.

    Returns
    -------
    Список RotationScoreEntry.
    """
    if not (len(image_ids) == len(angles_deg)
            == len(confidences) == len(methods)):
        raise ValueError("Все входные списки должны иметь одинаковую длину")
    return [
        make_entry(i, a, c, m)
        for i, a, c, m in zip(image_ids, angles_deg, confidences, methods)
    ]


def top_k_entries(
    entries: List[RotationScoreEntry],
    k:       int,
    cfg:     Optional[RotationScoreConfig] = None,
) -> List[RotationScoreEntry]:
    """Вернуть top-k записей по убыванию confidence.

    Parameters
    ----------
    entries:
        Список записей.
    k:
        Количество лучших записей.
    cfg:
        Конфигурация (применяется min_confidence).

    Returns
    -------
    Список из не более k RotationScoreEntry.
    """
    if cfg is None:
        cfg = RotationScoreConfig()
    filtered = filter_by_confidence(entries, cfg.min_confidence)
    return rank_by_confidence(filtered)[:k]


def group_by_method(
    entries: List[RotationScoreEntry],
) -> Dict[str, List[RotationScoreEntry]]:
    """Сгруппировать записи по методу оценки.

    Returns
    -------
    dict {method: [entries...]}
    """
    groups: Dict[str, List[RotationScoreEntry]] = {}
    for e in entries:
        groups.setdefault(e.method, []).append(e)
    return groups
