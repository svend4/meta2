"""Комбинирование дескрипторов фрагментов для улучшения распознавания.

Модуль объединяет несколько видов дескрипторов (форма, текстура, цвет,
градиент) в единый вектор признаков с опциональной нормализацией,
взвешиванием и понижением размерности.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── CombineConfig ────────────────────────────────────────────────────────────

@dataclass
class CombineConfig:
    """Параметры комбинирования дескрипторов.

    Атрибуты:
        weights:    Словарь {имя_дескриптора: вес} (все >= 0).
        normalize:  Нормализовать каждый дескриптор перед объединением.
        l2_final:   L2-нормировать итоговый вектор.
        pca_dim:    Размерность PCA (None = без PCA).
    """

    weights: Dict[str, float] = field(default_factory=dict)
    normalize: bool = True
    l2_final: bool = True
    pca_dim: Optional[int] = None

    def __post_init__(self) -> None:
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес дескриптора '{name}' должен быть >= 0, получено {w}"
                )
        if self.pca_dim is not None and self.pca_dim < 1:
            raise ValueError(
                f"pca_dim должен быть >= 1, получено {self.pca_dim}"
            )

    def weight_for(self, name: str) -> float:
        """Вернуть вес дескриптора (по умолчанию 1.0)."""
        return float(self.weights.get(name, 1.0))


# ─── DescriptorSet ────────────────────────────────────────────────────────────

@dataclass
class DescriptorSet:
    """Набор дескрипторов одного фрагмента.

    Атрибуты:
        fragment_id: ID фрагмента.
        descriptors: Словарь {имя: вектор numpy}.
    """

    fragment_id: int
    descriptors: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, vec in self.descriptors.items():
            if vec.ndim != 1:
                raise ValueError(
                    f"Дескриптор '{name}' должен быть 1D, получено ndim={vec.ndim}"
                )

    @property
    def names(self) -> List[str]:
        """Список имён дескрипторов."""
        return list(self.descriptors.keys())

    @property
    def total_dim(self) -> int:
        """Суммарная размерность всех дескрипторов."""
        return sum(v.shape[0] for v in self.descriptors.values())

    def has(self, name: str) -> bool:
        """True если дескриптор с данным именем присутствует."""
        return name in self.descriptors

    def get(self, name: str) -> Optional[np.ndarray]:
        """Вернуть дескриптор или None."""
        return self.descriptors.get(name)


# ─── CombineResult ────────────────────────────────────────────────────────────

@dataclass
class CombineResult:
    """Результат комбинирования дескрипторов.

    Атрибуты:
        fragment_id:   ID фрагмента.
        vector:        Итоговый вектор признаков.
        used_names:    Имена дескрипторов, включённых в вектор.
        original_dim:  Размерность до PCA.
    """

    fragment_id: int
    vector: np.ndarray
    used_names: List[str]
    original_dim: int

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(
                f"vector должен быть 1D, получено ndim={self.vector.ndim}"
            )
        if self.original_dim < 0:
            raise ValueError(
                f"original_dim должен быть >= 0, получено {self.original_dim}"
            )

    @property
    def dim(self) -> int:
        """Размерность итогового вектора."""
        return int(self.vector.shape[0])

    @property
    def is_reduced(self) -> bool:
        """True если была применена PCA-редукция."""
        return self.dim < self.original_dim

    @property
    def norm(self) -> float:
        """L2-норма итогового вектора."""
        return float(np.linalg.norm(self.vector))


# ─── _normalize_vec ───────────────────────────────────────────────────────────

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    """L2-нормализация вектора; вернуть нулевой при нулевой норме."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


# ─── _apply_weight ────────────────────────────────────────────────────────────

def _apply_weight(v: np.ndarray, weight: float) -> np.ndarray:
    """Умножить вектор на вес."""
    return v * weight


# ─── combine_descriptors ──────────────────────────────────────────────────────

def combine_descriptors(
    desc_set: DescriptorSet,
    cfg: Optional[CombineConfig] = None,
) -> CombineResult:
    """Объединить все дескрипторы из DescriptorSet в один вектор.

    Аргументы:
        desc_set: Набор дескрипторов фрагмента.
        cfg:      Параметры комбинирования.

    Возвращает:
        CombineResult с объединённым вектором.

    Исключения:
        ValueError: если descriptors пуст.
    """
    if cfg is None:
        cfg = CombineConfig()

    if not desc_set.descriptors:
        raise ValueError("DescriptorSet не должен быть пустым")

    parts: List[np.ndarray] = []
    used: List[str] = []

    for name, vec in desc_set.descriptors.items():
        v = vec.astype(float)
        if cfg.normalize:
            v = _normalize_vec(v)
        w = cfg.weight_for(name)
        parts.append(_apply_weight(v, w))
        used.append(name)

    combined = np.concatenate(parts)
    original_dim = int(combined.shape[0])

    if cfg.l2_final:
        combined = _normalize_vec(combined)

    return CombineResult(
        fragment_id=desc_set.fragment_id,
        vector=combined,
        used_names=used,
        original_dim=original_dim,
    )


# ─── combine_selected ─────────────────────────────────────────────────────────

def combine_selected(
    desc_set: DescriptorSet,
    names: List[str],
    cfg: Optional[CombineConfig] = None,
) -> CombineResult:
    """Объединить только выбранные дескрипторы.

    Аргументы:
        desc_set: Набор дескрипторов.
        names:    Имена дескрипторов для включения.
        cfg:      Параметры.

    Возвращает:
        CombineResult.

    Исключения:
        ValueError: если names пуст или ни один из дескрипторов не найден.
    """
    if not names:
        raise ValueError("Список names не должен быть пустым")

    selected = {n: v for n, v in desc_set.descriptors.items() if n in names}
    if not selected:
        raise ValueError(
            f"Ни один из дескрипторов {names} не найден в DescriptorSet"
        )

    sub_set = DescriptorSet(
        fragment_id=desc_set.fragment_id,
        descriptors=selected,
    )
    return combine_descriptors(sub_set, cfg)


# ─── batch_combine ────────────────────────────────────────────────────────────

def batch_combine(
    desc_sets: List[DescriptorSet],
    cfg: Optional[CombineConfig] = None,
) -> List[CombineResult]:
    """Объединить дескрипторы для списка фрагментов.

    Аргументы:
        desc_sets: Список DescriptorSet.
        cfg:       Параметры.

    Возвращает:
        Список CombineResult в том же порядке.
    """
    if cfg is None:
        cfg = CombineConfig()
    return [combine_descriptors(ds, cfg) for ds in desc_sets]


# ─── descriptor_distance ──────────────────────────────────────────────────────

def descriptor_distance(
    r1: CombineResult,
    r2: CombineResult,
    metric: str = "cosine",
) -> float:
    """Расстояние между двумя объединёнными дескрипторами.

    Аргументы:
        r1, r2: Результаты комбинирования.
        metric: "cosine", "euclidean" или "l1".

    Возвращает:
        Расстояние >= 0.

    Исключения:
        ValueError: при неизвестном metric.
    """
    v1, v2 = r1.vector.astype(float), r2.vector.astype(float)

    # Выровнять размерности (добить нулями)
    n = max(len(v1), len(v2))
    if len(v1) < n:
        v1 = np.concatenate([v1, np.zeros(n - len(v1))])
    if len(v2) < n:
        v2 = np.concatenate([v2, np.zeros(n - len(v2))])

    if metric == "cosine":
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 1.0
        return float(1.0 - np.dot(v1, v2) / (n1 * n2))
    if metric == "euclidean":
        return float(np.linalg.norm(v1 - v2))
    if metric == "l1":
        return float(np.sum(np.abs(v1 - v2)))

    raise ValueError(
        f"Неизвестный metric '{metric}'. Допустимые: 'cosine', 'euclidean', 'l1'"
    )


# ─── build_distance_matrix ────────────────────────────────────────────────────

def build_distance_matrix(
    results: List[CombineResult],
    metric: str = "cosine",
) -> np.ndarray:
    """Построить симметричную матрицу расстояний.

    Аргументы:
        results: Список CombineResult.
        metric:  Метрика расстояния.

    Возвращает:
        numpy-массив (N, N) float32.
    """
    n = len(results)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = descriptor_distance(results[i], results[j], metric)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


# ─── find_nearest ─────────────────────────────────────────────────────────────

def find_nearest(
    query: CombineResult,
    candidates: List[CombineResult],
    top_k: int = 5,
    metric: str = "cosine",
) -> List[Tuple[int, float]]:
    """Найти top_k ближайших кандидатов к query.

    Аргументы:
        query:      Запросный дескриптор.
        candidates: Список кандидатов.
        top_k:      Число ближайших соседей (>= 1).
        metric:     Метрика расстояния.

    Возвращает:
        Список (fragment_id, distance), отсортированный по расстоянию.

    Исключения:
        ValueError: если top_k < 1 или candidates пуст.
    """
    if top_k < 1:
        raise ValueError(f"top_k должен быть >= 1, получено {top_k}")
    if not candidates:
        raise ValueError("candidates не должен быть пустым")

    dists = [(c.fragment_id, descriptor_distance(query, c, metric))
             for c in candidates]
    dists.sort(key=lambda x: x[1])
    return dists[:top_k]
