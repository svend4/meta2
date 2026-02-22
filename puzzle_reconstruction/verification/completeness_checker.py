"""
Проверка полноты реконструкции документа из фрагментов.

Оценивает, насколько полно размещены фрагменты и охвачена площадь документа.

Экспортирует:
    CompletenessReport   — отчёт о полноте реконструкции
    check_fragment_coverage — доля размещённых фрагментов
    find_missing_fragments  — список не размещённых фрагментов
    check_spatial_coverage  — пиксельное покрытие целевой области
    find_uncovered_regions  — маска непокрытых областей
    completeness_score      — взвешенная оценка полноты
    generate_completeness_report — сформировать полный отчёт
    batch_check_coverage    — пакетная проверка нескольких наборов размещений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class CompletenessReport:
    """Отчёт о полноте реконструкции документа.

    Attributes:
        fragment_coverage:  Доля размещённых фрагментов ∈ [0, 1].
        spatial_coverage:   Доля охваченных пикселей ∈ [0, 1].
        total_score:        Итоговая взвешенная оценка ∈ [0, 1].
        n_placed:           Количество размещённых фрагментов.
        n_total:            Общее количество фрагментов.
        missing_ids:        Список идентификаторов не размещённых фрагментов.
        params:             Параметры расчёта.
    """
    fragment_coverage: float
    spatial_coverage: float
    total_score: float
    n_placed: int = 0
    n_total: int = 0
    missing_ids: List[int] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [
            ("fragment_coverage", self.fragment_coverage),
            ("spatial_coverage", self.spatial_coverage),
            ("total_score", self.total_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def is_complete(self, threshold: float = 1.0) -> bool:
        """Проверить, достигнута ли полнота выше порога."""
        return self.total_score >= threshold

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompletenessReport(score={self.total_score:.3f}, "
            f"placed={self.n_placed}/{self.n_total}, "
            f"spatial={self.spatial_coverage:.3f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def check_fragment_coverage(
    placed_ids: List[int],
    all_ids: List[int],
) -> float:
    """Вычислить долю размещённых фрагментов.

    Args:
        placed_ids: Список идентификаторов размещённых фрагментов.
        all_ids:    Полный список идентификаторов.

    Returns:
        Доля ∈ [0, 1]; при пустом all_ids → 1.0.

    Raises:
        ValueError: Если placed_ids содержит ID, отсутствующие в all_ids.
    """
    all_set: Set[int] = set(all_ids)
    placed_set: Set[int] = set(placed_ids)
    extra = placed_set - all_set
    if extra:
        raise ValueError(
            f"placed_ids contains IDs not in all_ids: {sorted(extra)}"
        )
    if not all_ids:
        return 1.0
    return float(len(placed_set & all_set)) / float(len(all_set))


def find_missing_fragments(
    placed_ids: List[int],
    all_ids: List[int],
) -> List[int]:
    """Найти не размещённые фрагменты.

    Args:
        placed_ids: Список идентификаторов размещённых фрагментов.
        all_ids:    Полный список идентификаторов.

    Returns:
        Сортированный список не размещённых ID.
    """
    placed_set = set(placed_ids)
    return sorted(i for i in all_ids if i not in placed_set)


def check_spatial_coverage(
    masks: List[np.ndarray],
    target_shape: Tuple[int, int],
) -> float:
    """Вычислить долю покрытых пикселей целевой области.

    Args:
        masks:        Список бинарных масок uint8 (H, W), по одной на фрагмент.
                      Ненулевые пиксели считаются покрытыми.
        target_shape: Форма целевого изображения (H, W).

    Returns:
        Доля покрытых пикселей ∈ [0, 1].

    Raises:
        ValueError: Если target_shape содержит нулевое или отрицательное
                    измерение.
    """
    th, tw = target_shape
    if th <= 0 or tw <= 0:
        raise ValueError(
            f"target_shape must have positive dimensions, got {target_shape}"
        )
    if not masks:
        return 0.0

    covered = np.zeros((th, tw), dtype=np.uint8)
    for mask in masks:
        h = min(mask.shape[0], th)
        w = min(mask.shape[1], tw)
        covered[:h, :w] = np.maximum(covered[:h, :w], (mask[:h, :w] > 0).astype(np.uint8))

    total_pixels = th * tw
    return float(np.sum(covered > 0)) / float(total_pixels)


def find_uncovered_regions(
    masks: List[np.ndarray],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Вернуть бинарную маску непокрытых областей.

    Args:
        masks:        Список масок uint8 (H, W).
        target_shape: Целевой размер (H, W).

    Returns:
        Маска uint8 (H, W): 255 — непокрытый пиксель, 0 — покрытый.

    Raises:
        ValueError: Если target_shape содержит нулевое измерение.
    """
    th, tw = target_shape
    if th <= 0 or tw <= 0:
        raise ValueError(
            f"target_shape must have positive dimensions, got {target_shape}"
        )
    covered = np.zeros((th, tw), dtype=np.uint8)
    for mask in masks:
        h = min(mask.shape[0], th)
        w = min(mask.shape[1], tw)
        covered[:h, :w] = np.maximum(covered[:h, :w],
                                     (mask[:h, :w] > 0).astype(np.uint8))
    uncovered = (covered == 0).astype(np.uint8) * 255
    return uncovered


def completeness_score(
    n_placed: int,
    n_total: int,
    pixel_coverage: float = 1.0,
    w_count: float = 0.5,
    w_pixel: float = 0.5,
) -> float:
    """Вычислить взвешенную оценку полноты реконструкции.

    Args:
        n_placed:       Количество размещённых фрагментов (≥ 0).
        n_total:        Общее количество фрагментов (> 0).
        pixel_coverage: Пиксельное покрытие ∈ [0, 1].
        w_count:        Вес счётчика фрагментов (≥ 0).
        w_pixel:        Вес пиксельного покрытия (≥ 0); w_count + w_pixel > 0.

    Returns:
        Итоговая оценка ∈ [0, 1].

    Raises:
        ValueError: Если параметры некорректны.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")
    if n_placed < 0:
        raise ValueError(f"n_placed must be >= 0, got {n_placed}")
    if n_placed > n_total:
        raise ValueError(
            f"n_placed ({n_placed}) must be <= n_total ({n_total})"
        )
    if not (0.0 <= pixel_coverage <= 1.0):
        raise ValueError(
            f"pixel_coverage must be in [0, 1], got {pixel_coverage}"
        )
    if w_count < 0:
        raise ValueError(f"w_count must be >= 0, got {w_count}")
    if w_pixel < 0:
        raise ValueError(f"w_pixel must be >= 0, got {w_pixel}")
    total_w = w_count + w_pixel
    if total_w <= 0:
        raise ValueError("w_count + w_pixel must be > 0")

    count_score = float(n_placed) / float(n_total)
    score = (w_count * count_score + w_pixel * pixel_coverage) / total_w
    return float(np.clip(score, 0.0, 1.0))


def generate_completeness_report(
    placed_ids: List[int],
    all_ids: List[int],
    masks: Optional[List[np.ndarray]] = None,
    target_shape: Tuple[int, int] = (0, 0),
    w_count: float = 0.5,
    w_pixel: float = 0.5,
) -> CompletenessReport:
    """Сформировать полный отчёт о полноте реконструкции.

    Args:
        placed_ids:   Список идентификаторов размещённых фрагментов.
        all_ids:      Полный список идентификаторов.
        masks:        Список масок фрагментов (опционально).
        target_shape: Целевой размер документа (H, W); игнорируется если 0.
        w_count:      Вес счётчика фрагментов.
        w_pixel:      Вес пиксельного покрытия.

    Returns:
        :class:`CompletenessReport`.
    """
    frag_cov = check_fragment_coverage(placed_ids, all_ids)
    missing = find_missing_fragments(placed_ids, all_ids)

    if masks and target_shape[0] > 0 and target_shape[1] > 0:
        spatial_cov = check_spatial_coverage(masks, target_shape)
    else:
        spatial_cov = frag_cov  # fallback

    n_placed = len(set(placed_ids) & set(all_ids))
    n_total = len(set(all_ids)) if all_ids else 0

    if n_total > 0:
        score = completeness_score(
            n_placed, n_total, spatial_cov, w_count, w_pixel
        )
    else:
        score = 1.0

    return CompletenessReport(
        fragment_coverage=frag_cov,
        spatial_coverage=float(np.clip(spatial_cov, 0.0, 1.0)),
        total_score=float(np.clip(score, 0.0, 1.0)),
        n_placed=n_placed,
        n_total=n_total,
        missing_ids=missing,
        params={"w_count": w_count, "w_pixel": w_pixel},
    )


def batch_check_coverage(
    placed_sets: List[List[int]],
    all_ids: List[int],
) -> List[float]:
    """Пакетно проверить покрытие для нескольких наборов размещений.

    Args:
        placed_sets: Список наборов размещённых ID.
        all_ids:     Полный список ID.

    Returns:
        Список долей покрытия ∈ [0, 1] (по одному на набор).
    """
    return [check_fragment_coverage(placed, all_ids) for placed in placed_sets]
