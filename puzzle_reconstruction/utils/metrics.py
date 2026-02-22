"""
Метрики качества восстановления документа из фрагментов.

Вычисляет количественные показатели, позволяющие сравнивать
результат автоматической сборки с эталонным (ground-truth).

Классы:
    ReconstructionMetrics — набор метрик качества одной сборки

Функции:
    placement_iou            — IoU двух прямоугольников фрагментов
    order_kendall_tau        — τ Кенделла для оценки порядка фрагментов
    permutation_distance     — нормированное расстояние Хэмминга двух перестановок
    assembly_precision_recall — точность и полнота правильных пар
    fragment_placement_accuracy — доля правильно размещённых фрагментов
    compute_reconstruction_metrics — итоговый набор всех метрик
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── ReconstructionMetrics ────────────────────────────────────────────────────

@dataclass
class ReconstructionMetrics:
    """
    Набор метрик качества одной сборки пазла.

    Attributes:
        precision:          Доля корректных пар «фрагмент-позиция» среди
                            предсказанных.
        recall:             Доля корректных пар среди эталонных.
        f1:                 Среднее гармоническое precision и recall.
        placement_accuracy: Доля фрагментов, размещённых в пределах
                            допуска от эталонной позиции.
        order_accuracy:     Нормированный τ Кенделла (1 = идеальный порядок).
        permutation_dist:   Нормированное расстояние Хэмминга (0 = идеально).
        n_correct:          Число правильно размещённых фрагментов.
        n_total:            Общее число эталонных фрагментов.
        extra:              Дополнительные числовые метрики.
    """
    precision:          float
    recall:             float
    f1:                 float
    placement_accuracy: float
    order_accuracy:     float
    permutation_dist:   float
    n_correct:          int
    n_total:            int
    extra:              Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ReconstructionMetrics("
                f"f1={self.f1:.3f}, "
                f"placement={self.placement_accuracy:.3f}, "
                f"order={self.order_accuracy:.3f}, "
                f"n={self.n_correct}/{self.n_total})")


# ─── placement_iou ────────────────────────────────────────────────────────────

def placement_iou(box1: Tuple[float, float, float, float],
                   box2: Tuple[float, float, float, float]) -> float:
    """
    IoU (Intersection over Union) двух прямоугольников.

    Args:
        box1, box2: (x, y, w, h) — левый верхний угол, ширина, высота.

    Returns:
        IoU ∈ [0, 1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    if union <= 0.0:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


# ─── order_kendall_tau ────────────────────────────────────────────────────────

def order_kendall_tau(pred_order: Sequence[int],
                       true_order: Sequence[int]) -> float:
    """
    Вычисляет нормированный τ Кенделла для двух порядков фрагментов.

    τ = (concordant - discordant) / (n*(n-1)/2)
    Нормированный результат: (τ + 1) / 2 ∈ [0, 1].

    Args:
        pred_order: Предсказанный порядок ID фрагментов.
        true_order: Эталонный порядок ID фрагментов.

    Returns:
        Нормированный τ ∈ [0, 1]. 1 = идеальное совпадение.
    """
    if len(pred_order) != len(true_order):
        raise ValueError(
            f"Длины порядков не совпадают: {len(pred_order)} vs {len(true_order)}"
        )
    n = len(pred_order)
    if n <= 1:
        return 1.0

    # Маппинг true_order → ранги
    rank_map: Dict[int, int] = {fid: r for r, fid in enumerate(true_order)}
    # Ранги предсказанного порядка в системе эталона
    try:
        pred_ranks = [rank_map[fid] for fid in pred_order]
    except KeyError:
        return 0.5   # неизвестные ID → случайный порядок

    concordant  = 0
    discordant  = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred_ranks[i] < pred_ranks[j]:
                concordant += 1
            elif pred_ranks[i] > pred_ranks[j]:
                discordant += 1

    pairs = n * (n - 1) // 2
    if pairs == 0:
        return 1.0
    tau = (concordant - discordant) / pairs
    return float(np.clip((tau + 1.0) / 2.0, 0.0, 1.0))


# ─── permutation_distance ─────────────────────────────────────────────────────

def permutation_distance(pred: Sequence[int],
                          true: Sequence[int]) -> float:
    """
    Нормированное расстояние Хэмминга между двумя перестановками.

    Расстояние = число позиций, в которых pred[i] ≠ true[i],
    нормированное на длину.

    Args:
        pred: Предсказанная перестановка ID фрагментов.
        true: Эталонная перестановка.

    Returns:
        Нормированное расстояние ∈ [0, 1]. 0 = идеальное совпадение.

    Raises:
        ValueError: Если длины различаются.
    """
    if len(pred) != len(true):
        raise ValueError(
            f"Длины перестановок не совпадают: {len(pred)} vs {len(true)}"
        )
    n = len(pred)
    if n == 0:
        return 0.0
    mismatches = sum(p != t for p, t in zip(pred, true))
    return float(mismatches) / n


# ─── assembly_precision_recall ────────────────────────────────────────────────

def assembly_precision_recall(
        pred_pairs: Sequence[Tuple[int, int]],
        true_pairs: Sequence[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Точность и полнота для пар смежных фрагментов.

    Пара считается «правильной» если (a, b) или (b, a) присутствует
    в эталонном наборе.

    Args:
        pred_pairs: Предсказанные смежные пары [(fid_i, fid_j), ...].
        true_pairs: Эталонные смежные пары.

    Returns:
        (precision, recall) ∈ [0, 1]².
    """
    if not pred_pairs and not true_pairs:
        return 1.0, 1.0

    # Нормализуем: (min, max) для симметрии
    true_set = {(min(a, b), max(a, b)) for a, b in true_pairs}
    pred_set = {(min(a, b), max(a, b)) for a, b in pred_pairs}

    true_positives = len(pred_set & true_set)

    precision = true_positives / max(1, len(pred_set))
    recall    = true_positives / max(1, len(true_set))

    return float(precision), float(recall)


# ─── fragment_placement_accuracy ──────────────────────────────────────────────

def fragment_placement_accuracy(
        pred_positions: Dict[int, Tuple[float, float]],
        true_positions: Dict[int, Tuple[float, float]],
        tolerance:      float = 20.0,
) -> float:
    """
    Доля фрагментов, размещённых в пределах Евклидова допуска.

    Для каждого фрагмента из эталона проверяет, есть ли в предсказании
    позиция в пределах `tolerance` пикселей.

    Args:
        pred_positions: {fid: (x, y)} — предсказанные позиции.
        true_positions: {fid: (x, y)} — эталонные позиции.
        tolerance:      Максимально допустимое расстояние (пикселей).

    Returns:
        Точность размещения ∈ [0, 1].
    """
    if not true_positions:
        return 1.0

    correct = 0
    for fid, (tx, ty) in true_positions.items():
        if fid not in pred_positions:
            continue
        px, py = pred_positions[fid]
        dist = float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))
        if dist <= tolerance:
            correct += 1

    return float(correct) / len(true_positions)


# ─── compute_reconstruction_metrics ──────────────────────────────────────────

def compute_reconstruction_metrics(
        pred_order:      Optional[Sequence[int]]              = None,
        true_order:      Optional[Sequence[int]]              = None,
        pred_pairs:      Optional[Sequence[Tuple[int, int]]]  = None,
        true_pairs:      Optional[Sequence[Tuple[int, int]]]  = None,
        pred_positions:  Optional[Dict[int, Tuple[float, float]]] = None,
        true_positions:  Optional[Dict[int, Tuple[float, float]]] = None,
        placement_tol:   float = 20.0,
) -> ReconstructionMetrics:
    """
    Вычисляет полный набор метрик восстановления.

    Все параметры опциональны: для отсутствующих метрика заполняется
    нейтральным значением (0.5 для τ, 0.0 для остальных).

    Args:
        pred_order:     Предсказанный порядок ID фрагментов.
        true_order:     Эталонный порядок.
        pred_pairs:     Предсказанные смежные пары.
        true_pairs:     Эталонные смежные пары.
        pred_positions: Предсказанные позиции {fid: (x, y)}.
        true_positions: Эталонные позиции {fid: (x, y)}.
        placement_tol:  Допуск размещения (пикселей).

    Returns:
        ReconstructionMetrics.
    """
    # Точность / полнота пар
    if pred_pairs is not None and true_pairs is not None:
        precision, recall = assembly_precision_recall(pred_pairs, true_pairs)
    else:
        precision, recall = 0.0, 0.0

    f1 = (2.0 * precision * recall / max(1e-9, precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Размещение
    if pred_positions is not None and true_positions is not None:
        placement_acc = fragment_placement_accuracy(
            pred_positions, true_positions, tolerance=placement_tol
        )
        n_correct = int(round(placement_acc * len(true_positions)))
        n_total   = len(true_positions)
    else:
        placement_acc = 0.0
        n_correct     = 0
        n_total       = 0

    # Порядок
    if pred_order is not None and true_order is not None:
        tau = order_kendall_tau(pred_order, true_order)
        perm_dist = permutation_distance(pred_order, true_order)
    else:
        tau       = 0.5
        perm_dist = 0.0

    return ReconstructionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        placement_accuracy=placement_acc,
        order_accuracy=tau,
        permutation_dist=perm_dist,
        n_correct=n_correct,
        n_total=n_total,
    )
