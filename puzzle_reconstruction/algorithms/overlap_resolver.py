"""
Разрешение перекрытий между размещёнными фрагментами.

Обнаруживает конфликты (пересечения контуров) и применяет итеративные
корректировки позиций для их устранения, не нарушая уже выстроенную сборку.

Классы:
    OverlapConflict — описание одного конфликта (пара, IoU, вектор сдвига)

Функции:
    compute_separation_vector  — вектор минимального разделения двух контуров
    detect_overlap_conflicts   — нахождение всех пар с IoU > порога
    resolve_single_conflict    — устранение одного конфликта сдвигом
    resolve_all_conflicts      — итеративное устранение всех конфликтов
    conflict_score             — скалярная мера суммарного перекрытия сборки
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..verification.overlap_checker import polygon_iou, check_overlap_pair
from ..assembly.assembly_state import (
    AssemblyState,
    place_fragment,
    PlacedFragment,
)


# ─── OverlapConflict ──────────────────────────────────────────────────────────

@dataclass
class OverlapConflict:
    """
    Описание конфликта перекрытия между двумя фрагментами.

    Attributes:
        idx1:         Индекс первого фрагмента.
        idx2:         Индекс второго фрагмента.
        iou:          Пересечение контуров (IoU) в данный момент.
        shift_vector: (dx, dy) — рекомендованный сдвиг для устранения.
    """
    idx1:         int
    idx2:         int
    iou:          float
    shift_vector: Tuple[float, float] = field(default=(0.0, 0.0))

    def __repr__(self) -> str:
        return (f"OverlapConflict(idx1={self.idx1}, idx2={self.idx2}, "
                f"iou={self.iou:.4f}, shift={self.shift_vector})")


# ─── _get_contour_for_fragment ─────────────────────────────────────────────────

def _get_placed_contour(
    frag:     PlacedFragment,
    contours: List[np.ndarray],
) -> Optional[np.ndarray]:
    """Возвращает контур фрагмента, сдвинутый на его позицию."""
    idx = frag.fragment_idx
    if idx >= len(contours):
        return None
    cnt = contours[idx].reshape(-1, 2).astype(np.float64)
    if cnt.size == 0:
        return None
    dx, dy = frag.position
    return cnt + np.array([[dx, dy]], dtype=np.float64)


# ─── compute_separation_vector ────────────────────────────────────────────────

def compute_separation_vector(
    contour1: np.ndarray,
    contour2: np.ndarray,
) -> Tuple[float, float]:
    """
    Вычисляет вектор минимального разделения двух перекрывающихся контуров.

    Использует вектор из центроида контура2 к центроиду контура1,
    масштабированный пропорционально расстоянию между centroid2 и
    ближайшей точкой контура1.

    Args:
        contour1: (N, 2) float — контур первого фрагмента.
        contour2: (M, 2) float — контур второго фрагмента.

    Returns:
        (dx, dy) — вектор, на который нужно сдвинуть второй фрагмент
        от первого.
    """
    pts1 = contour1.reshape(-1, 2).astype(np.float64)
    pts2 = contour2.reshape(-1, 2).astype(np.float64)

    if pts1.size == 0 or pts2.size == 0:
        return (0.0, 0.0)

    c1 = pts1.mean(axis=0)
    c2 = pts2.mean(axis=0)

    direction = c2 - c1
    dist = np.linalg.norm(direction)
    if dist < 1e-9:
        # Центроиды совпадают — сдвигаем по диагонали
        return (1.0, 1.0)

    # Нормированный вектор + масштаб по bbox
    bb1 = np.ptp(pts1, axis=0)   # [width, height] контура1
    scale = float(np.max(bb1)) * 0.1 if np.max(bb1) > 1e-9 else 1.0

    unit = direction / dist
    return (float(unit[0] * scale), float(unit[1] * scale))


# ─── detect_overlap_conflicts ─────────────────────────────────────────────────

def detect_overlap_conflicts(
    state:     AssemblyState,
    contours:  List[np.ndarray],
    threshold: float = 0.05,
) -> List[OverlapConflict]:
    """
    Находит все пары размещённых фрагментов с IoU > threshold.

    Args:
        state:     Текущее состояние сборки.
        contours:  Исходные контуры (в системе координат фрагмента).
        threshold: Порог IoU для объявления конфликта.

    Returns:
        Список OverlapConflict, отсортированный по убыванию IoU.
    """
    placed_ids = list(state.placed.keys())
    conflicts: List[OverlapConflict] = []

    for a, i in enumerate(placed_ids):
        frag_i = state.placed[i]
        cnt_i  = _get_placed_contour(frag_i, contours)
        if cnt_i is None:
            continue

        for j in placed_ids[a + 1:]:
            frag_j = state.placed[j]
            cnt_j  = _get_placed_contour(frag_j, contours)
            if cnt_j is None:
                continue

            result = check_overlap_pair(cnt_i, cnt_j, iou_thresh=threshold)
            if result.overlaps:
                shift = compute_separation_vector(cnt_i, cnt_j)
                conflicts.append(OverlapConflict(
                    idx1=i, idx2=j,
                    iou=float(result.iou),
                    shift_vector=shift,
                ))

    conflicts.sort(key=lambda c: -c.iou)
    return conflicts


# ─── resolve_single_conflict ──────────────────────────────────────────────────

def resolve_single_conflict(
    state:    AssemblyState,
    conflict: OverlapConflict,
    contours: List[np.ndarray],
    fixed:    Optional[int] = None,
) -> AssemblyState:
    """
    Устраняет один конфликт сдвигом второго фрагмента.

    Если fixed задан — сдвигается тот фрагмент, который не является fixed.
    Иначе сдвигается idx2.

    Args:
        state:    Текущее состояние сборки.
        conflict: Конфликт для устранения.
        contours: Исходные контуры фрагментов.
        fixed:    Индекс «якорного» (неподвижного) фрагмента.

    Returns:
        Новое AssemblyState с исправленной позицией.
    """
    idx1, idx2 = conflict.idx1, conflict.idx2
    dx, dy     = conflict.shift_vector

    # Определяем, какой фрагмент двигать
    if fixed is not None and fixed == idx1:
        move_idx = idx2
        sdx, sdy = dx, dy         # сдвигаем idx2 от idx1
    elif fixed is not None and fixed == idx2:
        move_idx = idx1
        sdx, sdy = -dx, -dy       # сдвигаем idx1 от idx2
    else:
        move_idx = idx2
        sdx, sdy = dx, dy

    if move_idx not in state.placed:
        return state

    frag = state.placed[move_idx]
    old_x, old_y = frag.position
    new_pos = (old_x + sdx, old_y + sdy)

    # Пересоздаём сборку с новой позицией
    new_placed = dict(state.placed)
    old_frag   = new_placed[move_idx]
    new_placed[move_idx] = PlacedFragment(
        fragment_idx=old_frag.fragment_idx,
        position=new_pos,
        rotation=old_frag.rotation,
        score=old_frag.score,
        meta=old_frag.meta,
    )

    from dataclasses import replace as dc_replace
    return dc_replace(state, placed=new_placed)


# ─── resolve_all_conflicts ────────────────────────────────────────────────────

def resolve_all_conflicts(
    state:     AssemblyState,
    contours:  List[np.ndarray],
    max_iter:  int   = 10,
    threshold: float = 0.05,
    fixed:     Optional[int] = None,
) -> AssemblyState:
    """
    Итеративно устраняет все конфликты перекрытия.

    Повторяет detect → resolve до тех пор, пока конфликты не исчезнут
    или не будет исчерпан лимит итераций.

    Args:
        state:     Начальное состояние сборки.
        contours:  Исходные контуры фрагментов.
        max_iter:  Максимальное число итераций.
        threshold: Порог IoU для объявления конфликта.
        fixed:     Индекс якорного фрагмента.

    Returns:
        AssemblyState без (или с минимальным числом) конфликтов.
    """
    current = state

    for _ in range(max_iter):
        conflicts = detect_overlap_conflicts(current, contours, threshold)
        if not conflicts:
            break
        for conflict in conflicts:
            current = resolve_single_conflict(
                current, conflict, contours, fixed=fixed
            )

    return current


# ─── conflict_score ───────────────────────────────────────────────────────────

def conflict_score(
    state:     AssemblyState,
    contours:  List[np.ndarray],
    threshold: float = 0.05,
) -> float:
    """
    Скалярная мера суммарного перекрытия: сумма IoU по всем конфликтным парам.

    0.0 → нет конфликтов. Чем меньше, тем лучше.

    Args:
        state:     Состояние сборки.
        contours:  Исходные контуры.
        threshold: Порог IoU.

    Returns:
        Суммарный IoU по конфликтным парам (float ≥ 0).
    """
    conflicts = detect_overlap_conflicts(state, contours, threshold)
    return float(sum(c.iou for c in conflicts))
