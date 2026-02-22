"""
Жадный алгоритм начальной сборки документа.

Стратегия: на каждом шаге берём самое уверенное совпадение краёв
из ещё несобранных фрагментов и добавляем соответствующий фрагмент
в сборку с оптимальной позицией и ориентацией.
"""
import numpy as np
from typing import List, Dict, Tuple
from ..models import Fragment, CompatEntry, Assembly


def greedy_assembly(fragments: List[Fragment],
                    entries: List[CompatEntry]) -> Assembly:
    """
    Жадная сборка по матрице совместимости.

    Args:
        fragments: Все фрагменты.
        entries:   Список CompatEntry, отсортированный по убыванию score.

    Returns:
        Assembly с начальными позициями.
    """
    placements: Dict[int, Tuple[np.ndarray, float]] = {}
    placed_ids = set()

    # Словарь: edge_id → (fragment_id, edge_obj)
    edge_to_frag = {}
    for frag in fragments:
        for edge in frag.edges:
            edge_to_frag[edge.edge_id] = frag

    if not fragments:
        return Assembly(fragments=fragments, placements={}, compat_matrix=np.array([]))

    # Начинаем с произвольного фрагмента — помещаем его в (0, 0)
    first = fragments[0]
    placements[first.fragment_id] = (np.array([0.0, 0.0]), 0.0)
    placed_ids.add(first.fragment_id)

    # Жадно добавляем фрагменты
    for entry in entries:
        frag_i = edge_to_frag.get(entry.edge_i.edge_id)
        frag_j = edge_to_frag.get(entry.edge_j.edge_id)

        if frag_i is None or frag_j is None:
            continue

        i_placed = frag_i.fragment_id in placed_ids
        j_placed = frag_j.fragment_id in placed_ids

        if i_placed and j_placed:
            continue  # Оба уже размещены
        if not i_placed and not j_placed:
            continue  # Ни один не привязан — пропустим пока

        # Один размещён, другой — нет
        anchor_frag = frag_i if i_placed else frag_j
        new_frag    = frag_j if i_placed else frag_i
        anchor_edge = entry.edge_i if i_placed else entry.edge_j
        new_edge    = entry.edge_j if i_placed else entry.edge_i

        # Вычисляем позицию нового фрагмента относительно якоря
        pos, angle = _compute_placement(anchor_frag, anchor_edge,
                                        new_frag, new_edge,
                                        placements[anchor_frag.fragment_id])
        placements[new_frag.fragment_id] = (pos, angle)
        placed_ids.add(new_frag.fragment_id)

        if len(placed_ids) == len(fragments):
            break

    # Фрагменты без привязки — размещаем произвольно
    _place_orphans(fragments, placements, placed_ids)

    total_score = sum(e.score for e in entries
                      if edge_to_frag.get(e.edge_i.edge_id) and
                         edge_to_frag.get(e.edge_j.edge_id) and
                         edge_to_frag[e.edge_i.edge_id].fragment_id in placed_ids and
                         edge_to_frag[e.edge_j.edge_id].fragment_id in placed_ids)

    return Assembly(
        fragments=fragments,
        placements=placements,
        compat_matrix=np.array([]),  # Заполняется снаружи
        total_score=float(total_score),
    )


def _compute_placement(anchor_frag: Fragment,
                        anchor_edge,
                        new_frag: Fragment,
                        new_edge,
                        anchor_placement: Tuple[np.ndarray, float]
                        ) -> Tuple[np.ndarray, float]:
    """
    Вычисляет (position, rotation) нового фрагмента так, чтобы
    new_edge совместился с anchor_edge.

    Упрощённая модель: используем длину и центр края для оценки позиции.
    """
    anchor_pos, anchor_angle = anchor_placement

    # Центр якорного края в мировых координатах
    anchor_curve_world = _transform_curve(anchor_edge.virtual_curve,
                                          anchor_pos, anchor_angle)
    anchor_mid = anchor_curve_world.mean(axis=0)

    # Угол поворота нового фрагмента: зеркально к якорному краю
    # Для простоты берём разворот на 180° + небольшую коррекцию
    new_angle = anchor_angle + np.pi

    # Позиция: центр нового края совпадает с центром якорного
    new_curve_local = new_edge.virtual_curve
    new_mid_local   = new_curve_local.mean(axis=0)
    c, s = np.cos(new_angle), np.sin(new_angle)
    R = np.array([[c, -s], [s, c]])
    new_mid_world = R @ new_mid_local

    new_pos = anchor_mid - new_mid_world
    return new_pos, float(new_angle)


def _transform_curve(curve: np.ndarray,
                     pos: np.ndarray,
                     angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R @ curve.T).T + pos


def _place_orphans(fragments: List[Fragment],
                   placements: Dict,
                   placed_ids: set) -> None:
    """Размещает непривязанные фрагменты в строку ниже собранного документа."""
    orphans = [f for f in fragments if f.fragment_id not in placed_ids]
    if not placements:
        y_offset = 0.0
    else:
        max_y = max(pos[1] for pos, _ in placements.values())
        y_offset = max_y + 200.0

    for k, frag in enumerate(orphans):
        placements[frag.fragment_id] = (np.array([k * 150.0, y_offset]), 0.0)
        placed_ids.add(frag.fragment_id)
