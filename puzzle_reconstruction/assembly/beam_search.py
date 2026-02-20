"""
Beam Search для глобальной сборки документа.

Beam Search исследует дерево частичных сборок, удерживая на каждом шаге
только B лучших кандидатов («лучей»). Точнее жадного алгоритма,
но дешевле полного перебора.

Параметры:
    beam_width (B): Число удерживаемых гипотез. B=1 → жадный, B=∞ → полный перебор.
    depth: Сколько фрагментов добавляем за один шаг.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
from dataclasses import dataclass, field

from ..models import Fragment, Assembly, CompatEntry, EdgeSignature


@dataclass
class Hypothesis:
    """Частичная сборка — один «луч» в Beam Search."""
    placements:    Dict[int, Tuple[np.ndarray, float]]  # frag_id → (pos, angle)
    placed_ids:    set
    score:         float = 0.0
    last_entries:  List[CompatEntry] = field(default_factory=list)


def beam_search(fragments: List[Fragment],
                entries: List[CompatEntry],
                beam_width: int = 10,
                max_depth: Optional[int] = None) -> Assembly:
    """
    Сборка методом Beam Search.

    Args:
        fragments:   Все фрагменты с заполненными edges.
        entries:     CompatEntry, отсортированные по убыванию score.
        beam_width:  Ширина луча (B).
        max_depth:   Максимальная глубина (None = len(fragments)).

    Returns:
        Assembly с наилучшей найденной конфигурацией.
    """
    if not fragments:
        return Assembly(fragments=[], placements={}, compat_matrix=np.array([]))

    if max_depth is None:
        max_depth = len(fragments)

    # Индекс: edge_id → Fragment
    edge_to_frag: Dict[int, Fragment] = {}
    for frag in fragments:
        for edge in frag.edges:
            edge_to_frag[edge.edge_id] = frag

    # Начальный луч: первый фрагмент в начале координат
    init_hyp = Hypothesis(
        placements={fragments[0].fragment_id: (np.array([0.0, 0.0]), 0.0)},
        placed_ids={fragments[0].fragment_id},
        score=0.0,
    )
    beam: List[Hypothesis] = [init_hyp]

    for depth in range(1, max_depth):
        if all(len(h.placed_ids) == len(fragments) for h in beam):
            break

        candidates: List[Hypothesis] = []

        for hyp in beam:
            # Ищем лучшие расширения этой гипотезы
            expansions = _expand(hyp, fragments, entries, edge_to_frag,
                                 n_expand=beam_width * 2)
            candidates.extend(expansions)

        if not candidates:
            break

        # Оставляем B лучших
        candidates.sort(key=lambda h: h.score, reverse=True)
        beam = candidates[:beam_width]

    # Лучшая гипотеза
    best = max(beam, key=lambda h: h.score)

    # Добавляем непривязанные фрагменты
    _fill_orphans(fragments, best)

    return Assembly(
        fragments=fragments,
        placements=best.placements,
        compat_matrix=np.array([]),
        total_score=best.score,
    )


def _expand(hyp: Hypothesis,
            fragments: List[Fragment],
            entries: List[CompatEntry],
            edge_to_frag: Dict[int, Fragment],
            n_expand: int) -> List[Hypothesis]:
    """
    Расширяет гипотезу: добавляет один новый фрагмент по лучшим совпадениям.
    Возвращает список новых гипотез.
    """
    expansions = []
    seen_new_frags = set()

    for entry in entries:
        if len(expansions) >= n_expand:
            break

        fi = edge_to_frag.get(entry.edge_i.edge_id)
        fj = edge_to_frag.get(entry.edge_j.edge_id)
        if fi is None or fj is None:
            continue

        i_placed = fi.fragment_id in hyp.placed_ids
        j_placed = fj.fragment_id in hyp.placed_ids

        if i_placed == j_placed:
            continue  # Либо оба, либо ни одного

        anchor_frag = fi if i_placed else fj
        new_frag    = fj if i_placed else fi
        anchor_edge = entry.edge_i if i_placed else entry.edge_j
        new_edge    = entry.edge_j if i_placed else entry.edge_i

        if new_frag.fragment_id in seen_new_frags:
            continue
        seen_new_frags.add(new_frag.fragment_id)

        # Вычисляем позицию нового фрагмента
        pos, angle = _compute_placement(
            anchor_frag, anchor_edge, new_frag, new_edge,
            hyp.placements[anchor_frag.fragment_id]
        )

        # Новая гипотеза
        new_placements = dict(hyp.placements)
        new_placements[new_frag.fragment_id] = (pos, angle)
        new_placed = set(hyp.placed_ids) | {new_frag.fragment_id}

        # Пересчитываем score: накопленный + новый стык
        new_score = hyp.score + entry.score

        expansions.append(Hypothesis(
            placements=new_placements,
            placed_ids=new_placed,
            score=new_score,
            last_entries=hyp.last_entries + [entry],
        ))

    return expansions


def _compute_placement(anchor_frag: Fragment,
                        anchor_edge: EdgeSignature,
                        new_frag: Fragment,
                        new_edge: EdgeSignature,
                        anchor_placement: Tuple) -> Tuple[np.ndarray, float]:
    """Вычисляет позицию нового фрагмента относительно якоря."""
    anchor_pos, anchor_angle = anchor_placement
    anchor_pos = np.asarray(anchor_pos)

    # Центр якорного края в мировых координатах
    c, s = np.cos(anchor_angle), np.sin(anchor_angle)
    R = np.array([[c, -s], [s, c]])
    anchor_mid = R @ anchor_edge.virtual_curve.mean(axis=0) + anchor_pos

    # Новый фрагмент: разворот на 180° + выравнивание центров
    new_angle = anchor_angle + np.pi
    c2, s2 = np.cos(new_angle), np.sin(new_angle)
    R2 = np.array([[c2, -s2], [s2, c2]])
    new_mid_world = R2 @ new_edge.virtual_curve.mean(axis=0)
    new_pos = anchor_mid - new_mid_world

    return new_pos, float(new_angle)


def _fill_orphans(fragments: List[Fragment], hyp: Hypothesis) -> None:
    """Добавляет непривязанные фрагменты в строку ниже."""
    orphans = [f for f in fragments if f.fragment_id not in hyp.placed_ids]
    if not hyp.placements:
        y_base = 0.0
    else:
        y_base = max(pos[1] for pos, _ in hyp.placements.values()) + 200.0

    for k, frag in enumerate(orphans):
        hyp.placements[frag.fragment_id] = (np.array([k * 200.0, y_base]), 0.0)
        hyp.placed_ids.add(frag.fragment_id)
