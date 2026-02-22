"""
Имитация отжига (Simulated Annealing) для глобальной оптимизации сборки.

Улучшает начальное решение, полученное жадным алгоритмом, путём случайных
перестановок с вероятностным принятием ухудшений (метрополисовский критерий).
"""
import numpy as np
import random
from typing import List, Dict, Tuple
from ..models import Fragment, Assembly, CompatEntry


def simulated_annealing(assembly: Assembly,
                        entries: List[CompatEntry],
                        T_max: float = 1000.0,
                        T_min: float = 0.1,
                        cooling: float = 0.995,
                        max_iter: int = 10_000,
                        seed: int = 42) -> Assembly:
    """
    Улучшает Assembly методом имитации отжига.

    На каждом шаге случайно:
    - Переставляем два фрагмента местами, или
    - Поворачиваем один фрагмент на 90°/180°/270°

    Args:
        assembly:   Начальная сборка (от greedy_assembly).
        entries:    Отсортированный список CompatEntry.
        T_max:      Начальная «температура».
        T_min:      Конечная температура (остановка).
        cooling:    Коэффициент охлаждения (< 1).
        max_iter:   Максимум итераций.
        seed:       Случайный seed.

    Returns:
        Улучшенная Assembly.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    fragments = assembly.fragments
    if len(fragments) < 2:
        return assembly

    # Строим lookup: edge → fragment
    edge_to_frag: Dict[int, int] = {}  # edge_id → fragment_id
    for frag in fragments:
        for edge in frag.edges:
            edge_to_frag[edge.edge_id] = frag.fragment_id

    # Текущая конфигурация
    placements = {fid: list(p) for fid, p in assembly.placements.items()}
    current_score = _evaluate(placements, entries, edge_to_frag, fragments)
    best_score    = current_score
    best_placement = {fid: (np.array(p[0]), float(p[1]))
                      for fid, p in placements.items()}

    T = T_max
    frag_ids = [f.fragment_id for f in fragments]

    for it in range(max_iter):
        if T < T_min:
            break

        # Случайный ход
        move = rng.choice(["swap", "rotate", "shift"])
        new_placements = {fid: list(v) for fid, v in placements.items()}

        if move == "swap" and len(frag_ids) >= 2:
            a, b = rng.sample(frag_ids, 2)
            new_placements[a], new_placements[b] = new_placements[b], new_placements[a]

        elif move == "rotate":
            fid = rng.choice(frag_ids)
            delta = rng.choice([np.pi / 2, np.pi, 3 * np.pi / 2])
            pos, angle = new_placements[fid]
            new_placements[fid] = [pos, angle + delta]

        else:  # shift
            fid = rng.choice(frag_ids)
            pos, angle = new_placements[fid]
            dx = rng.gauss(0, 20)
            dy = rng.gauss(0, 20)
            new_placements[fid] = [pos + np.array([dx, dy]), angle]

        new_score = _evaluate(new_placements, entries, edge_to_frag, fragments)
        dE = new_score - current_score

        # Метрополисовский критерий принятия
        if dE > 0 or rng.random() < np.exp(dE / T):
            placements = new_placements
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_placement = {fid: (np.array(p[0]), float(p[1]))
                                  for fid, p in placements.items()}

        T *= cooling

    return Assembly(
        fragments=fragments,
        placements=best_placement,
        compat_matrix=assembly.compat_matrix,
        total_score=best_score,
        ocr_score=assembly.ocr_score,
    )


def _evaluate(placements: Dict,
              entries: List[CompatEntry],
              edge_to_frag: Dict[int, int],
              fragments: List[Fragment]) -> float:
    """
    Быстрая оценка качества конфигурации.
    Суммирует оценки совместимости для всех смежных краёв
    с учётом их реального пространственного расстояния.
    """
    frag_pos: Dict[int, np.ndarray] = {}
    for fid, (pos, angle) in placements.items():
        frag_pos[fid] = np.array(pos) if not isinstance(pos, np.ndarray) else pos

    total = 0.0
    count = 0

    for entry in entries[:500]:  # Ограничиваем для скорости
        fid_i = edge_to_frag.get(entry.edge_i.edge_id)
        fid_j = edge_to_frag.get(entry.edge_j.edge_id)
        if fid_i is None or fid_j is None:
            continue
        pos_i = frag_pos.get(fid_i)
        pos_j = frag_pos.get(fid_j)
        if pos_i is None or pos_j is None:
            continue

        # Учитываем расстояние между фрагментами: ближе = лучше
        dist = float(np.linalg.norm(pos_i - pos_j))
        proximity = np.exp(-dist / 500.0)
        total += entry.score * proximity
        count += 1

    return total / (count + 1e-10)
