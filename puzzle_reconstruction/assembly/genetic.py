"""
Генетический алгоритм для сборки разорванного документа.

Источник: метод популяционной оптимизации, адаптированный для задачи
восстановления (Wu et al., 2021; Lo et al., 2023).

Ключевые операции:
    - Особь (individual): перестановка fragment_id-ов + вектор углов поворота
    - Фитнес: суммарный score всех смежных стыков
    - Выбор: турнирная селекция (размер турнира K)
    - Скрещивание: Order Crossover (OX) для перестановок
    - Мутация: swap двух позиций или случайный поворот
    - Элитизм: top-E особей переходят без изменений

Преимущества перед SA:
    - Исследует несколько областей пространства параллельно
    - Crossover объединяет хорошие «паттерны» из разных решений
    - Меньше шансов застрять в локальном минимуме

Ограничения:
    - Медленнее SA при малом N (накладные расходы)
    - Оптимально при N ≥ 10
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import Fragment, Assembly, CompatEntry


# ─── Тип «особь» ─────────────────────────────────────────────────────────

Individual = Tuple[np.ndarray, np.ndarray]
# individual = (order: int[N], angles: float[N])
# order[i] = fragment_id на позиции i в раскладке
# angles[i] = угол поворота (0.0, π/2, π, 3π/2) для i-го fragment_id


def genetic_assembly(fragments: List[Fragment],
                      entries: List[CompatEntry],
                      population_size: int = 50,
                      n_generations: int = 200,
                      elite_size: int = 5,
                      tournament_k: int = 3,
                      mutation_rate: float = 0.15,
                      crossover_rate: float = 0.85,
                      allow_rotation: bool = True,
                      seed: int = 42) -> Assembly:
    """
    Оптимизирует расстановку фрагментов с помощью генетического алгоритма.

    Args:
        fragments:       Список фрагментов.
        entries:         Отсортированный список CompatEntry.
        population_size: Число особей в популяции.
        n_generations:   Число поколений.
        elite_size:      Число лучших особей, переходящих без изменений.
        tournament_k:    Размер турнира при выборе родителей.
        mutation_rate:   Вероятность мутации для каждой особи.
        crossover_rate:  Вероятность скрещивания (иначе клон).
        allow_rotation:  Разрешать ли повороты на 90°/180°/270°.
        seed:            Random seed.

    Returns:
        Assembly с лучшей найденной конфигурацией.
    """
    if not fragments:
        return Assembly(fragments=[], placements={}, compat_matrix=np.array([]))

    rng      = np.random.RandomState(seed)
    frag_ids = np.array([f.fragment_id for f in fragments], dtype=int)
    n        = len(frag_ids)
    rotations = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])

    # ── Индекс score ─────────────────────────────────────────────────────
    edge_to_frag: Dict[int, int] = {
        e.edge_id: f.fragment_id
        for f in fragments for e in f.edges
    }
    score_map = _build_score_map(entries, edge_to_frag)

    # ── Инициализация популяции ───────────────────────────────────────────
    # Первая особь — жадное решение
    from .greedy import greedy_assembly
    greedy = greedy_assembly(fragments, entries)
    greedy_order = frag_ids.copy()  # Порядок по умолчанию
    greedy_angles = np.array([
        greedy.placements.get(fid, (None, 0.0))[1]
        for fid in frag_ids
    ], dtype=float)

    population = [_make_individual(frag_ids, rotations, rng)
                   for _ in range(population_size - 1)]
    population.insert(0, (greedy_order.copy(), greedy_angles.copy()))

    # ── Основной цикл ─────────────────────────────────────────────────────
    best_ind    = population[0]
    best_score  = _fitness(best_ind, score_map, frag_ids)

    for gen in range(n_generations):
        # Оцениваем популяцию
        scores = np.array([_fitness(ind, score_map, frag_ids)
                            for ind in population])

        # Обновляем лучшую особь
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_ind   = (population[best_idx][0].copy(),
                           population[best_idx][1].copy())

        # Следующее поколение
        next_gen: List[Individual] = []

        # Элитизм: сохраняем топ-K без изменений
        elite_idxs = np.argsort(scores)[-elite_size:]
        for i in elite_idxs:
            next_gen.append((population[i][0].copy(), population[i][1].copy()))

        # Скрещивание + мутация
        while len(next_gen) < population_size:
            p1 = _tournament_select(population, scores, tournament_k, rng)
            if rng.rand() < crossover_rate:
                p2       = _tournament_select(population, scores, tournament_k, rng)
                child    = _order_crossover(p1, p2, rng)
            else:
                child = (p1[0].copy(), p1[1].copy())

            if rng.rand() < mutation_rate:
                child = _mutate(child, rotations, rng,
                                 allow_rotation=allow_rotation)
            next_gen.append(child)

        population = next_gen

    # ── Формируем Assembly из лучшей особи ────────────────────────────────
    return _individual_to_assembly(best_ind, fragments, score_map, frag_ids,
                                    best_score)


# ─── Генетические операторы ───────────────────────────────────────────────

def _make_individual(frag_ids: np.ndarray,
                      rotations: np.ndarray,
                      rng: np.random.RandomState) -> Individual:
    """Создаёт случайную особь (случайная перестановка + случайные углы)."""
    order  = rng.permutation(frag_ids)
    angles = rng.choice(rotations, size=len(frag_ids))
    return (order, angles)


def _fitness(individual: Individual,
              score_map: Dict[Tuple[int, int], float],
              frag_ids: np.ndarray) -> float:
    """
    Оценивает особь: сумма score всех смежных пар.

    Смежными считаются соседи в линейном порядке (order[i] рядом с order[i+1]).
    """
    order, _ = individual
    total = 0.0
    for i in range(len(order) - 1):
        fi, fj = int(order[i]), int(order[i + 1])
        total += score_map.get((min(fi, fj), max(fi, fj)), 0.0)
    return total


def _tournament_select(population: List[Individual],
                        scores: np.ndarray,
                        k: int,
                        rng: np.random.RandomState) -> Individual:
    """Турнирная селекция: выбирает лучшего из k случайных особей."""
    idxs    = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best_i  = idxs[np.argmax(scores[idxs])]
    return population[best_i]


def _order_crossover(p1: Individual, p2: Individual,
                      rng: np.random.RandomState) -> Individual:
    """
    Order Crossover (OX) для перестановок.

    Алгоритм OX:
        1. Выбираем случайный отрезок [a, b] из p1.
        2. Копируем genes из p1[a:b] в child[a:b].
        3. Остаток заполняем из p2 в порядке, пропуская уже добавленные.

    Углы поворота берём из p1 (для copied) и из p2 (для остальных).
    """
    order1, angles1 = p1
    order2, angles2 = p2
    n = len(order1)

    if n <= 1:
        return (order1.copy(), angles1.copy())

    a, b = sorted(rng.randint(0, n, size=2))
    if a == b:
        b = min(b + 1, n)

    child_order  = np.full(n, -1, dtype=int)
    child_angles = np.zeros(n, dtype=float)

    # Копируем сегмент из p1
    child_order[a:b]  = order1[a:b]
    child_angles[a:b] = angles1[a:b]

    # Заполняем остаток из p2
    in_child = set(child_order[a:b])
    fill_pos = list(range(b, n)) + list(range(0, a))
    fill_src = [i for i in range(n) if order2[i] not in in_child]

    for pos, src_idx in zip(fill_pos, fill_src):
        child_order[pos]  = order2[src_idx]
        child_angles[pos] = angles2[src_idx]

    return (child_order, child_angles)


def _mutate(individual: Individual,
             rotations: np.ndarray,
             rng: np.random.RandomState,
             allow_rotation: bool = True) -> Individual:
    """
    Мутация: случайно swap двух позиций или поворот одного фрагмента.
    """
    order, angles = individual[0].copy(), individual[1].copy()
    n = len(order)

    if n < 2:
        return (order, angles)

    move = rng.randint(0, 3)

    if move == 0 and n >= 2:
        # Swap двух позиций в перестановке
        i, j = rng.choice(n, size=2, replace=False)
        order[i], order[j] = order[j], order[i]
        angles[i], angles[j] = angles[j], angles[i]

    elif move == 1 and allow_rotation:
        # Случайный поворот одного фрагмента
        i = rng.randint(0, n)
        angles[i] = rng.choice(rotations)

    else:
        # Обращение подпоследовательности (reverse segment)
        i, j = sorted(rng.choice(n, size=2, replace=False))
        order[i:j+1]  = order[i:j+1][::-1]
        angles[i:j+1] = angles[i:j+1][::-1]

    return (order, angles)


# ─── Вспомогательные функции ──────────────────────────────────────────────

def _build_score_map(entries: List[CompatEntry],
                      edge_to_frag: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    """Строит словарь {(min_fid, max_fid): best_score} из списка entries."""
    result: Dict[Tuple[int, int], float] = {}
    for e in entries:
        fi = edge_to_frag.get(e.edge_i.edge_id)
        fj = edge_to_frag.get(e.edge_j.edge_id)
        if fi is None or fj is None or fi == fj:
            continue
        key = (min(fi, fj), max(fi, fj))
        result[key] = max(result.get(key, 0.0), e.score)
    return result


def _individual_to_assembly(individual: Individual,
                              fragments: List[Fragment],
                              score_map: Dict,
                              frag_ids: np.ndarray,
                              total_score: float) -> Assembly:
    """Конвертирует особь в объект Assembly."""
    order, angles = individual
    spacing = 120.0
    n_cols  = max(1, math.ceil(math.sqrt(len(order))))

    placements: Dict[int, Tuple[np.ndarray, float]] = {}
    for idx, fid in enumerate(order):
        col = idx % n_cols
        row = idx // n_cols
        pos = np.array([col * spacing, row * spacing])
        placements[int(fid)] = (pos, float(angles[idx]))

    return Assembly(
        fragments=fragments,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=total_score,
    )
