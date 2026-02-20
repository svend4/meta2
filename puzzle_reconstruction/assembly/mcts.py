"""
Сборка документа методом Monte Carlo Tree Search (MCTS).

MCTS строит дерево частичных сборок, направляя поиск через UCB1-баланс
«исследование vs. эксплуатация». Каждый лист оценивается случайными
«роллаутами» (случайное завершение сборки), результаты обратно
распространяются вверх по дереву.

Алгоритм:
    1. Selection   — спускаемся по дереву, выбирая узлы по UCB1.
    2. Expansion   — расширяем не полностью исследованный листовой узел.
    3. Simulation  — случайный роллаут до конечного состояния.
    4. Backprop    — обновляем visits и total_score вверх по дереву.

Параметры:
    n_simulations   — число итераций MCTS (≥ 10 рекомендуется).
    exploration_c   — константа UCB1 (1.41 ≈ √2 — классический выбор).
    n_rollouts      — роллаутов на один лист (усредняется).
    seed            — для воспроизводимости.

Функции:
    mcts_assembly   — основная точка входа
    MCTSNode        — узел дерева (состояние частичной сборки)
    _ucb1           — формула UCB1
    _rollout        — случайный роллаут → оценка
    _greedy_score   — жадная оценка последовательности фрагментов
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..models import Assembly, CompatEntry, Fragment


# ─── MCTSNode ─────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """
    Узел дерева MCTS — частичная сборка.

    Attributes:
        order:        Упорядоченный список fragment_id, помещённых до сих пор.
        remaining:    Множество ещё не помещённых fragment_id.
        parent:       Родительский узел (None для корня).
        children:     {next_fid: MCTSNode}.
        visits:       Число раз, когда узел посещался.
        total_score:  Сумма оценок роллаутов через этот узел.
    """
    order:       List[int]
    remaining:   Set[int]
    parent:      Optional["MCTSNode"]           = field(default=None, repr=False)
    children:    Dict[int, "MCTSNode"]          = field(default_factory=dict, repr=False)
    visits:      int                            = 0
    total_score: float                          = 0.0

    @property
    def is_terminal(self) -> bool:
        return not self.remaining

    @property
    def is_fully_expanded(self) -> bool:
        return not self.remaining or len(self.children) == len(self.remaining)

    @property
    def mean_score(self) -> float:
        return self.total_score / max(1, self.visits)

    def ucb1(self, exploration_c: float) -> float:
        """UCB1 формула: mean + C * sqrt(ln(N_parent) / N)."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.mean_score + exploration_c * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

    def best_child(self, exploration_c: float = 0.0) -> Optional["MCTSNode"]:
        """Возвращает лучшего потомка по UCB1 (или greedy если c=0)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.ucb1(exploration_c))


# ─── Основная функция ─────────────────────────────────────────────────────────

def mcts_assembly(fragments:      List[Fragment],
                   entries:        List[CompatEntry],
                   n_simulations:  int   = 100,
                   exploration_c:  float = 1.4142135623730951,  # √2
                   n_rollouts:     int   = 3,
                   seed:           Optional[int] = None) -> Assembly:
    """
    Сборка документа методом Monte Carlo Tree Search.

    Args:
        fragments:     Список Fragment для сборки.
        entries:       Список CompatEntry (оценки совместимости краёв).
        n_simulations: Число итераций MCTS.
        exploration_c: Константа UCB1 (√2 ≈ 1.414 — стандарт).
        n_rollouts:    Число роллаутов для оценки листа.
        seed:          Random seed для воспроизводимости.

    Returns:
        Assembly с лучшей найденной последовательностью.
    """
    rng = np.random.RandomState(seed)

    if not fragments:
        return Assembly(fragments=[], placements={},
                        compat_matrix=np.array([]), total_score=0.0)

    if len(fragments) == 1:
        frag = fragments[0]
        return Assembly(
            fragments=fragments,
            placements={frag.fragment_id: (np.array([0.0, 0.0]), 0.0)},
            compat_matrix=np.array([]),
            total_score=1.0,
        )

    # ── Предварительная обработка entries ─────────────────────────────────
    score_map = _build_score_map(fragments, entries)
    all_fids  = [f.fragment_id for f in fragments]

    # ── Корень дерева ─────────────────────────────────────────────────────
    root = MCTSNode(order=[], remaining=set(all_fids))

    best_order: List[int] = list(all_fids)
    best_score: float     = 0.0

    # ── Основной цикл MCTS ────────────────────────────────────────────────
    for _ in range(n_simulations):
        # 1. Selection
        node = _select(root, exploration_c)

        # 2. Expansion
        if not node.is_terminal:
            node = _expand(node, rng)

        # 3. Simulation (n_rollouts роллаутов)
        rollout_scores = [
            _rollout(node.order, list(node.remaining), score_map, rng)
            for _ in range(n_rollouts)
        ]
        sim_score = float(np.mean(rollout_scores))

        # Обновляем лучший результат
        if node.is_terminal and sim_score > best_score:
            best_score = sim_score
            best_order = list(node.order)
        elif not node.is_terminal and sim_score > best_score:
            remaining_shuffled = list(node.remaining)
            rng.shuffle(remaining_shuffled)
            candidate_order = node.order + remaining_shuffled
            candidate_score = _greedy_score(candidate_order, score_map)
            if candidate_score > best_score:
                best_score = candidate_score
                best_order = candidate_order

        # 4. Backpropagation
        _backpropagate(node, sim_score)

    # ── Построение финальной Assembly ─────────────────────────────────────
    return _order_to_assembly(best_order, fragments, best_score)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _select(root: MCTSNode, exploration_c: float) -> MCTSNode:
    """Спускается по дереву, используя UCB1, до листового/неполного узла."""
    node = root
    while not node.is_terminal and node.is_fully_expanded and node.children:
        child = node.best_child(exploration_c)
        if child is None:
            break
        node = child
    return node


def _expand(node: MCTSNode, rng: np.random.RandomState) -> MCTSNode:
    """Добавляет один новый дочерний узел для непосещённого фрагмента."""
    tried   = set(node.children.keys())
    untried = list(node.remaining - tried)
    if not untried:
        return node

    fid = int(rng.choice(untried))
    child = MCTSNode(
        order=node.order + [fid],
        remaining=node.remaining - {fid},
        parent=node,
    )
    node.children[fid] = child
    return child


def _rollout(order:     List[int],
              remaining: List[int],
              score_map: Dict[Tuple[int, int], float],
              rng:       np.random.RandomState) -> float:
    """
    Случайный роллаут: случайно перемешиваем оставшиеся фрагменты
    и считаем жадную оценку полной последовательности.
    """
    shuffled = list(remaining)
    rng.shuffle(shuffled)
    full_order = order + shuffled
    return _greedy_score(full_order, score_map)


def _greedy_score(order: List[int],
                   score_map: Dict[Tuple[int, int], float]) -> float:
    """
    Оценивает последовательность как сумму попарных оценок соседей.

    score = Σ score_map[(fid_i, fid_{i+1})] / (N-1).
    """
    if len(order) < 2:
        return 1.0

    total = 0.0
    for i in range(len(order) - 1):
        a, b  = order[i], order[i + 1]
        total += score_map.get((a, b), 0.0) + score_map.get((b, a), 0.0)

    return total / (2.0 * (len(order) - 1))


def _backpropagate(node: MCTSNode, score: float) -> None:
    """Распространяет оценку вверх от листа до корня."""
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visits      += 1
        current.total_score += score
        current = current.parent


def _build_score_map(fragments: List[Fragment],
                      entries:   List[CompatEntry]) -> Dict[Tuple[int, int], float]:
    """
    Строит словарь {(fid_i, fid_j): max_score} из CompatEntry.

    Использует только fragment_id, закодированный в edge_id.
    """
    score_map: Dict[Tuple[int, int], float] = {}
    for e in entries:
        fid_i = e.edge_i.edge_id // 10
        fid_j = e.edge_j.edge_id // 10
        key   = (fid_i, fid_j)
        score_map[key] = max(score_map.get(key, 0.0), float(e.score))
    return score_map


def _order_to_assembly(order:     List[int],
                        fragments: List[Fragment],
                        score:     float) -> Assembly:
    """
    Преобразует порядок фрагментов в Assembly с сеточным расположением.

    Фрагменты располагаются горизонтально с отступом 10 пикселей.
    """
    frag_map = {f.fragment_id: f for f in fragments}
    placements: Dict[int, Tuple[np.ndarray, float]] = {}

    x = 0.0
    for fid in order:
        placements[fid] = (np.array([x, 0.0]), 0.0)
        frag = frag_map.get(fid)
        w    = frag.image.shape[1] if frag is not None and frag.image is not None else 100
        x   += w + 10.0

    # Добавляем не вошедшие фрагменты (защита от orphan)
    for frag in fragments:
        if frag.fragment_id not in placements:
            placements[frag.fragment_id] = (np.array([0.0, 0.0]), 0.0)

    return Assembly(
        fragments=fragments,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=score,
    )


def _ucb1(node: MCTSNode, exploration_c: float) -> float:
    """Внешний UCB1 для удобного доступа из тестов."""
    return node.ucb1(exploration_c)
