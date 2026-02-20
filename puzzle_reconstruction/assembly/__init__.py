"""
Алгоритмы сборки документа из фрагментов.

Доступные методы:
    greedy_assembly       — Жадная эвристика, O(N²), мгновенно
    simulated_annealing   — Имитация отжига, улучшает жадную сборку
    beam_search           — Beam search с шириной луча W
    gamma_optimizer       — Гамма-распределение (статья 2026, лучший SOTA)
    genetic_assembly      — Генетический алгоритм (OX crossover, элитизм)
    exhaustive_assembly   — Точный Branch & Bound (N ≤ 8)
    ant_colony_assembly   — Муравьиный алгоритм (феромонная матрица + эвристика)
    mcts_assembly         — Monte Carlo Tree Search (UCB1 + случайные роллауты)

Выбор метода:
    - ≤8 фрагментов:    exhaustive (точный, Branch & Bound)
    - 6-15 фрагментов:  beam, sa или mcts
    - 15–30 фрагментов: genetic, ant_colony, mcts или gamma
    - 30+ фрагментов:   gamma или sa с большим числом итераций
"""
from .greedy import greedy_assembly
from .annealing import simulated_annealing
from .beam_search import beam_search
from .gamma_optimizer import gamma_optimizer, GammaEdgeModel
from .exhaustive import exhaustive_assembly
from .genetic import genetic_assembly
from .ant_colony import ant_colony_assembly
from .mcts import mcts_assembly, MCTSNode

__all__ = [
    "greedy_assembly",
    "simulated_annealing",
    "beam_search",
    "gamma_optimizer",
    "GammaEdgeModel",
    "exhaustive_assembly",
    "genetic_assembly",
    "ant_colony_assembly",
    "mcts_assembly",
    "MCTSNode",
]
