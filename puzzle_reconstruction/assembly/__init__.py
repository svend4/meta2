"""
Алгоритмы сборки документа из фрагментов.

Доступные методы:
    greedy_assembly     — Жадная эвристика, O(N²), мгновенно
    simulated_annealing — Имитация отжига, улучшает жадную сборку
    beam_search         — Beam search с шириной луча W
    gamma_optimizer     — Гамма-распределение (статья 2026, лучший SOTA)

Выбор метода:
    - ≤8 фрагментов:   exhaustive (точный, Branch & Bound)
    - 6-15 фрагментов:  beam или sa
    - 15+ фрагментов:  gamma или sa с большим числом итераций
"""
from .greedy import greedy_assembly
from .annealing import simulated_annealing
from .beam_search import beam_search
from .gamma_optimizer import gamma_optimizer, GammaEdgeModel
from .exhaustive import exhaustive_assembly

__all__ = [
    "greedy_assembly",
    "simulated_annealing",
    "beam_search",
    "gamma_optimizer",
    "GammaEdgeModel",
    "exhaustive_assembly",
]
