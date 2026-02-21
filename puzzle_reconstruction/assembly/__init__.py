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
    run_all_methods       — параллельный запуск нескольких методов с выбором лучшего

Выбор метода:
    - ≤8 фрагментов:    exhaustive (точный, Branch & Bound)
    - 6-15 фрагментов:  beam, sa или mcts
    - 15–30 фрагментов: genetic, ant_colony, mcts или gamma
    - 30+ фрагментов:   gamma или sa с большим числом итераций

Вспомогательные модули:
    assembly_state — состояние сборки (PlacedFragment, AssemblyState,
                     create_state, place_fragment, remove_fragment,
                     add_adjacency, get_neighbors, compute_coverage,
                     is_complete, to_dict, from_dict)
    placement_optimizer — оптимизация порядка размещения (PlacementResult,
                          score_placement, find_best_next, greedy_place,
                          remove_worst_placed, iterative_place)
    cost_matrix         — матрицы стоимостей для алгоритмов сборки (CostMatrix,
                          build_from_scores, build_from_distances, build_combined,
                          apply_forbidden_mask, normalize_costs,
                          to_assignment_matrix, top_k_candidates)
    score_tracker       — отслеживание эволюции оценок (ScoreSnapshot, ScoreTracker,
                          create_tracker, record_snapshot, detect_convergence,
                          extract_best_iteration, summarize_tracker, smooth_scores)
"""
from .greedy import greedy_assembly
from .annealing import simulated_annealing
from .beam_search import beam_search
from .gamma_optimizer import gamma_optimizer, GammaEdgeModel
from .exhaustive import exhaustive_assembly
from .genetic import genetic_assembly
from .ant_colony import ant_colony_assembly
from .mcts import mcts_assembly, MCTSNode
from .assembly_state import (
    PlacedFragment,
    AssemblyState,
    create_state,
    place_fragment,
    remove_fragment,
    add_adjacency,
    get_neighbors,
    compute_coverage,
    is_complete,
    to_dict,
    from_dict,
)
from .parallel import (
    run_all_methods,
    run_selected,
    pick_best,
    pick_best_k,
    summary_table,
    MethodResult,
    AssemblyRacer,
    ALL_METHODS,
    DEFAULT_METHODS,
)
from .placement_optimizer import (
    PlacementResult,
    score_placement,
    find_best_next,
    greedy_place,
    remove_worst_placed,
    iterative_place,
)
from .cost_matrix import (
    CostMatrix,
    build_from_scores,
    build_from_distances,
    build_combined,
    apply_forbidden_mask,
    normalize_costs,
    to_assignment_matrix,
    top_k_candidates,
)
from .score_tracker import (
    ScoreSnapshot,
    ScoreTracker,
    create_tracker,
    record_snapshot,
    detect_convergence,
    extract_best_iteration,
    summarize_tracker,
    smooth_scores,
)

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
    "run_all_methods",
    "run_selected",
    "pick_best",
    "pick_best_k",
    "summary_table",
    "MethodResult",
    "AssemblyRacer",
    "ALL_METHODS",
    "DEFAULT_METHODS",
    # Состояние сборки
    "PlacedFragment",
    "AssemblyState",
    "create_state",
    "place_fragment",
    "remove_fragment",
    "add_adjacency",
    "get_neighbors",
    "compute_coverage",
    "is_complete",
    "to_dict",
    "from_dict",
    # Оптимизация размещения
    "PlacementResult",
    "score_placement",
    "find_best_next",
    "greedy_place",
    "remove_worst_placed",
    "iterative_place",
    # Матрицы стоимостей
    "CostMatrix",
    "build_from_scores",
    "build_from_distances",
    "build_combined",
    "apply_forbidden_mask",
    "normalize_costs",
    "to_assignment_matrix",
    "top_k_candidates",
    # Трекер оценок
    "ScoreSnapshot",
    "ScoreTracker",
    "create_tracker",
    "record_snapshot",
    "detect_convergence",
    "extract_best_iteration",
    "summarize_tracker",
    "smooth_scores",
]
