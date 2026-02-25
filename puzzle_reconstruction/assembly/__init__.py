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
    fragment_scorer     — оценка качества размещения фрагментов (ScoreConfig,
                          FragmentScore, AssemblyScore, score_fragment,
                          score_assembly, top_k_placed, bottom_k_placed,
                          batch_score)
    score_tracker       — отслеживание эволюции оценок (ScoreSnapshot, ScoreTracker,
                          create_tracker, record_snapshot, detect_convergence,
                          extract_best_iteration, summarize_tracker, smooth_scores)
    layout_builder      — построение 2D-компоновки (LayoutCell, AssemblyLayout,
                          create_layout, add_cell, remove_cell,
                          compute_bounding_box, snap_to_grid, render_layout_image,
                          layout_to_dict, dict_to_layout)
    fragment_sequencer  — определение порядка фрагментов (SequenceResult,
                          sequence_greedy, sequence_by_score,
                          compute_sequence_score, reverse_sequence,
                          rotate_sequence, sequence_to_pairs,
                          find_best_start, batch_sequence)
    candidate_filter    — фильтрация кандидатов сборки (Candidate, FilterResult,
                          filter_by_threshold, filter_top_k, filter_by_rank,
                          deduplicate_candidates, normalize_scores,
                          merge_candidate_lists, batch_filter)
    fragment_arranger   — расстановка фрагментов на холсте (ArrangementParams,
                          FragmentPlacement, arrange_grid, arrange_strip,
                          center_placements, group_bbox, shift_placements,
                          arrange, batch_arrange)
    collision_detector  — обнаружение коллизий AABB (PlacedRect, CollisionInfo,
                          aabb_overlap, compute_overlap, detect_collisions,
                          collision_graph, is_collision_free, total_overlap_area,
                          resolve_greedy, batch_detect)
    gap_analyzer        — анализ зазоров между фрагментами (FragmentBounds, GapInfo,
                          GapStats, compute_gap, find_adjacent, analyze_all_gaps,
                          gap_histogram, classify_gaps, summarize, batch_analyze)
    fragment_sorter     — сортировка фрагментов для порядка сборки (SortConfig,
                          FragmentSortInfo, SortedFragment, sort_by_id, sort_by_area,
                          sort_by_score, sort_random, sort_fragments,
                          assign_positions, reorder_by_positions, batch_sort)
    canvas_builder      — построение финального холста (CanvasConfig,
                          CanvasFragmentPlacement, CanvasResult, compute_canvas_size,
                          make_empty_canvas, place_fragment_on_canvas, build_canvas,
                          crop_to_content, batch_build_canvases)
    position_estimator  — оценка позиций фрагментов (PositionConfig, FragmentPosition,
                          PositionEstimate, snap_to_grid, estimate_grid_positions,
                          refine_positions, generate_position_candidates,
                          batch_estimate_positions)
    fragment_mapper     — маппинг фрагментов к зонам (MapConfig, FragmentZone,
                          MapResult, compute_zone_grid, assign_to_zone,
                          build_fragment_map, remap_fragments, score_mapping,
                          batch_build_fragment_maps)
    sequence_planner    — планирование порядка размещения (PlanConfig,
                          PlacementStep, PlacementPlan, build_placement_plan,
                          reorder_plan, filter_plan, export_plan, batch_build_plans)
    layout_refiner      — итеративное уточнение компоновки (RefineConfig,
                          FragmentPosition, RefineStep, RefineResult,
                          compute_layout_score, refine_layout, apply_offset,
                          compare_layouts)
    overlap_resolver    — разрешение перекрытий (ResolveConfig, BBox, Overlap,
                          ResolveResult, compute_overlap, detect_overlaps,
                          resolve_overlaps, compute_total_overlap, overlap_ratio)
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
from .fragment_scorer import (
    ScoreConfig,
    FragmentScore,
    AssemblyScore,
    score_fragment,
    score_assembly,
    top_k_placed,
    bottom_k_placed,
    batch_score as batch_score_assemblies,
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
from .layout_builder import (
    LayoutCell,
    AssemblyLayout,
    create_layout,
    add_cell,
    remove_cell,
    compute_bounding_box,
    snap_to_grid,
    render_layout_image,
    layout_to_dict,
    dict_to_layout,
)
from .fragment_sequencer import (
    SequenceResult,
    sequence_greedy,
    sequence_by_score,
    compute_sequence_score,
    reverse_sequence,
    rotate_sequence,
    sequence_to_pairs,
    find_best_start,
    batch_sequence,
)
from .candidate_filter import (
    Candidate,
    FilterResult,
    filter_by_threshold,
    filter_top_k,
    filter_by_rank,
    deduplicate_candidates,
    normalize_scores,
    merge_candidate_lists,
    batch_filter,
)
from .fragment_arranger import (
    ArrangementParams,
    FragmentPlacement,
    arrange_grid,
    arrange_strip,
    center_placements,
    group_bbox,
    shift_placements,
    arrange,
    batch_arrange,
)
from .collision_detector import (
    PlacedRect,
    CollisionInfo,
    aabb_overlap,
    compute_overlap,
    detect_collisions,
    collision_graph,
    is_collision_free,
    total_overlap_area,
    resolve_greedy,
    batch_detect,
)
from .gap_analyzer import (
    FragmentBounds,
    GapInfo,
    GapStats,
    compute_gap,
    find_adjacent,
    analyze_all_gaps,
    gap_histogram,
    classify_gaps,
    summarize,
    batch_analyze,
)
from .fragment_sorter import (
    SortConfig,
    FragmentSortInfo,
    SortedFragment,
    sort_by_id,
    sort_by_area,
    sort_by_score,
    sort_random,
    sort_fragments,
    assign_positions,
    reorder_by_positions,
    batch_sort,
)
from .canvas_builder import (
    CanvasConfig,
    FragmentPlacement as CanvasFragmentPlacement,
    CanvasResult,
    compute_canvas_size,
    make_empty_canvas,
    place_fragment as place_fragment_on_canvas,
    build_canvas,
    crop_to_content,
    batch_build_canvases,
)
from .position_estimator import (
    PositionConfig,
    FragmentPosition,
    PositionEstimate,
    snap_to_grid as snap_coords_to_grid,
    estimate_grid_positions,
    refine_positions,
    generate_position_candidates,
    batch_estimate_positions,
)
from .fragment_mapper import (
    MapConfig,
    FragmentZone,
    MapResult,
    compute_zone_grid,
    assign_to_zone,
    build_fragment_map,
    remap_fragments,
    score_mapping,
    batch_build_fragment_maps,
)
from .sequence_planner import (
    PlanConfig,
    PlacementStep,
    PlacementPlan,
    build_placement_plan,
    reorder_plan,
    filter_plan,
    export_plan,
    batch_build_plans,
)
from .layout_refiner import (
    RefineConfig,
    FragmentPosition,
    RefineStep,
    RefineResult,
    compute_layout_score,
    refine_layout,
    apply_offset,
    compare_layouts,
)
from .overlap_resolver import (
    ResolveConfig,
    BBox,
    Overlap,
    ResolveResult,
    compute_overlap,
    detect_overlaps,
    resolve_overlaps,
    compute_total_overlap,
    overlap_ratio,
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
    # Построение 2D-компоновки
    "LayoutCell",
    "AssemblyLayout",
    "create_layout",
    "add_cell",
    "remove_cell",
    "compute_bounding_box",
    "snap_to_grid",
    "render_layout_image",
    "layout_to_dict",
    "dict_to_layout",
    # Определение порядка фрагментов
    "SequenceResult",
    "sequence_greedy",
    "sequence_by_score",
    "compute_sequence_score",
    "reverse_sequence",
    "rotate_sequence",
    "sequence_to_pairs",
    "find_best_start",
    "batch_sequence",
    # Фильтрация кандидатов сборки
    "Candidate",
    "FilterResult",
    "filter_by_threshold",
    "filter_top_k",
    "filter_by_rank",
    "deduplicate_candidates",
    "normalize_scores",
    "merge_candidate_lists",
    "batch_filter",
    # Расстановка фрагментов на холсте
    "ArrangementParams",
    "FragmentPlacement",
    "arrange_grid",
    "arrange_strip",
    "center_placements",
    "group_bbox",
    "shift_placements",
    "arrange",
    "batch_arrange",
    # Обнаружение коллизий
    "PlacedRect",
    "CollisionInfo",
    "aabb_overlap",
    "compute_overlap",
    "detect_collisions",
    "collision_graph",
    "is_collision_free",
    "total_overlap_area",
    "resolve_greedy",
    "batch_detect",
    # Анализ зазоров между фрагментами
    "FragmentBounds",
    "GapInfo",
    "GapStats",
    "compute_gap",
    "find_adjacent",
    "analyze_all_gaps",
    "gap_histogram",
    "classify_gaps",
    "summarize",
    "batch_analyze",
    # Сортировка фрагментов
    "SortConfig",
    "FragmentSortInfo",
    "SortedFragment",
    "sort_by_id",
    "sort_by_area",
    "sort_by_score",
    "sort_random",
    "sort_fragments",
    "assign_positions",
    "reorder_by_positions",
    "batch_sort",
    # Построение финального холста
    "CanvasConfig",
    "CanvasFragmentPlacement",
    "CanvasResult",
    "compute_canvas_size",
    "make_empty_canvas",
    "place_fragment_on_canvas",
    "build_canvas",
    "crop_to_content",
    "batch_build_canvases",
    # Оценка позиций фрагментов
    "PositionConfig",
    "FragmentPosition",
    "PositionEstimate",
    "snap_coords_to_grid",
    "estimate_grid_positions",
    "refine_positions",
    "generate_position_candidates",
    "batch_estimate_positions",
    # Маппинг фрагментов к зонам
    "MapConfig",
    "FragmentZone",
    "MapResult",
    "compute_zone_grid",
    "assign_to_zone",
    "build_fragment_map",
    "remap_fragments",
    "score_mapping",
    "batch_build_fragment_maps",
    # Планирование последовательности размещения
    "PlanConfig",
    "PlacementStep",
    "PlacementPlan",
    "build_placement_plan",
    "reorder_plan",
    "filter_plan",
    "export_plan",
    "batch_build_plans",
    # Итеративное уточнение компоновки
    "RefineConfig",
    "FragmentPosition",
    "RefineStep",
    "RefineResult",
    "compute_layout_score",
    "refine_layout",
    "apply_offset",
    "compare_layouts",
    # Разрешение перекрытий
    "ResolveConfig",
    "BBox",
    "Overlap",
    "ResolveResult",
    "compute_overlap",
    "detect_overlaps",
    "resolve_overlaps",
    "compute_total_overlap",
    "overlap_ratio",
    # Оценка качества размещения фрагментов
    "ScoreConfig",
    "FragmentScore",
    "AssemblyScore",
    "score_fragment",
    "score_assembly",
    "top_k_placed",
    "bottom_k_placed",
    "batch_score_assemblies",
    # Bridge #10 — реестр 16 sleeping assembly-модулей
    "build_assembly_registry",
    "list_assembly_fns",
    "get_assembly_fn",
    "get_assembly_category",
    "ASSEMBLY_CATEGORIES",
    "ASSEMBLY_REGISTRY",
]

# Bridge #10 exports
from .bridge import (  # noqa: E402
    build_assembly_registry,
    list_assembly_fns,
    get_assembly_fn,
    get_assembly_category,
    ASSEMBLY_CATEGORIES,
    ASSEMBLY_REGISTRY,
)
