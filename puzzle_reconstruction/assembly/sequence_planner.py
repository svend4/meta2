"""Планирование последовательности размещения фрагментов.

Модуль предоставляет структуры и функции для построения порядка
размещения фрагментов в процессе сборки пазла: от «якорного» фрагмента
к последующим, с учётом оценок совместимости.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── PlanConfig ───────────────────────────────────────────────────────────────

@dataclass
class PlanConfig:
    """Параметры планировщика последовательности.

    Атрибуты:
        strategy:       Стратегия выбора следующего фрагмента:
                        "greedy" (максимальный счёт), "bfs" (в ширину).
        anchor_id:      ID якорного фрагмента (None = первый в списке).
        min_score:      Минимальный допустимый счёт совместимости (>= 0).
        allow_revisit:  Разрешить повторное добавление фрагмента.
    """

    strategy: str = "greedy"
    anchor_id: Optional[int] = None
    min_score: float = 0.0
    allow_revisit: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in ("greedy", "bfs"):
            raise ValueError(
                f"strategy должен быть 'greedy' или 'bfs', "
                f"получено '{self.strategy}'"
            )
        if self.min_score < 0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )


# ─── PlacementStep ────────────────────────────────────────────────────────────

@dataclass
class PlacementStep:
    """Один шаг последовательности размещения.

    Атрибуты:
        step:        Порядковый номер шага (>= 0).
        fragment_id: Размещаемый фрагмент (>= 0).
        score:       Счёт совместимости с уже размещёнными (>= 0).
        anchored_by: Список ID фрагментов, с которыми связан текущий.
    """

    step: int
    fragment_id: int
    score: float
    anchored_by: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(
                f"step должен быть >= 0, получено {self.step}"
            )
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.score < 0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )

    @property
    def is_anchor(self) -> bool:
        """True если это первый шаг (якорный фрагмент)."""
        return self.step == 0


# ─── PlacementPlan ────────────────────────────────────────────────────────────

@dataclass
class PlacementPlan:
    """Итоговая последовательность размещения.

    Атрибуты:
        steps:       Упорядоченный список PlacementStep.
        n_fragments: Число фрагментов для размещения (>= 0).
        n_placed:    Число фактически включённых в план (>= 0).
        strategy:    Использованная стратегия.
    """

    steps: List[PlacementStep]
    n_fragments: int
    n_placed: int
    strategy: str

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments должен быть >= 0, получено {self.n_fragments}"
            )
        if self.n_placed < 0:
            raise ValueError(
                f"n_placed должен быть >= 0, получено {self.n_placed}"
            )

    @property
    def placement_order(self) -> List[int]:
        """Список fragment_id в порядке размещения."""
        return [s.fragment_id for s in self.steps]

    @property
    def coverage(self) -> float:
        """Доля размещённых фрагментов (0 если n_fragments == 0)."""
        if self.n_fragments == 0:
            return 0.0
        return float(self.n_placed) / float(self.n_fragments)

    @property
    def mean_score(self) -> float:
        """Средний счёт совместимости (без якорного шага)."""
        non_anchor = [s.score for s in self.steps if not s.is_anchor]
        if not non_anchor:
            return 0.0
        return sum(non_anchor) / len(non_anchor)


# ─── _best_candidate ──────────────────────────────────────────────────────────

def _best_candidate(
    remaining: Set[int],
    placed: Set[int],
    scores: Dict[Tuple[int, int], float],
    min_score: float,
) -> Tuple[Optional[int], float, List[int]]:
    """Найти лучший кандидат среди непомещённых фрагментов."""
    best_id: Optional[int] = None
    best_sc = -1.0
    best_anchors: List[int] = []

    for fid in remaining:
        sc = 0.0
        anchors: List[int] = []
        for pid in placed:
            key = (min(fid, pid), max(fid, pid))
            val = scores.get(key, 0.0)
            if val > 0:
                anchors.append(pid)
                sc += val
        if anchors:
            sc /= len(anchors)
        if sc >= min_score and sc > best_sc:
            best_sc = sc
            best_id = fid
            best_anchors = anchors

    return best_id, max(best_sc, 0.0), best_anchors


# ─── _next_bfs ────────────────────────────────────────────────────────────────

def _next_bfs(
    remaining: Set[int],
    placed: Set[int],
    scores: Dict[Tuple[int, int], float],
    min_score: float,
) -> Tuple[Optional[int], float, List[int]]:
    """BFS: взять первый доступный кандидат с допустимым счётом."""
    for fid in sorted(remaining):
        sc = 0.0
        anchors: List[int] = []
        for pid in placed:
            key = (min(fid, pid), max(fid, pid))
            val = scores.get(key, 0.0)
            if val > 0:
                anchors.append(pid)
                sc += val
        if anchors:
            sc /= len(anchors)
        if sc >= min_score:
            return fid, sc, anchors
    # Нет связанных — взять просто первый
    if remaining:
        fid = min(remaining)
        return fid, 0.0, []
    return None, 0.0, []


# ─── build_placement_plan ─────────────────────────────────────────────────────

def build_placement_plan(
    fragment_ids: List[int],
    scores: Dict[Tuple[int, int], float],
    cfg: Optional[PlanConfig] = None,
) -> PlacementPlan:
    """Построить порядок размещения фрагментов.

    Аргументы:
        fragment_ids: Список идентификаторов фрагментов.
        scores:       Словарь {(id_a, id_b): оценка} с id_a <= id_b.
        cfg:          Параметры.

    Возвращает:
        PlacementPlan.

    Исключения:
        ValueError: Если fragment_ids пуст.
    """
    if not fragment_ids:
        return PlacementPlan(steps=[], n_fragments=0, n_placed=0,
                             strategy=(cfg or PlanConfig()).strategy)

    if cfg is None:
        cfg = PlanConfig()

    ids = list(fragment_ids)
    remaining: Set[int] = set(ids)
    placed: Set[int] = set()
    steps: List[PlacementStep] = []

    # Якорный фрагмент
    if cfg.anchor_id is not None and cfg.anchor_id in remaining:
        anchor = cfg.anchor_id
    else:
        anchor = ids[0]

    remaining.discard(anchor)
    placed.add(anchor)
    steps.append(PlacementStep(step=0, fragment_id=anchor, score=0.0))

    selector = _best_candidate if cfg.strategy == "greedy" else _next_bfs

    while remaining:
        fid, sc, anchors = selector(remaining, placed, scores, cfg.min_score)
        if fid is None:
            break
        if not cfg.allow_revisit:
            remaining.discard(fid)
        placed.add(fid)
        steps.append(PlacementStep(
            step=len(steps),
            fragment_id=fid,
            score=sc,
            anchored_by=anchors,
        ))

    return PlacementPlan(
        steps=steps,
        n_fragments=len(ids),
        n_placed=len(steps),
        strategy=cfg.strategy,
    )


# ─── reorder_plan ─────────────────────────────────────────────────────────────

def reorder_plan(
    plan: PlacementPlan,
    priority: List[int],
) -> PlacementPlan:
    """Переупорядочить план, выдвинув выбранные фрагменты в начало.

    Аргументы:
        plan:     Исходный PlacementPlan.
        priority: Список fragment_id, которые должны идти первыми.

    Возвращает:
        Новый PlacementPlan с переставленными шагами.
    """
    order = list(dict.fromkeys(priority))
    remaining_ids = [s.fragment_id for s in plan.steps
                     if s.fragment_id not in set(order)]
    new_order = order + remaining_ids

    step_by_id = {s.fragment_id: s for s in plan.steps}
    new_steps = []
    for i, fid in enumerate(new_order):
        if fid in step_by_id:
            old = step_by_id[fid]
            new_steps.append(PlacementStep(
                step=i,
                fragment_id=old.fragment_id,
                score=old.score,
                anchored_by=old.anchored_by,
            ))

    return PlacementPlan(
        steps=new_steps,
        n_fragments=plan.n_fragments,
        n_placed=len(new_steps),
        strategy=plan.strategy,
    )


# ─── filter_plan ──────────────────────────────────────────────────────────────

def filter_plan(
    plan: PlacementPlan,
    min_score: float = 0.0,
) -> PlacementPlan:
    """Убрать шаги с оценкой ниже порога (кроме якоря).

    Аргументы:
        plan:      Исходный PlacementPlan.
        min_score: Минимальный допустимый счёт (>= 0).

    Возвращает:
        Отфильтрованный PlacementPlan.

    Исключения:
        ValueError: Если min_score < 0.
    """
    if min_score < 0:
        raise ValueError(
            f"min_score должен быть >= 0, получено {min_score}"
        )
    kept = [s for s in plan.steps if s.is_anchor or s.score >= min_score]
    reindexed = [
        PlacementStep(step=i, fragment_id=s.fragment_id,
                      score=s.score, anchored_by=s.anchored_by)
        for i, s in enumerate(kept)
    ]
    return PlacementPlan(
        steps=reindexed,
        n_fragments=plan.n_fragments,
        n_placed=len(reindexed),
        strategy=plan.strategy,
    )


# ─── export_plan ──────────────────────────────────────────────────────────────

def export_plan(plan: PlacementPlan) -> List[dict]:
    """Экспортировать план в список словарей.

    Аргументы:
        plan: PlacementPlan.

    Возвращает:
        Список записей вида {"step", "fragment_id", "score", "anchored_by"}.
    """
    return [
        {
            "step": s.step,
            "fragment_id": s.fragment_id,
            "score": s.score,
            "anchored_by": list(s.anchored_by),
        }
        for s in plan.steps
    ]


# ─── batch_build_plans ────────────────────────────────────────────────────────

def batch_build_plans(
    fragment_id_lists: List[List[int]],
    score_dicts: List[Dict[Tuple[int, int], float]],
    cfg: Optional[PlanConfig] = None,
) -> List[PlacementPlan]:
    """Построить планы для нескольких наборов фрагментов.

    Аргументы:
        fragment_id_lists: Список списков ID фрагментов.
        score_dicts:       Список словарей оценок.
        cfg:               Общие параметры.

    Возвращает:
        Список PlacementPlan.

    Исключения:
        ValueError: Если длины списков не совпадают.
    """
    if len(fragment_id_lists) != len(score_dicts):
        raise ValueError(
            "fragment_id_lists и score_dicts должны иметь одинаковую длину"
        )
    return [
        build_placement_plan(ids, sc, cfg)
        for ids, sc in zip(fragment_id_lists, score_dicts)
    ]
