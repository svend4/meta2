"""Разрешение перекрытий между размещёнными фрагментами.

Модуль обнаруживает пересечения прямоугольных bounding-box'ов фрагментов
и вычисляет смещения для их устранения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── ResolveConfig ────────────────────────────────────────────────────────────

@dataclass
class ResolveConfig:
    """Параметры разрешения перекрытий.

    Атрибуты:
        max_iter:    Максимальное число итераций (>= 1).
        gap:         Целевой зазор между фрагментами (>= 0).
        step_scale:  Масштаб шага смещения (> 0).
        frozen_ids:  Идентификаторы фиксированных фрагментов.
    """

    max_iter: int = 10
    gap: float = 1.0
    step_scale: float = 0.5
    frozen_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter должен быть >= 1, получено {self.max_iter}"
            )
        if self.gap < 0.0:
            raise ValueError(
                f"gap должен быть >= 0, получено {self.gap}"
            )
        if self.step_scale <= 0.0:
            raise ValueError(
                f"step_scale должен быть > 0, получено {self.step_scale}"
            )


# ─── BBox ─────────────────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Ограничивающий прямоугольник фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента.
        x:           Левая граница.
        y:           Верхняя граница.
        w:           Ширина (> 0).
        h:           Высота (> 0).
    """

    fragment_id: int
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(
                f"w должен быть > 0, получено {self.w}"
            )
        if self.h <= 0:
            raise ValueError(
                f"h должен быть > 0, получено {self.h}"
            )

    @property
    def x2(self) -> float:
        """Правая граница."""
        return self.x + self.w

    @property
    def y2(self) -> float:
        """Нижняя граница."""
        return self.y + self.h

    @property
    def cx(self) -> float:
        """Центр по X."""
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        """Центр по Y."""
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        """Площадь."""
        return self.w * self.h

    def translate(self, dx: float, dy: float) -> "BBox":
        """Вернуть смещённый BBox (не изменяет исходный)."""
        return BBox(fragment_id=self.fragment_id,
                    x=self.x + dx, y=self.y + dy,
                    w=self.w, h=self.h)


# ─── Overlap ──────────────────────────────────────────────────────────────────

@dataclass
class Overlap:
    """Перекрытие двух фрагментов.

    Атрибуты:
        id_a:   ID первого фрагмента.
        id_b:   ID второго фрагмента.
        area:   Площадь перекрытия (>= 0).
        dx:     Компонента смещения по X для устранения.
        dy:     Компонента смещения по Y для устранения.
    """

    id_a: int
    id_b: int
    area: float
    dx: float
    dy: float

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Упорядоченная пара (min, max)."""
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))

    @property
    def has_overlap(self) -> bool:
        """True если площадь > 0."""
        return self.area > 0.0


# ─── ResolveResult ────────────────────────────────────────────────────────────

@dataclass
class ResolveResult:
    """Результат разрешения перекрытий.

    Атрибуты:
        boxes:        Финальный словарь {fragment_id: BBox}.
        n_iter:       Выполнено итераций.
        resolved:     True если перекрытий не осталось.
        history:      Список (iter, n_overlaps) для каждой итерации.
    """

    boxes: Dict[int, BBox]
    n_iter: int
    resolved: bool
    history: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_iter < 0:
            raise ValueError(
                f"n_iter должен быть >= 0, получено {self.n_iter}"
            )

    @property
    def final_n_overlaps(self) -> int:
        """Число перекрытий на последней итерации."""
        if not self.history:
            return 0
        return self.history[-1][1]

    @property
    def fragment_ids(self) -> List[int]:
        """Список ID фрагментов."""
        return list(self.boxes.keys())


# ─── compute_overlap ──────────────────────────────────────────────────────────

def compute_overlap(a: BBox, b: BBox, gap: float = 0.0) -> Overlap:
    """Вычислить перекрытие двух BBox с учётом требуемого зазора.

    Аргументы:
        a:   Первый BBox.
        b:   Второй BBox.
        gap: Требуемый зазор (фрагменты должны быть на расстоянии >= gap).

    Возвращает:
        Overlap (area == 0 если перекрытия нет).
    """
    # Перекрытие с учётом зазора
    ix = min(a.x2 + gap, b.x2 + gap) - max(a.x - gap, b.x - gap)
    iy = min(a.y2 + gap, b.y2 + gap) - max(a.y - gap, b.y - gap)

    if ix <= 0 or iy <= 0:
        return Overlap(id_a=a.fragment_id, id_b=b.fragment_id,
                       area=0.0, dx=0.0, dy=0.0)

    area = float(ix * iy)

    # Направление смещения: от центра A к центру B
    raw_dx = b.cx - a.cx
    raw_dy = b.cy - a.cy
    norm = (raw_dx ** 2 + raw_dy ** 2) ** 0.5
    if norm < 1e-12:
        dx = float(ix / 2.0)
        dy = float(iy / 2.0)
    else:
        # Смещение пропорционально перекрытию вдоль направления
        dx = float(raw_dx / norm * ix / 2.0)
        dy = float(raw_dy / norm * iy / 2.0)

    return Overlap(id_a=a.fragment_id, id_b=b.fragment_id,
                   area=area, dx=dx, dy=dy)


# ─── detect_overlaps ──────────────────────────────────────────────────────────

def detect_overlaps(
    boxes: Dict[int, BBox],
    gap: float = 0.0,
) -> List[Overlap]:
    """Обнаружить все перекрытия в наборе BBox.

    Аргументы:
        boxes: Словарь {fragment_id: BBox}.
        gap:   Требуемый зазор.

    Возвращает:
        Список Overlap (только с area > 0).
    """
    ids = list(boxes.keys())
    result: List[Overlap] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ov = compute_overlap(boxes[ids[i]], boxes[ids[j]], gap)
            if ov.has_overlap:
                result.append(ov)
    return result


# ─── _apply_step ──────────────────────────────────────────────────────────────

def _apply_step(
    boxes: Dict[int, BBox],
    overlaps: List[Overlap],
    cfg: ResolveConfig,
) -> Dict[int, BBox]:
    """Применить один шаг разрешения перекрытий."""
    # Аккумулировать смещения
    shifts: Dict[int, List[Tuple[float, float]]] = {k: [] for k in boxes}

    for ov in overlaps:
        # B отодвигается от A; A остаётся (или тоже двигается в обр. направлении)
        if ov.id_b not in cfg.frozen_ids:
            shifts[ov.id_b].append((ov.dx * cfg.step_scale,
                                    ov.dy * cfg.step_scale))
        if ov.id_a not in cfg.frozen_ids:
            shifts[ov.id_a].append((-ov.dx * cfg.step_scale,
                                    -ov.dy * cfg.step_scale))

    new_boxes: Dict[int, BBox] = {}
    for fid, box in boxes.items():
        if not shifts[fid]:
            new_boxes[fid] = box
        else:
            dx = sum(s[0] for s in shifts[fid]) / len(shifts[fid])
            dy = sum(s[1] for s in shifts[fid]) / len(shifts[fid])
            new_boxes[fid] = box.translate(dx, dy)

    return new_boxes


# ─── resolve_overlaps ─────────────────────────────────────────────────────────

def resolve_overlaps(
    boxes: Dict[int, BBox],
    cfg: Optional[ResolveConfig] = None,
) -> ResolveResult:
    """Итеративно устранить перекрытия между фрагментами.

    Аргументы:
        boxes: Словарь {fragment_id: BBox}.
        cfg:   Параметры.

    Возвращает:
        ResolveResult.
    """
    if cfg is None:
        cfg = ResolveConfig()

    # Глубокая копия через translate(0, 0)
    current: Dict[int, BBox] = {fid: b.translate(0.0, 0.0)
                                 for fid, b in boxes.items()}
    history: List[Tuple[int, int]] = []

    for it in range(cfg.max_iter):
        overlaps = detect_overlaps(current, cfg.gap)
        history.append((it, len(overlaps)))
        if not overlaps:
            return ResolveResult(boxes=current, n_iter=it + 1,
                                 resolved=True, history=history)
        current = _apply_step(current, overlaps, cfg)

    # Финальная проверка
    final_overlaps = detect_overlaps(current, cfg.gap)
    history.append((cfg.max_iter, len(final_overlaps)))
    return ResolveResult(boxes=current, n_iter=cfg.max_iter,
                         resolved=len(final_overlaps) == 0,
                         history=history)


# ─── compute_total_overlap ────────────────────────────────────────────────────

def compute_total_overlap(
    boxes: Dict[int, BBox],
    gap: float = 0.0,
) -> float:
    """Суммарная площадь перекрытий.

    Аргументы:
        boxes: Словарь {fragment_id: BBox}.
        gap:   Требуемый зазор.

    Возвращает:
        Суммарная площадь.
    """
    return sum(ov.area for ov in detect_overlaps(boxes, gap))


# ─── overlap_ratio ────────────────────────────────────────────────────────────

def overlap_ratio(
    boxes: Dict[int, BBox],
    gap: float = 0.0,
) -> float:
    """Доля перекрывающейся площади относительно суммарной площади.

    Аргументы:
        boxes: Словарь {fragment_id: BBox}.
        gap:   Требуемый зазор.

    Возвращает:
        Значение в [0, 1].
    """
    total_area = sum(b.area for b in boxes.values())
    if total_area < 1e-12:
        return 0.0
    return float(min(compute_total_overlap(boxes, gap) / total_area, 1.0))
