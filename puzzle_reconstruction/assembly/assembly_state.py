"""
Состояние текущей сборки пазла.

Хранит, какие фрагменты уже размещены, их позиции и трансформации,
граф смежности между соседними фрагментами, а также вспомогательные
метрики (покрытие, полнота).

Классы:
    PlacedFragment — один размещённый фрагмент (позиция, угол, масштаб)
    AssemblyState  — полное состояние сборки

Функции:
    create_state      — создать пустое состояние для N фрагментов
    place_fragment    — добавить фрагмент (возвращает новое состояние)
    remove_fragment   — убрать фрагмент (возвращает новое состояние)
    add_adjacency     — зарегистрировать смежность двух фрагментов
    get_neighbors     — список соседей данного фрагмента
    compute_coverage  — доля размещённых фрагментов ∈ [0,1]
    is_complete       — все ли N фрагментов размещены
    to_dict           — сериализация в словарь
    from_dict         — десериализация из словаря
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── PlacedFragment ───────────────────────────────────────────────────────────

@dataclass
class PlacedFragment:
    """
    Размещённый фрагмент документа.

    Attributes:
        idx:      Индекс фрагмента в исходном наборе.
        position: (x, y) — координаты верхнего-левого угла.
        angle:    Угол поворота в градусах (против часовой стрелки).
        scale:    Коэффициент масштабирования (1.0 = без изменений).
        meta:     Произвольные метаданные (score, side, метод, ...).
    """
    idx:      int
    position: Tuple[float, float]
    angle:    float = 0.0
    scale:    float = 1.0
    meta:     Dict  = field(default_factory=dict)

    def __repr__(self) -> str:
        x, y = self.position
        return (f"PlacedFragment(idx={self.idx}, "
                f"pos=({x:.1f},{y:.1f}), "
                f"angle={self.angle:.1f}°)")


# ─── AssemblyState ────────────────────────────────────────────────────────────

@dataclass
class AssemblyState:
    """
    Полное состояние сборки пазла.

    Attributes:
        n_fragments: Общее число фрагментов в наборе.
        placed:      {idx → PlacedFragment} для размещённых фрагментов.
        adjacency:   Неориентированный граф смежности {idx → set(соседей)}.
        step:        Счётчик шагов сборки.
    """
    n_fragments: int
    placed:      Dict[int, PlacedFragment]       = field(default_factory=dict)
    adjacency:   Dict[int, Set[int]]             = field(default_factory=dict)
    step:        int                              = 0

    def __repr__(self) -> str:
        return (f"AssemblyState(n={self.n_fragments}, "
                f"placed={len(self.placed)}, "
                f"step={self.step})")


# ─── create_state ─────────────────────────────────────────────────────────────

def create_state(n_fragments: int) -> AssemblyState:
    """
    Создаёт пустое состояние сборки.

    Args:
        n_fragments: Общее число фрагментов.

    Returns:
        AssemblyState с пустыми placed и adjacency.

    Raises:
        ValueError: Если n_fragments < 1.
    """
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments must be ≥ 1, got {n_fragments}."
        )
    return AssemblyState(n_fragments=n_fragments)


# ─── place_fragment ───────────────────────────────────────────────────────────

def place_fragment(state:    AssemblyState,
                    idx:      int,
                    position: Tuple[float, float],
                    angle:    float = 0.0,
                    scale:    float = 1.0,
                    **meta) -> AssemblyState:
    """
    Добавляет фрагмент в сборку; возвращает новое состояние.

    Args:
        state:    Текущее AssemblyState (не изменяется).
        idx:      Индекс размещаемого фрагмента.
        position: (x, y) — координаты размещения.
        angle:    Угол поворота (°).
        scale:    Масштаб.
        **meta:   Метаданные (score, side, ...).

    Returns:
        Новый AssemblyState с добавленным фрагментом.

    Raises:
        ValueError: Если idx уже размещён или вне диапазона.
    """
    if idx < 0 or idx >= state.n_fragments:
        raise ValueError(
            f"Fragment index {idx} out of range [0, {state.n_fragments})."
        )
    if idx in state.placed:
        raise ValueError(f"Fragment {idx} is already placed.")

    new_state = copy.deepcopy(state)
    new_state.placed[idx] = PlacedFragment(
        idx=idx, position=position, angle=angle, scale=scale, meta=dict(meta)
    )
    if idx not in new_state.adjacency:
        new_state.adjacency[idx] = set()
    new_state.step += 1
    return new_state


# ─── remove_fragment ──────────────────────────────────────────────────────────

def remove_fragment(state: AssemblyState,
                     idx:   int) -> AssemblyState:
    """
    Убирает фрагмент из сборки; возвращает новое состояние.

    Args:
        state: Текущее AssemblyState (не изменяется).
        idx:   Индекс удаляемого фрагмента.

    Returns:
        Новый AssemblyState без данного фрагмента.

    Raises:
        KeyError: Если фрагмент не размещён.
    """
    if idx not in state.placed:
        raise KeyError(f"Fragment {idx} is not placed.")

    new_state = copy.deepcopy(state)
    del new_state.placed[idx]

    # Удалить из всех списков смежности
    new_state.adjacency.pop(idx, None)
    for neighbors in new_state.adjacency.values():
        neighbors.discard(idx)

    new_state.step += 1
    return new_state


# ─── add_adjacency ────────────────────────────────────────────────────────────

def add_adjacency(state: AssemblyState,
                   idx1:  int,
                   idx2:  int) -> AssemblyState:
    """
    Регистрирует смежность двух фрагментов (неориентированное ребро).

    Args:
        state: Текущее AssemblyState.
        idx1:  Первый фрагмент.
        idx2:  Второй фрагмент.

    Returns:
        Новый AssemblyState с добавленным ребром смежности.

    Raises:
        ValueError: Если idx1 == idx2.
    """
    if idx1 == idx2:
        raise ValueError(f"Cannot add self-adjacency for fragment {idx1}.")

    new_state = copy.deepcopy(state)
    new_state.adjacency.setdefault(idx1, set()).add(idx2)
    new_state.adjacency.setdefault(idx2, set()).add(idx1)
    return new_state


# ─── get_neighbors ────────────────────────────────────────────────────────────

def get_neighbors(state: AssemblyState,
                   idx:   int) -> List[int]:
    """
    Возвращает список соседей данного фрагмента.

    Args:
        state: AssemblyState.
        idx:   Индекс фрагмента.

    Returns:
        Список индексов соседних фрагментов (может быть пустым).
    """
    return sorted(state.adjacency.get(idx, set()))


# ─── compute_coverage ─────────────────────────────────────────────────────────

def compute_coverage(state: AssemblyState) -> float:
    """
    Возвращает долю размещённых фрагментов ∈ [0,1].

    Args:
        state: AssemblyState.

    Returns:
        len(placed) / n_fragments.
    """
    return len(state.placed) / state.n_fragments


# ─── is_complete ──────────────────────────────────────────────────────────────

def is_complete(state: AssemblyState) -> bool:
    """
    Проверяет, размещены ли все фрагменты.

    Returns:
        True, если len(placed) == n_fragments.
    """
    return len(state.placed) == state.n_fragments


# ─── to_dict ──────────────────────────────────────────────────────────────────

def to_dict(state: AssemblyState) -> dict:
    """
    Сериализует AssemblyState в словарь (для JSON-совместимого хранения).

    Returns:
        dict с полями 'n_fragments', 'placed', 'adjacency', 'step'.
    """
    placed_serial = {}
    for idx, pf in state.placed.items():
        placed_serial[str(idx)] = {
            "idx":      pf.idx,
            "position": list(pf.position),
            "angle":    pf.angle,
            "scale":    pf.scale,
            "meta":     pf.meta,
        }

    adjacency_serial = {
        str(k): sorted(v)
        for k, v in state.adjacency.items()
    }

    return {
        "n_fragments": state.n_fragments,
        "placed":      placed_serial,
        "adjacency":   adjacency_serial,
        "step":        state.step,
    }


# ─── from_dict ────────────────────────────────────────────────────────────────

def from_dict(d: dict) -> AssemblyState:
    """
    Десериализует AssemblyState из словаря.

    Args:
        d: Словарь, созданный функцией to_dict.

    Returns:
        AssemblyState.
    """
    placed: Dict[int, PlacedFragment] = {}
    for key, pf_data in d.get("placed", {}).items():
        idx = int(key)
        placed[idx] = PlacedFragment(
            idx=pf_data["idx"],
            position=tuple(pf_data["position"]),
            angle=float(pf_data["angle"]),
            scale=float(pf_data["scale"]),
            meta=pf_data.get("meta", {}),
        )

    adjacency: Dict[int, Set[int]] = {}
    for key, neighbors in d.get("adjacency", {}).items():
        adjacency[int(key)] = set(int(n) for n in neighbors)

    return AssemblyState(
        n_fragments=int(d["n_fragments"]),
        placed=placed,
        adjacency=adjacency,
        step=int(d.get("step", 0)),
    )
