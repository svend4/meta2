"""Проверка согласованности сборки пазла.

Модуль проверяет, что расставленные фрагменты образуют корректную
сборку: нет дублирующихся идентификаторов, все фрагменты присутствуют,
зазоры соответствуют норме, оценки совместимости достаточны,
а позиции не выходят за границы холста.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─── ConsistencyIssue ─────────────────────────────────────────────────────────

@dataclass
class ConsistencyIssue:
    """Одна проблема согласованности.

    Атрибуты:
        code:        Код проблемы (например, 'DUPLICATE_ID', 'MISSING_FRAGMENT').
        description: Текстовое описание.
        fragment_ids: Идентификаторы фрагментов, к которым относится проблема.
        severity:    Уровень серьёзности ('error', 'warning', 'info').
    """

    code: str
    description: str
    fragment_ids: List[int] = field(default_factory=list)
    severity: str = "error"

    def __post_init__(self) -> None:
        valid = {"error", "warning", "info"}
        if self.severity not in valid:
            raise ValueError(
                f"severity должна быть одним из {sorted(valid)}, "
                f"получено '{self.severity}'"
            )
        if not self.code:
            raise ValueError("code не может быть пустым")


# ─── ConsistencyReport ────────────────────────────────────────────────────────

@dataclass
class ConsistencyReport:
    """Полный отчёт о согласованности сборки.

    Атрибуты:
        issues:          Список обнаруженных проблем.
        is_consistent:   True, если нет ошибок (только предупреждения/info).
        n_errors:        Количество ошибок.
        n_warnings:      Количество предупреждений.
        checked_pairs:   Количество проверенных пар фрагментов.
    """

    issues: List[ConsistencyIssue]
    is_consistent: bool
    n_errors: int
    n_warnings: int
    checked_pairs: int = 0

    def __post_init__(self) -> None:
        if self.n_errors < 0:
            raise ValueError(
                f"n_errors должно быть >= 0, получено {self.n_errors}"
            )
        if self.n_warnings < 0:
            raise ValueError(
                f"n_warnings должно быть >= 0, получено {self.n_warnings}"
            )
        if self.checked_pairs < 0:
            raise ValueError(
                f"checked_pairs должно быть >= 0, получено {self.checked_pairs}"
            )

    def __len__(self) -> int:
        return len(self.issues)


# ─── check_unique_ids ─────────────────────────────────────────────────────────

def check_unique_ids(fragment_ids: List[int]) -> List[ConsistencyIssue]:
    """Проверить уникальность идентификаторов фрагментов.

    Аргументы:
        fragment_ids: Список идентификаторов фрагментов в сборке.

    Возвращает:
        Список ConsistencyIssue (пустой, если дубликатов нет).
    """
    seen: Set[int] = set()
    duplicates: Set[int] = set()
    for fid in fragment_ids:
        if fid in seen:
            duplicates.add(fid)
        seen.add(fid)
    if duplicates:
        return [ConsistencyIssue(
            code="DUPLICATE_ID",
            description=f"Дублирующиеся идентификаторы: {sorted(duplicates)}",
            fragment_ids=sorted(duplicates),
            severity="error",
        )]
    return []


# ─── check_all_present ────────────────────────────────────────────────────────

def check_all_present(
    fragment_ids: List[int], expected_ids: List[int]
) -> List[ConsistencyIssue]:
    """Проверить, что все ожидаемые фрагменты присутствуют в сборке.

    Аргументы:
        fragment_ids:  Фактически присутствующие идентификаторы.
        expected_ids:  Ожидаемые идентификаторы.

    Возвращает:
        Список ConsistencyIssue для отсутствующих и лишних фрагментов.
    """
    present = set(fragment_ids)
    expected = set(expected_ids)
    issues: List[ConsistencyIssue] = []

    missing = sorted(expected - present)
    if missing:
        issues.append(ConsistencyIssue(
            code="MISSING_FRAGMENT",
            description=f"Отсутствующие фрагменты: {missing}",
            fragment_ids=missing,
            severity="error",
        ))

    extra = sorted(present - expected)
    if extra:
        issues.append(ConsistencyIssue(
            code="EXTRA_FRAGMENT",
            description=f"Лишние фрагменты: {extra}",
            fragment_ids=extra,
            severity="warning",
        ))

    return issues


# ─── check_canvas_bounds ──────────────────────────────────────────────────────

def check_canvas_bounds(
    positions: List[Tuple[int, int]],
    sizes: List[Tuple[int, int]],
    canvas_w: int,
    canvas_h: int,
) -> List[ConsistencyIssue]:
    """Проверить, что фрагменты не выходят за границы холста.

    Аргументы:
        positions: Список (x, y) для каждого фрагмента.
        sizes:     Список (w, h) для каждого фрагмента.
        canvas_w:  Ширина холста (>= 1).
        canvas_h:  Высота холста (>= 1).

    Возвращает:
        Список ConsistencyIssue для выходящих за границы фрагментов.

    Исключения:
        ValueError: Если длины списков не совпадают или canvas < 1.
    """
    if canvas_w < 1:
        raise ValueError(f"canvas_w должен быть >= 1, получено {canvas_w}")
    if canvas_h < 1:
        raise ValueError(f"canvas_h должен быть >= 1, получено {canvas_h}")
    if len(positions) != len(sizes):
        raise ValueError(
            f"Длины positions и sizes не совпадают: "
            f"{len(positions)} != {len(sizes)}"
        )

    issues: List[ConsistencyIssue] = []
    for i, ((x, y), (w, h)) in enumerate(zip(positions, sizes)):
        if x < 0 or y < 0 or x + w > canvas_w or y + h > canvas_h:
            issues.append(ConsistencyIssue(
                code="OUT_OF_BOUNDS",
                description=(
                    f"Фрагмент {i} выходит за границы холста "
                    f"({x},{y})+({w},{h}) > ({canvas_w},{canvas_h})"
                ),
                fragment_ids=[i],
                severity="error",
            ))
    return issues


# ─── check_score_threshold ────────────────────────────────────────────────────

def check_score_threshold(
    pair_scores: Dict[Tuple[int, int], float],
    min_score: float = 0.5,
) -> List[ConsistencyIssue]:
    """Проверить, что оценки совместимости пар не ниже порога.

    Аргументы:
        pair_scores: Словарь {(idx1, idx2): score}.
        min_score:   Минимально допустимая оценка (>= 0).

    Возвращает:
        Список ConsistencyIssue для пар с низкой оценкой.

    Исключения:
        ValueError: Если min_score < 0.
    """
    if min_score < 0.0:
        raise ValueError(f"min_score должен быть >= 0, получено {min_score}")

    issues: List[ConsistencyIssue] = []
    for (i, j), score in sorted(pair_scores.items()):
        if score < min_score:
            issues.append(ConsistencyIssue(
                code="LOW_SCORE",
                description=(
                    f"Низкая оценка совместимости пары ({i},{j}): "
                    f"{score:.4f} < {min_score}"
                ),
                fragment_ids=[i, j],
                severity="warning",
            ))
    return issues


# ─── check_gap_uniformity ─────────────────────────────────────────────────────

def check_gap_uniformity(
    gaps: List[float],
    max_std: float = 5.0,
) -> List[ConsistencyIssue]:
    """Проверить равномерность зазоров между фрагментами.

    Аргументы:
        gaps:    Список измеренных зазоров (px).
        max_std: Максимально допустимое стандартное отклонение (>= 0).

    Возвращает:
        Список ConsistencyIssue если разброс зазоров превышает max_std.

    Исключения:
        ValueError: Если max_std < 0.
    """
    if max_std < 0.0:
        raise ValueError(f"max_std должен быть >= 0, получено {max_std}")
    if len(gaps) < 2:
        return []

    std = float(np.std(gaps, ddof=0))
    if std > max_std:
        return [ConsistencyIssue(
            code="NONUNIFORM_GAP",
            description=(
                f"Неравномерные зазоры: std={std:.2f} > max_std={max_std}"
            ),
            severity="warning",
        )]
    return []


# ─── run_consistency_check ────────────────────────────────────────────────────

def run_consistency_check(
    fragment_ids: List[int],
    expected_ids: List[int],
    positions: List[Tuple[int, int]],
    sizes: List[Tuple[int, int]],
    canvas_w: int,
    canvas_h: int,
    pair_scores: Optional[Dict[Tuple[int, int], float]] = None,
    min_score: float = 0.5,
) -> ConsistencyReport:
    """Выполнить полную проверку согласованности сборки.

    Аргументы:
        fragment_ids:  Идентификаторы фрагментов в сборке.
        expected_ids:  Ожидаемые идентификаторы.
        positions:     Позиции (x, y) каждого фрагмента.
        sizes:         Размеры (w, h) каждого фрагмента.
        canvas_w:      Ширина холста.
        canvas_h:      Высота холста.
        pair_scores:   Оценки совместимости пар (опционально).
        min_score:     Порог оценки совместимости.

    Возвращает:
        ConsistencyReport с полным списком проблем.
    """
    all_issues: List[ConsistencyIssue] = []

    all_issues.extend(check_unique_ids(fragment_ids))
    all_issues.extend(check_all_present(fragment_ids, expected_ids))
    all_issues.extend(
        check_canvas_bounds(positions, sizes, canvas_w, canvas_h)
    )
    if pair_scores:
        score_issues = check_score_threshold(pair_scores, min_score=min_score)
        all_issues.extend(score_issues)

    n_errors = sum(1 for iss in all_issues if iss.severity == "error")
    n_warnings = sum(1 for iss in all_issues if iss.severity == "warning")
    checked_pairs = len(pair_scores) if pair_scores else 0

    return ConsistencyReport(
        issues=all_issues,
        is_consistent=(n_errors == 0),
        n_errors=n_errors,
        n_warnings=n_warnings,
        checked_pairs=checked_pairs,
    )


# ─── batch_consistency_check ──────────────────────────────────────────────────

def batch_consistency_check(
    assemblies: List[Dict],
) -> List[ConsistencyReport]:
    """Проверить согласованность нескольких сборок.

    Аргументы:
        assemblies: Список словарей с ключами, соответствующими аргументам
                    run_consistency_check (кроме pair_scores, min_score).

    Возвращает:
        Список ConsistencyReport.
    """
    results = []
    for asm in assemblies:
        report = run_consistency_check(
            fragment_ids=asm["fragment_ids"],
            expected_ids=asm["expected_ids"],
            positions=asm["positions"],
            sizes=asm["sizes"],
            canvas_w=asm["canvas_w"],
            canvas_h=asm["canvas_h"],
            pair_scores=asm.get("pair_scores"),
            min_score=asm.get("min_score", 0.5),
        )
        results.append(report)
    return results
