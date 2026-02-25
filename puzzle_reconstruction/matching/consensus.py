"""
Консенсусное голосование по результатам нескольких методов сборки.

Идея: запустить несколько методов сборки (greedy, SA, beam, gamma, genetic,
ACO) на одном наборе фрагментов и объединить их результаты голосованием.
Пары фрагментов, признанные соседними большинством методов, скорее всего
действительно соседние в оригинальном документе.

Алгоритм:
    1. По каждой Assembly формируем множество «соседних пар» (fid_i, fid_j).
    2. Считаем «рейтинг» пары = число Assembly, в которых она соседняя.
    3. Пары с рейтингом ≥ threshold × n_methods считаются консенсусными.
    4. По консенсусным парам строим финальную сборку через greedy,
       начиная с узлов-чемпионов (высокий рейтинг).

Классы / Функции:
    ConsensusResult        — результат голосования
    build_consensus        — основная функция
    assembly_to_pairs      — выделяет соседей из Assembly
    vote_on_pairs          — подсчёт рейтингов всех пар
    consensus_score_matrix — матрица консенсусных оценок N_frags × N_frags
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from ..models import Assembly, CompatEntry, Fragment


# ─── Тип «соседняя пара» ──────────────────────────────────────────────────────

Pair = FrozenSet[int]  # frozenset {frag_id_i, frag_id_j}


# ─── ConsensusResult ──────────────────────────────────────────────────────────

@dataclass
class ConsensusResult:
    """
    Результат голосования.

    Attributes:
        pair_votes:    {frozenset{i,j}: vote_count} — рейтинги всех пар.
        consensus_pairs: Пары с рейтингом ≥ threshold.
        n_methods:     Число проголосовавших методов.
        threshold:     Порог (доля голосов) для консенсуса.
        assembly:      Финальная Assembly (если build_assembly=True).
    """
    pair_votes:      Dict[Pair, int]
    consensus_pairs: Set[Pair]
    n_methods:       int
    threshold:       float
    assembly:        Optional[Assembly] = None

    # ── Удобные методы ────────────────────────────────────────────────────

    def vote_fraction(self, fid_a: int, fid_b: int) -> float:
        """Доля методов, признавших пару (fid_a, fid_b) соседней."""
        pair = frozenset({fid_a, fid_b})
        return self.pair_votes.get(pair, 0) / max(1, self.n_methods)

    def is_consensus(self, fid_a: int, fid_b: int) -> bool:
        """True если пара достигла порога консенсуса."""
        return frozenset({fid_a, fid_b}) in self.consensus_pairs

    def top_pairs(self, n: int = 10) -> List[Tuple[Pair, int]]:
        """Возвращает топ-n пар по убыванию числа голосов."""
        return sorted(self.pair_votes.items(), key=lambda x: x[1], reverse=True)[:n]

    def summary(self) -> str:
        """Краткое текстовое резюме."""
        n_total = len(self.pair_votes)
        n_cons  = len(self.consensus_pairs)
        return (f"ConsensusResult(methods={self.n_methods}, "
                f"threshold={self.threshold:.0%}, "
                f"pairs={n_total}, consensus={n_cons})")


# ─── Основная функция ─────────────────────────────────────────────────────────

def build_consensus(assemblies:     List[Assembly],
                    fragments:      List[Fragment],
                    entries:        List[CompatEntry],
                    threshold:      float = 0.5,
                    build_assembly: bool  = True) -> ConsensusResult:
    """
    Объединяет несколько Assembly через консенсусное голосование.

    Args:
        assemblies:     Список результатов сборки разных методов.
        fragments:      Исходные фрагменты (нужны для финальной сборки).
        entries:        CompatEntry (нужны для финальной greedy-сборки).
        threshold:      Порог голосов (0.5 = большинство).
        build_assembly: True → строить финальную Assembly из консенсусных пар.

    Returns:
        ConsensusResult.
    """
    if not assemblies:
        return ConsensusResult(
            pair_votes={}, consensus_pairs=set(),
            n_methods=0, threshold=threshold,
        )

    # ── Шаг 1–2: голосование ─────────────────────────────────────────────
    pair_votes = vote_on_pairs(assemblies)
    n_methods  = len(assemblies)

    # ── Шаг 3: отбор консенсусных пар ────────────────────────────────────
    min_votes = max(1, int(math.ceil(threshold * n_methods)))
    consensus_pairs = {
        pair for pair, votes in pair_votes.items()
        if votes >= min_votes
    }

    # ── Шаг 4: финальная сборка ───────────────────────────────────────────
    final_assembly = None
    if build_assembly and fragments and entries:
        # Фильтруем entries: оставляем только консенсусные пары,
        # повышая их score пропорционально числу голосов
        consensus_entries = _filter_entries_by_consensus(
            entries, pair_votes, n_methods, threshold,
        )
        try:
            sorted_entries = sorted(consensus_entries,
                                     key=lambda e: e.score, reverse=True)
        except Exception:
            sorted_entries = consensus_entries

        from ..assembly.greedy import greedy_assembly
        final_assembly = greedy_assembly(fragments, sorted_entries)

    return ConsensusResult(
        pair_votes=pair_votes,
        consensus_pairs=consensus_pairs,
        n_methods=n_methods,
        threshold=threshold,
        assembly=final_assembly,
    )


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def assembly_to_pairs(assembly: Assembly,
                       adjacency_threshold: float = 250.0) -> Set[Pair]:
    """
    Извлекает множество соседних пар фрагментов из Assembly.

    Два фрагмента считаются соседними если расстояние между их позициями
    меньше *adjacency_threshold* пикселей.

    Поддерживает два формата placements:
    - Dict[int, Tuple[np.ndarray, float]]  — {fid: (pos, angle)}
    - List[Placement]                       — список объектов Placement

    Args:
        assembly:             Сборка с placements.
        adjacency_threshold:  Максимальное расстояние (пиксели).

    Returns:
        Множество frozenset{fid_i, fid_j}.
    """
    pairs: Set[Pair] = set()
    placements = assembly.placements

    # Нормализуем в список [(fid, position_array)]
    items: list = []
    if isinstance(placements, dict):
        for fid, val in placements.items():
            pos = val[0] if isinstance(val, (tuple, list)) else val
            items.append((int(fid), np.asarray(pos, dtype=float)))
    elif isinstance(placements, list):
        for p in placements:
            if hasattr(p, "fragment_id") and hasattr(p, "position"):
                pos = np.asarray(p.position, dtype=float)
                items.append((int(p.fragment_id), pos))
    else:
        return pairs

    n = len(items)
    for i in range(n):
        fid_i, pos_i = items[i]
        for j in range(i + 1, n):
            fid_j, pos_j = items[j]
            dist = float(np.linalg.norm(pos_i - pos_j))
            if dist < adjacency_threshold:
                pairs.add(frozenset({fid_i, fid_j}))

    return pairs


def vote_on_pairs(assemblies: List[Assembly],
                   adjacency_threshold: float = 250.0) -> Dict[Pair, int]:
    """
    Подсчитывает, сколько Assembly признаёт каждую пару соседней.

    Args:
        assemblies:          Список объектов Assembly.
        adjacency_threshold: Порог расстояния для определения соседства.

    Returns:
        {frozenset{fid_i, fid_j}: count} — рейтинг каждой пары.
    """
    votes: Dict[Pair, int] = {}
    for assembly in assemblies:
        for pair in assembly_to_pairs(assembly, adjacency_threshold):
            votes[pair] = votes.get(pair, 0) + 1
    return votes


def consensus_score_matrix(result:    ConsensusResult,
                             fragments: List[Fragment]) -> np.ndarray:
    """
    Строит матрицу консенсусных оценок N×N.

    Элемент [i, j] = доля методов, признавших пару (i, j) соседней.

    Args:
        result:    ConsensusResult с pair_votes.
        fragments: Список фрагментов (задаёт порядок).

    Returns:
        (N, N) float64 симметричная матрица ∈ [0, 1].
    """
    n       = len(fragments)
    fid_to_idx = {f.fragment_id: i for i, f in enumerate(fragments)}
    mat     = np.zeros((n, n), dtype=np.float64)

    for pair, votes in result.pair_votes.items():
        ids = list(pair)
        if len(ids) != 2:
            continue
        ii = fid_to_idx.get(ids[0])
        ij = fid_to_idx.get(ids[1])
        if ii is None or ij is None:
            continue
        frac = votes / max(1, result.n_methods)
        mat[ii, ij] = frac
        mat[ij, ii] = frac

    return mat


def _filter_entries_by_consensus(entries:    List[CompatEntry],
                                  pair_votes: Dict[Pair, int],
                                  n_methods:  int,
                                  threshold:  float) -> List[CompatEntry]:
    """
    Повышает score консенсусных пар и оставляет их поверх остальных.

    Консенсусная пара: score += vote_fraction * 0.5 (бонус до 50%).
    Остальные пары: score *= 0.5 (штраф).
    """
    edge_to_frag: Dict[int, int] = {}
    # Строим карту edge_id → fragment_id из entries
    for e in entries:
        # Нет прямого доступа к fragment_id из EdgeSignature — используем
        # имеющиеся данные (edge_id уникален для фрагмента)
        pass

    result = []
    for e in entries:
        # Пробуем найти пару через edge_id
        # Используем id рёбер как приближение пары фрагментов
        fid_i = e.edge_i.edge_id // 10  # Предполагаем формат fid * 10 + side
        fid_j = e.edge_j.edge_id // 10
        pair  = frozenset({fid_i, fid_j})
        votes = pair_votes.get(pair, 0)
        frac  = votes / max(1, n_methods)

        new_score = float(np.clip(
            e.score + frac * 0.5 if frac >= threshold else e.score * 0.5,
            0.0, 1.0,
        ))
        result.append(CompatEntry(
            edge_i=e.edge_i,
            edge_j=e.edge_j,
            score=new_score,
            dtw_dist=e.dtw_dist,
            css_sim=e.css_sim,
            fd_diff=e.fd_diff,
            text_score=e.text_score,
        ))

    return result


# Импорт math откладываем, чтобы не создавать кружных зависимостей
import math  # noqa: E402 (перемещён в конец по стилистическим соображениям)
