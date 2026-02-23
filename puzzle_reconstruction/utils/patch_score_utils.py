"""Утилиты анализа и ранжирования результатов сопоставления патчей.

Предоставляет структуры данных и функции для хранения, фильтрации
и сравнения результатов попарного сопоставления патчей изображений.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class PatchScoreConfig:
    """Конфигурация для фильтрации и анализа результатов сопоставления патчей."""

    min_score: float = 0.0
    max_pairs: int = 1000
    method: str = "total"

    def __post_init__(self) -> None:
        if self.min_score < 0.0:
            raise ValueError("min_score must be >= 0.0")
        if self.max_pairs <= 0:
            raise ValueError("max_pairs must be > 0")
        if self.method not in ("total", "ncc", "ssd", "ssim"):
            raise ValueError(f"unknown method: {self.method!r}")


# ─── Entry ───────────────────────────────────────────────────────────────────


@dataclass
class PatchScoreEntry:
    """Запись об одном результате попарного сопоставления патчей."""

    pair_id: int
    idx1: int
    idx2: int
    side1: int
    side2: int
    ncc: float
    ssd: float
    ssim: float
    total_score: float
    rank: int = 0
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pair_id < 0:
            raise ValueError("pair_id must be >= 0")
        if self.idx1 < 0 or self.idx2 < 0:
            raise ValueError("idx1, idx2 must be >= 0")
        if not (0 <= self.ssd <= 1.0):
            raise ValueError("ssd must be in [0, 1]")
        if not (0 <= self.ssim <= 1.0):
            raise ValueError("ssim must be in [0, 1]")
        if not (0 <= self.total_score <= 1.0):
            raise ValueError("total_score must be in [0, 1]")

    @property
    def pair(self):
        """Возвращает пару индексов (idx1, idx2)."""
        return (self.idx1, self.idx2)

    @property
    def is_good(self) -> bool:
        """True если total_score > 0.5."""
        return self.total_score > 0.5


# ─── Summary ─────────────────────────────────────────────────────────────────


@dataclass
class PatchScoreSummary:
    """Агрегированные данные по нескольким результатам сопоставления патчей."""

    entries: List[PatchScoreEntry]
    n_total: int
    n_good: int
    n_poor: int
    mean_total: float
    max_total: float
    min_total: float
    mean_ncc: float
    mean_ssd: float
    mean_ssim: float

    def __repr__(self) -> str:
        return (
            f"PatchScoreSummary(n_total={self.n_total}, "
            f"n_good={self.n_good}, "
            f"mean_total={self.mean_total:.4f}, "
            f"max_total={self.max_total:.4f})"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────


def make_patch_entry(
    pair_id: int,
    idx1: int,
    idx2: int,
    side1: int,
    side2: int,
    ncc: float,
    ssd: float,
    ssim: float,
    total_score: float,
    rank: int = 0,
    meta: Optional[Dict] = None,
) -> PatchScoreEntry:
    """Создать запись PatchScoreEntry."""
    return PatchScoreEntry(
        pair_id=pair_id,
        idx1=idx1,
        idx2=idx2,
        side1=side1,
        side2=side2,
        ncc=float(ncc),
        ssd=float(ssd),
        ssim=float(ssim),
        total_score=float(total_score),
        rank=rank,
        meta=meta or {},
    )


def entries_from_patch_matches(
    patch_matches: List,
) -> List[PatchScoreEntry]:
    """Построить список записей из объектов PatchMatch.

    Ожидается, что у каждого объекта есть атрибуты
    ``idx1``, ``idx2``, ``side1``, ``side2``, ``ncc``, ``ssd``,
    ``ssim``, ``total_score``.
    """
    entries: List[PatchScoreEntry] = []
    for pid, pm in enumerate(patch_matches):
        entry = make_patch_entry(
            pair_id=pid,
            idx1=int(getattr(pm, "idx1", 0)),
            idx2=int(getattr(pm, "idx2", 0)),
            side1=int(getattr(pm, "side1", 0)),
            side2=int(getattr(pm, "side2", 0)),
            ncc=max(-1.0, min(1.0, float(getattr(pm, "ncc", 0.0)))),
            ssd=float(getattr(pm, "ssd", 0.0)),
            ssim=float(getattr(pm, "ssim", 0.0)),
            total_score=float(getattr(pm, "total_score", 0.0)),
        )
        entries.append(entry)
    # assign ranks descending by total_score
    for rank, e in enumerate(
        sorted(entries, key=lambda x: x.total_score, reverse=True), start=1
    ):
        e.rank = rank
    return entries


def summarise_patch_scores(
    entries: List[PatchScoreEntry],
) -> PatchScoreSummary:
    """Вычислить сводку по списку записей PatchScoreEntry."""
    n = len(entries)
    if n == 0:
        return PatchScoreSummary(
            entries=[],
            n_total=0, n_good=0, n_poor=0,
            mean_total=0.0, max_total=0.0, min_total=0.0,
            mean_ncc=0.0, mean_ssd=0.0, mean_ssim=0.0,
        )
    totals = [e.total_score for e in entries]
    n_good = sum(1 for e in entries if e.is_good)
    return PatchScoreSummary(
        entries=list(entries),
        n_total=n,
        n_good=n_good,
        n_poor=n - n_good,
        mean_total=sum(totals) / n,
        max_total=max(totals),
        min_total=min(totals),
        mean_ncc=sum(e.ncc for e in entries) / n,
        mean_ssd=sum(e.ssd for e in entries) / n,
        mean_ssim=sum(e.ssim for e in entries) / n,
    )


# ─── Filters ─────────────────────────────────────────────────────────────────


def filter_good_patch_scores(
    entries: List[PatchScoreEntry],
) -> List[PatchScoreEntry]:
    """Вернуть только записи с is_good == True."""
    return [e for e in entries if e.is_good]


def filter_poor_patch_scores(
    entries: List[PatchScoreEntry],
) -> List[PatchScoreEntry]:
    """Вернуть только записи с is_good == False."""
    return [e for e in entries if not e.is_good]


def filter_patch_by_score_range(
    entries: List[PatchScoreEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[PatchScoreEntry]:
    """Фильтр записей по диапазону total_score."""
    return [e for e in entries if lo <= e.total_score <= hi]


def filter_by_side_pair(
    entries: List[PatchScoreEntry],
    side1: int,
    side2: int,
) -> List[PatchScoreEntry]:
    """Фильтр записей по паре сторон (side1, side2)."""
    return [e for e in entries if e.side1 == side1 and e.side2 == side2]


def filter_by_ncc_range(
    entries: List[PatchScoreEntry],
    lo: float = -1.0,
    hi: float = 1.0,
) -> List[PatchScoreEntry]:
    """Фильтр записей по диапазону ncc."""
    return [e for e in entries if lo <= e.ncc <= hi]


# ─── Ranking ─────────────────────────────────────────────────────────────────


def top_k_patch_entries(
    entries: List[PatchScoreEntry],
    k: int,
) -> List[PatchScoreEntry]:
    """Вернуть топ-k записей по убыванию total_score."""
    if k <= 0:
        return []
    return sorted(entries, key=lambda e: e.total_score, reverse=True)[:k]


def best_patch_entry(
    entries: List[PatchScoreEntry],
) -> Optional[PatchScoreEntry]:
    """Вернуть запись с наибольшим total_score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.total_score)


# ─── Statistics ──────────────────────────────────────────────────────────────


def patch_score_stats(
    entries: List[PatchScoreEntry],
) -> Dict:
    """Вернуть словарь базовых статистик."""
    if not entries:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    totals = [e.total_score for e in entries]
    n = len(totals)
    mean = sum(totals) / n
    variance = sum((s - mean) ** 2 for s in totals) / n
    return {
        "n": n,
        "mean": mean,
        "min": min(totals),
        "max": max(totals),
        "std": variance ** 0.5,
    }


# ─── Comparison ──────────────────────────────────────────────────────────────


def compare_patch_summaries(
    a: PatchScoreSummary,
    b: PatchScoreSummary,
) -> Dict:
    """Сравнить две сводки; вернуть словарь с delta-значениями."""
    return {
        "delta_mean_total": a.mean_total - b.mean_total,
        "delta_max_total": a.max_total - b.max_total,
        "delta_n_good": a.n_good - b.n_good,
        "a_better": a.mean_total >= b.mean_total,
    }


# ─── Batch ───────────────────────────────────────────────────────────────────


def batch_summarise_patch_scores(
    groups: List[List[PatchScoreEntry]],
) -> List[PatchScoreSummary]:
    """Применить summarise_patch_scores к каждой группе записей."""
    return [summarise_patch_scores(g) for g in groups]
