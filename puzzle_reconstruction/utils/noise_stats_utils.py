"""Утилиты агрегации и сравнения статистик шумового анализа.

Обеспечивает структуры и функции для хранения, фильтрации, ранжирования
и агрегированного сравнения результатов анализа шума изображений.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class NoiseStatsConfig:
    """Конфигурация для агрегации шумовых статистик."""

    max_sigma: float = 50.0
    snr_threshold: float = 20.0
    quality_levels: int = 3

    def __post_init__(self) -> None:
        if self.max_sigma <= 0.0:
            raise ValueError("max_sigma must be > 0")
        if self.snr_threshold < 0.0:
            raise ValueError("snr_threshold must be >= 0")
        if self.quality_levels < 1:
            raise ValueError("quality_levels must be >= 1")


# ─── Entry ───────────────────────────────────────────────────────────────────


@dataclass
class NoiseStatsEntry:
    """Одна запись шумового анализа для изображения."""

    image_id: int
    sigma: float
    snr_db: float
    jpeg_level: float
    grain_level: float
    quality: str
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_id < 0:
            raise ValueError("image_id must be >= 0")
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if not (0.0 <= self.jpeg_level <= 1.0):
            raise ValueError("jpeg_level must be in [0, 1]")
        if not (0.0 <= self.grain_level <= 1.0):
            raise ValueError("grain_level must be in [0, 1]")
        if self.quality not in ("clean", "noisy", "very_noisy"):
            raise ValueError(f"unknown quality: {self.quality!r}")

    @property
    def is_clean(self) -> bool:
        return self.quality == "clean"

    @property
    def is_noisy(self) -> bool:
        return self.quality in ("noisy", "very_noisy")


# ─── Summary ─────────────────────────────────────────────────────────────────


@dataclass
class NoiseStatsSummary:
    """Агрегат по множеству записей шумового анализа."""

    entries: List[NoiseStatsEntry]
    n_total: int
    n_clean: int
    n_noisy: int
    mean_sigma: float
    max_sigma: float
    min_sigma: float
    mean_snr: float
    mean_jpeg: float
    mean_grain: float

    def __repr__(self) -> str:
        return (
            f"NoiseStatsSummary(n_total={self.n_total}, "
            f"n_clean={self.n_clean}, "
            f"mean_sigma={self.mean_sigma:.4f}, "
            f"mean_snr={self.mean_snr:.2f})"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────


def make_noise_entry(
    image_id: int,
    sigma: float,
    snr_db: float,
    jpeg_level: float,
    grain_level: float,
    quality: str,
    meta: Optional[Dict] = None,
) -> NoiseStatsEntry:
    """Создать запись NoiseStatsEntry."""
    return NoiseStatsEntry(
        image_id=image_id,
        sigma=float(sigma),
        snr_db=float(snr_db),
        jpeg_level=float(jpeg_level),
        grain_level=float(grain_level),
        quality=quality,
        meta=meta or {},
    )


def entries_from_analysis_results(
    results: List,
) -> List[NoiseStatsEntry]:
    """Построить список записей из объектов NoiseAnalysisResult.

    Ожидается, что у каждого объекта есть атрибуты
    ``noise_level``, ``snr_db``, ``jpeg_artifacts``,
    ``grain_level``, ``quality``.
    """
    entries: List[NoiseStatsEntry] = []
    for idx, r in enumerate(results):
        entry = make_noise_entry(
            image_id=idx,
            sigma=float(getattr(r, "noise_level", 0.0)),
            snr_db=float(getattr(r, "snr_db", 0.0)),
            jpeg_level=max(0.0, min(1.0, float(getattr(r, "jpeg_artifacts", 0.0)))),
            grain_level=max(0.0, min(1.0, float(getattr(r, "grain_level", 0.0)))),
            quality=str(getattr(r, "quality", "clean")),
        )
        entries.append(entry)
    return entries


# ─── Summarise ───────────────────────────────────────────────────────────────


def summarise_noise_stats(
    entries: List[NoiseStatsEntry],
) -> NoiseStatsSummary:
    """Вычислить сводку по списку записей."""
    n = len(entries)
    if n == 0:
        return NoiseStatsSummary(
            entries=[], n_total=0, n_clean=0, n_noisy=0,
            mean_sigma=0.0, max_sigma=0.0, min_sigma=0.0,
            mean_snr=0.0, mean_jpeg=0.0, mean_grain=0.0,
        )
    sigmas = [e.sigma for e in entries]
    n_clean = sum(1 for e in entries if e.is_clean)
    import math
    finite_snrs = [e.snr_db for e in entries if math.isfinite(e.snr_db)]
    mean_snr = (sum(finite_snrs) / len(finite_snrs)) if finite_snrs else 0.0
    return NoiseStatsSummary(
        entries=list(entries),
        n_total=n,
        n_clean=n_clean,
        n_noisy=n - n_clean,
        mean_sigma=sum(sigmas) / n,
        max_sigma=max(sigmas),
        min_sigma=min(sigmas),
        mean_snr=mean_snr,
        mean_jpeg=sum(e.jpeg_level for e in entries) / n,
        mean_grain=sum(e.grain_level for e in entries) / n,
    )


# ─── Filters ─────────────────────────────────────────────────────────────────


def filter_clean_entries(
    entries: List[NoiseStatsEntry],
) -> List[NoiseStatsEntry]:
    """Вернуть только записи с quality == 'clean'."""
    return [e for e in entries if e.is_clean]


def filter_noisy_entries(
    entries: List[NoiseStatsEntry],
) -> List[NoiseStatsEntry]:
    """Вернуть только записи с quality in {'noisy', 'very_noisy'}."""
    return [e for e in entries if e.is_noisy]


def filter_by_sigma_range(
    entries: List[NoiseStatsEntry],
    lo: float = 0.0,
    hi: float = 100.0,
) -> List[NoiseStatsEntry]:
    """Фильтр записей по диапазону sigma."""
    return [e for e in entries if lo <= e.sigma <= hi]


def filter_by_snr_range(
    entries: List[NoiseStatsEntry],
    lo: float = 0.0,
    hi: float = 200.0,
) -> List[NoiseStatsEntry]:
    """Фильтр записей по диапазону snr_db."""
    import math
    return [e for e in entries
            if math.isfinite(e.snr_db) and lo <= e.snr_db <= hi]


def filter_by_jpeg_threshold(
    entries: List[NoiseStatsEntry],
    max_jpeg: float = 0.5,
) -> List[NoiseStatsEntry]:
    """Записи с jpeg_level <= max_jpeg."""
    return [e for e in entries if e.jpeg_level <= max_jpeg]


# ─── Ranking ─────────────────────────────────────────────────────────────────


def top_k_cleanest(
    entries: List[NoiseStatsEntry],
    k: int,
) -> List[NoiseStatsEntry]:
    """Топ-k записей с наименьшей sigma."""
    if k <= 0:
        return []
    return sorted(entries, key=lambda e: e.sigma)[:k]


def top_k_noisiest(
    entries: List[NoiseStatsEntry],
    k: int,
) -> List[NoiseStatsEntry]:
    """Топ-k записей с наибольшей sigma."""
    if k <= 0:
        return []
    return sorted(entries, key=lambda e: e.sigma, reverse=True)[:k]


def best_snr_entry(
    entries: List[NoiseStatsEntry],
) -> Optional[NoiseStatsEntry]:
    """Запись с наибольшим snr_db (конечным)."""
    import math
    finite = [e for e in entries if math.isfinite(e.snr_db)]
    if not finite:
        return None
    return max(finite, key=lambda e: e.snr_db)


# ─── Statistics ──────────────────────────────────────────────────────────────


def noise_stats_dict(
    entries: List[NoiseStatsEntry],
) -> Dict:
    """Базовые статистики по sigma."""
    if not entries:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    sigmas = [e.sigma for e in entries]
    n = len(sigmas)
    mean = sum(sigmas) / n
    var = sum((s - mean) ** 2 for s in sigmas) / n
    return {
        "n": n,
        "mean": mean,
        "min": min(sigmas),
        "max": max(sigmas),
        "std": var ** 0.5,
    }


# ─── Comparison ──────────────────────────────────────────────────────────────


def compare_noise_summaries(
    a: NoiseStatsSummary,
    b: NoiseStatsSummary,
) -> Dict:
    """Сравнить две шумовые сводки."""
    return {
        "delta_mean_sigma": a.mean_sigma - b.mean_sigma,
        "delta_mean_snr": a.mean_snr - b.mean_snr,
        "delta_n_clean": a.n_clean - b.n_clean,
        "a_cleaner": a.mean_sigma <= b.mean_sigma,
    }


# ─── Batch ───────────────────────────────────────────────────────────────────


def batch_summarise_noise_stats(
    groups: List[List[NoiseStatsEntry]],
) -> List[NoiseStatsSummary]:
    """Применить summarise_noise_stats к каждой группе."""
    return [summarise_noise_stats(g) for g in groups]
