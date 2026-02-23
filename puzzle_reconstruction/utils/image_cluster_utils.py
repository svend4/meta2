"""Утилиты анализа результатов статистики изображений и кластеризации фрагментов.

Модуль предоставляет вспомогательные структуры и функции для:
- Агрегации и фильтрации результатов статистического анализа изображений
- Анализа и ранжирования результатов кластеризации фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Image Statistics Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageStatsAnalysisConfig:
    """Конфигурация анализа результатов статистики изображений.

    Атрибуты:
        min_sharpness:   Минимальный порог резкости для фильтрации.
        max_entropy:     Максимальный порог энтропии.
        min_contrast:    Минимальный порог контрастности.
    """
    min_sharpness: float = 0.0
    max_entropy: float = 8.0
    min_contrast: float = 0.0


@dataclass
class ImageStatsAnalysisEntry:
    """Запись результата статистического анализа одного изображения.

    Атрибуты:
        fragment_id: Идентификатор фрагмента.
        sharpness:   Показатель резкости.
        entropy:     Энтропия пикселей.
        contrast:    Контрастность (стандартное отклонение).
        mean:        Среднее значение пикселей.
        n_pixels:    Число пикселей.
        params:      Дополнительные параметры.
    """
    fragment_id: int
    sharpness: float
    entropy: float
    contrast: float
    mean: float
    n_pixels: int
    params: Dict = field(default_factory=dict)


@dataclass
class ImageStatsAnalysisSummary:
    """Сводка статистического анализа набора изображений.

    Атрибуты:
        n_images:        Число изображений.
        mean_sharpness:  Средняя резкость.
        mean_entropy:    Средняя энтропия.
        sharpest_id:     ID самого резкого фрагмента или None.
        blurriest_id:    ID наименее резкого фрагмента или None.
    """
    n_images: int
    mean_sharpness: float
    mean_entropy: float
    sharpest_id: Optional[int]
    blurriest_id: Optional[int]


def make_image_stats_entry(
    fragment_id: int,
    sharpness: float,
    entropy: float,
    contrast: float,
    mean: float,
    n_pixels: int,
    **params,
) -> ImageStatsAnalysisEntry:
    """Создать запись результата статистического анализа изображения.

    Args:
        fragment_id: Идентификатор фрагмента.
        sharpness:   Резкость.
        entropy:     Энтропия.
        contrast:    Контрастность.
        mean:        Среднее значение пикселей.
        n_pixels:    Число пикселей.
        **params:    Дополнительные параметры.

    Returns:
        :class:`ImageStatsAnalysisEntry`.
    """
    return ImageStatsAnalysisEntry(
        fragment_id=fragment_id,
        sharpness=float(sharpness),
        entropy=float(entropy),
        contrast=float(contrast),
        mean=float(mean),
        n_pixels=int(n_pixels),
        params=dict(params),
    )


def summarise_image_stats_entries(
    entries: Sequence[ImageStatsAnalysisEntry],
    cfg: Optional[ImageStatsAnalysisConfig] = None,
) -> ImageStatsAnalysisSummary:
    """Сформировать сводку по записям статистического анализа изображений.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → ImageStatsAnalysisConfig()).

    Returns:
        :class:`ImageStatsAnalysisSummary`.
    """
    if cfg is None:
        cfg = ImageStatsAnalysisConfig()
    if not entries:
        return ImageStatsAnalysisSummary(
            n_images=0,
            mean_sharpness=0.0,
            mean_entropy=0.0,
            sharpest_id=None,
            blurriest_id=None,
        )
    sharpness_vals = [e.sharpness for e in entries]
    entropy_vals = [e.entropy for e in entries]
    mean_sharp = sum(sharpness_vals) / len(sharpness_vals)
    mean_entr = sum(entropy_vals) / len(entropy_vals)
    sharpest = max(entries, key=lambda e: e.sharpness)
    blurriest = min(entries, key=lambda e: e.sharpness)
    return ImageStatsAnalysisSummary(
        n_images=len(entries),
        mean_sharpness=mean_sharp,
        mean_entropy=mean_entr,
        sharpest_id=sharpest.fragment_id,
        blurriest_id=blurriest.fragment_id,
    )


def filter_by_min_sharpness(
    entries: Sequence[ImageStatsAnalysisEntry],
    min_sharpness: float,
) -> List[ImageStatsAnalysisEntry]:
    """Отфильтровать записи по минимальной резкости."""
    return [e for e in entries if e.sharpness >= min_sharpness]


def filter_by_max_entropy(
    entries: Sequence[ImageStatsAnalysisEntry],
    max_entropy: float,
) -> List[ImageStatsAnalysisEntry]:
    """Отфильтровать записи по максимальной энтропии."""
    return [e for e in entries if e.entropy <= max_entropy]


def filter_by_min_contrast(
    entries: Sequence[ImageStatsAnalysisEntry],
    min_contrast: float,
) -> List[ImageStatsAnalysisEntry]:
    """Отфильтровать записи по минимальной контрастности."""
    return [e for e in entries if e.contrast >= min_contrast]


def top_k_sharpest(
    entries: Sequence[ImageStatsAnalysisEntry],
    k: int,
) -> List[ImageStatsAnalysisEntry]:
    """Вернуть k наиболее резких изображений."""
    return sorted(entries, key=lambda e: e.sharpness, reverse=True)[:k]


def best_image_stats_entry(
    entries: Sequence[ImageStatsAnalysisEntry],
) -> Optional[ImageStatsAnalysisEntry]:
    """Вернуть запись с наибольшей резкостью."""
    return max(entries, key=lambda e: e.sharpness) if entries else None


def image_stats_score_stats(
    entries: Sequence[ImageStatsAnalysisEntry],
) -> Dict:
    """Вычислить статистику резкости: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.sharpness for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_image_stats_summaries(
    a: ImageStatsAnalysisSummary,
    b: ImageStatsAnalysisSummary,
) -> Dict:
    """Сравнить две сводки анализа изображений."""
    return {
        "delta_mean_sharpness": b.mean_sharpness - a.mean_sharpness,
        "delta_mean_entropy": b.mean_entropy - a.mean_entropy,
        "delta_n_images": b.n_images - a.n_images,
    }


def batch_summarise_image_stats_entries(
    groups: Sequence[Sequence[ImageStatsAnalysisEntry]],
    cfg: Optional[ImageStatsAnalysisConfig] = None,
) -> List[ImageStatsAnalysisSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_image_stats_entries(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusteringAnalysisConfig:
    """Конфигурация анализа результатов кластеризации.

    Атрибуты:
        min_silhouette: Минимальный порог силуэтного коэффициента.
        max_inertia:    Максимально допустимая инерция.
    """
    min_silhouette: float = 0.0
    max_inertia: float = float("inf")


@dataclass
class ClusteringAnalysisEntry:
    """Запись результата кластеризации одного набора фрагментов.

    Атрибуты:
        run_id:       Идентификатор прогона.
        n_clusters:   Число кластеров.
        inertia:      Инерция кластеризации.
        silhouette:   Силуэтный коэффициент.
        algorithm:    Метод кластеризации.
        n_samples:    Число точек.
        params:       Дополнительные параметры.
    """
    run_id: int
    n_clusters: int
    inertia: float
    silhouette: float
    algorithm: str
    n_samples: int
    params: Dict = field(default_factory=dict)


@dataclass
class ClusteringAnalysisSummary:
    """Сводка результатов кластеризации.

    Атрибуты:
        n_runs:          Число прогонов.
        mean_inertia:    Средняя инерция.
        mean_silhouette: Средний силуэтный коэффициент.
        best_run_id:     ID лучшего прогона (наибольший силуэт) или None.
        worst_run_id:    ID худшего прогона (наименьший силуэт) или None.
    """
    n_runs: int
    mean_inertia: float
    mean_silhouette: float
    best_run_id: Optional[int]
    worst_run_id: Optional[int]


def make_clustering_entry(
    run_id: int,
    n_clusters: int,
    inertia: float,
    silhouette: float,
    algorithm: str,
    n_samples: int,
    **params,
) -> ClusteringAnalysisEntry:
    """Создать запись результата кластеризации.

    Args:
        run_id:      Идентификатор прогона.
        n_clusters:  Число кластеров.
        inertia:     Инерция.
        silhouette:  Силуэтный коэффициент.
        algorithm:   Метод кластеризации.
        n_samples:   Число точек.
        **params:    Дополнительные параметры.

    Returns:
        :class:`ClusteringAnalysisEntry`.
    """
    return ClusteringAnalysisEntry(
        run_id=run_id,
        n_clusters=n_clusters,
        inertia=float(inertia),
        silhouette=float(silhouette),
        algorithm=algorithm,
        n_samples=n_samples,
        params=dict(params),
    )


def summarise_clustering_entries(
    entries: Sequence[ClusteringAnalysisEntry],
    cfg: Optional[ClusteringAnalysisConfig] = None,
) -> ClusteringAnalysisSummary:
    """Сформировать сводку по записям результатов кластеризации.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → ClusteringAnalysisConfig()).

    Returns:
        :class:`ClusteringAnalysisSummary`.
    """
    if cfg is None:
        cfg = ClusteringAnalysisConfig()
    if not entries:
        return ClusteringAnalysisSummary(
            n_runs=0,
            mean_inertia=0.0,
            mean_silhouette=0.0,
            best_run_id=None,
            worst_run_id=None,
        )
    inertia_vals = [e.inertia for e in entries]
    sil_vals = [e.silhouette for e in entries]
    mean_in = sum(inertia_vals) / len(inertia_vals)
    mean_sil = sum(sil_vals) / len(sil_vals)
    best = max(entries, key=lambda e: e.silhouette)
    worst = min(entries, key=lambda e: e.silhouette)
    return ClusteringAnalysisSummary(
        n_runs=len(entries),
        mean_inertia=mean_in,
        mean_silhouette=mean_sil,
        best_run_id=best.run_id,
        worst_run_id=worst.run_id,
    )


def filter_clustering_by_min_silhouette(
    entries: Sequence[ClusteringAnalysisEntry],
    min_silhouette: float,
) -> List[ClusteringAnalysisEntry]:
    """Отфильтровать записи по минимальному силуэтному коэффициенту."""
    return [e for e in entries if e.silhouette >= min_silhouette]


def filter_clustering_by_max_inertia(
    entries: Sequence[ClusteringAnalysisEntry],
    max_inertia: float,
) -> List[ClusteringAnalysisEntry]:
    """Отфильтровать записи по максимальной инерции."""
    return [e for e in entries if e.inertia <= max_inertia]


def filter_clustering_by_algorithm(
    entries: Sequence[ClusteringAnalysisEntry],
    algorithm: str,
) -> List[ClusteringAnalysisEntry]:
    """Отфильтровать записи по алгоритму кластеризации."""
    return [e for e in entries if e.algorithm == algorithm]


def filter_clustering_by_n_clusters(
    entries: Sequence[ClusteringAnalysisEntry],
    n_clusters: int,
) -> List[ClusteringAnalysisEntry]:
    """Отфильтровать записи по числу кластеров."""
    return [e for e in entries if e.n_clusters == n_clusters]


def top_k_clustering_entries(
    entries: Sequence[ClusteringAnalysisEntry],
    k: int,
) -> List[ClusteringAnalysisEntry]:
    """Вернуть k прогонов с наибольшим силуэтным коэффициентом."""
    return sorted(entries, key=lambda e: e.silhouette, reverse=True)[:k]


def best_clustering_entry(
    entries: Sequence[ClusteringAnalysisEntry],
) -> Optional[ClusteringAnalysisEntry]:
    """Вернуть прогон с наибольшим силуэтным коэффициентом."""
    return max(entries, key=lambda e: e.silhouette) if entries else None


def clustering_score_stats(
    entries: Sequence[ClusteringAnalysisEntry],
) -> Dict:
    """Вычислить статистику силуэтных коэффициентов: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.silhouette for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_clustering_summaries(
    a: ClusteringAnalysisSummary,
    b: ClusteringAnalysisSummary,
) -> Dict:
    """Сравнить две сводки результатов кластеризации."""
    return {
        "delta_mean_inertia": b.mean_inertia - a.mean_inertia,
        "delta_mean_silhouette": b.mean_silhouette - a.mean_silhouette,
        "delta_n_runs": b.n_runs - a.n_runs,
        "same_best": a.best_run_id == b.best_run_id,
    }


def batch_summarise_clustering_entries(
    groups: Sequence[Sequence[ClusteringAnalysisEntry]],
    cfg: Optional[ClusteringAnalysisConfig] = None,
) -> List[ClusteringAnalysisSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_clustering_entries(g, cfg) for g in groups]
