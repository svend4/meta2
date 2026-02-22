"""Утилиты оценки качества фрагментов и изображений.

Provides lightweight dataclasses and helper functions for analysing
quality assessment results: blur, noise, contrast, completeness scores,
filtering by quality, and batch summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QualityScoreConfig:
    """Configuration for quality score analysis."""
    min_overall: float = 0.5
    min_blur: float = 0.0
    min_noise: float = 0.0
    min_contrast: float = 0.0
    min_completeness: float = 0.0

    def __post_init__(self) -> None:
        for name in ("min_overall", "min_blur", "min_noise",
                     "min_contrast", "min_completeness"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass
class QualityScoreEntry:
    """Score record for a single image/fragment quality assessment."""
    image_id: int
    blur_score: float
    noise_score: float
    contrast_score: float
    completeness: float
    overall: float
    is_acceptable: bool
    meta: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"QualityScoreEntry(id={self.image_id}, "
            f"overall={self.overall:.3f}, "
            f"acceptable={self.is_acceptable})"
        )


@dataclass
class QualitySummary:
    """Summary of quality scores across multiple fragments."""
    entries: List[QualityScoreEntry]
    n_total: int
    n_acceptable: int
    n_rejected: int
    mean_overall: float
    mean_blur: float
    mean_noise: float
    mean_contrast: float
    mean_completeness: float

    def __repr__(self) -> str:
        return (
            f"QualitySummary(n={self.n_total}, "
            f"acceptable={self.n_acceptable}, "
            f"mean_overall={self.mean_overall:.3f})"
        )


def make_quality_entry(
    image_id: int,
    blur_score: float,
    noise_score: float,
    contrast_score: float,
    completeness: float,
    overall: float,
    cfg: Optional[QualityScoreConfig] = None,
    meta: Optional[Dict] = None,
) -> QualityScoreEntry:
    """Create a single quality score entry."""
    cfg = cfg or QualityScoreConfig()
    is_acceptable = overall >= cfg.min_overall
    return QualityScoreEntry(
        image_id=image_id,
        blur_score=blur_score,
        noise_score=noise_score,
        contrast_score=contrast_score,
        completeness=completeness,
        overall=overall,
        is_acceptable=is_acceptable,
        meta=meta or {},
    )


def entries_from_reports(
    reports: List[Dict],
    cfg: Optional[QualityScoreConfig] = None,
) -> List[QualityScoreEntry]:
    """Convert a list of report dicts to QualityScoreEntry list.

    Expected keys: ``image_id``, ``blur_score``, ``noise_score``,
    ``contrast_score``, ``completeness``, ``overall``.
    """
    result = []
    for i, rep in enumerate(reports):
        entry = make_quality_entry(
            image_id=int(rep.get("image_id", i)),
            blur_score=float(rep.get("blur_score", 0.0)),
            noise_score=float(rep.get("noise_score", 0.0)),
            contrast_score=float(rep.get("contrast_score", 0.0)),
            completeness=float(rep.get("completeness", 0.0)),
            overall=float(rep.get("overall", 0.0)),
            cfg=cfg,
            meta={k: v for k, v in rep.items()
                  if k not in ("image_id", "blur_score", "noise_score",
                               "contrast_score", "completeness", "overall")},
        )
        result.append(entry)
    return result


def summarise_quality(
    entries: List[QualityScoreEntry],
) -> QualitySummary:
    """Compute a summary from a list of quality entries."""
    if not entries:
        return QualitySummary(
            entries=entries, n_total=0, n_acceptable=0, n_rejected=0,
            mean_overall=0.0, mean_blur=0.0, mean_noise=0.0,
            mean_contrast=0.0, mean_completeness=0.0,
        )
    n = len(entries)
    n_acc = sum(1 for e in entries if e.is_acceptable)
    return QualitySummary(
        entries=entries,
        n_total=n,
        n_acceptable=n_acc,
        n_rejected=n - n_acc,
        mean_overall=sum(e.overall for e in entries) / n,
        mean_blur=sum(e.blur_score for e in entries) / n,
        mean_noise=sum(e.noise_score for e in entries) / n,
        mean_contrast=sum(e.contrast_score for e in entries) / n,
        mean_completeness=sum(e.completeness for e in entries) / n,
    )


def filter_acceptable(
    entries: List[QualityScoreEntry],
) -> List[QualityScoreEntry]:
    """Return only acceptable entries."""
    return [e for e in entries if e.is_acceptable]


def filter_rejected(
    entries: List[QualityScoreEntry],
) -> List[QualityScoreEntry]:
    """Return only rejected entries."""
    return [e for e in entries if not e.is_acceptable]


def filter_by_overall(
    entries: List[QualityScoreEntry],
    min_overall: float = 0.5,
) -> List[QualityScoreEntry]:
    """Keep entries where overall >= min_overall."""
    return [e for e in entries if e.overall >= min_overall]


def filter_by_blur(
    entries: List[QualityScoreEntry],
    min_blur: float = 0.0,
) -> List[QualityScoreEntry]:
    """Keep entries where blur_score >= min_blur."""
    return [e for e in entries if e.blur_score >= min_blur]


def top_k_quality_entries(
    entries: List[QualityScoreEntry],
    k: int,
) -> List[QualityScoreEntry]:
    """Return top-k entries by overall score (descending)."""
    sorted_entries = sorted(entries, key=lambda e: e.overall, reverse=True)
    return sorted_entries[:max(0, k)]


def quality_score_stats(entries: List[QualityScoreEntry]) -> Dict:
    """Compute basic statistics over overall scores."""
    if not entries:
        return {
            "count": 0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0,
            "n_acceptable": 0, "n_rejected": 0,
        }
    overalls = [e.overall for e in entries]
    n = len(overalls)
    mean_o = sum(overalls) / n
    var = sum((o - mean_o) ** 2 for o in overalls) / n
    std_o = var ** 0.5
    n_acc = sum(1 for e in entries if e.is_acceptable)
    return {
        "count": n,
        "mean": mean_o,
        "std": std_o,
        "min": min(overalls),
        "max": max(overalls),
        "n_acceptable": n_acc,
        "n_rejected": n - n_acc,
    }


def compare_quality(
    summary_a: QualitySummary,
    summary_b: QualitySummary,
) -> Dict:
    """Compare two quality summaries."""
    return {
        "n_total_delta": summary_a.n_total - summary_b.n_total,
        "n_acceptable_delta": summary_a.n_acceptable - summary_b.n_acceptable,
        "mean_overall_delta": summary_a.mean_overall - summary_b.mean_overall,
        "mean_blur_delta": summary_a.mean_blur - summary_b.mean_blur,
        "mean_contrast_delta": summary_a.mean_contrast - summary_b.mean_contrast,
    }


def batch_summarise_quality(
    report_lists: List[List[Dict]],
    cfg: Optional[QualityScoreConfig] = None,
) -> List[QualitySummary]:
    """Summarise multiple quality report batches at once."""
    return [
        summarise_quality(entries_from_reports(reports, cfg))
        for reports in report_lists
    ]
