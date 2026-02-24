"""
Попарная оценка совместимости двух краёв фрагментов.

Базовая формула (дефолтные веса):
    MatchScore = 0.35·CSS + 0.30·DTW + 0.20·FD + 0.15·TEXT

Конфигурируемая формула (через MatchingConfig):
    MatchScore = Σ weight_i · matcher_i(e_i, e_j)
    где активные матчеры и их веса задаются в MatchingConfig.

Backward-compatible: вызов match_score(e_i, e_j) без cfg даёт тот же
результат, что и старая реализация.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional

from ..models import EdgeSignature, CompatEntry
from .dtw import dtw_distance_mirror
from ..algorithms.fractal.css import css_similarity_mirror

if TYPE_CHECKING:
    from ..config import MatchingConfig


# Дефолтные веса (для backward-compatibility без cfg)
_DEFAULT_WEIGHTS = {"css": 0.35, "dtw": 0.30, "fd": 0.20, "text": 0.15}
_DEFAULT_MATCHERS = ["css", "dtw", "fd", "text"]


def match_score(
    e_i: EdgeSignature,
    e_j: EdgeSignature,
    text_score: float = 0.0,
    cfg: Optional["MatchingConfig"] = None,
) -> CompatEntry:
    """
    Вычисляет полную оценку совместимости двух краёв.

    Args:
        e_i, e_j:    Края двух разных фрагментов.
        text_score:  Внешняя оценка связности текста (0..1), от OCR-модуля.
        cfg:         MatchingConfig для конфигурируемых весов и матчеров.
                     None → используются дефолтные веса (backward-compatible).

    Returns:
        CompatEntry с полной информацией о совместимости.
    """
    # Всегда вычисляем базовые компоненты (нужны для CompatEntry полей)
    css_sim = css_similarity_mirror(e_i.css_vec, e_j.css_vec)
    dtw_dist = dtw_distance_mirror(e_i.virtual_curve, e_j.virtual_curve)
    dtw_score = 1.0 / (1.0 + dtw_dist)
    fd_diff = abs(e_i.fd - e_j.fd)
    fd_score = 1.0 / (1.0 + fd_diff)
    ifs_dist = _ifs_distance_norm(e_i.ifs_coeffs, e_j.ifs_coeffs)
    ifs_score = 1.0 / (1.0 + ifs_dist)

    if cfg is None:
        # Backward-compatible путь: жёсткие веса как раньше
        score = (
            _DEFAULT_WEIGHTS["css"] * css_sim
            + _DEFAULT_WEIGHTS["dtw"] * (0.7 * dtw_score + 0.3 * ifs_score)
            + _DEFAULT_WEIGHTS["fd"] * fd_score
            + _DEFAULT_WEIGHTS["text"] * text_score
        )
    else:
        # Конфигурируемый путь: веса и матчеры из cfg
        weights = cfg.matcher_weights
        active = cfg.active_matchers
        combine = cfg.combine_method

        # Базовые оценки для стандартных матчеров
        base_scores = {
            "css": float(css_sim),
            "dtw": float(0.7 * dtw_score + 0.3 * ifs_score),
            "fd": float(fd_score),
            "text": float(text_score),
        }

        # Расширенные матчеры из реестра (если запрошены)
        extra_names = [m for m in active if m not in base_scores]
        if extra_names:
            try:
                from .matcher_registry import compute_scores
                extra = compute_scores(e_i, e_j, extra_names)
                base_scores.update(extra)
            except Exception:
                pass

        # Комбинирование
        if combine == "weighted":
            score = _weighted(base_scores, weights, active)
        elif combine == "rank":
            score = _rank_combine(base_scores, active)
        elif combine == "min":
            score = min(base_scores.get(m, 0.0) for m in active)
        elif combine == "max":
            score = max(base_scores.get(m, 0.0) for m in active)
        else:
            score = _weighted(base_scores, weights, active)

    # Штраф за сильно разные длины краёв
    len_ratio = min(e_i.length, e_j.length) / (max(e_i.length, e_j.length) + 1e-5)
    if len_ratio < 0.5:
        score *= len_ratio

    score = float(np.clip(score, 0.0, 1.0))

    return CompatEntry(
        edge_i=e_i,
        edge_j=e_j,
        score=score,
        dtw_dist=float(dtw_dist),
        css_sim=float(css_sim),
        fd_diff=float(fd_diff),
        text_score=float(text_score),
    )


# ── Вспомогательные функции комбинирования ────────────────────────────────────

def _weighted(
    scores: dict,
    weights: dict,
    active: list,
) -> float:
    """Взвешенное среднее активных матчеров."""
    total_w = sum(weights.get(m, 0.0) for m in active if m in scores)
    if total_w <= 0.0:
        return 0.0
    return sum(
        scores[m] * weights.get(m, 0.0)
        for m in active
        if m in scores
    ) / total_w


def _rank_combine(scores: dict, active: list) -> float:
    """Ранговое слияние: нормализованный средний ранг."""
    vals = [scores.get(m, 0.0) for m in active]
    if not vals:
        return 0.0
    # Нормализуем ранги в [0..1]
    n = len(vals)
    ranked = sorted(range(n), key=lambda i: vals[i])
    rank_scores = [0.0] * n
    for rank, idx in enumerate(ranked):
        rank_scores[idx] = rank / max(n - 1, 1)
    return float(np.mean(rank_scores))


# ── Внутренние утилиты ────────────────────────────────────────────────────────

def _ifs_distance_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Нормализованное расстояние между IFS-коэффициентами."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = a[:n] - b[:n]
    return float(np.linalg.norm(diff)) / (np.sqrt(n) + 1e-10)
