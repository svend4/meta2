"""
Реестр матчеров с единым интерфейсом.

Каждый матчер принимает два EdgeSignature и возвращает float [0..1].
Регистрация происходит через декоратор @register или register_fn().

Использование:
    from puzzle_reconstruction.matching.matcher_registry import (
        MATCHER_REGISTRY, get_matcher, list_matchers, register
    )

    @register("my_matcher")
    def my_matcher(e_i, e_j):
        return some_score(e_i, e_j)

    score = MATCHER_REGISTRY["my_matcher"](edge_a, edge_b)
    print(list_matchers())
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..models import EdgeSignature

# Тип матчера: (EdgeSignature, EdgeSignature) -> float [0..1]
MatcherFn = Callable[[EdgeSignature, EdgeSignature], float]

# Глобальный реестр
MATCHER_REGISTRY: Dict[str, MatcherFn] = {}


def register(name: str) -> Callable[[MatcherFn], MatcherFn]:
    """Декоратор для регистрации матчера в реестре."""
    def decorator(fn: MatcherFn) -> MatcherFn:
        MATCHER_REGISTRY[name] = fn
        return fn
    return decorator


def register_fn(name: str, fn: MatcherFn) -> None:
    """Регистрация матчера без декоратора."""
    MATCHER_REGISTRY[name] = fn


def get_matcher(name: str) -> Optional[MatcherFn]:
    """Возвращает матчер по имени или None."""
    return MATCHER_REGISTRY.get(name)


def list_matchers() -> List[str]:
    """Список всех зарегистрированных матчеров."""
    return sorted(MATCHER_REGISTRY.keys())


def _safe_score(fn: MatcherFn, e_i: EdgeSignature, e_j: EdgeSignature) -> float:
    """Вызывает матчер с защитой от исключений, возвращает 0.0 при ошибке."""
    try:
        s = fn(e_i, e_j)
        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.0


def compute_scores(
    e_i: EdgeSignature,
    e_j: EdgeSignature,
    matchers: List[str],
) -> Dict[str, float]:
    """
    Вычисляет оценки всех указанных матчеров для пары краёв.

    Args:
        e_i, e_j:  Пара EdgeSignature.
        matchers:  Список имён матчеров из MATCHER_REGISTRY.

    Returns:
        Словарь {matcher_name: score ∈ [0..1]}.
        Если матчер не найден — score = 0.0.
    """
    return {
        name: _safe_score(MATCHER_REGISTRY[name], e_i, e_j)
        if name in MATCHER_REGISTRY else 0.0
        for name in matchers
    }


def weighted_combine(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Взвешенная комбинация оценок матчеров.

    Args:
        scores:  {matcher_name: score}.
        weights: {matcher_name: weight}. Будут нормализованы если сумма ≠ 1.

    Returns:
        float [0..1] — взвешенное среднее.
    """
    total_w = sum(weights.get(k, 0.0) for k in scores)
    if total_w <= 0.0:
        return 0.0
    return sum(
        scores[k] * weights.get(k, 0.0)
        for k in scores
    ) / total_w


# ── Регистрация базовых матчеров при импорте ──────────────────────────────────

def _register_defaults() -> None:
    """Регистрирует стандартные матчеры из существующих модулей."""
    import numpy as np

    # ── CSS (Curvature Scale Space) ───────────────────────────────────────
    try:
        from ..algorithms.fractal.css import css_similarity_mirror

        @register("css")
        def _css(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(css_similarity_mirror(e_i.css_vec, e_j.css_vec))
    except Exception:
        pass

    # ── DTW (Dynamic Time Warping) ────────────────────────────────────────
    try:
        from .dtw import dtw_distance_mirror

        @register("dtw")
        def _dtw(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            dist = dtw_distance_mirror(e_i.virtual_curve, e_j.virtual_curve)
            return float(1.0 / (1.0 + dist))
    except Exception:
        pass

    # ── FD (Fractal Dimension) ────────────────────────────────────────────
    @register("fd")
    def _fd(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
        diff = abs(e_i.fd - e_j.fd)
        return float(1.0 / (1.0 + diff))

    # ── TEXT (OCR text coherence — external score, returns 0 if unavailable)
    @register("text")
    def _text(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
        # Текстовая связность — внешний сигнал, базовое значение 0.0
        # Реальное значение передаётся через pairwise.match_score(text_score=...)
        return 0.0

    # ── ICP (Iterative Closest Point) ────────────────────────────────────
    try:
        from .icp import icp_align

        @register("icp")
        def _icp(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            result = icp_align(e_i.virtual_curve, e_j.virtual_curve)
            dist = getattr(result, "final_error", None) or getattr(result, "error", 1.0)
            return float(1.0 / (1.0 + dist))
    except Exception:
        pass

    # ── Color match ───────────────────────────────────────────────────────
    try:
        from .color_match import color_match_score

        @register("color")
        def _color(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(color_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Texture match ─────────────────────────────────────────────────────
    try:
        from .texture_match import texture_match_score

        @register("texture")
        def _texture(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(texture_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Seam score ────────────────────────────────────────────────────────
    try:
        from .seam_score import seam_score

        @register("seam")
        def _seam(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(seam_score(e_i, e_j))
    except Exception:
        pass

    # ── Geometric match ───────────────────────────────────────────────────
    try:
        from .geometric_match import geometric_match_score

        @register("geometric")
        def _geometric(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(geometric_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Boundary matcher ──────────────────────────────────────────────────
    try:
        from .boundary_matcher import boundary_match_score

        @register("boundary")
        def _boundary(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(boundary_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Affine matcher ────────────────────────────────────────────────────
    try:
        from .affine_matcher import affine_match_score

        @register("affine")
        def _affine(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(affine_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Spectral matcher ──────────────────────────────────────────────────
    try:
        from .spectral_matcher import spectral_match_score

        @register("spectral")
        def _spectral(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(spectral_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Shape matcher (Shape Context) ─────────────────────────────────────
    try:
        from .shape_matcher import shape_match_score

        @register("shape_context")
        def _shape(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(shape_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Patch matcher ─────────────────────────────────────────────────────
    try:
        from .patch_matcher import patch_match_score

        @register("patch")
        def _patch(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(patch_match_score(e_i, e_j))
    except Exception:
        pass

    # ── Feature matcher ───────────────────────────────────────────────────
    try:
        from .feature_match import feature_match_score

        @register("feature")
        def _feature(e_i: EdgeSignature, e_j: EdgeSignature) -> float:
            return float(feature_match_score(e_i, e_j))
    except Exception:
        pass


_register_defaults()
