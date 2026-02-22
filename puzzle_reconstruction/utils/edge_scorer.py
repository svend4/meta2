"""Оценка совместимости краёв фрагментов.

Модуль предоставляет функции для вычисления различных метрик
совместимости пары краёв: пространственное перекрытие, сходство
кривизны, совместимость длин, близость концевых точек и агрегированная
оценка.

Публичный API:
    EdgeScoreConfig   — параметры оценки
    EdgeScoreResult   — результат сравнения одной пары краёв
    score_edge_overlap    — оценка пространственного перекрытия
    score_edge_curvature  — оценка сходства профилей кривизны
    score_edge_length     — оценка совместимости длин
    score_edge_endpoints  — оценка близости концевых точек
    aggregate_edge_scores — взвешенная агрегация составных оценок
    rank_edge_pairs       — ранжирование пар по убыванию оценки
    batch_score_edges     — пакетное вычисление оценок
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── EdgeScoreConfig ──────────────────────────────────────────────────────────

@dataclass
class EdgeScoreConfig:
    """Параметры оценки совместимости краёв.

    Атрибуты:
        n_samples:       Число точек для ресемплинга кривых (>= 2).
        length_tol:      Допустимое относительное отличие длин (>= 0).
        endpoint_sigma:  Масштаб (σ) для гауссовой оценки близости
                         концевых точек (> 0).
        weights:         Словарь весов составных оценок
                         (overlap, curvature, length, endpoints).
                         Значения >= 0; нормализуются автоматически.
    """

    n_samples: int = 64
    length_tol: float = 0.5
    endpoint_sigma: float = 10.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "overlap": 0.4,
            "curvature": 0.3,
            "length": 0.15,
            "endpoints": 0.15,
        }
    )

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(
                f"n_samples должен быть >= 2, получено {self.n_samples}"
            )
        if self.length_tol < 0:
            raise ValueError(
                f"length_tol должен быть >= 0, получено {self.length_tol}"
            )
        if self.endpoint_sigma <= 0:
            raise ValueError(
                f"endpoint_sigma должен быть > 0, получено {self.endpoint_sigma}"
            )
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(
                    f"weight[{k!r}] должен быть >= 0, получено {v}"
                )

    @property
    def normalized_weights(self) -> Dict[str, float]:
        """Нормализованные веса (сумма = 1, если хотя бы один > 0)."""
        total = sum(self.weights.values())
        if total <= 0:
            n = len(self.weights)
            return {k: 1.0 / n for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}


# ─── EdgeScoreResult ──────────────────────────────────────────────────────────

@dataclass
class EdgeScoreResult:
    """Результат сравнения одной пары краёв.

    Атрибуты:
        overlap:    Оценка пространственного перекрытия [0, 1].
        curvature:  Оценка сходства кривизны [0, 1].
        length:     Оценка совместимости длин [0, 1].
        endpoints:  Оценка близости концевых точек [0, 1].
        total:      Взвешенная агрегированная оценка [0, 1].
    """

    overlap: float = 0.0
    curvature: float = 0.0
    length: float = 0.0
    endpoints: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "overlap": self.overlap,
            "curvature": self.curvature,
            "length": self.length,
            "endpoints": self.endpoints,
            "total": self.total,
        }


# ─── score_edge_overlap ───────────────────────────────────────────────────────

def score_edge_overlap(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    cfg: Optional[EdgeScoreConfig] = None,
) -> float:
    """Оценивает пространственное перекрытие двух краёв.

    Кривые ресемплируются до одинакового числа точек, затем
    вычисляется нормализованная обратная L2-разность.

    Аргументы:
        curve_a: (N, 2) массив точек первого края.
        curve_b: (M, 2) массив точек второго края.
        cfg:     Параметры (None → EdgeScoreConfig()).

    Возвращает:
        score ∈ [0, 1], где 1 = идеальное совпадение.

    Исключения:
        ValueError: Если входной массив не (*, 2).
    """
    if cfg is None:
        cfg = EdgeScoreConfig()
    a = _validate_curve(curve_a, "curve_a")
    b = _validate_curve(curve_b, "curve_b")

    if len(a) == 0 or len(b) == 0:
        return 0.0

    a_r = _resample(a, cfg.n_samples)
    b_r = _resample(b[::-1], cfg.n_samples)  # зеркалируем b

    diffs = np.linalg.norm(a_r - b_r, axis=1)
    mean_diff = float(diffs.mean())

    # Нормализуем: характерный масштаб = диаметр описывающего прямоугольника
    scale = _bounding_scale(np.vstack([a, b]))
    if scale < 1e-9:
        return 1.0 if mean_diff < 1e-9 else 0.0

    return float(np.exp(-mean_diff / scale))


# ─── score_edge_curvature ─────────────────────────────────────────────────────

def score_edge_curvature(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    cfg: Optional[EdgeScoreConfig] = None,
) -> float:
    """Оценивает сходство профилей кривизны двух краёв.

    Кривизна вычисляется как угловая скорость изменения направления
    касательного вектора. Сравниваются нормализованные профили.

    Аргументы:
        curve_a: (N, 2) массив точек.
        curve_b: (M, 2) массив точек.
        cfg:     Параметры (None → EdgeScoreConfig()).

    Возвращает:
        score ∈ [0, 1].
    """
    if cfg is None:
        cfg = EdgeScoreConfig()
    a = _validate_curve(curve_a, "curve_a")
    b = _validate_curve(curve_b, "curve_b")

    if len(a) < 3 or len(b) < 3:
        return 0.5  # недостаточно точек для оценки кривизны

    ka = _curvature_profile(_resample(a, cfg.n_samples))
    kb = _curvature_profile(_resample(b[::-1], cfg.n_samples))

    # Нормализуем профили
    ka_n = _normalize_profile(ka)
    kb_n = _normalize_profile(kb)

    corr = float(np.dot(ka_n, kb_n))
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ─── score_edge_length ────────────────────────────────────────────────────────

def score_edge_length(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    cfg: Optional[EdgeScoreConfig] = None,
) -> float:
    """Оценивает совместимость длин двух краёв.

    Аргументы:
        curve_a: (N, 2) массив точек.
        curve_b: (M, 2) массив точек.
        cfg:     Параметры (None → EdgeScoreConfig()).

    Возвращает:
        score ∈ [0, 1], где 1 = одинаковые длины.
    """
    if cfg is None:
        cfg = EdgeScoreConfig()
    a = _validate_curve(curve_a, "curve_a")
    b = _validate_curve(curve_b, "curve_b")

    len_a = _arc_length(a)
    len_b = _arc_length(b)

    if len_a < 1e-9 and len_b < 1e-9:
        return 1.0
    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0

    ratio = min(len_a, len_b) / max(len_a, len_b)
    # ratio ∈ (0, 1]: 1 = совпадение
    if cfg.length_tol <= 0:
        return float(ratio)

    # Нормализуем: оценка = 1 если |ratio-1| <= length_tol
    deviation = abs(1.0 - ratio)
    score = float(np.exp(-deviation / (cfg.length_tol + 1e-9)))
    return float(np.clip(score, 0.0, 1.0))


# ─── score_edge_endpoints ─────────────────────────────────────────────────────

def score_edge_endpoints(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    cfg: Optional[EdgeScoreConfig] = None,
) -> float:
    """Оценивает близость концевых точек пары краёв.

    Проверяет, что начало curve_a близко к концу curve_b и наоборот
    (ориентированное соединение).

    Аргументы:
        curve_a: (N, 2) массив точек.
        curve_b: (M, 2) массив точек.
        cfg:     Параметры (None → EdgeScoreConfig()).

    Возвращает:
        score ∈ [0, 1].
    """
    if cfg is None:
        cfg = EdgeScoreConfig()
    a = _validate_curve(curve_a, "curve_a")
    b = _validate_curve(curve_b, "curve_b")

    if len(a) == 0 or len(b) == 0:
        return 0.0

    # Стыки: a[0] ↔ b[-1] и a[-1] ↔ b[0]
    d_start = float(np.linalg.norm(a[0] - b[-1]))
    d_end = float(np.linalg.norm(a[-1] - b[0]))
    d_mean = (d_start + d_end) / 2.0

    sigma = cfg.endpoint_sigma
    return float(np.exp(-d_mean / (sigma + 1e-9)))


# ─── aggregate_edge_scores ────────────────────────────────────────────────────

def aggregate_edge_scores(
    overlap: float,
    curvature: float,
    length: float,
    endpoints: float,
    cfg: Optional[EdgeScoreConfig] = None,
) -> float:
    """Вычисляет взвешенную агрегированную оценку.

    Аргументы:
        overlap:    Оценка перекрытия ∈ [0, 1].
        curvature:  Оценка кривизны ∈ [0, 1].
        length:     Оценка длины ∈ [0, 1].
        endpoints:  Оценка концевых точек ∈ [0, 1].
        cfg:        Параметры (None → EdgeScoreConfig()).

    Возвращает:
        Взвешенная сумма ∈ [0, 1].
    """
    if cfg is None:
        cfg = EdgeScoreConfig()
    w = cfg.normalized_weights
    total = (
        w.get("overlap", 0.0) * overlap
        + w.get("curvature", 0.0) * curvature
        + w.get("length", 0.0) * length
        + w.get("endpoints", 0.0) * endpoints
    )
    return float(np.clip(total, 0.0, 1.0))


# ─── rank_edge_pairs ──────────────────────────────────────────────────────────

def rank_edge_pairs(
    results: List[Tuple[int, int, EdgeScoreResult]],
) -> List[Tuple[int, int, EdgeScoreResult]]:
    """Ранжирует пары краёв по убыванию суммарной оценки.

    Аргументы:
        results: Список (id_a, id_b, EdgeScoreResult).

    Возвращает:
        Тот же список, отсортированный по result.total убыванию.
    """
    return sorted(results, key=lambda x: x[2].total, reverse=True)


# ─── batch_score_edges ────────────────────────────────────────────────────────

def batch_score_edges(
    curves_a: Sequence[np.ndarray],
    curves_b: Sequence[np.ndarray],
    cfg: Optional[EdgeScoreConfig] = None,
) -> List[EdgeScoreResult]:
    """Вычисляет оценки для нескольких пар краёв.

    Аргументы:
        curves_a: Список кривых первого края (каждая (N_i, 2)).
        curves_b: Список кривых второго края (каждая (M_i, 2)).
        cfg:      Параметры (None → EdgeScoreConfig()).

    Возвращает:
        Список EdgeScoreResult для каждой пары.

    Исключения:
        ValueError: Если len(curves_a) != len(curves_b).
    """
    if len(curves_a) != len(curves_b):
        raise ValueError(
            f"len(curves_a)={len(curves_a)} != len(curves_b)={len(curves_b)}"
        )
    if cfg is None:
        cfg = EdgeScoreConfig()

    results = []
    for a, b in zip(curves_a, curves_b):
        ov = score_edge_overlap(a, b, cfg)
        cu = score_edge_curvature(a, b, cfg)
        le = score_edge_length(a, b, cfg)
        ep = score_edge_endpoints(a, b, cfg)
        total = aggregate_edge_scores(ov, cu, le, ep, cfg)
        results.append(EdgeScoreResult(
            overlap=ov,
            curvature=cu,
            length=le,
            endpoints=ep,
            total=total,
        ))
    return results


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _validate_curve(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim == 1 and pts.shape[0] == 0:
        return pts.reshape(0, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"{name} должен иметь форму (N, 2), получено {pts.shape}"
        )
    return pts


def _resample(curve: np.ndarray, n: int) -> np.ndarray:
    """Ресемплирует кривую до n равномерных точек."""
    if len(curve) == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if len(curve) == 1:
        return np.tile(curve[0], (n, 1))

    # Вычисляем накопленные длины
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    if total < 1e-12:
        return np.tile(curve[0], (n, 1))

    target = np.linspace(0.0, total, n)
    x_new = np.interp(target, cumlen, curve[:, 0])
    y_new = np.interp(target, cumlen, curve[:, 1])
    return np.stack([x_new, y_new], axis=1)


def _arc_length(curve: np.ndarray) -> float:
    if len(curve) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def _bounding_scale(pts: np.ndarray) -> float:
    if len(pts) == 0:
        return 1.0
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    return max(diag, 1.0)


def _curvature_profile(curve: np.ndarray) -> np.ndarray:
    """Вычисляет профиль кривизны как угол поворота касательной."""
    if len(curve) < 3:
        return np.zeros(max(0, len(curve) - 2))
    tangents = np.diff(curve, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents_n = tangents / norms
    # Угол между последовательными касательными
    dots = np.clip(
        (tangents_n[:-1] * tangents_n[1:]).sum(axis=1), -1.0, 1.0
    )
    return np.arccos(dots)


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(profile)
    if norm < 1e-12:
        n = len(profile)
        return np.full(n, 1.0 / math.sqrt(max(1, n)))
    return profile / norm
