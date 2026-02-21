"""Сравнение пар краёв фрагментов пазла.

Модуль вычисляет меры совместимости между двумя EdgeSignature:
DTW-расстояние по виртуальным кривым, косинусное сходство CSS-дескрипторов,
разность фрактальных размерностей и IFS-совместимость.
Финальная оценка — взвешенная сумма каналов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import EdgeSignature, CompatEntry


# ─── CompareConfig ────────────────────────────────────────────────────────────

@dataclass
class CompareConfig:
    """Параметры сравнения краёв.

    Атрибуты:
        w_dtw:   Вес DTW-канала (>= 0).
        w_css:   Вес CSS-канала (>= 0).
        w_fd:    Вес FD-канала (>= 0).
        w_ifs:   Вес IFS-канала (>= 0).
        dtw_band: Ширина полосы Сакое–Чибы для DTW (0 = без ограничений).
        fd_sigma: Сигма для нормализации разности FD (> 0).
    """

    w_dtw: float = 0.4
    w_css: float = 0.3
    w_fd: float = 0.15
    w_ifs: float = 0.15
    dtw_band: int = 0
    fd_sigma: float = 0.5

    def __post_init__(self) -> None:
        for name, val in (
            ("w_dtw", self.w_dtw),
            ("w_css", self.w_css),
            ("w_fd", self.w_fd),
            ("w_ifs", self.w_ifs),
        ):
            if val < 0.0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )
        if self.dtw_band < 0:
            raise ValueError(
                f"dtw_band должен быть >= 0, получено {self.dtw_band}"
            )
        if self.fd_sigma <= 0.0:
            raise ValueError(
                f"fd_sigma должна быть > 0, получено {self.fd_sigma}"
            )

    @property
    def total_weight(self) -> float:
        return self.w_dtw + self.w_css + self.w_fd + self.w_ifs


# ─── EdgeCompareResult ────────────────────────────────────────────────────────

@dataclass
class EdgeCompareResult:
    """Результат сравнения двух краёв.

    Атрибуты:
        edge_id_a:  ID первого края.
        edge_id_b:  ID второго края.
        dtw_dist:   DTW-расстояние (>= 0).
        css_sim:    Косинусное сходство CSS [0, 1].
        fd_diff:    Абсолютная разность FD (>= 0).
        ifs_sim:    IFS-сходство [0, 1].
        score:      Взвешенная итоговая оценка [0, 1].
    """

    edge_id_a: int
    edge_id_b: int
    dtw_dist: float
    css_sim: float
    fd_diff: float
    ifs_sim: float
    score: float

    def __post_init__(self) -> None:
        if self.dtw_dist < 0.0:
            raise ValueError(
                f"dtw_dist должен быть >= 0, получено {self.dtw_dist}"
            )
        if not (0.0 <= self.css_sim <= 1.0):
            raise ValueError(
                f"css_sim должен быть в [0, 1], получено {self.css_sim}"
            )
        if self.fd_diff < 0.0:
            raise ValueError(
                f"fd_diff должен быть >= 0, получено {self.fd_diff}"
            )
        if not (0.0 <= self.ifs_sim <= 1.0):
            raise ValueError(
                f"ifs_sim должен быть в [0, 1], получено {self.ifs_sim}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    @property
    def pair_key(self) -> Tuple[int, int]:
        return (min(self.edge_id_a, self.edge_id_b),
                max(self.edge_id_a, self.edge_id_b))

    @property
    def is_compatible(self) -> bool:
        """True если score >= 0.6."""
        return self.score >= 0.6


# ─── dtw_distance ─────────────────────────────────────────────────────────────

def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    band: int = 0,
) -> float:
    """Вычислить DTW-расстояние между двумя кривыми (N, 2).

    Аргументы:
        a:    Первая кривая (N, 2).
        b:    Вторая кривая (M, 2).
        band: Полоса Сакое–Чибы (0 = без ограничений).

    Возвращает:
        DTW-расстояние (>= 0).

    Исключения:
        ValueError: При некорректной форме массивов.
    """
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(
            f"a должен быть (N, 2), получено {a.shape}"
        )
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(
            f"b должен быть (M, 2), получено {b.shape}"
        )
    if band < 0:
        raise ValueError(f"band должен быть >= 0, получено {band}")

    n, m = len(a), len(b)
    inf = float("inf")
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = 1 if band == 0 else max(1, i - band)
        j_hi = m if band == 0 else min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


# ─── css_similarity ───────────────────────────────────────────────────────────

def css_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """Косинусное сходство двух CSS-дескрипторов.

    Аргументы:
        vec_a: Дескриптор (D,).
        vec_b: Дескриптор (D,).

    Возвращает:
        Сходство в [0, 1].

    Исключения:
        ValueError: При несовпадении размерностей или неверной форме.
    """
    va = np.asarray(vec_a, dtype=float).ravel()
    vb = np.asarray(vec_b, dtype=float).ravel()
    if va.shape != vb.shape:
        raise ValueError(
            f"Размерности не совпадают: {va.shape} != {vb.shape}"
        )
    na = np.linalg.norm(va) + 1e-12
    nb = np.linalg.norm(vb) + 1e-12
    return float(np.clip(np.dot(va / na, vb / nb), 0.0, 1.0))


# ─── fd_score ─────────────────────────────────────────────────────────────────

def fd_score(
    fd_a: float,
    fd_b: float,
    sigma: float = 0.5,
) -> float:
    """Преобразовать разность FD в оценку сходства [0, 1] через Гауссову функцию.

    Аргументы:
        fd_a:  FD первого края.
        fd_b:  FD второго края.
        sigma: Параметр масштаба (> 0).

    Возвращает:
        Оценка exp(-(fd_a - fd_b)² / (2 σ²)) ∈ (0, 1].

    Исключения:
        ValueError: Если sigma <= 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma должна быть > 0, получено {sigma}")
    diff = fd_a - fd_b
    return float(np.exp(-(diff ** 2) / (2.0 * sigma ** 2)))


# ─── ifs_similarity ───────────────────────────────────────────────────────────

def ifs_similarity(
    coeffs_a: np.ndarray,
    coeffs_b: np.ndarray,
) -> float:
    """Косинусное сходство IFS-коэффициентов.

    Аргументы:
        coeffs_a: Коэффициенты (M,).
        coeffs_b: Коэффициенты (M,).

    Возвращает:
        Сходство в [0, 1].
    """
    va = np.asarray(coeffs_a, dtype=float).ravel()
    vb = np.asarray(coeffs_b, dtype=float).ravel()
    # Выровнять до минимальной длины
    min_len = min(len(va), len(vb))
    if min_len == 0:
        return 0.0
    va, vb = va[:min_len], vb[:min_len]
    na = np.linalg.norm(va) + 1e-12
    nb = np.linalg.norm(vb) + 1e-12
    return float(np.clip(np.dot(va / na, vb / nb), 0.0, 1.0))


# ─── compare_edges ────────────────────────────────────────────────────────────

def compare_edges(
    edge_a: EdgeSignature,
    edge_b: EdgeSignature,
    cfg: Optional[CompareConfig] = None,
) -> EdgeCompareResult:
    """Сравнить два края и вернуть детализированный результат.

    Аргументы:
        edge_a: Первый EdgeSignature.
        edge_b: Второй EdgeSignature.
        cfg:    Параметры (None → CompareConfig()).

    Возвращает:
        EdgeCompareResult.
    """
    if cfg is None:
        cfg = CompareConfig()

    # DTW
    raw_dtw = dtw_distance(edge_a.virtual_curve, edge_b.virtual_curve,
                           band=cfg.dtw_band)
    dtw_norm = raw_dtw / (len(edge_a.virtual_curve) + 1e-12)

    # CSS
    css_sim = css_similarity(edge_a.css_vec, edge_b.css_vec)

    # FD
    fd_diff = abs(edge_a.fd - edge_b.fd)
    fd_sim = fd_score(edge_a.fd, edge_b.fd, cfg.fd_sigma)

    # IFS
    ifs_sim = ifs_similarity(edge_a.ifs_coeffs, edge_b.ifs_coeffs)

    # DTW → сходство: экспоненциальное затухание
    dtw_sim = float(np.exp(-dtw_norm))

    # Взвешенная сумма
    tw = cfg.total_weight + 1e-12
    score = (
        cfg.w_dtw * dtw_sim
        + cfg.w_css * css_sim
        + cfg.w_fd * fd_sim
        + cfg.w_ifs * ifs_sim
    ) / tw

    score = float(np.clip(score, 0.0, 1.0))

    return EdgeCompareResult(
        edge_id_a=edge_a.edge_id,
        edge_id_b=edge_b.edge_id,
        dtw_dist=raw_dtw,
        css_sim=css_sim,
        fd_diff=fd_diff,
        ifs_sim=ifs_sim,
        score=score,
    )


# ─── build_compat_matrix ──────────────────────────────────────────────────────

def build_compat_matrix(
    edges: List[EdgeSignature],
    cfg: Optional[CompareConfig] = None,
) -> np.ndarray:
    """Построить матрицу совместимости (N × N).

    Аргументы:
        edges: Список EdgeSignature.
        cfg:   Параметры.

    Возвращает:
        Матрица float32 (N, N), compat[i, j] = score(edges[i], edges[j]).
    """
    if cfg is None:
        cfg = CompareConfig()
    n = len(edges)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                r = compare_edges(edges[i], edges[j], cfg)
                mat[i, j] = r.score
                mat[j, i] = r.score
    return mat


# ─── top_k_matches ────────────────────────────────────────────────────────────

def top_k_matches(
    query: EdgeSignature,
    candidates: List[EdgeSignature],
    k: int = 5,
    cfg: Optional[CompareConfig] = None,
) -> List[EdgeCompareResult]:
    """Найти k лучших совпадений для query среди candidates.

    Аргументы:
        query:      Запросный край.
        candidates: Список кандидатов.
        k:          Число лучших результатов (>= 1).
        cfg:        Параметры.

    Возвращает:
        Список EdgeCompareResult, отсортированный по убыванию score.

    Исключения:
        ValueError: Если k < 1 или candidates пуст.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    if not candidates:
        raise ValueError("candidates не должен быть пустым")
    if cfg is None:
        cfg = CompareConfig()

    results = [compare_edges(query, c, cfg) for c in candidates]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
