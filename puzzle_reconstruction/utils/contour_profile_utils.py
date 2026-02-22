"""contour_profile_utils — утилиты для работы с профилями контуров и рёбер.

Предоставляет функции для вычисления, сравнения и преобразования профилей
(последовательностей точек или значений вдоль контура фрагмента).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class ProfileConfig:
    """Параметры вычисления профиля контура."""
    n_samples: int = 64
    smooth_window: int = 3
    normalize: bool = True
    pad_mode: str = "edge"


# ─── Результат сравнения профилей ─────────────────────────────────────────────

@dataclass
class ProfileMatchResult:
    """Результат сопоставления двух профилей."""
    score: float
    offset: int
    distance: float
    method: str = "dtw"
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ProfileMatchResult(score={self.score:.4f}, "
                f"offset={self.offset}, method={self.method!r})")


# ─── Утилиты для работы с профилями ──────────────────────────────────────────

def sample_profile_along_contour(
    contour: np.ndarray,
    n_samples: int = 64,
) -> np.ndarray:
    """Равномерно распределить n_samples точек вдоль контура.

    Parameters
    ----------
    contour:
        Массив точек контура shape (N, 2), float.
    n_samples:
        Количество равномерно распределённых точек.

    Returns
    -------
    np.ndarray shape (n_samples, 2), float64.

    Raises
    ------
    ValueError
        Если contour пуст или n_samples < 1.
    """
    if len(contour) == 0:
        raise ValueError("contour must not be empty")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    pts = np.asarray(contour, dtype=np.float64)
    if len(pts) == 1:
        return np.tile(pts, (n_samples, 1))
    # Кумулятивная длина дуги
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]
    if total == 0.0:
        return np.tile(pts[0:1], (n_samples, 1))
    target = np.linspace(0.0, total, n_samples)
    xs = np.interp(target, cum_lengths, pts[:, 0])
    ys = np.interp(target, cum_lengths, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def contour_curvature(contour: np.ndarray) -> np.ndarray:
    """Вычислить дискретную кривизну вдоль контура.

    Parameters
    ----------
    contour:
        Массив точек shape (N, 2), float.

    Returns
    -------
    np.ndarray shape (N,), float64 — знаковая кривизна в каждой точке.
    """
    pts = np.asarray(contour, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float64)
    # Центральные разности (периодические)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    d1 = pts - prev_pts
    d2 = next_pts - pts
    # Знаковая кривизна через векторное произведение
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denom = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return cross / denom


def smooth_profile(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Сгладить одномерный профиль скользящим средним.

    Parameters
    ----------
    values:
        Одномерный массив значений.
    window:
        Ширина окна (нечётное). Должно быть >= 1.

    Returns
    -------
    np.ndarray той же длины, float64.

    Raises
    ------
    ValueError
        Если window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0 or window == 1:
        return v.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(v, kernel, mode="same")


def normalize_profile(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Нормализовать профиль в диапазон [0, 1].

    Parameters
    ----------
    values:
        Входной одномерный массив.
    eps:
        Минимальный диапазон (защита от деления на ноль).

    Returns
    -------
    np.ndarray float64.
    """
    v = np.asarray(values, dtype=np.float64)
    mn, mx = v.min(), v.max()
    rng = mx - mn
    if rng < eps:
        return np.ones_like(v)
    return (v - mn) / rng


def profile_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Евклидово расстояние между двумя профилями одинаковой длины.

    Parameters
    ----------
    a, b:
        Одномерные массивы одинаковой длины.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        Если длины не совпадают.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"profiles must have the same length: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def profile_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Косинусное сходство между двумя профилями.

    Parameters
    ----------
    a, b:
        Одномерные массивы (любая длина, но одинаковая).
    eps:
        Защита от деления на ноль.

    Returns
    -------
    float в диапазоне [-1, 1].
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_cyclic_offset(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    """Найти лучший циклический сдвиг массива b относительно a.

    Минимизирует L2-расстояние между a и cyclic_shift(b, offset).

    Parameters
    ----------
    a, b:
        Одномерные массивы одинаковой длины.

    Returns
    -------
    (offset, min_distance) — лучший сдвиг и соответствующее расстояние.

    Raises
    ------
    ValueError
        Если длины не совпадают или массивы пусты.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    n = len(a)
    if n == 0:
        raise ValueError("arrays must not be empty")
    best_offset = 0
    best_dist = float("inf")
    for k in range(n):
        shifted = np.roll(b, k)
        dist = float(np.sqrt(np.sum((a - shifted) ** 2)))
        if dist < best_dist:
            best_dist = dist
            best_offset = k
    return best_offset, best_dist


def align_profiles(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Выровнять два профиля по лучшему циклическому сдвигу.

    Parameters
    ----------
    a, b:
        Одномерные массивы одинаковой длины.

    Returns
    -------
    (a_aligned, b_aligned, offset)
    """
    offset, _ = best_cyclic_offset(a, b)
    b_aligned = np.roll(np.asarray(b, dtype=np.float64), offset)
    return np.asarray(a, dtype=np.float64).copy(), b_aligned, offset


def match_profiles(
    a: np.ndarray,
    b: np.ndarray,
    cyclic: bool = False,
    normalize: bool = True,
    cfg: Optional[ProfileConfig] = None,
) -> ProfileMatchResult:
    """Сопоставить два профиля и вернуть оценку сходства.

    Parameters
    ----------
    a, b:
        Одномерные профили одинаковой длины.
    cyclic:
        Если True — перебирать циклические сдвиги.
    normalize:
        Нормализовать профили в [0, 1] перед сравнением.
    cfg:
        Конфигурация (если передана, overrides normalize).

    Returns
    -------
    ProfileMatchResult
    """
    if cfg is not None:
        normalize = cfg.normalize
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if normalize:
        va = normalize_profile(va)
        vb = normalize_profile(vb)
    if cyclic:
        offset, dist = best_cyclic_offset(va, vb)
    else:
        offset = 0
        dist = profile_l2_distance(va, vb)
    n = len(va)
    max_dist = float(np.sqrt(n))  # максимально возможное расстояние для [0,1]-данных
    score = float(max(0.0, 1.0 - dist / (max_dist + 1e-8)))
    return ProfileMatchResult(
        score=score,
        offset=offset,
        distance=dist,
        method="cyclic" if cyclic else "l2",
        params={"normalize": normalize, "n": n},
    )


def batch_match_profiles(
    reference: np.ndarray,
    candidates: List[np.ndarray],
    cyclic: bool = False,
    normalize: bool = True,
) -> List[ProfileMatchResult]:
    """Сопоставить опорный профиль со списком кандидатов.

    Parameters
    ----------
    reference:
        Одномерный опорный профиль.
    candidates:
        Список профилей-кандидатов.
    cyclic, normalize:
        Передаются в match_profiles.

    Returns
    -------
    Список ProfileMatchResult, по одному на кандидата.
    """
    return [
        match_profiles(reference, c, cyclic=cyclic, normalize=normalize)
        for c in candidates
    ]


def top_k_profile_matches(
    results: List[ProfileMatchResult],
    k: int,
) -> List[ProfileMatchResult]:
    """Вернуть top-k результатов по убыванию score.

    Parameters
    ----------
    results:
        Список ProfileMatchResult.
    k:
        Количество лучших результатов.

    Returns
    -------
    Список из не более k результатов.
    """
    return sorted(results, key=lambda r: r.score, reverse=True)[:k]
