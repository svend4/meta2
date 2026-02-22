"""
Профилирование краёв фрагментов документов.

Предоставляет инструменты для вычисления числовых профилей краёв:
яркостного, градиентного и разностного (diff), а также агрегацию
нескольких профилей и их нормализацию.

Классы:
    ProfileConfig   — параметры профилирования
    EdgeProfile     — профиль одного края

Функции:
    compute_brightness_profile  — профиль яркости вдоль края
    compute_gradient_profile    — профиль градиента (Собель) вдоль края
    compute_diff_profile        — профиль разности соседних пикселей
    normalize_profile           — нормализация профиля в [0, 1]
    aggregate_profiles          — взвешенная комбинация нескольких профилей
    compare_profiles            — сходство двух профилей (1 - нормированная L2)
    batch_profile_edges         — профилирование списка краёв
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── ProfileConfig ────────────────────────────────────────────────────────────

_VALID_PROFILE_TYPES = frozenset({"brightness", "gradient", "diff"})


@dataclass
class ProfileConfig:
    """Параметры профилирования края.

    Attributes:
        n_samples:    Число точек профиля (>= 2).
        profile_type: Тип профиля: 'brightness', 'gradient' или 'diff'.
        normalize:    Нормализовать профиль в [0, 1].
        strip_width:  Ширина полосы вдоль края (пикселей, >= 1).
    """
    n_samples:    int  = 32
    profile_type: str  = "brightness"
    normalize:    bool = True
    strip_width:  int  = 4

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(
                f"n_samples must be >= 2, got {self.n_samples}"
            )
        if self.profile_type not in _VALID_PROFILE_TYPES:
            raise ValueError(
                f"Unknown profile_type {self.profile_type!r}. "
                f"Choose one of {sorted(_VALID_PROFILE_TYPES)}."
            )
        if self.strip_width < 1:
            raise ValueError(
                f"strip_width must be >= 1, got {self.strip_width}"
            )


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

@dataclass
class EdgeProfile:
    """Профиль одного края фрагмента.

    Attributes:
        profile:      Числовой профиль (float64, длина = n_samples).
        edge_id:      Идентификатор края (>= 0).
        profile_type: Тип профиля.
        n_samples:    Длина профиля.
        params:       Дополнительные параметры.
    """
    profile:      np.ndarray
    edge_id:      int
    profile_type: str
    n_samples:    int
    params:       Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_id < 0:
            raise ValueError(
                f"edge_id must be >= 0, got {self.edge_id}"
            )
        if len(self.profile) != self.n_samples:
            raise ValueError(
                f"profile length {len(self.profile)} != n_samples {self.n_samples}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EdgeProfile(edge_id={self.edge_id}, type={self.profile_type!r}, "
            f"n={self.n_samples}, min={self.profile.min():.3f}, "
            f"max={self.profile.max():.3f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def compute_brightness_profile(
    strip: np.ndarray,
    n_samples: int = 32,
    axis: int = 1,
) -> np.ndarray:
    """Профиль средней яркости вдоль края.

    Args:
        strip:     2-D grayscale массив (полоса вдоль края).
        n_samples: Число точек профиля (>= 2).
        axis:      Ось вдоль которой берётся среднее (0 или 1).

    Returns:
        Массив float64 длины n_samples.

    Raises:
        ValueError: Если strip не 2-D или n_samples < 2.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    strip = np.asarray(strip, dtype=np.float64)
    if strip.ndim != 2:
        raise ValueError(f"strip must be 2-D, got ndim={strip.ndim}")
    if strip.size == 0:
        return np.zeros(n_samples, dtype=np.float64)

    collapsed = strip.mean(axis=axis)
    indices = np.round(
        np.linspace(0, len(collapsed) - 1, n_samples)
    ).astype(int)
    return collapsed[indices].astype(np.float64)


def compute_gradient_profile(
    strip: np.ndarray,
    n_samples: int = 32,
    axis: int = 1,
) -> np.ndarray:
    """Профиль магнитуды градиента вдоль края (через np.gradient).

    Args:
        strip:     2-D grayscale массив.
        n_samples: Число точек профиля (>= 2).
        axis:      Ось вдоль которой берётся среднее.

    Returns:
        Массив float64 длины n_samples (неотрицательный).

    Raises:
        ValueError: Если strip не 2-D или n_samples < 2.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    strip = np.asarray(strip, dtype=np.float64)
    if strip.ndim != 2:
        raise ValueError(f"strip must be 2-D, got ndim={strip.ndim}")
    if strip.size == 0:
        return np.zeros(n_samples, dtype=np.float64)

    gy, gx = np.gradient(strip)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    collapsed = magnitude.mean(axis=axis)
    indices = np.round(
        np.linspace(0, len(collapsed) - 1, n_samples)
    ).astype(int)
    return collapsed[indices].astype(np.float64)


def compute_diff_profile(
    strip: np.ndarray,
    n_samples: int = 32,
    axis: int = 1,
) -> np.ndarray:
    """Профиль абсолютных разностей соседних пикселей вдоль края.

    Args:
        strip:     2-D grayscale массив.
        n_samples: Число точек профиля (>= 2).
        axis:      Ось вдоль которой берётся среднее.

    Returns:
        Массив float64 длины n_samples (>= 0).

    Raises:
        ValueError: Если strip не 2-D или n_samples < 2.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    strip = np.asarray(strip, dtype=np.float64)
    if strip.ndim != 2:
        raise ValueError(f"strip must be 2-D, got ndim={strip.ndim}")
    if strip.shape[1] < 2:
        return np.zeros(n_samples, dtype=np.float64)

    diffs = np.abs(np.diff(strip, axis=1))
    collapsed = diffs.mean(axis=axis - 1 if axis == 1 else axis)
    if len(collapsed) < 1:
        return np.zeros(n_samples, dtype=np.float64)
    indices = np.round(
        np.linspace(0, len(collapsed) - 1, n_samples)
    ).astype(int)
    return collapsed[indices].astype(np.float64)


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Нормализовать профиль в [0, 1] по min-max.

    Args:
        profile: 1-D массив float.

    Returns:
        Нормализованный массив float64 той же длины.
        Если все значения одинаковы — нулевой массив.

    Raises:
        ValueError: Если profile не 1-D.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"profile must be 1-D, got ndim={profile.ndim}")
    if profile.size == 0:
        return profile.copy()
    mn, mx = profile.min(), profile.max()
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(profile)
    return (profile - mn) / (mx - mn)


def aggregate_profiles(
    profiles: List[np.ndarray],
    weights:  Optional[List[float]] = None,
) -> np.ndarray:
    """Взвешенная комбинация нескольких профилей одинаковой длины.

    Args:
        profiles: Список 1-D массивов одинаковой длины.
        weights:  Веса (нормализуются до суммы 1.0); None → равные.

    Returns:
        Комбинированный профиль float64.

    Raises:
        ValueError: Если profiles пустой, длины различаются или сумма весов = 0.
    """
    if not profiles:
        raise ValueError("profiles must not be empty.")
    n = len(profiles[0])
    for p in profiles[1:]:
        if len(p) != n:
            raise ValueError(
                f"All profiles must have the same length ({n}), got {len(p)}."
            )

    if weights is None:
        w = np.full(len(profiles), 1.0 / len(profiles), dtype=np.float64)
    else:
        w = np.array(weights, dtype=np.float64)
        s = w.sum()
        if s < 1e-12:
            raise ValueError("Sum of weights must be > 0.")
        w = w / s

    combined = np.zeros(n, dtype=np.float64)
    for p, wi in zip(profiles, w):
        combined += float(wi) * np.asarray(p, dtype=np.float64)
    return combined


def compare_profiles(
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Вычислить сходство двух профилей одинаковой длины.

    similarity = 1 - ||p1 - p2|| / (||p1|| + ||p2|| + ε)

    Args:
        p1: Профиль 1 (1-D).
        p2: Профиль 2 (1-D).

    Returns:
        Значение сходства в [0, 1] (1 = идентичны).

    Raises:
        ValueError: Если длины профилей различаются или массивы не 1-D.
    """
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64).ravel()
    if len(p1) != len(p2):
        raise ValueError(
            f"Profiles must have the same length, got {len(p1)} and {len(p2)}."
        )
    diff_norm  = float(np.linalg.norm(p1 - p2))
    denom      = float(np.linalg.norm(p1) + np.linalg.norm(p2)) + 1e-10
    similarity = 1.0 - diff_norm / denom
    return float(np.clip(similarity, 0.0, 1.0))


def batch_profile_edges(
    strips:    List[np.ndarray],
    cfg:       Optional[ProfileConfig] = None,
    edge_ids:  Optional[List[int]] = None,
) -> List[EdgeProfile]:
    """Профилировать список полос краёв.

    Args:
        strips:   Список 2-D grayscale массивов.
        cfg:      Конфигурация профилирования; None → ProfileConfig().
        edge_ids: Список идентификаторов краёв; None → [0, 1, 2, ...].

    Returns:
        Список EdgeProfile.
    """
    if cfg is None:
        cfg = ProfileConfig()
    if edge_ids is None:
        edge_ids = list(range(len(strips)))

    _dispatch = {
        "brightness": lambda s: compute_brightness_profile(s, cfg.n_samples),
        "gradient":   lambda s: compute_gradient_profile(s, cfg.n_samples),
        "diff":       lambda s: compute_diff_profile(s, cfg.n_samples),
    }
    fn = _dispatch[cfg.profile_type]

    results: List[EdgeProfile] = []
    for eid, strip in zip(edge_ids, strips):
        raw = fn(strip)
        if cfg.normalize:
            raw = normalize_profile(raw)
        results.append(EdgeProfile(
            profile=raw,
            edge_id=eid,
            profile_type=cfg.profile_type,
            n_samples=cfg.n_samples,
            params={"strip_width": cfg.strip_width, "normalize": cfg.normalize},
        ))
    return results
