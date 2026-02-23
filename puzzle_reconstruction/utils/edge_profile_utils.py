"""edge_profile_utils — утилиты профилей краёв фрагментов.

Предоставляет функции для построения, сравнения и пакетной обработки
профилей краёв (edge profiles) на основе геометрических точек контура.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class EdgeProfileConfig:
    """Параметры построения профиля края.

    Attributes
    ----------
    n_samples:
        Число точек равномерной дискретизации профиля.
    smooth_sigma:
        Сигма гауссовского сглаживания (0 — без сглаживания).
    normalize:
        Нормализовать ли профиль в [0, 1].
    """
    n_samples:    int   = 64
    smooth_sigma: float = 1.0
    normalize:    bool  = True


# ─── Профиль края ─────────────────────────────────────────────────────────────

@dataclass
class EdgeProfile:
    """Профиль одного края фрагмента.

    Attributes
    ----------
    values:
        1-D массив float32 — дискретизированный профиль.
    side:
        Сторона ('top', 'bottom', 'left', 'right', 'unknown').
    n_samples:
        Длина профиля.
    meta:
        Дополнительные данные.
    """
    values:   np.ndarray
    side:     str  = "unknown"
    n_samples: int = 0
    meta:     Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32).ravel()
        valid_sides = {"top", "bottom", "left", "right", "unknown"}
        if self.side not in valid_sides:
            raise ValueError(f"side должен быть одним из {valid_sides}, получено '{self.side}'")
        self.n_samples = len(self.values)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (f"EdgeProfile(side={self.side!r}, "
                f"n_samples={self.n_samples}, "
                f"values_min={self.values.min():.3f}, "
                f"values_max={self.values.max():.3f})")


# ─── Построение профиля ───────────────────────────────────────────────────────

def _smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Простое гауссовское сглаживание 1-D массива (ручная свёртка)."""
    if sigma < 1e-8 or len(arr) < 2:
        return arr.copy()
    radius = max(1, int(3 * sigma))
    kernel_size = 2 * radius + 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr.astype(np.float64), radius, mode="edge")
    result = np.convolve(padded, kernel, mode="valid")
    return result[:len(arr)].astype(np.float32)


def build_edge_profile(
    points: np.ndarray,
    side:   str = "unknown",
    cfg:    Optional[EdgeProfileConfig] = None,
) -> EdgeProfile:
    """Построить профиль края из набора точек контура.

    Для горизонтальных сторон ('top', 'bottom') профилем служат
    y-координаты, интерполированные в n_samples равномерных позиций по x.
    Для вертикальных — x-координаты по y.

    Parameters
    ----------
    points:
        (N, 2) или (N, 1, 2) массив точек контура (float32/float64).
    side:
        Сторона края.
    cfg:
        Конфигурация.

    Returns
    -------
    EdgeProfile
    """
    if cfg is None:
        cfg = EdgeProfileConfig()

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points должен иметь форму (N,2) или (N,1,2), получено {pts.shape}")
    if len(pts) == 0:
        values = np.zeros(cfg.n_samples, dtype=np.float32)
        return EdgeProfile(values=values, side=side)

    if side in ("top", "bottom"):
        primary, secondary = pts[:, 0], pts[:, 1]
    else:
        primary, secondary = pts[:, 1], pts[:, 0]

    order = np.argsort(primary)
    px = primary[order]
    sy = secondary[order]

    # Удалить дубликаты по px
    _, unique_idx = np.unique(px, return_index=True)
    px = px[unique_idx]
    sy = sy[unique_idx]

    x_new = np.linspace(px[0], px[-1], cfg.n_samples)
    if len(px) == 1:
        values = np.full(cfg.n_samples, sy[0], dtype=np.float32)
    else:
        values = np.interp(x_new, px, sy).astype(np.float32)

    if cfg.smooth_sigma > 0:
        values = _smooth_1d(values, cfg.smooth_sigma)

    if cfg.normalize:
        mn, mx = values.min(), values.max()
        rng = mx - mn
        if rng > 1e-8:
            values = ((values - mn) / rng).astype(np.float32)
        else:
            values = np.zeros_like(values)

    return EdgeProfile(values=values, side=side)


# ─── Расстояния между профилями ───────────────────────────────────────────────

def profile_l2_distance(a: EdgeProfile, b: EdgeProfile) -> float:
    """Евклидово расстояние между профилями одинаковой длины.

    Parameters
    ----------
    a, b:
        Два EdgeProfile с одинаковым n_samples.

    Returns
    -------
    float ≥ 0.

    Raises
    ------
    ValueError
        Если длины профилей не совпадают.
    """
    if a.n_samples != b.n_samples:
        raise ValueError(
            f"Длины профилей не совпадают: {a.n_samples} != {b.n_samples}"
        )
    return float(np.linalg.norm(a.values - b.values))


def profile_cosine_similarity(a: EdgeProfile, b: EdgeProfile) -> float:
    """Косинусное сходство двух профилей.

    Parameters
    ----------
    a, b:
        Два EdgeProfile с одинаковым n_samples.

    Returns
    -------
    float в [-1, 1]; 1 — идентичные профили.
    """
    if a.n_samples != b.n_samples:
        raise ValueError(
            f"Длины профилей не совпадают: {a.n_samples} != {b.n_samples}"
        )
    na = np.linalg.norm(a.values)
    nb = np.linalg.norm(b.values)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a.values, b.values) / (na * nb))


def profile_correlation(a: EdgeProfile, b: EdgeProfile) -> float:
    """Коэффициент корреляции Пирсона двух профилей.

    Returns
    -------
    float в [-1, 1]; 1 — линейно зависимые профили.
    """
    if a.n_samples != b.n_samples:
        raise ValueError(
            f"Длины профилей не совпадают: {a.n_samples} != {b.n_samples}"
        )
    if a.n_samples < 2:
        return 1.0
    c = np.corrcoef(a.values, b.values)
    val = float(c[0, 1])
    if np.isnan(val):
        return 0.0
    return val


# ─── Операции с профилями ─────────────────────────────────────────────────────

def resample_profile(profile: EdgeProfile, n_samples: int) -> EdgeProfile:
    """Изменить число отсчётов профиля через линейную интерполяцию.

    Parameters
    ----------
    profile:
        Исходный EdgeProfile.
    n_samples:
        Новое число отсчётов (≥ 1).

    Returns
    -------
    Новый EdgeProfile.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples должен быть ≥ 1, получено {n_samples}")
    x_old = np.linspace(0, 1, len(profile.values))
    x_new = np.linspace(0, 1, n_samples)
    new_values = np.interp(x_new, x_old, profile.values).astype(np.float32)
    return EdgeProfile(values=new_values, side=profile.side,
                       meta=dict(profile.meta))


def flip_profile(profile: EdgeProfile) -> EdgeProfile:
    """Отразить профиль (реверс отсчётов).

    Returns
    -------
    Новый EdgeProfile с перевёрнутым вектором values.
    """
    return EdgeProfile(values=profile.values[::-1].copy(),
                       side=profile.side, meta=dict(profile.meta))


def mean_profile(profiles: List[EdgeProfile]) -> EdgeProfile:
    """Вычислить средний профиль из списка.

    Все профили должны иметь одинаковое n_samples.

    Returns
    -------
    EdgeProfile со средними значениями.
    """
    if not profiles:
        return EdgeProfile(values=np.zeros(0, dtype=np.float32))
    n = profiles[0].n_samples
    for p in profiles[1:]:
        if p.n_samples != n:
            raise ValueError("Все профили должны иметь одинаковое n_samples")
    stack = np.stack([p.values for p in profiles], axis=0)
    return EdgeProfile(values=stack.mean(axis=0).astype(np.float32),
                       side=profiles[0].side)


# ─── Пакетная обработка ───────────────────────────────────────────────────────

def batch_build_profiles(
    point_sets: Sequence[np.ndarray],
    sides:      Optional[Sequence[str]] = None,
    cfg:        Optional[EdgeProfileConfig] = None,
) -> List[EdgeProfile]:
    """Пакетно построить профили для списка наборов точек.

    Parameters
    ----------
    point_sets:
        Список массивов точек (N, 2).
    sides:
        Список сторон; если None, используется 'unknown' для всех.
    cfg:
        Конфигурация.

    Returns
    -------
    Список EdgeProfile.
    """
    if sides is None:
        sides = ["unknown"] * len(point_sets)
    return [
        build_edge_profile(pts, side=side, cfg=cfg)
        for pts, side in zip(point_sets, sides)
    ]


def pairwise_l2_matrix(profiles: List[EdgeProfile]) -> np.ndarray:
    """Построить матрицу L2-расстояний между всеми парами профилей.

    Parameters
    ----------
    profiles:
        Список EdgeProfile одинаковой длины.

    Returns
    -------
    (N, N) массив float64.
    """
    n = len(profiles)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = profile_l2_distance(profiles[i], profiles[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def best_matching_profile(
    query:      EdgeProfile,
    candidates: List[EdgeProfile],
) -> Tuple[int, float]:
    """Найти наиболее похожий профиль из списка кандидатов (по L2).

    Parameters
    ----------
    query:
        Запрашиваемый профиль.
    candidates:
        Список кандидатов.

    Returns
    -------
    (index, distance) — индекс лучшего кандидата и расстояние до него.
    """
    if not candidates:
        raise ValueError("Список кандидатов пуст")
    distances = [profile_l2_distance(query, c) for c in candidates]
    best_idx = int(np.argmin(distances))
    return best_idx, distances[best_idx]
