"""
Извлечение и анализ 1D профилей краёв фрагментов.

Представляет каждый край фрагмента в виде одномерного сигнала,
который описывает яркость, градиент или текстуру вдоль края.
Профили используются для точного сопоставления соседних краёв.

Классы:
    EdgeProfile        — 1D профиль одного края
    ProfileMatchResult — результат сравнения двух профилей

Функции:
    extract_intensity_profile — яркостный профиль вдоль края
    extract_gradient_profile  — профиль градиентной силы
    extract_texture_profile   — профиль локальной текстурной сложности
    normalize_profile         — z-нормализация профиля
    profile_correlation       — нормированная кросс-корреляция ∈ [0,1]
    profile_dtw               — DTW-схожесть ∈ [0,1]
    match_edge_profiles       — взвешенное сравнение трёх профилей
    batch_profile_match       — пакетное сравнение пар краёв
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

@dataclass
class EdgeProfile:
    """
    Одномерный профиль края фрагмента.

    Attributes:
        signal:    float32 массив длиной n_samples.
        side:      Сторона: 0=верх, 1=право, 2=низ, 3=лево.
        n_samples: Длина сигнала.
        method:    Тип профиля ('intensity', 'gradient', 'texture').
        params:    Словарь параметров.
    """
    signal:    np.ndarray
    side:      int
    n_samples: int
    method:    str
    params:    Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"EdgeProfile(side={self.side}, n={self.n_samples}, "
                f"method={self.method!r}, "
                f"mean={self.signal.mean():.2f})")


# ─── ProfileMatchResult ───────────────────────────────────────────────────────

@dataclass
class ProfileMatchResult:
    """
    Результат сравнения двух краевых профилей.

    Attributes:
        score:       Итоговая взвешенная схожесть ∈ [0,1].
        correlation: Нормированная кросс-корреляция ∈ [0,1].
        dtw_score:   DTW-схожесть ∈ [0,1].
        method:      Идентификатор метода.
        params:      Словарь параметров.
    """
    score:       float
    correlation: float
    dtw_score:   float
    method:      str = "profile"
    params:      Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ProfileMatchResult(score={self.score:.3f}, "
                f"corr={self.correlation:.3f}, dtw={self.dtw_score:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _extract_strip(img: np.ndarray, side: int,
                    border_frac: float = 0.08) -> np.ndarray:
    """Извлекает полосу вдоль стороны side (0=верх,1=право,2=низ,3=лево)."""
    h, w = img.shape[:2]
    n    = max(1, int(min(h, w) * border_frac))
    if side == 0:
        return img[:n, :]
    elif side == 1:
        return img[:, w - n:]
    elif side == 2:
        return img[h - n:, :]
    else:
        return img[:, :n]


def _resample_1d(signal: np.ndarray, n: int) -> np.ndarray:
    """Ресэмплирует 1D сигнал до длины n методом линейной интерполяции."""
    if len(signal) == n:
        return signal.astype(np.float32)
    src = signal.astype(np.float32).reshape(1, -1)
    dst = cv2.resize(src, (n, 1), interpolation=cv2.INTER_LINEAR)
    return dst.flatten()


# ─── extract_intensity_profile ───────────────────────────────────────────────

def extract_intensity_profile(img: np.ndarray,
                               side: int,
                               border_frac: float = 0.08,
                               n_samples: int = 64) -> EdgeProfile:
    """
    Яркостный профиль: среднее значение пикселей вдоль края.

    Для горизонтальных сторон (верх/низ) усредняет по строкам;
    для вертикальных (право/лево) — по столбцам.

    Args:
        img:         BGR или grayscale изображение.
        side:        Сторона (0=верх,1=право,2=низ,3=лево).
        border_frac: Доля min(h,w) для ширины полосы.
        n_samples:   Длина результирующего вектора.

    Returns:
        EdgeProfile с method='intensity'.
    """
    gray  = _to_gray(_extract_strip(img, side, border_frac))
    # Проекция: для горизонтальных сторон — по столбцам; для вертикальных — по строкам
    if side in (0, 2):
        raw = gray.mean(axis=0)   # длина = ширина изображения
    else:
        raw = gray.mean(axis=1)   # длина = высота изображения

    signal = _resample_1d(raw, n_samples)

    return EdgeProfile(
        signal=signal,
        side=side,
        n_samples=n_samples,
        method="intensity",
        params={"border_frac": border_frac, "n_samples": n_samples},
    )


# ─── extract_gradient_profile ─────────────────────────────────────────────────

def extract_gradient_profile(img: np.ndarray,
                              side: int,
                              border_frac: float = 0.08,
                              n_samples: int = 64) -> EdgeProfile:
    """
    Профиль перпендикулярного градиента: сила Sobel в направлении к краю.

    Args:
        img:         BGR или grayscale изображение.
        side:        Сторона (0=верх,1=право,2=низ,3=лево).
        border_frac: Доля min(h,w) для ширины полосы.
        n_samples:   Длина результирующего вектора.

    Returns:
        EdgeProfile с method='gradient'.
    """
    gray = _to_gray(_extract_strip(img, side, border_frac)).astype(np.float32)

    # Градиент перпендикулярно к краю
    if side in (0, 2):   # горизонтальные стороны → dy
        gx = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:                # вертикальные стороны → dx
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

    mag = np.abs(gx)

    if side in (0, 2):
        raw = mag.mean(axis=0)
    else:
        raw = mag.mean(axis=1)

    signal = _resample_1d(raw, n_samples)

    return EdgeProfile(
        signal=signal,
        side=side,
        n_samples=n_samples,
        method="gradient",
        params={"border_frac": border_frac, "n_samples": n_samples},
    )


# ─── extract_texture_profile ──────────────────────────────────────────────────

def extract_texture_profile(img: np.ndarray,
                              side: int,
                              border_frac: float = 0.08,
                              n_samples: int = 64,
                              window: int = 8) -> EdgeProfile:
    """
    Профиль текстурной сложности: скользящее стандартное отклонение яркости.

    Args:
        img:         BGR или grayscale изображение.
        side:        Сторона (0=верх,1=право,2=низ,3=лево).
        border_frac: Доля min(h,w) для ширины полосы.
        n_samples:   Длина результирующего вектора.
        window:      Размер скользящего окна для σ.

    Returns:
        EdgeProfile с method='texture'.
    """
    gray = _to_gray(_extract_strip(img, side, border_frac))
    h, w = gray.shape

    if side in (0, 2):
        # Среднее по строкам → вектор длиной w; скользящий σ вдоль ширины
        col_std = np.array([
            float(gray[:, max(0, j - window // 2):j + window // 2 + 1].std())
            for j in range(w)
        ], dtype=np.float32)
        raw = col_std
    else:
        row_std = np.array([
            float(gray[max(0, i - window // 2):i + window // 2 + 1, :].std())
            for i in range(h)
        ], dtype=np.float32)
        raw = row_std

    signal = _resample_1d(raw, n_samples)

    return EdgeProfile(
        signal=signal,
        side=side,
        n_samples=n_samples,
        method="texture",
        params={"border_frac": border_frac, "n_samples": n_samples,
                "window": window},
    )


# ─── normalize_profile ────────────────────────────────────────────────────────

def normalize_profile(profile: EdgeProfile) -> EdgeProfile:
    """
    Z-нормализация профиля: (signal − μ) / σ.

    Если σ < ε, возвращает нулевой сигнал (постоянный профиль).

    Args:
        profile: EdgeProfile.

    Returns:
        Новый EdgeProfile с нормированным сигналом.
    """
    mu  = profile.signal.mean()
    sig = profile.signal.std()
    if sig < 1e-9:
        normalized = np.zeros_like(profile.signal)
    else:
        normalized = (profile.signal - mu) / sig

    return EdgeProfile(
        signal=normalized.astype(np.float32),
        side=profile.side,
        n_samples=profile.n_samples,
        method=profile.method + "_normalized",
        params={**profile.params, "normalized": True},
    )


# ─── profile_correlation ──────────────────────────────────────────────────────

def profile_correlation(p1: EdgeProfile, p2: EdgeProfile) -> float:
    """
    Нормированная кросс-корреляция двух профилей ∈ [0,1].

    Схожесть = (ρ + 1) / 2, где ρ — Pearson r ∈ [−1, 1].
    Равна 1 для идентичных профилей, 0.5 для некоррелированных.

    Args:
        p1, p2: EdgeProfile одинаковой длины n_samples.

    Returns:
        float ∈ [0,1].
    """
    s1 = p1.signal.astype(np.float64)
    s2 = p2.signal.astype(np.float64)

    n = len(s1)
    if n == 0:
        return 0.5

    mu1, mu2 = s1.mean(), s2.mean()
    std1 = s1.std()
    std2 = s2.std()

    if std1 < 1e-9 or std2 < 1e-9:
        # Constant profiles: identical → 1, different → 0
        return 1.0 if np.allclose(s1, s2) else 0.5

    rho = float(((s1 - mu1) * (s2 - mu2)).mean() / (std1 * std2))
    rho = float(np.clip(rho, -1.0, 1.0))
    return (rho + 1.0) / 2.0


# ─── profile_dtw ──────────────────────────────────────────────────────────────

def profile_dtw(p1: EdgeProfile, p2: EdgeProfile,
                window: Optional[int] = None) -> float:
    """
    DTW (Dynamic Time Warping) схожесть двух профилей ∈ [0,1].

    Используется зависимость exp(−dtw_cost/scale), где scale = n.

    Args:
        p1, p2: EdgeProfile.
        window: Ширина полосы Sakoe-Chiba (None → без ограничений).

    Returns:
        float ∈ [0,1]; 1 = идентичные.
    """
    s1 = p1.signal.astype(np.float64)
    s2 = p2.signal.astype(np.float64)
    n, m = len(s1), len(s2)

    if n == 0 or m == 0:
        return 0.0

    w = window if window is not None else max(n, m)

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])

    scale = max(n, 1)
    dist  = float(dtw[n, m])
    return float(np.exp(-dist / (scale * scale)))


# ─── match_edge_profiles ──────────────────────────────────────────────────────

def match_edge_profiles(img1: np.ndarray,
                         img2: np.ndarray,
                         side1: int = 1,
                         side2: int = 3,
                         border_frac: float = 0.08,
                         n_samples: int = 64,
                         w_intensity: float = 0.4,
                         w_gradient: float = 0.3,
                         w_texture: float = 0.3) -> ProfileMatchResult:
    """
    Взвешенное сравнение трёх типов краевых профилей.

    Вычисляет среднее корреляции и DTW по трём профилям (intensity,
    gradient, texture) с весами w_intensity, w_gradient, w_texture.

    Args:
        img1, img2:  BGR или grayscale изображения.
        side1:       Сторона первого фрагмента.
        side2:       Сторона второго фрагмента.
        border_frac: Доля для полосы края.
        n_samples:   Длина профилей.
        w_intensity / w_gradient / w_texture: Веса профилей (сумма = 1).

    Returns:
        ProfileMatchResult с score ∈ [0,1].
    """
    # Интенсивность
    ip1  = extract_intensity_profile(img1, side1, border_frac, n_samples)
    ip2  = extract_intensity_profile(img2, side2, border_frac, n_samples)
    ic   = profile_correlation(ip1, ip2)
    id_  = profile_dtw(ip1, ip2)
    i_sc = (ic + id_) / 2.0

    # Градиент
    gp1  = extract_gradient_profile(img1, side1, border_frac, n_samples)
    gp2  = extract_gradient_profile(img2, side2, border_frac, n_samples)
    gc   = profile_correlation(gp1, gp2)
    gd   = profile_dtw(gp1, gp2)
    g_sc = (gc + gd) / 2.0

    # Текстура
    tp1  = extract_texture_profile(img1, side1, border_frac, n_samples)
    tp2  = extract_texture_profile(img2, side2, border_frac, n_samples)
    tc   = profile_correlation(tp1, tp2)
    td   = profile_dtw(tp1, tp2)
    t_sc = (tc + td) / 2.0

    score = float(w_intensity * i_sc + w_gradient * g_sc + w_texture * t_sc)

    return ProfileMatchResult(
        score=float(np.clip(score, 0.0, 1.0)),
        correlation=float((ic + gc + tc) / 3.0),
        dtw_score=float((id_ + gd + td) / 3.0),
        method="profile",
        params={
            "side1": side1, "side2": side2,
            "border_frac": border_frac, "n_samples": n_samples,
            "w_intensity": w_intensity,
            "w_gradient":  w_gradient,
            "w_texture":   w_texture,
        },
    )


# ─── batch_profile_match ──────────────────────────────────────────────────────

def batch_profile_match(images: List[np.ndarray],
                         side_pairs: List[Tuple[int, int, int, int]],
                         border_frac: float = 0.08,
                         n_samples: int = 64) -> List[ProfileMatchResult]:
    """
    Пакетное сравнение пар краёв.

    Args:
        images:     Список изображений.
        side_pairs: Список кортежей (idx1, side1, idx2, side2).
        border_frac: Доля для полосы края.
        n_samples:  Длина профилей.

    Returns:
        Список ProfileMatchResult по одному на кортеж.
    """
    results = []
    for idx1, s1, idx2, s2 in side_pairs:
        r = match_edge_profiles(
            images[idx1], images[idx2],
            side1=s1, side2=s2,
            border_frac=border_frac,
            n_samples=n_samples,
        )
        results.append(r)
    return results
