"""
Комплексная оценка шва между двумя фрагментами документа.

Объединяет несколько независимых сигналов (профиль яркости, цвет,
текстура, градиент) в единую взвешенную оценку совместимости шва,
поддерживает ранжирование кандидатов и пакетную обработку.

Классы:
    SeamScoreResult — результат комплексной оценки шва

Функции:
    compute_seam_score    — взвешенная оценка одного шва
    seam_score_matrix     — разреженная матрица оценок для пар фрагментов
    normalize_seam_scores — нормализация вектора оценок в [0,1]
    rank_candidates       — ранжирование кандидатов по убыванию оценки
    batch_seam_scores     — пакетная обработка с одинаковыми весами
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── SeamScoreResult ──────────────────────────────────────────────────────────

@dataclass
class SeamScoreResult:
    """
    Результат комплексной оценки шва.

    Attributes:
        score:             Итоговая взвешенная оценка ∈ [0,1].
        component_scores:  Оценки компонент:
                           'profile'  — корреляция профиля яркости,
                           'color'    — близость гистограмм серого,
                           'texture'  — схожесть локальной дисперсии,
                           'gradient' — корреляция профиля градиента.
        side1:             Сторона первого фрагмента (0=верх,…,3=лево).
        side2:             Сторона второго фрагмента.
        method:            Всегда 'seam'.
        params:            Веса и вспомогательные параметры.
    """
    score:            float
    component_scores: Dict[str, float]
    side1:            int  = 1
    side2:            int  = 3
    method:           str  = "seam"
    params:           Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        comps = ", ".join(f"{k}={v:.3f}" for k, v in self.component_scores.items())
        return (f"SeamScoreResult(score={self.score:.3f}, "
                f"sides=({self.side1},{self.side2}), "
                f"[{comps}])")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _extract_edge_strip(gray: np.ndarray,
                         side: int,
                         n_samples: int,
                         border_frac: float = 0.08) -> np.ndarray:
    """
    Извлекает 1D-полосу пикселей вдоль заданного края.

    Args:
        gray:       Grayscale изображение.
        side:       Сторона (0=верх,1=право,2=низ,3=лево).
        n_samples:  Число точек в выходном профиле.
        border_frac: Доля размера изображения для ширины полосы.

    Returns:
        Вектор float32 длины n_samples.
    """
    h, w    = gray.shape
    border  = max(1, int(min(h, w) * border_frac))
    g_f     = gray.astype(np.float32)

    if side == 0:     # верх
        strip = g_f[:border, :].mean(axis=0)
    elif side == 1:   # право
        strip = g_f[:, -border:].mean(axis=1)
    elif side == 2:   # низ
        strip = g_f[-border:, :].mean(axis=0)
    else:             # лево (side == 3)
        strip = g_f[:, :border].mean(axis=1)

    n = len(strip)
    if n == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if n == n_samples:
        return strip.astype(np.float32)

    xs     = np.linspace(0, n - 1, n_samples, dtype=np.float64)
    xp     = np.arange(n, dtype=np.float64)
    return np.interp(xs, xp, strip).astype(np.float32)


def _extract_gradient_strip(gray: np.ndarray,
                              side: int,
                              n_samples: int) -> np.ndarray:
    """Вычисляет профиль градиентной магнитуды вдоль края."""
    gy  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gy ** 2 + gx ** 2)
    return _extract_edge_strip(mag.astype(np.uint8), side, n_samples)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Нормализованная корреляция Пирсона → [0,1]. 0.5 при нулевой дисперсии."""
    if len(a) == 0:
        return 0.5
    a_c = a - a.mean()
    b_c = b - b.mean()
    na  = float(np.linalg.norm(a_c))
    nb  = float(np.linalg.norm(b_c))
    if na < 1e-9 or nb < 1e-9:
        return 0.5
    r = float(np.dot(a_c, b_c) / (na * nb))
    return float(np.clip((r + 1.0) / 2.0, 0.0, 1.0))


def _hist_similarity(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Сходство гистограмм серого (Bhattacharyya → [0,1])."""
    ha = np.histogram(a, bins=bins, range=(0, 256))[0].astype(np.float32) + 1e-9
    hb = np.histogram(b, bins=bins, range=(0, 256))[0].astype(np.float32) + 1e-9
    ha /= ha.sum()
    hb /= hb.sum()
    bc  = float(np.sum(np.sqrt(ha * hb)))
    return float(np.clip(bc, 0.0, 1.0))


def _texture_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Сходство зернистости (по отношению стандартных отклонений)."""
    sa = float(a.std())
    sb = float(b.std())
    mx = max(sa, sb)
    if mx < 1e-9:
        return 1.0
    return float(np.clip(min(sa, sb) / mx, 0.0, 1.0))


# ─── compute_seam_score ───────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "profile":  0.35,
    "color":    0.25,
    "texture":  0.20,
    "gradient": 0.20,
}


def compute_seam_score(img1:        np.ndarray,
                        img2:        np.ndarray,
                        side1:       int  = 1,
                        side2:       int  = 3,
                        weights:     Optional[Dict[str, float]] = None,
                        n_samples:   int  = 64,
                        border_frac: float = 0.08) -> SeamScoreResult:
    """
    Вычисляет комплексную оценку шва между двумя фрагментами.

    Четыре компоненты:
      - 'profile'  : нормализованная корреляция профилей яркости,
      - 'color'    : Bhattacharyya-сходство гистограмм серого,
      - 'texture'  : схожесть стандартного отклонения вдоль края,
      - 'gradient' : нормализованная корреляция профилей градиента.

    Args:
        img1, img2:   BGR или grayscale изображения.
        side1:        Сторона первого фрагмента (0=верх,1=право,2=низ,3=лево).
        side2:        Сторона второго фрагмента.
        weights:      Веса компонент (None → равные веса).
        n_samples:    Число точек профиля вдоль края.
        border_frac:  Доля ширины/высоты изображения для полосы края.

    Returns:
        SeamScoreResult с итоговой оценкой и компонентами.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update({k: v for k, v in weights.items() if k in w})

    # Нормировка весов
    total = sum(w.values())
    if total < 1e-9:
        total = 1.0
    w = {k: v / total for k, v in w.items()}

    g1 = _to_gray(img1)
    g2 = _to_gray(img2)

    s1 = _extract_edge_strip(g1, side1, n_samples, border_frac)
    s2 = _extract_edge_strip(g2, side2, n_samples, border_frac)

    grad1 = _extract_gradient_strip(g1, side1, n_samples)
    grad2 = _extract_gradient_strip(g2, side2, n_samples)

    comps: Dict[str, float] = {
        "profile":  _safe_correlation(s1, s2),
        "color":    _hist_similarity(s1, s2),
        "texture":  _texture_similarity(s1, s2),
        "gradient": _safe_correlation(grad1, grad2),
    }

    score = float(sum(w[k] * comps[k] for k in comps))
    score = float(np.clip(score, 0.0, 1.0))

    return SeamScoreResult(
        score=score,
        component_scores=comps,
        side1=side1, side2=side2,
        method="seam",
        params={
            "weights": w,
            "n_samples": n_samples,
            "border_frac": border_frac,
        },
    )


# ─── seam_score_matrix ────────────────────────────────────────────────────────

def seam_score_matrix(images:      List[np.ndarray],
                       pairs:       List[Tuple[int, int, int, int]],
                       weights:     Optional[Dict[str, float]] = None,
                       n_samples:   int   = 64,
                       border_frac: float = 0.08) -> List[SeamScoreResult]:
    """
    Вычисляет оценки шва для списка пар (idx1, side1, idx2, side2).

    Args:
        images:     Список изображений.
        pairs:      Список кортежей (i, side1, j, side2).
        weights:    Веса компонент.
        n_samples:  Число точек профиля.
        border_frac: Доля ширины для полосы.

    Returns:
        Список SeamScoreResult (по одному на пару).
    """
    return [
        compute_seam_score(images[i], images[j],
                            side1=s1, side2=s2,
                            weights=weights,
                            n_samples=n_samples,
                            border_frac=border_frac)
        for i, s1, j, s2 in pairs
    ]


# ─── normalize_seam_scores ────────────────────────────────────────────────────

def normalize_seam_scores(scores: List[float]) -> List[float]:
    """
    Нормализует список оценок в диапазон [0,1] (min-max).

    Если все оценки одинаковы, возвращает список из единиц.

    Args:
        scores: Список числовых оценок.

    Returns:
        Нормализованный список той же длины.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if abs(hi - lo) < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ─── rank_candidates ──────────────────────────────────────────────────────────

def rank_candidates(scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Ранжирует кандидатов по убыванию оценки.

    Args:
        scores: Список пар (fragment_id, score).

    Returns:
        Тот же список, отсортированный по убыванию score.
    """
    return sorted(scores, key=lambda x: x[1], reverse=True)


# ─── batch_seam_scores ────────────────────────────────────────────────────────

def batch_seam_scores(images:      List[np.ndarray],
                       pairs:       List[Tuple[int, int, int, int]],
                       weights:     Optional[Dict[str, float]] = None,
                       n_samples:   int   = 64,
                       border_frac: float = 0.08) -> List[SeamScoreResult]:
    """
    Псевдоним для seam_score_matrix с идентичным интерфейсом.

    Вычисляет оценки шва для всех указанных пар с едиными параметрами.
    Пустой список пар возвращает пустой список.
    """
    return seam_score_matrix(images, pairs, weights=weights,
                              n_samples=n_samples, border_frac=border_frac)
