"""
Оценка совместимости двух краёв фрагментов документа.

Вычисляет многоканальную оценку совместимости пары краёв (полоса
пикселей вдоль указанных сторон двух фрагментов) по нескольким
признакам: цвет, градиент и текстура. Итоговая оценка — взвешенная
сумма ∈ [0,1].

Классы:
    EdgeScore — результат оценки одной пары краёв

Функции:
    score_color_compat    — цветовая совместимость [0,1]
    score_gradient_compat — градиентная совместимость [0,1]
    score_texture_compat  — текстурная совместимость [0,1]
    score_edge_pair       — итоговая взвешенная оценка → EdgeScore
    batch_score_edges     — пакетная оценка списка пар изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── EdgeScore ────────────────────────────────────────────────────────────────

@dataclass
class EdgeScore:
    """
    Результат оценки совместимости одной пары краёв.

    Attributes:
        idx1:           Индекс первого фрагмента.
        idx2:           Индекс второго фрагмента.
        side1:          Сторона первого фрагмента (0=верх,1=право,2=низ,3=лево).
        side2:          Сторона второго фрагмента.
        color_score:    Цветовая совместимость ∈ [0,1].
        gradient_score: Градиентная совместимость ∈ [0,1].
        texture_score:  Текстурная совместимость ∈ [0,1].
        total_score:    Итоговая взвешенная оценка ∈ [0,1].
        method:         Всегда 'weighted'.
        params:         Использованные параметры.
    """
    idx1:           int
    idx2:           int
    side1:          int
    side2:          int
    color_score:    float
    gradient_score: float
    texture_score:  float
    total_score:    float
    method:         str  = "weighted"
    params:         Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"EdgeScore(({self.idx1},{self.idx2}), "
                f"sides=({self.side1},{self.side2}), "
                f"total={self.total_score:.4f})")


# ─── _extract_strip ───────────────────────────────────────────────────────────

def _extract_strip(img: np.ndarray, side: int, border_px: int) -> np.ndarray:
    """Вырезает полосу пикселей вдоль указанной стороны."""
    h, w = img.shape[:2]
    bp   = max(1, border_px)
    if side == 0:
        return img[:bp, :]
    elif side == 1:
        return img[:, w - bp:]
    elif side == 2:
        return img[h - bp:, :]
    elif side == 3:
        return img[:, :bp]
    else:
        raise ValueError(f"Unknown side {side}. Must be 0-3.")


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _normalize_hist(strip: np.ndarray, bins: int = 64) -> np.ndarray:
    """Нормализованная гистограмма для grayscale или первого канала."""
    gray = _to_gray(strip)
    h    = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    total = h.sum()
    return (h / total).astype(np.float32) if total > 0 else h.astype(np.float32)


# ─── score_color_compat ───────────────────────────────────────────────────────

def score_color_compat(img1:      np.ndarray,
                        img2:      np.ndarray,
                        side1:     int = 2,
                        side2:     int = 0,
                        border_px: int = 10,
                        bins:      int = 64) -> float:
    """
    Цветовая совместимость двух краёв (гистограммная корреляция).

    Args:
        img1:      Первое изображение uint8 (BGR или grayscale).
        img2:      Второе изображение uint8.
        side1:     Сторона первого изображения (0-3).
        side2:     Сторона второго изображения.
        border_px: Ширина полосы в пикселях.
        bins:      Число бинов гистограммы.

    Returns:
        Оценка ∈ [0,1]; 1 = идентичные распределения.
    """
    s1 = _extract_strip(img1, side1, border_px)
    s2 = _extract_strip(img2, side2, border_px)
    h1 = _normalize_hist(s1, bins)
    h2 = _normalize_hist(s2, bins)
    corr = float(cv2.compareHist(
        h1.reshape(-1, 1), h2.reshape(-1, 1), cv2.HISTCMP_CORREL
    ))
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ─── score_gradient_compat ────────────────────────────────────────────────────

def score_gradient_compat(img1:      np.ndarray,
                           img2:      np.ndarray,
                           side1:     int = 2,
                           side2:     int = 0,
                           border_px: int = 10) -> float:
    """
    Градиентная совместимость краёв (нормализованная корреляция профилей).

    Усредняет градиент по длине полосы и сравнивает средние профили
    двух краёв через нормализованную корреляцию.

    Args:
        img1:      Первое изображение.
        img2:      Второе изображение.
        side1:     Сторона первого.
        side2:     Сторона второго.
        border_px: Ширина полосы.

    Returns:
        Оценка ∈ [0,1].
    """
    s1 = _to_gray(_extract_strip(img1, side1, border_px)).astype(np.float32)
    s2 = _to_gray(_extract_strip(img2, side2, border_px)).astype(np.float32)

    # Усредняем профиль вдоль короткой оси полосы
    p1 = s1.mean(axis=0) if side1 in (0, 2) else s1.mean(axis=1)
    p2 = s2.mean(axis=0) if side2 in (0, 2) else s2.mean(axis=1)

    # Обрезаем до минимальной длины
    n = min(len(p1), len(p2))
    if n < 2:
        return 0.5

    p1, p2 = p1[:n], p2[:n]

    std1 = p1.std()
    std2 = p2.std()
    if std1 < 1e-6 or std2 < 1e-6:
        # Один из профилей плоский
        return 0.5

    corr = float(np.corrcoef(p1, p2)[0, 1])
    if not np.isfinite(corr):
        return 0.5
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ─── score_texture_compat ─────────────────────────────────────────────────────

def score_texture_compat(img1:      np.ndarray,
                          img2:      np.ndarray,
                          side1:     int = 2,
                          side2:     int = 0,
                          border_px: int = 10) -> float:
    """
    Текстурная совместимость краёв (сравнение локальной дисперсии).

    Использует стандартное отклонение пикселей вдоль полосы как
    прокси текстурной сложности. Схожие текстуры → близкие σ.

    Args:
        img1:      Первое изображение.
        img2:      Второе изображение.
        side1:     Сторона первого.
        side2:     Сторона второго.
        border_px: Ширина полосы.

    Returns:
        Оценка ∈ [0,1]; 1 = одинаковая текстурная сложность.
    """
    s1 = _to_gray(_extract_strip(img1, side1, border_px)).astype(np.float32)
    s2 = _to_gray(_extract_strip(img2, side2, border_px)).astype(np.float32)

    std1 = float(s1.std())
    std2 = float(s2.std())

    if std1 + std2 < 1e-9:
        return 1.0   # оба однородны — совместимы

    diff = abs(std1 - std2)
    norm = max(std1, std2, 1.0)
    return float(np.clip(1.0 - diff / norm, 0.0, 1.0))


# ─── score_edge_pair ──────────────────────────────────────────────────────────

def score_edge_pair(img1:      np.ndarray,
                     img2:      np.ndarray,
                     idx1:      int = 0,
                     idx2:      int = 1,
                     side1:     int = 2,
                     side2:     int = 0,
                     border_px: int = 10,
                     bins:      int = 64,
                     weights:   Optional[Dict[str, float]] = None) -> EdgeScore:
    """
    Вычисляет итоговую оценку совместимости двух краёв.

    Args:
        img1:      Первое изображение uint8.
        img2:      Второе изображение uint8.
        idx1:      Индекс первого фрагмента.
        idx2:      Индекс второго фрагмента.
        side1:     Сторона первого фрагмента.
        side2:     Сторона второго фрагмента.
        border_px: Ширина полосы.
        bins:      Бины гистограммы для цвета.
        weights:   {'color': w, 'gradient': w, 'texture': w};
                   None → {'color': 0.4, 'gradient': 0.4, 'texture': 0.2}.

    Returns:
        EdgeScore с отдельными каналами и итоговой оценкой.
    """
    if weights is None:
        weights = {"color": 0.4, "gradient": 0.4, "texture": 0.2}

    c_score = score_color_compat(img1, img2, side1, side2, border_px, bins)
    g_score = score_gradient_compat(img1, img2, side1, side2, border_px)
    t_score = score_texture_compat(img1, img2, side1, side2, border_px)

    w_c = float(weights.get("color",    0.4))
    w_g = float(weights.get("gradient", 0.4))
    w_t = float(weights.get("texture",  0.2))
    total_w = w_c + w_g + w_t

    if total_w <= 0.0:
        total = 0.0
    else:
        total = (c_score * w_c + g_score * w_g + t_score * w_t) / total_w

    total = float(np.clip(total, 0.0, 1.0))

    return EdgeScore(
        idx1=idx1, idx2=idx2,
        side1=side1, side2=side2,
        color_score=c_score,
        gradient_score=g_score,
        texture_score=t_score,
        total_score=total,
        method="weighted",
        params={
            "border_px": border_px,
            "bins":      bins,
            "weights":   dict(weights),
        },
    )


# ─── batch_score_edges ────────────────────────────────────────────────────────

def batch_score_edges(images:     List[np.ndarray],
                       pairs:      List[Tuple[int, int]],
                       side_pairs: Optional[List[Tuple[int, int]]] = None,
                       border_px:  int  = 10,
                       bins:       int  = 64,
                       weights:    Optional[Dict[str, float]] = None) -> List[EdgeScore]:
    """
    Пакетная оценка совместимости списка пар фрагментов.

    Args:
        images:     Список изображений (индексируются как в pairs).
        pairs:      Список (idx1, idx2) — пары для оценки.
        side_pairs: Список (side1, side2) для каждой пары;
                    None → каждая пара (2, 0) [низ→верх].
        border_px:  Ширина полосы.
        bins:       Бины гистограммы.
        weights:    Веса каналов.

    Returns:
        Список EdgeScore той же длины, что pairs.
    """
    if side_pairs is None:
        side_pairs = [(2, 0)] * len(pairs)

    results: List[EdgeScore] = []
    for k, (i, j) in enumerate(pairs):
        s1, s2 = side_pairs[k]
        results.append(
            score_edge_pair(
                images[i], images[j],
                idx1=i, idx2=j,
                side1=s1, side2=s2,
                border_px=border_px,
                bins=bins,
                weights=weights,
            )
        )
    return results
