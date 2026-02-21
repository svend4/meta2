"""
Классификатор типов фрагментов документов.

Анализирует геометрические и текстурные характеристики фрагмента
для определения его роли (угол, край, внутренний, полная страница),
наличия текста и числа строк.

Классы:
    FragmentType     — перечисление типов (CORNER/EDGE/INNER/FULL/UNKNOWN)
    FragmentFeatures — вектор признаков фрагмента (12 элементов)
    ClassifyResult   — результат классификации

Функции:
    estimate_noise_level      — σ шума через дисперсию Лапласиана
    compute_texture_features  — дисперсия текстуры + LBP-равномерность
    compute_edge_features     — плотность/прямолинейность краёв по 4 сторонам
    compute_shape_features    — aspect_ratio, fill_ratio, dominant_angle
    detect_text_presence      — обнаружение текста по локальной дисперсии
    classify_fragment_type    — тип по профилю краёв
    classify_fragment         — полная классификация одного фрагмента
    batch_classify            — классификация списка фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ─── FragmentType ─────────────────────────────────────────────────────────────

class FragmentType(str, Enum):
    """Тип фрагмента документа по позиции на странице."""
    CORNER  = "corner"   # Угловой: 2 соседних прямых края
    EDGE    = "edge"     # Краевой: 1 прямой край
    INNER   = "inner"    # Внутренний: 0 прямых краёв
    FULL    = "full"     # Полная страница: 4 прямых края
    UNKNOWN = "unknown"  # Не удалось определить


# ─── FragmentFeatures ─────────────────────────────────────────────────────────

@dataclass
class FragmentFeatures:
    """
    Вектор признаков фрагмента.

    Атрибуты (все вещественные, нормированы или в понятных единицах):
        edge_densities:    Плотность краёв по 4 сторонам (top,right,bot,left).
        edge_straightness: Прямолинейность краёв [0, 1] по 4 сторонам.
        texture_variance:  Дисперсия Лапласиана (мера резкости).
        text_density:      Доля «текстовых» блоков [0, 1].
        aspect_ratio:      w / h.
        fill_ratio:        Доля непустых пикселей.
        dominant_angle:    Преобладающий угол градиента (°, –90…90).
        lbp_uniformity:    Доля равномерных LBP-паттернов [0, 1].
    """
    edge_densities:    Tuple[float, float, float, float] = (0., 0., 0., 0.)
    edge_straightness: Tuple[float, float, float, float] = (0., 0., 0., 0.)
    texture_variance:  float = 0.0
    text_density:      float = 0.0
    aspect_ratio:      float = 1.0
    fill_ratio:        float = 1.0
    dominant_angle:    float = 0.0
    lbp_uniformity:    float = 0.0

    def as_vector(self) -> np.ndarray:
        """Плоский массив из 12 элементов (float32)."""
        return np.array([
            *self.edge_densities,
            *self.edge_straightness,
            self.texture_variance,
            self.text_density,
            self.aspect_ratio,
            self.fill_ratio,
            self.dominant_angle,
            self.lbp_uniformity,
        ], dtype=np.float32)


# ─── ClassifyResult ───────────────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """
    Результат классификации фрагмента.

    Attributes:
        fragment_type:  Тип фрагмента (FragmentType).
        confidence:     Уверенность [0, 1].
        has_text:       True если обнаружен текст.
        text_lines:     Оценочное число строк текста.
        features:       Вычисленные признаки.
        straight_sides: Индексы прямых сторон (0=top, 1=right, 2=bot, 3=left).
    """
    fragment_type:  FragmentType
    confidence:     float
    has_text:       bool
    text_lines:     int
    features:       FragmentFeatures
    straight_sides: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"ClassifyResult(type={self.fragment_type.value!r}, "
                f"conf={self.confidence:.3f}, "
                f"text={self.has_text}, "
                f"straight={self.straight_sides})")


# ─── Текстурные признаки ──────────────────────────────────────────────────────

def compute_texture_features(gray: np.ndarray) -> Tuple[float, float]:
    """
    Вычисляет текстурные признаки grayscale-изображения.

    Returns:
        (texture_variance, lbp_uniformity)
        texture_variance — дисперсия Лапласиана.
        lbp_uniformity   — доля «активных» соседей (упрощённый LBP) [0, 1].
    """
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    texture_variance = float(np.var(lap))

    h, w = gray.shape
    if h < 3 or w < 3:
        return texture_variance, 0.0

    center = gray[1:-1, 1:-1].astype(np.int32)
    neighbors = [
        gray[:-2, 1:-1], gray[2:, 1:-1],
        gray[1:-1, :-2], gray[1:-1, 2:],
    ]
    active_total = 0
    total        = 0
    for nb in neighbors:
        bits = (nb.astype(np.int32) >= center).astype(np.uint8)
        active_total += int(bits.sum())
        total        += bits.size

    lbp_uniformity = float(active_total) / max(1, total)
    return texture_variance, lbp_uniformity


# ─── Признаки краёв ───────────────────────────────────────────────────────────

_BORDER_FRAC = 0.12   # Ширина полосы вдоль края (доля размера изображения)


def _border_strip(img: np.ndarray, side: int,
                  frac: float = _BORDER_FRAC) -> np.ndarray:
    """Вырезает полосу вдоль заданной стороны (0=top,1=right,2=bot,3=left)."""
    h, w = img.shape[:2]
    bh = max(1, int(h * frac))
    bw = max(1, int(w * frac))
    if side == 0:
        return img[:bh, :]
    elif side == 1:
        return img[:, w - bw:]
    elif side == 2:
        return img[h - bh:, :]
    else:
        return img[:, :bw]


def compute_edge_features(
        gray:     np.ndarray,
        canny_lo: int = 30,
        canny_hi: int = 100,
) -> Tuple[Tuple[float, float, float, float],
           Tuple[float, float, float, float]]:
    """
    Плотность и прямолинейность краёв по 4 сторонам фрагмента.

    Returns:
        (edge_densities, edge_straightness) — по 4 значения на сторону.
    """
    edge_map = cv2.Canny(gray, canny_lo, canny_hi)
    densities     = []
    straightnesses = []

    for side in range(4):
        strip_gray  = _border_strip(gray,     side)
        strip_edges = _border_strip(edge_map, side)

        density = float(strip_edges.sum()) / (255.0 * max(1, strip_edges.size))
        densities.append(density)

        if strip_edges.sum() == 0:
            straightnesses.append(0.0)
            continue

        # Прямолинейность — через нормированную дисперсию проекции краёв
        proj = (strip_edges.sum(axis=1)
                if side in (0, 2)
                else strip_edges.sum(axis=0)).astype(np.float32)
        if proj.max() > 0:
            proj /= proj.max()
        straight = float(np.clip(1.0 - float(np.std(proj)) * 2.0, 0.0, 1.0))
        straightnesses.append(straight)

    return tuple(densities), tuple(straightnesses)   # type: ignore[return-value]


# ─── Признаки формы ───────────────────────────────────────────────────────────

def compute_shape_features(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Геометрические признаки фрагмента.

    Returns:
        (aspect_ratio, fill_ratio, dominant_angle)
    """
    h, w = gray.shape
    aspect_ratio = float(w) / max(1, h)

    _, binary  = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    fill_ratio = float((binary > 0).sum()) / max(1, gray.size)

    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)

    if magnitudes.sum() > 0:
        angles     = np.arctan2(gy, gx) * 180.0 / np.pi
        flat_a     = angles.ravel()
        flat_m     = magnitudes.ravel()
        order      = np.argsort(flat_a)
        cumsum     = np.cumsum(flat_m[order])
        mid_idx    = np.searchsorted(cumsum, cumsum[-1] * 0.5)
        mid_idx    = min(mid_idx, len(order) - 1)
        dom_angle  = float(np.clip(flat_a[order[mid_idx]], -90.0, 90.0))
    else:
        dom_angle = 0.0

    return aspect_ratio, fill_ratio, dom_angle


# ─── Обнаружение текста ───────────────────────────────────────────────────────

def detect_text_presence(
        gray:       np.ndarray,
        block_size: int   = 16,
        var_thresh: float = 100.0,
) -> Tuple[bool, float, int]:
    """
    Обнаруживает текст по локальной дисперсии блоков.

    Разбивает изображение на блоки и считает «текстовыми» те,
    у которых дисперсия яркости выше порога.

    Args:
        gray:       Grayscale изображение.
        block_size: Размер блока (пикселей).
        var_thresh: Порог дисперсии для «текстового» блока.

    Returns:
        (has_text, text_density, n_text_rows)
    """
    h, w = gray.shape
    bs   = max(1, block_size)

    text_blocks  = 0
    total_blocks = 0
    row_has_text: List[bool] = []

    for y in range(0, h, bs):
        row_text = False
        for x in range(0, w, bs):
            block = gray[y: y + bs, x: x + bs].astype(np.float32)
            if float(np.var(block)) > var_thresh:
                text_blocks += 1
                row_text = True
            total_blocks += 1
        row_has_text.append(row_text)

    text_density = float(text_blocks) / max(1, total_blocks)
    has_text     = text_density > 0.05

    n_text_rows = 0
    prev = False
    for val in row_has_text:
        if val and not prev:
            n_text_rows += 1
        prev = val

    return has_text, text_density, n_text_rows


# ─── Классификация типа ───────────────────────────────────────────────────────

_STRAIGHT_SCORE_THRESH = 0.30   # порог combined_score (density × straightness)


def classify_fragment_type(
        edge_densities:    Tuple[float, float, float, float],
        edge_straightness: Tuple[float, float, float, float],
        aspect_ratio:      float,
) -> Tuple[FragmentType, float, List[int]]:
    """
    Определяет тип фрагмента по профилю краёв.

    Returns:
        (fragment_type, confidence, straight_sides)
        straight_sides — индексы сторон, признанных «прямыми».
    """
    scores   = [d * s for d, s in zip(edge_densities, edge_straightness)]
    straight = [i for i, sc in enumerate(scores) if sc > _STRAIGHT_SCORE_THRESH]
    n        = len(straight)

    if n >= 4:
        conf = float(np.clip(min(scores), 0.0, 1.0))
        return FragmentType.FULL, max(conf, 0.85), straight

    if n == 2:
        i, j      = straight[0], straight[1]
        adjacent  = (abs(i - j) == 1) or (i == 0 and j == 3)
        if adjacent:
            conf = float(np.clip((scores[i] + scores[j]) / 2.0, 0.0, 1.0))
            return FragmentType.CORNER, conf, straight
        # Противоположные прямые стороны — нетипичная ситуация
        return FragmentType.UNKNOWN, 0.35, straight

    if n == 1:
        conf = float(np.clip(scores[straight[0]], 0.0, 1.0))
        return FragmentType.EDGE, conf, straight

    # n == 0
    max_score = max(scores) if scores else 0.0
    if max_score < 0.05:
        return FragmentType.INNER, 0.70, []
    return FragmentType.UNKNOWN, 0.40, []


# ─── Полная классификация ─────────────────────────────────────────────────────

def classify_fragment(img:          np.ndarray,
                       border_frac:  float = _BORDER_FRAC,
                       var_thresh:   float = 100.0,
                       canny_lo:     int   = 30,
                       canny_hi:     int   = 100) -> ClassifyResult:
    """
    Полная классификация одного фрагмента документа.

    Args:
        img:        BGR или grayscale изображение.
        border_frac: Ширина полосы краёв (доля размера).
        var_thresh:  Порог дисперсии для текстовых блоков.
        canny_lo:    Нижний порог Canny.
        canny_hi:    Верхний порог Canny.

    Returns:
        ClassifyResult.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    tex_var,  lbp_uni  = compute_texture_features(gray)
    edge_den, edge_str = compute_edge_features(gray, canny_lo, canny_hi)
    asp, fill, dom_ang = compute_shape_features(gray)
    has_txt, txt_den, n_rows = detect_text_presence(gray, var_thresh=var_thresh)

    features = FragmentFeatures(
        edge_densities=edge_den,
        edge_straightness=edge_str,
        texture_variance=tex_var,
        text_density=txt_den,
        aspect_ratio=asp,
        fill_ratio=fill,
        dominant_angle=dom_ang,
        lbp_uniformity=lbp_uni,
    )

    ftype, conf, straight = classify_fragment_type(edge_den, edge_str, asp)

    return ClassifyResult(
        fragment_type=ftype,
        confidence=conf,
        has_text=has_txt,
        text_lines=n_rows,
        features=features,
        straight_sides=straight,
    )


# ─── batch_classify ───────────────────────────────────────────────────────────

def batch_classify(images:      List[np.ndarray],
                    border_frac: float = _BORDER_FRAC,
                    var_thresh:  float = 100.0) -> List[ClassifyResult]:
    """
    Классифицирует список изображений фрагментов.

    Args:
        images:      Список BGR или grayscale изображений.
        border_frac: Ширина полосы краёв.
        var_thresh:  Порог дисперсии для текста.

    Returns:
        Список ClassifyResult.
    """
    return [
        classify_fragment(img, border_frac=border_frac,
                          var_thresh=var_thresh)
        for img in images
    ]
