"""
Коррекция наклона (skew) изображений фрагментов документа.

Оценивает угол наклона текстовых строк с помощью проекционного профиля
(перебор углов, максимум дисперсии) или метода линий Хафа, затем
применяет поворот для выравнивания.

Классы:
    DeskewResult — результат коррекции одного изображения

Функции:
    estimate_skew_projection — оценка угла по проекционному профилю
    estimate_skew_hough      — оценка угла по линиям Хафа
    deskew_image             — применение поворота на заданный угол
    auto_deskew              — автоматическая коррекция (выбор метода)
    batch_deskew             — пакетная коррекция списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── DeskewResult ─────────────────────────────────────────────────────────────

@dataclass
class DeskewResult:
    """
    Результат коррекции наклона одного изображения.

    Attributes:
        corrected:  Выровненное изображение (uint8, та же форма).
        angle:      Применённый угол коррекции (°). Положительный = CCW.
        method:     'projection' | 'hough' | 'auto'.
        confidence: Уверенность в оценке ∈ [0,1].
        params:     Использованные параметры.
    """
    corrected:  np.ndarray
    angle:      float
    method:     str
    confidence: float = 0.0
    params:     Dict  = field(default_factory=dict)

    def __repr__(self) -> str:
        h, w = self.corrected.shape[:2]
        return (f"DeskewResult(angle={self.angle:.2f}°, "
                f"method={self.method!r}, "
                f"conf={self.confidence:.2f}, "
                f"shape=({h},{w}))")


# ─── _to_binary ───────────────────────────────────────────────────────────────

def _to_binary(img: np.ndarray,
                block_size: int = 15,
                C:          int = 10) -> np.ndarray:
    """Бинаризует изображение (адаптивный порог Otsu для простоты)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


# ─── estimate_skew_projection ─────────────────────────────────────────────────

def estimate_skew_projection(img:         np.ndarray,
                               angle_range: Tuple[float, float] = (-15.0, 15.0),
                               n_angles:   int   = 60) -> Tuple[float, float]:
    """
    Оценивает угол наклона методом проекционного профиля.

    Перебирает n_angles углов в angle_range. Для каждого угла вращает
    бинаризованное изображение и вычисляет дисперсию горизонтальной
    проекции (суммы по строкам). Максимальная дисперсия соответствует
    наилучшему выравниванию строк.

    Args:
        img:         BGR или grayscale изображение uint8.
        angle_range: (min_angle, max_angle) в градусах.
        n_angles:    Количество проверяемых углов.

    Returns:
        (angle, confidence) — угол наклона (°) и уверенность ∈ [0,1].
    """
    bw    = _to_binary(img)
    h, w  = bw.shape
    cx, cy = float(w) / 2, float(h) / 2

    angles = np.linspace(angle_range[0], angle_range[1], max(n_angles, 2))
    scores = np.zeros(len(angles), dtype=np.float64)

    for i, a in enumerate(angles):
        M   = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(bw, M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
        proj       = rot.sum(axis=1).astype(np.float64)
        scores[i]  = float(proj.var())

    best_idx = int(np.argmax(scores))
    best_angle = float(angles[best_idx])

    score_range = scores.max() - scores.min()
    confidence  = float(np.clip(score_range / (scores.max() + 1e-9), 0.0, 1.0))

    return best_angle, confidence


# ─── estimate_skew_hough ──────────────────────────────────────────────────────

def estimate_skew_hough(img:         np.ndarray,
                         angle_range: Tuple[float, float] = (-15.0, 15.0),
                         threshold:   int   = 50,
                         min_length:  float = 0.3) -> Tuple[float, float]:
    """
    Оценивает угол наклона по линиям, обнаруженным методом Хафа.

    Бинаризует изображение, находит линии с помощью probabilistic HoughLinesP
    и медиану их углов.

    Args:
        img:         BGR или grayscale изображение uint8.
        angle_range: Диапазон допустимых углов (°).
        threshold:   Порог HoughLinesP.
        min_length:  Минимальная длина линии как доля ширины изображения.

    Returns:
        (angle, confidence) — угол наклона (°) и уверенность ∈ [0,1].
        Если линий не найдено, возвращает (0.0, 0.0).
    """
    bw   = _to_binary(img)
    h, w = bw.shape
    min_len = max(1, int(w * min_length))

    lines = cv2.HoughLinesP(
        bw, rho=1, theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return 0.0, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        a = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        if angle_range[0] <= a <= angle_range[1]:
            angles.append(a)

    if not angles:
        return 0.0, 0.0

    median_angle = float(np.median(angles))
    # Уверенность = согласованность углов (1 - нормированная дисперсия)
    std_a     = float(np.std(angles))
    range_a   = float(angle_range[1] - angle_range[0])
    confidence = float(np.clip(1.0 - std_a / max(range_a, 1e-9), 0.0, 1.0))

    return median_angle, confidence


# ─── deskew_image ─────────────────────────────────────────────────────────────

def deskew_image(img:   np.ndarray,
                  angle: float,
                  fill:  int = 255) -> np.ndarray:
    """
    Применяет поворот на заданный угол для устранения наклона.

    Args:
        img:   BGR или grayscale изображение uint8.
        angle: Угол коррекции (°) против часовой стрелки.
        fill:  Заполнение граничных пикселей.

    Returns:
        Выровненное изображение той же формы и dtype.
    """
    h, w = img.shape[:2]
    cx, cy = float(w) / 2, float(h) / 2
    M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=fill)


# ─── auto_deskew ──────────────────────────────────────────────────────────────

def auto_deskew(img:         np.ndarray,
                 method:      str = "projection",
                 angle_range: Tuple[float, float] = (-15.0, 15.0),
                 fill:        int = 255,
                 **kwargs) -> DeskewResult:
    """
    Автоматически оценивает и корректирует наклон изображения.

    Args:
        img:         BGR или grayscale изображение uint8.
        method:      'projection' | 'hough'.
        angle_range: Диапазон допустимых углов.
        fill:        Заполнение граничных пикселей.
        **kwargs:    Параметры конкретного метода (n_angles, threshold, ...).

    Returns:
        DeskewResult с выровненным изображением и метаданными.

    Raises:
        ValueError: Неизвестный метод.
    """
    if method == "projection":
        n_angles = int(kwargs.get("n_angles", 60))
        angle, conf = estimate_skew_projection(img, angle_range=angle_range,
                                                n_angles=n_angles)
        params = {"n_angles": n_angles, "angle_range": list(angle_range)}
    elif method == "hough":
        threshold  = int(kwargs.get("threshold", 50))
        min_length = float(kwargs.get("min_length", 0.3))
        angle, conf = estimate_skew_hough(img, angle_range=angle_range,
                                           threshold=threshold,
                                           min_length=min_length)
        params = {"threshold": threshold, "min_length": min_length,
                  "angle_range": list(angle_range)}
    else:
        raise ValueError(
            f"Unknown deskew method {method!r}. "
            f"Choose 'projection' or 'hough'."
        )

    corrected = deskew_image(img, angle, fill=fill)
    return DeskewResult(
        corrected=corrected,
        angle=angle,
        method=method,
        confidence=conf,
        params=params,
    )


# ─── batch_deskew ─────────────────────────────────────────────────────────────

def batch_deskew(images: List[np.ndarray],
                  method: str = "projection",
                  **kwargs) -> List[DeskewResult]:
    """
    Применяет auto_deskew ко всем изображениям в списке.

    Args:
        images: Список BGR или grayscale изображений uint8.
        method: 'projection' | 'hough'.
        **kwargs: Параметры auto_deskew (angle_range, n_angles, ...).

    Returns:
        Список DeskewResult той же длины.

    Raises:
        ValueError: Неизвестный метод.
    """
    # Проверяем метод заранее, чтобы сразу сообщить об ошибке
    if method not in ("projection", "hough"):
        raise ValueError(
            f"Unknown deskew method {method!r}. "
            f"Choose 'projection' or 'hough'."
        )
    return [auto_deskew(img, method=method, **kwargs) for img in images]
