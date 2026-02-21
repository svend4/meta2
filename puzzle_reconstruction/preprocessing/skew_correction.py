"""
Коррекция наклона (skew) фрагментов документа.

Определяет угол поворота страницы тремя независимыми методами
и корректирует изображение поворотом вокруг центра.

Методы определения угла:
    hough      — линии Хафа; медиана угла горизонтальных линий
    projection — проекция на горизонтальные полосы; максимизация дисперсии
    fft        — спектральный анализ; доминирующее направление в ПФ

Функции:
    detect_skew_hough      → float (угол в градусах)
    detect_skew_projection → float
    detect_skew_fft        → float
    correct_skew           — поворачивает изображение на заданный угол
    auto_correct_skew      — полный пайплайн: detect + correct → SkewResult
    skew_confidence        — оценка надёжности угла (согласованность методов)

Класс:
    SkewResult — corrected_image, angle_deg, confidence, method, params
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── SkewResult ───────────────────────────────────────────────────────────────

@dataclass
class SkewResult:
    """
    Результат коррекции наклона.

    Attributes:
        corrected_image: Исправленное изображение (BGR или grayscale).
        angle_deg:       Обнаруженный угол наклона в градусах.
                         Положительный → против часовой стрелки.
        confidence:      ∈ [0, 1] — уверенность в оценке угла.
        method:          Использованный метод ('hough'/'projection'/'fft'/'auto').
        params:          Дополнительные параметры метода.
    """
    corrected_image: np.ndarray
    angle_deg:       float
    confidence:      float
    method:          str
    params:          Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        h, w = self.corrected_image.shape[:2]
        return (f"SkewResult(method={self.method!r}, "
                f"angle={self.angle_deg:.2f}°, "
                f"confidence={self.confidence:.3f}, "
                f"size={w}×{h})")


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Конвертирует BGR/BGRA в grayscale; grayscale возвращает без изменений."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _clamp_angle(angle_deg: float,
                  lo: float = -45.0,
                  hi: float = 45.0) -> float:
    """Приводит угол в диапазон [lo, hi]."""
    while angle_deg > hi:
        angle_deg -= 90.0
    while angle_deg < lo:
        angle_deg += 90.0
    return angle_deg


# ─── Определение угла ─────────────────────────────────────────────────────────

def detect_skew_hough(img:          np.ndarray,
                       threshold:    int   = 100,
                       min_line_len: int   = 50,
                       max_gap:      int   = 10,
                       angle_range:  float = 45.0) -> float:
    """
    Определяет угол наклона методом линий Хафа.

    Алгоритм:
        1. Binarize (Canny).
        2. HoughLinesP → множество углов.
        3. Фильтрация «горизонтальных» линий (|θ| < angle_range).
        4. Медиана угла.

    Args:
        img:          BGR или grayscale изображение.
        threshold:    Порог для HoughLinesP (мин. число пересечений).
        min_line_len: Минимальная длина линии в пикселях.
        max_gap:      Максимальный допустимый разрыв между точками линии.
        angle_range:  Максимальный угол отклонения от горизонтали (°).

    Returns:
        Угол наклона в градусах. 0.0 если линии не найдены.
    """
    gray  = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                              threshold=threshold,
                              minLineLength=min_line_len,
                              maxLineGap=max_gap)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= angle_range:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def detect_skew_projection(img:      np.ndarray,
                             n_angles: int   = 180,
                             lo:       float = -45.0,
                             hi:       float = 45.0) -> float:
    """
    Определяет угол наклона методом горизонтальных проекций.

    Принцип: для правильно ориентированного текста дисперсия горизонтальных
    проекций (суммы пикселей по строкам) максимальна — текстовые строки
    отображаются в чёткие пики.

    Args:
        img:      BGR или grayscale изображение.
        n_angles: Число угловых шагов в диапазоне [lo, hi].
        lo, hi:   Диапазон поиска угла в градусах.

    Returns:
        Угол (в °) с максимальной дисперсией проекций.
    """
    gray = _to_gray(img)
    # Бинаризация Otsu
    _, bw = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w  = bw.shape
    cx, cy = w // 2, h // 2

    best_angle = 0.0
    best_score = -1.0

    angles = np.linspace(lo, hi, n_angles)
    for angle in angles:
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(bw, M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderValue=0)
        proj  = rot.sum(axis=1).astype(np.float64)
        score = float(proj.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle


def detect_skew_fft(img:         np.ndarray,
                     sigma:       float = 2.0,
                     n_angles:    int   = 180,
                     peak_radius: float = 0.1) -> float:
    """
    Определяет угол наклона через спектральный анализ (ПФ).

    Алгоритм:
        1. Гауссовое размытие → Canny.
        2. 2D ПФ, логарифм амплитуды.
        3. Накапливаем суммы по радиальным полосам под разными углами.
        4. Угол максимальной суммы − 90° = угол наклона текста.

    Args:
        img:         BGR или grayscale изображение.
        sigma:       Гауссовый размыв (σ) перед ПФ.
        n_angles:    Число угловых шагов (0..180°).
        peak_radius: Доля радиуса изображения для выборки пика.

    Returns:
        Угол наклона в градусах.
    """
    gray = _to_gray(img)
    if sigma > 0:
        k    = max(3, int(sigma * 4) | 1)  # нечётное
        gray = cv2.GaussianBlur(gray, (k, k), sigma)

    edges = cv2.Canny(gray, 30, 100)
    h, w  = edges.shape

    # Паддинг до степени двойки (быстрый FFT)
    fft   = np.fft.fft2(edges.astype(np.float64))
    fft   = np.fft.fftshift(fft)
    mag   = np.log1p(np.abs(fft))

    cy, cx  = h // 2, w // 2
    radius  = min(cx, cy) * peak_radius
    best_a  = 0.0
    best_s  = -1.0

    theta_vals = np.linspace(0, math.pi, n_angles, endpoint=False)
    for theta in theta_vals:
        # Прямая через центр под углом theta
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        steps = np.arange(int(-radius * 1.5), int(radius * 1.5) + 1)
        xs    = np.clip((cx + steps * cos_t).astype(int), 0, w - 1)
        ys    = np.clip((cy + steps * sin_t).astype(int), 0, h - 1)
        score = float(mag[ys, xs].sum())
        if score > best_s:
            best_s = score
            best_a = math.degrees(theta)

    # Угол спектрального пика ⊥ текстовым строкам
    skew = _clamp_angle(best_a - 90.0)
    return skew


# ─── Коррекция изображения ────────────────────────────────────────────────────

def correct_skew(img:         np.ndarray,
                  angle_deg:   float,
                  border_mode: int = cv2.BORDER_REPLICATE,
                  scale:       float = 1.0) -> np.ndarray:
    """
    Корректирует наклон поворотом изображения вокруг центра.

    Args:
        img:         Входное изображение (BGR или grayscale).
        angle_deg:   Угол поворота в градусах (положительный = CCW).
        border_mode: cv2.BORDER_* для заполнения пустых областей.
        scale:       Масштаб при повороте (1.0 = без масштабирования).

    Returns:
        Повёрнутое изображение того же размера.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale)
    return cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=border_mode)


# ─── Оценка надёжности ────────────────────────────────────────────────────────

def skew_confidence(angles: List[float], tol: float = 2.0) -> float:
    """
    Оценивает надёжность оценки угла на основе согласованности методов.

    Args:
        angles: Углы, полученные от нескольких методов.
        tol:    Допустимое отклонение в градусах (диапазон ≤ tol → confidence=1).

    Returns:
        confidence ∈ [0, 1].
    """
    if not angles:
        return 0.0
    if len(angles) == 1:
        return 0.5  # Один метод — средняя уверенность

    spread = max(angles) - min(angles)
    # Плавное убывание от 1 до 0 при spread > tol
    return float(max(0.0, 1.0 - spread / (tol * 4)))


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def auto_correct_skew(img:         np.ndarray,
                       method:      str   = "hough",
                       border_mode: int   = cv2.BORDER_REPLICATE,
                       **method_kwargs) -> SkewResult:
    """
    Полный пайплайн: определение угла + коррекция.

    Args:
        img:            BGR или grayscale изображение.
        method:         'hough' | 'projection' | 'fft' | 'auto'.
                        'auto' усредняет все три метода.
        border_mode:    Заполнение пустых областей после поворота.
        **method_kwargs: Параметры для конкретного детектора.

    Returns:
        SkewResult.

    Raises:
        ValueError: Если method не распознан.
    """
    if method not in ("hough", "projection", "fft", "auto"):
        raise ValueError(f"Неизвестный метод: {method!r}. "
                          f"Допустимые: 'hough', 'projection', 'fft', 'auto'")

    params: Dict = {"method": method, **method_kwargs}

    if method == "hough":
        angle      = detect_skew_hough(img, **{k: v for k, v in method_kwargs.items()
                                                if k in ("threshold", "min_line_len",
                                                          "max_gap", "angle_range")})
        confidence = 0.7

    elif method == "projection":
        angle = detect_skew_projection(
            img,
            **{k: v for k, v in method_kwargs.items()
               if k in ("n_angles", "lo", "hi")},
        )
        confidence = 0.8

    elif method == "fft":
        angle = detect_skew_fft(
            img,
            **{k: v for k, v in method_kwargs.items()
               if k in ("sigma", "n_angles", "peak_radius")},
        )
        confidence = 0.6

    else:  # auto
        a_hough = detect_skew_hough(img)
        a_proj  = detect_skew_projection(img, n_angles=90)
        a_fft   = detect_skew_fft(img)
        angles  = [a_hough, a_proj, a_fft]
        angle   = float(np.median(angles))
        confidence = skew_confidence(angles)
        params["angles_all"] = angles

    corrected = correct_skew(img, angle, border_mode=border_mode)

    return SkewResult(
        corrected_image=corrected,
        angle_deg=angle,
        confidence=confidence,
        method=method,
        params=params,
    )


def batch_correct_skew(images:  List[np.ndarray],
                        method:  str = "hough",
                        **kwargs) -> List[SkewResult]:
    """
    Корректирует наклон у списка изображений.

    Args:
        images: Список BGR/grayscale изображений.
        method: Метод определения угла.
        **kwargs: Передаётся в auto_correct_skew.

    Returns:
        Список SkewResult.
    """
    return [auto_correct_skew(img, method=method, **kwargs) for img in images]
