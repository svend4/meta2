"""
Коррекция перспективных искажений фрагментов документа.

Исправляет трапециевидные деформации, возникающие при фотографировании
страниц под углом. Ключевой шаг перед сопоставлением краёв.

Методы обнаружения углов:
    contour  — контурный детектор (Canny → findContours → approxPolyDP)
    hough    — линии Хафа → четыре доминирующих линии → пересечения

Функции:
    order_corners        — упорядочивает 4 точки: (tl, tr, br, bl)
    four_point_transform — перспективное преобразование по 4 точкам
    detect_corners_contour — обнаружение по контурам
    detect_corners_hough   — обнаружение через Hough-линии
    correct_perspective  — полный пайплайн: detect + warp → PerspectiveResult
    auto_correct_perspective — выбор метода автоматически

Класс:
    PerspectiveResult — corrected, homography, src_pts, dst_pts, method
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── PerspectiveResult ────────────────────────────────────────────────────────

@dataclass
class PerspectiveResult:
    """
    Результат коррекции перспективы.

    Attributes:
        corrected:   Выправленное изображение.
        homography:  3×3 матрица гомографии (src → dst).
        src_pts:     Исходные 4 угловые точки (tl, tr, br, bl).
        dst_pts:     Целевые 4 угловые точки.
        method:      'contour' | 'hough' | 'manual' | 'none'
        confidence:  ∈ [0, 1] — уверенность в качестве коррекции.
        params:      Дополнительные параметры.
    """
    corrected:  np.ndarray
    homography: np.ndarray
    src_pts:    np.ndarray
    dst_pts:    np.ndarray
    method:     str
    confidence: float = 1.0
    params:     Dict  = field(default_factory=dict)

    def __repr__(self) -> str:
        h, w = self.corrected.shape[:2]
        return (f"PerspectiveResult(method={self.method!r}, "
                f"confidence={self.confidence:.3f}, "
                f"output={w}×{h})")


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки в порядке (tl, tr, br, bl).

    Args:
        pts: (4, 2) float32 или float64 массив точек.

    Returns:
        (4, 2) float32 в порядке [top-left, top-right, bottom-right, bottom-left].
    """
    pts  = np.array(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]    # tl: min(x+y)
    rect[2] = pts[np.argmax(s)]    # br: max(x+y)
    rect[1] = pts[np.argmin(diff)] # tr: min(y-x)
    rect[3] = pts[np.argmax(diff)] # bl: max(y-x)

    return rect


def _dst_rect(src_rect: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Вычисляет целевой прямоугольник и размеры выходного изображения.

    Returns:
        (dst_pts, width, height)
    """
    tl, tr, br, bl = src_rect

    w_top    = np.linalg.norm(tr - tl)
    w_bottom = np.linalg.norm(br - bl)
    h_left   = np.linalg.norm(bl - tl)
    h_right  = np.linalg.norm(br - tr)

    max_w = max(int(w_top), int(w_bottom)) + 1
    max_h = max(int(h_left), int(h_right)) + 1

    dst = np.array([
        [0,         0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0,         max_h - 1],
    ], dtype=np.float32)

    return dst, max_w, max_h


def four_point_transform(img: np.ndarray,
                           pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполняет перспективное преобразование по 4 точкам.

    Args:
        img: Входное изображение (BGR или grayscale).
        pts: (4, 2) угловые точки (любой порядок — будут упорядочены).

    Returns:
        (warped, H, dst_pts):
            warped   — выровненное изображение.
            H        — матрица гомографии 3×3.
            dst_pts  — целевые точки.
    """
    rect    = order_corners(pts)
    dst, w, h = _dst_rect(rect)

    H = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return warped, H, dst


# ─── Обнаружение углов ────────────────────────────────────────────────────────

def detect_corners_contour(img:           np.ndarray,
                             canny_lo:     int   = 50,
                             canny_hi:     int   = 150,
                             approx_eps:   float = 0.02,
                             min_area_frac: float = 0.1) -> Optional[np.ndarray]:
    """
    Обнаруживает 4 угловые точки документа через контурный анализ.

    Алгоритм:
        1. Grayscale + GaussianBlur.
        2. Canny.
        3. findContours → сортировка по площади (убывание).
        4. approxPolyDP → ищем первый 4-угольник с площадью > min_area_frac.

    Args:
        img:            BGR или grayscale изображение.
        canny_lo/hi:    Пороги Canny.
        approx_eps:     Коэффициент аппроксимации полигона.
        min_area_frac:  Минимальная доля площади изображения.

    Returns:
        (4, 2) float32 массив угловых точек или None.
    """
    gray  = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * min_area_frac

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def detect_corners_hough(img:         np.ndarray,
                           threshold:   int   = 100,
                           min_length:  int   = 50,
                           max_gap:     int   = 10) -> Optional[np.ndarray]:
    """
    Обнаруживает 4 угловые точки через линии Хафа.

    Алгоритм:
        1. Canny.
        2. HoughLinesP → множество линий.
        3. Кластеризация по углу в 4 группы (≈ горизонт. и вертикал.).
        4. Пересечения доминирующих линий → 4 угла.

    Args:
        img:        BGR или grayscale изображение.
        threshold:  Порог HoughLinesP.
        min_length: Минимальная длина линии.
        max_gap:    Максимальный разрыв.

    Returns:
        (4, 2) float32 массив угловых точек или None.
    """
    gray   = _to_gray(img)
    edges  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines  = cv2.HoughLinesP(edges, 1, math.pi / 180,
                               threshold=threshold,
                               minLineLength=min_length,
                               maxLineGap=max_gap)

    if lines is None or len(lines) < 4:
        return None

    # Кластеризация: горизонтальные (|θ| < 30°) и вертикальные (|θ-90°| < 30°)
    horiz: List[np.ndarray] = []
    vert:  List[np.ndarray] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            horiz.append(line[0])
        elif 60 < angle < 120:
            vert.append(line[0])

    if len(horiz) < 2 or len(vert) < 2:
        return None

    def _median_line(seg_list):
        arr = np.array(seg_list, dtype=np.float32)
        return arr[len(arr) // 2]

    horiz.sort(key=lambda s: (s[1] + s[3]) / 2)
    vert.sort(key=lambda  s: (s[0] + s[2]) / 2)

    top_line    = _median_line(horiz[:len(horiz)//2 or 1])
    bottom_line = _median_line(horiz[len(horiz)//2:])
    left_line   = _median_line(vert[:len(vert)//2 or 1])
    right_line  = _median_line(vert[len(vert)//2:])

    def _intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return [x, y]

    tl = _intersect(top_line,    left_line)
    tr = _intersect(top_line,    right_line)
    br = _intersect(bottom_line, right_line)
    bl = _intersect(bottom_line, left_line)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def correct_perspective(img:     np.ndarray,
                         corners: Optional[np.ndarray] = None,
                         method:  str = "contour",
                         **kwargs) -> PerspectiveResult:
    """
    Корректирует перспективные искажения.

    Если corners переданы явно — использует их напрямую.
    Иначе — вызывает детектор.

    Args:
        img:     BGR или grayscale изображение.
        corners: (4, 2) float32 или None (автодетектирование).
        method:  'contour' | 'hough' (используется только при corners=None).
        **kwargs: Параметры детектора.

    Returns:
        PerspectiveResult. Если углы не найдены — возвращает исходное
        изображение с единичной гомографией.
    """
    if method not in ("contour", "hough", "manual", "none"):
        raise ValueError(f"Неизвестный метод: {method!r}. "
                          f"Допустимые: 'contour', 'hough', 'manual', 'none'")

    h, w = img.shape[:2]
    identity_corners = np.array([[0, 0], [w - 1, 0],
                                   [w - 1, h - 1], [0, h - 1]],
                                  dtype=np.float32)

    if corners is None and method in ("contour", "hough"):
        if method == "contour":
            corners = detect_corners_contour(img, **{
                k: v for k, v in kwargs.items()
                if k in ("canny_lo", "canny_hi", "approx_eps", "min_area_frac")
            })
        else:
            corners = detect_corners_hough(img, **{
                k: v for k, v in kwargs.items()
                if k in ("threshold", "min_length", "max_gap")
            })

    if corners is None:
        # Углы не найдены — возвращаем оригинал с identity H
        H = np.eye(3, dtype=np.float64)
        return PerspectiveResult(
            corrected=img.copy(),
            homography=H,
            src_pts=identity_corners,
            dst_pts=identity_corners,
            method=method,
            confidence=0.0,
            params={"corners_detected": False},
        )

    src_rect = order_corners(corners)
    warped, H, dst_pts = four_point_transform(img, src_rect)

    # Оцениваем уверенность по отношению площади к исходной
    warped_area = warped.shape[0] * warped.shape[1]
    src_area    = cv2.contourArea(src_rect)
    confidence  = float(np.clip(src_area / max(h * w, 1), 0.0, 1.0))

    return PerspectiveResult(
        corrected=warped,
        homography=H,
        src_pts=src_rect,
        dst_pts=dst_pts,
        method=method,
        confidence=confidence,
        params={"corners_detected": True,
                "src_area": float(src_area),
                "warped_size": (warped.shape[1], warped.shape[0])},
    )


def auto_correct_perspective(img: np.ndarray,
                               **kwargs) -> PerspectiveResult:
    """
    Автоматически выбирает метод коррекции (contour → hough → none).

    Пробует contour; если не нашёл углы — пробует hough; если и он не
    нашёл — возвращает оригинал с confidence=0.

    Args:
        img:     Входное изображение.
        **kwargs: Параметры детекторов.

    Returns:
        PerspectiveResult.
    """
    for method in ("contour", "hough"):
        result = correct_perspective(img, method=method, **kwargs)
        if result.confidence > 0.0:
            return result

    return correct_perspective(img, corners=None, method="none")


def batch_correct_perspective(images: List[np.ndarray],
                                method: str = "contour",
                                **kwargs) -> List[PerspectiveResult]:
    """
    Корректирует перспективу у списка изображений.

    Args:
        images: Список BGR/grayscale изображений.
        method: Метод детектирования.
        **kwargs: Параметры.

    Returns:
        Список PerspectiveResult.
    """
    return [correct_perspective(img, method=method, **kwargs) for img in images]
