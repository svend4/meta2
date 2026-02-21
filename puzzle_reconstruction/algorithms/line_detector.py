"""
Обнаружение текстовых строк в фрагментах документа.

Выявляет горизонтальные строки текста двумя методами:
проекция яркости (быстрый, надёжный) и преобразование Хафа (точный угол).
Результат используется для выравнивания фрагментов и оценки согласованности.

Классы:
    TextLine            — описание одной обнаруженной строки
    LineDetectionResult — результат обнаружения (строки + метрики)

Функции:
    detect_lines_projection — обнаружение через горизонтальную проекцию
    detect_lines_hough      — обнаружение через преобразование Хафа
    estimate_line_metrics   — средняя высота строки и межстрочный интервал
    filter_lines            — фильтрация строк по размеру и ширине
    detect_text_lines       — автоматический выбор метода + постобработка
    batch_detect_lines      — пакетная обработка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── TextLine ─────────────────────────────────────────────────────────────────

@dataclass
class TextLine:
    """
    Описание одной обнаруженной строки текста.

    Attributes:
        y_top:      Верхняя граница строки (пиксель).
        y_bottom:   Нижняя граница строки (пиксель).
        x_left:     Левая граница строки.
        x_right:    Правая граница строки.
        height:     Высота строки = y_bottom − y_top.
        width:      Ширина строки = x_right − x_left.
        confidence: Уверенность в обнаружении ∈ [0,1].
    """
    y_top:      int
    y_bottom:   int
    x_left:     int
    x_right:    int
    height:     int
    width:      int
    confidence: float = 1.0

    @property
    def center_y(self) -> float:
        return (self.y_top + self.y_bottom) / 2.0

    def __repr__(self) -> str:
        return (f"TextLine(y=[{self.y_top},{self.y_bottom}], "
                f"x=[{self.x_left},{self.x_right}], "
                f"h={self.height}, conf={self.confidence:.2f})")


# ─── LineDetectionResult ──────────────────────────────────────────────────────

@dataclass
class LineDetectionResult:
    """
    Результат обнаружения строк.

    Attributes:
        lines:        Список обнаруженных TextLine.
        method:       Использованный метод ('projection' | 'hough' | 'auto').
        line_height:  Средняя высота строки (пикс).
        line_spacing: Средний межстрочный интервал (пикс).
        skew_angle:   Оценка угла наклона текста (градусы).
        n_lines:      Число обнаруженных строк.
        params:       Параметры метода.
    """
    lines:        List[TextLine]
    method:       str
    line_height:  float
    line_spacing: float
    skew_angle:   float
    n_lines:      int
    params:       Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"LineDetectionResult(n={self.n_lines}, "
                f"method={self.method!r}, "
                f"height={self.line_height:.1f}, "
                f"spacing={self.line_spacing:.1f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Быстрая бинаризация: Otsu + инверсия (текст → 255, фон → 0)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


# ─── estimate_line_metrics ────────────────────────────────────────────────────

def estimate_line_metrics(lines: List[TextLine]) -> Tuple[float, float]:
    """
    Оценивает среднюю высоту строки и межстрочный интервал.

    Args:
        lines: Список TextLine, отсортированных по y_top.

    Returns:
        (line_height, line_spacing) — средние значения в пикселях.
        Если строк < 2, spacing = 0.0.
    """
    if not lines:
        return (0.0, 0.0)

    heights = [float(ln.height) for ln in lines]
    avg_h   = float(np.mean(heights))

    if len(lines) < 2:
        return (avg_h, 0.0)

    sorted_lines = sorted(lines, key=lambda l: l.y_top)
    spacings     = []
    for i in range(1, len(sorted_lines)):
        gap = sorted_lines[i].y_top - sorted_lines[i - 1].y_bottom
        if gap >= 0:
            spacings.append(float(gap))

    avg_s = float(np.mean(spacings)) if spacings else 0.0
    return (avg_h, avg_s)


# ─── filter_lines ─────────────────────────────────────────────────────────────

def filter_lines(lines:          List[TextLine],
                  min_height:    int   = 4,
                  max_height:    int   = 80,
                  min_width_frac: float = 0.2,
                  img_width:     int   = 100) -> List[TextLine]:
    """
    Фильтрует строки по высоте и относительной ширине.

    Args:
        lines:          Исходный список строк.
        min_height:     Минимальная высота строки (пикс).
        max_height:     Максимальная высота строки (пикс).
        min_width_frac: Минимальная доля ширины изображения для строки.
        img_width:      Ширина изображения (пикс).

    Returns:
        Отфильтрованный список TextLine.
    """
    min_w = int(img_width * min_width_frac)
    return [
        ln for ln in lines
        if min_height <= ln.height <= max_height and ln.width >= min_w
    ]


# ─── detect_lines_projection ─────────────────────────────────────────────────

def detect_lines_projection(img:           np.ndarray,
                              min_height:   int   = 4,
                              max_height:   int   = 80,
                              min_width_frac: float = 0.3,
                              bw_thresh:    int   = 128) -> LineDetectionResult:
    """
    Обнаруживает строки через горизонтальную проекцию (RLE).

    Горизонтальная проекция = сумма пикселей по каждой строке (axis=1).
    Строки с проекцией > bw_thresh считаются «текстовыми».
    Последовательные «текстовые» строки группируются в TextLine.

    Args:
        img:            BGR или grayscale изображение.
        min_height:     Мин. высота строки.
        max_height:     Макс. высота строки.
        min_width_frac: Мин. доля ширины для валидной строки.
        bw_thresh:      Порог суммы проекции.

    Returns:
        LineDetectionResult с method='projection'.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    bw   = _binarize(gray)
    proj = bw.sum(axis=1).astype(np.float32)   # (h,)

    # Бинарная маска: строки с проекцией > bw_thresh считаются текстовыми
    text_mask = proj > bw_thresh

    lines: List[TextLine] = []
    in_line = False
    y_start = 0

    for y in range(h):
        if text_mask[y] and not in_line:
            in_line = True
            y_start = y
        elif not text_mask[y] and in_line:
            in_line = False
            y_end = y
            ht    = y_end - y_start
            # Ширина — от левого до правого ненулевого пикселя на этом участке
            strip = bw[y_start:y_end, :]
            cols  = np.where(strip.sum(axis=0) > 0)[0]
            if len(cols) > 0:
                x_left  = int(cols[0])
                x_right = int(cols[-1]) + 1
            else:
                x_left  = 0
                x_right = w
            conf = float(np.clip(proj[y_start:y_end].mean() / max(proj.max(), 1.0),
                                 0.0, 1.0))
            lines.append(TextLine(
                y_top=y_start, y_bottom=y_end,
                x_left=x_left, x_right=x_right,
                height=ht, width=x_right - x_left,
                confidence=conf,
            ))

    if in_line:
        ht   = h - y_start
        strip = bw[y_start:, :]
        cols  = np.where(strip.sum(axis=0) > 0)[0]
        if len(cols) > 0:
            x_left, x_right = int(cols[0]), int(cols[-1]) + 1
        else:
            x_left, x_right = 0, w
        lines.append(TextLine(
            y_top=y_start, y_bottom=h,
            x_left=x_left, x_right=x_right,
            height=ht, width=x_right - x_left,
            confidence=0.5,
        ))

    lines = filter_lines(lines, min_height=min_height, max_height=max_height,
                          min_width_frac=min_width_frac, img_width=w)
    line_h, spacing = estimate_line_metrics(lines)

    return LineDetectionResult(
        lines=lines,
        method="projection",
        line_height=line_h,
        line_spacing=spacing,
        skew_angle=0.0,   # Проекция не оценивает угол
        n_lines=len(lines),
        params={
            "min_height": min_height,
            "max_height": max_height,
            "min_width_frac": min_width_frac,
            "bw_thresh": bw_thresh,
        },
    )


# ─── detect_lines_hough ───────────────────────────────────────────────────────

def detect_lines_hough(img:             np.ndarray,
                        threshold:      int   = 80,
                        min_len_frac:   float = 0.3,
                        max_gap:        int   = 10,
                        min_height:     int   = 4,
                        max_height:     int   = 80) -> LineDetectionResult:
    """
    Обнаруживает строки через преобразование Хафа (HoughLinesP).

    Горизонтальные отрезки группируются по Y-координате в строки.
    Оценивается угол наклона текста.

    Args:
        img:          BGR или grayscale изображение.
        threshold:    Порог аккумулятора Хафа.
        min_len_frac: Мин. длина отрезка как доля ширины.
        max_gap:      Макс. зазор между отрезками.
        min_height:   Мин. толщина сгруппированного кластера (пикс).
        max_height:   Макс. толщина кластера (пикс).

    Returns:
        LineDetectionResult с method='hough'.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150)
    min_len = int(w * min_len_frac)

    raw = cv2.HoughLinesP(edges, 1, np.pi / 180,
                           threshold=threshold,
                           minLineLength=max(min_len, 10),
                           maxLineGap=max_gap)

    angles: List[float] = []
    y_groups: List[List[int]] = []   # список y-координат отрезков

    if raw is not None:
        for seg in raw:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = float(np.degrees(np.arctan2(dy, dx)))
            # Только близкие к горизонтальным (±30°)
            if abs(angle) < 30:
                angles.append(angle)
                y_mid = (y1 + y2) // 2
                # Найдём существующую группу
                placed = False
                for grp in y_groups:
                    if abs(y_mid - int(np.mean(grp))) < 10:
                        grp.append(y_mid)
                        placed = True
                        break
                if not placed:
                    y_groups.append([y_mid])

    skew_angle = float(np.median(angles)) if angles else 0.0

    lines: List[TextLine] = []
    for grp in y_groups:
        y_mid_g = int(np.mean(grp))
        ht_g    = max(min_height, min(int(np.std(grp)) * 2 + min_height, max_height))
        y_top   = max(0, y_mid_g - ht_g // 2)
        y_bot   = min(h, y_top + ht_g)
        lines.append(TextLine(
            y_top=y_top, y_bottom=y_bot,
            x_left=0, x_right=w,
            height=y_bot - y_top, width=w,
            confidence=float(np.clip(len(grp) / 5.0, 0.0, 1.0)),
        ))

    lines = [ln for ln in lines if min_height <= ln.height <= max_height]
    lines.sort(key=lambda l: l.y_top)

    line_h, spacing = estimate_line_metrics(lines)

    return LineDetectionResult(
        lines=lines,
        method="hough",
        line_height=line_h,
        line_spacing=spacing,
        skew_angle=skew_angle,
        n_lines=len(lines),
        params={
            "threshold": threshold,
            "min_len_frac": min_len_frac,
            "max_gap": max_gap,
        },
    )


# ─── detect_text_lines ────────────────────────────────────────────────────────

def detect_text_lines(img:             np.ndarray,
                       method:         str   = "auto",
                       min_height:     int   = 4,
                       max_height:     int   = 80,
                       min_width_frac: float = 0.3) -> LineDetectionResult:
    """
    Автоматическое обнаружение строк с выбором метода.

    'auto': сначала пробует projection; если строк < 2, пробует hough.

    Args:
        img:            BGR или grayscale изображение.
        method:         'projection' | 'hough' | 'auto'.
        min_height:     Мин. высота строки.
        max_height:     Макс. высота строки.
        min_width_frac: Мин. ширина строки как доля ширины изображения.

    Returns:
        LineDetectionResult с method='auto' или именем конкретного метода.

    Raises:
        ValueError: Если метод неизвестен.
    """
    if method == "projection":
        return detect_lines_projection(img, min_height=min_height,
                                        max_height=max_height,
                                        min_width_frac=min_width_frac)
    elif method == "hough":
        return detect_lines_hough(img, min_height=min_height,
                                   max_height=max_height)
    elif method == "auto":
        r = detect_lines_projection(img, min_height=min_height,
                                     max_height=max_height,
                                     min_width_frac=min_width_frac)
        if r.n_lines >= 2:
            # Достаточно строк — возвращаем результат с методом 'auto'
            return LineDetectionResult(
                lines=r.lines, method="auto",
                line_height=r.line_height, line_spacing=r.line_spacing,
                skew_angle=r.skew_angle, n_lines=r.n_lines,
                params={**r.params, "fallback": False},
            )
        # Fallback на Hough
        r2 = detect_lines_hough(img, min_height=min_height,
                                  max_height=max_height)
        return LineDetectionResult(
            lines=r2.lines, method="auto",
            line_height=r2.line_height, line_spacing=r2.line_spacing,
            skew_angle=r2.skew_angle, n_lines=r2.n_lines,
            params={**r2.params, "fallback": True},
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'projection', 'hough', or 'auto'."
        )


# ─── batch_detect_lines ───────────────────────────────────────────────────────

def batch_detect_lines(images: List[np.ndarray],
                        method: str = "auto",
                        **kwargs) -> List[LineDetectionResult]:
    """
    Пакетное обнаружение строк для списка изображений.

    Args:
        images: Список BGR или grayscale изображений.
        method: 'projection' | 'hough' | 'auto'.
        **kwargs: Параметры, передаваемые в detect_text_lines.

    Returns:
        Список LineDetectionResult (по одному на изображение).

    Raises:
        ValueError: Если метод неизвестен.
    """
    return [detect_text_lines(img, method=method, **kwargs) for img in images]
