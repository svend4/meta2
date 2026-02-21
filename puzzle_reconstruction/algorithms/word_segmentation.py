"""
Сегментация слов и строк на восстановленном изображении документа.

Работает поверх классических морфологических операций OpenCV — без OCR —
что позволяет применять её на ранних этапах пайплайна для оценки связности
текста.

Алгоритм:
    1. Бинаризация Otsu / Sauvola (адаптивная).
    2. Морфологическое закрытие (dilation) вдоль горизонтали → слияние букв
       в слова.
    3. Нахождение контуров — каждый контур = одно слово.
    4. Группировка слов в строки по Y-coordinate clustering.

Классы:
    WordBox              — ограничивающий прямоугольник слова
    LineSegment          — строка документа (слова + общий bbox)
    WordSegmentationResult — полный результат (слова, строки, текст)

Функции:
    binarize             — Otsu / adaptive / sauvola бинаризация
    segment_words        — возвращает список WordBox
    merge_line_words     — группирует WordBox в LineSegment
    segment_lines        — обёртка: words → lines
    words_to_text        — dummy-текст из количества слов (без OCR)
    segment_document     — полный пайплайн → WordSegmentationResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── WordBox ──────────────────────────────────────────────────────────────────

@dataclass
class WordBox:
    """
    Ограничивающий прямоугольник предполагаемого слова.

    Attributes:
        x, y:       Левый верхний угол.
        w, h:       Ширина и высота.
        line_idx:   Индекс строки (заполняется после группировки).
        confidence: Приблизительная уверенность (0.0–1.0).
    """
    x:          int
    y:          int
    w:          int
    h:          int
    line_idx:   int   = -1
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else 0.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h)"""
        return (self.x, self.y, self.w, self.h)

    def iou(self, other: "WordBox") -> float:
        """Intersection over Union."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        return (f"WordBox(x={self.x}, y={self.y}, "
                f"w={self.w}, h={self.h}, line={self.line_idx})")


# ─── LineSegment ──────────────────────────────────────────────────────────────

@dataclass
class LineSegment:
    """
    Одна строка документа.

    Attributes:
        line_idx: Индекс строки (сверху вниз).
        words:    Слова в строке, отсортированные слева направо.
        bbox:     (x, y, w, h) общий bbox строки.
    """
    line_idx: int
    words:    List[WordBox] = field(default_factory=list)
    bbox:     Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def y_center(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0

    @property
    def x_start(self) -> int:
        return self.bbox[0]

    def avg_word_height(self) -> float:
        if not self.words:
            return 0.0
        return float(np.mean([w.h for w in self.words]))

    def __repr__(self) -> str:
        return (f"LineSegment(line={self.line_idx}, "
                f"n_words={self.n_words}, "
                f"bbox={self.bbox})")


# ─── WordSegmentationResult ───────────────────────────────────────────────────

@dataclass
class WordSegmentationResult:
    """
    Итог сегментации слов и строк.

    Attributes:
        words:         Все найденные WordBox.
        lines:         Строки (LineSegment), отсортированные сверху вниз.
        n_words:       Число слов.
        n_lines:       Число строк.
        image_shape:   (h, w) исходного изображения.
        binarize_method: Метод бинаризации, использованный при сегментации.
    """
    words:            List[WordBox]
    lines:            List[LineSegment]
    image_shape:      Tuple[int, int]
    binarize_method:  str = "otsu"

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def avg_words_per_line(self) -> float:
        if not self.lines:
            return 0.0
        return self.n_words / self.n_lines

    def words_on_line(self, line_idx: int) -> List[WordBox]:
        if 0 <= line_idx < len(self.lines):
            return self.lines[line_idx].words
        return []

    def summary(self) -> str:
        h, w = self.image_shape
        return (f"WordSegmentationResult("
                f"n_words={self.n_words}, "
                f"n_lines={self.n_lines}, "
                f"avg_words_per_line={self.avg_words_per_line:.1f}, "
                f"image={w}×{h})")

    def __repr__(self) -> str:
        return self.summary()


# ─── Бинаризация ──────────────────────────────────────────────────────────────

def binarize(img:    np.ndarray,
              method: str   = "otsu",
              block:  int   = 51,
              c:      float = 10.0) -> np.ndarray:
    """
    Бинаризует изображение.

    Args:
        img:    BGR или grayscale изображение.
        method: 'otsu' | 'adaptive' | 'sauvola'
        block:  Размер блока для адаптивных методов (нечётное).
        c:      Константа вычитания (для adaptive/sauvola).

    Returns:
        uint8 бинарное изображение (255=текст, 0=фон). Форма = gray.shape.

    Raises:
        ValueError: Если method не распознан.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Лёгкое размытие — уберём пикселный шум
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = block if block % 2 == 1 else block + 1
        bw    = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       block, int(c))
    elif method == "sauvola":
        # Реализация через интегральные изображения
        block  = block if block % 2 == 1 else block + 1
        half   = block // 2
        h, w   = gray.shape
        gf     = gray.astype(np.float64)
        # Интегральные изображения суммы и суммы квадратов
        s1 = cv2.integral(gf)
        s2 = cv2.integral(gf ** 2)

        n   = block * block
        bw  = np.zeros_like(gray)

        for y in range(h):
            for x in range(w):
                x1 = max(0, x - half)
                y1 = max(0, y - half)
                x2 = min(w, x + half + 1)
                y2 = min(h, y + half + 1)
                area  = (x2 - x1) * (y2 - y1)
                sm    = (s1[y2, x2] - s1[y1, x2]
                          - s1[y2, x1] + s1[y1, x1])
                sm2   = (s2[y2, x2] - s2[y1, x2]
                          - s2[y2, x1] + s2[y1, x1])
                mean  = sm / area
                var   = sm2 / area - mean ** 2
                std   = math.sqrt(max(var, 0.0))
                t     = mean * (1.0 + c * (std / 128.0 - 1.0))
                bw[y, x] = 255 if gray[y, x] < t else 0
    else:
        raise ValueError(f"Неизвестный метод бинаризации: {method!r}. "
                          f"Допустимые: 'otsu', 'adaptive', 'sauvola'")

    return bw


# ─── Сегментация слов ─────────────────────────────────────────────────────────

def segment_words(img:           np.ndarray,
                   binarize_method: str = "otsu",
                   dilation_w:    int   = 20,
                   dilation_h:    int   = 3,
                   min_area:      int   = 50,
                   min_width:     int   = 5,
                   min_height:    int   = 5) -> List[WordBox]:
    """
    Находит ограничивающие прямоугольники слов на изображении.

    Алгоритм:
        1. Бинаризация.
        2. Горизонтальное dilate → слияние соседних символов в слова.
        3. findContours → bounding rect.
        4. Фильтрация по area / width / height.

    Args:
        img:              BGR или grayscale изображение.
        binarize_method:  Метод бинаризации.
        dilation_w:       Ширина ядра дилатации (горизонтальное закрытие).
        dilation_h:       Высота ядра дилатации.
        min_area:         Минимальная площадь bbox слова.
        min_width:        Минимальная ширина bbox.
        min_height:       Минимальная высота bbox.

    Returns:
        Список WordBox, отсортированных по (y, x).
    """
    bw      = binarize(img, method=binarize_method)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_w, dilation_h))
    dilated = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    words: List[WordBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        if w < min_width or h < min_height:
            continue
        words.append(WordBox(x=x, y=y, w=w, h=h))

    # Сортировка по (y, x)
    words.sort(key=lambda wb: (wb.y, wb.x))
    return words


# ─── Группировка в строки ─────────────────────────────────────────────────────

def merge_line_words(words:    List[WordBox],
                      line_gap: Optional[float] = None) -> List[LineSegment]:
    """
    Группирует WordBox в строки по Y-center.

    Алгоритм: жадная кластеризация — новая строка, если cy текущего слова
    отличается от медианы cy текущей строки более чем на line_gap.

    Args:
        words:    Список WordBox (рекомендуется отсортированный по y).
        line_gap: Максимальное расстояние по Y для отнесения к одной строке.
                  Если None — auto (среднее h слова * 0.8).

    Returns:
        Список LineSegment, отсортированных по y_center.
    """
    if not words:
        return []

    if line_gap is None:
        avg_h    = float(np.mean([w.h for w in words]))
        line_gap = avg_h * 0.8

    rows: List[List[WordBox]] = []
    for wb in sorted(words, key=lambda w: w.cy):
        placed = False
        for row in rows:
            median_cy = float(np.median([w.cy for w in row]))
            if abs(wb.cy - median_cy) <= line_gap:
                row.append(wb)
                placed = True
                break
        if not placed:
            rows.append([wb])

    lines: List[LineSegment] = []
    for idx, row in enumerate(rows):
        row.sort(key=lambda w: w.x)
        for wb in row:
            wb.line_idx = idx
        # Общий bbox строки
        xs = min(w.x  for w in row)
        ys = min(w.y  for w in row)
        xe = max(w.x2 for w in row)
        ye = max(w.y2 for w in row)
        lines.append(LineSegment(
            line_idx=idx,
            words=row,
            bbox=(xs, ys, xe - xs, ye - ys),
        ))

    lines.sort(key=lambda ln: ln.y_center)
    return lines


def segment_lines(img:             np.ndarray,
                   binarize_method: str = "otsu",
                   **word_kwargs) -> List[LineSegment]:
    """
    Обёртка: изображение → List[LineSegment].

    Args:
        img:              Входное изображение.
        binarize_method:  Метод бинаризации.
        **word_kwargs:    Параметры для segment_words.

    Returns:
        Список LineSegment.
    """
    words = segment_words(img, binarize_method=binarize_method, **word_kwargs)
    return merge_line_words(words)


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def segment_document(img:             np.ndarray,
                      binarize_method: str = "otsu",
                      dilation_w:      int = 20,
                      dilation_h:      int = 3,
                      min_area:        int = 50,
                      line_gap:        Optional[float] = None) -> WordSegmentationResult:
    """
    Полный пайплайн: изображение → WordSegmentationResult.

    Args:
        img:              BGR или grayscale изображение.
        binarize_method:  'otsu' | 'adaptive' | 'sauvola'.
        dilation_w:       Ширина горизонтального ядра дилатации.
        dilation_h:       Высота ядра.
        min_area:         Минимальная площадь bbox слова.
        line_gap:         Порог Y-расстояния для группировки в строки.

    Returns:
        WordSegmentationResult.
    """
    words = segment_words(img,
                           binarize_method=binarize_method,
                           dilation_w=dilation_w,
                           dilation_h=dilation_h,
                           min_area=min_area)

    lines = merge_line_words(words, line_gap=line_gap)
    h, w  = img.shape[:2]

    return WordSegmentationResult(
        words=words,
        lines=lines,
        image_shape=(h, w),
        binarize_method=binarize_method,
    )
