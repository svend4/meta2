"""
Утилиты визуализации для отладки пайплайна сборки документа.

Все функции работают с NumPy-массивами (BGR) и возвращают аннотированные
изображения. Не зависят от matplotlib — только OpenCV.

Функции:
    draw_word_boxes          — рамки слов + номера строк
    draw_fragment_boxes      — рамки фрагментов из FragmentBox
    draw_edge_matches        — side-by-side с нарисованными соответствиями
    draw_contour             — наложение контура
    draw_assembly_layout     — схема размещения фрагментов
    draw_skew_angle          — аннотация угла наклона
    draw_confidence_bar      — цветная шкала confidence
    tile_images              — мозаика нескольких изображений

Класс:
    VisConfig — цвета, толщины, шрифт
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ─── VisConfig ────────────────────────────────────────────────────────────────

@dataclass
class VisConfig:
    """
    Параметры отображения (цвета, толщины, шрифт).

    Все цвета в формате BGR (OpenCV).
    """
    word_box_color:     Tuple[int, int, int] = (0, 200, 0)
    frag_box_color:     Tuple[int, int, int] = (255, 80, 0)
    contour_color:      Tuple[int, int, int] = (0, 120, 255)
    match_line_color:   Tuple[int, int, int] = (0, 255, 255)
    text_color:         Tuple[int, int, int] = (0, 0, 0)
    bg_color:           Tuple[int, int, int] = (255, 255, 255)
    line_thickness:     int   = 1
    font:               int   = cv2.FONT_HERSHEY_SIMPLEX
    font_scale:         float = 0.4
    font_thickness:     int   = 1
    tile_gap:           int   = 4
    tile_bg:            Tuple[int, int, int] = (200, 200, 200)


_DEFAULT_CFG = VisConfig()


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Конвертирует grayscale → BGR, RGB → BGR; BGR возвращает без изменений."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _put_label(img:       np.ndarray,
                text:      str,
                x:         int,
                y:         int,
                cfg:       VisConfig = _DEFAULT_CFG,
                bg_alpha:  float     = 0.5) -> None:
    """Рисует текст с прозрачным фоном."""
    (tw, th), _ = cv2.getTextSize(text, cfg.font, cfg.font_scale,
                                   cfg.font_thickness)
    x2 = min(x + tw + 4, img.shape[1] - 1)
    y2 = min(y + 2, img.shape[0] - 1)
    y1 = max(y - th - 2, 0)
    roi = img[y1:y2, x:x2]
    if roi.size > 0:
        overlay = roi.copy()
        overlay[:] = cfg.bg_color
        cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)
    cv2.putText(img, text, (x, y), cfg.font, cfg.font_scale,
                cfg.text_color, cfg.font_thickness, cv2.LINE_AA)


# ─── Рисование боксов слов ────────────────────────────────────────────────────

def draw_word_boxes(img:    np.ndarray,
                     words:  Sequence,
                     cfg:    VisConfig = _DEFAULT_CFG,
                     label:  bool      = True) -> np.ndarray:
    """
    Рисует рамки WordBox и подписи строк.

    Args:
        img:   Входное изображение (не изменяется).
        words: Список WordBox из word_segmentation.
        cfg:   Параметры отображения.
        label: Если True — подписывает номер строки.

    Returns:
        Аннотированное изображение.
    """
    out = _ensure_bgr(img)
    for wb in words:
        x, y, w, h = wb.x, wb.y, wb.w, wb.h
        cv2.rectangle(out, (x, y), (x + w, y + h),
                       cfg.word_box_color, cfg.line_thickness)
        if label and wb.line_idx >= 0:
            _put_label(out, str(wb.line_idx), x, y - 2, cfg)
    return out


# ─── Рисование боксов фрагментов ─────────────────────────────────────────────

def draw_fragment_boxes(img:    np.ndarray,
                         boxes:  Sequence,
                         cfg:    VisConfig = _DEFAULT_CFG,
                         label:  bool      = True) -> np.ndarray:
    """
    Рисует рамки FragmentBox.

    Args:
        img:   Холст (обычно пустой белый).
        boxes: Список FragmentBox из layout_verifier.
        cfg:   Параметры отображения.
        label: Если True — подписывает fid.

    Returns:
        Аннотированное изображение.
    """
    out = _ensure_bgr(img)
    for fb in boxes:
        x1, y1 = int(fb.x), int(fb.y)
        x2, y2 = int(fb.x2), int(fb.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2),
                       cfg.frag_box_color, cfg.line_thickness)
        if label:
            _put_label(out, f"fid={fb.fid}", x1 + 2, y1 + 14, cfg)
    return out


# ─── Рисование совпадений ─────────────────────────────────────────────────────

def draw_edge_matches(img1:    np.ndarray,
                       img2:    np.ndarray,
                       matches: Sequence,
                       cfg:     VisConfig = _DEFAULT_CFG,
                       max_matches: int   = 50) -> np.ndarray:
    """
    Рисует side-by-side изображения с линиями совпадений (KeypointMatch).

    Args:
        img1, img2:  Входные изображения (не обязаны быть одного размера).
        matches:     Список KeypointMatch.
        cfg:         Параметры.
        max_matches: Максимальное число рисуемых соответствий.

    Returns:
        Широкое изображение (img1 | gap | img2) с соединительными линиями.
    """
    i1 = _ensure_bgr(img1)
    i2 = _ensure_bgr(img2)
    h1, w1 = i1.shape[:2]
    h2, w2 = i2.shape[:2]
    h = max(h1, h2)
    gap = cfg.tile_gap

    canvas = np.full((h, w1 + gap + w2, 3), cfg.tile_bg, dtype=np.uint8)
    canvas[:h1, :w1] = i1
    canvas[:h2, w1 + gap:w1 + gap + w2] = i2

    for m in list(matches)[:max_matches]:
        pt1 = (int(m.pt_src[0]),       int(m.pt_src[1]))
        pt2 = (int(m.pt_dst[0]) + w1 + gap, int(m.pt_dst[1]))
        # Цвет зависит от confidence: зелёный → высокая, красный → низкая
        r = int((1.0 - m.confidence) * 255)
        g = int(m.confidence * 255)
        color = (0, g, r)
        cv2.line(canvas, pt1, pt2, color, cfg.line_thickness)
        cv2.circle(canvas, pt1, 2, color, -1)
        cv2.circle(canvas, pt2, 2, color, -1)

    return canvas


# ─── Рисование контура ────────────────────────────────────────────────────────

def draw_contour(img:          np.ndarray,
                  contour:      np.ndarray,
                  cfg:          VisConfig          = _DEFAULT_CFG,
                  closed:       bool               = True,
                  fill_alpha:   float              = 0.0) -> np.ndarray:
    """
    Накладывает контур на изображение.

    Args:
        img:        Входное изображение.
        contour:    (N, 2) или (N, 1, 2) массив точек.
        cfg:        Параметры.
        closed:     Замкнуть контур.
        fill_alpha: Если > 0 — заполняет контур полупрозрачно.

    Returns:
        Аннотированное изображение.
    """
    out = _ensure_bgr(img)
    pts = np.array(contour, dtype=np.int32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)

    if fill_alpha > 0.0:
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], cfg.contour_color)
        cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

    cv2.polylines(out, [pts], closed, cfg.contour_color,
                   cfg.line_thickness, cv2.LINE_AA)
    return out


# ─── Схема размещения фрагментов ─────────────────────────────────────────────

def draw_assembly_layout(boxes:      Sequence,
                          canvas_wh:  Tuple[int, int] = (500, 400),
                          cfg:        VisConfig        = _DEFAULT_CFG) -> np.ndarray:
    """
    Рисует схему размещения фрагментов на белом холсте.

    Каждый FragmentBox отображается как цветной прямоугольник с подписью fid.
    Цвет циклически меняется по 8 заранее заданным оттенкам.

    Args:
        boxes:     Список FragmentBox.
        canvas_wh: (width, height) холста.
        cfg:       Параметры.

    Returns:
        BGR изображение.
    """
    palette = [
        (220,  80,  80), ( 80, 200,  80), ( 80,  80, 220),
        (220, 200,  80), (200,  80, 220), ( 80, 220, 200),
        (180, 100,  40), ( 40, 100, 180),
    ]
    w, h   = canvas_wh
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, fb in enumerate(boxes):
        color = palette[i % len(palette)]
        x1, y1 = int(fb.x),  int(fb.y)
        x2, y2 = int(fb.x2), int(fb.y2)

        # Отсекаем за пределами холста
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)

        if x2c > x1c and y2c > y1c:
            roi = canvas[y1c:y2c, x1c:x2c]
            overlay = roi.copy()
            overlay[:] = color
            cv2.addWeighted(overlay, 0.3, roi, 0.7, 0, roi)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, cfg.line_thickness + 1)
        _put_label(canvas, f"fid={fb.fid}", max(x1, 0) + 2,
                    max(y1, 0) + 14, cfg)

    return canvas


# ─── Аннотация угла наклона ───────────────────────────────────────────────────

def draw_skew_angle(img:       np.ndarray,
                     angle_deg: float,
                     cfg:       VisConfig = _DEFAULT_CFG) -> np.ndarray:
    """
    Рисует линию под углом angle_deg и подпись значения угла.

    Args:
        img:       Входное изображение.
        angle_deg: Угол наклона в градусах.
        cfg:       Параметры.

    Returns:
        Аннотированное изображение.
    """
    out = _ensure_bgr(img)
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    length = min(w, h) // 3

    rad = math.radians(angle_deg)
    x1  = int(cx - length * math.cos(rad))
    y1  = int(cy - length * math.sin(rad))
    x2  = int(cx + length * math.cos(rad))
    y2  = int(cy + length * math.sin(rad))

    cv2.line(out, (x1, y1), (x2, y2), (0, 0, 200), cfg.line_thickness + 1,
              cv2.LINE_AA)
    _put_label(out, f"skew={angle_deg:.1f}°", 5, 20, cfg)
    return out


# ─── Полоска confidence ───────────────────────────────────────────────────────

def draw_confidence_bar(img:        np.ndarray,
                         confidence: float,
                         grade:      str = "",
                         bar_height: int = 20,
                         cfg:        VisConfig = _DEFAULT_CFG) -> np.ndarray:
    """
    Добавляет горизонтальную цветную полоску confidence внизу изображения.

    Цвет: зелёный (1.0) → жёлтый (0.5) → красный (0.0).

    Args:
        img:        Входное изображение.
        confidence: ∈ [0, 1].
        grade:      Буква оценки (опционально).
        bar_height: Высота полоски в пикселях.
        cfg:        Параметры.

    Returns:
        Изображение с дополнительной полоской снизу.
    """
    out  = _ensure_bgr(img)
    h, w = out.shape[:2]
    bar  = np.full((bar_height, w, 3), cfg.bg_color, dtype=np.uint8)

    fill_w = max(0, min(w, int(w * float(np.clip(confidence, 0, 1)))))
    # Интерполяция цвета: красный→жёлтый→зелёный (clamped to valid uint8 range)
    c = float(np.clip(confidence, 0.0, 1.0))
    r = int(255 * (1.0 - c))
    g = int(255 * c)
    bar[:, :fill_w] = (0, g, r)

    label = f"conf={confidence:.2f}"
    if grade:
        label += f" [{grade}]"
    _put_label(bar, label, 4, bar_height - 4, cfg)

    return np.vstack([out, bar])


# ─── Мозаика изображений ─────────────────────────────────────────────────────

def tile_images(images:     Sequence[np.ndarray],
                 n_cols:     int         = 3,
                 labels:     Optional[Sequence[str]] = None,
                 cfg:        VisConfig   = _DEFAULT_CFG) -> np.ndarray:
    """
    Собирает мозаику из нескольких изображений.

    Все изображения масштабируются до одного размера (размер первого).
    Grayscale конвертируются в BGR.

    Args:
        images: Список изображений.
        n_cols: Число столбцов.
        labels: Подписи под каждым изображением (опционально).
        cfg:    Параметры.

    Returns:
        Одно BGR изображение — мозаика.
    """
    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 200

    bgr_images = [_ensure_bgr(img) for img in images]
    th, tw     = bgr_images[0].shape[:2]
    gap        = cfg.tile_gap

    n_imgs = len(bgr_images)
    n_rows = math.ceil(n_imgs / n_cols)

    total_w = n_cols * tw + (n_cols - 1) * gap
    total_h = n_rows * th + (n_rows - 1) * gap
    canvas  = np.full((total_h, total_w, 3), cfg.tile_bg, dtype=np.uint8)

    for idx, img in enumerate(bgr_images):
        row = idx // n_cols
        col = idx % n_cols
        x   = col * (tw + gap)
        y   = row * (th + gap)

        # Масштабируем до (tw, th)
        resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        canvas[y:y + th, x:x + tw] = resized

        if labels is not None and idx < len(labels):
            _put_label(canvas, labels[idx], x + 2, y + 14, cfg)

    return canvas
