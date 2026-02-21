"""
Проверка согласованности расположения фрагментов документа.

Анализирует совместимость смежных фрагментов по характеристикам текста:
межстрочному интервалу, высоте символов, направлению текста,
выравниванию полей.

Классы:
    ConsistencyType      — типы нарушений согласованности
    ConsistencyViolation — одно обнаруженное нарушение
    ConsistencyResult    — итоговый отчёт проверки

Функции:
    estimate_line_spacing   — оценка межстрочного интервала изображения
    estimate_char_height    — оценка средней высоты символов (строк текста)
    estimate_text_angle     — оценка угла текстового блока через Hough
    check_line_spacing      — проверка равномерности межстрочного интервала
    check_char_height       — проверка однородности высоты символов
    check_text_angle        — проверка ориентации текста
    check_margin_alignment  — проверка выравнивания полей по горизонтали
    check_consistency       — полная проверка одного набора фрагментов
    batch_check_consistency — проверка нескольких наборов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ─── ConsistencyType ──────────────────────────────────────────────────────────

class ConsistencyType(str, Enum):
    """Тип нарушения согласованности."""
    LINE_SPACING    = "line_spacing"    # Разные межстрочные интервалы
    CHAR_HEIGHT     = "char_height"     # Разная высота символов
    TEXT_ANGLE      = "text_angle"      # Несогласованные углы текста
    MARGIN_ALIGN    = "margin_align"    # Несогласованное выравнивание полей
    INSUFFICIENT    = "insufficient"    # Недостаточно данных для проверки


# ─── ConsistencyViolation ─────────────────────────────────────────────────────

@dataclass
class ConsistencyViolation:
    """
    Одно обнаруженное нарушение согласованности.

    Attributes:
        type:         Тип нарушения.
        severity:     Тяжесть [0, 1]. 1 = максимальная.
        fragment_ids: ID фрагментов, участвующих в нарушении.
        description:  Человекочитаемое описание.
        values:       Измеренные значения {fid: value}.
    """
    type:         ConsistencyType
    severity:     float
    fragment_ids: List[int] = field(default_factory=list)
    description:  str       = ""
    values:       Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ConsistencyViolation(type={self.type.value!r}, "
                f"severity={self.severity:.3f}, "
                f"frags={self.fragment_ids})")


# ─── ConsistencyResult ────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    """
    Итоговый отчёт проверки согласованности.

    Attributes:
        violations:    Список нарушений.
        score:         Оценка согласованности ∈ [0, 1].
                       1 = полная согласованность (нет нарушений).
        n_checked:     Число проверенных пар фрагментов.
        method_scores: Оценки по отдельным методам.
    """
    violations:    List[ConsistencyViolation]
    score:         float
    n_checked:     int
    method_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def max_severity(self) -> float:
        if not self.violations:
            return 0.0
        return max(v.severity for v in self.violations)

    def __repr__(self) -> str:
        return (f"ConsistencyResult(score={self.score:.3f}, "
                f"violations={self.n_violations}, "
                f"max_severity={self.max_severity:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """Горизонтальная проекция бинарного изображения (число белых пикселей)."""
    return binary.astype(np.uint8).sum(axis=1).astype(np.float32)


def _peaks_from_projection(proj: np.ndarray,
                             min_val: float = 0.0) -> List[int]:
    """Находит индексы пиков (строк с максимальным числом белых пикселей)."""
    peaks = []
    n = len(proj)
    for i in range(1, n - 1):
        if proj[i] > min_val and proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1]:
            if not peaks or i - peaks[-1] > 1:
                peaks.append(i)
    return peaks


# ─── estimate_line_spacing ────────────────────────────────────────────────────

def estimate_line_spacing(img: np.ndarray,
                            min_line_height: int = 5) -> float:
    """
    Оценивает средний межстрочный интервал через горизонтальную проекцию.

    Args:
        img:             BGR или grayscale изображение.
        min_line_height: Минимальная высота строки (пикселей).

    Returns:
        Средний межстрочный интервал (пикселей). 0 если строки не найдены.
    """
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = _horizontal_projection(binary)

    # Минимальное значение = 5% от максимума
    threshold = float(proj.max()) * 0.05
    text_rows = [i for i, v in enumerate(proj) if v > threshold]

    if len(text_rows) < 2:
        return 0.0

    # Разбиваем на группы строк (разрыв > min_line_height)
    groups: List[List[int]] = [[text_rows[0]]]
    for r in text_rows[1:]:
        if r - groups[-1][-1] <= min_line_height:
            groups[-1].append(r)
        else:
            groups.append([r])

    if len(groups) < 2:
        return 0.0

    # Центры групп
    centers = [float(np.mean(g)) for g in groups]
    intervals = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    return float(np.mean(intervals)) if intervals else 0.0


# ─── estimate_char_height ─────────────────────────────────────────────────────

def estimate_char_height(img: np.ndarray,
                           min_height: int = 4,
                           max_height: int = 200) -> float:
    """
    Оценивает среднюю высоту символов через анализ связных компонент.

    Args:
        img:        BGR или grayscale изображение.
        min_height: Минимальная высота компоненты (символа).
        max_height: Максимальная высота компоненты.

    Returns:
        Средняя высота символов (пикселей). 0 если нет подходящих компонент.
    """
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    heights = []
    for i in range(1, n_labels):   # 0 = фон
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Фильтр: реалистичный символ
        if min_height <= h <= max_height and area >= 4:
            # Отсеиваем линии (слишком широкие относительно высоты)
            if w <= h * 8:
                heights.append(h)

    if not heights:
        return 0.0
    # Медиана устойчивее к выбросам
    return float(np.median(heights))


# ─── estimate_text_angle ──────────────────────────────────────────────────────

def estimate_text_angle(img:       np.ndarray,
                         threshold: int   = 80,
                         min_len:   int   = 30) -> float:
    """
    Оценивает угол текстового блока через линии Хаффа.

    Args:
        img:       BGR или grayscale изображение.
        threshold: Порог накопителя Хаффа.
        min_len:   Минимальная длина линии.

    Returns:
        Медианный угол (°, относительно горизонтали). 0 если нет линий.
    """
    gray  = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=5,
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Нормируем к [-45, 45]
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90
            angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


# ─── check_line_spacing ───────────────────────────────────────────────────────

def check_line_spacing(fragment_ids: List[int],
                        images:       List[np.ndarray],
                        tol_ratio:    float = 0.25) -> List[ConsistencyViolation]:
    """
    Проверяет равномерность межстрочного интервала между фрагментами.

    Args:
        fragment_ids: ID фрагментов.
        images:       Соответствующие изображения.
        tol_ratio:    Допустимое относительное отклонение интервала.

    Returns:
        Список нарушений (пустой если всё в норме).
    """
    spacings: Dict[int, float] = {}
    for fid, img in zip(fragment_ids, images):
        sp = estimate_line_spacing(img)
        if sp > 0:
            spacings[fid] = sp

    if len(spacings) < 2:
        return []

    values = list(spacings.values())
    mean_sp = float(np.mean(values))
    if mean_sp <= 0:
        return []

    violations = []
    outliers = [fid for fid, sp in spacings.items()
                if abs(sp - mean_sp) / mean_sp > tol_ratio]

    if outliers:
        std_sp   = float(np.std(values))
        severity = float(np.clip(std_sp / max(1.0, mean_sp), 0.0, 1.0))
        violations.append(ConsistencyViolation(
            type=ConsistencyType.LINE_SPACING,
            severity=severity,
            fragment_ids=outliers,
            description=(f"Межстрочный интервал: μ={mean_sp:.1f}px, "
                         f"σ={std_sp:.1f}px, отклонение > {tol_ratio*100:.0f}%"),
            values=spacings,
        ))
    return violations


# ─── check_char_height ────────────────────────────────────────────────────────

def check_char_height(fragment_ids: List[int],
                       images:       List[np.ndarray],
                       tol_ratio:    float = 0.30) -> List[ConsistencyViolation]:
    """
    Проверяет однородность высоты символов между фрагментами.

    Args:
        fragment_ids: ID фрагментов.
        images:       Соответствующие изображения.
        tol_ratio:    Допустимое относительное отклонение.

    Returns:
        Список нарушений.
    """
    heights: Dict[int, float] = {}
    for fid, img in zip(fragment_ids, images):
        ch = estimate_char_height(img)
        if ch > 0:
            heights[fid] = ch

    if len(heights) < 2:
        return []

    values   = list(heights.values())
    mean_h   = float(np.mean(values))
    if mean_h <= 0:
        return []

    outliers = [fid for fid, h in heights.items()
                if abs(h - mean_h) / mean_h > tol_ratio]

    if not outliers:
        return []

    std_h    = float(np.std(values))
    severity = float(np.clip(std_h / max(1.0, mean_h), 0.0, 1.0))
    return [ConsistencyViolation(
        type=ConsistencyType.CHAR_HEIGHT,
        severity=severity,
        fragment_ids=outliers,
        description=(f"Высота символов: μ={mean_h:.1f}px, "
                     f"σ={std_h:.1f}px, отклонение > {tol_ratio*100:.0f}%"),
        values=heights,
    )]


# ─── check_text_angle ─────────────────────────────────────────────────────────

def check_text_angle(fragment_ids: List[int],
                      images:       List[np.ndarray],
                      max_angle:    float = 3.0) -> List[ConsistencyViolation]:
    """
    Проверяет, что углы текста согласованы между фрагментами.

    Args:
        fragment_ids: ID фрагментов.
        images:       Соответствующие изображения.
        max_angle:    Максимально допустимое угловое отклонение (°).

    Returns:
        Список нарушений.
    """
    angles: Dict[int, float] = {}
    for fid, img in zip(fragment_ids, images):
        ang = estimate_text_angle(img)
        angles[fid] = ang

    if len(angles) < 2:
        return []

    values    = list(angles.values())
    mean_ang  = float(np.mean(values))
    outliers  = [fid for fid, ang in angles.items()
                 if abs(ang - mean_ang) > max_angle]

    if not outliers:
        return []

    max_dev   = max(abs(ang - mean_ang) for ang in values)
    severity  = float(np.clip(max_dev / max(1.0, 45.0), 0.0, 1.0))
    return [ConsistencyViolation(
        type=ConsistencyType.TEXT_ANGLE,
        severity=severity,
        fragment_ids=outliers,
        description=(f"Угол текста: μ={mean_ang:.1f}°, "
                     f"допуск={max_angle}°, макс. откл.={max_dev:.1f}°"),
        values=angles,
    )]


# ─── check_margin_alignment ───────────────────────────────────────────────────

def check_margin_alignment(fragment_ids: List[int],
                             images:       List[np.ndarray],
                             tol_px:       float = 10.0) -> List[ConsistencyViolation]:
    """
    Проверяет горизонтальное выравнивание левых полей текста.

    Оценивает левое поле каждого фрагмента через горизонтальную проекцию
    и проверяет, что они согласованы.

    Args:
        fragment_ids: ID фрагментов.
        images:       Соответствующие изображения.
        tol_px:       Допустимое отклонение в пикселях.

    Returns:
        Список нарушений.
    """
    margins: Dict[int, float] = {}
    for fid, img in zip(fragment_ids, images):
        gray = _to_gray(img)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Вертикальная проекция → ищем первый ненулевой столбец
        v_proj = binary.sum(axis=0)
        nz = np.where(v_proj > 0)[0]
        if len(nz) > 0:
            margins[fid] = float(nz[0])

    if len(margins) < 2:
        return []

    values    = list(margins.values())
    mean_m    = float(np.mean(values))
    outliers  = [fid for fid, m in margins.items()
                 if abs(m - mean_m) > tol_px]

    if not outliers:
        return []

    std_m    = float(np.std(values))
    severity = float(np.clip(std_m / max(1.0, img.shape[1] * 0.1), 0.0, 1.0))
    return [ConsistencyViolation(
        type=ConsistencyType.MARGIN_ALIGN,
        severity=severity,
        fragment_ids=outliers,
        description=(f"Поле: μ={mean_m:.1f}px, σ={std_m:.1f}px, "
                     f"допуск={tol_px:.0f}px"),
        values=margins,
    )]


# ─── check_consistency ────────────────────────────────────────────────────────

def check_consistency(fragment_ids:   List[int],
                       images:         List[np.ndarray],
                       spacing_tol:    float = 0.25,
                       height_tol:     float = 0.30,
                       angle_max:      float = 3.0,
                       margin_tol:     float = 15.0) -> ConsistencyResult:
    """
    Полная проверка согласованности набора фрагментов.

    Args:
        fragment_ids: ID фрагментов.
        images:       Соответствующие изображения.
        spacing_tol:  Допуск межстрочного интервала (доля).
        height_tol:   Допуск высоты символов (доля).
        angle_max:    Максимальный угол несогласованности (°).
        margin_tol:   Допуск выравнивания полей (пикселей).

    Returns:
        ConsistencyResult.
    """
    if len(fragment_ids) != len(images):
        raise ValueError(
            f"fragment_ids ({len(fragment_ids)}) и images ({len(images)}) "
            f"должны быть одной длины."
        )

    all_violations: List[ConsistencyViolation] = []
    method_scores: Dict[str, float] = {}

    # Межстрочный интервал
    sp_viol = check_line_spacing(fragment_ids, images, tol_ratio=spacing_tol)
    all_violations.extend(sp_viol)
    method_scores["line_spacing"] = 0.0 if sp_viol else 1.0

    # Высота символов
    ch_viol = check_char_height(fragment_ids, images, tol_ratio=height_tol)
    all_violations.extend(ch_viol)
    method_scores["char_height"] = 0.0 if ch_viol else 1.0

    # Угол текста
    ang_viol = check_text_angle(fragment_ids, images, max_angle=angle_max)
    all_violations.extend(ang_viol)
    method_scores["text_angle"] = 0.0 if ang_viol else 1.0

    # Поля
    mar_viol = check_margin_alignment(fragment_ids, images, tol_px=margin_tol)
    all_violations.extend(mar_viol)
    method_scores["margin_align"] = 0.0 if mar_viol else 1.0

    # Итоговая оценка
    if all_violations:
        severities = [v.severity for v in all_violations]
        violation_score = 0.6 * max(severities) + 0.4 * float(np.mean(severities))
        score = float(np.clip(1.0 - violation_score, 0.0, 1.0))
    else:
        score = 1.0

    n_checked = max(0, len(fragment_ids) * (len(fragment_ids) - 1) // 2)

    return ConsistencyResult(
        violations=all_violations,
        score=score,
        n_checked=n_checked,
        method_scores=method_scores,
    )


# ─── batch_check_consistency ──────────────────────────────────────────────────

def batch_check_consistency(
        fragment_id_groups: List[List[int]],
        image_groups:       List[List[np.ndarray]],
        **kwargs,
) -> List[ConsistencyResult]:
    """
    Проверяет согласованность нескольких наборов фрагментов.

    Args:
        fragment_id_groups: Списки ID фрагментов (по одному на группу).
        image_groups:       Соответствующие группы изображений.
        **kwargs:           Параметры для check_consistency.

    Returns:
        Список ConsistencyResult.

    Raises:
        ValueError: Если число групп не совпадает.
    """
    if len(fragment_id_groups) != len(image_groups):
        raise ValueError(
            f"Число групп ID ({len(fragment_id_groups)}) != "
            f"число групп изображений ({len(image_groups)})."
        )
    return [
        check_consistency(fids, imgs, **kwargs)
        for fids, imgs in zip(fragment_id_groups, image_groups)
    ]
