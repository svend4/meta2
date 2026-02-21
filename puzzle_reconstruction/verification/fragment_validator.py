"""
Валидация отдельных фрагментов документа.

Проверяет размеры, соотношение сторон, покрытие содержимым, качество
контура и другие базовые свойства перед передачей фрагментов в пайплайн.

Экспортирует:
    ValidationIssue     — один выявленный дефект
    ValidationResult    — итоговый отчёт по фрагменту
    FragmentValidatorParams — параметры валидатора
    validate_dimensions    — проверка размеров изображения
    validate_aspect_ratio  — проверка соотношения сторон
    validate_content_coverage — оценка покрытия содержимым (маска)
    validate_contour       — проверка контура фрагмента
    validate_fragment      — полная валидация по набору правил
    batch_validate         — пакетная валидация
    filter_valid           — фильтрация: только прошедшие валидацию
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """Один выявленный дефект при валидации.

    Attributes:
        code:    Краткий машиночитаемый код проблемы (например, ``'too_small'``).
        message: Подробное описание.
        severity: ``'error'`` (валидация не пройдена) или ``'warning'``.
    """
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def __repr__(self) -> str:  # pragma: no cover
        return f"ValidationIssue({self.severity}: {self.code})"


@dataclass
class ValidationResult:
    """Итоговый отчёт валидации одного фрагмента.

    Attributes:
        fragment_idx: Индекс фрагмента (если задан).
        passed:       ``True``, если нет ошибок (предупреждения допускаются).
        issues:       Список выявленных проблем.
        metrics:      Измеренные метрики (например, площадь, AR и т.д.).
    """
    fragment_idx: int = -1
    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    def __repr__(self) -> str:  # pragma: no cover
        status = "PASS" if self.passed else "FAIL"
        return f"ValidationResult(idx={self.fragment_idx}, {status}, issues={len(self.issues)})"


@dataclass
class FragmentValidatorParams:
    """Параметры валидатора фрагментов.

    Attributes:
        min_width:          Минимальная допустимая ширина (пикс.).
        min_height:         Минимальная допустимая высота (пикс.).
        max_width:          Максимальная допустимая ширина (0 = без ограничений).
        max_height:         Максимальная допустимая высота (0 = без ограничений).
        min_aspect_ratio:   Минимальное соотношение сторон (min(w,h)/max(w,h)).
        min_coverage:       Минимальная доля ненулевых пикселей (0–1).
        min_contour_points: Минимальное число точек контура.
        max_contour_points: Максимальное число точек контура (0 = без лимита).
        min_contour_area:   Минимальная площадь контура (пикс²).
    """
    min_width: int = 16
    min_height: int = 16
    max_width: int = 0
    max_height: int = 0
    min_aspect_ratio: float = 0.05
    min_coverage: float = 0.05
    min_contour_points: int = 3
    max_contour_points: int = 0
    min_contour_area: float = 10.0

    def __post_init__(self) -> None:
        if self.min_width < 1:
            raise ValueError(f"min_width must be >= 1, got {self.min_width}")
        if self.min_height < 1:
            raise ValueError(f"min_height must be >= 1, got {self.min_height}")
        if not (0.0 <= self.min_aspect_ratio <= 1.0):
            raise ValueError(
                f"min_aspect_ratio must be in [0, 1], got {self.min_aspect_ratio}"
            )
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError(
                f"min_coverage must be in [0, 1], got {self.min_coverage}"
            )
        if self.min_contour_points < 3:
            raise ValueError(
                f"min_contour_points must be >= 3, got {self.min_contour_points}"
            )


# ─── Публичные функции ────────────────────────────────────────────────────────

def validate_dimensions(
    img: np.ndarray,
    params: Optional[FragmentValidatorParams] = None,
) -> ValidationResult:
    """Проверить размеры изображения.

    Args:
        img:    Изображение uint8 (2D или 3D).
        params: Параметры валидатора.

    Returns:
        :class:`ValidationResult` с полями ``width``, ``height``, ``area``.
    """
    if params is None:
        params = FragmentValidatorParams()
    result = ValidationResult()
    h, w = img.shape[:2]
    result.metrics["width"] = float(w)
    result.metrics["height"] = float(h)
    result.metrics["area"] = float(w * h)

    if w < params.min_width:
        result.add_issue(ValidationIssue(
            code="too_narrow",
            message=f"Width {w} < min_width {params.min_width}",
        ))
    if h < params.min_height:
        result.add_issue(ValidationIssue(
            code="too_short",
            message=f"Height {h} < min_height {params.min_height}",
        ))
    if params.max_width > 0 and w > params.max_width:
        result.add_issue(ValidationIssue(
            code="too_wide",
            message=f"Width {w} > max_width {params.max_width}",
        ))
    if params.max_height > 0 and h > params.max_height:
        result.add_issue(ValidationIssue(
            code="too_tall",
            message=f"Height {h} > max_height {params.max_height}",
        ))
    return result


def validate_aspect_ratio(
    img: np.ndarray,
    params: Optional[FragmentValidatorParams] = None,
) -> ValidationResult:
    """Проверить соотношение сторон.

    Соотношение вычисляется как min(w,h)/max(w,h) ∈ (0, 1].

    Args:
        img:    Изображение uint8.
        params: Параметры валидатора.

    Returns:
        :class:`ValidationResult` с полем ``aspect_ratio``.
    """
    if params is None:
        params = FragmentValidatorParams()
    result = ValidationResult()
    h, w = img.shape[:2]
    ar = float(min(w, h)) / float(max(w, h)) if max(w, h) > 0 else 0.0
    result.metrics["aspect_ratio"] = ar

    if ar < params.min_aspect_ratio:
        result.add_issue(ValidationIssue(
            code="extreme_aspect_ratio",
            message=f"Aspect ratio {ar:.4f} < min {params.min_aspect_ratio}",
        ))
    return result


def validate_content_coverage(
    img: np.ndarray,
    params: Optional[FragmentValidatorParams] = None,
    threshold: int = 10,
) -> ValidationResult:
    """Проверить долю информативных (ненулевых) пикселей.

    Args:
        img:       Изображение uint8 (2D или 3D).
        params:    Параметры валидатора.
        threshold: Минимальная яркость пикселя для считывания «содержимым».

    Returns:
        :class:`ValidationResult` с полем ``coverage``.
    """
    if params is None:
        params = FragmentValidatorParams()
    result = ValidationResult()

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size
    content_px = int(np.sum(gray > threshold))
    coverage = content_px / total if total > 0 else 0.0
    result.metrics["coverage"] = coverage
    result.metrics["content_pixels"] = float(content_px)

    if coverage < params.min_coverage:
        result.add_issue(ValidationIssue(
            code="insufficient_coverage",
            message=f"Coverage {coverage:.4f} < min {params.min_coverage}",
        ))
    return result


def validate_contour(
    contour: np.ndarray,
    params: Optional[FragmentValidatorParams] = None,
) -> ValidationResult:
    """Проверить контур фрагмента.

    Args:
        contour: Контур формы (N, 2) или (N, 1, 2).
        params:  Параметры валидатора.

    Returns:
        :class:`ValidationResult` с полями ``n_points`` и ``contour_area``.
    """
    if params is None:
        params = FragmentValidatorParams()
    result = ValidationResult()

    pts = contour.reshape(-1, 2)
    n = len(pts)
    result.metrics["n_points"] = float(n)

    if n < params.min_contour_points:
        result.add_issue(ValidationIssue(
            code="too_few_contour_points",
            message=f"Contour has {n} points, min is {params.min_contour_points}",
        ))

    if params.max_contour_points > 0 and n > params.max_contour_points:
        result.add_issue(ValidationIssue(
            code="too_many_contour_points",
            message=f"Contour has {n} points, max is {params.max_contour_points}",
            severity="warning",
        ))

    # Contour area
    if n >= 3:
        area = float(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32)))
    else:
        area = 0.0
    result.metrics["contour_area"] = area

    if area < params.min_contour_area:
        result.add_issue(ValidationIssue(
            code="degenerate_contour",
            message=f"Contour area {area:.2f} < min {params.min_contour_area}",
        ))

    # Check for duplicate / collinear degenerate contour
    if n >= 3:
        unique_pts = len(np.unique(pts, axis=0))
        if unique_pts < 3:
            result.add_issue(ValidationIssue(
                code="degenerate_contour",
                message=f"Contour has only {unique_pts} unique points",
            ))

    return result


def validate_fragment(
    img: np.ndarray,
    contour: Optional[np.ndarray] = None,
    fragment_idx: int = -1,
    params: Optional[FragmentValidatorParams] = None,
) -> ValidationResult:
    """Полная валидация фрагмента.

    Последовательно выполняет: проверку размеров, соотношения сторон,
    покрытия содержимым и (если задан) контура. Объединяет все результаты.

    Args:
        img:          Изображение uint8.
        contour:      Контур фрагмента (необязательно).
        fragment_idx: Индекс фрагмента.
        params:       Параметры валидатора.

    Returns:
        Единый :class:`ValidationResult` со всеми метриками и проблемами.
    """
    if params is None:
        params = FragmentValidatorParams()

    result = ValidationResult(fragment_idx=fragment_idx)

    for sub in (
        validate_dimensions(img, params),
        validate_aspect_ratio(img, params),
        validate_content_coverage(img, params),
    ):
        result.metrics.update(sub.metrics)
        for issue in sub.issues:
            result.add_issue(issue)

    if contour is not None:
        sub = validate_contour(contour, params)
        result.metrics.update(sub.metrics)
        for issue in sub.issues:
            result.add_issue(issue)

    return result


def batch_validate(
    images: List[np.ndarray],
    contours: Optional[List[Optional[np.ndarray]]] = None,
    params: Optional[FragmentValidatorParams] = None,
) -> List[ValidationResult]:
    """Пакетная валидация списка изображений.

    Args:
        images:   Список изображений uint8.
        contours: Опциональный список контуров (``None`` для пропуска).
        params:   Параметры валидатора.

    Returns:
        Список :class:`ValidationResult` той же длины.
    """
    if contours is None:
        contours = [None] * len(images)
    return [
        validate_fragment(img, contour=c, fragment_idx=i, params=params)
        for i, (img, c) in enumerate(zip(images, contours))
    ]


def filter_valid(results: List[ValidationResult]) -> List[int]:
    """Вернуть индексы прошедших валидацию фрагментов.

    Args:
        results: Список :class:`ValidationResult`.

    Returns:
        Список индексов (``fragment_idx``) фрагментов, у которых ``passed=True``.
    """
    return [r.fragment_idx for r in results if r.passed]
