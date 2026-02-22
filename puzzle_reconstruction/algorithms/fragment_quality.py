"""Оценка качества отдельных фрагментов документа.

Модуль вычисляет метрики качества изображения фрагмента:
размытость, контрастность, покрытие маской и резкость краёв.
Используется для ранжирования и фильтрации фрагментов перед сборкой.

Классы:
    QualityConfig  — параметры оценки качества
    QualityReport  — результат оценки одного фрагмента

Функции:
    measure_blur          — оценка размытости (выше = чётче)
    measure_contrast      — RMS-контрастность в [0, 1]
    measure_mask_coverage — доля ненулевых пикселей маски
    measure_edge_sharpness — средняя величина градиента на маске
    assess_fragment       — полная оценка фрагмента
    rank_fragments        — ранжирование по агрегированному score
    batch_assess          — пакетная оценка списка фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── QualityConfig ────────────────────────────────────────────────────────────

@dataclass
class QualityConfig:
    """Параметры оценки качества фрагмента.

    Атрибуты:
        w_blur:     Вес метрики чёткости (>= 0).
        w_contrast: Вес контрастности (>= 0).
        w_coverage: Вес покрытия маской (>= 0).
        w_sharpness: Вес резкости краёв (>= 0).
        blur_ref:   Эталонное значение Лапласиана для нормировки (> 0).
        grad_ref:   Эталонное значение градиента для нормировки (> 0).
    """
    w_blur:     float = 1.0
    w_contrast: float = 1.0
    w_coverage: float = 1.0
    w_sharpness: float = 1.0
    blur_ref:   float = 500.0
    grad_ref:   float = 50.0

    def __post_init__(self) -> None:
        for name in ("w_blur", "w_contrast", "w_coverage", "w_sharpness"):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(
                    f"Вес '{name}' должен быть >= 0, получено {val}"
                )
        total = self.w_blur + self.w_contrast + self.w_coverage + self.w_sharpness
        if total == 0.0:
            raise ValueError("Сумма весов должна быть > 0")
        if self.blur_ref <= 0.0:
            raise ValueError(
                f"blur_ref должен быть > 0, получено {self.blur_ref}"
            )
        if self.grad_ref <= 0.0:
            raise ValueError(
                f"grad_ref должен быть > 0, получено {self.grad_ref}"
            )

    @property
    def total_weight(self) -> float:
        """Суммарный вес."""
        return self.w_blur + self.w_contrast + self.w_coverage + self.w_sharpness


# ─── QualityReport ────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Результат оценки качества фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        score:       Агрегированная оценка ∈ [0, 1].
        blur:        Оценка чёткости ∈ [0, 1].
        contrast:    Оценка контрастности ∈ [0, 1].
        coverage:    Покрытие маской ∈ [0, 1].
        sharpness:   Оценка резкости краёв ∈ [0, 1].
        params:      Дополнительные параметры.
    """
    fragment_id: int
    score:       float
    blur:        float
    contrast:    float
    coverage:    float
    sharpness:   float
    params:      Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        for name in ("score", "blur", "contrast", "coverage", "sharpness"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"'{name}' должен быть в [0, 1], получено {val}"
                )

    @property
    def is_usable(self) -> bool:
        """True если score >= 0.3 (минимально приемлемое качество)."""
        return self.score >= 0.3

    def summary(self) -> str:
        """Краткое текстовое резюме."""
        return (
            f"QualityReport(id={self.fragment_id}, score={self.score:.3f}, "
            f"blur={self.blur:.3f}, contrast={self.contrast:.3f}, "
            f"coverage={self.coverage:.3f}, sharpness={self.sharpness:.3f})"
        )


# ─── measure_blur ─────────────────────────────────────────────────────────────

def measure_blur(img: np.ndarray, ref: float = 500.0) -> float:
    """Оценить чёткость изображения через дисперсию Лапласиана.

    Аргументы:
        img: BGR или grayscale изображение (uint8).
        ref: Эталонное значение для нормировки (> 0).

    Возвращает:
        Оценка ∈ [0, 1]; 1 = максимально чёткое.

    Исключения:
        ValueError: Если ref <= 0.
    """
    if ref <= 0.0:
        raise ValueError(f"ref должен быть > 0, получено {ref}")

    img = np.asarray(img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    lap_var = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())
    return float(np.clip(lap_var / ref, 0.0, 1.0))


# ─── measure_contrast ────────────────────────────────────────────────────────

def measure_contrast(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Вычислить RMS-контрастность изображения ∈ [0, 1].

    Аргументы:
        img:  BGR или grayscale изображение (uint8).
        mask: Бинарная маска (uint8, 255=активная область). None → вся область.

    Возвращает:
        RMS-контрастность / 128 ∈ [0, 1].
    """
    img = np.asarray(img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    if mask is not None:
        m = np.asarray(mask) > 0
        pixels = gray[m]
    else:
        pixels = gray.ravel()

    if len(pixels) == 0:
        return 0.0

    rms = float(np.sqrt(((pixels - pixels.mean()) ** 2).mean()))
    return float(np.clip(rms / 128.0, 0.0, 1.0))


# ─── measure_mask_coverage ───────────────────────────────────────────────────

def measure_mask_coverage(mask: np.ndarray) -> float:
    """Вычислить долю ненулевых пикселей маски ∈ [0, 1].

    Аргументы:
        mask: Бинарная маска (uint8).

    Возвращает:
        Доля ∈ [0, 1]. Пустая маска → 0.

    Исключения:
        ValueError: Если mask не 2-D.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"mask должна быть 2-D, получено ndim={mask.ndim}"
        )
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum()) / float(total)


# ─── measure_edge_sharpness ──────────────────────────────────────────────────

def measure_edge_sharpness(
    img:     np.ndarray,
    mask:    Optional[np.ndarray] = None,
    ref:     float = 50.0,
) -> float:
    """Вычислить среднюю величину градиента на активной области.

    Аргументы:
        img:  BGR или grayscale изображение (uint8).
        mask: Маска активной области. None → вся область.
        ref:  Эталонное значение для нормировки (> 0).

    Возвращает:
        Оценка ∈ [0, 1].

    Исключения:
        ValueError: Если ref <= 0.
    """
    if ref <= 0.0:
        raise ValueError(f"ref должен быть > 0, получено {ref}")

    img = np.asarray(img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    if mask is not None:
        m = np.asarray(mask) > 0
        if not m.any():
            return 0.0
        mean_grad = float(mag[m].mean())
    else:
        mean_grad = float(mag.mean())

    return float(np.clip(mean_grad / ref, 0.0, 1.0))


# ─── assess_fragment ─────────────────────────────────────────────────────────

def assess_fragment(
    img:         np.ndarray,
    mask:        Optional[np.ndarray] = None,
    cfg:         Optional[QualityConfig] = None,
    fragment_id: int = 0,
) -> QualityReport:
    """Полная оценка качества фрагмента.

    Аргументы:
        img:         Изображение фрагмента (uint8).
        mask:        Маска фрагмента. None → вся область.
        cfg:         Конфигурация (None → QualityConfig()).
        fragment_id: Идентификатор фрагмента (>= 0).

    Возвращает:
        QualityReport.
    """
    if cfg is None:
        cfg = QualityConfig()

    blur      = measure_blur(img, ref=cfg.blur_ref)
    contrast  = measure_contrast(img, mask=mask)
    coverage  = measure_mask_coverage(mask) if mask is not None else 1.0
    sharpness = measure_edge_sharpness(img, mask=mask, ref=cfg.grad_ref)

    t = cfg.total_weight
    score = float(np.clip(
        (cfg.w_blur * blur
         + cfg.w_contrast * contrast
         + cfg.w_coverage * coverage
         + cfg.w_sharpness * sharpness) / t,
        0.0, 1.0,
    ))

    return QualityReport(
        fragment_id=fragment_id,
        score=score,
        blur=blur,
        contrast=contrast,
        coverage=coverage,
        sharpness=sharpness,
        params={
            "w_blur": cfg.w_blur,
            "w_contrast": cfg.w_contrast,
            "w_coverage": cfg.w_coverage,
            "w_sharpness": cfg.w_sharpness,
        },
    )


# ─── rank_fragments ──────────────────────────────────────────────────────────

def rank_fragments(
    reports:  List[QualityReport],
    indices:  Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Ранжировать фрагменты по убыванию агрегированного score.

    Аргументы:
        reports: Список QualityReport.
        indices: Подмножество индексов для ранжирования
                 (None → все reports по порядку).

    Возвращает:
        [(fragment_id, score), ...] отсортированный по убыванию score.

    Исключения:
        ValueError: Если indices не пустой и его длина != len(reports).
    """
    if indices is not None and len(indices) != len(reports):
        raise ValueError(
            f"Длина indices ({len(indices)}) != len(reports) ({len(reports)})"
        )

    if indices is None:
        items = [(r.fragment_id, r.score) for r in reports]
    else:
        items = [(indices[i], reports[i].score) for i in range(len(reports))]

    items.sort(key=lambda x: x[1], reverse=True)
    return items


# ─── batch_assess ────────────────────────────────────────────────────────────

def batch_assess(
    images: List[np.ndarray],
    masks:  Optional[List[Optional[np.ndarray]]] = None,
    cfg:    Optional[QualityConfig] = None,
) -> List[QualityReport]:
    """Пакетная оценка качества списка фрагментов.

    Аргументы:
        images: Список изображений (uint8).
        masks:  Список масок (None → без масок для всех).
        cfg:    Конфигурация (None → QualityConfig()).

    Возвращает:
        Список QualityReport; fragment_id = индекс в списке.

    Исключения:
        ValueError: Если masks не None и len(masks) != len(images).
    """
    if cfg is None:
        cfg = QualityConfig()

    if masks is not None and len(masks) != len(images):
        raise ValueError(
            f"Длина masks ({len(masks)}) != len(images) ({len(images)})"
        )

    results = []
    for i, img in enumerate(images):
        mask = masks[i] if masks is not None else None
        results.append(assess_fragment(img, mask=mask, cfg=cfg, fragment_id=i))
    return results
