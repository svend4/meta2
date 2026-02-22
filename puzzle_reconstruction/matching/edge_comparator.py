"""Сравнение краёв фрагментов по интенсивности, градиенту и текстуре.

Модуль предоставляет структуры и функции для извлечения одномерных профилей
краёв фрагментов и их парного сравнения с несколькими метриками.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── EdgeCompConfig ───────────────────────────────────────────────────────────

@dataclass
class EdgeCompConfig:
    """Параметры сравнения краёв.

    Атрибуты:
        strip_width:   Ширина полосы вдоль края в пикселях (>= 1).
        n_samples:     Число точек выборки вдоль края (>= 1).
        use_gradient:  Использовать градиент в сравнении.
        use_texture:   Использовать текстуру (ст. отклонение) в сравнении.
        normalize:     Нормализовать профили перед сравнением.
    """

    strip_width: int = 4
    n_samples: int = 32
    use_gradient: bool = True
    use_texture: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.strip_width < 1:
            raise ValueError(
                f"strip_width должен быть >= 1, получено {self.strip_width}"
            )
        if self.n_samples < 1:
            raise ValueError(
                f"n_samples должен быть >= 1, получено {self.n_samples}"
            )


# ─── EdgeSample ───────────────────────────────────────────────────────────────

@dataclass
class EdgeSample:
    """Профили, извлечённые из полосы вдоль края.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        intensity:   Средняя интенсивность вдоль края (shape: n_samples).
        gradient:    Средний градиент (shape: n_samples).
        texture:     Текстурная дисперсия (shape: n_samples).
    """

    fragment_id: int
    intensity: np.ndarray
    gradient: np.ndarray
    texture: np.ndarray

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.intensity.ndim != 1:
            raise ValueError("intensity должен быть одномерным массивом")
        if self.gradient.shape != self.intensity.shape:
            raise ValueError("gradient должен иметь ту же форму, что и intensity")
        if self.texture.shape != self.intensity.shape:
            raise ValueError("texture должен иметь ту же форму, что и intensity")

    @property
    def n_samples(self) -> int:
        """Число точек выборки."""
        return len(self.intensity)

    @property
    def mean_intensity(self) -> float:
        """Средняя интенсивность профиля."""
        return float(np.mean(self.intensity))


# ─── EdgeCompResult ───────────────────────────────────────────────────────────

@dataclass
class EdgeCompResult:
    """Результат сравнения двух краёв.

    Атрибуты:
        pair:              Пара (fragment_id_a, fragment_id_b).
        intensity_score:   Оценка по интенсивности [0, 1].
        gradient_score:    Оценка по градиенту [0, 1].
        texture_score:     Оценка по текстуре [0, 1].
        total_score:       Итоговая взвешенная оценка [0, 1].
        scores:            Словарь всех компонент.
    """

    pair: Tuple[int, int]
    intensity_score: float
    gradient_score: float
    texture_score: float
    total_score: float
    scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in (
            ("intensity_score", self.intensity_score),
            ("gradient_score", self.gradient_score),
            ("texture_score", self.texture_score),
            ("total_score", self.total_score),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )

    @property
    def fragment_a(self) -> int:
        """Первый фрагмент."""
        return self.pair[0]

    @property
    def fragment_b(self) -> int:
        """Второй фрагмент."""
        return self.pair[1]

    @property
    def is_good_match(self) -> bool:
        """True если total_score >= 0.7."""
        return self.total_score >= 0.7


# ─── _ncc ─────────────────────────────────────────────────────────────────────

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная кросс-корреляция двух профилей [0, 1]."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.clip((np.dot(a, b) / denom + 1.0) / 2.0, 0.0, 1.0))


def _l1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированное L1-сходство [0, 1]."""
    diff = float(np.mean(np.abs(a - b)))
    rng = max(float(np.ptp(a)), float(np.ptp(b)), 1e-12)
    return float(np.clip(1.0 - diff / rng, 0.0, 1.0))


# ─── extract_edge_sample ──────────────────────────────────────────────────────

def extract_edge_sample(
    image: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[EdgeCompConfig] = None,
) -> EdgeSample:
    """Извлечь профиль из полосы вдоль правого края изображения.

    Аргументы:
        image:       Изображение фрагмента (H×W или H×W×C), dtype float/uint8.
        fragment_id: Идентификатор фрагмента.
        cfg:         Параметры сравнения.

    Возвращает:
        EdgeSample.

    Исключения:
        ValueError: Если изображение имеет неподдерживаемый формат.
    """
    if cfg is None:
        cfg = EdgeCompConfig()
    if image.ndim not in (2, 3):
        raise ValueError("image должен быть 2D или 3D массивом")

    # Привести к 2D grayscale
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(float)
    else:
        gray = image.astype(float)

    h, w = gray.shape
    strip_w = min(cfg.strip_width, w)

    # Полоса: правые strip_w столбцов
    strip = gray[:, max(w - strip_w, 0):]

    # Ресемплинг по высоте до n_samples
    indices = np.linspace(0, h - 1, cfg.n_samples).astype(int)
    intensity_raw = np.array([float(np.mean(strip[i])) for i in indices])

    # Градиент (разности)
    if len(intensity_raw) > 1:
        gradient_raw = np.gradient(intensity_raw)
    else:
        gradient_raw = np.zeros_like(intensity_raw)

    # Текстура (стандартное отклонение строки)
    texture_raw = np.array([float(np.std(strip[i])) for i in indices])

    if cfg.normalize:
        def _norm(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            rng = mx - mn
            if rng < 1e-12:
                return np.zeros_like(arr)
            return (arr - mn) / rng

        intensity_raw = _norm(intensity_raw)
        gradient_raw = _norm(gradient_raw)
        texture_raw = _norm(texture_raw)

    return EdgeSample(
        fragment_id=fragment_id,
        intensity=intensity_raw,
        gradient=gradient_raw,
        texture=texture_raw,
    )


# ─── compare_edge_intensity ───────────────────────────────────────────────────

def compare_edge_intensity(a: EdgeSample, b: EdgeSample) -> float:
    """Сравнить края по интенсивности.

    Аргументы:
        a, b: EdgeSample одинаковой длины.

    Возвращает:
        Оценка [0, 1].
    """
    return _ncc(a.intensity, b.intensity)


# ─── compare_edge_gradient ────────────────────────────────────────────────────

def compare_edge_gradient(a: EdgeSample, b: EdgeSample) -> float:
    """Сравнить края по градиенту.

    Аргументы:
        a, b: EdgeSample.

    Возвращает:
        Оценка [0, 1].
    """
    return _ncc(a.gradient, b.gradient)


# ─── compare_edge_texture ─────────────────────────────────────────────────────

def compare_edge_texture(a: EdgeSample, b: EdgeSample) -> float:
    """Сравнить края по текстурной дисперсии.

    Аргументы:
        a, b: EdgeSample.

    Возвращает:
        Оценка [0, 1].
    """
    return _l1_similarity(a.texture, b.texture)


# ─── score_edge_comparison ────────────────────────────────────────────────────

def score_edge_comparison(
    intensity_score: float,
    gradient_score: float,
    texture_score: float,
    cfg: Optional[EdgeCompConfig] = None,
) -> float:
    """Вычислить взвешенную итоговую оценку из компонент.

    Аргументы:
        intensity_score: Оценка по интенсивности.
        gradient_score:  Оценка по градиенту.
        texture_score:   Оценка по текстуре.
        cfg:             Параметры (use_gradient, use_texture).

    Возвращает:
        Взвешенная оценка [0, 1].
    """
    if cfg is None:
        cfg = EdgeCompConfig()
    total = intensity_score
    count = 1
    if cfg.use_gradient:
        total += gradient_score
        count += 1
    if cfg.use_texture:
        total += texture_score
        count += 1
    return float(np.clip(total / count, 0.0, 1.0))


# ─── compare_edge_pair ────────────────────────────────────────────────────────

def compare_edge_pair(
    sample_a: EdgeSample,
    sample_b: EdgeSample,
    cfg: Optional[EdgeCompConfig] = None,
) -> EdgeCompResult:
    """Сравнить два EdgeSample и вернуть EdgeCompResult.

    Аргументы:
        sample_a: Первый профиль.
        sample_b: Второй профиль.
        cfg:      Параметры сравнения.

    Возвращает:
        EdgeCompResult.
    """
    if cfg is None:
        cfg = EdgeCompConfig()

    i_score = compare_edge_intensity(sample_a, sample_b)
    g_score = compare_edge_gradient(sample_a, sample_b)
    t_score = compare_edge_texture(sample_a, sample_b)
    total = score_edge_comparison(i_score, g_score, t_score, cfg)

    return EdgeCompResult(
        pair=(sample_a.fragment_id, sample_b.fragment_id),
        intensity_score=i_score,
        gradient_score=g_score,
        texture_score=t_score,
        total_score=total,
        scores={
            "intensity": i_score,
            "gradient": g_score,
            "texture": t_score,
        },
    )


# ─── batch_compare_edges ──────────────────────────────────────────────────────

def batch_compare_edges(
    samples: List[EdgeSample],
    cfg: Optional[EdgeCompConfig] = None,
) -> List[EdgeCompResult]:
    """Попарно сравнить все EdgeSample из списка.

    Аргументы:
        samples: Список EdgeSample.
        cfg:     Параметры сравнения.

    Возвращает:
        Список EdgeCompResult для каждой пары (i, j), i < j.
    """
    results: List[EdgeCompResult] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            results.append(compare_edge_pair(samples[i], samples[j], cfg))
    return results
