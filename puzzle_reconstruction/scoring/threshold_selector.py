"""Выбор порогового значения для бинаризации оценок совместимости.

Модуль предоставляет несколько стратегий выбора порога: фиксированный,
по процентилю, методом Отсу, по максимальной F-меры и адаптивный
(ансамблевый средний из нескольких методов).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


_VALID_METHODS = {"fixed", "percentile", "otsu", "f1", "adaptive"}


# ─── ThresholdConfig ──────────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """Параметры выбора порога.

    Атрибуты:
        method:      'fixed' | 'percentile' | 'otsu' | 'f1' | 'adaptive'.
        fixed_value: Фиксированный порог (для method='fixed'; >= 0).
        percentile:  Процентиль (0–100; для method='percentile').
        n_bins:      Число бинов гистограммы для Отсу (>= 2).
        beta:        Вес полноты в F-мере (> 0).
    """

    method: str = "percentile"
    fixed_value: float = 0.5
    percentile: float = 50.0
    n_bins: int = 256
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method должен быть одним из {_VALID_METHODS}, "
                f"получено '{self.method}'"
            )
        if self.fixed_value < 0.0:
            raise ValueError(
                f"fixed_value должен быть >= 0, получено {self.fixed_value}"
            )
        if not (0.0 <= self.percentile <= 100.0):
            raise ValueError(
                f"percentile должен быть в [0, 100], "
                f"получено {self.percentile}"
            )
        if self.n_bins < 2:
            raise ValueError(
                f"n_bins должен быть >= 2, получено {self.n_bins}"
            )
        if self.beta <= 0.0:
            raise ValueError(
                f"beta должен быть > 0, получено {self.beta}"
            )


# ─── ThresholdResult ──────────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    """Результат выбора порога.

    Атрибуты:
        threshold:  Выбранное значение (>= 0).
        method:     Использованный метод.
        n_above:    Число оценок >= threshold (>= 0).
        n_below:    Число оценок < threshold (>= 0).
        n_total:    Общее число оценок (>= 0).
    """

    threshold: float
    method: str
    n_above: int
    n_below: int
    n_total: int

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError(
                f"threshold должен быть >= 0, получено {self.threshold}"
            )
        for name, val in (
            ("n_above", self.n_above),
            ("n_below", self.n_below),
            ("n_total", self.n_total),
        ):
            if val < 0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )

    @property
    def acceptance_ratio(self) -> float:
        """Доля оценок выше порога (0.0 если n_total == 0)."""
        if self.n_total == 0:
            return 0.0
        return float(self.n_above) / float(self.n_total)

    @property
    def rejection_ratio(self) -> float:
        """Доля оценок ниже порога (0.0 если n_total == 0)."""
        if self.n_total == 0:
            return 0.0
        return float(self.n_below) / float(self.n_total)


# ─── _make_result ─────────────────────────────────────────────────────────────

def _make_result(scores: np.ndarray, threshold: float, method: str) -> ThresholdResult:
    """Вспомогательная функция: подсчёт n_above/n_below."""
    n_above = int(np.sum(scores >= threshold))
    n_below = int(np.sum(scores < threshold))
    return ThresholdResult(
        threshold=float(threshold),
        method=method,
        n_above=n_above,
        n_below=n_below,
        n_total=len(scores),
    )


# ─── select_fixed_threshold ───────────────────────────────────────────────────

def select_fixed_threshold(
    scores: np.ndarray,
    value: float = 0.5,
) -> ThresholdResult:
    """Использовать фиксированный порог.

    Аргументы:
        scores: Массив оценок (1D).
        value:  Порог (>= 0).

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если value < 0 или scores пустой.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) == 0:
        raise ValueError("scores не должен быть пустым")
    if value < 0.0:
        raise ValueError(f"value должен быть >= 0, получено {value}")
    return _make_result(scores, value, "fixed")


# ─── select_percentile_threshold ──────────────────────────────────────────────

def select_percentile_threshold(
    scores: np.ndarray,
    percentile: float = 50.0,
) -> ThresholdResult:
    """Порог по заданному процентилю распределения оценок.

    Аргументы:
        scores:     Массив оценок (1D).
        percentile: Процентиль в [0, 100].

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если scores пустой или percentile вне [0, 100].
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) == 0:
        raise ValueError("scores не должен быть пустым")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(
            f"percentile должен быть в [0, 100], получено {percentile}"
        )
    threshold = float(np.percentile(scores, percentile))
    return _make_result(scores, threshold, "percentile")


# ─── select_otsu_threshold ────────────────────────────────────────────────────

def select_otsu_threshold(
    scores: np.ndarray,
    n_bins: int = 256,
) -> ThresholdResult:
    """Порог методом Отсу (максимизация межклассовой дисперсии).

    Аргументы:
        scores: Массив оценок (1D).
        n_bins: Число бинов (>= 2).

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если scores пустой или n_bins < 2.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) == 0:
        raise ValueError("scores не должен быть пустым")
    if n_bins < 2:
        raise ValueError(f"n_bins должен быть >= 2, получено {n_bins}")

    s_min, s_max = scores.min(), scores.max()
    if s_min == s_max:
        return _make_result(scores, float(s_min), "otsu")

    edges = np.linspace(s_min, s_max, n_bins + 1)
    hist, _ = np.histogram(scores, bins=edges)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return _make_result(scores, float(s_min), "otsu")

    # Отсу по гистограмме
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    w_cumsum = np.cumsum(hist) / total
    mu_cumsum = np.cumsum(hist * bin_centers) / total

    mu_total = mu_cumsum[-1]
    w1 = w_cumsum[:-1]
    w2 = 1.0 - w1
    mu1 = np.divide(mu_cumsum[:-1], w1, out=np.zeros_like(w1), where=w1 > 0)
    mu2 = np.divide(
        (mu_total - mu_cumsum[:-1]),
        w2,
        out=np.zeros_like(w2),
        where=w2 > 0,
    )

    sigma_b2 = w1 * w2 * (mu1 - mu2) ** 2
    best_idx = int(np.argmax(sigma_b2))
    threshold = float(bin_centers[best_idx])
    return _make_result(scores, threshold, "otsu")


# ─── select_f1_threshold ──────────────────────────────────────────────────────

def select_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_candidates: int = 50,
    beta: float = 1.0,
) -> ThresholdResult:
    """Порог по максимуму F_beta на наборе кандидатов.

    Аргументы:
        scores:       Массив оценок (1D).
        labels:       Бинарные метки (0/1; len == len(scores)).
        n_candidates: Число кандидатов-порогов (>= 2).
        beta:         Вес полноты (> 0).

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если размеры не совпадают, beta <= 0 или n_candidates < 2.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if len(scores) == 0:
        raise ValueError("scores не должен быть пустым")
    if len(scores) != len(labels):
        raise ValueError(
            f"Длины scores ({len(scores)}) и labels ({len(labels)}) "
            f"не совпадают"
        )
    if beta <= 0.0:
        raise ValueError(f"beta должен быть > 0, получено {beta}")
    if n_candidates < 2:
        raise ValueError(
            f"n_candidates должен быть >= 2, получено {n_candidates}"
        )

    candidates = np.linspace(scores.min(), scores.max(), n_candidates)
    best_thresh = candidates[0]
    best_fb = -1.0
    b2 = beta ** 2

    for t in candidates:
        pred = (scores >= t).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        denom = (1.0 + b2) * tp + b2 * fn + fp
        fb = (1.0 + b2) * tp / denom if denom > 0 else 0.0
        if fb > best_fb:
            best_fb = fb
            best_thresh = t

    return _make_result(scores, float(best_thresh), "f1")


# ─── select_adaptive_threshold ────────────────────────────────────────────────

def select_adaptive_threshold(
    scores: np.ndarray,
    n_bins: int = 256,
) -> ThresholdResult:
    """Адаптивный порог — среднее из методов percentile и otsu.

    Аргументы:
        scores: Массив оценок (1D).
        n_bins: Число бинов для метода Отсу (>= 2).

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если scores пустой.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) == 0:
        raise ValueError("scores не должен быть пустым")

    t_pct = select_percentile_threshold(scores, 50.0).threshold
    t_otsu = select_otsu_threshold(scores, n_bins).threshold
    threshold = (t_pct + t_otsu) / 2.0
    return _make_result(scores, threshold, "adaptive")


# ─── select_threshold ─────────────────────────────────────────────────────────

def select_threshold(
    scores: np.ndarray,
    cfg: Optional[ThresholdConfig] = None,
    labels: Optional[np.ndarray] = None,
) -> ThresholdResult:
    """Выбрать порог согласно конфигурации.

    Аргументы:
        scores: Массив оценок (1D).
        cfg:    Параметры (None → ThresholdConfig()).
        labels: Метки для method='f1' (обязательны при method='f1').

    Возвращает:
        ThresholdResult.

    Исключения:
        ValueError: Если method='f1' и labels не переданы.
    """
    if cfg is None:
        cfg = ThresholdConfig()

    scores = np.asarray(scores, dtype=float).ravel()

    if cfg.method == "fixed":
        return select_fixed_threshold(scores, cfg.fixed_value)
    elif cfg.method == "percentile":
        return select_percentile_threshold(scores, cfg.percentile)
    elif cfg.method == "otsu":
        return select_otsu_threshold(scores, cfg.n_bins)
    elif cfg.method == "f1":
        if labels is None:
            raise ValueError(
                "labels обязательны при method='f1'"
            )
        return select_f1_threshold(scores, labels, beta=cfg.beta)
    else:  # adaptive
        return select_adaptive_threshold(scores, cfg.n_bins)


# ─── apply_threshold ──────────────────────────────────────────────────────────

def apply_threshold(
    scores: np.ndarray,
    result: ThresholdResult,
) -> np.ndarray:
    """Применить порог к массиву оценок.

    Аргументы:
        scores: Массив оценок (1D).
        result: ThresholdResult.

    Возвращает:
        Булев массив (True = выше порога).
    """
    return np.asarray(scores, dtype=float).ravel() >= result.threshold


# ─── batch_select_thresholds ──────────────────────────────────────────────────

def batch_select_thresholds(
    score_arrays: List[np.ndarray],
    cfg: Optional[ThresholdConfig] = None,
) -> List[ThresholdResult]:
    """Выбрать пороги для списка массивов оценок.

    Аргументы:
        score_arrays: Список массивов оценок.
        cfg:          Параметры.

    Возвращает:
        Список ThresholdResult.
    """
    return [select_threshold(s, cfg) for s in score_arrays]
