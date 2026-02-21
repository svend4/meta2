"""Сопоставление фрагментов на основе ориентации краёв.

Модуль предоставляет структуры и функции для оценки совместимости ориентаций
двух фрагментов с опциональным перебором углов поворота.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── OrientConfig ─────────────────────────────────────────────────────────────

@dataclass
class OrientConfig:
    """Параметры сопоставления по ориентации.

    Атрибуты:
        n_bins:        Число бинов гистограммы ориентаций (>= 2).
        angle_step:    Шаг перебора углов в градусах (> 0).
        max_angle:     Максимальный угол поворота в градусах (>= 0).
        normalize:     Нормализовать гистограммы.
        use_flip:      Проверять также зеркальное отражение.
    """

    n_bins: int = 36
    angle_step: float = 10.0
    max_angle: float = 180.0
    normalize: bool = True
    use_flip: bool = False

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(
                f"n_bins должен быть >= 2, получено {self.n_bins}"
            )
        if self.angle_step <= 0:
            raise ValueError(
                f"angle_step должен быть > 0, получено {self.angle_step}"
            )
        if self.max_angle < 0:
            raise ValueError(
                f"max_angle должен быть >= 0, получено {self.max_angle}"
            )


# ─── OrientProfile ────────────────────────────────────────────────────────────

@dataclass
class OrientProfile:
    """Гистограмма ориентаций краёв фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        histogram:   Массив ориентаций (shape: n_bins).
        dominant:    Угол доминирующего направления в градусах [0, 360).
    """

    fragment_id: int
    histogram: np.ndarray
    dominant: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.histogram.ndim != 1:
            raise ValueError("histogram должен быть одномерным массивом")
        if len(self.histogram) < 2:
            raise ValueError(
                f"histogram должен иметь >= 2 элементов, получено {len(self.histogram)}"
            )
        if not (0.0 <= self.dominant < 360.0):
            raise ValueError(
                f"dominant должен быть в [0, 360), получено {self.dominant}"
            )

    @property
    def n_bins(self) -> int:
        """Число бинов."""
        return len(self.histogram)

    @property
    def is_uniform(self) -> bool:
        """True если гистограмма равномерная (нет явного доминирующего угла)."""
        if self.histogram.sum() < 1e-12:
            return True
        norm = self.histogram / (self.histogram.sum() + 1e-12)
        return float(norm.std()) < 0.05


# ─── OrientMatchResult ────────────────────────────────────────────────────────

@dataclass
class OrientMatchResult:
    """Результат сопоставления ориентаций двух фрагментов.

    Атрибуты:
        pair:          Пара (fragment_id_a, fragment_id_b).
        best_angle:    Лучший угол выравнивания в градусах.
        best_score:    Оценка при лучшем угле [0, 1].
        angle_scores:  Словарь {угол: оценка} для всех проверенных углов.
        is_flipped:    True если лучший результат достигнут с отражением.
    """

    pair: Tuple[int, int]
    best_angle: float
    best_score: float
    angle_scores: Dict[float, float] = field(default_factory=dict)
    is_flipped: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.best_score <= 1.0):
            raise ValueError(
                f"best_score должен быть в [0, 1], получено {self.best_score}"
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
    def n_angles_tested(self) -> int:
        """Число проверенных углов."""
        return len(self.angle_scores)


# ─── _histogram_intersection ──────────────────────────────────────────────────

def _histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Пересечение нормированных гистограмм [0, 1]."""
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum < 1e-12 or b_sum < 1e-12:
        return 0.0
    a_n = a / a_sum
    b_n = b / b_sum
    return float(np.minimum(a_n, b_n).sum())


def _shift_histogram(hist: np.ndarray, bins: int) -> np.ndarray:
    """Сдвинуть гистограмму на bins позиций вправо."""
    return np.roll(hist, bins)


# ─── compute_orient_profile ───────────────────────────────────────────────────

def compute_orient_profile(
    image: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[OrientConfig] = None,
) -> OrientProfile:
    """Вычислить профиль ориентаций краёв изображения.

    Аргументы:
        image:       2D или 3D изображение (H×W или H×W×C).
        fragment_id: Идентификатор фрагмента.
        cfg:         Параметры.

    Возвращает:
        OrientProfile.

    Исключения:
        ValueError: Если image имеет неподдерживаемый формат.
    """
    if cfg is None:
        cfg = OrientConfig()
    if image.ndim not in (2, 3):
        raise ValueError("image должен быть 2D или 3D массивом")

    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(float)
    else:
        gray = image.astype(float)

    # Градиенты Собела
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)

    # Угол в диапазоне [0, 360)
    angles = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)

    # Взвешенная гистограмма
    hist, _ = np.histogram(
        angles.ravel(),
        bins=cfg.n_bins,
        range=(0.0, 360.0),
        weights=magnitudes.ravel(),
    )
    hist = hist.astype(float)

    if cfg.normalize and hist.sum() > 1e-12:
        hist = hist / hist.sum()

    dominant_bin = int(np.argmax(hist))
    dominant = float(dominant_bin / cfg.n_bins * 360.0)

    return OrientProfile(
        fragment_id=fragment_id,
        histogram=hist,
        dominant=dominant,
    )


# ─── orient_similarity ────────────────────────────────────────────────────────

def orient_similarity(
    profile_a: OrientProfile,
    profile_b: OrientProfile,
    angle_deg: float = 0.0,
    cfg: Optional[OrientConfig] = None,
) -> float:
    """Оценить схожесть двух профилей при заданном угле поворота.

    Аргументы:
        profile_a:  Первый профиль.
        profile_b:  Второй профиль.
        angle_deg:  Угол поворота профиля b в градусах.
        cfg:        Параметры.

    Возвращает:
        Оценка [0, 1].
    """
    if cfg is None:
        cfg = OrientConfig()
    n = profile_a.n_bins
    shift_bins = int(round(angle_deg / 360.0 * n)) % n
    shifted_b = _shift_histogram(profile_b.histogram, shift_bins)
    return _histogram_intersection(profile_a.histogram, shifted_b)


# ─── best_orient_angle ────────────────────────────────────────────────────────

def best_orient_angle(
    profile_a: OrientProfile,
    profile_b: OrientProfile,
    cfg: Optional[OrientConfig] = None,
) -> Tuple[float, float]:
    """Найти угол поворота, дающий наилучшее совпадение.

    Аргументы:
        profile_a: Первый профиль.
        profile_b: Второй профиль.
        cfg:       Параметры.

    Возвращает:
        Кортеж (best_angle_degrees, best_score).
    """
    if cfg is None:
        cfg = OrientConfig()
    if cfg.angle_step <= 0:
        raise ValueError("angle_step должен быть > 0")

    best_angle = 0.0
    best_score = -1.0
    angle = 0.0
    while angle <= cfg.max_angle:
        score = orient_similarity(profile_a, profile_b, angle, cfg)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += cfg.angle_step

    return best_angle, float(np.clip(best_score, 0.0, 1.0))


# ─── match_orient_pair ────────────────────────────────────────────────────────

def match_orient_pair(
    profile_a: OrientProfile,
    profile_b: OrientProfile,
    cfg: Optional[OrientConfig] = None,
) -> OrientMatchResult:
    """Сопоставить два профиля ориентаций по всем углам.

    Аргументы:
        profile_a: Первый профиль.
        profile_b: Второй профиль.
        cfg:       Параметры.

    Возвращает:
        OrientMatchResult.
    """
    if cfg is None:
        cfg = OrientConfig()

    angle_scores: Dict[float, float] = {}
    best_angle = 0.0
    best_score = -1.0
    is_flipped = False

    # Прямое сравнение
    angle = 0.0
    while angle <= cfg.max_angle:
        score = orient_similarity(profile_a, profile_b, angle, cfg)
        angle_scores[angle] = float(score)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += cfg.angle_step

    # Зеркальное
    if cfg.use_flip:
        flipped_hist = np.flip(profile_b.histogram)
        flipped_profile = OrientProfile(
            fragment_id=profile_b.fragment_id,
            histogram=flipped_hist,
            dominant=profile_b.dominant,
        )
        angle = 0.0
        while angle <= cfg.max_angle:
            score = orient_similarity(profile_a, flipped_profile, angle, cfg)
            flip_key = -(angle + 1.0)  # Отличительный ключ
            angle_scores[flip_key] = float(score)
            if score > best_score:
                best_score = score
                best_angle = angle
                is_flipped = True
            angle += cfg.angle_step

    return OrientMatchResult(
        pair=(profile_a.fragment_id, profile_b.fragment_id),
        best_angle=best_angle,
        best_score=float(np.clip(best_score, 0.0, 1.0)),
        angle_scores=angle_scores,
        is_flipped=is_flipped,
    )


# ─── batch_orient_match ───────────────────────────────────────────────────────

def batch_orient_match(
    profiles: List[OrientProfile],
    cfg: Optional[OrientConfig] = None,
) -> List[OrientMatchResult]:
    """Попарно сопоставить все профили.

    Аргументы:
        profiles: Список OrientProfile.
        cfg:      Параметры.

    Возвращает:
        Список OrientMatchResult для каждой пары (i, j), i < j.
    """
    results: List[OrientMatchResult] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            results.append(match_orient_pair(profiles[i], profiles[j], cfg))
    return results
