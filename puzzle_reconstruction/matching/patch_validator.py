"""Валидация совместимости патчей на границах фрагментов.

Модуль проверяет, насколько хорошо патч одного фрагмента стыкуется
с патчем другого, используя цветовые, текстурные и градиентные метрики.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── PatchValidConfig ─────────────────────────────────────────────────────────

@dataclass
class PatchValidConfig:
    """Параметры валидации патчей.

    Атрибуты:
        color_weight:    Вес цветовой метрики [0, 1].
        texture_weight:  Вес текстурной метрики [0, 1].
        gradient_weight: Вес градиентной метрики [0, 1].
        min_patch_size:  Минимальный размер патча в пикселях (>= 1).
        threshold:       Порог суммарной оценки [0, 1]; ниже → отклонить.
    """

    color_weight: float = 0.4
    texture_weight: float = 0.3
    gradient_weight: float = 0.3
    min_patch_size: int = 4
    threshold: float = 0.5

    def __post_init__(self) -> None:
        for name, val in (
            ("color_weight", self.color_weight),
            ("texture_weight", self.texture_weight),
            ("gradient_weight", self.gradient_weight),
        ):
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )
        if self.min_patch_size < 1:
            raise ValueError(
                f"min_patch_size должен быть >= 1, получено {self.min_patch_size}"
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold должен быть в [0, 1], получено {self.threshold}"
            )

    @property
    def weight_sum(self) -> float:
        """Сумма весов."""
        return self.color_weight + self.texture_weight + self.gradient_weight

    def normalized_weights(self) -> Tuple[float, float, float]:
        """Нормированные веса (сумма = 1)."""
        s = self.weight_sum
        if s < 1e-12:
            return (1.0 / 3, 1.0 / 3, 1.0 / 3)
        return (self.color_weight / s,
                self.texture_weight / s,
                self.gradient_weight / s)


# ─── PatchScore ───────────────────────────────────────────────────────────────

@dataclass
class PatchScore:
    """Оценки совместимости двух патчей.

    Атрибуты:
        color_score:    Цветовая совместимость [0, 1].
        texture_score:  Текстурная совместимость [0, 1].
        gradient_score: Градиентная совместимость [0, 1].
        total_score:    Взвешенная итоговая оценка [0, 1].
        valid:          True если total_score >= threshold.
    """

    color_score: float
    texture_score: float
    gradient_score: float
    total_score: float
    valid: bool = True

    def __post_init__(self) -> None:
        for name, val in (
            ("color_score", self.color_score),
            ("texture_score", self.texture_score),
            ("gradient_score", self.gradient_score),
            ("total_score", self.total_score),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )

    @property
    def is_strong(self) -> bool:
        """True если total_score > 0.8."""
        return self.total_score > 0.8

    @property
    def dominant_channel(self) -> str:
        """Канал с наибольшей оценкой."""
        ch = {"color": self.color_score,
              "texture": self.texture_score,
              "gradient": self.gradient_score}
        return max(ch, key=lambda k: ch[k])


# ─── PatchValidResult ─────────────────────────────────────────────────────────

@dataclass
class PatchValidResult:
    """Результат валидации пары фрагментов.

    Атрибуты:
        fragment_a:  ID первого фрагмента.
        fragment_b:  ID второго фрагмента.
        score:       PatchScore с детализацией.
        n_patches:   Число проверенных патчей.
        passed:      True если валидация пройдена.
    """

    fragment_a: int
    fragment_b: int
    score: PatchScore
    n_patches: int
    passed: bool

    def __post_init__(self) -> None:
        if self.n_patches < 0:
            raise ValueError(
                f"n_patches должен быть >= 0, получено {self.n_patches}"
            )

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Упорядоченная пара (min, max)."""
        a, b = self.fragment_a, self.fragment_b
        return (min(a, b), max(a, b))

    @property
    def avg_score(self) -> float:
        """Средняя оценка по трём каналам."""
        return (self.score.color_score
                + self.score.texture_score
                + self.score.gradient_score) / 3.0


# ─── _color_similarity ────────────────────────────────────────────────────────

def _color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Цветовая близость двух патчей (нормированное среднее отклонение)."""
    fa = a.astype(float).ravel()
    fb = b.astype(float).ravel()
    n = min(fa.size, fb.size)
    if n == 0:
        return 0.0
    diff = np.abs(fa[:n] - fb[:n])
    # Нормируем на максимально возможное отклонение (255)
    max_val = max(float(fa[:n].max()), float(fb[:n].max()), 1.0)
    return float(1.0 - np.mean(diff) / max_val)


# ─── _texture_similarity ──────────────────────────────────────────────────────

def _texture_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Текстурная близость: близость стандартных отклонений."""
    sa = float(a.astype(float).std())
    sb = float(b.astype(float).std())
    denom = max(sa, sb, 1e-6)
    return float(1.0 - abs(sa - sb) / denom)


# ─── _gradient_similarity ─────────────────────────────────────────────────────

def _gradient_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Градиентная близость: разность средних абсолютных градиентов."""
    def _mag(arr: np.ndarray) -> float:
        f = arr.astype(float)
        if f.ndim == 3:
            f = f.mean(axis=2)
        if f.shape[0] < 2 or f.shape[1] < 2:
            return 0.0
        gx = np.abs(np.diff(f, axis=1)).mean()
        gy = np.abs(np.diff(f, axis=0)).mean()
        return float((gx + gy) / 2.0)

    ga, gb = _mag(a), _mag(b)
    denom = max(ga, gb, 1e-6)
    return float(1.0 - abs(ga - gb) / denom)


# ─── compute_patch_score ──────────────────────────────────────────────────────

def compute_patch_score(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    cfg: Optional[PatchValidConfig] = None,
) -> PatchScore:
    """Вычислить оценку совместимости двух патчей.

    Аргументы:
        patch_a: Патч первого фрагмента (2D или 3D).
        patch_b: Патч второго фрагмента (2D или 3D).
        cfg:     Параметры.

    Возвращает:
        PatchScore.

    Исключения:
        ValueError: если патч меньше min_patch_size.
    """
    if cfg is None:
        cfg = PatchValidConfig()

    min_sz = cfg.min_patch_size
    if patch_a.size < min_sz or patch_b.size < min_sz:
        raise ValueError(
            f"Патч меньше min_patch_size={min_sz}: "
            f"a={patch_a.size}, b={patch_b.size}"
        )

    c_score = float(np.clip(_color_similarity(patch_a, patch_b), 0.0, 1.0))
    t_score = float(np.clip(_texture_similarity(patch_a, patch_b), 0.0, 1.0))
    g_score = float(np.clip(_gradient_similarity(patch_a, patch_b), 0.0, 1.0))

    wc, wt, wg = cfg.normalized_weights()
    total = float(np.clip(wc * c_score + wt * t_score + wg * g_score, 0.0, 1.0))
    valid = total > cfg.threshold

    return PatchScore(
        color_score=c_score,
        texture_score=t_score,
        gradient_score=g_score,
        total_score=total,
        valid=valid,
    )


# ─── aggregate_patch_scores ───────────────────────────────────────────────────

def aggregate_patch_scores(
    scores: List[PatchScore],
    cfg: Optional[PatchValidConfig] = None,
) -> PatchScore:
    """Агрегировать список PatchScore в один итоговый.

    Аргументы:
        scores: Список PatchScore.
        cfg:    Параметры (threshold используется для valid).

    Возвращает:
        PatchScore (среднее по всем оценкам).

    Исключения:
        ValueError: если scores пуст.
    """
    if not scores:
        raise ValueError("scores не должен быть пустым")
    if cfg is None:
        cfg = PatchValidConfig()

    c = float(np.mean([s.color_score for s in scores]))
    t = float(np.mean([s.texture_score for s in scores]))
    g = float(np.mean([s.gradient_score for s in scores]))

    wc, wt, wg = cfg.normalized_weights()
    total = float(np.clip(wc * c + wt * t + wg * g, 0.0, 1.0))
    valid = total > cfg.threshold

    return PatchScore(color_score=c, texture_score=t, gradient_score=g,
                      total_score=total, valid=valid)


# ─── validate_patch_pair ──────────────────────────────────────────────────────

def validate_patch_pair(
    fragment_a: int,
    fragment_b: int,
    patches_a: List[np.ndarray],
    patches_b: List[np.ndarray],
    cfg: Optional[PatchValidConfig] = None,
) -> PatchValidResult:
    """Валидировать совместимость патчей двух фрагментов.

    Аргументы:
        fragment_a: ID фрагмента A.
        fragment_b: ID фрагмента B.
        patches_a:  Патчи фрагмента A.
        patches_b:  Патчи фрагмента B.
        cfg:        Параметры.

    Возвращает:
        PatchValidResult.
    """
    if cfg is None:
        cfg = PatchValidConfig()

    n = min(len(patches_a), len(patches_b))
    individual: List[PatchScore] = []
    for pa, pb in zip(patches_a[:n], patches_b[:n]):
        try:
            individual.append(compute_patch_score(pa, pb, cfg))
        except ValueError:
            continue  # патч слишком мал — пропустить

    if not individual:
        zero = PatchScore(color_score=0.0, texture_score=0.0,
                          gradient_score=0.0, total_score=0.0, valid=False)
        return PatchValidResult(fragment_a=fragment_a, fragment_b=fragment_b,
                                score=zero, n_patches=0, passed=False)

    agg = aggregate_patch_scores(individual, cfg)
    return PatchValidResult(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        score=agg,
        n_patches=len(individual),
        passed=agg.valid,
    )


# ─── batch_validate_patches ───────────────────────────────────────────────────

def batch_validate_patches(
    pairs: List[Tuple[int, int]],
    patch_map: Dict[int, List[np.ndarray]],
    cfg: Optional[PatchValidConfig] = None,
) -> Dict[Tuple[int, int], PatchValidResult]:
    """Валидировать патчи для списка пар фрагментов.

    Аргументы:
        pairs:     Список пар (id_a, id_b).
        patch_map: Словарь {fragment_id: [patches]}.
        cfg:       Параметры.

    Возвращает:
        Словарь {(id_a, id_b): PatchValidResult}.
    """
    if cfg is None:
        cfg = PatchValidConfig()

    results: Dict[Tuple[int, int], PatchValidResult] = {}
    for a, b in pairs:
        pa = patch_map.get(a, [])
        pb = patch_map.get(b, [])
        results[(a, b)] = validate_patch_pair(a, b, pa, pb, cfg)
    return results


# ─── filter_valid_pairs ───────────────────────────────────────────────────────

def filter_valid_pairs(
    results: Dict[Tuple[int, int], PatchValidResult],
) -> List[Tuple[int, int]]:
    """Вернуть только пары, прошедшие валидацию.

    Аргументы:
        results: Словарь {(a, b): PatchValidResult}.

    Возвращает:
        Список пар (a, b) у которых passed=True.
    """
    return [pair for pair, res in results.items() if res.passed]
