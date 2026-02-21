"""Патч-матчинг изображений фрагментов пазла.

Модуль реализует поиск соответствий на основе скользящего окна
(template matching) с поддержкой NCC, SSD и SAD метрик.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


_METHODS = {"ncc", "ssd", "sad"}


# ─── PatchConfig ──────────────────────────────────────────────────────────────

@dataclass
class PatchConfig:
    """Параметры патч-матчинга.

    Атрибуты:
        patch_size: Размер патча (нечётное >= 3).
        stride:     Шаг скользящего окна (>= 1).
        method:     Метрика: 'ncc', 'ssd', 'sad'.
        max_matches: Максимальное число результатов (>= 1).
    """

    patch_size: int = 17
    stride: int = 4
    method: str = "ncc"
    max_matches: int = 50

    def __post_init__(self) -> None:
        if self.patch_size < 3:
            raise ValueError(
                f"patch_size должен быть >= 3, получено {self.patch_size}"
            )
        if self.patch_size % 2 == 0:
            raise ValueError(
                f"patch_size должен быть нечётным, получено {self.patch_size}"
            )
        if self.stride < 1:
            raise ValueError(
                f"stride должен быть >= 1, получено {self.stride}"
            )
        if self.method not in _METHODS:
            raise ValueError(
                f"method должен быть одним из {_METHODS}, получено '{self.method}'"
            )
        if self.max_matches < 1:
            raise ValueError(
                f"max_matches должен быть >= 1, получено {self.max_matches}"
            )


# ─── PatchMatch ───────────────────────────────────────────────────────────────

@dataclass
class PatchMatch:
    """Результат сопоставления одного патча.

    Атрибуты:
        row1, col1: Координаты (верхний левый угол) в первом изображении.
        row2, col2: Координаты лучшего совпадения во втором изображении.
        score:      Оценка сходства (NCC: [-1,1]; SSD/SAD: >= 0).
        method:     Использованная метрика.
    """

    row1: int
    col1: int
    row2: int
    col2: int
    score: float
    method: str = "ncc"

    def __post_init__(self) -> None:
        for name, val in (("row1", self.row1), ("col1", self.col1),
                          ("row2", self.row2), ("col2", self.col2)):
            if val < 0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )
        if self.method not in _METHODS:
            raise ValueError(
                f"method должен быть одним из {_METHODS}, получено '{self.method}'"
            )

    @property
    def src_pos(self) -> Tuple[int, int]:
        """(row, col) в первом изображении."""
        return (self.row1, self.col1)

    @property
    def dst_pos(self) -> Tuple[int, int]:
        """(row, col) во втором изображении."""
        return (self.row2, self.col2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Конвертировать цветное изображение в grayscale float32."""
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


# ─── extract_patch ────────────────────────────────────────────────────────────

def extract_patch(
    img: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
) -> np.ndarray:
    """Извлечь патч из изображения.

    Аргументы:
        img:        Изображение (2-D или 3-D).
        row, col:   Координаты верхнего левого угла патча.
        patch_size: Размер патча (нечётное >= 3).

    Возвращает:
        Патч (patch_size × patch_size, float32).

    Исключения:
        ValueError: Если координаты выходят за границы или patch_size < 3.
    """
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(
            f"patch_size должен быть нечётным и >= 3, получено {patch_size}"
        )
    g = _to_gray(img)
    h, w = g.shape[:2]
    if row < 0 or col < 0 or row + patch_size > h or col + patch_size > w:
        raise ValueError(
            f"Патч ({row},{col},{patch_size}) выходит за границы "
            f"изображения {h}×{w}"
        )
    return g[row:row + patch_size, col:col + patch_size].copy()


# ─── Score functions ──────────────────────────────────────────────────────────

def ncc_score(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """Нормированная кросс-корреляция двух патчей одинакового размера.

    Возвращает:
        float в [-1, 1].

    Исключения:
        ValueError: Если формы патчей не совпадают.
    """
    if patch1.shape != patch2.shape:
        raise ValueError(
            f"Формы патчей не совпадают: {patch1.shape} vs {patch2.shape}"
        )
    p1 = patch1.astype(np.float64)
    p2 = patch2.astype(np.float64)
    p1 -= p1.mean()
    p2 -= p2.mean()
    denom = (np.linalg.norm(p1) * np.linalg.norm(p2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(p1 * p2) / denom)


def ssd_score(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """Сумма квадратов разностей (меньше = лучше).

    Возвращает:
        float >= 0.

    Исключения:
        ValueError: Если формы патчей не совпадают.
    """
    if patch1.shape != patch2.shape:
        raise ValueError(
            f"Формы патчей не совпадают: {patch1.shape} vs {patch2.shape}"
        )
    diff = patch1.astype(np.float64) - patch2.astype(np.float64)
    return float(np.sum(diff ** 2))


def sad_score(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """Сумма абсолютных разностей (меньше = лучше).

    Возвращает:
        float >= 0.

    Исключения:
        ValueError: Если формы патчей не совпадают.
    """
    if patch1.shape != patch2.shape:
        raise ValueError(
            f"Формы патчей не совпадают: {patch1.shape} vs {patch2.shape}"
        )
    diff = patch1.astype(np.float64) - patch2.astype(np.float64)
    return float(np.sum(np.abs(diff)))


# ─── match_patch_in_image ─────────────────────────────────────────────────────

def match_patch_in_image(
    template: np.ndarray,
    img: np.ndarray,
    method: str = "ncc",
    stride: int = 1,
) -> Tuple[int, int, float]:
    """Найти лучшее совпадение шаблона в изображении (скользящее окно).

    Аргументы:
        template: Шаблон (H×W, float32/uint8).
        img:      Целевое изображение (>= H×W).
        method:   'ncc', 'ssd', 'sad'.
        stride:   Шаг окна (>= 1).

    Возвращает:
        (best_row, best_col, best_score).

    Исключения:
        ValueError: Если метод неизвестен или stride < 1.
    """
    if method not in _METHODS:
        raise ValueError(
            f"method должен быть одним из {_METHODS}, получено '{method}'"
        )
    if stride < 1:
        raise ValueError(
            f"stride должен быть >= 1, получено {stride}"
        )

    tmpl = _to_gray(template)
    target = _to_gray(img)
    th, tw = tmpl.shape
    h, w = target.shape

    if th > h or tw > w:
        raise ValueError(
            f"Шаблон ({th}×{tw}) больше целевого изображения ({h}×{w})"
        )

    score_fn = {"ncc": ncc_score, "ssd": ssd_score, "sad": sad_score}[method]
    best_score = -np.inf if method == "ncc" else np.inf
    best_row, best_col = 0, 0

    for r in range(0, h - th + 1, stride):
        for c in range(0, w - tw + 1, stride):
            patch = target[r:r + th, c:c + tw]
            s = score_fn(tmpl, patch)
            if method == "ncc":
                if s > best_score:
                    best_score, best_row, best_col = s, r, c
            else:
                if s < best_score:
                    best_score, best_row, best_col = s, r, c

    return best_row, best_col, float(best_score)


# ─── find_matches ─────────────────────────────────────────────────────────────

def find_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    cfg: Optional[PatchConfig] = None,
) -> List[PatchMatch]:
    """Найти совпадения патчей между двумя изображениями.

    Извлекает патчи из img1 с заданным шагом и ищет лучшее совпадение
    каждого в img2.

    Аргументы:
        img1, img2: Изображения (2-D или 3-D).
        cfg:        Параметры (None → PatchConfig()).

    Возвращает:
        Список PatchMatch.
    """
    if cfg is None:
        cfg = PatchConfig()

    g1 = _to_gray(img1)
    g2 = _to_gray(img2)
    h1, w1 = g1.shape
    ps = cfg.patch_size

    matches: List[PatchMatch] = []
    for r in range(0, h1 - ps + 1, cfg.stride):
        for c in range(0, w1 - ps + 1, cfg.stride):
            tmpl = g1[r:r + ps, c:c + ps]
            try:
                br, bc, sc = match_patch_in_image(
                    tmpl, g2, method=cfg.method, stride=cfg.stride
                )
            except ValueError:
                continue
            matches.append(
                PatchMatch(
                    row1=r, col1=c,
                    row2=br, col2=bc,
                    score=sc,
                    method=cfg.method,
                )
            )

    return matches


# ─── top_matches ──────────────────────────────────────────────────────────────

def top_matches(
    matches: List[PatchMatch],
    k: int,
    method: str = "ncc",
) -> List[PatchMatch]:
    """Вернуть топ-K лучших совпадений.

    Для NCC: сортировка по убыванию score.
    Для SSD/SAD: сортировка по возрастанию score.

    Аргументы:
        matches: Список PatchMatch.
        k:       Количество лучших (>= 1).
        method:  Метрика для определения порядка.

    Возвращает:
        Отсортированный список (не более k элементов).

    Исключения:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    if not matches:
        return []
    reverse = (method == "ncc")
    return sorted(matches, key=lambda m: m.score, reverse=reverse)[:k]


# ─── batch_patch_match ────────────────────────────────────────────────────────

def batch_patch_match(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Optional[PatchConfig] = None,
) -> List[List[PatchMatch]]:
    """Поиск патч-совпадений для нескольких пар изображений.

    Аргументы:
        pairs: Список пар (img1, img2).
        cfg:   Параметры (None → PatchConfig()).

    Возвращает:
        Список списков PatchMatch.
    """
    return [find_matches(i1, i2, cfg) for i1, i2 in pairs]
