"""Доминантные цвета фрагментов документа.

Модуль извлекает палитру доминантных цветов из изображения и вычисляет
попарное сходство палитр для ранжирования кандидатов при сборке пазла.

Классы:
    ColorPaletteConfig  — параметры алгоритма
    ColorPalette        — результат: массив цветов + веса

Функции:
    extract_dominant_colors  — k-means-подобная кластеризация цветов
    palette_distance         — расстояние между двумя палитрами
    compute_palette          — полный пайплайн для одного изображения
    batch_compute_palettes   — пакетная обработка списка изображений
    rank_by_palette          — ранжирование кандидатов по сходству палитры
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── ColorPaletteConfig ───────────────────────────────────────────────────────

@dataclass
class ColorPaletteConfig:
    """Параметры извлечения палитры доминантных цветов.

    Атрибуты:
        n_colors:  Количество доминантных цветов (>= 2).
        max_iter:  Максимальное число итераций k-means (>= 1).
        tol:       Допуск сходимости (>= 0).
        seed:      Начальное значение генератора случайных чисел.
    """
    n_colors: int   = 8
    max_iter: int   = 20
    tol:      float = 1e-4
    seed:     int   = 0

    def __post_init__(self) -> None:
        if self.n_colors < 2:
            raise ValueError(
                f"n_colors должен быть >= 2, получено {self.n_colors}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter должен быть >= 1, получено {self.max_iter}"
            )
        if self.tol < 0.0:
            raise ValueError(
                f"tol должен быть >= 0, получено {self.tol}"
            )


# ─── ColorPalette ─────────────────────────────────────────────────────────────

@dataclass
class ColorPalette:
    """Палитра доминантных цветов изображения.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        colors:      Массив формы (n_colors, C) float32,
                     где C = 1 (grayscale) или 3 (BGR).
        weights:     Массив формы (n_colors,) float32, сумма = 1.
        n_colors:    Количество цветов в палитре.
        params:      Дополнительные параметры.
    """
    fragment_id: int
    colors:      np.ndarray
    weights:     np.ndarray
    n_colors:    int
    params:      Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.n_colors < 2:
            raise ValueError(
                f"n_colors должен быть >= 2, получено {self.n_colors}"
            )
        self.colors = np.asarray(self.colors, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.colors.ndim != 2:
            raise ValueError(
                f"colors должен быть 2-D (n_colors × C), "
                f"получено ndim={self.colors.ndim}"
            )
        if self.weights.ndim != 1:
            raise ValueError(
                f"weights должен быть 1-D, получено ndim={self.weights.ndim}"
            )
        if len(self.weights) != len(self.colors):
            raise ValueError(
                f"Длины colors ({len(self.colors)}) и weights "
                f"({len(self.weights)}) должны совпадать"
            )

    @property
    def dominant(self) -> np.ndarray:
        """Наиболее весомый цвет палитры (1-D массив float32)."""
        return self.colors[int(np.argmax(self.weights))]


# ─── extract_dominant_colors ──────────────────────────────────────────────────

def extract_dominant_colors(
    img:      np.ndarray,
    n_colors: int = 8,
    max_iter: int = 20,
    tol:      float = 1e-4,
    seed:     int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Извлечь доминантные цвета с помощью k-means.

    Аргументы:
        img:      Изображение (2-D grayscale или 3-D BGR), uint8.
        n_colors: Число кластеров (>= 2).
        max_iter: Максимальное число итераций.
        tol:      Допуск изменения центроидов.
        seed:     Начальное значение RNG для воспроизводимости.

    Возвращает:
        colors:  ndarray (n_colors, C) float32 — центроиды кластеров.
        weights: ndarray (n_colors,) float32  — доли пикселей в кластерах.

    Исключения:
        ValueError: Если n_colors < 2 или изображение пустое.
    """
    if n_colors < 2:
        raise ValueError(f"n_colors должен быть >= 2, получено {n_colors}")

    img_arr = np.asarray(img, dtype=np.float32)
    if img_arr.ndim == 2:
        pixels = img_arr.reshape(-1, 1)
    elif img_arr.ndim == 3:
        pixels = img_arr.reshape(-1, img_arr.shape[2])
    else:
        raise ValueError(
            f"img должен быть 2-D или 3-D, получено ndim={img_arr.ndim}"
        )

    n_pixels = len(pixels)
    if n_pixels == 0:
        raise ValueError("Изображение пустое — нет пикселей для кластеризации")

    k = min(n_colors, n_pixels)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pixels, size=k, replace=False)
    centers = pixels[idx].copy()

    for _ in range(max_iter):
        # Расстояния до каждого центра: (n_pixels, k)
        diffs = pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = pixels[mask].mean(axis=0)
            else:
                new_centers[c] = centers[c]

        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        if shift < tol:
            break

    # Веса = доли пикселей в кластерах
    counts = np.zeros(k, dtype=np.float32)
    for c in range(k):
        counts[c] = float(np.sum(labels == c))
    weights = counts / max(float(counts.sum()), 1e-9)

    # Дополнить до n_colors нулями, если k < n_colors (мало пикселей)
    if k < n_colors:
        pad = n_colors - k
        centers = np.vstack([
            centers,
            np.zeros((pad, centers.shape[1]), dtype=np.float32),
        ])
        weights = np.concatenate([weights, np.zeros(pad, dtype=np.float32)])

    return centers.astype(np.float32), weights.astype(np.float32)


# ─── palette_distance ─────────────────────────────────────────────────────────

def palette_distance(
    palette_a: ColorPalette,
    palette_b: ColorPalette,
) -> float:
    """Расстояние между двумя палитрами по взвешенным центроидам.

    Вычисляет взвешенную L2-разность между доминантными цветами двух
    палитр одинакового размера. Оба объекта должны иметь одинаковое
    число цветов.

    Аргументы:
        palette_a: Первая палитра.
        palette_b: Вторая палитра.

    Возвращает:
        Неотрицательное вещественное расстояние.

    Исключения:
        ValueError: Если n_colors у палитр различаются.
    """
    if palette_a.n_colors != palette_b.n_colors:
        raise ValueError(
            f"n_colors палитр должны совпадать: "
            f"{palette_a.n_colors} != {palette_b.n_colors}"
        )
    diff = palette_a.colors - palette_b.colors          # (n_colors, C)
    sq_dist = np.sum(diff ** 2, axis=1)                 # (n_colors,)
    w_avg = (palette_a.weights + palette_b.weights) / 2.0
    weighted = float(np.sum(w_avg * sq_dist))
    return float(np.sqrt(max(weighted, 0.0)))


# ─── compute_palette ──────────────────────────────────────────────────────────

def compute_palette(
    img:         np.ndarray,
    fragment_id: int = 0,
    cfg:         Optional[ColorPaletteConfig] = None,
) -> ColorPalette:
    """Вычислить палитру доминантных цветов изображения.

    Аргументы:
        img:         Изображение (2-D или 3-D, uint8).
        fragment_id: Идентификатор фрагмента (>= 0).
        cfg:         Конфигурация (None → ColorPaletteConfig()).

    Возвращает:
        :class:`ColorPalette` с центроидами и весами кластеров.
    """
    if cfg is None:
        cfg = ColorPaletteConfig()

    colors, weights = extract_dominant_colors(
        img,
        n_colors=cfg.n_colors,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        seed=cfg.seed,
    )

    h = img.shape[0] if img.ndim >= 1 else 0
    w = img.shape[1] if img.ndim >= 2 else 0

    return ColorPalette(
        fragment_id=fragment_id,
        colors=colors,
        weights=weights,
        n_colors=cfg.n_colors,
        params={
            "n_pixels": h * w,
            "img_shape": (h, w) + (img.shape[2:] if img.ndim == 3 else ()),
        },
    )


# ─── batch_compute_palettes ───────────────────────────────────────────────────

def batch_compute_palettes(
    images: List[np.ndarray],
    cfg:    Optional[ColorPaletteConfig] = None,
) -> List[ColorPalette]:
    """Пакетное вычисление палитр для списка изображений.

    Аргументы:
        images: Список изображений (2-D или 3-D, uint8).
        cfg:    Конфигурация (None → ColorPaletteConfig()).

    Возвращает:
        Список :class:`ColorPalette`, fragment_id = индекс в списке.
    """
    if cfg is None:
        cfg = ColorPaletteConfig()
    return [compute_palette(img, fragment_id=i, cfg=cfg)
            for i, img in enumerate(images)]


# ─── rank_by_palette ──────────────────────────────────────────────────────────

def rank_by_palette(
    query:      ColorPalette,
    candidates: List[ColorPalette],
    indices:    Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Ранжировать кандидатов по сходству палитры с запросом.

    Сходство = 1 / (1 + palette_distance(query, candidate)).

    Аргументы:
        query:      Эталонная палитра.
        candidates: Список палитр кандидатов.
        indices:    Идентификаторы кандидатов (None → 0, 1, …).

    Возвращает:
        Список кортежей (idx, similarity), отсортированный по убыванию.

    Исключения:
        ValueError: Если len(indices) != len(candidates).
    """
    if indices is None:
        indices = list(range(len(candidates)))
    if len(indices) != len(candidates):
        raise ValueError(
            f"Длины indices ({len(indices)}) и candidates "
            f"({len(candidates)}) должны совпадать"
        )
    if not candidates:
        return []

    scored = []
    for idx, cand in zip(indices, candidates):
        try:
            dist = palette_distance(query, cand)
        except ValueError:
            dist = float("inf")
        similarity = 1.0 / (1.0 + dist)
        scored.append((int(idx), float(similarity)))

    return sorted(scored, key=lambda x: x[1], reverse=True)
