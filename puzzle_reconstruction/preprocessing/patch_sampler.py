"""Выборка патчей из изображений фрагментов пазла.

Модуль предоставляет несколько стратегий выборки прямоугольных патчей:
равномерная сетка, случайная, вдоль границ и скользящее окно с заданным шагом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


_VALID_MODES = {"grid", "random", "border", "stride"}


# ─── SampleConfig ─────────────────────────────────────────────────────────────

@dataclass
class SampleConfig:
    """Параметры выборки патчей.

    Атрибуты:
        patch_size: Размер патча (нечётное >= 3, или чётное >= 2 для совместимости).
                    В данном модуле принимается любое >= 2.
        n_patches:  Желаемое число патчей (>= 1; для grid/border/stride — максимум).
        mode:       'grid' | 'random' | 'border' | 'stride'.
        stride:     Шаг скользящего окна (>= 1).
        seed:       Зерно генератора для режима 'random' (>= 0).
    """

    patch_size: int = 32
    n_patches: int = 16
    mode: str = "grid"
    stride: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.patch_size < 2:
            raise ValueError(
                f"patch_size должен быть >= 2, получено {self.patch_size}"
            )
        if self.n_patches < 1:
            raise ValueError(
                f"n_patches должен быть >= 1, получено {self.n_patches}"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode должен быть одним из {_VALID_MODES}, "
                f"получено '{self.mode}'"
            )
        if self.stride < 1:
            raise ValueError(
                f"stride должен быть >= 1, получено {self.stride}"
            )
        if self.seed < 0:
            raise ValueError(
                f"seed должен быть >= 0, получено {self.seed}"
            )


# ─── PatchSample ──────────────────────────────────────────────────────────────

@dataclass
class PatchSample:
    """Описание одного выбранного патча.

    Атрибуты:
        idx: Порядковый номер патча (>= 0).
        x:   Левая граница (>= 0).
        y:   Верхняя граница (>= 0).
        w:   Ширина (>= 1).
        h:   Высота (>= 1).
    """

    idx: int
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.idx < 0:
            raise ValueError(f"idx должен быть >= 0, получено {self.idx}")
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.w < 1:
            raise ValueError(f"w должен быть >= 1, получено {self.w}")
        if self.h < 1:
            raise ValueError(f"h должен быть >= 1, получено {self.h}")

    @property
    def x2(self) -> int:
        """Правая граница (не включительно)."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Нижняя граница (не включительно)."""
        return self.y + self.h

    @property
    def area(self) -> int:
        """Площадь патча."""
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Центр патча (cx, cy)."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


# ─── SampleResult ─────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Результат выборки патчей из одного изображения.

    Атрибуты:
        samples:     Список PatchSample.
        image_shape: Форма исходного изображения (H, W) или (H, W, C).
        n_patches:   Фактическое число патчей (>= 0).
    """

    samples: List[PatchSample]
    image_shape: Tuple[int, ...]
    n_patches: int

    def __post_init__(self) -> None:
        if self.n_patches < 0:
            raise ValueError(
                f"n_patches должен быть >= 0, получено {self.n_patches}"
            )

    @property
    def coverage_ratio(self) -> float:
        """Отношение суммарной площади патчей к площади изображения (может быть > 1)."""
        h, w = self.image_shape[:2]
        img_area = h * w
        if img_area == 0:
            return 0.0
        total = sum(s.area for s in self.samples)
        return float(total) / float(img_area)


# ─── sample_grid_patches ──────────────────────────────────────────────────────

def sample_grid_patches(
    image_h: int,
    image_w: int,
    patch_size: int,
    max_patches: int = 64,
) -> List[PatchSample]:
    """Выбрать патчи по равномерной сетке.

    Аргументы:
        image_h, image_w: Размеры изображения (>= 1).
        patch_size:       Размер патча (>= 2).
        max_patches:      Максимальное число патчей (>= 1).

    Возвращает:
        Список PatchSample.

    Исключения:
        ValueError: Если размеры некорректны.
    """
    if image_h < 1 or image_w < 1:
        raise ValueError(
            f"image_h и image_w должны быть >= 1, "
            f"получено ({image_h}, {image_w})"
        )
    if patch_size < 2:
        raise ValueError(f"patch_size должен быть >= 2, получено {patch_size}")
    if max_patches < 1:
        raise ValueError(f"max_patches должен быть >= 1, получено {max_patches}")

    ps = min(patch_size, image_h, image_w)
    if ps < 1:
        return []

    samples: List[PatchSample] = []
    idx = 0
    y = 0
    while y + ps <= image_h and len(samples) < max_patches:
        x = 0
        while x + ps <= image_w and len(samples) < max_patches:
            samples.append(PatchSample(idx=idx, x=x, y=y, w=ps, h=ps))
            idx += 1
            x += ps
        y += ps
    return samples


# ─── sample_random_patches ────────────────────────────────────────────────────

def sample_random_patches(
    image_h: int,
    image_w: int,
    patch_size: int,
    n_patches: int,
    seed: int = 0,
) -> List[PatchSample]:
    """Выбрать патчи случайным образом.

    Аргументы:
        image_h, image_w: Размеры изображения (>= 1).
        patch_size:       Размер патча (>= 2).
        n_patches:        Желаемое число патчей (>= 1).
        seed:             Зерно (>= 0).

    Возвращает:
        Список PatchSample.

    Исключения:
        ValueError: Если размеры некорректны или seed < 0.
    """
    if image_h < 1 or image_w < 1:
        raise ValueError(
            f"image_h и image_w должны быть >= 1, "
            f"получено ({image_h}, {image_w})"
        )
    if patch_size < 2:
        raise ValueError(f"patch_size должен быть >= 2, получено {patch_size}")
    if n_patches < 1:
        raise ValueError(f"n_patches должен быть >= 1, получено {n_patches}")
    if seed < 0:
        raise ValueError(f"seed должен быть >= 0, получено {seed}")

    ps = min(patch_size, image_h, image_w)
    max_x = image_w - ps
    max_y = image_h - ps
    if max_x < 0 or max_y < 0:
        return []

    rng = np.random.default_rng(seed)
    xs = rng.integers(0, max_x + 1, n_patches) if max_x >= 0 else np.zeros(n_patches, int)
    ys = rng.integers(0, max_y + 1, n_patches) if max_y >= 0 else np.zeros(n_patches, int)
    return [
        PatchSample(idx=i, x=int(xs[i]), y=int(ys[i]), w=ps, h=ps)
        for i in range(n_patches)
    ]


# ─── sample_border_patches ────────────────────────────────────────────────────

def sample_border_patches(
    image_h: int,
    image_w: int,
    patch_size: int,
    max_patches: int = 64,
) -> List[PatchSample]:
    """Выбрать патчи вдоль границ изображения.

    Аргументы:
        image_h, image_w: Размеры изображения (>= 1).
        patch_size:       Размер патча (>= 2).
        max_patches:      Максимальное число (>= 1).

    Возвращает:
        Список PatchSample.
    """
    if image_h < 1 or image_w < 1:
        raise ValueError(
            f"image_h и image_w должны быть >= 1, "
            f"получено ({image_h}, {image_w})"
        )
    if patch_size < 2:
        raise ValueError(f"patch_size должен быть >= 2, получено {patch_size}")
    if max_patches < 1:
        raise ValueError(f"max_patches должен быть >= 1, получено {max_patches}")

    ps = min(patch_size, image_h, image_w)
    positions: List[Tuple[int, int]] = []

    # Top and bottom rows
    x = 0
    while x + ps <= image_w:
        positions.append((x, 0))                      # top
        if ps < image_h:
            positions.append((x, image_h - ps))       # bottom
        x += ps

    # Left and right columns (excluding corners already covered)
    y = ps
    while y + ps <= image_h - ps:
        positions.append((0, y))                       # left
        if ps < image_w:
            positions.append((image_w - ps, y))        # right
        y += ps

    # Deduplicate preserving order
    seen: set = set()
    unique: List[Tuple[int, int]] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    unique = unique[:max_patches]
    return [
        PatchSample(idx=i, x=p[0], y=p[1], w=ps, h=ps)
        for i, p in enumerate(unique)
    ]


# ─── sample_stride_patches ────────────────────────────────────────────────────

def sample_stride_patches(
    image_h: int,
    image_w: int,
    patch_size: int,
    stride: int = 8,
    max_patches: int = 256,
) -> List[PatchSample]:
    """Выбрать патчи скользящим окном.

    Аргументы:
        image_h, image_w: Размеры изображения (>= 1).
        patch_size:       Размер патча (>= 2).
        stride:           Шаг (>= 1).
        max_patches:      Максимум (>= 1).

    Возвращает:
        Список PatchSample.
    """
    if image_h < 1 or image_w < 1:
        raise ValueError(
            f"image_h и image_w должны быть >= 1, "
            f"получено ({image_h}, {image_w})"
        )
    if patch_size < 2:
        raise ValueError(f"patch_size должен быть >= 2, получено {patch_size}")
    if stride < 1:
        raise ValueError(f"stride должен быть >= 1, получено {stride}")
    if max_patches < 1:
        raise ValueError(f"max_patches должен быть >= 1, получено {max_patches}")

    ps = min(patch_size, image_h, image_w)
    samples: List[PatchSample] = []
    idx = 0
    y = 0
    while y + ps <= image_h and len(samples) < max_patches:
        x = 0
        while x + ps <= image_w and len(samples) < max_patches:
            samples.append(PatchSample(idx=idx, x=x, y=y, w=ps, h=ps))
            idx += 1
            x += stride
        y += stride
    return samples


# ─── sample_patches ───────────────────────────────────────────────────────────

def sample_patches(
    image: np.ndarray,
    cfg: Optional[SampleConfig] = None,
) -> SampleResult:
    """Выбрать патчи из изображения согласно конфигурации.

    Аргументы:
        image: Изображение (2D или 3D numpy array).
        cfg:   Параметры (None → SampleConfig()).

    Возвращает:
        SampleResult.

    Исключения:
        ValueError: Если image не 2D/3D.
    """
    if cfg is None:
        cfg = SampleConfig()

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(
            f"image должно быть 2D или 3D, получено ndim={img.ndim}"
        )

    h, w = img.shape[:2]

    if cfg.mode == "grid":
        samples = sample_grid_patches(h, w, cfg.patch_size, cfg.n_patches)
    elif cfg.mode == "random":
        samples = sample_random_patches(h, w, cfg.patch_size, cfg.n_patches, cfg.seed)
    elif cfg.mode == "border":
        samples = sample_border_patches(h, w, cfg.patch_size, cfg.n_patches)
    else:  # stride
        samples = sample_stride_patches(h, w, cfg.patch_size, cfg.stride, cfg.n_patches)

    return SampleResult(
        samples=samples,
        image_shape=tuple(img.shape),
        n_patches=len(samples),
    )


# ─── extract_patch_images ─────────────────────────────────────────────────────

def extract_patch_images(
    image: np.ndarray,
    result: SampleResult,
) -> List[np.ndarray]:
    """Извлечь фактические патчи из изображения.

    Аргументы:
        image:  Изображение-источник.
        result: SampleResult (описания патчей).

    Возвращает:
        Список numpy arrays (по одному на патч).
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    patches: List[np.ndarray] = []
    for s in result.samples:
        y1, y2 = max(0, s.y), min(h, s.y2)
        x1, x2 = max(0, s.x), min(w, s.x2)
        patches.append(img[y1:y2, x1:x2].copy())
    return patches


# ─── batch_sample_patches ────────────────────────────────────────────────────

def batch_sample_patches(
    images: List[np.ndarray],
    cfg: Optional[SampleConfig] = None,
) -> List[SampleResult]:
    """Выбрать патчи для списка изображений.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры.

    Возвращает:
        Список SampleResult.
    """
    return [sample_patches(img, cfg) for img in images]
