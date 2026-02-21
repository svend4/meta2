"""
Утилиты извлечения патчей (подизображений) из фрагментов.

Поддерживает несколько стратегий выборки: равномерная сетка,
скользящее окно, случайные позиции и погранопосы вдоль краёв.

Классы:
    Patch    — один патч с координатами и метаданными
    PatchSet — упорядоченный набор патчей одного источника

Функции:
    extract_grid_patches    — равномерная сетка без перекрытия
    extract_sliding_patches — скользящее окно с заданным шагом
    extract_random_patches  — случайные позиции
    extract_border_patches  — равномерно вдоль краёв (4 стороны)
    filter_patches          — фильтрация по яркости и энтропии
    batch_extract_patches   — пакетная обработка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Patch ────────────────────────────────────────────────────────────────────

@dataclass
class Patch:
    """
    Один извлечённый патч.

    Attributes:
        image:     Пиксели патча (uint8, тот же формат каналов, что и источник).
        x:         Левая граница патча в исходном изображении.
        y:         Верхняя граница патча.
        w:         Ширина патча (пикс).
        h:         Высота патча (пикс).
        source_id: Идентификатор исходного изображения.
        meta:      Дополнительные поля (side, index и т.п.).
    """
    image:     np.ndarray
    x:         int
    y:         int
    w:         int
    h:         int
    source_id: int = 0
    meta:      Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Patch(xy=({self.x},{self.y}), wh=({self.w},{self.h}), "
                f"src={self.source_id})")


# ─── PatchSet ─────────────────────────────────────────────────────────────────

@dataclass
class PatchSet:
    """
    Набор патчей одного изображения.

    Attributes:
        patches:     Список Patch.
        source_id:   Идентификатор источника.
        image_shape: (h, w) исходного изображения.
        method:      Стратегия выборки.
    """
    patches:     List[Patch]
    source_id:   int
    image_shape: Tuple[int, int]
    method:      str

    @property
    def n_patches(self) -> int:
        return len(self.patches)

    def __repr__(self) -> str:
        return (f"PatchSet(n={self.n_patches}, "
                f"method={self.method!r}, "
                f"src={self.source_id}, "
                f"shape={self.image_shape})")


# ─── extract_grid_patches ────────────────────────────────────────────────────

def extract_grid_patches(img:        np.ndarray,
                          patch_size: int = 32,
                          stride:     Optional[int] = None,
                          source_id:  int = 0) -> PatchSet:
    """
    Извлекает патчи по равномерной сетке.

    По умолчанию stride = patch_size (без перекрытия).
    Патчи, выходящие за пределы изображения, пропускаются.

    Args:
        img:        BGR или grayscale изображение.
        patch_size: Размер патча (квадрат, пикс).
        stride:     Шаг сетки (None → patch_size).
        source_id:  ID источника.

    Returns:
        PatchSet с method='grid'.
    """
    h, w = img.shape[:2]
    s    = stride if stride is not None else patch_size
    s    = max(1, s)

    patches: List[Patch] = []
    y = 0
    while y + patch_size <= h:
        x = 0
        while x + patch_size <= w:
            crop = img[y:y + patch_size, x:x + patch_size]
            patches.append(Patch(
                image=crop.copy(), x=x, y=y,
                w=patch_size, h=patch_size,
                source_id=source_id,
            ))
            x += s
        y += s

    return PatchSet(patches=patches, source_id=source_id,
                    image_shape=(h, w), method="grid")


# ─── extract_sliding_patches ──────────────────────────────────────────────────

def extract_sliding_patches(img:        np.ndarray,
                              patch_size: int = 32,
                              stride:     int = 16,
                              source_id:  int = 0) -> PatchSet:
    """
    Извлекает патчи скользящим окном с перекрытием.

    Args:
        img:        BGR или grayscale изображение.
        patch_size: Размер патча (квадрат, пикс).
        stride:     Шаг скользящего окна.
        source_id:  ID источника.

    Returns:
        PatchSet с method='sliding'.
    """
    pset = extract_grid_patches(img, patch_size=patch_size,
                                  stride=stride, source_id=source_id)
    return PatchSet(patches=pset.patches, source_id=source_id,
                    image_shape=pset.image_shape, method="sliding")


# ─── extract_random_patches ───────────────────────────────────────────────────

def extract_random_patches(img:        np.ndarray,
                            patch_size: int = 32,
                            n_patches:  int = 50,
                            seed:       Optional[int] = None,
                            source_id:  int = 0) -> PatchSet:
    """
    Извлекает случайные патчи из изображения.

    Args:
        img:        BGR или grayscale изображение.
        patch_size: Размер патча (квадрат, пикс).
        n_patches:  Целевое число патчей.
        seed:       Зерно генератора случайных чисел (None → случайное).
        source_id:  ID источника.

    Returns:
        PatchSet с method='random'.
        Если изображение меньше patch_size × patch_size, возвращает пустой PatchSet.
    """
    h, w = img.shape[:2]
    patches: List[Patch] = []

    if h < patch_size or w < patch_size:
        return PatchSet(patches=[], source_id=source_id,
                        image_shape=(h, w), method="random")

    rng = np.random.default_rng(seed)
    ys  = rng.integers(0, h - patch_size + 1, size=n_patches)
    xs  = rng.integers(0, w - patch_size + 1, size=n_patches)

    for y, x in zip(ys, xs):
        crop = img[y:y + patch_size, x:x + patch_size]
        patches.append(Patch(
            image=crop.copy(), x=int(x), y=int(y),
            w=patch_size, h=patch_size,
            source_id=source_id,
        ))

    return PatchSet(patches=patches, source_id=source_id,
                    image_shape=(h, w), method="random")


# ─── extract_border_patches ───────────────────────────────────────────────────

def extract_border_patches(img:          np.ndarray,
                            patch_size:   int = 32,
                            n_per_side:   int = 8,
                            source_id:    int = 0) -> PatchSet:
    """
    Извлекает патчи вдоль четырёх краёв изображения.

    По n_per_side патчей равномерно вдоль каждого из 4 краёв.
    Метаданные патча содержат 'side' (0=верх,1=право,2=низ,3=лево).

    Args:
        img:        BGR или grayscale изображение.
        patch_size: Размер патча (квадрат, пикс).
        n_per_side: Число патчей на каждую сторону.
        source_id:  ID источника.

    Returns:
        PatchSet с method='border' (до 4 × n_per_side патчей).
    """
    h, w = img.shape[:2]
    p    = patch_size
    patches: List[Patch] = []

    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    # Верхняя сторона (side=0)
    for i in range(n_per_side):
        x = int(i * (w - p) / max(n_per_side - 1, 1)) if n_per_side > 1 else (w - p) // 2
        x = _clamp(x, 0, max(0, w - p))
        y = 0
        if y + p <= h and x + p <= w:
            patches.append(Patch(
                image=img[y:y + p, x:x + p].copy(),
                x=x, y=y, w=p, h=p,
                source_id=source_id, meta={"side": 0, "index": i},
            ))

    # Правая сторона (side=1)
    for i in range(n_per_side):
        y = int(i * (h - p) / max(n_per_side - 1, 1)) if n_per_side > 1 else (h - p) // 2
        y = _clamp(y, 0, max(0, h - p))
        x = max(0, w - p)
        if y + p <= h and x + p <= w:
            patches.append(Patch(
                image=img[y:y + p, x:x + p].copy(),
                x=x, y=y, w=p, h=p,
                source_id=source_id, meta={"side": 1, "index": i},
            ))

    # Нижняя сторона (side=2)
    for i in range(n_per_side):
        x = int(i * (w - p) / max(n_per_side - 1, 1)) if n_per_side > 1 else (w - p) // 2
        x = _clamp(x, 0, max(0, w - p))
        y = max(0, h - p)
        if y + p <= h and x + p <= w:
            patches.append(Patch(
                image=img[y:y + p, x:x + p].copy(),
                x=x, y=y, w=p, h=p,
                source_id=source_id, meta={"side": 2, "index": i},
            ))

    # Левая сторона (side=3)
    for i in range(n_per_side):
        y = int(i * (h - p) / max(n_per_side - 1, 1)) if n_per_side > 1 else (h - p) // 2
        y = _clamp(y, 0, max(0, h - p))
        x = 0
        if y + p <= h and x + p <= w:
            patches.append(Patch(
                image=img[y:y + p, x:x + p].copy(),
                x=x, y=y, w=p, h=p,
                source_id=source_id, meta={"side": 3, "index": i},
            ))

    return PatchSet(patches=patches, source_id=source_id,
                    image_shape=(h, w), method="border")


# ─── filter_patches ───────────────────────────────────────────────────────────

def _patch_entropy(patch: np.ndarray) -> float:
    """Shannon entropy от гистограммы яркости."""
    gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0].astype(np.float64)
    if len(hist) == 0:
        return 0.0
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def filter_patches(patches:        List[Patch],
                    min_brightness: float = 10.0,
                    max_brightness: float = 245.0,
                    min_entropy:    float = 0.5) -> List[Patch]:
    """
    Фильтрует патчи, удаляя пустые и перенасыщенные.

    Критерии отбора:
      - Средняя яркость ∈ [min_brightness, max_brightness].
      - Энтропия гистограммы ≥ min_entropy.

    Args:
        patches:        Список Patch.
        min_brightness: Нижний порог средней яркости.
        max_brightness: Верхний порог средней яркости.
        min_entropy:    Мин. Shannon-энтропия (биты).

    Returns:
        Отфильтрованный список Patch.
    """
    result: List[Patch] = []
    for p in patches:
        gray = (p.image if p.image.ndim == 2
                else cv2.cvtColor(p.image, cv2.COLOR_BGR2GRAY))
        brightness = float(gray.mean())
        if brightness < min_brightness or brightness > max_brightness:
            continue
        entropy = _patch_entropy(p.image)
        if entropy < min_entropy:
            continue
        result.append(p)
    return result


# ─── batch_extract_patches ────────────────────────────────────────────────────

_DISPATCH = {
    "grid":    extract_grid_patches,
    "sliding": extract_sliding_patches,
    "random":  extract_random_patches,
    "border":  extract_border_patches,
}


def batch_extract_patches(images:    List[np.ndarray],
                           method:   str = "grid",
                           **kwargs) -> List[PatchSet]:
    """
    Пакетное извлечение патчей из списка изображений.

    Args:
        images: Список BGR или grayscale изображений.
        method: 'grid' | 'sliding' | 'random' | 'border'.
        **kwargs: Параметры, передаваемые в выбранную функцию.

    Returns:
        Список PatchSet (по одному на изображение).

    Raises:
        ValueError: Если метод неизвестен.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Available: {sorted(_DISPATCH.keys())}"
        )
    fn = _DISPATCH[method]
    return [fn(img, source_id=i, **kwargs) for i, img in enumerate(images)]
