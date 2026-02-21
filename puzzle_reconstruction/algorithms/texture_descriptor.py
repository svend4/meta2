"""Вычисление текстурных дескрипторов фрагментов.

Модуль предоставляет функции для описания текстуры патчей изображений:
LBP (Local Binary Pattern), GLCM-признаки (контраст, однородность,
энергия, корреляция), статистики Гора (mean/std по каналам),
а также пакетную обработку и нормализацию дескрипторов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── TextureDescriptor ────────────────────────────────────────────────────────

@dataclass
class TextureDescriptor:
    """Текстурный дескриптор изображения.

    Атрибуты:
        vector:      Вектор признаков (float32, 1-D).
        method:      Метод вычисления ('lbp', 'glcm', 'stats', 'combined').
        image_id:    Идентификатор изображения (>= 0).
        params:      Дополнительные параметры.
    """

    vector: np.ndarray
    method: str = "combined"
    image_id: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.ndim != 1:
            raise ValueError(
                f"vector должен быть 1-D, получено ndim={self.vector.ndim}"
            )
        if self.image_id < 0:
            raise ValueError(
                f"image_id должен быть >= 0, получено {self.image_id}"
            )

    def __len__(self) -> int:
        return len(self.vector)


# ─── _to_gray ────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Преобразовать изображение в оттенки серого (uint8)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3:
        # Взвешенная формула luma
        return (
            0.2989 * img[:, :, 0].astype(np.float64)
            + 0.5870 * img[:, :, 1].astype(np.float64)
            + 0.1140 * img[:, :, 2].astype(np.float64)
        ).clip(0, 255).astype(np.uint8)
    raise ValueError(f"Изображение должно быть 2-D или 3-D, получено ndim={img.ndim}")


# ─── compute_lbp ──────────────────────────────────────────────────────────────

def compute_lbp(img: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Вычислить LBP-гистограмму изображения (radius=1, P=8).

    Аргументы:
        img:    Изображение (uint8, 2-D или 3-D).
        n_bins: Количество бинов гистограммы (>= 2).

    Возвращает:
        Нормализованная гистограмма LBP (float32, shape=(n_bins,)).

    Исключения:
        ValueError: Если n_bins < 2 или img некорректен.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins должен быть >= 2, получено {n_bins}")
    gray = _to_gray(img)
    H, W = gray.shape

    # Соседи по часовой стрелке (radius=1)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0,  1),  (1,  1), (1,  0),
                 (1, -1),  (0, -1)]

    lbp_map = np.zeros((H - 2, W - 2), dtype=np.uint8)
    center = gray[1:-1, 1:-1].astype(np.int16)

    for bit, (dr, dc) in enumerate(neighbors):
        neighbor = gray[1 + dr : H - 1 + dr, 1 + dc : W - 1 + dc].astype(np.int16)
        lbp_map |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, 256))
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float32) / total
    return hist.astype(np.float32)


# ─── compute_glcm_features ────────────────────────────────────────────────────

def compute_glcm_features(
    img: np.ndarray, levels: int = 16, distance: int = 1
) -> np.ndarray:
    """Вычислить признаки GLCM: контраст, однородность, энергию, корреляцию.

    Аргументы:
        img:      Изображение (uint8, 2-D или 3-D).
        levels:   Количество уровней квантования (>= 2, <= 256).
        distance: Расстояние для GLCM (>= 1).

    Возвращает:
        Вектор из 4 признаков (float32): [contrast, homogeneity, energy, correlation].

    Исключения:
        ValueError: Если параметры некорректны.
    """
    if levels < 2 or levels > 256:
        raise ValueError(f"levels должен быть в [2, 256], получено {levels}")
    if distance < 1:
        raise ValueError(f"distance должен быть >= 1, получено {distance}")

    gray = _to_gray(img)
    # Квантуем до levels уровней
    q = (gray.astype(np.float64) * (levels - 1) / 255.0).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    # GLCM по направлению 0° (горизонталь)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    H, W = q.shape
    if W - distance > 0:
        rows_i = q[:, :W - distance].ravel()
        rows_j = q[:, distance:].ravel()
        for i, j in zip(rows_i, rows_j):
            glcm[i, j] += 1.0
        # Симметризуем
        glcm = (glcm + glcm.T)
    total = glcm.sum()
    if total > 0:
        glcm /= total

    # Признаки
    i_idx = np.arange(levels)[:, None]
    j_idx = np.arange(levels)[None, :]

    contrast = float(((i_idx - j_idx) ** 2 * glcm).sum())
    homogeneity = float((glcm / (1.0 + (i_idx - j_idx) ** 2)).sum())
    energy = float((glcm ** 2).sum())

    mu_i = (i_idx * glcm).sum()
    mu_j = (j_idx * glcm).sum()
    std_i = np.sqrt(((i_idx - mu_i) ** 2 * glcm).sum())
    std_j = np.sqrt(((j_idx - mu_j) ** 2 * glcm).sum())
    if std_i < 1e-10 or std_j < 1e-10:
        correlation = 0.0
    else:
        correlation = float(((i_idx - mu_i) * (j_idx - mu_j) * glcm).sum()
                            / (std_i * std_j))

    return np.array([contrast, homogeneity, energy, correlation], dtype=np.float32)


# ─── compute_stats_descriptor ─────────────────────────────────────────────────

def compute_stats_descriptor(img: np.ndarray) -> np.ndarray:
    """Вычислить статистический дескриптор (mean, std) по каналам.

    Аргументы:
        img: Изображение (uint8, 2-D или 3-D).

    Возвращает:
        Вектор (float32): [mean_c0, std_c0, ...] для каждого канала.

    Исключения:
        ValueError: Если img не 2-D или 3-D.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        channels = [img.astype(np.float32)]
    elif img.ndim == 3:
        channels = [img[:, :, c].astype(np.float32) for c in range(img.shape[2])]
    else:
        raise ValueError(f"img должен быть 2-D или 3-D, получено ndim={img.ndim}")

    result = []
    for ch in channels:
        result.append(float(ch.mean()))
        result.append(float(ch.std()))
    return np.array(result, dtype=np.float32)


# ─── compute_texture_descriptor ──────────────────────────────────────────────

def compute_texture_descriptor(
    img: np.ndarray,
    method: str = "combined",
    lbp_bins: int = 32,
    glcm_levels: int = 16,
    image_id: int = 0,
) -> TextureDescriptor:
    """Вычислить текстурный дескриптор изображения.

    Аргументы:
        img:         Изображение (uint8, 2-D или 3-D).
        method:      'lbp', 'glcm', 'stats' или 'combined'.
        lbp_bins:    Число бинов для LBP (>= 2).
        glcm_levels: Число уровней для GLCM (>= 2, <= 256).
        image_id:    Идентификатор изображения (>= 0).

    Возвращает:
        TextureDescriptor.

    Исключения:
        ValueError: Если method неизвестен.
    """
    valid = {"lbp", "glcm", "stats", "combined"}
    if method not in valid:
        raise ValueError(
            f"Неизвестный метод '{method}'. Допустимые: {sorted(valid)}"
        )

    parts = []
    if method in ("lbp", "combined"):
        parts.append(compute_lbp(img, n_bins=lbp_bins))
    if method in ("glcm", "combined"):
        parts.append(compute_glcm_features(img, levels=glcm_levels))
    if method in ("stats", "combined"):
        parts.append(compute_stats_descriptor(img))

    vector = np.concatenate(parts).astype(np.float32)
    return TextureDescriptor(vector=vector, method=method, image_id=image_id)


# ─── normalize_descriptor ────────────────────────────────────────────────────

def normalize_descriptor(desc: TextureDescriptor) -> TextureDescriptor:
    """L2-нормализация вектора дескриптора.

    Аргументы:
        desc: TextureDescriptor.

    Возвращает:
        Новый TextureDescriptor с L2-нормализованным вектором.
    """
    norm = np.linalg.norm(desc.vector)
    if norm < 1e-12:
        vec = desc.vector.copy()
    else:
        vec = (desc.vector / norm).astype(np.float32)
    return TextureDescriptor(vector=vec, method=desc.method,
                             image_id=desc.image_id, params=dict(desc.params))


# ─── descriptor_distance ─────────────────────────────────────────────────────

def descriptor_distance(a: TextureDescriptor, b: TextureDescriptor) -> float:
    """L2-расстояние между двумя дескрипторами.

    Аргументы:
        a: Первый TextureDescriptor.
        b: Второй TextureDescriptor.

    Возвращает:
        Евклидово расстояние (float >= 0).

    Исключения:
        ValueError: Если длины дескрипторов не совпадают.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Длины дескрипторов не совпадают: {len(a)} != {len(b)}"
        )
    return float(np.linalg.norm(a.vector.astype(np.float64)
                                - b.vector.astype(np.float64)))


# ─── batch_compute_descriptors ────────────────────────────────────────────────

def batch_compute_descriptors(
    images: List[np.ndarray],
    method: str = "combined",
    lbp_bins: int = 32,
    glcm_levels: int = 16,
) -> List[TextureDescriptor]:
    """Вычислить дескрипторы для списка изображений.

    Аргументы:
        images:      Список изображений (uint8).
        method:      Метод вычисления дескриптора.
        lbp_bins:    Число бинов LBP.
        glcm_levels: Число уровней GLCM.

    Возвращает:
        Список TextureDescriptor, по одному на каждое изображение.
    """
    return [
        compute_texture_descriptor(img, method=method,
                                   lbp_bins=lbp_bins, glcm_levels=glcm_levels,
                                   image_id=i)
        for i, img in enumerate(images)
    ]
