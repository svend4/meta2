"""Повышение резкости краёв фрагментов документа перед сравнением.

Восстанавливает чёткость краёв, размытых при сканировании или сжатии,
что улучшает точность последующего сопоставления фрагментов.

Методы:
    unsharp_mask        — нерезкое маскирование (классика)
    laplacian_sharpen   — добавление лапласиана к исходному изображению
    high_pass_sharpen   — усиление высокочастотных деталей

Публичный API:
    SharpenerConfig     — параметры повышения резкости
    SharpenerResult     — результат обработки изображения
    unsharp_mask        — нерезкое маскирование
    laplacian_sharpen   — лапласиановое заострение
    high_pass_sharpen   — усиление высоких частот
    sharpen_image       — единый вход с выбором метода
    sharpen_edges       — выборочное заострение по маске краёв
    batch_sharpen       — пакетная обработка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ─── SharpenerConfig ──────────────────────────────────────────────────────────

@dataclass
class SharpenerConfig:
    """Параметры повышения резкости изображения.

    Атрибуты:
        method:   Метод заострения: 'unsharp', 'laplacian' или 'high_pass'.
        strength: Сила заострения (>= 0; 0 → без обработки, возврат исходника).
        sigma:    СКО гауссова размытия для unsharp/high_pass (> 0).
        ksize:    Размер ядра для laplacian (1, 3 или 5).
    """

    method: str = "unsharp"
    strength: float = 1.0
    sigma: float = 1.0
    ksize: int = 3

    def __post_init__(self) -> None:
        valid_methods = {"unsharp", "laplacian", "high_pass"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method должен быть одним из {sorted(valid_methods)}, "
                f"получено {self.method!r}"
            )
        if self.strength < 0.0:
            raise ValueError(
                f"strength должен быть >= 0, получено {self.strength}"
            )
        if self.sigma <= 0.0:
            raise ValueError(
                f"sigma должен быть > 0, получено {self.sigma}"
            )
        if self.ksize not in (1, 3, 5):
            raise ValueError(
                f"ksize должен быть 1, 3 или 5, получено {self.ksize}"
            )


# ─── SharpenerResult ──────────────────────────────────────────────────────────

@dataclass
class SharpenerResult:
    """Результат повышения резкости изображения.

    Атрибуты:
        image:      Обработанное изображение uint8.
        method:     Использованный метод.
        strength:   Применённая сила заострения.
        params:     Дополнительные параметры обработки.
    """

    image: np.ndarray
    method: str
    strength: float
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(
                f"strength должен быть >= 0, получено {self.strength}"
            )

    @property
    def shape(self):
        """Форма изображения (H, W) или (H, W, C)."""
        return self.image.shape

    @property
    def dtype(self):
        """Тип данных изображения."""
        return self.image.dtype

    @property
    def is_sharpened(self) -> bool:
        """True если strength > 0 (обработка применена)."""
        return self.strength > 0.0


# ─── unsharp_mask ─────────────────────────────────────────────────────────────

def unsharp_mask(
    img: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Нерезкое маскирование: result = img + strength × (img − blur(img)).

    Аргументы:
        img:      Изображение uint8 (2D или 3D BGR).
        sigma:    СКО гауссова размытия (> 0).
        strength: Сила заострения (>= 0).

    Возвращает:
        Изображение uint8 с повышенной резкостью.

    Исключения:
        ValueError: Если sigma <= 0 или strength < 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma должен быть > 0, получено {sigma}")
    if strength < 0.0:
        raise ValueError(f"strength должен быть >= 0, получено {strength}")

    if strength == 0.0:
        return img.copy()

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    img_f = img.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    result = img_f + strength * (img_f - blur_f)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── laplacian_sharpen ────────────────────────────────────────────────────────

def laplacian_sharpen(
    img: np.ndarray,
    ksize: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Заострение через лапласиан: result = img − alpha × Laplacian(img).

    Аргументы:
        img:   Изображение uint8 (2D или 3D BGR).
        ksize: Размер ядра лапласиана (1, 3 или 5).
        alpha: Сила заострения (>= 0).

    Возвращает:
        Изображение uint8 с повышенной резкостью.

    Исключения:
        ValueError: Если ksize не в {1, 3, 5} или alpha < 0.
    """
    if ksize not in (1, 3, 5):
        raise ValueError(f"ksize должен быть 1, 3 или 5, получено {ksize}")
    if alpha < 0.0:
        raise ValueError(f"alpha должен быть >= 0, получено {alpha}")

    if alpha == 0.0:
        return img.copy()

    img_f = img.astype(np.float32)

    if img.ndim == 3:
        lap = np.zeros_like(img_f)
        for c in range(img.shape[2]):
            lap[:, :, c] = cv2.Laplacian(img_f[:, :, c], cv2.CV_32F, ksize=ksize)
    else:
        lap = cv2.Laplacian(img_f, cv2.CV_32F, ksize=ksize)

    result = img_f - alpha * lap
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── high_pass_sharpen ────────────────────────────────────────────────────────

def high_pass_sharpen(
    img: np.ndarray,
    sigma: float = 2.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Усиление высоких частот: result = img + strength × HighPass(img).

    Высокочастотная составляющая: img − GaussBlur(img, sigma2),
    где sigma2 = sigma × 2.

    Аргументы:
        img:      Изображение uint8 (2D или 3D BGR).
        sigma:    Базовый σ гауссова размытия (> 0).
        strength: Сила усиления (>= 0).

    Возвращает:
        Изображение uint8 с повышенной резкостью.

    Исключения:
        ValueError: Если sigma <= 0 или strength < 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma должен быть > 0, получено {sigma}")
    if strength < 0.0:
        raise ValueError(f"strength должен быть >= 0, получено {strength}")

    if strength == 0.0:
        return img.copy()

    low = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma * 2, sigmaY=sigma * 2)
    img_f = img.astype(np.float32)
    low_f = low.astype(np.float32)
    high = img_f - low_f
    result = img_f + strength * high
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── sharpen_image ────────────────────────────────────────────────────────────

def sharpen_image(
    img: np.ndarray,
    cfg: Optional[SharpenerConfig] = None,
) -> SharpenerResult:
    """Повысить резкость изображения по конфигурации.

    Аргументы:
        img: Изображение uint8 (2D или 3D BGR).
        cfg: Параметры обработки (None → SharpenerConfig()).

    Возвращает:
        SharpenerResult с обработанным изображением.
    """
    if cfg is None:
        cfg = SharpenerConfig()

    if cfg.strength == 0.0:
        return SharpenerResult(
            image=img.copy(),
            method=cfg.method,
            strength=0.0,
            params={"sigma": cfg.sigma, "ksize": cfg.ksize},
        )

    if cfg.method == "unsharp":
        out = unsharp_mask(img, sigma=cfg.sigma, strength=cfg.strength)
    elif cfg.method == "laplacian":
        out = laplacian_sharpen(img, ksize=cfg.ksize, alpha=cfg.strength)
    else:  # high_pass
        out = high_pass_sharpen(img, sigma=cfg.sigma, strength=cfg.strength)

    return SharpenerResult(
        image=out,
        method=cfg.method,
        strength=cfg.strength,
        params={"sigma": cfg.sigma, "ksize": cfg.ksize},
    )


# ─── sharpen_edges ────────────────────────────────────────────────────────────

def sharpen_edges(
    img: np.ndarray,
    edge_mask: Optional[np.ndarray] = None,
    strength: float = 1.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Выборочное повышение резкости по маске краёв.

    Применяет нерезкое маскирование только в области краёв (edge_mask > 0).
    Вне маски изображение остаётся неизменным.

    Аргументы:
        img:       Изображение uint8 (2D или 3D).
        edge_mask: Бинарная маска (H, W) uint8; None → авто-Canny.
        strength:  Сила заострения (>= 0).
        sigma:     СКО гауссова ядра (> 0).

    Возвращает:
        Изображение uint8 с выборочно повышенной резкостью.

    Исключения:
        ValueError: Если sigma <= 0 или strength < 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma должен быть > 0, получено {sigma}")
    if strength < 0.0:
        raise ValueError(f"strength должен быть >= 0, получено {strength}")

    if edge_mask is None:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_mask = cv2.Canny(gray, threshold1=50, threshold2=150)

    sharpened = unsharp_mask(img, sigma=sigma, strength=strength)
    mask_bool = edge_mask.astype(bool)

    result = img.copy()
    if img.ndim == 2:
        result[mask_bool] = sharpened[mask_bool]
    else:
        result[mask_bool] = sharpened[mask_bool]

    return result


# ─── batch_sharpen ────────────────────────────────────────────────────────────

def batch_sharpen(
    images: List[np.ndarray],
    cfg: Optional[SharpenerConfig] = None,
) -> List[np.ndarray]:
    """Пакетная обработка: применить sharpen_image ко всем изображениям.

    Аргументы:
        images: Список изображений uint8.
        cfg:    Параметры обработки (None → SharpenerConfig()).

    Возвращает:
        Список обработанных изображений той же длины.
    """
    if cfg is None:
        cfg = SharpenerConfig()
    return [sharpen_image(img, cfg).image for img in images]
