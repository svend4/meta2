"""
Специализированное детектирование краёв фрагментов документа.

Стандартный Canny хорошо работает на фотографиях, но для сканов документов
часто требуется адаптивная настройка порогов и пост-обработка.

Функции:
    detect_edges          — единая точка входа (выбор метода)
    adaptive_canny        — Canny с автоматическими порогами (σ-метод Otsu)
    sobel_edges           — Sobel градиент (амплитуда)
    laplacian_edges       — Laplacian of Gaussian (LoG)
    structured_forest_edges — заглушка (требует обученной модели)
    refine_edge_contour   — морфологическая пост-обработка края
    edge_density          — доля пикселей-краёв в регионе
    edge_orientation_hist — гистограмма направлений граней (HOG-lite)

Классы:
    EdgeDetectionResult   — результат детектирования (edge_map, method, stats)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ─── Тип метода ───────────────────────────────────────────────────────────────

EdgeMethod = Literal["canny", "adaptive_canny", "sobel", "laplacian", "auto"]


# ─── EdgeDetectionResult ──────────────────────────────────────────────────────

@dataclass
class EdgeDetectionResult:
    """
    Результат детектирования краёв.

    Attributes:
        edge_map:   Бинарная карта краёв (uint8, 0/255).
        method:     Использованный метод ('canny', 'sobel', …).
        params:     Параметры, переданные методу.
        density:    Доля пикселей-краёв ∈ [0, 1].
        n_contours: Число найденных связных компонент.
    """
    edge_map:   np.ndarray
    method:     str
    params:     Dict = field(default_factory=dict)
    density:    float = 0.0
    n_contours: int   = 0

    def __post_init__(self):
        if self.edge_map.size > 0:
            self.density = float((self.edge_map > 0).mean())
        if _CV2:
            contours, _ = cv2.findContours(self.edge_map, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
            self.n_contours = len(contours)

    def __repr__(self) -> str:
        h, w = self.edge_map.shape[:2]
        return (f"EdgeDetectionResult(method={self.method!r}, "
                f"size=({h},{w}), density={self.density:.3f}, "
                f"n_contours={self.n_contours})")


# ─── Главная точка входа ──────────────────────────────────────────────────────

def detect_edges(image:     np.ndarray,
                  method:   EdgeMethod = "adaptive_canny",
                  **kwargs) -> EdgeDetectionResult:
    """
    Детектирует края фрагмента документа.

    Args:
        image:  BGR или grayscale uint8 изображение.
        method: Метод детектирования:
                  'canny'          — классический Canny (нужны порог1, порог2)
                  'adaptive_canny' — автоматические пороги на основе Otsu (default)
                  'sobel'          — градиент Sobel (амплитуда)
                  'laplacian'      — Laplacian of Gaussian
                  'auto'           — выбирает метод по снимку
        **kwargs: Дополнительные параметры метода.

    Returns:
        EdgeDetectionResult с бинарной картой.
    """
    if not _CV2:
        h, w = image.shape[:2]
        return EdgeDetectionResult(
            edge_map=np.zeros((h, w), dtype=np.uint8),
            method=method,
        )

    gray = _to_gray(image)

    if method == "canny":
        t1 = kwargs.get("threshold1", 50)
        t2 = kwargs.get("threshold2", 150)
        em = cv2.Canny(gray, t1, t2)
        return EdgeDetectionResult(edge_map=em, method=method,
                                    params={"threshold1": t1, "threshold2": t2})

    if method in ("adaptive_canny", "auto"):
        return adaptive_canny(gray, **kwargs)

    if method == "sobel":
        return sobel_edges(gray, **kwargs)

    if method == "laplacian":
        return laplacian_edges(gray, **kwargs)

    raise ValueError(f"Неизвестный метод детектирования краёв: {method!r}")


# ─── Адаптивный Canny ─────────────────────────────────────────────────────────

def adaptive_canny(image:   np.ndarray,
                    sigma:   float = 0.33,
                    blur_k:  int   = 5,
                    l2grad:  bool  = False) -> EdgeDetectionResult:
    """
    Canny с порогами, вычисляемыми автоматически из гистограммы яркостей.

    Метод σ-порогов (Dollar & Zitnick, 2013):
        median = медиана пикселей
        low  = max(0, (1 - σ) × median)
        high = min(255, (1 + σ) × median)

    Args:
        image:  Grayscale или BGR uint8.
        sigma:  Чувствительность (0.0 → строгий, 1.0 → мягкий).
        blur_k: Размер ядра Gaussian blur для пред-обработки.
        l2grad: Использовать L2-норму градиента (точнее, медленнее).

    Returns:
        EdgeDetectionResult.
    """
    if not _CV2:
        h, w = image.shape[:2]
        return EdgeDetectionResult(np.zeros((h, w), dtype=np.uint8),
                                    method="adaptive_canny")

    gray = _to_gray(image)

    # Сглаживание для уменьшения шума
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Вычисление порогов
    median = float(np.median(gray))
    low    = max(0.0,   (1.0 - sigma) * median)
    high   = min(255.0, (1.0 + sigma) * median)

    edge_map = cv2.Canny(gray, low, high, L2gradient=l2grad)

    return EdgeDetectionResult(
        edge_map=edge_map,
        method="adaptive_canny",
        params={"sigma": sigma, "blur_k": blur_k,
                "threshold1": low, "threshold2": high},
    )


# ─── Sobel ────────────────────────────────────────────────────────────────────

def sobel_edges(image:      np.ndarray,
                 threshold:  float = 50.0,
                 ksize:      int   = 3,
                 blur_k:     int   = 3) -> EdgeDetectionResult:
    """
    Детектирование краёв через градиент Sobel.

    Args:
        image:     Grayscale или BGR uint8.
        threshold: Порог бинаризации амплитуды (0–255).
        ksize:     Размер ядра Sobel (1, 3, 5 или 7).
        blur_k:    Предварительное сглаживание (0 → без сглаживания).

    Returns:
        EdgeDetectionResult с бинарной картой.
    """
    if not _CV2:
        h, w = image.shape[:2]
        return EdgeDetectionResult(np.zeros((h, w), dtype=np.uint8),
                                    method="sobel")

    gray = _to_gray(image)
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx**2 + gy**2)

    # Нормализация и бинаризация
    mag_norm = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mag_norm, int(threshold), 255, cv2.THRESH_BINARY)

    return EdgeDetectionResult(
        edge_map=binary,
        method="sobel",
        params={"threshold": threshold, "ksize": ksize},
    )


# ─── Laplacian of Gaussian ────────────────────────────────────────────────────

def laplacian_edges(image:     np.ndarray,
                     sigma:     float = 1.4,
                     threshold: float = 15.0) -> EdgeDetectionResult:
    """
    Детектирование краёв через Laplacian of Gaussian (LoG).

    Нули пересечения LoG соответствуют границам объектов.

    Args:
        image:     Grayscale или BGR uint8.
        sigma:     Стандартное отклонение Gaussian (размытие).
        threshold: Минимальная амплитуда LoG для включения в маску.

    Returns:
        EdgeDetectionResult с бинарной картой.
    """
    if not _CV2:
        h, w = image.shape[:2]
        return EdgeDetectionResult(np.zeros((h, w), dtype=np.uint8),
                                    method="laplacian")

    gray = _to_gray(image)

    # Gaussian blur
    k = int(6 * sigma + 1) | 1  # Нечётное ядро, покрывает ≈3σ
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigma)

    # Laplacian
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap_abs = np.abs(lap)
    _, binary = cv2.threshold(
        np.clip(lap_abs, 0, 255).astype(np.uint8),
        int(threshold), 255, cv2.THRESH_BINARY,
    )

    return EdgeDetectionResult(
        edge_map=binary,
        method="laplacian",
        params={"sigma": sigma, "threshold": threshold},
    )


# ─── Пост-обработка ───────────────────────────────────────────────────────────

def refine_edge_contour(edge_map:    np.ndarray,
                         close_iter:  int = 2,
                         dilate_iter: int = 1,
                         min_area:    int = 50) -> np.ndarray:
    """
    Морфологическая пост-обработка карты краёв.

    Шаги:
      1. Закрытие (close) для соединения разрывов.
      2. Опциональное дилатирование для утолщения.
      3. Удаление маленьких связных компонент (< min_area).

    Args:
        edge_map:    Бинарная карта краёв (uint8, 0/255).
        close_iter:  Итерации морфологического закрытия.
        dilate_iter: Итерации дилатирования (0 → без дилатации).
        min_area:    Минимальная площадь связной компоненты.

    Returns:
        Уточнённая бинарная карта (uint8, 0/255).
    """
    if not _CV2:
        return edge_map.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Закрытие для соединения разрывов
    refined = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel,
                                iterations=close_iter)

    # Опциональное дилатирование
    if dilate_iter > 0:
        refined = cv2.dilate(refined, kernel, iterations=dilate_iter)

    # Удаление маленьких компонент
    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            refined, connectivity=8,
        )
        mask = np.zeros_like(refined)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == lbl] = 255
        refined = mask

    return refined


# ─── Аналитические функции ────────────────────────────────────────────────────

def edge_density(region: np.ndarray,
                  edge_map: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет долю пикселей-краёв в регионе.

    Args:
        region:   Бинарная карта или изображение (uint8).
                  Если edge_map=None, применяет adaptive_canny к region.
        edge_map: Готовая карта краёв (если уже вычислена).

    Returns:
        Плотность краёв ∈ [0, 1].
    """
    if edge_map is None:
        result = adaptive_canny(region)
        em = result.edge_map
    else:
        em = edge_map

    if em.size == 0:
        return 0.0
    return float((em > 0).mean())


def edge_orientation_hist(edge_map: np.ndarray,
                            n_bins:   int = 8,
                            normalize: bool = True) -> np.ndarray:
    """
    Гистограмма направлений краёв (HOG-подобная, упрощённая).

    Вычисляет градиенты Sobel и строит гистограмму их направлений
    (ориентации) в диапазоне [0, π) (беззнаковые направления).

    Args:
        edge_map:  Бинарная карта краёв (uint8) или серое изображение.
        n_bins:    Число бинов гистограммы (по умолчанию 8 × 22.5°).
        normalize: True → нормализовать до суммы 1.0.

    Returns:
        np.ndarray формы (n_bins,) — гистограмма направлений.
    """
    if not _CV2:
        return np.zeros(n_bins, dtype=np.float64)

    gray = edge_map if edge_map.ndim == 2 else _to_gray(edge_map)

    gx  = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(np.abs(gy), np.abs(gx))  # [0, π/2] → разворачиваем

    # Беззнаковая ориентация ∈ [0, π)
    ang_full = np.arctan2(gy, gx) % np.pi      # [0, π)

    # Гистограмма, взвешенная по амплитуде
    hist, _ = np.histogram(ang_full.ravel(), bins=n_bins,
                             range=(0, np.pi), weights=mag.ravel())

    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float64)


# ─── Внутренние утилиты ───────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Конвертирует BGR → grayscale; grayscale оставляет как есть."""
    if not _CV2:
        return image if image.ndim == 2 else image.mean(axis=2).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
