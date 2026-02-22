"""
Шумоподавление отсканированных фрагментов документа.

Реальные сканы содержат:
    - Цифровой шум датчика (пятна, зёрна)
    - JPEG-артефакты (блочность, ореолы)
    - Пятна и следы на бумаге
    - Муар от сетчатого растра

Методы (от быстрых к медленным):
    gaussian_denoise()   — Гауссово размытие (быстро, потеря деталей)
    median_denoise()     — Медианный фильтр (хорош для «соли и перца»)
    bilateral_denoise()  — Билатеральный фильтр (сохраняет края)
    nlmeans_denoise()    — Non-Local Means (лучшее качество, медленно)
    auto_denoise()       — Автовыбор метода по SNR изображения

Рекомендации:
    Для сканов высокого качества: bilateral_denoise()
    Для сильно зашумлённых: nlmeans_denoise()
    Для быстрой обработки: gaussian_denoise()
    По умолчанию в пайплайне: auto_denoise()
"""
from __future__ import annotations

import numpy as np
import cv2


# ─── Гауссово размытие ────────────────────────────────────────────────────

def gaussian_denoise(image: np.ndarray,
                      sigma: float = 1.0,
                      kernel_size: int = 0) -> np.ndarray:
    """
    Гауссово размытие для подавления высокочастотного шума.

    Простейший метод. Хорошо работает для мелкого равномерного шума,
    но размывает и полезные детали (текст, края).

    Args:
        image:       BGR или grayscale uint8.
        sigma:       Стандартное отклонение Гауссова ядра (пиксели).
        kernel_size: Размер ядра. 0 = автовычисление из sigma.

    Returns:
        Сглаженное изображение uint8.
    """
    if sigma <= 0:
        return image
    ksize = kernel_size if kernel_size > 0 else 0  # 0 → авто
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


# ─── Медианный фильтр ─────────────────────────────────────────────────────

def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Медианный фильтр — лучший выбор для шума «соль и перец».

    Полностью убирает случайные выбросы (чёрные/белые точки),
    сохраняя края лучше, чем Гауссово размытие.

    Args:
        image: BGR или grayscale uint8.
        ksize: Размер окна (нечётное ≥ 3). Больше → сильнее фильтрация.

    Returns:
        Отфильтрованное изображение uint8.
    """
    ksize = max(3, ksize | 1)  # Гарантируем нечётность ≥ 3
    return cv2.medianBlur(image, ksize)


# ─── Билатеральный фильтр ─────────────────────────────────────────────────

def bilateral_denoise(image: np.ndarray,
                       d: int = 9,
                       sigma_color: float = 75.0,
                       sigma_space: float = 75.0) -> np.ndarray:
    """
    Билатеральный фильтр — сглаживает шум, сохраняя края.

    Ключевое свойство: учитывает одновременно пространственную близость
    и близость по цвету. Пиксели за краем (другой цвет) не учитываются
    при вычислении среднего → контуры сохраняются.

    Идеален для сканов с однородными областями (листы бумаги).

    Args:
        image:       BGR uint8.
        d:           Диаметр пространственного окна (пиксели).
        sigma_color: Диапазон допустимых цветовых различий.
        sigma_space: Пространственная полуширина Гауссова ядра.

    Returns:
        Сглаженное изображение uint8 с сохранёнными краями.
    """
    if image.ndim == 2:
        # Grayscale: обрабатываем напрямую
        return cv2.bilateralFilter(image, d=d,
                                    sigmaColor=sigma_color,
                                    sigmaSpace=sigma_space)
    return cv2.bilateralFilter(image, d=d,
                                sigmaColor=sigma_color,
                                sigmaSpace=sigma_space)


# ─── Non-Local Means ──────────────────────────────────────────────────────

def nlmeans_denoise(image: np.ndarray,
                     h: float = 10.0,
                     h_color: float = 10.0,
                     template_size: int = 7,
                     search_size: int = 21) -> np.ndarray:
    """
    Non-Local Means (NLM) — метод высочайшего качества.

    Для каждого пикселя ищет похожие патчи во всём изображении
    (не только в локальном окне) и усредняет их. Сохраняет текстуры
    и тонкие детали лучше любого локального метода.

    Минус: медленно (~1–5 сек на изображение 1000×800).

    Args:
        image:         BGR uint8.
        h:             Параметр фильтрации (сила). 5–15 для типичных сканов.
        h_color:       Параметр для цветовых каналов.
        template_size: Размер патча сравнения (нечётное).
        search_size:   Размер поискового окна (нечётное).

    Returns:
        Отфильтрованное изображение uint8.
    """
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image, h=h,
            templateWindowSize=template_size,
            searchWindowSize=search_size,
        )
    return cv2.fastNlMeansDenoisingColored(
        image, h=h, hColor=h_color,
        templateWindowSize=template_size,
        searchWindowSize=search_size,
    )


# ─── Авто-выбор ───────────────────────────────────────────────────────────

def auto_denoise(image: np.ndarray,
                  aggressive: bool = False) -> np.ndarray:
    """
    Автоматически выбирает метод шумоподавления по оценке SNR.

    Оценка шума: стандартное отклонение разности image - GaussBlur(image).
    Высокий σ_noise → сильный шум → NLM или медиана.
    Низкий σ_noise → слабый шум → билатеральный или ничего.

    Args:
        image:      BGR uint8.
        aggressive: True → всегда NLM (лучшее качество, медленно).

    Returns:
        Обработанное изображение uint8.
    """
    if image is None or image.size == 0:
        return image

    if aggressive:
        return nlmeans_denoise(image)

    noise_level = _estimate_noise(image)

    if noise_level < 2.0:
        return image                           # Практически чистое
    elif noise_level < 8.0:
        return bilateral_denoise(image,        # Лёгкий шум
                                  d=7, sigma_color=50, sigma_space=50)
    elif noise_level < 20.0:
        return bilateral_denoise(image,        # Умеренный шум
                                  d=9, sigma_color=75, sigma_space=75)
    else:
        return nlmeans_denoise(image, h=15.0)  # Сильный шум → NLM


def _estimate_noise(image: np.ndarray) -> float:
    """
    Оценивает уровень шума методом Лапласиана.

    Высокочастотная составляющая Лапласиана содержит в основном шум.
    Дисперсия Лапласиана / 6.0 ≈ σ_noise^2 для изотропного гауссова шума.

    Returns:
        σ_noise — стандартное отклонение шума (0 = чистый сигнал).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap  = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.sqrt(np.abs(lap.var()) / 6.0))


# ─── Пакетная обработка ───────────────────────────────────────────────────

def denoise_batch(images: list[np.ndarray],
                   method: str = "auto",
                   **kwargs) -> list[np.ndarray]:
    """
    Применяет выбранный метод шумоподавления к каждому изображению.

    Args:
        images: Список BGR uint8 изображений.
        method: "gaussian" | "median" | "bilateral" | "nlmeans" | "auto".
        **kwargs: Дополнительные параметры метода.

    Returns:
        Список обработанных изображений.

    Raises:
        ValueError: При неизвестном методе.
    """
    dispatch = {
        "gaussian":  gaussian_denoise,
        "median":    median_denoise,
        "bilateral": bilateral_denoise,
        "nlmeans":   nlmeans_denoise,
        "auto":      auto_denoise,
    }
    if method not in dispatch:
        raise ValueError(f"Неизвестный метод: '{method}'. "
                          f"Доступны: {list(dispatch)}")

    fn = dispatch[method]
    return [fn(img, **kwargs) if img is not None else img for img in images]
