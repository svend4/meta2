"""
Шумоподавление изображений фрагментов документов.

Модуль предоставляет несколько алгоритмов фильтрации и удаления шума,
а также автоматический выбор метода по оценённому уровню шума.

Классы:
    DenoiseResult — результат шумоподавления (изображение + метрики)

Функции:
    estimate_noise_level   — оценка уровня шума через дисперсию Лапласиана
    denoise_gaussian       — фильтр Гаусса
    denoise_median         — медианный фильтр (импульсный шум)
    denoise_nlm            — Non-Local Means (гауссовый шум)
    denoise_bilateral      — билатеральный фильтр (сохраняет края)
    denoise_morphological  — морфологическая очистка (открытие/закрытие)
    smart_denoise          — автовыбор метода по уровню шума
    batch_denoise          — обработка списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


# ─── DenoiseResult ────────────────────────────────────────────────────────────

@dataclass
class DenoiseResult:
    """
    Результат шумоподавления одного изображения.

    Attributes:
        denoised:     Обработанное изображение (dtype совпадает со входом).
        method:       Название применённого метода.
        noise_before: Оценка уровня шума до обработки.
        noise_after:  Оценка уровня шума после обработки.
        params:       Параметры метода.
    """
    denoised:     np.ndarray
    method:       str
    noise_before: float
    noise_after:  float
    params:       Dict = field(default_factory=dict)

    @property
    def noise_reduction_ratio(self) -> float:
        """Относительное снижение шума ∈ [0, 1]. 0 → нет улучшения."""
        if self.noise_before <= 0.0:
            return 0.0
        ratio = (self.noise_before - self.noise_after) / self.noise_before
        return float(np.clip(ratio, 0.0, 1.0))

    def __repr__(self) -> str:
        return (f"DenoiseResult(method={self.method!r}, "
                f"noise={self.noise_before:.2f}→{self.noise_after:.2f}, "
                f"reduction={self.noise_reduction_ratio:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


# ─── estimate_noise_level ─────────────────────────────────────────────────────

def estimate_noise_level(img: np.ndarray) -> float:
    """
    Оценивает стандартное отклонение шума через Лапласиан изображения.

    Метод: σ ≈ std(Laplacian(I)) / sqrt(sqrt(N)),
    где N — число пикселей (нормировка для независимости от размера).

    Args:
        img: BGR или grayscale изображение.

    Returns:
        Оценка σ (меньше = чище изображение).
    """
    gray = _to_gray(img).astype(np.float32)
    lap  = cv2.Laplacian(gray, cv2.CV_32F)
    n    = float(gray.size)
    sigma = float(np.std(lap)) / max(1.0, n ** 0.25)
    return max(0.0, sigma)


# ─── denoise_gaussian ─────────────────────────────────────────────────────────

def denoise_gaussian(img:    np.ndarray,
                      ksize: int   = 5,
                      sigma: float = 0.0) -> DenoiseResult:
    """
    Фильтр Гаусса.

    Args:
        img:   BGR или grayscale изображение.
        ksize: Размер ядра (нечётное ≥ 1).
        sigma: σ; 0 = авто по ksize.

    Returns:
        DenoiseResult.
    """
    ksize = _ensure_odd(ksize)
    noise_before = estimate_noise_level(img)
    denoised     = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    noise_after  = estimate_noise_level(denoised)
    return DenoiseResult(
        denoised=denoised,
        method="gaussian",
        noise_before=noise_before,
        noise_after=noise_after,
        params={"ksize": ksize, "sigma": sigma},
    )


# ─── denoise_median ───────────────────────────────────────────────────────────

def denoise_median(img:   np.ndarray,
                    ksize: int = 5) -> DenoiseResult:
    """
    Медианный фильтр — эффективно устраняет импульсный (соль-перец) шум.

    Args:
        img:   BGR или grayscale изображение.
        ksize: Размер ядра (нечётное, 3–21).

    Returns:
        DenoiseResult.
    """
    ksize = _ensure_odd(max(3, ksize))
    noise_before = estimate_noise_level(img)
    denoised     = cv2.medianBlur(img, ksize)
    noise_after  = estimate_noise_level(denoised)
    return DenoiseResult(
        denoised=denoised,
        method="median",
        noise_before=noise_before,
        noise_after=noise_after,
        params={"ksize": ksize},
    )


# ─── denoise_nlm ──────────────────────────────────────────────────────────────

def denoise_nlm(img:          np.ndarray,
                 h:            float = 10.0,
                 template_win: int   = 7,
                 search_win:   int   = 21) -> DenoiseResult:
    """
    Non-Local Means (NLM) деноиз — лучший результат при гауссовом шуме.

    Args:
        img:          BGR или grayscale изображение.
        h:            Параметр фильтрации (выше = сильнее; теряет детали).
        template_win: Размер патча (нечётное).
        search_win:   Размер окна поиска (нечётное).

    Returns:
        DenoiseResult.
    """
    template_win = _ensure_odd(template_win)
    search_win   = _ensure_odd(search_win)
    noise_before = estimate_noise_level(img)

    if img.ndim == 2:
        denoised = cv2.fastNlMeansDenoising(
            img, h=h,
            templateWindowSize=template_win,
            searchWindowSize=search_win,
        )
    else:
        denoised = cv2.fastNlMeansDenoisingColored(
            img, h=h, hColor=h,
            templateWindowSize=template_win,
            searchWindowSize=search_win,
        )

    noise_after = estimate_noise_level(denoised)
    return DenoiseResult(
        denoised=denoised,
        method="nlm",
        noise_before=noise_before,
        noise_after=noise_after,
        params={"h": h, "template_win": template_win,
                "search_win": search_win},
    )


# ─── denoise_bilateral ────────────────────────────────────────────────────────

def denoise_bilateral(img:         np.ndarray,
                       d:           int   = 9,
                       sigma_color: float = 75.0,
                       sigma_space: float = 75.0) -> DenoiseResult:
    """
    Билатеральный фильтр — сохраняет края, устраняет гауссовый шум.

    Args:
        img:         BGR или grayscale изображение.
        d:           Диаметр окрестности.
        sigma_color: σ по цвету (большое = сглаживаются далёкие цвета).
        sigma_space: σ по пространству.

    Returns:
        DenoiseResult.
    """
    noise_before = estimate_noise_level(img)

    if img.ndim == 2:
        src      = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        filtered = cv2.bilateralFilter(src, d, sigma_color, sigma_space)
        denoised = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    else:
        denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    noise_after = estimate_noise_level(denoised)
    return DenoiseResult(
        denoised=denoised,
        method="bilateral",
        noise_before=noise_before,
        noise_after=noise_after,
        params={"d": d, "sigma_color": sigma_color,
                "sigma_space": sigma_space},
    )


# ─── denoise_morphological ────────────────────────────────────────────────────

_MORPH_OPS = {
    "open":     cv2.MORPH_OPEN,
    "close":    cv2.MORPH_CLOSE,
    "tophat":   cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def denoise_morphological(img:   np.ndarray,
                            ksize: int = 3,
                            op:    str = "open") -> DenoiseResult:
    """
    Морфологическая очистка (opening/closing/tophat/blackhat).

    Opening  (эрозия → дилатация) удаляет мелкие светлые артефакты.
    Closing  (дилатация → эрозия) заполняет мелкие тёмные дыры.

    Args:
        img:   BGR или grayscale изображение.
        ksize: Размер структурного элемента (эллипс).
        op:    'open' | 'close' | 'tophat' | 'blackhat'.

    Returns:
        DenoiseResult.

    Raises:
        ValueError: Если op не из допустимого набора.
    """
    if op not in _MORPH_OPS:
        raise ValueError(
            f"Неизвестная операция: {op!r}. Допустимые: {list(_MORPH_OPS)}"
        )
    ksize  = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    noise_before = estimate_noise_level(img)
    denoised     = cv2.morphologyEx(img, _MORPH_OPS[op], kernel)
    noise_after  = estimate_noise_level(denoised)

    return DenoiseResult(
        denoised=denoised,
        method=f"morphological_{op}",
        noise_before=noise_before,
        noise_after=noise_after,
        params={"ksize": ksize, "op": op},
    )


# ─── smart_denoise ────────────────────────────────────────────────────────────

# Пороги для автовыбора метода
_THRESH_LOW    = 2.0
_THRESH_MEDIUM = 6.0
_THRESH_HIGH   = 12.0


def smart_denoise(img:         np.ndarray,
                   light_ksize: int   = 3,
                   strong_h:    float = 15.0,
                   bilateral_d: int   = 9) -> DenoiseResult:
    """
    Автоматически выбирает метод шумоподавления по уровню шума.

    Уровень шума:
        ≤ 2.0  → без обработки (копия)
        ≤ 6.0  → медианный фильтр
        ≤ 12.0 → билатеральный фильтр
        > 12.0 → Non-Local Means

    Args:
        img:         Входное изображение.
        light_ksize: Ядро для медианного фильтра.
        strong_h:    h для NLM при сильном шуме.
        bilateral_d: d для билатерального фильтра.

    Returns:
        DenoiseResult с выбранным методом.
    """
    noise = estimate_noise_level(img)

    if noise <= _THRESH_LOW:
        return DenoiseResult(
            denoised=img.copy(),
            method="none",
            noise_before=noise,
            noise_after=noise,
            params={"reason": "noise below threshold"},
        )
    elif noise <= _THRESH_MEDIUM:
        return denoise_median(img, ksize=light_ksize)
    elif noise <= _THRESH_HIGH:
        return denoise_bilateral(img, d=bilateral_d)
    else:
        return denoise_nlm(img, h=strong_h)


# ─── batch_denoise ────────────────────────────────────────────────────────────

_DISPATCH = {
    "smart":         smart_denoise,
    "gaussian":      denoise_gaussian,
    "median":        denoise_median,
    "nlm":           denoise_nlm,
    "bilateral":     denoise_bilateral,
    "morphological": denoise_morphological,
}


def batch_denoise(images: List[np.ndarray],
                   method: str = "smart",
                   **kwargs) -> List[DenoiseResult]:
    """
    Применяет шумоподавление к списку изображений.

    Args:
        images: Список изображений.
        method: 'smart' | 'gaussian' | 'median' | 'nlm' |
                'bilateral' | 'morphological'.
        **kwargs: Параметры для выбранного метода.

    Returns:
        Список DenoiseResult.

    Raises:
        ValueError: Если method не из допустимого набора.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Неизвестный метод: {method!r}. Допустимые: {list(_DISPATCH)}"
        )
    fn = _DISPATCH[method]
    return [fn(img, **kwargs) for img in images]
