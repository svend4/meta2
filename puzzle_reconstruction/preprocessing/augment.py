"""
Аугментация данных для синтетического тестирования и обучения.

Цель: генерация вариаций сканов, имитирующих реальные условия захвата
      (разное освещение, небольшое дрожание сканера, пыль, JPEG-артефакты).

Функции:
    random_crop           — случайная вырезка с сохранением области интереса
    random_rotate         — небольшой случайный поворот
    add_gaussian_noise    — добавление гауссового шума
    add_salt_pepper       — добавление шума «соль и перец»
    brightness_jitter     — случайная коррекция яркости/гаммы
    simulate_scan_noise   — комплексная имитация дефектов сканирования
    jpeg_compress         — имитация JPEG-артефактов
    augment_batch         — аугментация списка изображений

Все функции работают с BGR uint8 изображениями OpenCV.
"""
from __future__ import annotations

import io
import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── random_crop ──────────────────────────────────────────────────────────────

def random_crop(image: np.ndarray,
                min_scale: float = 0.75,
                max_scale: float = 1.0,
                rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Случайная вырезка прямоугольника из изображения с последующим масштабированием.

    Полезно для имитации разных полей отсканированных фрагментов.

    Args:
        image:     BGR uint8 (H, W, 3) или grayscale (H, W).
        min_scale: Минимальная доля стороны от оригинала (0.5–1.0).
        max_scale: Максимальная доля стороны.
        rng:       Генератор случайных чисел. None → np.random.

    Returns:
        Вырезанное и масштабированное изображение того же размера (H, W, ...).
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w  = image.shape[:2]
    scale = rng.uniform(min_scale, max_scale)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    y0 = rng.randint(0, max(1, h - new_h + 1))
    x0 = rng.randint(0, max(1, w - new_w + 1))

    cropped = image[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─── random_rotate ────────────────────────────────────────────────────────────

def random_rotate(image: np.ndarray,
                  max_angle: float = 10.0,
                  expand: bool = False,
                  border_color: Tuple[int, int, int] = (255, 255, 255),
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Случайный поворот изображения на малый угол (имитация дрожания сканера).

    Args:
        image:        BGR uint8.
        max_angle:    Максимальный угол поворота (±, градусы).
        expand:       True → увеличить холст, чтобы сохранить углы.
        border_color: Цвет фона после поворота (BGR).
        rng:          Генератор случайных чисел.

    Returns:
        Повёрнутое изображение (тот же размер если expand=False).
    """
    if rng is None:
        rng = np.random.RandomState()

    angle = rng.uniform(-max_angle, max_angle)
    h, w  = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    if expand:
        cos_a = abs(math.cos(math.radians(angle)))
        sin_a = abs(math.sin(math.radians(angle)))
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        out_size = (new_w, new_h)
    else:
        out_size = (w, h)

    flags  = cv2.INTER_LINEAR
    border = cv2.BORDER_CONSTANT
    value  = border_color if image.ndim == 3 else border_color[0]

    return cv2.warpAffine(image, M, out_size,
                           flags=flags, borderMode=border,
                           borderValue=value)


# ─── add_gaussian_noise ───────────────────────────────────────────────────────

def add_gaussian_noise(image: np.ndarray,
                        sigma: float = 10.0,
                        rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Добавляет нормальный шум σ (имитация теплового шума сенсора).

    Args:
        image: BGR uint8.
        sigma: Стандартное отклонение шума (0–50 для типичных сканов).
        rng:   Генератор случайных чисел.

    Returns:
        BGR uint8 с добавленным шумом.
    """
    if rng is None:
        rng = np.random.RandomState()
    if sigma <= 0:
        return image

    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ─── add_salt_pepper ──────────────────────────────────────────────────────────

def add_salt_pepper(image: np.ndarray,
                     amount: float = 0.01,
                     salt_ratio: float = 0.5,
                     rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Добавляет шум «соль и перец» (имитация пыли, битых пикселей сканера).

    Args:
        image:       BGR uint8.
        amount:      Доля испорченных пикселей (0.0–1.0).
        salt_ratio:  Доля белых пикселей среди испорченных (0.5 → 50/50).
        rng:         Генератор случайных чисел.

    Returns:
        Изображение с шумом uint8.
    """
    if rng is None:
        rng = np.random.RandomState()

    out   = image.copy()
    h, w  = image.shape[:2]
    n     = int(h * w * amount)

    # Соль (белые пиксели)
    n_salt = int(n * salt_ratio)
    ys = rng.randint(0, h, n_salt)
    xs = rng.randint(0, w, n_salt)
    out[ys, xs] = 255

    # Перец (чёрные пиксели)
    n_pepper = n - n_salt
    yp = rng.randint(0, h, n_pepper)
    xp = rng.randint(0, w, n_pepper)
    out[yp, xp] = 0

    return out


# ─── brightness_jitter ────────────────────────────────────────────────────────

def brightness_jitter(image: np.ndarray,
                       factor_range: Tuple[float, float] = (0.7, 1.3),
                       gamma_range:  Tuple[float, float] = (0.8, 1.2),
                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Случайная коррекция яркости и гаммы (имитация разных условий освещения).

    Args:
        image:        BGR uint8.
        factor_range: Диапазон линейного множителя яркости.
        gamma_range:  Диапазон показателя гаммы (1.0 = без изменений).
        rng:          Генератор случайных чисел.

    Returns:
        Скорректированное изображение uint8.
    """
    if rng is None:
        rng = np.random.RandomState()

    factor = rng.uniform(*factor_range)
    gamma  = rng.uniform(*gamma_range)

    # Линейная яркость
    out = np.clip(image.astype(np.float32) * factor, 0, 255)

    # Гамма-коррекция через LUT
    lut = np.array([
        min(255, int(255 * ((i / 255.0) ** (1.0 / gamma))))
        for i in range(256)
    ], dtype=np.uint8)

    out = cv2.LUT(out.astype(np.uint8), lut)
    return out


# ─── jpeg_compress ────────────────────────────────────────────────────────────

def jpeg_compress(image: np.ndarray,
                   quality: int = 70) -> np.ndarray:
    """
    Имитирует JPEG-сжатие с потерями (артефакты блочности, «звон»).

    Args:
        image:   BGR uint8.
        quality: Качество JPEG 1–100 (ниже → сильнее артефакты).

    Returns:
        Декодированное изображение с артефактами uint8.
    """
    quality = int(np.clip(quality, 1, 100))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ─── simulate_scan_noise ──────────────────────────────────────────────────────

def simulate_scan_noise(image:         np.ndarray,
                         gaussian_sigma: float = 5.0,
                         sp_amount:      float = 0.002,
                         jpeg_quality:   int   = 85,
                         yellowing:      float = 0.05,
                         rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Комплексная имитация артефактов реального сканирования:
        1. Гауссовый шум сенсора.
        2. Одиночные выбросы (пыль на стекле).
        3. Лёгкое пожелтение (старение бумаги).
        4. JPEG-сжатие (типичная пересылка скана).

    Args:
        image:          BGR uint8.
        gaussian_sigma: Σ гауссова шума (0 = отключить).
        sp_amount:      Доля «соль-перец» пикселей (0 = отключить).
        jpeg_quality:   Качество JPEG (100 = без сжатия).
        yellowing:      Сила пожелтения, 0.0–0.3.
        rng:            Генератор случайных чисел.

    Returns:
        Аугментированное изображение uint8.
    """
    if rng is None:
        rng = np.random.RandomState()

    out = image.copy()

    # 1. Гауссовый шум
    if gaussian_sigma > 0:
        out = add_gaussian_noise(out, sigma=gaussian_sigma, rng=rng)

    # 2. Пыль (соль и перец, только «соль»)
    if sp_amount > 0:
        out = add_salt_pepper(out, amount=sp_amount, salt_ratio=0.7, rng=rng)

    # 3. Пожелтение: сдвиг B-канала вниз, R-канала вверх
    if yellowing > 0 and out.ndim == 3:
        out = out.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * (1.0 - yellowing), 0, 255)  # B↓
        out[:, :, 2] = np.clip(out[:, :, 2] * (1.0 + yellowing * 0.5), 0, 255)  # R↑
        out = out.astype(np.uint8)

    # 4. JPEG
    if jpeg_quality < 100:
        out = jpeg_compress(out, quality=jpeg_quality)

    return out


# ─── augment_batch ────────────────────────────────────────────────────────────

def augment_batch(images:     List[np.ndarray],
                   n_augments: int = 3,
                   rotate:     bool = True,
                   crop:       bool = True,
                   noise:      bool = True,
                   jitter:     bool = True,
                   jpeg:       bool = False,
                   max_angle:  float = 8.0,
                   noise_sigma: float = 8.0,
                   seed:       Optional[int] = None) -> List[np.ndarray]:
    """
    Генерирует аугментированные копии каждого изображения в списке.

    Args:
        images:      Список BGR uint8 изображений.
        n_augments:  Число аугментированных копий на каждое изображение.
        rotate:      Включить случайный поворот.
        crop:        Включить случайную вырезку.
        noise:       Включить гауссовый шум.
        jitter:      Включить коррекцию яркости.
        jpeg:        Включить JPEG-артефакты.
        max_angle:   Максимальный угол поворота (градусы).
        noise_sigma: Σ гауссова шума.
        seed:        Начальное значение ГСЧ (None → случайное).

    Returns:
        Список: оригинальные изображения + все аугментированные копии.
        Итого: len(images) * (1 + n_augments).
    """
    rng    = np.random.RandomState(seed)
    result = list(images)

    for img in images:
        for _ in range(n_augments):
            aug = img.copy()
            if rotate:
                aug = random_rotate(aug, max_angle=max_angle, rng=rng)
            if crop:
                aug = random_crop(aug, min_scale=0.80, rng=rng)
            if noise:
                aug = add_gaussian_noise(aug, sigma=noise_sigma, rng=rng)
            if jitter:
                aug = brightness_jitter(aug, rng=rng)
            if jpeg:
                q   = int(rng.randint(60, 95))
                aug = jpeg_compress(aug, quality=q)
            result.append(aug)

    return result
