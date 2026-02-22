"""
Нормализация цвета сканированных фрагментов документа.

Проблема: разные фрагменты могут быть отсканированы при разном освещении,
с пожелтением бумаги, чернилами разной плотности и т.д.
Нормализация выравнивает цветовое пространство перед сравнением краёв.

Методы:
    normalize_color()       — автовыбор лучшего метода
    clahe_normalize()       — CLAHE (локальное выравнивание гистограммы)
    white_balance()         — белый баланс методом Gray World
    gamma_correction()      — гамма-коррекция
    normalize_brightness()  — нормализация средней яркости к целевому уровню

Рекомендованный порядок:
    1. white_balance()         → убрать цветовой сдвиг
    2. clahe_normalize()       → выровнять локальный контраст
    3. normalize_brightness()  → привести к единой средней яркости
"""
from __future__ import annotations

import numpy as np
import cv2


def normalize_color(image: np.ndarray,
                     target_brightness: float = 200.0,
                     clahe_clip: float = 2.0,
                     clahe_tile: int = 8) -> np.ndarray:
    """
    Применяет полный стек нормализации цвета.

    Шаги:
        1. Gray World white balance
        2. CLAHE на L-канале (LAB)
        3. Нормализация яркости к target_brightness

    Args:
        image:              BGR изображение, dtype=uint8.
        target_brightness:  Целевая средняя яркость (0-255).
        clahe_clip:         Clip limit для CLAHE (больше → резче, больше шума).
        clahe_tile:         Размер тайла CLAHE (пиксели).

    Returns:
        Нормализованное BGR изображение, dtype=uint8.
    """
    if image is None or image.size == 0:
        return image

    out = white_balance(image)
    out = clahe_normalize(out, clip_limit=clahe_clip, tile_size=clahe_tile)
    out = normalize_brightness(out, target=target_brightness)
    return out


# ─── CLAHE ───────────────────────────────────────────────────────────────

def clahe_normalize(image: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) в пространстве LAB.

    Применяется только к L-каналу, A и B не трогаются — это
    сохраняет цветовую информацию при выравнивании контраста.

    Args:
        image:       BGR uint8.
        clip_limit:  Порог ограничения контраста. 1.0 = выкл, 2-4 = норма.
        tile_size:   Размер локального окна (пиксели).

    Returns:
        BGR uint8 с выровненным контрастом.
    """
    if image.ndim == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                  tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)

    lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_size, tile_size))
    l_eq  = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─── White Balance ────────────────────────────────────────────────────────

def white_balance(image: np.ndarray) -> np.ndarray:
    """
    Коррекция белого баланса методом Gray World.

    Предположение: средний цвет изображения должен быть нейтральным серым.
    Масштабируем каждый канал так, чтобы среднее R = G = B = global_mean.

    Работает хорошо для документов: белая бумага часто имеет цветовой сдвиг
    при сканировании с лампой определённой температуры.

    Args:
        image: BGR uint8.

    Returns:
        Скорректированное BGR uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    img_f = image.astype(np.float32)
    b_mean = img_f[:, :, 0].mean()
    g_mean = img_f[:, :, 1].mean()
    r_mean = img_f[:, :, 2].mean()

    global_mean = (b_mean + g_mean + r_mean) / 3.0

    # Избегаем деления на ноль
    if b_mean < 1.0 or g_mean < 1.0 or r_mean < 1.0:
        return image

    scale_b = global_mean / b_mean
    scale_g = global_mean / g_mean
    scale_r = global_mean / r_mean

    out = img_f.copy()
    out[:, :, 0] *= scale_b
    out[:, :, 1] *= scale_g
    out[:, :, 2] *= scale_r

    return np.clip(out, 0, 255).astype(np.uint8)


# ─── Гамма-коррекция ──────────────────────────────────────────────────────

def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Применяет степенную гамма-коррекцию: out = (in / 255)^gamma * 255.

    gamma < 1.0 → осветление (типично для потемневших сканов)
    gamma > 1.0 → затемнение
    gamma = 1.0 → без изменений

    Args:
        image: BGR или grayscale uint8.
        gamma: Коэффициент гаммы.

    Returns:
        Скорректированное изображение uint8.
    """
    if abs(gamma - 1.0) < 1e-4:
        return image

    lut = np.array(
        [min(255, int((i / 255.0) ** gamma * 255 + 0.5)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


# ─── Нормализация яркости ─────────────────────────────────────────────────

def normalize_brightness(image: np.ndarray,
                          target: float = 200.0,
                          mask: np.ndarray = None) -> np.ndarray:
    """
    Масштабирует яркость так, чтобы средняя яркость = target.

    Используется для выравнивания между фрагментами одного документа.
    По маске считается только область фрагмента (не фон).

    Args:
        image:   BGR uint8.
        target:  Целевая средняя яркость пикселей (0-255).
        mask:    Бинарная маска области интереса (1 = включить в расчёт).

    Returns:
        Нормализованное BGR uint8.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    if mask is not None and mask.any():
        current = float(gray[mask > 0].mean())
    else:
        current = float(gray.mean())

    if current < 1.0:
        return image

    scale = target / current
    out   = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return out


# ─── Батч-нормализация ────────────────────────────────────────────────────

def batch_normalize(images: list[np.ndarray],
                     reference_idx: int = 0) -> list[np.ndarray]:
    """
    Нормализует яркость набора фрагментов относительно опорного изображения.

    Полезно когда все фрагменты должны иметь одинаковую яркость
    (например, при совместном рендере результата).

    Args:
        images:        Список BGR изображений.
        reference_idx: Индекс опорного изображения (его яркость сохраняется).

    Returns:
        Список нормализованных изображений.
    """
    if not images:
        return []

    ref_gray   = cv2.cvtColor(images[reference_idx], cv2.COLOR_BGR2GRAY)
    target_lum = float(ref_gray.mean())

    result = []
    for img in images:
        normalized = normalize_color(img, target_brightness=target_lum)
        result.append(normalized)
    return result
