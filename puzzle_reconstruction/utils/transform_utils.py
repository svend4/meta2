"""
Утилиты геометрических преобразований изображений.

Обёртки над OpenCV для вращения, отражения, масштабирования,
обрезки, составления аффинных преобразований и применения гомографий.
Используются повторно во всём пайплайне вместо дублирования кода.

Функции:
    rotate_image       — вращение вокруг центра (без изменения размера)
    flip_image         — горизонтальное/вертикальное/оба отражения
    scale_image        — масштабирование (sx, sy) с выбором интерполяции
    crop_region        — вырезание прямоугольной области (x,y,w,h)
    affine_from_params — матрица аффинного преобразования из параметров
    compose_affines    — последовательная композиция аффинных матриц
    apply_affine       — применение аффинной матрицы к изображению
    apply_homography   — применение гомографии (3×3) к изображению
    batch_rotate       — пакетное вращение списка изображений
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── rotate_image ─────────────────────────────────────────────────────────────

def rotate_image(img:    np.ndarray,
                  angle:  float,
                  center: Optional[Tuple[float, float]] = None,
                  fill:   int = 255) -> np.ndarray:
    """
    Поворачивает изображение вокруг заданного центра.

    Размер выходного изображения совпадает с входным; пиксели за
    границей заполняются значением fill.

    Args:
        img:    BGR или grayscale изображение (uint8).
        angle:  Угол поворота в градусах (против часовой стрелки).
        center: (cx, cy) центра вращения; None → центр изображения.
        fill:   Значение заполнения пустых пикселей (0..255).

    Returns:
        Повёрнутое изображение той же формы и dtype.
    """
    h, w = img.shape[:2]
    cx, cy = (float(w) / 2.0, float(h) / 2.0) if center is None else center
    M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=fill)


# ─── flip_image ───────────────────────────────────────────────────────────────

def flip_image(img:  np.ndarray,
                mode: int = 1) -> np.ndarray:
    """
    Отражает изображение.

    Args:
        img:  BGR или grayscale изображение.
        mode: 0 → по вертикали (flip around x-axis),
              1 → по горизонтали (flip around y-axis),
             -1 → оба направления.

    Returns:
        Отражённое изображение той же формы и dtype.
    """
    return cv2.flip(img, mode)


# ─── scale_image ──────────────────────────────────────────────────────────────

def scale_image(img:           np.ndarray,
                 sx:            float = 1.0,
                 sy:            Optional[float] = None,
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Масштабирует изображение по коэффициентам sx и sy.

    Args:
        img:           BGR или grayscale изображение.
        sx:            Коэффициент масштабирования по X.
        sy:            Коэффициент по Y; None → равен sx (пропорционально).
        interpolation: Метод интерполяции (cv2.INTER_*).

    Returns:
        Масштабированное изображение uint8.
    """
    if sy is None:
        sy = sx
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


# ─── crop_region ──────────────────────────────────────────────────────────────

def crop_region(img:    np.ndarray,
                 x:      int,
                 y:      int,
                 w:      int,
                 h:      int,
                 clamp:  bool = True) -> np.ndarray:
    """
    Вырезает прямоугольную область из изображения.

    Args:
        img:   BGR или grayscale изображение.
        x, y:  Верхний левый угол области.
        w, h:  Ширина и высота области.
        clamp: Зажать координаты в границах изображения (по умолчанию True).

    Returns:
        Вырезанный фрагмент (view или copy).

    Raises:
        ValueError: Если w ≤ 0 или h ≤ 0 после зажатия.
    """
    ih, iw = img.shape[:2]
    if clamp:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(iw, x + w)
        y1 = min(ih, y + h)
    else:
        x0, y0, x1, y1 = x, y, x + w, y + h

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Empty crop region after clamp: "
            f"x=[{x0},{x1}), y=[{y0},{y1})"
        )
    return img[y0:y1, x0:x1]


# ─── affine_from_params ───────────────────────────────────────────────────────

def affine_from_params(angle:  float = 0.0,
                        tx:     float = 0.0,
                        ty:     float = 0.0,
                        sx:     float = 1.0,
                        sy:     Optional[float] = None,
                        cx:     float = 0.0,
                        cy:     float = 0.0) -> np.ndarray:
    """
    Строит матрицу аффинного преобразования (2×3) из параметров.

    Порядок: масштабирование → поворот → перенос.
    Вращение выполняется вокруг точки (cx, cy).

    Args:
        angle: Угол поворота (градусы, против часовой стрелки).
        tx:    Перенос по X (пикс).
        ty:    Перенос по Y (пикс).
        sx:    Масштаб по X.
        sy:    Масштаб по Y (None → sx).
        cx:    X-координата центра вращения.
        cy:    Y-координата центра вращения.

    Returns:
        Матрица 2×3 типа float32.
    """
    if sy is None:
        sy = sx
    rad  = float(angle) * np.pi / 180.0
    cos_ = float(np.cos(rad))
    sin_ = float(np.sin(rad))

    # Масштабирование + вращение
    M = np.array([
        [sx * cos_, -sy * sin_, 0.0],
        [sx * sin_,  sy * cos_, 0.0],
    ], dtype=np.float32)

    # Поправка на центр вращения + перенос
    M[0, 2] = tx + cx - M[0, 0] * cx - M[0, 1] * cy
    M[1, 2] = ty + cy - M[1, 0] * cx - M[1, 1] * cy

    return M


# ─── compose_affines ──────────────────────────────────────────────────────────

def compose_affines(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Последовательно компонует список аффинных матриц (2×3).

    Применяется в порядке: matrices[0], затем matrices[1], и т.д.

    Args:
        matrices: Список матриц 2×3 (float32 или float64).

    Returns:
        Результирующая матрица 2×3 float32.

    Raises:
        ValueError: Если список пуст.
    """
    if not matrices:
        raise ValueError("Cannot compose empty list of affine matrices.")

    def _to_3x3(M: np.ndarray) -> np.ndarray:
        """2×3 → 3×3 (добавить строку [0,0,1])."""
        return np.vstack([M.astype(np.float64),
                          [[0.0, 0.0, 1.0]]])

    result = _to_3x3(matrices[0])
    for M in matrices[1:]:
        result = _to_3x3(M) @ result

    return result[:2, :].astype(np.float32)


# ─── apply_affine ─────────────────────────────────────────────────────────────

def apply_affine(img:    np.ndarray,
                  M:      np.ndarray,
                  size:   Optional[Tuple[int, int]] = None,
                  fill:   int = 255) -> np.ndarray:
    """
    Применяет аффинную матрицу 2×3 к изображению.

    Args:
        img:  BGR или grayscale изображение (uint8).
        M:    Матрица аффинного преобразования 2×3.
        size: (width, height) выходного изображения;
              None → размер входного изображения.
        fill: Заполнение граничных пикселей.

    Returns:
        Преобразованное изображение.
    """
    h, w = img.shape[:2]
    out_size = size if size is not None else (w, h)
    return cv2.warpAffine(img, M, out_size,
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=fill)


# ─── apply_homography ─────────────────────────────────────────────────────────

def apply_homography(img:  np.ndarray,
                      H:    np.ndarray,
                      size: Optional[Tuple[int, int]] = None,
                      fill: int = 255) -> np.ndarray:
    """
    Применяет гомографию (матрица 3×3) к изображению.

    Args:
        img:  BGR или grayscale изображение (uint8).
        H:    Матрица гомографии 3×3 (float32 или float64).
        size: (width, height) выходного изображения;
              None → размер входного.
        fill: Значение заполнения.

    Returns:
        Трансформированное изображение.
    """
    h, w = img.shape[:2]
    out_size = size if size is not None else (w, h)
    return cv2.warpPerspective(img, H.astype(np.float32), out_size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=fill)


# ─── batch_rotate ─────────────────────────────────────────────────────────────

def batch_rotate(images: List[np.ndarray],
                  angle:  float,
                  fill:   int = 255) -> List[np.ndarray]:
    """
    Поворачивает список изображений на один и тот же угол.

    Args:
        images: Список BGR или grayscale изображений.
        angle:  Угол поворота (градусы, против часовой стрелки).
        fill:   Заполнение граничных пикселей.

    Returns:
        Список повёрнутых изображений той же длины.
    """
    return [rotate_image(img, angle, fill=fill) for img in images]
