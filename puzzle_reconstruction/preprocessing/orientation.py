"""
Определение ориентации текста на фрагменте.
По строкам текста определяем, как фрагмент был ориентирован в исходном документе.
"""
import cv2
import numpy as np
from typing import Optional


def estimate_orientation(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Оценивает угол ориентации текстового содержимого фрагмента.

    Стратегия:
    1. Выделяем горизонтальные структуры (строки текста) через морфологию.
    2. Находим линии через преобразование Хафа.
    3. Возвращаем медианный угол.

    Returns:
        angle_rad: Угол поворота в радианах (0 = текст горизонтален).
    """
    gray = _to_gray(image)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Адаптивная бинаризация: выделяем тёмные пиксели (текст)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Морфологическое закрытие по горизонтали — «склеиваем» буквы в строку
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    text_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)

    # Ищем линии преобразованием Хафа
    lines = cv2.HoughLinesP(
        text_lines, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=30, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        # Fallback: PCA по бинарным пикселям
        return _pca_angle(binary)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        if abs(angle) < np.pi / 4:  # Только почти горизонтальные линии
            angles.append(angle)

    if not angles:
        return _pca_angle(binary)

    return float(np.median(angles))


def rotate_to_upright(image: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Поворачивает изображение так, чтобы текст стал горизонтальным.
    """
    angle_deg = np.degrees(angle_rad)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return rotated


# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _pca_angle(binary: np.ndarray) -> float:
    """PCA по белым пикселям — запасной метод определения ориентации."""
    pts = np.column_stack(np.where(binary > 0)).astype(np.float32)
    if len(pts) < 10:
        return 0.0
    mean, eigenvectors = cv2.PCACompute(pts, mean=np.array([]))
    angle = np.arctan2(eigenvectors[0, 0], eigenvectors[0, 1])
    return float(angle)
