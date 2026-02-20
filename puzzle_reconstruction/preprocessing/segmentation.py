"""
Сегментация фрагментов: выделение маски бумажного кусочка из фона.
"""
import cv2
import numpy as np


def segment_fragment(image: np.ndarray,
                     method: str = "otsu",
                     morph_kernel: int = 3) -> np.ndarray:
    """
    Возвращает бинарную маску фрагмента (255 = бумага, 0 = фон).

    Args:
        image:        BGR или grayscale изображение.
        method:       "otsu" | "adaptive" | "grabcut"
        morph_kernel: Размер ядра для морфологической очистки.

    Returns:
        mask: uint8 массив (H, W), 255 на пикселях фрагмента.
    """
    gray = _to_gray(image)

    if method == "otsu":
        mask = _otsu_mask(gray)
    elif method == "adaptive":
        mask = _adaptive_mask(gray)
    elif method == "grabcut":
        mask = _grabcut_mask(image, gray)
    else:
        raise ValueError(f"Неизвестный метод сегментации: {method!r}")

    mask = _morphological_clean(mask, morph_kernel)
    mask = _keep_largest_component(mask)
    return mask


# ---------------------------------------------------------------------------
# Приватные функции
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    """Бинаризация по методу Оцу — работает для белой бумаги на белом/сером фоне."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _adaptive_mask(gray: np.ndarray) -> np.ndarray:
    """Адаптивная бинаризация — для неравномерного освещения."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )


def _grabcut_mask(image: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """GrabCut — точнее, но медленнее. Использует прямоугольник ROI."""
    h, w = image.shape[:2]
    margin = max(5, min(h, w) // 20)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    gc_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0)
    return binary.astype(np.uint8)


def _morphological_clean(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Удаление мелких артефактов и заполнение дыр."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Закрытие: заполняем дыры внутри
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Открытие: убираем мелкий шум
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    return opened


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Оставляет только самую большую связную компоненту."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    # Компонента 0 — фон, ищем самую большую среди остальных
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)
