"""
Утилиты для работы с numpy-массивами изображений и патчей.

Предоставляет функции нормализации, выравнивания размеров, разбивки
на чанки и агрегации, которые часто нужны в пайплайне реконструкции.

Функции:
    normalize_array        — нормализация массива в заданный диапазон
    pad_to_shape           — дополнение массива до целевой формы
    crop_center            — вырезка центральной области
    stack_arrays           — стек массивов с выравниванием по максимальному размеру
    chunk_array            — разбивка на чанки фиксированного размера
    sliding_window         — скользящее окно по оси
    flatten_images         — преобразование списка изображений в матрицу строк
    unflatten_images       — обратное преобразование матрицы строк в изображения
    compute_pairwise_norms — попарные L2-расстояния между строками матрицы
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np


# ─── normalize_array ──────────────────────────────────────────────────────────

def normalize_array(
    arr:     np.ndarray,
    low:     float = 0.0,
    high:    float = 1.0,
    dtype:   Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Нормализует значения массива в диапазон [low, high].

    Если массив вырожден (min == max), все элементы устанавливаются в low.

    Args:
        arr:   Входной массив (любая форма).
        low:   Нижняя граница выходного диапазона.
        high:  Верхняя граница выходного диапазона.
        dtype: Тип данных результата. None → float64.

    Returns:
        Нормализованный массив той же формы, dtype float64 (или dtype).
    """
    out_dtype = dtype if dtype is not None else np.float64
    a = arr.astype(np.float64)
    mn, mx = a.min(), a.max()
    if abs(mx - mn) < 1e-12:
        result = np.full_like(a, fill_value=low, dtype=np.float64)
    else:
        result = (a - mn) / (mx - mn) * (high - low) + low
    return result.astype(out_dtype)


# ─── pad_to_shape ─────────────────────────────────────────────────────────────

def pad_to_shape(
    arr:     np.ndarray,
    shape:   Tuple[int, ...],
    value:   float = 0.0,
    align:   str   = "topleft",
) -> np.ndarray:
    """
    Дополняет массив нулями (или value) до целевой формы.

    Args:
        arr:   Входной массив. Форма должна быть ≤ shape по каждой оси.
        shape: Целевая форма.
        value: Значение заполнения.
        align: 'topleft' — массив помещается в левый верхний угол;
               'center'  — массив центрируется.

    Returns:
        Новый массив формы shape с исходными данными внутри.

    Raises:
        ValueError: Если shape короче, чем форма arr по какой-либо оси.
    """
    for i, (s, t) in enumerate(zip(arr.shape, shape)):
        if s > t:
            raise ValueError(
                f"Axis {i}: arr.shape[{i}]={s} > target shape[{i}]={t}.")

    result = np.full(shape, fill_value=value, dtype=arr.dtype)

    if align == "center":
        offsets = tuple((t - s) // 2
                        for s, t in zip(arr.shape, shape))
    else:  # topleft
        offsets = (0,) * len(shape)

    slices = tuple(slice(o, o + s)
                   for o, s in zip(offsets, arr.shape))
    result[slices] = arr
    return result


# ─── crop_center ──────────────────────────────────────────────────────────────

def crop_center(
    arr:  np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Вырезает центральную прямоугольную область из 2D/3D массива.

    Args:
        arr:  Массив формы (H, W) или (H, W, C).
        size: (crop_h, crop_w) — размер вырезаемой области.

    Returns:
        Массив формы (crop_h, crop_w) или (crop_h, crop_w, C).

    Raises:
        ValueError: Если size превышает размеры arr.
    """
    h, w = arr.shape[:2]
    ch, cw = size
    if ch > h or cw > w:
        raise ValueError(
            f"crop size ({ch}, {cw}) exceeds array size ({h}, {w}).")
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return arr[y0: y0 + ch, x0: x0 + cw]


# ─── stack_arrays ─────────────────────────────────────────────────────────────

def stack_arrays(
    arrays: List[np.ndarray],
    axis:   int   = 0,
    value:  float = 0.0,
) -> np.ndarray:
    """
    Стекает список 2D/3D массивов, выравнивая их до максимального размера.

    Массивы меньшего размера дополняются значением value (pad в правый нижний).

    Args:
        arrays: Список массивов (H, W) или (H, W, C). Одинаковый ndim.
        axis:   Ось стекирования (0 → новая ось, 1 → H, 2 → W).
        value:  Значение заполнения.

    Returns:
        Стек формы (N, max_H, max_W, ...) при axis=0.

    Raises:
        ValueError: Если список пуст.
    """
    if not arrays:
        raise ValueError("arrays must not be empty.")

    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)
    target = (max_h, max_w) + arrays[0].shape[2:]

    padded = [pad_to_shape(a, target, value=value) for a in arrays]
    return np.stack(padded, axis=axis)


# ─── chunk_array ──────────────────────────────────────────────────────────────

def chunk_array(
    arr:        np.ndarray,
    chunk_size: int,
    axis:       int = 0,
) -> List[np.ndarray]:
    """
    Разбивает массив на чанки фиксированного размера вдоль оси.

    Последний чанк может быть меньше chunk_size.

    Args:
        arr:        Входной массив.
        chunk_size: Размер одного чанка.
        axis:       Ось разбивки.

    Returns:
        Список массивов-чанков.

    Raises:
        ValueError: Если chunk_size < 1.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    n    = arr.shape[axis]
    idxs = list(range(0, n, chunk_size))
    return [np.take(arr, list(range(i, min(i + chunk_size, n))), axis=axis)
            for i in idxs]


# ─── sliding_window ───────────────────────────────────────────────────────────

def sliding_window(
    arr:    np.ndarray,
    size:   int,
    step:   int = 1,
    axis:   int = 0,
) -> Iterator[np.ndarray]:
    """
    Генератор скользящего окна вдоль оси.

    Args:
        arr:  Входной массив.
        size: Размер окна.
        step: Шаг сдвига.
        axis: Ось сдвига.

    Yields:
        Подмассивы размером size вдоль axis.

    Raises:
        ValueError: Если size < 1 или step < 1.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}.")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}.")
    n = arr.shape[axis]
    for start in range(0, n - size + 1, step):
        yield np.take(arr, list(range(start, start + size)), axis=axis)


# ─── flatten_images ───────────────────────────────────────────────────────────

def flatten_images(
    images: List[np.ndarray],
    dtype:  Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Преобразует список изображений в 2D матрицу (N, H*W*C).

    Все изображения должны иметь одинаковую форму.

    Args:
        images: Список изображений.
        dtype:  Тип данных результата. None → dtype первого изображения.

    Returns:
        np.ndarray форма (N, D), где D = H*W*C.

    Raises:
        ValueError: Если список пуст или изображения разных форм.
    """
    if not images:
        raise ValueError("images list must not be empty.")
    shape0 = images[0].shape
    for i, img in enumerate(images[1:], start=1):
        if img.shape != shape0:
            raise ValueError(
                f"Image {i} has shape {img.shape}, expected {shape0}.")
    out_dtype = dtype if dtype is not None else images[0].dtype
    return np.stack([img.ravel() for img in images]).astype(out_dtype)


# ─── unflatten_images ─────────────────────────────────────────────────────────

def unflatten_images(
    matrix:     np.ndarray,
    img_shape:  Tuple[int, ...],
) -> List[np.ndarray]:
    """
    Обратное преобразование матрицы строк (N, D) → список изображений.

    Args:
        matrix:    np.ndarray форма (N, D).
        img_shape: Целевая форма каждого изображения.

    Returns:
        Список из N изображений формы img_shape.

    Raises:
        ValueError: Если D не совпадает с произведением img_shape.
    """
    d = int(np.prod(img_shape))
    if matrix.shape[1] != d:
        raise ValueError(
            f"Matrix column count {matrix.shape[1]} != "
            f"product of img_shape {img_shape} = {d}.")
    return [matrix[i].reshape(img_shape) for i in range(matrix.shape[0])]


# ─── compute_pairwise_norms ───────────────────────────────────────────────────

def compute_pairwise_norms(
    matrix: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    """
    Вычисляет попарные расстояния между строками матрицы.

    Args:
        matrix: np.ndarray форма (N, D).
        metric: 'l2' — евклидово расстояние;
                'l1' — манхэттенское расстояние;
                'cosine' — косинусное расстояние (1 - cos).

    Returns:
        Матрица расстояний форма (N, N) float64.

    Raises:
        ValueError: Неизвестная метрика.
    """
    if metric not in ("l2", "l1", "cosine"):
        raise ValueError(
            f"Unknown metric {metric!r}. Choose 'l2', 'l1', or 'cosine'.")

    a = matrix.astype(np.float64)
    n = a.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)

    if metric == "l2":
        # ||a - b||² = ||a||² + ||b||² - 2*a·b
        sq = (a ** 2).sum(axis=1, keepdims=True)
        gram = a @ a.T
        sq_dist = sq + sq.T - 2.0 * gram
        # Числовые погрешности
        sq_dist = np.clip(sq_dist, 0.0, None)
        dist = np.sqrt(sq_dist)
    elif metric == "l1":
        for i in range(n):
            dist[i] = np.abs(a - a[i]).sum(axis=1)
    else:  # cosine
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        a_norm = a / norms
        dist = 1.0 - a_norm @ a_norm.T
        dist = np.clip(dist, 0.0, 2.0)

    return dist
