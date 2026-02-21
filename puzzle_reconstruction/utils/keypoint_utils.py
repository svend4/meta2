"""
Утилиты работы с ключевыми точками (keypoints) и дескрипторами.

Поддерживает детектирование через ORB (всегда доступен в OpenCV),
базовую фильтрацию, сопоставление дескрипторов и геометрическую верификацию.

Экспортирует:
    KeypointSet          — набор ключевых точек с дескрипторами
    detect_keypoints     — детектирование ORB / SIFT ключевых точек
    filter_by_response   — фильтрация по силе отклика
    filter_by_region     — фильтрация по пространственной маске
    describe_keypoints   — вычисление дескрипторов для заданных точек
    match_descriptors    — сопоставление двух наборов дескрипторов (BF + ratio)
    filter_matches_ransac — геометрическая верификация через RANSAC
    keypoints_to_array   — cv2.KeyPoint → ndarray (N, 2)
    array_to_keypoints   — ndarray (N, 2) → list[cv2.KeyPoint]
    compute_match_score  — сводная оценка качества сопоставления
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class KeypointSet:
    """Набор ключевых точек с дескрипторами.

    Attributes:
        keypoints:   Список ``cv2.KeyPoint``.
        descriptors: Матрица дескрипторов (N, D) float32 или uint8, либо ``None``.
        detector:    Имя детектора, которым получены точки.
        params:      Параметры детектирования.
    """
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    detector: str = "orb"
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keypoints)

    def __repr__(self) -> str:  # pragma: no cover
        n_desc = 0 if self.descriptors is None else len(self.descriptors)
        return (
            f"KeypointSet(n_kp={len(self.keypoints)}, "
            f"n_desc={n_desc}, detector={self.detector!r})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def detect_keypoints(
    img: np.ndarray,
    detector: str = "orb",
    max_keypoints: int = 500,
    **kwargs,
) -> KeypointSet:
    """Обнаружить ключевые точки и вычислить дескрипторы.

    Args:
        img:          Изображение uint8 (2D или BGR 3D).
        detector:     ``'orb'`` (default) или ``'sift'`` (требует opencv-contrib).
        max_keypoints: Максимальное число точек (> 0).
        **kwargs:     Дополнительные параметры детектора.

    Returns:
        :class:`KeypointSet`.

    Raises:
        ValueError: Если ``detector`` неизвестен или ``max_keypoints`` < 1.
    """
    if max_keypoints < 1:
        raise ValueError(f"max_keypoints must be >= 1, got {max_keypoints}")
    valid_detectors = {"orb", "sift"}
    if detector not in valid_detectors:
        raise ValueError(
            f"detector must be one of {sorted(valid_detectors)}, got {detector!r}"
        )

    gray = _to_gray(img)
    det = _build_detector(detector, max_keypoints, **kwargs)
    kps, descs = det.detectAndCompute(gray, None)

    if kps is None:
        kps = []
    # Limit to max_keypoints by response
    if len(kps) > max_keypoints:
        kps = sorted(kps, key=lambda k: k.response, reverse=True)[:max_keypoints]
        descs = descs[:max_keypoints] if descs is not None else None

    return KeypointSet(
        keypoints=list(kps),
        descriptors=descs,
        detector=detector,
        params={"max_keypoints": max_keypoints, **kwargs},
    )


def filter_by_response(
    kpset: KeypointSet,
    min_response: float = 0.0,
    top_k: int = 0,
) -> KeypointSet:
    """Фильтровать ключевые точки по силе отклика.

    Args:
        kpset:        Исходный набор.
        min_response: Минимальный отклик (включительно).
        top_k:        Оставить только top-k лучших (0 = без ограничений).

    Returns:
        Новый :class:`KeypointSet` с отфильтрованными точками.

    Raises:
        ValueError: Если ``top_k`` < 0.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    kps = kpset.keypoints
    descs = kpset.descriptors

    # Фильтр по минимальному отклику
    indices = [i for i, kp in enumerate(kps) if kp.response >= min_response]
    kps = [kps[i] for i in indices]
    if descs is not None and len(indices) > 0:
        descs = descs[indices]
    elif len(indices) == 0:
        descs = None

    # Сортировка и обрезка до top_k
    if top_k > 0 and len(kps) > top_k:
        order = sorted(range(len(kps)), key=lambda i: kps[i].response, reverse=True)[:top_k]
        kps = [kps[i] for i in order]
        if descs is not None:
            descs = descs[order]

    return KeypointSet(
        keypoints=kps,
        descriptors=descs if (descs is not None and len(kps) > 0) else None,
        detector=kpset.detector,
        params=dict(kpset.params),
    )


def filter_by_region(
    kpset: KeypointSet,
    mask: np.ndarray,
) -> KeypointSet:
    """Фильтровать ключевые точки по бинарной маске.

    Оставляет только точки, попадающие в ненулевые пиксели маски.

    Args:
        kpset: Исходный набор.
        mask:  Бинарная маска (H, W) uint8.

    Returns:
        Новый :class:`KeypointSet`.

    Raises:
        ValueError: Если маска не 2D.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got ndim={mask.ndim}")

    h, w = mask.shape
    kps = kpset.keypoints
    descs = kpset.descriptors
    keep = []
    for i, kp in enumerate(kps):
        xi = int(np.clip(round(kp.pt[0]), 0, w - 1))
        yi = int(np.clip(round(kp.pt[1]), 0, h - 1))
        if mask[yi, xi] > 0:
            keep.append(i)

    new_kps = [kps[i] for i in keep]
    new_descs = descs[keep] if (descs is not None and len(keep) > 0) else None

    return KeypointSet(
        keypoints=new_kps,
        descriptors=new_descs,
        detector=kpset.detector,
        params=dict(kpset.params),
    )


def describe_keypoints(
    img: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    detector: str = "orb",
) -> Optional[np.ndarray]:
    """Вычислить дескрипторы для заданных ключевых точек.

    Args:
        img:       Изображение uint8.
        keypoints: Список ``cv2.KeyPoint``.
        detector:  Тип детектора/дескриптора.

    Returns:
        Матрица дескрипторов (N, D) или ``None``, если точек нет.

    Raises:
        ValueError: Если ``detector`` неизвестен.
    """
    if not keypoints:
        return None
    valid_detectors = {"orb", "sift"}
    if detector not in valid_detectors:
        raise ValueError(
            f"detector must be one of {sorted(valid_detectors)}, got {detector!r}"
        )
    gray = _to_gray(img)
    det = _build_detector(detector, max_keypoints=len(keypoints) + 10)
    _, descs = det.compute(gray, keypoints)
    return descs


def match_descriptors(
    descs1: np.ndarray,
    descs2: np.ndarray,
    ratio_thresh: float = 0.75,
    cross_check: bool = False,
) -> List[cv2.DMatch]:
    """Сопоставить два набора дескрипторов методом BF + тест Лоу.

    Args:
        descs1:       Первый набор дескрипторов (N, D).
        descs2:       Второй набор дескрипторов (M, D).
        ratio_thresh: Порог теста соотношения Лоу (0 < threshold < 1).
        cross_check:  Если ``True``, дополнительно проводит перекрёстную проверку.

    Returns:
        Список ``cv2.DMatch`` — отобранных совпадений.

    Raises:
        ValueError: Если ``ratio_thresh`` не в (0, 1) или дескрипторы пустые.
    """
    if not (0 < ratio_thresh < 1):
        raise ValueError(f"ratio_thresh must be in (0, 1), got {ratio_thresh}")
    if descs1 is None or descs2 is None:
        raise ValueError("Descriptors must not be None")
    if len(descs1) == 0 or len(descs2) == 0:
        return []

    # Определить метрику по типу дескрипторов
    norm_type = cv2.NORM_HAMMING if descs1.dtype == np.uint8 else cv2.NORM_L2
    bf = cv2.BFMatcher(norm_type, crossCheck=False)

    if len(descs2) == 1:
        # knnMatch требует k ≤ len(descs2)
        raw = bf.match(descs1, descs2)
        return raw

    raw = bf.knnMatch(descs1, descs2, k=2)
    good: List[cv2.DMatch] = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        elif len(pair) == 1:
            good.append(pair[0])

    if cross_check:
        raw2 = bf.knnMatch(descs2, descs1, k=2)
        reverse_set: set = set()
        for pair in raw2:
            if len(pair) >= 1:
                reverse_set.add(pair[0].trainIdx)
        good = [m for m in good if m.queryIdx in reverse_set]

    return good


def filter_matches_ransac(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    reproj_threshold: float = 3.0,
) -> Tuple[List[cv2.DMatch], Optional[np.ndarray]]:
    """Геометрическая верификация совпадений через RANSAC (гомография).

    Args:
        kps1:             Ключевые точки первого изображения.
        kps2:             Ключевые точки второго изображения.
        matches:          Исходные совпадения.
        reproj_threshold: Порог ошибки репроекции (пикс.).

    Returns:
        Кортеж (inlier_matches, H):
        - ``inlier_matches`` — список валидных совпадений.
        - ``H`` — матрица гомографии 3×3 float64 или ``None``, если не найдена.

    Raises:
        ValueError: Если ``reproj_threshold`` ≤ 0.
    """
    if reproj_threshold <= 0:
        raise ValueError(f"reproj_threshold must be > 0, got {reproj_threshold}")
    if len(matches) < 4:
        return matches, None

    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
    if H is None or mask is None:
        return [], None

    inliers = [m for m, v in zip(matches, mask.ravel()) if v]
    return inliers, H


def keypoints_to_array(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """Преобразовать список ``cv2.KeyPoint`` в массив координат.

    Args:
        keypoints: Список ключевых точек.

    Returns:
        Массив float32 формы (N, 2) с координатами (x, y).
        Для пустого списка — массив (0, 2).
    """
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def array_to_keypoints(
    pts: np.ndarray,
    size: float = 1.0,
    response: float = 1.0,
) -> List[cv2.KeyPoint]:
    """Преобразовать массив координат в список ``cv2.KeyPoint``.

    Args:
        pts:      Массив (N, 2) с координатами (x, y).
        size:     Размер ключевой точки.
        response: Отклик.

    Returns:
        Список ``cv2.KeyPoint``.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return [cv2.KeyPoint(float(p[0]), float(p[1]), size, response=response) for p in pts]


def compute_match_score(
    matches: List[cv2.DMatch],
    n_kp1: int,
    n_kp2: int,
) -> float:
    """Вычислить нормированную оценку качества сопоставления.

    Метрика: |matches| / max(n_kp1, n_kp2) ∈ [0, 1].

    Args:
        matches: Список совпадений.
        n_kp1:   Число ключевых точек в первом наборе.
        n_kp2:   Число ключевых точек во втором наборе.

    Returns:
        Значение от 0.0 (нет совпадений) до 1.0 (идеальное покрытие).

    Raises:
        ValueError: Если ``n_kp1`` или ``n_kp2`` < 0.
    """
    if n_kp1 < 0 or n_kp2 < 0:
        raise ValueError(f"n_kp1 and n_kp2 must be >= 0, got {n_kp1}, {n_kp2}")
    denom = max(n_kp1, n_kp2)
    if denom == 0:
        return 0.0
    return float(min(len(matches), denom)) / denom


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _build_detector(detector: str, max_keypoints: int, **kwargs) -> cv2.Feature2D:
    if detector == "orb":
        return cv2.ORB_create(nfeatures=max_keypoints, **kwargs)
    if detector == "sift":
        return cv2.SIFT_create(nfeatures=max_keypoints, **kwargs)
    raise ValueError(f"Unknown detector: {detector!r}")
