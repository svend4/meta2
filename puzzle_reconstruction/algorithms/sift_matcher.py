"""Сопоставление фрагментов с помощью SIFT-дескрипторов.

Модуль предоставляет функции для извлечения ключевых точек SIFT,
вычисления дескрипторов, сопоставления пар (ratio test Лоуэ),
оценки качества совпадения (RANSAC homography), фильтрации совпадений
и пакетного сопоставления.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── SiftConfig ───────────────────────────────────────────────────────────────

@dataclass
class SiftConfig:
    """Параметры детектора SIFT и сопоставления.

    Атрибуты:
        n_features:    Максимальное число ключевых точек (>= 0; 0 = все).
        n_octave_layers: Уровней в октаве (>= 1).
        contrast_threshold: Порог контрастности (> 0).
        edge_threshold:  Порог края (> 0).
        sigma:           Начальная сигма (> 0).
        ratio_thresh:    Ratio test Лоуэ (0 < ratio_thresh < 1).
        min_matches:     Минимальное число совпадений для RANSAC (>= 4).
    """

    n_features: int = 500
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    ratio_thresh: float = 0.75
    min_matches: int = 4

    def __post_init__(self) -> None:
        if self.n_features < 0:
            raise ValueError(
                f"n_features должен быть >= 0, получено {self.n_features}"
            )
        if self.n_octave_layers < 1:
            raise ValueError(
                f"n_octave_layers должен быть >= 1, получено {self.n_octave_layers}"
            )
        if self.contrast_threshold <= 0.0:
            raise ValueError(
                f"contrast_threshold должен быть > 0, получено {self.contrast_threshold}"
            )
        if self.edge_threshold <= 0.0:
            raise ValueError(
                f"edge_threshold должен быть > 0, получено {self.edge_threshold}"
            )
        if self.sigma <= 0.0:
            raise ValueError(
                f"sigma должен быть > 0, получено {self.sigma}"
            )
        if not (0.0 < self.ratio_thresh < 1.0):
            raise ValueError(
                f"ratio_thresh должен быть в (0, 1), получено {self.ratio_thresh}"
            )
        if self.min_matches < 4:
            raise ValueError(
                f"min_matches должен быть >= 4, получено {self.min_matches}"
            )


# ─── MatchResult ──────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Результат сопоставления двух изображений.

    Атрибуты:
        n_matches:    Количество хороших совпадений после ratio test.
        n_inliers:    Количество инлаеров после RANSAC.
        score:        Оценка качества совпадения в [0, 1].
        homography:   Матрица гомографии (3×3) или None, если не вычислена.
        src_pts:      Точки на исходном изображении (N×2, float32).
        dst_pts:      Точки на целевом изображении (N×2, float32).
    """

    n_matches: int
    n_inliers: int
    score: float
    homography: Optional[np.ndarray] = None
    src_pts: Optional[np.ndarray] = None
    dst_pts: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.n_matches < 0:
            raise ValueError(
                f"n_matches должен быть >= 0, получено {self.n_matches}"
            )
        if self.n_inliers < 0:
            raise ValueError(
                f"n_inliers должен быть >= 0, получено {self.n_inliers}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    def is_reliable(self, min_inliers: int = 4) -> bool:
        """True если число инлаеров >= min_inliers."""
        return self.n_inliers >= min_inliers


# ─── _to_gray ─────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"img должен быть 2-D или 3-D, получено ndim={img.ndim}")


# ─── extract_keypoints ────────────────────────────────────────────────────────

def extract_keypoints(
    img: np.ndarray, cfg: Optional[SiftConfig] = None
) -> Tuple[List, Optional[np.ndarray]]:
    """Извлечь SIFT ключевые точки и дескрипторы.

    Аргументы:
        img: Изображение (uint8, 2-D или 3-D).
        cfg: Параметры SIFT. None → SiftConfig() по умолчанию.

    Возвращает:
        Кортеж (keypoints, descriptors):
          - keypoints: Список cv2.KeyPoint.
          - descriptors: ndarray (N×128, float32) или None если точек нет.

    Исключения:
        ValueError: Если img некорректен.
    """
    if cfg is None:
        cfg = SiftConfig()
    gray = _to_gray(img)
    sift = cv2.SIFT_create(
        nfeatures=cfg.n_features,
        nOctaveLayers=cfg.n_octave_layers,
        contrastThreshold=cfg.contrast_threshold,
        edgeThreshold=cfg.edge_threshold,
        sigma=cfg.sigma,
    )
    kps, descs = sift.detectAndCompute(gray, None)
    return list(kps), descs


# ─── match_descriptors ────────────────────────────────────────────────────────

def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> List[cv2.DMatch]:
    """Сопоставить дескрипторы методом kNN + ratio test Лоуэ.

    Аргументы:
        desc1:        Дескрипторы первого изображения (M×128, float32).
        desc2:        Дескрипторы второго изображения (N×128, float32).
        ratio_thresh: Порог отношения расстояний (0 < ratio_thresh < 1).

    Возвращает:
        Список «хороших» совпадений cv2.DMatch.

    Исключения:
        ValueError: Если ratio_thresh вне (0, 1) или дескрипторы пусты.
    """
    if not (0.0 < ratio_thresh < 1.0):
        raise ValueError(
            f"ratio_thresh должен быть в (0, 1), получено {ratio_thresh}"
        )
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    return good


# ─── compute_homography ────────────────────────────────────────────────────────

def compute_homography(
    kps1: List,
    kps2: List,
    matches: List[cv2.DMatch],
    min_matches: int = 4,
    ransac_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Вычислить гомографию RANSAC по совпадающим точкам.

    Аргументы:
        kps1:          Ключевые точки первого изображения.
        kps2:          Ключевые точки второго изображения.
        matches:       Список совпадений.
        min_matches:   Минимальное число совпадений (>= 4).
        ransac_thresh: Порог репроекции RANSAC (px, > 0).

    Возвращает:
        Кортеж (H, mask):
          - H: Матрица гомографии (3×3, float64) или None.
          - mask: Булева маска инлаеров.

    Исключения:
        ValueError: Если min_matches < 4 или ransac_thresh <= 0.
    """
    if min_matches < 4:
        raise ValueError(
            f"min_matches должен быть >= 4, получено {min_matches}"
        )
    if ransac_thresh <= 0.0:
        raise ValueError(
            f"ransac_thresh должен быть > 0, получено {ransac_thresh}"
        )

    if len(matches) < min_matches:
        return None, np.array([], dtype=bool)

    src = np.float32([kps1[m.queryIdx].pt for m in matches])
    dst = np.float32([kps2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    mask_bool = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
    return H, mask_bool


# ─── sift_match_pair ──────────────────────────────────────────────────────────

def sift_match_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    cfg: Optional[SiftConfig] = None,
) -> MatchResult:
    """Полное SIFT-сопоставление двух изображений.

    Аргументы:
        img1: Первое изображение (uint8).
        img2: Второе изображение (uint8).
        cfg:  Параметры SIFT. None → SiftConfig().

    Возвращает:
        MatchResult с числом совпадений, инлаеров, оценкой и гомографией.
    """
    if cfg is None:
        cfg = SiftConfig()

    kps1, desc1 = extract_keypoints(img1, cfg)
    kps2, desc2 = extract_keypoints(img2, cfg)
    matches = match_descriptors(desc1, desc2, ratio_thresh=cfg.ratio_thresh)

    n_matches = len(matches)
    if n_matches < cfg.min_matches:
        return MatchResult(n_matches=n_matches, n_inliers=0, score=0.0)

    H, mask = compute_homography(kps1, kps2, matches,
                                  min_matches=cfg.min_matches)
    n_inliers = int(mask.sum()) if len(mask) > 0 else 0
    score = n_inliers / max(n_matches, 1)

    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches])[mask] if len(mask) > 0 else None
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches])[mask] if len(mask) > 0 else None

    return MatchResult(
        n_matches=n_matches,
        n_inliers=n_inliers,
        score=float(np.clip(score, 0.0, 1.0)),
        homography=H,
        src_pts=src_pts,
        dst_pts=dst_pts,
    )


# ─── filter_matches_by_distance ───────────────────────────────────────────────

def filter_matches_by_distance(
    matches: List[cv2.DMatch], max_distance: float
) -> List[cv2.DMatch]:
    """Отфильтровать совпадения по максимальному расстоянию дескриптора.

    Аргументы:
        matches:      Список совпадений.
        max_distance: Максимально допустимое расстояние (>= 0).

    Возвращает:
        Отфильтрованный список совпадений.

    Исключения:
        ValueError: Если max_distance < 0.
    """
    if max_distance < 0.0:
        raise ValueError(
            f"max_distance должен быть >= 0, получено {max_distance}"
        )
    return [m for m in matches if m.distance <= max_distance]


# ─── batch_sift_match ─────────────────────────────────────────────────────────

def batch_sift_match(
    images: List[np.ndarray],
    cfg: Optional[SiftConfig] = None,
) -> Dict[Tuple[int, int], MatchResult]:
    """Сопоставить все пары изображений (N·(N-1)/2 пар).

    Аргументы:
        images: Список изображений (uint8).
        cfg:    Параметры SIFT.

    Возвращает:
        Словарь {(i, j): MatchResult} для всех i < j.
    """
    if cfg is None:
        cfg = SiftConfig()
    results: Dict[Tuple[int, int], MatchResult] = {}
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            results[(i, j)] = sift_match_pair(images[i], images[j], cfg)
    return results
