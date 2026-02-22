"""
Дескрипторное сопоставление фрагментов (feature-based matching).

Использует локальные дескрипторы (ORB, SIFT, AKAZE) для нахождения
соответствий между краями фрагментов. На основе найденных пар точек
оценивается гомография (RANSAC) и итоговый score совместимости.

Алгоритм:
    1. Извлечение ключевых точек и дескрипторов из ROI каждого края.
    2. Сопоставление дескрипторов (BruteForce или FLANN).
    3. Тест Лоу (ratio test, по умолчанию 0.75) для отсева плохих совпадений.
    4. RANSAC для оценки гомографии и маски инлайеров.
    5. score = inlier_ratio × (n_inliers / n_matches) — нормированный [0,1].

Классы:
    KeypointMatch     — одно совпадение (kp_src, kp_dst, distance, confidence)
    FeatureMatchResult — полный результат (matches, homography, score, method)

Функции:
    extract_features    — ключевые точки + дескрипторы (ORB/SIFT/AKAZE)
    match_descriptors   — ratio test, возвращает List[KeypointMatch]
    estimate_homography — RANSAC гомография из KeypointMatch
    feature_match_pair  — полный пайплайн для двух изображений
    edge_feature_score  — score совместимости двух краёв (для CompatEntry)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── KeypointMatch ────────────────────────────────────────────────────────────

@dataclass
class KeypointMatch:
    """
    Одно совпадение между ключевыми точками двух изображений.

    Attributes:
        pt_src:     (x, y) точка на исходном изображении.
        pt_dst:     (x, y) точка на целевом изображении.
        distance:   Расстояние между дескрипторами (меньше = лучше).
        confidence: Нормированная уверенность ∈ [0, 1].
    """
    pt_src:     Tuple[float, float]
    pt_dst:     Tuple[float, float]
    distance:   float
    confidence: float = 1.0

    def __repr__(self) -> str:
        return (f"KeypointMatch(src={self.pt_src}, dst={self.pt_dst}, "
                f"d={self.distance:.2f}, conf={self.confidence:.3f})")


# ─── FeatureMatchResult ───────────────────────────────────────────────────────

@dataclass
class FeatureMatchResult:
    """
    Результат дескрипторного сопоставления двух изображений.

    Attributes:
        matches:     Список KeypointMatch (после ratio test).
        homography:  3×3 матрица гомографии или None.
        inlier_mask: bool-массив инлайеров для RANSAC (len = len(matches)).
        score:       Итоговый score ∈ [0, 1].
        method:      Метод извлечения дескрипторов ('orb'/'sift'/'akaze').
        n_keypoints: Число ключевых точек (пара: src, dst).
    """
    matches:     List[KeypointMatch]
    homography:  Optional[np.ndarray]
    inlier_mask: np.ndarray
    score:       float
    method:      str
    n_keypoints: Tuple[int, int] = (0, 0)

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_inliers(self) -> int:
        if self.inlier_mask is None or len(self.inlier_mask) == 0:
            return 0
        return int(self.inlier_mask.sum())

    @property
    def inlier_ratio(self) -> float:
        if self.n_matches == 0:
            return 0.0
        return self.n_inliers / self.n_matches

    def __repr__(self) -> str:
        return (f"FeatureMatchResult(method={self.method!r}, "
                f"n_matches={self.n_matches}, "
                f"n_inliers={self.n_inliers}, "
                f"score={self.score:.4f})")


# ─── Извлечение дескрипторов ──────────────────────────────────────────────────

def _make_detector(method: str, n_features: int = 500):
    """Создаёт детектор + дескриптор для заданного метода."""
    m = method.lower()
    if m == "orb":
        return cv2.ORB_create(nfeatures=n_features)
    if m == "akaze":
        return cv2.AKAZE_create()
    if m == "sift":
        try:
            return cv2.SIFT_create(nfeatures=n_features)
        except AttributeError:
            # Старые версии OpenCV
            return cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
    raise ValueError(f"Неизвестный метод дескрипторов: {method!r}. "
                      f"Допустимые: 'orb', 'sift', 'akaze'")


def extract_features(img:        np.ndarray,
                      method:     str = "orb",
                      n_features: int = 500,
                      mask:       Optional[np.ndarray] = None
                      ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Извлекает ключевые точки и дескрипторы.

    Args:
        img:        BGR или grayscale изображение.
        method:     'orb' | 'sift' | 'akaze'.
        n_features: Максимальное число ключевых точек.
        mask:       Маска ROI (uint8, 255 = активная область).

    Returns:
        (keypoints, descriptors) — пустые списки если ничего не найдено.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    try:
        detector = _make_detector(method, n_features)
        kps, descs = detector.detectAndCompute(gray, mask)
    except Exception:
        return [], None

    if kps is None:
        kps = []
    return list(kps), descs


# ─── Сопоставление дескрипторов ───────────────────────────────────────────────

def match_descriptors(desc1:     np.ndarray,
                       desc2:     np.ndarray,
                       kps1:      List[cv2.KeyPoint],
                       kps2:      List[cv2.KeyPoint],
                       method:    str   = "orb",
                       ratio:     float = 0.75,
                       matcher:   str   = "bf") -> List[KeypointMatch]:
    """
    Сопоставляет дескрипторы с ratio test Лоу.

    Args:
        desc1, desc2: Матрицы дескрипторов (uint8 для ORB, float32 для SIFT).
        kps1, kps2:   Соответствующие ключевые точки.
        method:       Метод дескрипторов (влияет на норму).
        ratio:        Порог Лоу (0.75 — стандарт).
        matcher:      'bf' (BruteForce) или 'flann'.

    Returns:
        Список KeypointMatch после ratio test.
    """
    if (desc1 is None or desc2 is None
            or len(desc1) == 0 or len(desc2) == 0):
        return []

    # Нормировка для Brute-Force
    norm = cv2.NORM_HAMMING if method.lower() == "orb" else cv2.NORM_L2

    if matcher == "flann":
        # FLANN требует float32
        d1 = desc1.astype(np.float32)
        d2 = desc2.astype(np.float32)
        index_params  = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            raw = flann.knnMatch(d1, d2, k=2)
        except Exception:
            return []
    else:
        bf  = cv2.BFMatcher(norm, crossCheck=False)
        raw = bf.knnMatch(desc1, desc2, k=2)

    matches: List[KeypointMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            pt1 = kps1[m.queryIdx].pt
            pt2 = kps2[m.trainIdx].pt
            # confidence: инвертированное расстояние (нормировано к [0,1])
            conf = 1.0 - min(1.0, m.distance / max(n.distance, 1e-6))
            matches.append(KeypointMatch(
                pt_src=pt1,
                pt_dst=pt2,
                distance=float(m.distance),
                confidence=float(np.clip(conf, 0.0, 1.0)),
            ))

    return matches


# ─── Гомография ───────────────────────────────────────────────────────────────

def estimate_homography(matches:     List[KeypointMatch],
                         min_inliers: int   = 4,
                         ransac_thresh: float = 5.0
                         ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Оценивает гомографию из пар точек методом RANSAC.

    Args:
        matches:       Список KeypointMatch.
        min_inliers:   Минимальное число инлайеров для успеха.
        ransac_thresh: Порог RANSAC (пиксели).

    Returns:
        (H, mask) — гомография 3×3 и bool-маска инлайеров.
        Если недостаточно точек — (None, пустая маска).
    """
    if len(matches) < min_inliers:
        return None, np.zeros(len(matches), dtype=bool)

    src = np.array([m.pt_src for m in matches], dtype=np.float32)
    dst = np.array([m.pt_dst for m in matches], dtype=np.float32)

    try:
        H, mask = cv2.findHomography(src, dst,
                                      cv2.RANSAC, ransac_thresh)
    except Exception:
        return None, np.zeros(len(matches), dtype=bool)

    if H is None:
        return None, np.zeros(len(matches), dtype=bool)

    return H, mask.ravel().astype(bool)


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def feature_match_pair(img1:       np.ndarray,
                        img2:       np.ndarray,
                        method:     str   = "orb",
                        ratio:      float = 0.75,
                        n_features: int   = 500,
                        min_inliers: int  = 4) -> FeatureMatchResult:
    """
    Полный пайплайн дескрипторного сопоставления двух изображений.

    Args:
        img1, img2:  Входные изображения (BGR или grayscale).
        method:      'orb' | 'sift' | 'akaze'.
        ratio:       Ratio test Лоу.
        n_features:  Максимальное число ключевых точек.
        min_inliers: Порог для попытки оценки гомографии.

    Returns:
        FeatureMatchResult.
    """
    kps1, descs1 = extract_features(img1, method=method, n_features=n_features)
    kps2, descs2 = extract_features(img2, method=method, n_features=n_features)

    matches = match_descriptors(descs1, descs2, kps1, kps2,
                                  method=method, ratio=ratio)

    H, mask = estimate_homography(matches, min_inliers=min_inliers)

    # score = inlier_ratio × log(1 + n_inliers) / log(1 + n_features)
    n_inliers = int(mask.sum()) if len(mask) > 0 else 0
    if len(matches) > 0 and n_inliers >= min_inliers:
        inlier_ratio = n_inliers / len(matches)
        coverage     = math.log1p(n_inliers) / math.log1p(n_features)
        score        = float(np.clip(0.5 * inlier_ratio + 0.5 * coverage, 0.0, 1.0))
    else:
        score = 0.0

    return FeatureMatchResult(
        matches=matches,
        homography=H,
        inlier_mask=mask,
        score=score,
        method=method,
        n_keypoints=(len(kps1), len(kps2)),
    )


def edge_feature_score(edge_img1:  np.ndarray,
                        edge_img2:  np.ndarray,
                        method:     str   = "orb",
                        ratio:      float = 0.75) -> float:
    """
    Вычисляет score совместимости двух изображений краёв.

    Упрощённая версия feature_match_pair, возвращает только float score.

    Args:
        edge_img1, edge_img2: Небольшие изображения краёв (ROI).
        method: Метод дескрипторов.
        ratio:  Ratio test.

    Returns:
        score ∈ [0, 1].
    """
    result = feature_match_pair(edge_img1, edge_img2,
                                  method=method, ratio=ratio,
                                  n_features=200, min_inliers=3)
    return result.score
