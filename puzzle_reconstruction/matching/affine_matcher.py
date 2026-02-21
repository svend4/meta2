"""
Аффинное сопоставление фрагментов документа.

Оценивает аффинные преобразования между парами фрагментов по ключевым
точкам и дескрипторам. Вычисляет качество выравнивания, геометрическую
согласованность и итоговую оценку совместимости.

Экспортирует:
    AffineMatchResult   — результат аффинного сопоставления
    estimate_affine     — оценить матрицу аффинного преобразования
    apply_affine_pts    — применить аффин к набору точек
    affine_reprojection_error — ошибка репроекции по парам точек
    score_affine_match  — итоговая оценка (0–1)
    match_fragments_affine — полный пайплайн: ключевые точки → оценка
    batch_affine_match  — пакетное сопоставление одного фрагмента со списком
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class AffineMatchResult:
    """Результат аффинного сопоставления двух фрагментов.

    Attributes:
        idx1:        Индекс первого фрагмента.
        idx2:        Индекс второго фрагмента.
        M:           Матрица аффинного преобразования 2×3 (float64) или ``None``.
        n_inliers:   Число совпадений-инлайеров.
        reprojection_error: Средняя ошибка репроекции (пикс.) для инлайеров.
        score:       Нормированная оценка совместимости [0, 1].
        params:      Параметры вычисления.
    """
    idx1: int
    idx2: int
    M: Optional[np.ndarray]
    n_inliers: int
    reprojection_error: float
    score: float
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AffineMatchResult(idx1={self.idx1}, idx2={self.idx2}, "
            f"inliers={self.n_inliers}, score={self.score:.4f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def estimate_affine(
    pts1: np.ndarray,
    pts2: np.ndarray,
    method: str = "ransac",
    ransac_threshold: float = 3.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Оценить матрицу аффинного преобразования по парам точек.

    Args:
        pts1:             Исходные точки (N, 2) float32.
        pts2:             Целевые точки (N, 2) float32.
        method:           ``'ransac'`` или ``'lmeds'``.
        ransac_threshold: Порог ошибки репроекции для RANSAC (пикс.).

    Returns:
        Кортеж (M, inlier_mask):
        - M — матрица аффинного преобразования 2×3 float64, или ``None``.
        - inlier_mask — булев массив формы (N,) или ``None``.

    Raises:
        ValueError: Если ``method`` не ``'ransac'`` / ``'lmeds'``,
                    или числo точек < 3, или формы не совпадают.
    """
    if method not in ("ransac", "lmeds"):
        raise ValueError(f"method must be 'ransac' or 'lmeds', got {method!r}")
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p2.shape:
        raise ValueError(
            f"pts1 and pts2 must have the same shape, got {p1.shape} vs {p2.shape}"
        )
    if len(p1) < 3:
        raise ValueError(f"At least 3 point pairs are required, got {len(p1)}")

    cv_method = cv2.RANSAC if method == "ransac" else cv2.LMEDS
    M, mask = cv2.estimateAffine2D(p1, p2, method=cv_method,
                                   ransacReprojThreshold=ransac_threshold)
    if M is None:
        return None, None
    inlier_mask = mask.ravel().astype(bool) if mask is not None else None
    return M, inlier_mask


def apply_affine_pts(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить аффинное преобразование к набору точек.

    Args:
        M:   Матрица аффинного преобразования 2×3.
        pts: Точки (N, 2) или (N, 1, 2) float32.

    Returns:
        Преобразованные точки (N, 2) float32.

    Raises:
        ValueError: Если форма ``M`` не (2, 3).
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError(f"M must have shape (2, 3), got {M.shape}")
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(p), 1), dtype=np.float32)
    ph = np.hstack([p, ones])  # (N, 3)
    result = (M @ ph.T).T       # (N, 2)
    return result.astype(np.float32)


def affine_reprojection_error(
    M: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
) -> float:
    """Вычислить среднюю ошибку репроекции аффинного преобразования.

    Args:
        M:           Матрица аффинного преобразования 2×3.
        pts1:        Исходные точки (N, 2).
        pts2:        Целевые точки (N, 2).
        inlier_mask: Булев массив (N,) — если задан, учитываются только инлайеры.

    Returns:
        Средняя евклидова ошибка (пикс.). 0.0, если нет точек.

    Raises:
        ValueError: Если форма ``M`` не (2, 3) или размеры точек не совпадают.
    """
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p2.shape:
        raise ValueError(
            f"pts1 and pts2 shapes must match, got {p1.shape} vs {p2.shape}"
        )
    if len(p1) == 0:
        return 0.0
    projected = apply_affine_pts(M, p1)
    errors = np.linalg.norm(projected - p2, axis=1)
    if inlier_mask is not None:
        mask = np.asarray(inlier_mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        errors = errors[mask]
    return float(errors.mean())


def score_affine_match(
    n_inliers: int,
    reprojection_error: float,
    max_inliers: int = 100,
    max_error: float = 5.0,
    w_inliers: float = 0.6,
    w_error: float = 0.4,
) -> float:
    """Вычислить нормированную оценку аффинного сопоставления.

    Оценка = w_inliers × (n_inliers / max_inliers)
             + w_error × max(0, 1 − reprojection_error / max_error)

    Args:
        n_inliers:          Число инлайеров.
        reprojection_error: Средняя ошибка репроекции (пикс.).
        max_inliers:        Нормировочный максимум инлайеров (> 0).
        max_error:          Ошибка репроекции, при которой компонент = 0 (> 0).
        w_inliers:          Вес компонента инлайеров.
        w_error:            Вес компонента ошибки.

    Returns:
        Оценка ∈ [0, 1].

    Raises:
        ValueError: Если ``max_inliers`` ≤ 0, ``max_error`` ≤ 0,
                    или веса не суммируются примерно к 1.
    """
    if max_inliers <= 0:
        raise ValueError(f"max_inliers must be > 0, got {max_inliers}")
    if max_error <= 0:
        raise ValueError(f"max_error must be > 0, got {max_error}")
    if abs(w_inliers + w_error - 1.0) > 0.01:
        raise ValueError(
            f"Weights w_inliers + w_error must sum to 1.0, "
            f"got {w_inliers + w_error:.4f}"
        )
    inlier_score = min(float(n_inliers) / max_inliers, 1.0)
    error_score = max(0.0, 1.0 - reprojection_error / max_error)
    return float(w_inliers * inlier_score + w_error * error_score)


def match_fragments_affine(
    img1: np.ndarray,
    img2: np.ndarray,
    idx1: int = 0,
    idx2: int = 1,
    max_keypoints: int = 200,
    ratio_thresh: float = 0.75,
    ransac_threshold: float = 3.0,
    max_inliers: int = 100,
    max_error: float = 5.0,
) -> AffineMatchResult:
    """Полный пайплайн аффинного сопоставления двух изображений-фрагментов.

    ORB детектирование → BF+Лоу сопоставление → RANSAC аффин → оценка.

    Args:
        img1:             Первый фрагмент uint8 (2D или BGR).
        img2:             Второй фрагмент uint8.
        idx1, idx2:       Индексы фрагментов.
        max_keypoints:    Максимальное число ORB ключевых точек.
        ratio_thresh:     Порог теста Лоу.
        ransac_threshold: Порог RANSAC (пикс.).
        max_inliers:      Нормировочный максимум для scoring.
        max_error:        Максимальная ошибка для scoring.

    Returns:
        :class:`AffineMatchResult`.
    """
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)

    orb = cv2.ORB_create(nfeatures=max_keypoints)
    kps1, descs1 = orb.detectAndCompute(gray1, None)
    kps2, descs2 = orb.detectAndCompute(gray2, None)

    params = {
        "max_keypoints": max_keypoints,
        "ratio_thresh": ratio_thresh,
        "ransac_threshold": ransac_threshold,
    }

    if (descs1 is None or descs2 is None
            or len(kps1) < 3 or len(kps2) < 3):
        return AffineMatchResult(
            idx1=idx1, idx2=idx2, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0, params=params,
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if len(kps2) == 1:
        raw_matches = bf.match(descs1, descs2)
        good_matches = raw_matches
    else:
        raw_matches = bf.knnMatch(descs1, descs2, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
                good_matches.append(pair[0])

    if len(good_matches) < 3:
        return AffineMatchResult(
            idx1=idx1, idx2=idx2, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0, params=params,
        )

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    M, inlier_mask = estimate_affine(pts1, pts2,
                                     method="ransac",
                                     ransac_threshold=ransac_threshold)
    if M is None:
        return AffineMatchResult(
            idx1=idx1, idx2=idx2, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0, params=params,
        )

    n_in = int(inlier_mask.sum()) if inlier_mask is not None else len(good_matches)
    err = affine_reprojection_error(M, pts1, pts2, inlier_mask)
    score = score_affine_match(
        n_in, err, max_inliers=max_inliers, max_error=max_error
    )

    return AffineMatchResult(
        idx1=idx1, idx2=idx2, M=M,
        n_inliers=n_in, reprojection_error=err,
        score=score, params=params,
    )


def batch_affine_match(
    query: np.ndarray,
    candidates: List[np.ndarray],
    query_idx: int = 0,
    max_keypoints: int = 200,
    ratio_thresh: float = 0.75,
) -> List[AffineMatchResult]:
    """Сопоставить один фрагмент со списком кандидатов.

    Args:
        query:         Фрагмент-запрос uint8.
        candidates:    Список фрагментов-кандидатов uint8.
        query_idx:     Индекс фрагмента-запроса.
        max_keypoints: Максимальное число ORB точек.
        ratio_thresh:  Порог теста Лоу.

    Returns:
        Список :class:`AffineMatchResult` длиной ``len(candidates)``.
    """
    return [
        match_fragments_affine(
            query, cand,
            idx1=query_idx, idx2=i,
            max_keypoints=max_keypoints,
            ratio_thresh=ratio_thresh,
        )
        for i, cand in enumerate(candidates)
    ]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
