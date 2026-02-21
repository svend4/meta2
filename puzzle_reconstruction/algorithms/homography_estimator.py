"""Оценка гомографии между парами фрагментов пазла.

Модуль вычисляет проективное преобразование (гомография) между двумя
изображениями или наборами точечных соответствий методами DLT и RANSAC,
а также раскладывает гомографию на компоненты (поворот, масштаб, сдвиг).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


_METHODS = {"dlt", "ransac", "lmeds"}


# ─── HomographyConfig ─────────────────────────────────────────────────────────

@dataclass
class HomographyConfig:
    """Параметры оценки гомографии.

    Атрибуты:
        method:         'dlt' | 'ransac' | 'lmeds'.
        ransac_thresh:  Порог репроекции RANSAC в пикселях (> 0).
        max_iters:      Максимальное число итераций RANSAC (>= 1).
        confidence:     Уровень уверенности RANSAC (0, 1).
        min_inliers:    Минимальное число инлаеров для успеха (>= 4).
    """

    method: str = "ransac"
    ransac_thresh: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.995
    min_inliers: int = 4

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                f"method должен быть одним из {_METHODS}, получено '{self.method}'"
            )
        if self.ransac_thresh <= 0:
            raise ValueError(
                f"ransac_thresh должен быть > 0, получено {self.ransac_thresh}"
            )
        if self.max_iters < 1:
            raise ValueError(
                f"max_iters должен быть >= 1, получено {self.max_iters}"
            )
        if not (0.0 < self.confidence < 1.0):
            raise ValueError(
                f"confidence должен быть в (0, 1), получено {self.confidence}"
            )
        if self.min_inliers < 4:
            raise ValueError(
                f"min_inliers должен быть >= 4, получено {self.min_inliers}"
            )


# ─── HomographyResult ─────────────────────────────────────────────────────────

@dataclass
class HomographyResult:
    """Результат оценки гомографии.

    Атрибуты:
        H:          Матрица гомографии 3×3 (None если оценить не удалось).
        n_inliers:  Число инлаеров (>= 0).
        is_valid:   True если H найдена и n_inliers >= min_inliers.
        reproj_err: Средняя ошибка репроекции инлаеров (>= 0).
        method:     Использованный метод.
    """

    H: Optional[np.ndarray]
    n_inliers: int
    is_valid: bool
    reproj_err: float
    method: str = "ransac"

    def __post_init__(self) -> None:
        if self.n_inliers < 0:
            raise ValueError(
                f"n_inliers должен быть >= 0, получено {self.n_inliers}"
            )
        if self.reproj_err < 0.0:
            raise ValueError(
                f"reproj_err должен быть >= 0, получено {self.reproj_err}"
            )

    @property
    def has_homography(self) -> bool:
        """True если матрица H доступна."""
        return self.H is not None


# ─── normalize_points ─────────────────────────────────────────────────────────

def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Нормализовать точки для численной устойчивости (изотропное масштабирование).

    Аргументы:
        pts: Массив точек (N × 2).

    Возвращает:
        (pts_norm, T) — нормализованные точки и матрица преобразования 3×3.

    Исключения:
        ValueError: Если pts не 2-D или менее 2 точек.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"pts должны быть (N, 2), получено shape={pts.shape}"
        )
    if pts.shape[0] < 2:
        raise ValueError("Требуется минимум 2 точки")

    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    scale = np.sqrt(2.0) / (np.linalg.norm(shifted, axis=1).mean() + 1e-10)
    T = np.array([
        [scale, 0.0,   -scale * centroid[0]],
        [0.0,   scale, -scale * centroid[1]],
        [0.0,   0.0,    1.0],
    ], dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T[:, :2]
    return pts_norm, T


# ─── dlt_homography ───────────────────────────────────────────────────────────

def dlt_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> Optional[np.ndarray]:
    """Оценить гомографию методом DLT (Direct Linear Transform).

    Аргументы:
        src_pts: Точки источника (N × 2, N >= 4).
        dst_pts: Точки назначения (N × 2).

    Возвращает:
        Матрица H (3×3, float64) или None при неудаче.

    Исключения:
        ValueError: Если точек меньше 4 или размеры не совпадают.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    if src.shape != dst.shape:
        raise ValueError(
            f"Формы src_pts и dst_pts не совпадают: {src.shape} vs {dst.shape}"
        )
    if src.shape[0] < 4:
        raise ValueError(
            f"Требуется минимум 4 точки, получено {src.shape[0]}"
        )

    src_n, T_src = normalize_points(src)
    dst_n, T_dst = normalize_points(dst)

    N = src.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = src_n[i]
        xp, yp = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_n @ T_src
    if abs(H[2, 2]) < 1e-10:
        return None
    H /= H[2, 2]
    return H


# ─── compute_reprojection_error ───────────────────────────────────────────────

def compute_reprojection_error(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> float:
    """Вычислить среднюю ошибку репроекции.

    Аргументы:
        H:       Матрица гомографии 3×3.
        src_pts: Точки источника (N × 2).
        dst_pts: Точки назначения (N × 2).

    Возвращает:
        Средняя евклидова ошибка (float >= 0).

    Исключения:
        ValueError: Если H не 3×3.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H должна быть 3×3, получено {H.shape}")
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ src_h.T).T
    denom = proj[:, 2:3]
    denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
    proj_2d = proj[:, :2] / denom
    err = np.linalg.norm(proj_2d - dst, axis=1)
    return float(err.mean())


# ─── estimate_homography ──────────────────────────────────────────────────────

def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cfg: Optional[HomographyConfig] = None,
) -> HomographyResult:
    """Оценить гомографию между двумя наборами точек.

    Аргументы:
        src_pts: Точки источника (N × 2).
        dst_pts: Точки назначения (N × 2).
        cfg:     Параметры (None → HomographyConfig()).

    Возвращает:
        HomographyResult.
    """
    if cfg is None:
        cfg = HomographyConfig()

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)

    if src.shape[0] < cfg.min_inliers:
        return HomographyResult(
            H=None, n_inliers=0, is_valid=False,
            reproj_err=0.0, method=cfg.method,
        )

    if cfg.method == "dlt":
        H = dlt_homography(src, dst)
        if H is None:
            return HomographyResult(H=None, n_inliers=0, is_valid=False,
                                    reproj_err=0.0, method=cfg.method)
        n_in = len(src)
        err = compute_reprojection_error(H, src.astype(np.float64),
                                         dst.astype(np.float64))
    else:
        cv_method = (cv2.RANSAC if cfg.method == "ransac" else cv2.LMEDS)
        H_cv, mask = cv2.findHomography(
            src, dst,
            method=cv_method,
            ransacReprojThreshold=cfg.ransac_thresh,
            maxIters=cfg.max_iters,
            confidence=cfg.confidence,
        )
        if H_cv is None:
            return HomographyResult(H=None, n_inliers=0, is_valid=False,
                                    reproj_err=0.0, method=cfg.method)
        H = H_cv.astype(np.float64)
        n_in = int(mask.sum()) if mask is not None else 0
        inlier_mask = (mask.ravel() == 1) if mask is not None else np.ones(len(src), bool)
        if inlier_mask.sum() > 0:
            err = compute_reprojection_error(
                H,
                src[inlier_mask].astype(np.float64),
                dst[inlier_mask].astype(np.float64),
            )
        else:
            err = 0.0

    valid = n_in >= cfg.min_inliers
    return HomographyResult(
        H=H, n_inliers=n_in, is_valid=valid,
        reproj_err=err, method=cfg.method,
    )


# ─── decompose_homography ─────────────────────────────────────────────────────

def decompose_homography(H: np.ndarray) -> dict:
    """Разложить гомографию на компоненты аффинного приближения.

    Возвращает словарь:
        scale_x, scale_y: масштабные факторы по осям.
        rotation_deg:     угол поворота (градусы).
        shear:            коэффициент сдвига.
        tx, ty:           трансляция.

    Аргументы:
        H: Матрица гомографии 3×3.

    Исключения:
        ValueError: Если H не 3×3.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H должна быть 3×3, получено {H.shape}")

    a, b, tx = H[0, 0], H[0, 1], H[0, 2]
    c, d, ty = H[1, 0], H[1, 1], H[1, 2]

    scale_x = float(np.sqrt(a ** 2 + c ** 2))
    scale_y_raw = float(a * d - b * c)
    scale_y = scale_y_raw / scale_x if scale_x > 1e-10 else 0.0
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    shear = float((a * b + c * d) / (scale_x ** 2)) if scale_x > 1e-10 else 0.0

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "shear": shear,
        "tx": float(tx),
        "ty": float(ty),
    }


# ─── warp_points ──────────────────────────────────────────────────────────────

def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить гомографию к набору точек.

    Аргументы:
        H:   Матрица гомографии 3×3.
        pts: Точки (N × 2).

    Возвращает:
        Преобразованные точки (N × 2, float64).

    Исключения:
        ValueError: Если H не 3×3 или pts не (N, 2).
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H должна быть 3×3, получено {H.shape}")
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts должны быть (N, 2), получено shape={pts.shape}")

    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    denom = proj[:, 2:3]
    denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
    return (proj[:, :2] / denom).astype(np.float64)


# ─── batch_estimate_homographies ──────────────────────────────────────────────

def batch_estimate_homographies(
    point_pairs: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Optional[HomographyConfig] = None,
) -> List[HomographyResult]:
    """Оценить гомографии для нескольких пар точек.

    Аргументы:
        point_pairs: Список пар (src_pts, dst_pts).
        cfg:         Параметры.

    Возвращает:
        Список HomographyResult.
    """
    return [estimate_homography(s, d, cfg) for s, d in point_pairs]
