"""Описание контуров фрагментов методом дескрипторов Фурье (FD).

Дескрипторы Фурье представляют контур в виде комплексной функции
и используют коэффициенты FFT для компактного, инвариантного к
трансляции и масштабу описания формы.

Классы:
    FourierConfig      — параметры вычисления дескрипторов
    FourierDescriptor  — дескриптор Фурье одного контура

Функции:
    complex_representation  — перевод (N, 2) точек в комплексную запись
    compute_contour_centroid — центроид набора точек
    compute_fd              — вычислить дескриптор для контура
    fd_similarity           — оценка сходства двух дескрипторов ∈ [0, 1]
    batch_compute_fd        — пакетное вычисление дескрипторов
    rank_by_fd              — ранжирование по убыванию сходства с эталоном
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── FourierConfig ────────────────────────────────────────────────────────────

@dataclass
class FourierConfig:
    """Параметры дескриптора Фурье.

    Атрибуты:
        n_coeffs:  Число удерживаемых коэффициентов (>= 4).
        normalize: True → нормировать на первый ненулевой коэффициент
                   (инвариантность к масштабу и повороту).
    """
    n_coeffs:  int  = 32
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.n_coeffs < 4:
            raise ValueError(
                f"n_coeffs должен быть >= 4, получено {self.n_coeffs}"
            )


# ─── FourierDescriptor ────────────────────────────────────────────────────────

@dataclass
class FourierDescriptor:
    """Дескриптор Фурье контура.

    Атрибуты:
        fragment_id:  Идентификатор фрагмента (>= 0).
        edge_id:      Идентификатор края (>= 0).
        coefficients: 1D массив float32, n_coeffs комплексных компонент
                      уложенных как [Re(c0), Im(c0), Re(c1), Im(c1), ...].
        n_coeffs:     Число коэффициентов (>= 4).
        params:       Дополнительные параметры.
    """
    fragment_id:  int
    edge_id:      int
    coefficients: np.ndarray   # shape (2 * n_coeffs,), float32
    n_coeffs:     int
    params:       Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.edge_id < 0:
            raise ValueError(
                f"edge_id должен быть >= 0, получено {self.edge_id}"
            )
        if self.n_coeffs < 4:
            raise ValueError(
                f"n_coeffs должен быть >= 4, получено {self.n_coeffs}"
            )
        self.coefficients = np.asarray(self.coefficients, dtype=np.float32)
        if self.coefficients.ndim != 1:
            raise ValueError(
                "coefficients должен быть 1-D массивом, "
                f"получено ndim={self.coefficients.ndim}"
            )

    @property
    def magnitude(self) -> np.ndarray:
        """Амплитуды коэффициентов (n_coeffs,), float32."""
        c = self.coefficients.reshape(-1, 2)
        return np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2).astype(np.float32)

    @property
    def dim(self) -> int:
        """Полная длина вектора коэффициентов (2 × n_coeffs)."""
        return len(self.coefficients)


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def compute_contour_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Вычислить центроид набора точек.

    Аргументы:
        points: Массив формы (N, 2) или (N, 1, 2).

    Возвращает:
        (cx, cy) — координаты центроида.

    Исключения:
        ValueError: Если points пуст.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("points не должен быть пустым")
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    return cx, cy


def complex_representation(points: np.ndarray) -> np.ndarray:
    """Преобразовать контур (N, 2) в комплексный вектор (N,).

    Аргументы:
        points: Массив (N, 2) или (N, 1, 2) с координатами (x, y).

    Возвращает:
        Комплексный массив (N,): z = x + i·y.

    Исключения:
        ValueError: Если points пуст.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("points не должен быть пустым")
    return pts[:, 0] + 1j * pts[:, 1]


# ─── compute_fd ───────────────────────────────────────────────────────────────

def compute_fd(
    points: np.ndarray,
    fragment_id: int = 0,
    edge_id:     int = 0,
    cfg:         Optional[FourierConfig] = None,
) -> FourierDescriptor:
    """Вычислить дескриптор Фурье для контура.

    Алгоритм:
        1. Перевести в комплексную запись z = x + i·y.
        2. Вычесть центроид (инвариантность к трансляции).
        3. FFT → взять первые n_coeffs компонент.
        4. При normalize=True — делить на амплитуду первого коэф.

    Аргументы:
        points:      Контур (N, 2) или (N, 1, 2), N >= 4.
        fragment_id: Идентификатор фрагмента (>= 0).
        edge_id:     Идентификатор края (>= 0).
        cfg:         Конфигурация (None → FourierConfig()).

    Возвращает:
        FourierDescriptor.

    Исключения:
        ValueError: Если points содержит < 4 точек.
    """
    if cfg is None:
        cfg = FourierConfig()

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 4:
        raise ValueError(
            f"points должен содержать >= 4 точек, получено {n}"
        )

    # Комплексное представление
    z = pts[:, 0] + 1j * pts[:, 1]

    # Вычтем центроид
    z -= z.mean()

    # FFT
    F = np.fft.fft(z)

    # Берём n_coeffs частот (с обёрткой при n < n_coeffs)
    k = cfg.n_coeffs
    indices = np.arange(k) % n
    coeffs_c = F[indices]  # (k,) complex

    # Нормировка по первому ненулевому коэффициенту
    if cfg.normalize:
        first_amp = abs(coeffs_c[0]) if abs(coeffs_c[0]) > 1e-10 else (
            abs(coeffs_c[1]) if k > 1 else 1.0
        )
        if first_amp > 1e-10:
            coeffs_c = coeffs_c / first_amp

    # Укладываем Re/Im в float32 вектор
    flat = np.zeros(2 * k, dtype=np.float32)
    flat[0::2] = coeffs_c.real.astype(np.float32)
    flat[1::2] = coeffs_c.imag.astype(np.float32)

    return FourierDescriptor(
        fragment_id=fragment_id,
        edge_id=edge_id,
        coefficients=flat,
        n_coeffs=k,
        params={"n_points": n, "normalize": cfg.normalize},
    )


# ─── fd_similarity ────────────────────────────────────────────────────────────

def fd_similarity(a: FourierDescriptor, b: FourierDescriptor) -> float:
    """Оценить сходство двух дескрипторов Фурье.

    Использует косинусное сходство амплитудных спектров.

    Аргументы:
        a: Первый дескриптор.
        b: Второй дескриптор.

    Возвращает:
        Оценка ∈ [0, 1] (1 = идентичные).

    Исключения:
        ValueError: Если n_coeffs дескрипторов не совпадают.
    """
    if a.n_coeffs != b.n_coeffs:
        raise ValueError(
            f"n_coeffs дескрипторов не совпадают: {a.n_coeffs} vs {b.n_coeffs}"
        )

    mag_a = a.magnitude.astype(np.float64)
    mag_b = b.magnitude.astype(np.float64)

    norm_a = np.linalg.norm(mag_a)
    norm_b = np.linalg.norm(mag_b)

    if norm_a < 1e-12 and norm_b < 1e-12:
        return 1.0
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cosine = float(np.dot(mag_a, mag_b) / (norm_a * norm_b))
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


# ─── batch_compute_fd ─────────────────────────────────────────────────────────

def batch_compute_fd(
    points_list: List[np.ndarray],
    cfg:         Optional[FourierConfig] = None,
) -> List[FourierDescriptor]:
    """Пакетное вычисление дескрипторов Фурье.

    Аргументы:
        points_list: Список контуров (каждый (N, 2)).
        cfg:         Конфигурация (None → FourierConfig()).

    Возвращает:
        Список FourierDescriptor; edge_id = позиция в списке.
    """
    if cfg is None:
        cfg = FourierConfig()
    return [
        compute_fd(pts, fragment_id=0, edge_id=i, cfg=cfg)
        for i, pts in enumerate(points_list)
    ]


# ─── rank_by_fd ───────────────────────────────────────────────────────────────

def rank_by_fd(
    query:       FourierDescriptor,
    candidates:  List[FourierDescriptor],
    indices:     Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Ранжировать кандидатов по сходству с эталоном.

    Аргументы:
        query:      Эталонный дескриптор.
        candidates: Список дескрипторов для сравнения.
        indices:    Индексы кандидатов (по умолчанию 0, 1, …, len-1).

    Возвращает:
        Список кортежей (idx, score), отсортированный по убыванию score.

    Исключения:
        ValueError: Если lengths indices и candidates не совпадают.
    """
    if indices is None:
        indices = list(range(len(candidates)))
    if len(indices) != len(candidates):
        raise ValueError(
            f"Длины indices ({len(indices)}) и candidates "
            f"({len(candidates)}) должны совпадать"
        )

    scores = [fd_similarity(query, cand) for cand in candidates]
    ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    return [(int(idx), float(sc)) for idx, sc in ranked]
