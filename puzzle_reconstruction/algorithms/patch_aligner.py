"""Выравнивание патчей фрагментов документа методом фазовой корреляции и NCC.

Модуль вычисляет субпиксельное смещение между двумя патчами изображения
и оценивает качество совмещения (пиковое соотношение сигнал/шум, NCC).

Публичный API:
    AlignConfig      — параметры выравнивания
    AlignResult      — результат выравнивания (смещение, качество, метод)
    phase_correlate  — субпиксельное смещение через фазовую корреляцию
    ncc_score        — нормированная кросс-корреляция патчей ∈ [–1, 1]
    align_patches    — объединённый метод (phase + NCC-верификация)
    refine_alignment — улучшение смещения методом полного перебора в окрестности
    batch_align      — пакетное выравнивание списка пар патчей
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── AlignConfig ──────────────────────────────────────────────────────────────

@dataclass
class AlignConfig:
    """Параметры алгоритма выравнивания патчей.

    Атрибуты:
        method:          Метод: 'phase' | 'ncc' | 'combined'.
        max_shift:       Максимально допустимое смещение (пикс., > 0).
        upsample_factor: Коэффициент увеличения точности фазовой корреляции (>= 1).
        ncc_threshold:   Минимальная NCC-оценка для «хорошего» совмещения (∈ [0, 1]).
        refine_radius:   Радиус перебора при уточнении (>= 0, целое).
    """

    method: str = "combined"
    max_shift: float = 20.0
    upsample_factor: int = 1
    ncc_threshold: float = 0.5
    refine_radius: int = 2

    _VALID_METHODS = frozenset({"phase", "ncc", "combined"})

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"method должен быть одним из {sorted(self._VALID_METHODS)}, "
                f"получено {self.method!r}"
            )
        if self.max_shift <= 0.0:
            raise ValueError(
                f"max_shift должен быть > 0, получено {self.max_shift}"
            )
        if self.upsample_factor < 1:
            raise ValueError(
                f"upsample_factor должен быть >= 1, получено {self.upsample_factor}"
            )
        if not (0.0 <= self.ncc_threshold <= 1.0):
            raise ValueError(
                f"ncc_threshold должен быть в [0, 1], получено {self.ncc_threshold}"
            )
        if self.refine_radius < 0:
            raise ValueError(
                f"refine_radius должен быть >= 0, получено {self.refine_radius}"
            )


# ─── AlignResult ──────────────────────────────────────────────────────────────

@dataclass
class AlignResult:
    """Результат выравнивания двух патчей.

    Атрибуты:
        shift:      Смещение (dy, dx) в пикселях.
        ncc:        NCC-оценка совмещения ∈ [–1, 1].
        psnr:       Пиковое соотношение сигнал/шум (дБ, >= 0) или 0 при неудаче.
        success:    True если NCC >= ncc_threshold.
        method:     Применённый метод.
        params:     Дополнительные данные.
    """

    shift: Tuple[float, float]
    ncc: float
    psnr: float
    success: bool
    method: str
    params: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.ncc <= 1.0):
            raise ValueError(
                f"ncc должен быть в [–1, 1], получено {self.ncc}"
            )
        if self.psnr < 0.0:
            raise ValueError(
                f"psnr должен быть >= 0, получено {self.psnr}"
            )

    @property
    def shift_magnitude(self) -> float:
        """Длина вектора смещения (Евклидова норма)."""
        dy, dx = self.shift
        return float(np.sqrt(dy ** 2 + dx ** 2))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AlignResult(shift=({self.shift[0]:.2f},{self.shift[1]:.2f}), "
            f"ncc={self.ncc:.3f}, success={self.success})"
        )


# ─── _to_gray_f32 ─────────────────────────────────────────────────────────────

def _to_gray_f32(patch: np.ndarray) -> np.ndarray:
    """Привести патч к grayscale float32 2-D массиву."""
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim == 3:
        # Взвешенное преобразование RGB → Gray
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    if arr.ndim != 2:
        raise ValueError(
            f"patch должен быть 2-D или 3-D, получено ndim={patch.ndim}"
        )
    return arr


# ─── phase_correlate ──────────────────────────────────────────────────────────

def phase_correlate(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    upsample_factor: int = 1,
) -> Tuple[float, float]:
    """Вычислить субпиксельное смещение через фазовую корреляцию.

    Аргументы:
        patch_a:         Опорный патч (H, W) или (H, W, C).
        patch_b:         Целевой патч (H, W) или (H, W, C).
        upsample_factor: Коэффициент увеличения (>= 1; текущая реализация
                         поддерживает 1 и upsampling через zero-padding FFT).

    Возвращает:
        Кортеж (dy, dx) в пикселях.

    Исключения:
        ValueError: При несовпадении форм или некорректных аргументах.
    """
    if upsample_factor < 1:
        raise ValueError(
            f"upsample_factor должен быть >= 1, получено {upsample_factor}"
        )

    a = _to_gray_f32(patch_a)
    b = _to_gray_f32(patch_b)

    if a.shape != b.shape:
        raise ValueError(
            f"Формы патчей не совпадают: {a.shape} != {b.shape}"
        )

    # FFT-based phase correlation
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    cross_power = fa * np.conj(fb)
    norm = np.abs(cross_power) + 1e-12
    cross_power /= norm

    # Обратное преобразование
    corr = np.fft.ifft2(cross_power).real

    # Найти пик
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = float(peak_idx[0]), float(peak_idx[1])

    # Перевести в диапазон [-H/2, H/2)
    h, w = a.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    if upsample_factor > 1:
        # Упрощённый subpixel: параболическая интерполяция пика корреляции
        r, c = int(peak_idx[0]), int(peak_idx[1])
        r_prev = (r - 1) % h
        r_next = (r + 1) % h
        c_prev = (c - 1) % w
        c_next = (c + 1) % w
        denom_y = 2.0 * corr[r, c] - corr[r_prev, c] - corr[r_next, c]
        denom_x = 2.0 * corr[r, c] - corr[r, c_prev] - corr[r, c_next]
        if abs(denom_y) > 1e-9:
            sub_dy = 0.5 * (corr[r_next, c] - corr[r_prev, c]) / denom_y
            dy += sub_dy
        if abs(denom_x) > 1e-9:
            sub_dx = 0.5 * (corr[r, c_next] - corr[r, c_prev]) / denom_x
            dx += sub_dx

    return float(dy), float(dx)


# ─── ncc_score ────────────────────────────────────────────────────────────────

def ncc_score(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Нормированная кросс-корреляция двух патчей ∈ [–1, 1].

    При нулевом стандартном отклонении одного из патчей возвращает 0.0.

    Аргументы:
        patch_a: Патч (H, W) или (H, W, C) — опорный.
        patch_b: Патч той же формы — целевой.

    Возвращает:
        NCC ∈ [–1, 1].

    Исключения:
        ValueError: При несовпадении форм.
    """
    a = _to_gray_f32(patch_a).ravel()
    b = _to_gray_f32(patch_b).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"Формы патчей не совпадают: {patch_a.shape} != {patch_b.shape}"
        )

    a_c = a - a.mean()
    b_c = b - b.mean()
    na = float(np.linalg.norm(a_c))
    nb = float(np.linalg.norm(b_c))

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    return float(np.clip(np.dot(a_c, b_c) / (na * nb), -1.0, 1.0))


# ─── _compute_psnr ────────────────────────────────────────────────────────────

def _compute_psnr(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """PSNR (дБ) между двумя патчами одинаковой формы. 0.0 при нулевом MSE."""
    a = _to_gray_f32(patch_a)
    b = _to_gray_f32(patch_b)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-9:
        return 0.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


# ─── align_patches ────────────────────────────────────────────────────────────

def align_patches(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    cfg: Optional[AlignConfig] = None,
) -> AlignResult:
    """Выровнять два патча и оценить качество совмещения.

    Аргументы:
        patch_a: Опорный патч.
        patch_b: Целевой патч (должен иметь ту же форму).
        cfg:     Параметры (None → AlignConfig()).

    Возвращает:
        :class:`AlignResult`.
    """
    if cfg is None:
        cfg = AlignConfig()

    method = cfg.method

    if method in ("phase", "combined"):
        dy, dx = phase_correlate(patch_a, patch_b,
                                 upsample_factor=cfg.upsample_factor)
    else:
        dy, dx = 0.0, 0.0

    # Ограничить смещение
    shift_mag = float(np.sqrt(dy ** 2 + dx ** 2))
    if shift_mag > cfg.max_shift:
        scale = cfg.max_shift / (shift_mag + 1e-12)
        dy, dx = dy * scale, dx * scale

    ncc = ncc_score(patch_a, patch_b)
    psnr = _compute_psnr(patch_a, patch_b)
    success = ncc >= cfg.ncc_threshold

    return AlignResult(
        shift=(float(dy), float(dx)),
        ncc=float(ncc),
        psnr=float(psnr),
        success=success,
        method=method,
        params={
            "max_shift": cfg.max_shift,
            "upsample_factor": cfg.upsample_factor,
            "ncc_threshold": cfg.ncc_threshold,
        },
    )


# ─── refine_alignment ─────────────────────────────────────────────────────────

def refine_alignment(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    initial_shift: Tuple[float, float],
    radius: int = 2,
) -> Tuple[float, float]:
    """Уточнить смещение перебором в целочисленной окрестности.

    Аргументы:
        patch_a:       Опорный патч (H, W).
        patch_b:       Целевой патч (H, W).
        initial_shift: Начальное смещение (dy, dx).
        radius:        Радиус поиска в пикселях (>= 0).

    Возвращает:
        Уточнённое смещение (dy, dx).

    Исключения:
        ValueError: Если radius < 0.
    """
    if radius < 0:
        raise ValueError(f"radius должен быть >= 0, получено {radius}")

    a = _to_gray_f32(patch_a)
    b = _to_gray_f32(patch_b)

    h, w = a.shape
    idy = int(round(initial_shift[0]))
    idx = int(round(initial_shift[1]))

    best_ncc = -2.0
    best_dy, best_dx = float(idy), float(idx)

    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            dy_i = idy + ddy
            dx_i = idx + ddx
            # Сдвинуть patch_b
            shift_y = int(np.clip(-dy_i, -h + 1, h - 1))
            shift_x = int(np.clip(-dx_i, -w + 1, w - 1))
            b_shifted = np.roll(np.roll(b, shift_y, axis=0), shift_x, axis=1)
            score = ncc_score(a, b_shifted)
            if score > best_ncc:
                best_ncc = score
                best_dy, best_dx = float(dy_i), float(dx_i)

    return best_dy, best_dx


# ─── batch_align ──────────────────────────────────────────────────────────────

def batch_align(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Optional[AlignConfig] = None,
) -> List[AlignResult]:
    """Пакетное выравнивание списка пар патчей.

    Аргументы:
        pairs: Список пар (patch_a, patch_b).
        cfg:   Параметры выравнивания.

    Возвращает:
        Список :class:`AlignResult` той же длины.
    """
    if cfg is None:
        cfg = AlignConfig()
    return [align_patches(a, b, cfg) for a, b in pairs]
