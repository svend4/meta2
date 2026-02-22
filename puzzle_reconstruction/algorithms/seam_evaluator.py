"""Оценка качества шва между двумя фрагментами документа.

Модуль реализует многоканальную оценку стыка фрагментов по
трём характеристикам краевых полос: непрерывность цвета,
непрерывность градиента и непрерывность текстуры.

Классы:
    SeamConfig  — веса каналов и ширина полосы
    SeamScore   — результат оценки одного шва

Функции:
    extract_seam_strip      — извлечение полосы вдоль заданной стороны
    color_continuity        — оценка цветового совпадения
    gradient_continuity     — оценка совпадения градиентов
    texture_continuity      — оценка совпадения текстуры
    evaluate_seam           — комплексная оценка пары (img_a, img_b)
    batch_evaluate_seams    — пакетная оценка списка пар
    rank_seams              — ранжирование по убыванию оценки
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── SeamConfig ───────────────────────────────────────────────────────────────

@dataclass
class SeamConfig:
    """Конфигурация оценки шва.

    Атрибуты:
        w_color:      Вес канала цвета (>= 0).
        w_gradient:   Вес канала градиента (>= 0).
        w_texture:    Вес канала текстуры (>= 0).
        blend_width:  Ширина полосы (пикселей, >= 1).
    """
    w_color:     float = 0.40
    w_gradient:  float = 0.35
    w_texture:   float = 0.25
    blend_width: int   = 8

    def __post_init__(self) -> None:
        for name, val in [("w_color", self.w_color),
                          ("w_gradient", self.w_gradient),
                          ("w_texture", self.w_texture)]:
            if val < 0.0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")
        if self.blend_width < 1:
            raise ValueError(
                f"blend_width должен быть >= 1, получено {self.blend_width}"
            )

    @property
    def total_weight(self) -> float:
        return self.w_color + self.w_gradient + self.w_texture


# ─── SeamScore ────────────────────────────────────────────────────────────────

@dataclass
class SeamScore:
    """Результат оценки шва.

    Атрибуты:
        score:          Итоговая оценка ∈ [0, 1] (выше — лучше).
        color_score:    Оценка канала цвета ∈ [0, 1].
        gradient_score: Оценка канала градиента ∈ [0, 1].
        texture_score:  Оценка канала текстуры ∈ [0, 1].
        params:         Дополнительные параметры.
    """
    score:          float
    color_score:    float
    gradient_score: float
    texture_score:  float
    params:         Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [("score", self.score),
                          ("color_score", self.color_score),
                          ("gradient_score", self.gradient_score),
                          ("texture_score", self.texture_score)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Преобразует изображение в float64 grayscale."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return (0.2989 * img[:, :, 0]
                + 0.5870 * img[:, :, 1]
                + 0.1140 * img[:, :, 2])
    raise ValueError(f"img должен быть 2-D или 3-D, получено ndim={img.ndim}")


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Вычисляет магнитуду градиента Собела (простая реализация)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    h, w = gray.shape

    def _conv(img, kernel):
        ph = kernel.shape[0] // 2
        pw = kernel.shape[1] // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
        out = np.zeros_like(img)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                out += kernel[i, j] * padded[i : i + h, j : j + w]
        return out

    gx = _conv(gray, kx)
    gy = _conv(gray, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


# ─── extract_seam_strip ───────────────────────────────────────────────────────

def extract_seam_strip(img: np.ndarray,
                        side: int,
                        width: int = 8) -> np.ndarray:
    """Извлечь полосу пикселей вдоль заданной стороны изображения.

    Аргументы:
        img:   Изображение (2-D или 3-D, uint8).
        side:  Сторона (0=верх, 1=право, 2=низ, 3=лево).
        width: Ширина полосы в пикселях (>= 1).

    Возвращает:
        Массив float32 той же размерности, что и img.

    Исключения:
        ValueError: Если side не в {0, 1, 2, 3} или width < 1.
    """
    if side not in (0, 1, 2, 3):
        raise ValueError(f"side должен быть в {{0,1,2,3}}, получено {side}")
    if width < 1:
        raise ValueError(f"width должен быть >= 1, получено {width}")

    h, w_img = img.shape[:2]
    b = max(1, min(width, min(h, w_img) // 2))

    if side == 0:
        strip = img[:b, :]
    elif side == 1:
        strip = img[:, -b:]
    elif side == 2:
        strip = img[-b:, :]
    else:
        strip = img[:, :b]

    return strip.astype(np.float32)


# ─── color_continuity ─────────────────────────────────────────────────────────

def color_continuity(strip_a: np.ndarray,
                      strip_b: np.ndarray) -> float:
    """Оценить непрерывность цвета между двумя полосами.

    Вычисляет 1 - |mean_a - mean_b| / 255 (по всем пикселям и каналам).

    Аргументы:
        strip_a: Полоса первого фрагмента (float32).
        strip_b: Полоса второго фрагмента (float32).

    Возвращает:
        Оценка ∈ [0, 1].
    """
    if strip_a.size == 0 or strip_b.size == 0:
        return 1.0
    mean_a = float(np.mean(strip_a))
    mean_b = float(np.mean(strip_b))
    return float(np.clip(1.0 - abs(mean_a - mean_b) / 255.0, 0.0, 1.0))


# ─── gradient_continuity ──────────────────────────────────────────────────────

def gradient_continuity(strip_a: np.ndarray,
                         strip_b: np.ndarray) -> float:
    """Оценить непрерывность градиентов на шве.

    Вычисляет сходство средних магнитуд градиента как 1 - |g_a - g_b| / max_g.

    Аргументы:
        strip_a: Полоса первого фрагмента (float32, 2-D или 3-D).
        strip_b: Полоса второго фрагмента (float32, 2-D или 3-D).

    Возвращает:
        Оценка ∈ [0, 1].
    """
    if strip_a.size == 0 or strip_b.size == 0:
        return 1.0

    ga_gray = _to_gray_float(strip_a)
    gb_gray = _to_gray_float(strip_b)

    mag_a = float(np.mean(_sobel_magnitude(ga_gray)))
    mag_b = float(np.mean(_sobel_magnitude(gb_gray)))

    max_mag = max(mag_a, mag_b, 1e-9)
    return float(np.clip(1.0 - abs(mag_a - mag_b) / max_mag, 0.0, 1.0))


# ─── texture_continuity ───────────────────────────────────────────────────────

def texture_continuity(strip_a: np.ndarray,
                        strip_b: np.ndarray) -> float:
    """Оценить непрерывность текстуры между двумя полосами.

    Вычисляет соотношение стандартных отклонений (меньшее / большее).

    Аргументы:
        strip_a: Полоса первого фрагмента (float32).
        strip_b: Полоса второго фрагмента (float32).

    Возвращает:
        Оценка ∈ [0, 1].
    """
    if strip_a.size == 0 or strip_b.size == 0:
        return 1.0

    std_a = float(np.std(strip_a))
    std_b = float(np.std(strip_b))

    if std_a < 1e-9 and std_b < 1e-9:
        return 1.0
    min_std = min(std_a, std_b)
    max_std = max(std_a, std_b)
    if max_std < 1e-9:
        return 1.0
    return float(np.clip(min_std / max_std, 0.0, 1.0))


# ─── evaluate_seam ────────────────────────────────────────────────────────────

def evaluate_seam(img_a:  np.ndarray,
                   side_a: int,
                   img_b:  np.ndarray,
                   side_b: int,
                   cfg:    Optional[SeamConfig] = None) -> SeamScore:
    """Комплексная оценка шва между двумя фрагментами.

    Аргументы:
        img_a:  Первое изображение фрагмента (uint8, 2-D или 3-D).
        side_a: Сторона первого фрагмента (0=верх, 1=право, 2=низ, 3=лево).
        img_b:  Второе изображение фрагмента (uint8, 2-D или 3-D).
        side_b: Сторона второго фрагмента.
        cfg:    Конфигурация (по умолчанию SeamConfig()).

    Возвращает:
        SeamScore с компонентными и итоговой оценками.
    """
    if cfg is None:
        cfg = SeamConfig()

    sa = extract_seam_strip(img_a, side_a, cfg.blend_width)
    sb = extract_seam_strip(img_b, side_b, cfg.blend_width)

    cs = color_continuity(sa, sb)
    gs = gradient_continuity(sa, sb)
    ts = texture_continuity(sa, sb)

    total_w = cfg.total_weight
    if total_w < 1e-9:
        score = 0.0
    else:
        score = (cfg.w_color * cs
                 + cfg.w_gradient * gs
                 + cfg.w_texture * ts) / total_w

    score = float(np.clip(score, 0.0, 1.0))

    return SeamScore(
        score=score,
        color_score=cs,
        gradient_score=gs,
        texture_score=ts,
        params={
            "side_a": side_a,
            "side_b": side_b,
            "blend_width": cfg.blend_width,
        },
    )


# ─── batch_evaluate_seams ─────────────────────────────────────────────────────

def batch_evaluate_seams(
    pairs: List[Tuple[np.ndarray, int, np.ndarray, int]],
    cfg:   Optional[SeamConfig] = None,
) -> List[SeamScore]:
    """Пакетная оценка швов для списка пар фрагментов.

    Аргументы:
        pairs: Список кортежей (img_a, side_a, img_b, side_b).
        cfg:   Конфигурация (по умолчанию SeamConfig()).

    Возвращает:
        Список SeamScore, по одному на пару.
    """
    if cfg is None:
        cfg = SeamConfig()
    return [evaluate_seam(img_a, sa, img_b, sb, cfg)
            for img_a, sa, img_b, sb in pairs]


# ─── rank_seams ───────────────────────────────────────────────────────────────

def rank_seams(
    scores:  List[SeamScore],
    indices: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Ранжировать оценки швов по убыванию.

    Аргументы:
        scores:  Список SeamScore.
        indices: Индексы пар (по умолчанию 0, 1, …, len-1).

    Возвращает:
        Список кортежей (pair_id, score), отсортированный по убыванию score.

    Исключения:
        ValueError: Если lengths indices и scores не совпадают.
    """
    if indices is None:
        indices = list(range(len(scores)))
    if len(indices) != len(scores):
        raise ValueError(
            f"Длины indices ({len(indices)}) и scores ({len(scores)}) "
            f"должны совпадать"
        )
    ranked = sorted(
        zip(indices, [s.score for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(int(idx), float(sc)) for idx, sc in ranked]
