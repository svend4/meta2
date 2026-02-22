"""Построение финального холста из размещённых фрагментов пазла.

Модуль растеризует набор фрагментов по заданным позициям, выполняет
их склейку на общем холсте и вычисляет статистику покрытия/заполненности.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── CanvasConfig ─────────────────────────────────────────────────────────────

@dataclass
class CanvasConfig:
    """Параметры построения холста.

    Атрибуты:
        bg_color:    Цвет фона (B, G, R) — каждое значение 0–255.
        blend_mode:  'overwrite' | 'average' — режим наложения при перекрытии.
        padding:     Отступ вокруг содержимого (>= 0).
        dtype:       Тип выходного массива ('uint8' | 'float32').
    """

    bg_color: Tuple[int, int, int] = (255, 255, 255)
    blend_mode: str = "overwrite"
    padding: int = 0
    dtype: str = "uint8"

    def __post_init__(self) -> None:
        for i, v in enumerate(self.bg_color):
            if not (0 <= v <= 255):
                raise ValueError(
                    f"bg_color[{i}] должен быть в [0, 255], получено {v}"
                )
        if self.blend_mode not in ("overwrite", "average"):
            raise ValueError(
                f"blend_mode должен быть 'overwrite' или 'average', "
                f"получено '{self.blend_mode}'"
            )
        if self.padding < 0:
            raise ValueError(
                f"padding должен быть >= 0, получено {self.padding}"
            )
        if self.dtype not in ("uint8", "float32"):
            raise ValueError(
                f"dtype должен быть 'uint8' или 'float32', "
                f"получено '{self.dtype}'"
            )


# ─── FragmentPlacement ────────────────────────────────────────────────────────

@dataclass
class FragmentPlacement:
    """Позиционированный фрагмент для вставки на холст.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        image:       Изображение фрагмента (uint8, H×W или H×W×3).
        x:           Левая граница на холсте (>= 0).
        y:           Верхняя граница на холсте (>= 0).
    """

    fragment_id: int
    image: np.ndarray
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.image.ndim not in (2, 3):
            raise ValueError(
                f"image должно быть 2D или 3D, получено ndim={self.image.ndim}"
            )

    @property
    def h(self) -> int:
        """Высота фрагмента."""
        return int(self.image.shape[0])

    @property
    def w(self) -> int:
        """Ширина фрагмента."""
        return int(self.image.shape[1])

    @property
    def x2(self) -> int:
        """Правая граница (не включительно)."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Нижняя граница (не включительно)."""
        return self.y + self.h


# ─── CanvasResult ─────────────────────────────────────────────────────────────

@dataclass
class CanvasResult:
    """Результат построения холста.

    Атрибуты:
        canvas:           Итоговый холст (numpy array).
        coverage:         Доля пикселей холста, заполненных фрагментами [0, 1].
        n_placed:         Число успешно размещённых фрагментов (>= 0).
        canvas_w:         Ширина холста (>= 1).
        canvas_h:         Высота холста (>= 1).
    """

    canvas: np.ndarray
    coverage: float
    n_placed: int
    canvas_w: int
    canvas_h: int

    def __post_init__(self) -> None:
        if self.n_placed < 0:
            raise ValueError(
                f"n_placed должен быть >= 0, получено {self.n_placed}"
            )
        if not (0.0 <= self.coverage <= 1.0 + 1e-9):
            raise ValueError(
                f"coverage должен быть в [0, 1], получено {self.coverage}"
            )
        if self.canvas_w < 1:
            raise ValueError(
                f"canvas_w должен быть >= 1, получено {self.canvas_w}"
            )
        if self.canvas_h < 1:
            raise ValueError(
                f"canvas_h должен быть >= 1, получено {self.canvas_h}"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Форма холста."""
        return self.canvas.shape


# ─── compute_canvas_size ──────────────────────────────────────────────────────

def compute_canvas_size(
    placements: List[FragmentPlacement],
    padding: int = 0,
) -> Tuple[int, int]:
    """Вычислить минимальный размер холста для всех размещений.

    Аргументы:
        placements: Список размещений.
        padding:    Дополнительный отступ (>= 0).

    Возвращает:
        (width, height) холста.

    Исключения:
        ValueError: Если placements пустой или padding < 0.
    """
    if not placements:
        raise ValueError("placements не должен быть пустым")
    if padding < 0:
        raise ValueError(f"padding должен быть >= 0, получено {padding}")

    max_x2 = max(p.x2 for p in placements)
    max_y2 = max(p.y2 for p in placements)
    return (int(max_x2 + padding), int(max_y2 + padding))


# ─── make_empty_canvas ────────────────────────────────────────────────────────

def make_empty_canvas(
    width: int,
    height: int,
    cfg: Optional[CanvasConfig] = None,
) -> np.ndarray:
    """Создать пустой холст заданного размера.

    Аргументы:
        width:  Ширина в пикселях (>= 1).
        height: Высота в пикселях (>= 1).
        cfg:    Параметры (None → CanvasConfig()).

    Возвращает:
        Цветной холст H×W×3, заполненный bg_color.

    Исключения:
        ValueError: Если width < 1 или height < 1.
    """
    if width < 1:
        raise ValueError(f"width должен быть >= 1, получено {width}")
    if height < 1:
        raise ValueError(f"height должен быть >= 1, получено {height}")
    if cfg is None:
        cfg = CanvasConfig()

    canvas = np.full((height, width, 3), cfg.bg_color, dtype=np.uint8)
    if cfg.dtype == "float32":
        canvas = canvas.astype(np.float32) / 255.0
    return canvas


# ─── place_fragment ───────────────────────────────────────────────────────────

def place_fragment(
    canvas: np.ndarray,
    placement: FragmentPlacement,
    blend_mode: str = "overwrite",
) -> np.ndarray:
    """Поместить один фрагмент на холст.

    Аргументы:
        canvas:     Холст (H×W×3, модифицируется in-place).
        placement:  Фрагмент и его позиция.
        blend_mode: 'overwrite' | 'average'.

    Возвращает:
        Тот же холст (модифицированный).

    Исключения:
        ValueError: Если blend_mode неизвестен.
    """
    if blend_mode not in ("overwrite", "average"):
        raise ValueError(
            f"blend_mode должен быть 'overwrite' или 'average', "
            f"получено '{blend_mode}'"
        )

    h_c, w_c = canvas.shape[:2]
    img = placement.image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(canvas.dtype)

    # Clip to canvas bounds
    dst_x1 = max(0, placement.x)
    dst_y1 = max(0, placement.y)
    dst_x2 = min(w_c, placement.x2)
    dst_y2 = min(h_c, placement.y2)

    src_x1 = dst_x1 - placement.x
    src_y1 = dst_y1 - placement.y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas  # Fully outside

    src_patch = img[src_y1:src_y2, src_x1:src_x2]
    dst_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]

    if blend_mode == "average":
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = (
            (dst_region.astype(np.float32) + src_patch.astype(np.float32)) / 2.0
        ).astype(canvas.dtype)
    else:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = src_patch

    return canvas


# ─── build_canvas ─────────────────────────────────────────────────────────────

def build_canvas(
    placements: List[FragmentPlacement],
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    cfg: Optional[CanvasConfig] = None,
) -> CanvasResult:
    """Построить финальный холст из списка размещений.

    Аргументы:
        placements: Список FragmentPlacement.
        canvas_w:   Ширина холста (None → авто).
        canvas_h:   Высота холста (None → авто).
        cfg:        Параметры (None → CanvasConfig()).

    Возвращает:
        CanvasResult.

    Исключения:
        ValueError: Если placements пуст.
    """
    if not placements:
        raise ValueError("placements не должен быть пустым")
    if cfg is None:
        cfg = CanvasConfig()

    if canvas_w is None or canvas_h is None:
        auto_w, auto_h = compute_canvas_size(placements, cfg.padding)
        canvas_w = canvas_w or auto_w
        canvas_h = canvas_h or auto_h

    canvas = make_empty_canvas(canvas_w, canvas_h, cfg)
    coverage_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    n_placed = 0
    for p in placements:
        canvas = place_fragment(canvas, p, cfg.blend_mode)

        # Update coverage mask
        x1 = max(0, p.x)
        y1 = max(0, p.y)
        x2 = min(canvas_w, p.x2)
        y2 = min(canvas_h, p.y2)
        if x2 > x1 and y2 > y1:
            coverage_mask[y1:y2, x1:x2] = 1
            n_placed += 1

    canvas_area = canvas_w * canvas_h
    coverage = float(coverage_mask.sum()) / float(canvas_area) if canvas_area > 0 else 0.0

    return CanvasResult(
        canvas=canvas,
        coverage=coverage,
        n_placed=n_placed,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )


# ─── crop_to_content ──────────────────────────────────────────────────────────

def crop_to_content(
    result: CanvasResult,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Обрезать холст по границам содержимого (не-фонового цвета).

    Аргументы:
        result:   CanvasResult.
        bg_color: Цвет фона для определения маски содержимого.

    Возвращает:
        Обрезанный холст (numpy array).
    """
    canvas = result.canvas
    if canvas.ndim == 2:
        mask = canvas != bg_color[0]
    else:
        mask = np.any(
            canvas != np.array(bg_color, dtype=canvas.dtype),
            axis=2,
        )

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return canvas  # Entirely background

    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    r_min, r_max = int(row_idx[0]), int(row_idx[-1])
    c_min, c_max = int(col_idx[0]), int(col_idx[-1])
    return canvas[r_min: r_max + 1, c_min: c_max + 1]


# ─── batch_build_canvases ─────────────────────────────────────────────────────

def batch_build_canvases(
    placement_lists: List[List[FragmentPlacement]],
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    cfg: Optional[CanvasConfig] = None,
) -> List[CanvasResult]:
    """Построить несколько холстов из списка наборов размещений.

    Аргументы:
        placement_lists: Список наборов FragmentPlacement.
        canvas_w:        Общая ширина (None → авто для каждого).
        canvas_h:        Общая высота (None → авто для каждого).
        cfg:             Параметры.

    Возвращает:
        Список CanvasResult.
    """
    return [
        build_canvas(pl, canvas_w, canvas_h, cfg)
        for pl in placement_lists
    ]
