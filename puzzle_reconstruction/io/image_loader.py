"""Загрузка и базовая предобработка изображений фрагментов.

Модуль предоставляет функции для чтения изображений из файлов и
директорий, конвертации форматов, изменения размера, нормализации
диапазона значений и пакетной загрузки.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── LoadConfig ───────────────────────────────────────────────────────────────

_VALID_COLOR_MODES = frozenset({"gray", "bgr", "rgb"})


@dataclass
class LoadConfig:
    """Параметры загрузки изображений.

    Атрибуты:
        color_mode:   Цветовой режим ('gray', 'bgr', 'rgb').
        target_size:  Целевой размер (w, h) или None (без изменения размера).
        normalize:    Нормировать значения к [0, 1] (dtype float32).
        extensions:   Допустимые расширения файлов.
        params:       Дополнительные параметры.
    """

    color_mode: str = "bgr"
    target_size: Optional[Tuple[int, int]] = None
    normalize: bool = False
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(
                f"Неизвестный color_mode '{self.color_mode}'. "
                f"Допустимые: {sorted(_VALID_COLOR_MODES)}"
            )
        if self.target_size is not None:
            w, h = self.target_size
            if w < 1 or h < 1:
                raise ValueError(
                    f"target_size должен быть (w>=1, h>=1), получено {self.target_size}"
                )


# ─── LoadedImage ──────────────────────────────────────────────────────────────

@dataclass
class LoadedImage:
    """Загруженное изображение с метаданными.

    Атрибуты:
        data:       Массив изображения (uint8 или float32).
        path:       Путь к исходному файлу (или '' если из памяти).
        image_id:   Числовой идентификатор (>= 0).
        original_size: Исходный размер (w, h) до масштабирования.
        color_mode: Цветовой режим.
    """

    data: np.ndarray
    path: str
    image_id: int
    original_size: Tuple[int, int]
    color_mode: str = "bgr"

    def __post_init__(self) -> None:
        if self.image_id < 0:
            raise ValueError(
                f"image_id должен быть >= 0, получено {self.image_id}"
            )
        if self.data.ndim not in (2, 3):
            raise ValueError(
                f"data должен быть 2-D или 3-D, получено ndim={self.data.ndim}"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) загруженного изображения."""
        h, w = self.data.shape[:2]
        return (w, h)

    def __len__(self) -> int:
        return self.data.size


# ─── load_image ───────────────────────────────────────────────────────────────

def load_image(
    path: str,
    cfg: Optional[LoadConfig] = None,
    image_id: int = 0,
) -> LoadedImage:
    """Загрузить одно изображение из файла.

    Аргументы:
        path:     Путь к файлу.
        cfg:      Параметры загрузки (None → LoadConfig()).
        image_id: Числовой идентификатор изображения (>= 0).

    Возвращает:
        LoadedImage.

    Исключения:
        FileNotFoundError: Если файл не существует.
        ValueError: Если файл не удалось прочитать или image_id < 0.
    """
    if cfg is None:
        cfg = LoadConfig()
    if image_id < 0:
        raise ValueError(f"image_id должен быть >= 0, получено {image_id}")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    # Приводим к нужному формату
    if img.ndim == 2:
        original_size = (img.shape[1], img.shape[0])
    else:
        original_size = (img.shape[1], img.shape[0])

    img = _apply_color_mode(img, cfg.color_mode)
    img = _apply_resize(img, cfg.target_size)
    if cfg.normalize:
        img = img.astype(np.float32) / 255.0

    return LoadedImage(
        data=img,
        path=str(p),
        image_id=image_id,
        original_size=original_size,
        color_mode=cfg.color_mode,
    )


# ─── load_from_array ──────────────────────────────────────────────────────────

def load_from_array(
    arr: np.ndarray,
    cfg: Optional[LoadConfig] = None,
    image_id: int = 0,
) -> LoadedImage:
    """Создать LoadedImage из уже загруженного массива.

    Аргументы:
        arr:      Массив изображения (uint8, 2-D или 3-D).
        cfg:      Параметры (None → LoadConfig()).
        image_id: Идентификатор (>= 0).

    Возвращает:
        LoadedImage.

    Исключения:
        ValueError: Если arr некорректен.
    """
    if cfg is None:
        cfg = LoadConfig()
    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"arr должен быть 2-D или 3-D, получено ndim={arr.ndim}"
        )
    if image_id < 0:
        raise ValueError(f"image_id должен быть >= 0, получено {image_id}")

    original_size = (arr.shape[1], arr.shape[0])
    img = _apply_color_mode(arr.astype(np.uint8), cfg.color_mode)
    img = _apply_resize(img, cfg.target_size)
    if cfg.normalize:
        img = img.astype(np.float32) / 255.0

    return LoadedImage(
        data=img,
        path="",
        image_id=image_id,
        original_size=original_size,
        color_mode=cfg.color_mode,
    )


# ─── list_image_files ─────────────────────────────────────────────────────────

def list_image_files(
    directory: str,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    recursive: bool = False,
) -> List[str]:
    """Перечислить файлы изображений в директории.

    Аргументы:
        directory:  Путь к директории.
        extensions: Допустимые расширения файлов (нижний регистр).
        recursive:  Рекурсивный поиск в поддиректориях.

    Возвращает:
        Отсортированный список абсолютных путей.

    Исключения:
        NotADirectoryError: Если путь не является директорией.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Не является директорией: {directory}")

    exts = {e.lower() for e in extensions}
    pattern = "**/*" if recursive else "*"
    files = []
    for f in d.glob(pattern):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(str(f.resolve()))
    return sorted(files)


# ─── batch_load ───────────────────────────────────────────────────────────────

def batch_load(
    paths: List[str],
    cfg: Optional[LoadConfig] = None,
) -> List[LoadedImage]:
    """Загрузить список изображений по путям.

    Аргументы:
        paths: Список путей к файлам.
        cfg:   Параметры загрузки.

    Возвращает:
        Список LoadedImage (в том же порядке).

    Исключения:
        FileNotFoundError: Если хотя бы один файл не найден.
    """
    return [load_image(p, cfg=cfg, image_id=i) for i, p in enumerate(paths)]


# ─── load_from_directory ──────────────────────────────────────────────────────

def load_from_directory(
    directory: str,
    cfg: Optional[LoadConfig] = None,
    recursive: bool = False,
) -> List[LoadedImage]:
    """Загрузить все изображения из директории.

    Аргументы:
        directory: Путь к директории.
        cfg:       Параметры загрузки.
        recursive: Рекурсивный поиск.

    Возвращает:
        Список LoadedImage.

    Исключения:
        NotADirectoryError: Если путь не является директорией.
    """
    if cfg is None:
        cfg = LoadConfig()
    paths = list_image_files(directory, extensions=cfg.extensions,
                             recursive=recursive)
    return batch_load(paths, cfg=cfg)


# ─── resize_image ─────────────────────────────────────────────────────────────

def resize_image(
    img: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """Изменить размер изображения.

    Аргументы:
        img:         Массив изображения.
        target_size: Целевой размер (w, h), оба >= 1.

    Возвращает:
        Изменённое изображение (тот же dtype).

    Исключения:
        ValueError: Если target_size содержит значения < 1.
    """
    w, h = target_size
    if w < 1 or h < 1:
        raise ValueError(
            f"target_size должен быть (w>=1, h>=1), получено {target_size}"
        )
    return cv2.resize(img, (w, h))


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _apply_color_mode(img: np.ndarray, mode: str) -> np.ndarray:
    """Привести изображение к заданному цветовому режиму."""
    if mode == "gray":
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    if mode == "rgb":
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # bgr (default)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _apply_resize(img: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_size is None:
        return img
    return resize_image(img, target_size)
