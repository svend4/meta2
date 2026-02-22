"""
Утилиты загрузки и сохранения изображений фрагментов документа.

Поддерживает пакетную загрузку из директорий, определение формата по
расширению, изменение размера с сохранением пропорций и конвенции
именования фрагментов (fragment_NNN.ext → целочисленный ID).

Классы:
    ImageRecord — описание загруженного изображения (путь, массив, мета)

Функции:
    load_image          — загрузка одного изображения (с проверкой)
    save_image          — сохранение с автосозданием директорий
    load_directory      — пакетная загрузка всех изображений в папке
    filter_by_extension — фильтрация списка путей по расширению
    parse_fragment_id   — извлечение целочисленного ID из имени файла
    resize_to_max       — уменьшение до максимальной стороны (aspect-safe)
    batch_resize        — пакетный resize
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Поддерживаемые форматы ───────────────────────────────────────────────────

_SUPPORTED_EXT: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
)


# ─── ImageRecord ──────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    """
    Контейнер загруженного изображения.

    Attributes:
        path:  Абсолютный путь к файлу.
        image: Пиксели (BGR или grayscale, uint8).
        meta:  Произвольные поля (fragment_id, source и т.п.).
    """
    path:  str
    image: np.ndarray
    meta:  Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.image.shape)

    def __repr__(self) -> str:
        name = os.path.basename(self.path)
        h, w = self.image.shape[:2]
        return f"ImageRecord(name={name!r}, shape=({h}×{w}))"


# ─── load_image ───────────────────────────────────────────────────────────────

def load_image(path: str,
               flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Загружает одно изображение с OpenCV.

    Args:
        path:  Путь к файлу изображения.
        flags: Флаги cv2.imread (по умолчанию IMREAD_COLOR → BGR).

    Returns:
        Массив numpy uint8 в формате BGR или grayscale.

    Raises:
        FileNotFoundError: Если файл не существует.
        IOError:           Если OpenCV не смог декодировать файл.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path!r}")
    img = cv2.imread(path, flags)
    if img is None:
        raise IOError(f"Failed to load image: {path!r}")
    return img


# ─── save_image ───────────────────────────────────────────────────────────────

def save_image(path:  str,
               img:   np.ndarray,
               mkdir: bool = True) -> bool:
    """
    Сохраняет изображение на диск.

    Args:
        path:  Путь назначения (включая имя файла и расширение).
        img:   Массив numpy uint8 (BGR или grayscale).
        mkdir: Создать директории при необходимости (по умолчанию True).

    Returns:
        True при успехе, False если cv2.imwrite вернул False.
    """
    if mkdir:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return bool(cv2.imwrite(path, img))


# ─── filter_by_extension ──────────────────────────────────────────────────────

def filter_by_extension(paths:      List[str],
                          extensions: Tuple[str, ...] = _SUPPORTED_EXT) -> List[str]:
    """
    Возвращает только пути с указанными расширениями (без учёта регистра).

    Args:
        paths:      Список путей.
        extensions: Кортеж расширений (с точкой, например '.png').

    Returns:
        Отфильтрованный список путей.
    """
    exts_lower = {e.lower() for e in extensions}
    return [p for p in paths
            if os.path.splitext(p)[1].lower() in exts_lower]


# ─── parse_fragment_id ────────────────────────────────────────────────────────

_ID_RE = re.compile(r"(\d+)")


def parse_fragment_id(filename: str) -> Optional[int]:
    """
    Извлекает целочисленный ID из имени файла.

    Правило: берётся последовательность цифр, ближайшая к концу имени
    (без расширения). Например:
        'fragment_042.png' → 42
        'img003.jpg'       → 3
        'scan.png'         → None

    Args:
        filename: Имя файла или полный путь.

    Returns:
        Целое число или None, если цифр не найдено.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    matches = _ID_RE.findall(base)
    if not matches:
        return None
    return int(matches[-1])


# ─── load_directory ───────────────────────────────────────────────────────────

def load_directory(directory:  str,
                    extensions: Tuple[str, ...] = _SUPPORTED_EXT,
                    flags:      int = cv2.IMREAD_COLOR,
                    sort:       bool = True) -> List[ImageRecord]:
    """
    Загружает все поддерживаемые изображения из директории.

    Файлы с ошибками загрузки пропускаются (без исключений).

    Args:
        directory:  Путь к директории.
        extensions: Принимаемые расширения.
        flags:      Флаги cv2.imread.
        sort:       Сортировать по имени файла (по умолчанию True).

    Returns:
        Список ImageRecord, отсортированных по имени файла.

    Raises:
        NotADirectoryError: Если путь не является директорией.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory!r}")

    names = os.listdir(directory)
    if sort:
        names = sorted(names)

    paths    = filter_by_extension(
        [os.path.join(directory, n) for n in names], extensions
    )
    records: List[ImageRecord] = []
    for p in paths:
        try:
            img = cv2.imread(p, flags)
            if img is None:
                continue
            fid  = parse_fragment_id(p)
            meta = {"fragment_id": fid} if fid is not None else {}
            records.append(ImageRecord(path=p, image=img, meta=meta))
        except Exception:
            continue

    return records


# ─── resize_to_max ────────────────────────────────────────────────────────────

def resize_to_max(img:      np.ndarray,
                   max_side: int = 1024) -> np.ndarray:
    """
    Уменьшает изображение так, чтобы max(h, w) ≤ max_side.

    Сохраняет пропорции; если изображение уже меньше, возвращает копию.

    Args:
        img:      BGR или grayscale изображение (uint8).
        max_side: Максимальный размер длинной стороны (пикс).

    Returns:
        Изменённое (или исходное) изображение uint8.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img.copy()
    scale  = max_side / longest
    new_w  = max(1, int(round(w * scale)))
    new_h  = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─── batch_resize ─────────────────────────────────────────────────────────────

def batch_resize(images:   List[np.ndarray],
                  max_side: int = 1024) -> List[np.ndarray]:
    """
    Пакетное изменение размера списка изображений.

    Args:
        images:   Список BGR или grayscale изображений.
        max_side: Максимальный размер длинной стороны.

    Returns:
        Список изменённых изображений той же длины.
    """
    return [resize_to_max(img, max_side=max_side) for img in images]
