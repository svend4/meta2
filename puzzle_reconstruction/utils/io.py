"""
Утилиты ввода-вывода для коллекций фрагментов и сборок.

Функции:
    load_image_dir       — загрузка изображений из директории (jpg, png, tiff …)
    fragments_from_images — создание минимальных Fragment из изображений
    save_assembly_json   — сериализация Assembly → JSON (placements + scores)
    load_assembly_json   — десериализация JSON → Assembly
    save_fragments_npz   — сохранение изображений + масок в .npz
    load_fragments_npz   — загрузка из .npz → List[Fragment]
    FragmentSetInfo      — мета-информация о наборе фрагментов

Все функции используют стандартные библиотеки (pathlib, json, numpy)
без дополнительных зависимостей, кроме OpenCV для работы с изображениями.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..models import Assembly, Fragment

logger = logging.getLogger(__name__)

# Поддерживаемые расширения изображений
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ─── Метаданные набора ────────────────────────────────────────────────────────

@dataclass
class FragmentSetInfo:
    """
    Информация о загруженном наборе фрагментов.

    Attributes:
        n_fragments:    Число загруженных фрагментов.
        source_dir:     Путь к директории-источнику (если применимо).
        image_sizes:    Список размеров (H, W) для каждого фрагмента.
        total_pixels:   Суммарное число пикселей.
        failed_paths:   Пути к файлам, которые не удалось загрузить.
    """
    n_fragments:   int
    source_dir:    Optional[str]    = None
    image_sizes:   List[Tuple[int, int]] = field(default_factory=list)
    total_pixels:  int              = 0
    failed_paths:  List[str]        = field(default_factory=list)

    def summary(self) -> str:
        return (f"FragmentSetInfo(n={self.n_fragments}, "
                f"failed={len(self.failed_paths)}, "
                f"total_px={self.total_pixels:,})")


# ─── Загрузка изображений ─────────────────────────────────────────────────────

def load_image_dir(path:          str,
                    extensions:   Optional[set] = None,
                    sort:         bool = True,
                    recursive:    bool = False,
                    max_images:   int  = 0) -> Tuple[List[np.ndarray], FragmentSetInfo]:
    """
    Загружает все изображения из директории как BGR uint8.

    Args:
        path:       Путь к директории.
        extensions: Множество допустимых расширений (None → IMAGE_EXTENSIONS).
        sort:       True → сортировать по имени файла.
        recursive:  True → обходить поддиректории.
        max_images: Максимальное число изображений (0 = без ограничений).

    Returns:
        (images: List[np.ndarray], info: FragmentSetInfo)
    """
    exts = extensions or IMAGE_EXTENSIONS
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Директория не найдена: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Не директория: {path}")

    # Собираем файлы
    if recursive:
        candidates = [p for p in root.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in root.iterdir() if p.is_file()]

    image_paths = [p for p in candidates if p.suffix.lower() in exts]
    if sort:
        image_paths.sort(key=lambda p: p.name)
    if max_images > 0:
        image_paths = image_paths[:max_images]

    images:       List[np.ndarray] = []
    failed:       List[str]        = []
    image_sizes:  List[Tuple[int, int]] = []
    total_pixels: int              = 0

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Не удалось загрузить: %s", p)
            failed.append(str(p))
            continue
        images.append(img)
        h, w = img.shape[:2]
        image_sizes.append((h, w))
        total_pixels += h * w

    info = FragmentSetInfo(
        n_fragments=len(images),
        source_dir=str(root),
        image_sizes=image_sizes,
        total_pixels=total_pixels,
        failed_paths=failed,
    )
    return images, info


def fragments_from_images(images:       List[np.ndarray],
                           start_id:    int = 0,
                           auto_mask:   bool = True) -> List[Fragment]:
    """
    Создаёт минимальные объекты Fragment из списка изображений.

    Маска: белый фон (255) отсекается, если auto_mask=True (порог Otsu).

    Args:
        images:    Список BGR uint8 изображений.
        start_id:  Начальный fragment_id (инкрементируется).
        auto_mask: True → автоматически строить маску (Otsu).

    Returns:
        List[Fragment] с заполненными image, mask, contour.
        Дескрипторы (tangram, fractal, edges) не вычисляются.
    """
    fragments: List[Fragment] = []

    for i, img in enumerate(images):
        fid = start_id + i

        if auto_mask:
            mask = _compute_otsu_mask(img)
        else:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        contour = _extract_main_contour(mask)

        frag = Fragment(
            fragment_id=fid,
            image=img,
            mask=mask,
            contour=contour,
        )
        fragments.append(frag)

    return fragments


# ─── Сериализация Assembly ────────────────────────────────────────────────────

def save_assembly_json(assembly: Assembly,
                        path:    str,
                        indent:  int = 2) -> None:
    """
    Сохраняет Assembly в JSON (только placements и scores, без пикселей).

    Формат:
        {
          "total_score": 0.87,
          "ocr_score":   0.72,
          "n_fragments": 6,
          "placements": {
            "0": {"position": [x, y], "angle_rad": 0.0},
            ...
          }
        }

    Args:
        assembly: Assembly для сохранения.
        path:     Путь к выходному JSON-файлу.
        indent:   Отступ JSON (2 = читабельный).
    """
    placements_serial: Dict[str, dict] = {}
    for fid, (pos, angle) in assembly.placements.items():
        placements_serial[str(fid)] = {
            "position":  [float(pos[0]), float(pos[1])],
            "angle_rad": float(angle),
        }

    data = {
        "total_score":  float(assembly.total_score),
        "ocr_score":    float(assembly.ocr_score),
        "n_fragments":  len(assembly.fragments),
        "placements":   placements_serial,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.info("Assembly сохранена: %s (%d фрагментов)", out, len(placements_serial))


def load_assembly_json(path:      str,
                        fragments: List[Fragment]) -> Assembly:
    """
    Восстанавливает Assembly из JSON-файла.

    Args:
        path:      Путь к JSON-файлу (сохранённому save_assembly_json).
        fragments: Список Fragment (для поля Assembly.fragments).

    Returns:
        Assembly с заполненными placements, total_score, ocr_score.

    Raises:
        FileNotFoundError: Если файл не найден.
        KeyError:          Если JSON не содержит обязательных полей.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    placements: Dict[int, Tuple[np.ndarray, float]] = {}
    for str_fid, entry in data["placements"].items():
        fid   = int(str_fid)
        pos   = np.array(entry["position"], dtype=np.float64)
        angle = float(entry["angle_rad"])
        placements[fid] = (pos, angle)

    return Assembly(
        fragments=fragments,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=float(data.get("total_score", 0.0)),
        ocr_score=float(data.get("ocr_score", 0.0)),
    )


# ─── Сохранение/загрузка фрагментов (.npz) ───────────────────────────────────

def save_fragments_npz(fragments: List[Fragment],
                        path:     str) -> None:
    """
    Сохраняет изображения и маски фрагментов в бинарный файл .npz.

    Формат .npz:
        images[i]   — BGR uint8 (H_i, W_i, 3)
        masks[i]    — uint8 (H_i, W_i)
        contours[i] — float64 (N_i, 2)
        fragment_ids — int (n_fragments,)

    Args:
        fragments: Список Fragment.
        path:      Путь к выходному .npz-файлу (расширение добавляется если нет).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    images_list:   List[np.ndarray] = []
    masks_list:    List[np.ndarray] = []
    contours_list: List[np.ndarray] = []
    ids_list:      List[int]        = []

    for frag in fragments:
        images_list.append(frag.image)
        masks_list.append(frag.mask)
        contours_list.append(frag.contour.astype(np.float64))
        ids_list.append(frag.fragment_id)

    # Сохраняем как object-массив (изображения могут иметь разный размер)
    images_arr   = np.empty(len(fragments), dtype=object)
    masks_arr    = np.empty(len(fragments), dtype=object)
    contours_arr = np.empty(len(fragments), dtype=object)

    for i, (img, mask, contour) in enumerate(
            zip(images_list, masks_list, contours_list)):
        images_arr[i]   = img
        masks_arr[i]    = mask
        contours_arr[i] = contour

    np.savez_compressed(
        str(out),
        fragment_ids=np.array(ids_list, dtype=np.int64),
        images=images_arr,
        masks=masks_arr,
        contours=contours_arr,
    )

    logger.info("Фрагменты сохранены: %s.npz (%d шт.)", out, len(fragments))


def load_fragments_npz(path: str) -> List[Fragment]:
    """
    Загружает фрагменты из .npz-файла (сохранённого save_fragments_npz).

    Args:
        path: Путь к .npz-файлу.

    Returns:
        List[Fragment] с заполненными image, mask, contour, fragment_id.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    p = Path(path)
    if not p.exists() and not p.with_suffix(".npz").exists():
        raise FileNotFoundError(f"NPZ-файл не найден: {path}")

    data = np.load(str(p), allow_pickle=True)

    fragment_ids: np.ndarray = data["fragment_ids"]
    images_arr:   np.ndarray = data["images"]
    masks_arr:    np.ndarray = data["masks"]
    contours_arr: np.ndarray = data["contours"]

    fragments: List[Fragment] = []
    for i, fid in enumerate(fragment_ids):
        frag = Fragment(
            fragment_id=int(fid),
            image=images_arr[i],
            mask=masks_arr[i],
            contour=contours_arr[i],
        )
        fragments.append(frag)

    logger.info("Загружено фрагментов: %d из %s", len(fragments), p)
    return fragments


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _compute_otsu_mask(image: np.ndarray) -> np.ndarray:
    """Строит маску переднего плана методом Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Морфологическое закрытие для заполнения дыр
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)


def _extract_main_contour(mask: np.ndarray) -> np.ndarray:
    """Извлекает самый большой контур из бинарной маски."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze().astype(np.float64)


def _iter_image_files(directory: Path,
                       extensions: set,
                       recursive: bool) -> Iterator[Path]:
    """Итерирует файлы изображений в директории."""
    glob_fn = directory.rglob if recursive else directory.glob
    for p in sorted(glob_fn("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            yield p
