"""Rendering and canvas-composition utility functions.

Provides low-level helpers for compositing fragment images onto a canvas,
generating thumbnails, computing canvas geometry, and saving output images.
Used by the export pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CanvasConfig:
    """Parameters for canvas rendering."""
    bg_color: Tuple[int, int, int] = (240, 240, 240)
    margin: int = 20
    dtype: type = np.uint8


@dataclass
class MosaicConfig:
    """Parameters for mosaic / thumbnail-grid rendering."""
    thumb_size: int = 64
    max_cols: int = 8
    gap: int = 8
    bg_color: Tuple[int, int, int] = (200, 200, 200)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    """2×2 rotation matrix for *angle_rad*."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def bounding_box_of_rotated(w: int, h: int, angle_rad: float) -> Tuple[int, int]:
    """Return (new_w, new_h) of the axis-aligned bounding box after rotation."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    R = rotation_matrix_2d(angle_rad)
    rotated = corners @ R.T
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    new_w = int(math.ceil(max_xy[0] - min_xy[0]))
    new_h = int(math.ceil(max_xy[1] - min_xy[1]))
    return new_w, new_h


def compute_canvas_size(
    placements: dict,   # {frag_id: (position_xy, angle_rad)}
    frag_sizes: dict,   # {frag_id: (w, h)}
    margin: int = 20,
) -> Tuple[int, int]:
    """Compute the minimum canvas size to contain all placed fragments.

    Parameters
    ----------
    placements : dict  frag_id → (np.ndarray([x, y]), angle_rad)
    frag_sizes : dict  frag_id → (width, height) in pixels
    margin     : int   padding added on all sides

    Returns
    -------
    (canvas_w, canvas_h) in pixels
    """
    if not placements:
        return 2 * margin or 100, 2 * margin or 100

    xs, ys = [], []
    for fid, (pos, angle) in placements.items():
        if fid not in frag_sizes:
            continue
        fw, fh = frag_sizes[fid]
        rw, rh = bounding_box_of_rotated(fw, fh, angle)
        xs.extend([pos[0], pos[0] + rw])
        ys.extend([pos[1], pos[1] + rh])

    if not xs:
        return 2 * margin or 100, 2 * margin or 100

    canvas_w = int(math.ceil(max(xs) - min(xs))) + 2 * margin
    canvas_h = int(math.ceil(max(ys) - min(ys))) + 2 * margin
    return max(canvas_w, 1), max(canvas_h, 1)


# ─── Image helpers ────────────────────────────────────────────────────────────

def make_blank_canvas(
    width: int,
    height: int,
    color: Tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Create a blank BGR canvas filled with *color*."""
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas


def resize_keep_aspect(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize *image* so the longer side equals *target_size* (keeps aspect ratio)."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image.copy()
    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def pad_to_square(image: np.ndarray,
                   size: int,
                   fill: int = 200) -> np.ndarray:
    """Place *image* in the top-left corner of a *size × size* canvas."""
    if image.ndim == 3:
        canvas = np.full((size, size, image.shape[2]), fill, dtype=np.uint8)
    else:
        canvas = np.full((size, size), fill, dtype=np.uint8)
    h = min(image.shape[0], size)
    w = min(image.shape[1], size)
    canvas[:h, :w] = image[:h, :w]
    return canvas


def make_thumbnail(image: np.ndarray, thumb_size: int = 64) -> np.ndarray:
    """Return a square thumbnail of *image* with side *thumb_size*."""
    resized = resize_keep_aspect(image, thumb_size)
    return pad_to_square(resized, thumb_size)


def paste_with_mask(
    canvas: np.ndarray,
    fragment: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Paste *fragment* onto *canvas* at (x, y) using *mask* (uint8 0/255).

    Returns the modified canvas (in-place modification + return).
    """
    fh, fw = fragment.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clamp to canvas bounds
    x0, y0 = max(x, 0), max(y, 0)
    x1 = min(x + fw, cw)
    y1 = min(y + fh, ch)
    if x1 <= x0 or y1 <= y0:
        return canvas

    fx0, fy0 = x0 - x, y0 - y
    fx1, fy1 = fx0 + (x1 - x0), fy0 + (y1 - y0)

    roi = canvas[y0:y1, x0:x1]
    frag_roi = fragment[fy0:fy1, fx0:fx1]
    mask_roi = mask[fy0:fy1, fx0:fx1]

    alpha = (mask_roi / 255.0).astype(np.float32)
    if frag_roi.ndim == 3 and roi.ndim == 3:
        alpha = alpha[:, :, np.newaxis]

    canvas[y0:y1, x0:x1] = (
        alpha * frag_roi.astype(np.float32) +
        (1 - alpha) * roi.astype(np.float32)
    ).astype(np.uint8)
    return canvas


# ─── Mosaic layout ────────────────────────────────────────────────────────────

def compute_grid_layout(
    n: int,
    max_cols: int = 8,
) -> Tuple[int, int]:
    """Return (n_rows, n_cols) for a grid of *n* items."""
    if n == 0:
        return 0, 0
    n_cols = min(n, max_cols)
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


def make_mosaic(
    images: List[np.ndarray],
    cfg: Optional[MosaicConfig] = None,
) -> np.ndarray:
    """Lay out *images* in a thumbnail grid.

    Parameters
    ----------
    images : list of np.ndarray  – source images (any size / channels)
    cfg    : MosaicConfig

    Returns
    -------
    np.ndarray  – BGR mosaic image
    """
    cfg = cfg or MosaicConfig()
    if not images:
        s = cfg.thumb_size
        return make_blank_canvas(s, s, cfg.bg_color)

    thumbs = [make_thumbnail(img, cfg.thumb_size) for img in images]
    # Ensure all are BGR
    bgr_thumbs = []
    for t in thumbs:
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        bgr_thumbs.append(t)

    n_rows, n_cols = compute_grid_layout(len(bgr_thumbs), cfg.max_cols)
    cell = cfg.thumb_size + cfg.gap
    total_w = n_cols * cell + cfg.gap
    total_h = n_rows * cell + cfg.gap
    mosaic = make_blank_canvas(total_w, total_h, cfg.bg_color)

    for idx, thumb in enumerate(bgr_thumbs):
        row, col = divmod(idx, n_cols)
        x = cfg.gap + col * cell
        y = cfg.gap + row * cell
        h, w = thumb.shape[:2]
        mosaic[y:y + h, x:x + w] = thumb

    return mosaic


# ─── Output helpers ───────────────────────────────────────────────────────────

def save_image(image: np.ndarray, path, quality: int = 95) -> None:
    """Save *image* to *path* (PNG or JPEG based on extension).

    Parameters
    ----------
    image   : np.ndarray  – BGR or grayscale
    path    : str | Path
    quality : int         – JPEG quality (ignored for PNG)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image)


def horizontal_concat(images: List[np.ndarray], gap: int = 8,
                       bg: Tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """Concatenate images horizontally, all resized to the same height."""
    if not images:
        return make_blank_canvas(10, 10, bg)
    target_h = max(img.shape[0] for img in images)
    strips = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            scale = target_h / h
            nw = max(1, int(round(w * scale)))
            img = cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        strips.append(img)
        if gap > 0:
            sep = make_blank_canvas(gap, target_h, bg)
            strips.append(sep)
    if gap > 0 and strips:
        strips = strips[:-1]  # remove trailing gap
    return np.concatenate(strips, axis=1) if strips else make_blank_canvas(10, target_h, bg)
