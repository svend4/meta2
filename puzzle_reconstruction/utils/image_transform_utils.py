"""
Утилиты геометрических преобразований изображений.

Geometric image transformation utilities: rotation, reflection, perspective
correction, padding, and batch helpers used in preprocessing and alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ImageTransformConfig:
    """Parameters controlling image transformation operations."""

    border_value: int = 255
    """Fill value for areas outside the original image after rotation."""

    interpolation: int = cv2.INTER_LINEAR
    """OpenCV interpolation flag for warp operations."""

    expand: bool = False
    """If True, expand the canvas to fit the rotated content fully."""

    def __post_init__(self) -> None:
        if not (0 <= self.border_value <= 255):
            raise ValueError("border_value must be in [0, 255]")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """Result of an image transform operation."""

    image: np.ndarray
    angle_rad: float
    scale: float
    translation: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "shape": list(self.image.shape),
            "angle_deg": math.degrees(self.angle_rad),
            "scale": self.scale,
            "translation": list(self.translation),
        }


# ---------------------------------------------------------------------------
# Core transform functions
# ---------------------------------------------------------------------------

def rotate_image(
    image: np.ndarray,
    angle_rad: float,
    cfg: Optional[ImageTransformConfig] = None,
) -> np.ndarray:
    """
    Rotate an image by *angle_rad* radians around its centre.

    Parameters
    ----------
    image     : (H, W) or (H, W, C) uint8 array.
    angle_rad : rotation angle in radians (counter-clockwise positive).
    cfg       : optional :class:`ImageTransformConfig`.

    Returns
    -------
    ndarray — rotated image with same shape as input.
    """
    if cfg is None:
        cfg = ImageTransformConfig()
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    angle_deg = math.degrees(angle_rad)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    border = (
        (cfg.border_value, cfg.border_value, cfg.border_value)
        if image.ndim == 3
        else cfg.border_value
    )
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cfg.interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Flip image horizontally (left ↔ right)."""
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """Flip image vertically (top ↔ bottom)."""
    return cv2.flip(image, 0)


def pad_image(
    image: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    fill: int = 255,
) -> np.ndarray:
    """
    Add padding around an image.

    Parameters
    ----------
    image                : input image.
    top, bottom, left, right : padding in pixels per side.
    fill                 : fill value (0–255).

    Returns
    -------
    padded image.
    """
    if image.ndim == 3:
        border_val = (fill, fill, fill)
    else:
        border_val = fill
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=border_val,
    )


def crop_image(
    image: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> np.ndarray:
    """
    Crop a region from *image*.

    Parameters
    ----------
    image : input image.
    y0, x0 : top-left corner (inclusive).
    y1, x1 : bottom-right corner (exclusive).

    Returns
    -------
    Cropped image as a numpy array.
    """
    h, w = image.shape[:2]
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(h, y1)
    x1 = min(w, x1)
    return image[y0:y1, x0:x1].copy()


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize image to *target_size* = (width, height).

    Returns
    -------
    Resized image as uint8 array.
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def resize_to_max_side(
    image: np.ndarray,
    max_side: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize image so its longest side equals *max_side*, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return image.copy()
    scale = max_side / long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


# ---------------------------------------------------------------------------
# Affine / perspective helpers
# ---------------------------------------------------------------------------

def apply_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    cfg: Optional[ImageTransformConfig] = None,
) -> np.ndarray:
    """
    Apply a 2×3 affine transform *matrix* to *image*.

    Parameters
    ----------
    image  : input image.
    matrix : (2, 3) float32 affine matrix.
    cfg    : optional config.

    Returns
    -------
    Transformed image, same spatial size as input.
    """
    if cfg is None:
        cfg = ImageTransformConfig()
    h, w = image.shape[:2]
    border = (
        (cfg.border_value,) * 3 if image.ndim == 3 else cfg.border_value
    )
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cfg.interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def rotation_matrix_2x3(
    angle_rad: float,
    cx: float,
    cy: float,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Build a 2×3 OpenCV rotation matrix.

    Parameters
    ----------
    angle_rad : rotation angle in radians.
    cx, cy    : rotation centre.
    scale     : uniform scale factor.

    Returns
    -------
    (2, 3) float64 matrix.
    """
    angle_deg = math.degrees(angle_rad)
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_rotate(
    images: List[np.ndarray],
    angle_rad: float,
    cfg: Optional[ImageTransformConfig] = None,
) -> List[np.ndarray]:
    """Rotate a list of images by the same angle."""
    return [rotate_image(img, angle_rad, cfg) for img in images]


def batch_pad(
    images: List[np.ndarray],
    pad: int,
    fill: int = 255,
) -> List[np.ndarray]:
    """Pad all images in *images* uniformly on all sides."""
    return [pad_image(img, pad, pad, pad, pad, fill) for img in images]


def batch_resize_to_max(
    images: List[np.ndarray],
    max_side: int,
) -> List[np.ndarray]:
    """Resize all images so their longest side equals *max_side*."""
    return [resize_to_max_side(img, max_side) for img in images]
