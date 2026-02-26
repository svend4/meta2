"""
Multi-scale image segmentation with voting for robust fragment detection.

Segments a document fragment image at multiple resolution scales,
then combines the results via majority voting. More robust than
single-scale methods for uneven backgrounds and varying illumination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MultiscaleConfig:
    """Configuration for multi-scale segmentation.

    Attributes:
        scales:         List of scale factors to use (1.0 = original size).
        vote_threshold: Fraction of scales that must agree to label a pixel as
                        foreground (0.0–1.0).
        method:         Thresholding method: ``"otsu"``, ``"adaptive"``, or
                        ``"triangle"``.
        min_area:       Minimum connected-component area in pixels.  Components
                        smaller than this are removed from the final mask.
        smooth_final:   If ``True``, apply morphological smoothing to the final
                        mask.
    """
    scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    vote_threshold: float = 0.5
    method: str = "otsu"
    min_area: int = 100
    smooth_final: bool = True

    def __post_init__(self) -> None:
        if not self.scales:
            raise ValueError("scales must contain at least one value")
        valid_methods = {"otsu", "adaptive", "triangle"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}, got {self.method!r}"
            )
        if not 0.0 <= self.vote_threshold <= 1.0:
            raise ValueError(
                f"vote_threshold must be in [0.0, 1.0], got {self.vote_threshold}"
            )
        if self.min_area < 0:
            raise ValueError(f"min_area must be >= 0, got {self.min_area}")


@dataclass
class MultiscaleSegmentationResult:
    """Result of multi-scale segmentation.

    Attributes:
        mask:            Boolean foreground mask, same spatial shape as input.
        confidence_map:  Per-pixel confidence in [0, 1]: fraction of scales that
                         voted foreground.
        n_scales_used:   Number of scales that were actually processed.
        scales:          The scale values used.
    """
    mask: np.ndarray          # bool, shape (H, W)
    confidence_map: np.ndarray  # float32, shape (H, W), values in [0, 1]
    n_scales_used: int
    scales: List[float]


# ─── Private utilities ────────────────────────────────────────────────────────

def _resize_bilinear(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize a 2-D array to (new_h, new_w) via bilinear interpolation (numpy)."""
    h, w = image.shape
    if h == new_h and w == new_w:
        return image.copy()
    row_idx = np.linspace(0, h - 1, new_h)
    col_idx = np.linspace(0, w - 1, new_w)
    r0 = np.floor(row_idx).astype(int).clip(0, h - 1)
    r1 = (r0 + 1).clip(0, h - 1)
    c0 = np.floor(col_idx).astype(int).clip(0, w - 1)
    c1 = (c0 + 1).clip(0, w - 1)
    dr = (row_idx - r0)[:, np.newaxis].astype(np.float32)
    dc = (col_idx - c0)[np.newaxis, :].astype(np.float32)
    img_f = image.astype(np.float32)
    return (
        img_f[r0][:, c0] * (1 - dr) * (1 - dc)
        + img_f[r1][:, c0] * dr * (1 - dc)
        + img_f[r0][:, c1] * (1 - dr) * dc
        + img_f[r1][:, c1] * dr * dc
    )


def _morph_erode_dilate(mask: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    """Minimal morphological erosion/dilation on a bool mask (numpy)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = radius
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant",
                    constant_values=0 if dilate else 1)
    size = 2 * radius + 1
    try:
        windows = sliding_window_view(padded, (size, size))
        if dilate:
            return windows.max(axis=(-2, -1)) > 0
        else:
            return windows.min(axis=(-2, -1)) > 0
    except Exception:
        # Fallback for very small images or old numpy
        result = np.zeros_like(mask)
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                y0, y1 = max(0, y - radius), min(h, y + radius + 1)
                x0, x1 = max(0, x - radius), min(w, x + radius + 1)
                region = mask[y0:y1, x0:x1]
                if dilate:
                    result[y, x] = region.any()
                else:
                    result[y, x] = region.all()
        return result


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels (pure numpy)."""
    if min_area <= 0:
        return mask.copy()
    # Simple flood-fill based connected components (4-connectivity)
    labeled = np.zeros_like(mask, dtype=np.int32)
    current_label = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labeled[y, x] == 0:
                current_label += 1
                # BFS
                queue = [(y, x)]
                labeled[y, x] = current_label
                while queue:
                    cy, cx = queue.pop()
                    for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = current_label
                            queue.append((ny, nx))

    result = np.zeros_like(mask)
    for lbl in range(1, current_label + 1):
        comp = labeled == lbl
        if comp.sum() >= min_area:
            result |= comp
    return result


# ─── Main class ───────────────────────────────────────────────────────────────

class MultiscaleSegmenter:
    """Segment document fragment images using multi-scale voting.

    Parameters
    ----------
    config:
        Segmentation configuration; ``None`` uses defaults.
    """

    def __init__(self, config: Optional[MultiscaleConfig] = None) -> None:
        self.config = config if config is not None else MultiscaleConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, image: np.ndarray) -> MultiscaleSegmentationResult:
        """Segment a document fragment image.

        Args:
            image: Input image (H, W) grayscale or (H, W, C) colour.

        Returns:
            :class:`MultiscaleSegmentationResult` with boolean mask and
            confidence map.
        """
        gray = self._to_gray(image)
        original_shape = gray.shape  # (H, W)

        cfg = self.config
        masks_at_scale: List[np.ndarray] = []  # each (H, W) bool

        for scale in cfg.scales:
            try:
                mask = self._segment_at_scale(gray, scale)
                resized = self._resize_mask(mask, original_shape)
                masks_at_scale.append(resized)
            except Exception:
                # Skip failing scales rather than crashing
                pass

        n_used = len(masks_at_scale)
        if n_used == 0:
            # Last-resort: return empty mask
            return MultiscaleSegmentationResult(
                mask=np.zeros(original_shape, dtype=bool),
                confidence_map=np.zeros(original_shape, dtype=np.float32),
                n_scales_used=0,
                scales=list(cfg.scales),
            )

        confidence = self._vote(masks_at_scale, original_shape)
        mask = confidence >= cfg.vote_threshold

        if cfg.smooth_final:
            try:
                dilated = _morph_erode_dilate(mask, radius=1, dilate=True)
                mask = _morph_erode_dilate(dilated, radius=1, dilate=False)
            except Exception:
                pass

        if cfg.min_area > 0:
            try:
                mask = _remove_small_components(mask, cfg.min_area)
            except Exception:
                pass

        return MultiscaleSegmentationResult(
            mask=mask.astype(bool),
            confidence_map=confidence.astype(np.float32),
            n_scales_used=n_used,
            scales=list(cfg.scales),
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to single-channel grayscale.

        Args:
            image: (H, W) or (H, W, C) array.

        Returns:
            uint8 grayscale array of shape (H, W).
        """
        if image.ndim == 2:
            return image.astype(np.uint8)
        # Weighted RGB→gray (ITU-R BT.601 coefficients)
        if image.shape[2] >= 3:
            gray = (
                0.299 * image[:, :, 0].astype(np.float32)
                + 0.587 * image[:, :, 1].astype(np.float32)
                + 0.114 * image[:, :, 2].astype(np.float32)
            )
        else:
            gray = image[:, :, 0].astype(np.float32)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _segment_at_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Segment a grayscale image at a given resolution scale.

        Args:
            image: Grayscale uint8 array (H, W).
            scale: Scale factor; 1.0 = original size.

        Returns:
            Boolean foreground mask at the scaled resolution.
        """
        h, w = image.shape
        if scale != 1.0:
            new_h = max(4, int(round(h * scale)))
            new_w = max(4, int(round(w * scale)))
            scaled = _resize_bilinear(image.astype(np.float32), new_h, new_w)
            gray = np.clip(scaled, 0, 255).astype(np.uint8)
        else:
            gray = image

        method = self.config.method
        if method == "otsu":
            return self._threshold_otsu(gray)
        elif method == "adaptive":
            return self._threshold_adaptive(gray)
        elif method == "triangle":
            return self._threshold_triangle(gray)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    def _threshold_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Pure-numpy Otsu's thresholding.

        Args:
            gray: Grayscale uint8 array.

        Returns:
            Boolean mask (True = foreground).
        """
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return np.zeros_like(gray, dtype=bool)

        # Probability of each intensity
        p = hist / total
        # Cumulative sums
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256, dtype=np.float64))
        mu_total = mu[-1]

        # Between-class variance for each threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_b2 = np.where(
                (omega > 0) & (omega < 1),
                (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega)),
                0.0,
            )

        threshold = int(np.argmax(sigma_b2))
        return gray > threshold

    def _threshold_adaptive(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding with local mean (numpy fallback or cv2).

        Args:
            gray: Grayscale uint8 array.

        Returns:
            Boolean mask (True = foreground).
        """
        try:
            import cv2
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=max(3, min(51, (min(gray.shape) // 4) | 1)),
                C=10,
            )
            return binary > 0
        except Exception:
            pass

        # Numpy fallback: local mean thresholding with block size 15
        block = max(3, min(15, (min(gray.shape) // 4) | 1))
        from numpy.lib.stride_tricks import sliding_window_view
        pad = block // 2
        padded = np.pad(gray.astype(np.float32), pad, mode="reflect")
        try:
            windows = sliding_window_view(padded, (block, block))
            local_mean = windows.mean(axis=(-2, -1))
        except Exception:
            local_mean = np.full_like(gray, float(gray.mean()), dtype=np.float32)
        return gray.astype(np.float32) < (local_mean - 10)

    def _threshold_triangle(self, gray: np.ndarray) -> np.ndarray:
        """Triangle (Zack) thresholding — works well for skewed histograms.

        Args:
            gray: Grayscale uint8 array.

        Returns:
            Boolean mask (True = foreground).
        """
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)

        # Find peak of histogram
        peak_idx = int(np.argmax(hist))
        peak_val = hist[peak_idx]

        # Determine search direction: go toward the sparser tail
        if peak_idx < 128:
            # Right side is the tail
            end_idx = 255
            while end_idx > peak_idx and hist[end_idx] == 0:
                end_idx -= 1
            # Line from peak to end
            d_idx = end_idx - peak_idx
            d_val = 0.0 - peak_val
            best_dist = -1.0
            best_t = peak_idx
            for i in range(peak_idx, end_idx + 1):
                t = i - peak_idx
                # Perpendicular distance from point (i, hist[i]) to line
                line_y = peak_val + d_val * (t / max(d_idx, 1))
                dist = abs(hist[i] - line_y)
                if dist > best_dist:
                    best_dist = dist
                    best_t = i
            threshold = best_t
        else:
            # Left side is the tail
            start_idx = 0
            while start_idx < peak_idx and hist[start_idx] == 0:
                start_idx += 1
            d_idx = peak_idx - start_idx
            d_val = peak_val - 0.0
            best_dist = -1.0
            best_t = peak_idx
            for i in range(start_idx, peak_idx + 1):
                t = i - start_idx
                line_y = d_val * (t / max(d_idx, 1))
                dist = abs(hist[i] - line_y)
                if dist > best_dist:
                    best_dist = dist
                    best_t = i
            threshold = best_t

        return gray > threshold

    def _vote(
        self,
        masks: List[np.ndarray],
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Combine per-scale masks into a confidence map via majority voting.

        Args:
            masks:          List of boolean masks, each of shape *original_shape*.
            original_shape: Target spatial shape (H, W).

        Returns:
            Float32 confidence map in [0, 1], shape *original_shape*.
        """
        if not masks:
            return np.zeros(original_shape, dtype=np.float32)
        stack = np.stack([m.astype(np.float32) for m in masks], axis=0)
        return stack.mean(axis=0).astype(np.float32)

    def _resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize a boolean mask to *target_shape* using nearest-neighbour.

        Args:
            mask:         Boolean mask of any shape.
            target_shape: (H, W) target shape.

        Returns:
            Boolean mask of shape *target_shape*.
        """
        h, w = mask.shape
        th, tw = target_shape
        if h == th and w == tw:
            return mask.copy()
        row_idx = np.round(np.linspace(0, h - 1, th)).astype(int).clip(0, h - 1)
        col_idx = np.round(np.linspace(0, w - 1, tw)).astype(int).clip(0, w - 1)
        return mask[row_idx][:, col_idx]


# ─── Module-level convenience function ───────────────────────────────────────

def segment_multiscale(
    image: np.ndarray,
    config: Optional[MultiscaleConfig] = None,
) -> MultiscaleSegmentationResult:
    """Segment a document fragment image using multi-scale voting.

    Convenience wrapper around :class:`MultiscaleSegmenter`.

    Args:
        image:  Input image (H, W) or (H, W, C).
        config: Segmentation configuration; ``None`` uses defaults.

    Returns:
        :class:`MultiscaleSegmentationResult`.
    """
    return MultiscaleSegmenter(config=config).segment(image)
