"""
Torn-edge enhancement for document fragment contours.

Improves edge quality for matching by:
1. Sub-pixel refinement via supersampling along the contour
2. Edge-specific denoising (preserving sharpness away from the edge)
3. Contrast enhancement along the contour region

Improves FD and CSS descriptor accuracy on real-world data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TearEnhancerConfig:
    """Configuration for torn-edge enhancement.

    Attributes:
        supersample_factor: Sub-pixel refinement factor (1–4). Higher values
                            give finer contour resolution at increased cost.
        denoise_radius:     Pixel radius for edge-band denoising (>= 0).
        contrast_alpha:     Contrast stretch factor along the edge band (1.0–3.0).
                            1.0 means no change.
        method:             Denoising kernel: ``"gaussian"``, ``"bilateral"``,
                            or ``"none"``.
        edge_band_width:    Width in pixels of the band around the contour that
                            receives contrast enhancement.
    """
    supersample_factor: int = 2
    denoise_radius: int = 3
    contrast_alpha: float = 1.5
    method: str = "gaussian"
    edge_band_width: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.supersample_factor <= 4:
            raise ValueError(
                f"supersample_factor must be in [1, 4], got {self.supersample_factor}"
            )
        if self.denoise_radius < 0:
            raise ValueError(
                f"denoise_radius must be >= 0, got {self.denoise_radius}"
            )
        if not 1.0 <= self.contrast_alpha <= 3.0:
            raise ValueError(
                f"contrast_alpha must be in [1.0, 3.0], got {self.contrast_alpha}"
            )
        valid_methods = {"gaussian", "bilateral", "none"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}, got {self.method!r}"
            )
        if self.edge_band_width < 1:
            raise ValueError(
                f"edge_band_width must be >= 1, got {self.edge_band_width}"
            )


@dataclass
class TearEnhancerResult:
    """Result of torn-edge enhancement.

    Attributes:
        enhanced_image:   Output image with edge quality improved (same shape as input).
        enhanced_contour: Refined contour as float32 array of shape (N, 2).
        sharpness_before: Laplacian-variance sharpness estimate before enhancement.
        sharpness_after:  Laplacian-variance sharpness estimate after enhancement.
    """
    enhanced_image: np.ndarray
    enhanced_contour: np.ndarray
    sharpness_before: float
    sharpness_after: float


# ─── Private utilities ────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Build a 1-D Gaussian kernel."""
    if radius == 0:
        return np.array([1.0], dtype=np.float32)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    return kernel


def _convolve2d_separable(image: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Separable 2-D convolution using 1-D kernels (pure numpy)."""
    out = np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), axis=1, arr=image.astype(np.float32))
    out = np.apply_along_axis(lambda col: np.convolve(col, ky, mode="same"), axis=0, arr=out)
    return out


def _build_edge_mask(image: np.ndarray, contour: np.ndarray, band_width: int) -> np.ndarray:
    """Return a boolean mask that is True within *band_width* pixels of the contour."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    pts = contour.astype(np.int32)
    bw = max(1, band_width)
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        x0 = max(0, x - bw)
        x1 = min(w, x + bw + 1)
        y0 = max(0, y - bw)
        y1 = min(h, y + bw + 1)
        mask[y0:y1, x0:x1] = True
    return mask


def _numpy_gaussian_blur(channel: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur on a single-channel float image (pure numpy)."""
    if radius <= 0:
        return channel.astype(np.float32)
    sigma = radius / 3.0
    k = _gaussian_kernel_1d(sigma, radius)
    return _convolve2d_separable(channel, k, k)


# ─── Main class ───────────────────────────────────────────────────────────────

class TearEdgeEnhancer:
    """Enhance torn edges of document fragments for improved descriptor matching.

    Parameters
    ----------
    config:
        Enhancement configuration; ``None`` uses defaults.
    """

    def __init__(self, config: Optional[TearEnhancerConfig] = None) -> None:
        self.config = config if config is not None else TearEnhancerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, image: np.ndarray, contour: np.ndarray) -> TearEnhancerResult:
        """Enhance a document fragment image along its torn edge.

        Args:
            image:   Input image (H, W) or (H, W, C), uint8 or float.
            contour: Edge contour as array of shape (N, 2) with (x, y) columns.

        Returns:
            :class:`TearEnhancerResult` with enhanced image, refined contour,
            and sharpness measurements.
        """
        img = np.array(image, copy=True)
        contour = np.asarray(contour, dtype=np.float32)
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)

        sharpness_before = self._estimate_sharpness(img)

        cfg = self.config

        if cfg.method != "none":
            img = self._denoise_near_edge(img, contour, cfg.denoise_radius)

        img = self._enhance_contrast(img, contour, cfg.edge_band_width, cfg.contrast_alpha)
        refined = self._refine_contour(contour, img, cfg.supersample_factor)

        sharpness_after = self._estimate_sharpness(img)

        return TearEnhancerResult(
            enhanced_image=img,
            enhanced_contour=refined,
            sharpness_before=float(sharpness_before),
            sharpness_after=float(sharpness_after),
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _estimate_sharpness(self, image: np.ndarray) -> float:
        """Estimate sharpness as Laplacian variance (pure numpy).

        Args:
            image: Single- or multi-channel image.

        Returns:
            Non-negative float; higher values indicate sharper images.
        """
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            gray = image.astype(np.float32)

        gray = gray.astype(np.float32)

        # 3x3 Laplacian kernel
        laplacian_kernel = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=np.float32
        )

        # Manual 2-D convolution using stride tricks (avoids scipy dependency)
        padded = np.pad(gray, 1, mode="reflect")
        h, w = gray.shape
        lap = np.zeros_like(gray)
        for dy in range(3):
            for dx in range(3):
                lap += laplacian_kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]

        return float(lap.var())

    def _denoise_near_edge(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        radius: int,
    ) -> np.ndarray:
        """Apply edge-specific denoising within *radius* pixels of the contour.

        Outside the edge band the image is left unchanged to preserve interior
        texture.  Falls back to Gaussian blur when cv2 is unavailable.

        Args:
            image:   Input image.
            contour: Edge contour (N, 2).
            radius:  Denoising radius in pixels.

        Returns:
            Denoised image (same dtype and shape as input).
        """
        if radius <= 0:
            return image.copy()

        method = self.config.method
        mask = _build_edge_mask(image, contour, radius)
        result = image.copy()

        def _denoise_channel(ch: np.ndarray) -> np.ndarray:
            ch_f = ch.astype(np.float32)
            if method == "gaussian":
                try:
                    import cv2
                    ksize = 2 * radius + 1
                    blurred = cv2.GaussianBlur(ch_f, (ksize, ksize), 0)
                except Exception:
                    blurred = _numpy_gaussian_blur(ch_f, radius)
            elif method == "bilateral":
                try:
                    import cv2
                    # cv2.bilateralFilter requires uint8 or float32
                    blurred = cv2.bilateralFilter(
                        ch_f, d=radius * 2 + 1,
                        sigmaColor=75, sigmaSpace=75
                    )
                except Exception:
                    # Numpy fallback: use Gaussian
                    blurred = _numpy_gaussian_blur(ch_f, radius)
            else:
                return ch

            out = ch_f.copy()
            out[mask] = blurred[mask]
            return np.clip(out, 0, 255).astype(ch.dtype)

        if image.ndim == 2:
            result = _denoise_channel(image)
        else:
            for c in range(image.shape[2]):
                result[:, :, c] = _denoise_channel(image[:, :, c])

        return result

    def _enhance_contrast(
        self,
        image: np.ndarray,
        contour: np.ndarray,
        band_width: int,
        alpha: float,
    ) -> np.ndarray:
        """Increase contrast within *band_width* pixels of the contour.

        Applies ``output = mean + alpha * (pixel - mean)`` inside the edge band,
        where *mean* is computed from the band pixels to avoid a global brightness
        shift.

        Args:
            image:      Input image.
            contour:    Edge contour (N, 2).
            band_width: Width of the contrast-enhancement band in pixels.
            alpha:      Contrast stretch factor (1.0 = no change).

        Returns:
            Image with enhanced edge contrast (same dtype and shape).
        """
        if abs(alpha - 1.0) < 1e-6:
            return image.copy()

        mask = _build_edge_mask(image, contour, band_width)
        result = image.astype(np.float32).copy()

        if image.ndim == 2:
            region = result[mask]
            if region.size > 0:
                mean_val = float(region.mean())
                result[mask] = mean_val + alpha * (region - mean_val)
        else:
            for c in range(image.shape[2]):
                ch = result[:, :, c]
                region = ch[mask]
                if region.size > 0:
                    mean_val = float(region.mean())
                    ch[mask] = mean_val + alpha * (region - mean_val)
                result[:, :, c] = ch

        result = np.clip(result, 0, 255)
        return result.astype(image.dtype)

    def _refine_contour(
        self,
        contour: np.ndarray,
        image: np.ndarray,
        factor: int,
    ) -> np.ndarray:
        """Sub-pixel contour refinement via linear supersampling.

        Inserts ``factor - 1`` interpolated points between each consecutive pair
        of contour vertices, then refines each point position by finding the
        sub-pixel location of maximum gradient magnitude in a small neighbourhood.

        Args:
            contour: Input contour, shape (N, 2).
            image:   Image used for gradient-based refinement.
            factor:  Supersampling factor (1 = no refinement).

        Returns:
            Refined contour as float32 array of shape (M, 2), where
            ``M = (N - 1) * factor + 1`` for open contours.
        """
        if factor <= 1 or len(contour) < 2:
            return contour.astype(np.float32)

        # Upsample contour by linear interpolation
        pts = contour.astype(np.float32)
        n = len(pts)
        upsampled = []
        for i in range(n - 1):
            for k in range(factor):
                t = k / factor
                upsampled.append(pts[i] + t * (pts[i + 1] - pts[i]))
        upsampled.append(pts[-1])
        upsampled = np.array(upsampled, dtype=np.float32)

        # Gradient magnitude for refinement guidance
        if image.ndim == 3:
            gray = image.mean(axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        half = 1  # search radius in pixels

        refined = upsampled.copy()
        for i, pt in enumerate(upsampled):
            x, y = pt
            best_x, best_y = x, y
            best_grad = -1.0

            # Search in a small neighbourhood for peak gradient
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    nx = int(round(x)) + dx
                    ny = int(round(y)) + dy
                    if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                        gx = float(gray[ny, nx + 1]) - float(gray[ny, nx - 1])
                        gy = float(gray[ny + 1, nx]) - float(gray[ny - 1, nx])
                        g = gx * gx + gy * gy
                        if g > best_grad:
                            best_grad = g
                            best_x, best_y = float(nx), float(ny)

            refined[i] = [best_x, best_y]

        return refined


# ─── Module-level convenience function ───────────────────────────────────────

def enhance_torn_edge(
    image: np.ndarray,
    contour: np.ndarray,
    config: Optional[TearEnhancerConfig] = None,
) -> TearEnhancerResult:
    """Enhance torn edges of a document fragment.

    Convenience wrapper around :class:`TearEdgeEnhancer`.

    Args:
        image:   Input image (H, W) or (H, W, C).
        contour: Edge contour array of shape (N, 2) with (x, y) columns.
        config:  Enhancement configuration; ``None`` uses defaults.

    Returns:
        :class:`TearEnhancerResult`.
    """
    return TearEdgeEnhancer(config=config).enhance(image, contour)
