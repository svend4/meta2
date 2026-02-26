"""
Illumination equalization across multiple document fragments.

Brings all fragments to a common illumination baseline so that
cross-fragment color comparisons are meaningful.

Methods:
    histogram  — match histograms to a reference fragment
    retinex    — Multi-Scale Retinex normalization per fragment
    clahe      — Contrast Limited Adaptive Histogram Equalization per fragment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class IlluminationEqualizerConfig:
    """Configuration for illumination equalization.

    Attributes:
        method:          Equalization method: ``"histogram"``, ``"retinex"``,
                         or ``"clahe"``.
        reference_idx:   Index of the reference fragment used by the
                         ``"histogram"`` method.
        retinex_scales:  Gaussian sigma values for Multi-Scale Retinex.
        clahe_clip:      Clip limit for CLAHE (only used when cv2 is available).
        clahe_grid:      Tile grid size for CLAHE as ``(rows, cols)``.
    """
    method: str = "histogram"
    reference_idx: int = 0
    retinex_scales: List[float] = field(default_factory=lambda: [15.0, 80.0, 250.0])
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        valid_methods = {"histogram", "retinex", "clahe"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}, got {self.method!r}"
            )
        if self.reference_idx < 0:
            raise ValueError(
                f"reference_idx must be >= 0, got {self.reference_idx}"
            )
        if not self.retinex_scales:
            raise ValueError("retinex_scales must not be empty")
        if self.clahe_clip <= 0:
            raise ValueError(f"clahe_clip must be > 0, got {self.clahe_clip}")


@dataclass
class IlluminationEqualizerResult:
    """Result of illumination equalization.

    Attributes:
        images:            List of equalized images in the same order as input.
        uniformity_scores: Per-image uniformity score in [0, 1].  Higher values
                           indicate more uniform illumination.
        method:            The equalization method that was applied.
    """
    images: List[np.ndarray]
    uniformity_scores: List[float]
    method: str


# ─── Private utilities ────────────────────────────────────────────────────────

def _ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _to_float32(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)


def _numpy_gaussian_blur_1ch(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Single-channel Gaussian blur (pure numpy, separable)."""
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    out = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1,
        arr=channel.astype(np.float32)
    )
    out = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=out
    )
    return out


# ─── Main class ───────────────────────────────────────────────────────────────

class IlluminationEqualizer:
    """Equalize illumination across a collection of document fragment images.

    Parameters
    ----------
    config:
        Equalization configuration; ``None`` uses defaults.
    """

    def __init__(self, config: Optional[IlluminationEqualizerConfig] = None) -> None:
        self.config = config if config is not None else IlluminationEqualizerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def equalize(self, images: List[np.ndarray]) -> IlluminationEqualizerResult:
        """Equalize illumination across a list of fragment images.

        Args:
            images: List of images (each (H, W) or (H, W, C), uint8 or float).

        Returns:
            :class:`IlluminationEqualizerResult` with equalized images and
            uniformity scores.

        Raises:
            ValueError: If *images* is empty or *reference_idx* is out of range.
        """
        if not images:
            raise ValueError("images must contain at least one image")

        cfg = self.config
        method = cfg.method

        if method == "histogram":
            ref_idx = cfg.reference_idx
            if ref_idx >= len(images):
                raise ValueError(
                    f"reference_idx {ref_idx} out of range for {len(images)} images"
                )
            reference = images[ref_idx]
            out_images = []
            for i, img in enumerate(images):
                if i == ref_idx:
                    out_images.append(img.copy())
                else:
                    out_images.append(self._match_histogram(img, reference))
        elif method == "retinex":
            out_images = [self._apply_retinex(img) for img in images]
        elif method == "clahe":
            out_images = [self._apply_clahe(img) for img in images]
        else:
            raise ValueError(f"Unknown method: {method!r}")

        uniformity_scores = [self._uniformity_score(img) for img in out_images]

        return IlluminationEqualizerResult(
            images=out_images,
            uniformity_scores=uniformity_scores,
            method=method,
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _match_histogram(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Match the histogram of *source* to that of *reference* (pure numpy).

        Operates channel-by-channel for colour images.

        Args:
            source:    Source image to be transformed.
            reference: Reference image whose histogram will be matched.

        Returns:
            Transformed image with the same shape and dtype as *source*.
        """
        def _match_channel(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
            src_f = src_ch.astype(np.float32).ravel()
            ref_f = ref_ch.astype(np.float32).ravel()

            # CDFs
            src_hist, src_bins = np.histogram(src_f, bins=256, range=(0, 256))
            ref_hist, ref_bins = np.histogram(ref_f, bins=256, range=(0, 256))

            src_cdf = np.cumsum(src_hist).astype(np.float64)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)

            # Normalise to [0, 1]
            src_cdf /= max(src_cdf[-1], 1.0)
            ref_cdf /= max(ref_cdf[-1], 1.0)

            # For each source level, find closest matching reference level
            lut = np.zeros(256, dtype=np.float32)
            for src_val in range(256):
                diff = np.abs(ref_cdf - src_cdf[src_val])
                lut[src_val] = float(np.argmin(diff))

            mapped = lut[np.clip(src_ch, 0, 255).astype(np.uint8)]
            return np.clip(mapped, 0, 255).astype(src_ch.dtype)

        if source.ndim == 2:
            # If reference is 3D, convert to gray
            ref_gray = reference if reference.ndim == 2 else reference.mean(axis=2)
            return _match_channel(source, ref_gray)

        # Colour image — match each channel independently
        result = source.copy()
        n_ch = source.shape[2]
        for c in range(n_ch):
            ref_ch = reference[:, :, c] if reference.ndim == 3 and reference.shape[2] > c else reference.mean(axis=2) if reference.ndim == 3 else reference
            result[:, :, c] = _match_channel(source[:, :, c], ref_ch)
        return result

    def _apply_retinex(self, image: np.ndarray) -> np.ndarray:
        """Apply Multi-Scale Retinex normalisation (pure numpy).

        Computes ``log(I) - mean_sigma(log(GaussBlur_sigma(I)))`` for each
        configured sigma, then normalises back to [0, 255].

        Args:
            image: Input image (H, W) or (H, W, C), uint8.

        Returns:
            Retinex-corrected uint8 image of same shape.
        """
        scales = self.config.retinex_scales

        def _retinex_channel(ch: np.ndarray) -> np.ndarray:
            ch_f = ch.astype(np.float32) + 1.0  # avoid log(0)
            log_img = np.log(ch_f)
            retinex = np.zeros_like(log_img)
            for sigma in scales:
                try:
                    import cv2
                    k = _ensure_odd(int(sigma * 6) + 1)
                    blurred = cv2.GaussianBlur(ch_f, (k, k), sigmaX=float(sigma))
                except Exception:
                    blurred = _numpy_gaussian_blur_1ch(ch_f, sigma)
                retinex += log_img - np.log(blurred.astype(np.float32) + 1.0)
            retinex /= len(scales)

            r_min, r_max = retinex.min(), retinex.max()
            if r_max - r_min > 1e-6:
                retinex = (retinex - r_min) / (r_max - r_min) * 255.0
            else:
                retinex = np.full_like(retinex, 128.0)
            return np.clip(retinex, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return _retinex_channel(image)

        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _retinex_channel(image[:, :, c])
        return result

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE per channel; falls back to a numpy approximation if cv2 is absent.

        Args:
            image: Input image (H, W) or (H, W, C), uint8.

        Returns:
            CLAHE-equalized uint8 image.
        """
        cfg = self.config

        def _clahe_channel(ch: np.ndarray) -> np.ndarray:
            try:
                import cv2
                clahe_obj = cv2.createCLAHE(
                    clipLimit=float(cfg.clahe_clip),
                    tileGridSize=tuple(cfg.clahe_grid),
                )
                return clahe_obj.apply(ch.astype(np.uint8))
            except Exception:
                pass
            # Numpy fallback: tile-based histogram equalization with clip
            return self._numpy_clahe(ch, cfg.clahe_clip, cfg.clahe_grid)

        if image.ndim == 2:
            return _clahe_channel(image)

        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _clahe_channel(image[:, :, c])
        return result

    def _numpy_clahe(
        self,
        channel: np.ndarray,
        clip_limit: float,
        grid: Tuple[int, int],
    ) -> np.ndarray:
        """Tile-based CLAHE approximation using pure numpy.

        Divides the image into tiles, clips and equalises each tile's histogram,
        then bilinearly interpolates between tile responses.

        Args:
            channel:    Single-channel uint8 image.
            clip_limit: Normalised clip limit (relative to average histogram value).
            grid:       ``(rows, cols)`` tile grid.

        Returns:
            Equalized uint8 channel.
        """
        h, w = channel.shape
        grid_r, grid_c = max(1, grid[0]), max(1, grid[1])
        tile_h = max(1, h // grid_r)
        tile_w = max(1, w // grid_c)

        ch_u8 = channel.astype(np.uint8)
        luts = {}  # (tr, tc) -> LUT array of length 256

        for tr in range(grid_r):
            for tc in range(grid_c):
                y0, y1 = tr * tile_h, min(h, (tr + 1) * tile_h)
                x0, x1 = tc * tile_w, min(w, (tc + 1) * tile_w)
                tile = ch_u8[y0:y1, x0:x1]
                hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))
                n_pixels = hist.sum()
                clip_val = max(1, int(clip_limit * n_pixels / 256))
                excess = hist - clip_val
                excess[excess < 0] = 0
                hist = np.clip(hist, 0, clip_val)
                hist += excess.sum() // 256
                cdf = np.cumsum(hist).astype(np.float64)
                cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
                cdf_max = max(cdf[-1], 1)
                lut = np.clip(
                    np.round((cdf - cdf_min) / (cdf_max - cdf_min) * 255), 0, 255
                ).astype(np.uint8)
                luts[(tr, tc)] = lut

        # Apply LUTs with bilinear interpolation between tile centres
        result = np.zeros_like(ch_u8, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # Find surrounding tile indices and fractional positions
                fx = (x - tile_w / 2.0) / max(tile_w, 1)
                fy = (y - tile_h / 2.0) / max(tile_h, 1)
                tc0 = int(np.floor(fx))
                tr0 = int(np.floor(fy))
                tc1 = tc0 + 1
                tr1 = tr0 + 1
                dx = fx - tc0
                dy = fy - tr0

                def get_lut_val(tr_: int, tc_: int, pix: int) -> float:
                    tr_ = max(0, min(grid_r - 1, tr_))
                    tc_ = max(0, min(grid_c - 1, tc_))
                    return float(luts[(tr_, tc_)][pix])

                pix = int(ch_u8[y, x])
                v00 = get_lut_val(tr0, tc0, pix)
                v01 = get_lut_val(tr0, tc1, pix)
                v10 = get_lut_val(tr1, tc0, pix)
                v11 = get_lut_val(tr1, tc1, pix)
                result[y, x] = (
                    v00 * (1 - dy) * (1 - dx)
                    + v01 * (1 - dy) * dx
                    + v10 * dy * (1 - dx)
                    + v11 * dy * dx
                )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _uniformity_score(self, image: np.ndarray) -> float:
        """Compute an illumination uniformity score in [0, 1].

        Estimates the background field via strong Gaussian blurring and returns
        ``1 - coefficient_of_variation`` of that field.  A score of 1 means
        perfectly uniform illumination.

        Args:
            image: Image to evaluate.

        Returns:
            Float in [0, 1].
        """
        if image.ndim == 3:
            gray = image.mean(axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Approximate background by heavy blurring
        sigma = max(5.0, min(gray.shape) / 6.0)
        try:
            import cv2
            k = _ensure_odd(int(sigma * 6) + 1)
            bg = cv2.GaussianBlur(gray, (k, k), sigmaX=float(sigma))
        except Exception:
            bg = _numpy_gaussian_blur_1ch(gray, sigma)

        mean_bg = float(bg.mean())
        if mean_bg < 1.0:
            return 1.0
        cv = float(bg.std()) / mean_bg  # coefficient of variation
        return float(np.clip(1.0 - cv, 0.0, 1.0))


# ─── Module-level convenience function ───────────────────────────────────────

def equalize_fragments(
    images: List[np.ndarray],
    method: str = "histogram",
    config: Optional[IlluminationEqualizerConfig] = None,
) -> List[np.ndarray]:
    """Equalize illumination across a list of document fragment images.

    Convenience wrapper around :class:`IlluminationEqualizer`.

    Args:
        images: List of images to equalize.
        method: Equalization method (used only when *config* is ``None``).
        config: Full configuration; if provided, *method* is ignored.

    Returns:
        List of equalized images in the same order as input.
    """
    if config is None:
        config = IlluminationEqualizerConfig(method=method)
    return IlluminationEqualizer(config=config).equalize(images).images
