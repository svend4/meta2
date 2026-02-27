"""
Color continuity verification across fragment seams.

Checks that color transitions across fragment boundaries are smooth,
which indicates correctly matched fragments.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ColorContinuityConfig:
    """Configuration for ColorContinuityVerifier."""
    seam_width: int = 3          # Pixels to sample on each side of the seam
    method: str = "lab"          # Color space: "lab", "rgb", "hsv"
    threshold: float = 30.0      # Max acceptable mean delta to consider valid
    weight_spatial: bool = True  # Weight samples by proximity to seam centre


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ColorContinuityResult:
    """Result of color continuity verification across a seam."""
    mean_delta: float   # Mean per-pixel color difference across the seam
    max_delta: float    # Maximum per-pixel color difference across the seam
    score: float        # Overall quality score ∈ [0, 1]
    is_valid: bool      # True when mean_delta < config.threshold
    n_samples: int      # Number of pixel pairs compared


# ---------------------------------------------------------------------------
# Color-space conversion helpers (pure NumPy, no cv2 / scipy)
# ---------------------------------------------------------------------------

def _rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) uint8-or-float RGB array to CIE L*a*b* using the
    standard sRGB → XYZ → D65 → Lab pipeline.

    All arithmetic is done in float64.
    """
    rgb = pixels.astype(np.float64) / 255.0

    # Linearise sRGB (inverse gamma)
    mask = rgb > 0.04045
    rgb[mask]  = ((rgb[mask]  + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # sRGB → XYZ (D65 illuminant)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ M.T   # (N, 3)

    # Normalise by D65 white point
    xyz[:, 0] /= 0.95047
    xyz[:, 1] /= 1.00000
    xyz[:, 2] /= 1.08883

    # XYZ → Lab
    epsilon = 0.008856
    kappa   = 903.3

    def _f(t: np.ndarray) -> np.ndarray:
        out = np.empty_like(t)
        m = t > epsilon
        out[m]  = t[m] ** (1.0 / 3.0)
        out[~m] = (kappa * t[~m] + 16.0) / 116.0
        return out

    fx = _f(xyz[:, 0])
    fy = _f(xyz[:, 1])
    fz = _f(xyz[:, 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.column_stack([L, a, b])


def _rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) uint8-or-float RGB array to HSV.
    H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1].
    """
    rgb = pixels.astype(np.float64) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros(len(rgb))
    m = delta > 0

    mask_r = m & (cmax == r)
    mask_g = m & (cmax == g)
    mask_b = m & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)

    # Value
    v = cmax

    return np.column_stack([h, s, v])


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class ColorContinuityVerifier:
    """
    Verifies that colors on opposite sides of a fragment seam are
    consistent, indicating a correct match.
    """

    def __init__(self, config: Optional[ColorContinuityConfig] = None) -> None:
        self.config = config or ColorContinuityConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_seam(
        self,
        pixels_a: np.ndarray,
        pixels_b: np.ndarray,
    ) -> ColorContinuityResult:
        """
        Verify color continuity between two strips of pixels at a seam.

        Args:
            pixels_a: (N, 3) array of RGB colors on one side of the seam.
            pixels_b: (N, 3) array of RGB colors on the other side.

        Returns:
            ColorContinuityResult.
        """
        pixels_a = np.asarray(pixels_a, dtype=np.float64)
        pixels_b = np.asarray(pixels_b, dtype=np.float64)

        n = min(len(pixels_a), len(pixels_b))
        if n == 0:
            return ColorContinuityResult(
                mean_delta=float("inf"),
                max_delta=float("inf"),
                score=0.0,
                is_valid=False,
                n_samples=0,
            )

        pa = pixels_a[:n]
        pb = pixels_b[:n]

        # Convert to the configured color space
        pa_cs = self._convert_color_space(pa, self.config.method)
        pb_cs = self._convert_color_space(pb, self.config.method)

        # Per-sample Euclidean distance in color space
        deltas = np.linalg.norm(pa_cs - pb_cs, axis=1)   # (n,)

        mean_d = float(deltas.mean())
        max_d  = float(deltas.max())

        score    = self.score_from_delta(mean_d, self.config.threshold)
        is_valid = mean_d < self.config.threshold

        return ColorContinuityResult(
            mean_delta=mean_d,
            max_delta=max_d,
            score=score,
            is_valid=is_valid,
            n_samples=n,
        )

    def _convert_color_space(
        self,
        pixels: np.ndarray,
        space: str,
    ) -> np.ndarray:
        """
        Convert an (N, 3) RGB pixel array to the requested color space.

        Args:
            pixels: (N, 3) float64 array (values 0–255).
            space:  One of "lab", "rgb", "hsv".

        Returns:
            (N, 3) float64 array in the target color space.
        """
        space = space.lower()
        if space == "lab":
            return _rgb_to_lab(pixels)
        if space == "hsv":
            return _rgb_to_hsv(pixels)
        # Default: keep RGB (normalised to [0, 255] for consistent deltas)
        return pixels.astype(np.float64)

    def _color_delta(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Compute the mean Euclidean color difference between two pixel arrays.

        Args:
            a: (N, 3) pixel array in the working color space.
            b: (N, 3) pixel array in the working color space.

        Returns:
            Mean Euclidean distance (float).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size == 0 or b.size == 0:
            return float("inf")
        n = min(len(a), len(b))
        return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())

    @staticmethod
    def score_from_delta(delta: float, threshold: float) -> float:
        """
        Map a color delta to a quality score in [0, 1] using exponential decay.

        score = exp(−delta / threshold)

        Args:
            delta:     Mean color difference.
            threshold: Maximum acceptable delta (config value).

        Returns:
            Float in [0, 1].
        """
        if threshold <= 0:
            return 0.0 if delta > 0 else 1.0
        score = float(np.exp(-delta / threshold))
        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def verify_color_continuity(
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    config: Optional[ColorContinuityConfig] = None,
) -> ColorContinuityResult:
    """
    Verify color continuity between two seam-adjacent pixel strips.

    Args:
        pixels_a: (N, 3) RGB colors on one side of the seam.
        pixels_b: (N, 3) RGB colors on the other side.
        config:   Optional configuration; defaults used if None.

    Returns:
        ColorContinuityResult.
    """
    verifier = ColorContinuityVerifier(config)
    return verifier.verify_seam(pixels_a, pixels_b)
