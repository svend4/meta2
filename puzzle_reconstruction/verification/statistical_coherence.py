"""
Statistical coherence verification for fragment assemblies.

Checks that adjacent fragments share compatible statistical properties
(brightness distribution, texture statistics, noise level).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StatisticalCoherenceConfig:
    """Configuration for StatisticalCoherenceVerifier."""
    n_bins: int = 32             # Number of histogram bins per channel
    method: str = "histogram"    # "histogram", "moments", or "both"
    threshold: float = 0.3       # Minimum overall_score to consider coherent
    use_texture: bool = True     # Include texture statistics in overall score


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StatisticalCoherenceResult:
    """Result of statistical coherence verification between two patches."""
    histogram_similarity: float  # Bhattacharyya coefficient ∈ [0, 1]
    moment_similarity: float     # Similarity based on mean/std/skew ∈ [0, 1]
    texture_similarity: float    # GLCM-inspired texture similarity ∈ [0, 1]
    overall_score: float         # Weighted combination ∈ [0, 1]
    is_coherent: bool            # True when overall_score ≥ config.threshold


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _to_gray_flat(arr: np.ndarray) -> np.ndarray:
    """
    Flatten an array to a 1-D float64 grayscale signal in [0, 255].

    Accepts:
      - 1-D arrays  (already flat)
      - 2-D arrays  (H × W grayscale)
      - 3-D arrays  (H × W × C color) — averaged over channels
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.ravel()
    if arr.ndim == 3:
        return arr.mean(axis=2).ravel()
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _bhattacharyya_coefficient(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
) -> float:
    """
    Compute the Bhattacharyya coefficient between two normalised histograms.

    BC = Σ sqrt(p_i * q_i)

    Returns a value in [0, 1] where 1 means identical distributions.
    """
    sum_a = float(hist_a.sum())
    sum_b = float(hist_b.sum())
    if sum_a == 0 or sum_b == 0:
        return 0.0
    p = hist_a / sum_a
    q = hist_b / sum_b
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.clip(bc, 0.0, 1.0))


def _skewness(x: np.ndarray) -> float:
    """Compute the (bias-corrected) skewness of a 1-D array."""
    n = len(x)
    if n < 3:
        return 0.0
    mu  = float(x.mean())
    std = float(x.std())
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def _glcm_contrast(gray_8bit: np.ndarray, d: int = 1) -> float:
    """
    Estimate a GLCM-like *contrast* statistic without cv2/scipy.

    Uses the mean squared difference of horizontally adjacent pixel pairs
    as a lightweight proxy for the GLCM contrast feature.

    Args:
        gray_8bit: 2-D uint8-like array (values in [0, 255]).
        d:         Pixel displacement (number of steps).

    Returns:
        Normalised contrast value in [0, 1].
    """
    arr = np.asarray(gray_8bit, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= d:
        # Fall back: treat 1-D flat array
        flat = arr.ravel()
        if len(flat) <= d:
            return 0.0
        diffs = flat[d:].astype(np.float64) - flat[:-d].astype(np.float64)
        contrast = float(np.mean(diffs ** 2))
        return float(np.clip(contrast / (255.0 ** 2), 0.0, 1.0))

    left  = arr[:, :-d]
    right = arr[:, d:]
    contrast = float(np.mean((left - right) ** 2))
    return float(np.clip(contrast / (255.0 ** 2), 0.0, 1.0))


def _glcm_energy(gray_8bit: np.ndarray, d: int = 1) -> float:
    """
    Lightweight proxy for GLCM *energy* (uniformity).

    Computed as 1 / (1 + variance of adjacent-pair differences).
    Returns a value in (0, 1]; higher means more uniform texture.
    """
    arr = np.asarray(gray_8bit, dtype=np.float64)
    flat = arr.ravel()
    if len(flat) <= d:
        return 1.0
    diffs = flat[d:] - flat[:-d]
    var   = float(np.var(diffs))
    return float(1.0 / (1.0 + var))


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class StatisticalCoherenceVerifier:
    """
    Verifies that two image patches share compatible statistical
    properties, as expected from correctly adjacent puzzle fragments.
    """

    def __init__(self, config: Optional[StatisticalCoherenceConfig] = None) -> None:
        self.config = config or StatisticalCoherenceConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        patch_a: np.ndarray,
        patch_b: np.ndarray,
    ) -> StatisticalCoherenceResult:
        """
        Compare statistical properties of two patches.

        Args:
            patch_a: 1-D, 2-D (H×W), or 3-D (H×W×C) numpy array.
            patch_b: Same shape convention as patch_a.

        Returns:
            StatisticalCoherenceResult.
        """
        method = self.config.method.lower()

        hist_sim    = 0.5   # neutral defaults if not computed
        moment_sim  = 0.5
        texture_sim = 0.5

        if method in ("histogram", "both"):
            hist_sim = self._histogram_similarity(patch_a, patch_b)

        if method in ("moments", "both"):
            moment_sim = self._moment_similarity(patch_a, patch_b)

        if self.config.use_texture:
            texture_sim = self._texture_similarity(patch_a, patch_b)

        # Compute overall score depending on method
        if method == "histogram":
            if self.config.use_texture:
                overall = 0.6 * hist_sim + 0.4 * texture_sim
            else:
                overall = float(hist_sim)

        elif method == "moments":
            if self.config.use_texture:
                overall = 0.6 * moment_sim + 0.4 * texture_sim
            else:
                overall = float(moment_sim)

        else:  # "both"
            if self.config.use_texture:
                overall = 0.4 * hist_sim + 0.3 * moment_sim + 0.3 * texture_sim
            else:
                overall = 0.5 * hist_sim + 0.5 * moment_sim

        overall = float(np.clip(overall, 0.0, 1.0))
        is_coherent = overall >= self.config.threshold

        return StatisticalCoherenceResult(
            histogram_similarity=float(np.clip(hist_sim, 0.0, 1.0)),
            moment_similarity=float(np.clip(moment_sim, 0.0, 1.0)),
            texture_similarity=float(np.clip(texture_sim, 0.0, 1.0)),
            overall_score=overall,
            is_coherent=is_coherent,
        )

    # ------------------------------------------------------------------
    # Sub-metrics
    # ------------------------------------------------------------------

    def _histogram_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Compute histogram similarity using the Bhattacharyya coefficient.

        Args:
            a, b: Input patches (any supported shape).

        Returns:
            Float in [0, 1].
        """
        fa = _to_gray_flat(a)
        fb = _to_gray_flat(b)

        # Use combined range so both histograms share the same bins
        lo = min(float(fa.min()), float(fb.min()))
        hi = max(float(fa.max()), float(fb.max()))
        if hi == lo:
            return 1.0   # Both patches are uniform — identical histograms

        bins = self.config.n_bins
        hist_a, _ = np.histogram(fa, bins=bins, range=(lo, hi))
        hist_b, _ = np.histogram(fb, bins=bins, range=(lo, hi))

        return _bhattacharyya_coefficient(
            hist_a.astype(np.float64),
            hist_b.astype(np.float64),
        )

    def _moment_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Compute similarity based on first three statistical moments.

        Uses mean, standard deviation, and skewness. Differences are
        normalised and converted to a similarity score via exponential decay.

        Args:
            a, b: Input patches.

        Returns:
            Float in [0, 1].
        """
        fa = _to_gray_flat(a)
        fb = _to_gray_flat(b)

        def _moments(x: np.ndarray):
            mu  = float(x.mean())
            std = float(x.std())
            sk  = _skewness(x)
            return np.array([mu, std, sk], dtype=np.float64)

        ma = _moments(fa)
        mb = _moments(fb)

        # Normalise each moment to comparable scale
        #   mean / 255, std / 255, skew / 2  (typical skew range ≈ ±2)
        scale = np.array([255.0, 255.0, 2.0])
        diff  = np.abs(ma - mb) / scale
        diff  = np.clip(diff, 0.0, None)

        mean_diff = float(diff.mean())
        similarity = float(np.exp(-3.0 * mean_diff))   # decay rate 3
        return float(np.clip(similarity, 0.0, 1.0))

    def _texture_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Compute texture similarity using GLCM-inspired statistics.

        Compares contrast and energy proxies from both patches.

        Args:
            a, b: Input patches.

        Returns:
            Float in [0, 1].
        """
        def _to_2d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                # Treat as a 1 × N "image"
                return arr.reshape(1, -1)
            if arr.ndim == 2:
                return arr
            # 3-D: average channels
            return arr.mean(axis=2)

        a2 = _to_2d(a)
        b2 = _to_2d(b)

        contrast_a = _glcm_contrast(a2)
        contrast_b = _glcm_contrast(b2)
        energy_a   = _glcm_energy(a2)
        energy_b   = _glcm_energy(b2)

        # Similarity = 1 - normalised absolute difference for each feature
        contrast_sim = 1.0 - abs(contrast_a - contrast_b)
        energy_sim   = 1.0 - abs(energy_a   - energy_b)

        similarity = float(np.clip(0.5 * contrast_sim + 0.5 * energy_sim, 0.0, 1.0))
        return similarity


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def cohere_score(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    config: Optional[StatisticalCoherenceConfig] = None,
) -> float:
    """
    Compute the statistical coherence score between two patches.

    Args:
        patch_a: First patch (1-D, 2-D, or 3-D numpy array).
        patch_b: Second patch (same convention).
        config:  Optional configuration; defaults used if None.

    Returns:
        Float ∈ [0, 1] — the overall coherence score.
    """
    verifier = StatisticalCoherenceVerifier(config)
    result   = verifier.verify(patch_a, patch_b)
    return result.overall_score
