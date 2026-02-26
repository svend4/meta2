"""
Wavelet-based shape descriptor for edge contours.

Uses a discrete wavelet transform (DWT) to build a multi-resolution
descriptor of edge contour shapes.  Unlike Fourier descriptors, wavelets
capture local shape features at multiple scales simultaneously.

Algorithm:
    1. Resample the contour to a power-of-2 length.
    2. Compute the complex envelope (x(t) + j·y(t)).
    3. Apply DWT (Haar by default) to the real and imaginary parts.
    4. Concatenate and L2-normalise the detail coefficients.

The resulting descriptor is:
    - Translation invariant (compute on arc-length parameterised contour).
    - Scale invariant (normalise to unit bounding box).
    - Rotation sensitive by default (or can be made rotation invariant
      by phase normalisation).

Reference:
    Chuang & Kuo, "Wavelet Descriptor of Planar Curves", IEEE Trans.
    Image Processing, 1996.
"""
from __future__ import annotations

import numpy as np
from typing import List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class WaveletDescriptor(NamedTuple):
    """Wavelet descriptor of a contour."""
    coeffs: np.ndarray          # Concatenated DWT detail coefficients, L2-normalised
    energy_per_level: np.ndarray  # Energy fraction at each decomposition level
    n_levels: int               # Number of decomposition levels used


# ---------------------------------------------------------------------------
# Haar DWT (pure numpy, no external deps)
# ---------------------------------------------------------------------------

def _haar_dwt_1d(signal: np.ndarray) -> List[np.ndarray]:
    """
    1-D multi-level Haar DWT.

    Args:
        signal: 1-D array of length 2^k.

    Returns:
        List of detail coefficient arrays [level-1, level-2, ...].
        Each level has half the length of the previous.
    """
    details = []
    x = signal.copy().astype(float)
    n = len(x)
    while n >= 2:
        half = n // 2
        approx  = (x[:n:2] + x[1:n:2]) / np.sqrt(2.0)
        detail  = (x[:n:2] - x[1:n:2]) / np.sqrt(2.0)
        details.append(detail)
        x = approx
        n = half
    return details


def _next_pow2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Contour resampling
# ---------------------------------------------------------------------------

def _resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a (K, 2) contour to n equally-spaced points.

    Args:
        contour: (K, 2) array of 2-D points.
        n:       Target number of points (should be power of 2).

    Returns:
        (n, 2) resampled contour.
    """
    if len(contour) < 2:
        return np.zeros((n, 2))
    diffs  = np.diff(contour, axis=0)
    segs   = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(segs)])
    total  = cumlen[-1]
    if total == 0:
        return np.tile(contour[0], (n, 1))
    t_new = np.linspace(0.0, total, n)
    x_new = np.interp(t_new, cumlen, contour[:, 0])
    y_new = np.interp(t_new, cumlen, contour[:, 1])
    return np.column_stack([x_new, y_new])


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise_contour(contour: np.ndarray) -> np.ndarray:
    """
    Translate to centroid and scale to unit bounding box diagonal.

    Args:
        contour: (N, 2) array.

    Returns:
        Normalised (N, 2) array.
    """
    c = contour - contour.mean(axis=0)
    diag = np.linalg.norm(c.max(axis=0) - c.min(axis=0))
    if diag > 0:
        c = c / diag
    return c


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def compute_wavelet_descriptor(
    contour: np.ndarray,
    n_points: Optional[int] = None,
    n_levels: int = 4,
) -> WaveletDescriptor:
    """
    Compute wavelet descriptor of a contour.

    Args:
        contour:  (N, 2) array of contour points.
        n_points: Number of points to resample to (must be power of 2).
                  If None, uses next power of 2 >= len(contour), capped at 256.
        n_levels: Number of DWT levels to use.

    Returns:
        WaveletDescriptor.
    """
    if len(contour) < 2:
        empty = np.zeros(2 ** n_levels)
        return WaveletDescriptor(
            coeffs=empty,
            energy_per_level=np.zeros(n_levels),
            n_levels=n_levels,
        )

    # Determine resampling size
    if n_points is None:
        n_points = min(_next_pow2(len(contour)), 256)
    else:
        n_points = _next_pow2(n_points)  # ensure power of 2

    # Resample and normalise
    c = _resample_contour(contour, n_points)
    c = _normalise_contour(c)

    # DWT on x and y channels
    details_x = _haar_dwt_1d(c[:, 0])
    details_y = _haar_dwt_1d(c[:, 1])

    # Use only the first n_levels detail bands
    n_use = min(n_levels, len(details_x))
    used_x = details_x[:n_use]
    used_y = details_y[:n_use]

    # Concatenate detail coefficients from both channels
    all_details = np.concatenate([d for pair in zip(used_x, used_y) for d in pair])

    # Energy per level
    energy_per_level = np.array([
        float(np.sum(details_x[k] ** 2) + np.sum(details_y[k] ** 2))
        for k in range(n_use)
    ])
    total_energy = energy_per_level.sum()
    if total_energy > 0:
        energy_per_level = energy_per_level / total_energy

    # L2 normalise coefficients
    norm = np.linalg.norm(all_details)
    if norm > 0:
        all_details = all_details / norm

    return WaveletDescriptor(
        coeffs=all_details,
        energy_per_level=energy_per_level,
        n_levels=n_use,
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def wavelet_similarity(
    desc_a: WaveletDescriptor,
    desc_b: WaveletDescriptor,
) -> float:
    """
    Cosine similarity between two wavelet descriptors.

    Args:
        desc_a, desc_b: WaveletDescriptor objects.

    Returns:
        Similarity ∈ [0, 1].
    """
    a, b = desc_a.coeffs, desc_b.coeffs
    if a.shape != b.shape:
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    dot  = float(np.dot(a, b))
    na   = float(np.linalg.norm(a))
    nb   = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(dot / (na * nb), 0.0, 1.0))


def wavelet_similarity_mirror(
    desc_a: WaveletDescriptor,
    desc_b: WaveletDescriptor,
) -> float:
    """
    Mirror-aware similarity: max of direct and reversed similarity.

    Useful for matching torn edges where one edge is the mirror of another.

    Args:
        desc_a, desc_b: WaveletDescriptor objects.

    Returns:
        Similarity ∈ [0, 1].
    """
    direct   = wavelet_similarity(desc_a, desc_b)
    # Reversed: reverse the coefficient ordering within each level
    rev_b = WaveletDescriptor(
        coeffs=desc_b.coeffs[::-1],
        energy_per_level=desc_b.energy_per_level[::-1],
        n_levels=desc_b.n_levels,
    )
    mirrored = wavelet_similarity(desc_a, rev_b)
    return max(direct, mirrored)


def batch_wavelet_similarity(
    query: WaveletDescriptor,
    candidates: List[WaveletDescriptor],
    use_mirror: bool = False,
) -> np.ndarray:
    """
    Compute similarity between query and all candidates.

    Args:
        query:      Reference descriptor.
        candidates: List of candidate descriptors.
        use_mirror: If True, use mirror-aware similarity.

    Returns:
        (N,) array of similarity scores.
    """
    fn = wavelet_similarity_mirror if use_mirror else wavelet_similarity
    return np.array([fn(query, c) for c in candidates])
