"""
Zernike moment descriptor for fragment edge contours.

Zernike moments are orthogonal moments defined on the unit disk,
providing rotation-invariant shape descriptors.

Reference:
    Teague, M.R., "Image Analysis via the General Theory of Moments",
    J. Optical Society of America, 1980.
"""
from __future__ import annotations

import numpy as np
from typing import List, NamedTuple, Optional, Tuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ZernikeDescriptor(NamedTuple):
    """Zernike moment descriptor of a contour."""
    moments: np.ndarray      # Complex Zernike moments for all valid (n, m) pairs
    magnitudes: np.ndarray   # Absolute values of moments (rotation invariant)
    order: int               # Maximum radial order used


# ---------------------------------------------------------------------------
# Helper: factorial
# ---------------------------------------------------------------------------

def _factorial(n: int) -> float:
    """
    Compute factorial of n.

    Args:
        n: Non-negative integer.

    Returns:
        n! as a float.
    """
    if n < 0:
        raise ValueError(f"Factorial undefined for negative integer {n}")
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


# ---------------------------------------------------------------------------
# Zernike radial polynomial
# ---------------------------------------------------------------------------

def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute the Zernike radial polynomial R_nm(rho).

    The radial polynomial is defined for n >= 0, |m| <= n, (n - m) even:

        R_nm(rho) = sum_{s=0}^{(n-|m|)/2}
            (-1)^s * (n-s)! / (s! * ((n+|m|)/2 - s)! * ((n-|m|)/2 - s)!)
            * rho^(n - 2*s)

    Args:
        n:   Radial order (non-negative integer).
        m:   Azimuthal frequency (integer, |m| <= n, (n-m) even).
        rho: 1-D array of radii in [0, 1].

    Returns:
        Array of same shape as rho containing R_nm(rho).
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho, dtype=float)

    rho = np.asarray(rho, dtype=float)
    result = np.zeros_like(rho)
    n_terms = (n - m_abs) // 2

    for s in range(n_terms + 1):
        num = _factorial(n - s)
        den = (
            _factorial(s)
            * _factorial((n + m_abs) // 2 - s)
            * _factorial((n - m_abs) // 2 - s)
        )
        coeff = ((-1) ** s) * num / den
        result += coeff * rho ** (n - 2 * s)

    return result


# ---------------------------------------------------------------------------
# Zernike basis function
# ---------------------------------------------------------------------------

def _zernike_basis(
    n: int,
    m: int,
    rho: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute the complex Zernike basis function V_nm(rho, theta).

        V_nm(rho, theta) = R_nm(rho) * exp(j * m * theta)

    Points with rho > 1 contribute zero (outside the unit disk).

    Args:
        n:     Radial order.
        m:     Azimuthal frequency.
        rho:   1-D array of radii.
        theta: 1-D array of angles (radians), same length as rho.

    Returns:
        Complex array of same length as rho.
    """
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)

    r_poly = _radial_polynomial(n, m, rho)
    basis = r_poly * np.exp(1j * m * theta)

    # Zero out points outside the unit disk
    basis[rho > 1.0] = 0.0

    return basis


# ---------------------------------------------------------------------------
# Valid (n, m) index pairs
# ---------------------------------------------------------------------------

def _valid_nm_pairs(order: int) -> List[Tuple[int, int]]:
    """
    Return all valid (n, m) pairs up to given order.

    A pair is valid when: n >= 0, |m| <= n, (n - m) is even.

    Args:
        order: Maximum radial order.

    Returns:
        Sorted list of (n, m) tuples.
    """
    pairs = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if (n - m) % 2 == 0:
                pairs.append((n, m))
    return pairs


# ---------------------------------------------------------------------------
# Contour resampling (shared utility)
# ---------------------------------------------------------------------------

def _resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a (K, 2) contour to n equally-spaced points.

    Args:
        contour: (K, 2) array of 2-D points.
        n:       Target number of points.

    Returns:
        (n, 2) resampled contour.
    """
    if len(contour) < 2:
        return np.zeros((n, 2))

    diffs = np.diff(contour, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(segs)])
    total = cumlen[-1]

    if total == 0.0:
        return np.tile(contour[0], (n, 1))

    t_new = np.linspace(0.0, total, n)
    x_new = np.interp(t_new, cumlen, contour[:, 0])
    y_new = np.interp(t_new, cumlen, contour[:, 1])
    return np.column_stack([x_new, y_new])


# ---------------------------------------------------------------------------
# Main descriptor computation
# ---------------------------------------------------------------------------

def zernike_moments(
    contour: np.ndarray,
    order: int = 10,
    n_points: int = 64,
) -> ZernikeDescriptor:
    """
    Compute Zernike moments of a contour.

    The contour is resampled to n_points, centred, and mapped to the unit
    disk.  Moments are then computed for every valid (n, m) pair with
    n <= order.

    Args:
        contour:  (N, 2) array of 2-D contour points.
        order:    Maximum Zernike order (non-negative integer).
        n_points: Number of points to use after resampling.

    Returns:
        ZernikeDescriptor with complex moments, their magnitudes, and the
        order used.
    """
    order = max(0, int(order))
    n_points = max(1, int(n_points))

    pairs = _valid_nm_pairs(order)
    n_moments = len(pairs)

    # Degenerate input: too short
    contour = np.asarray(contour, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 2:
        moments_arr = np.zeros(n_moments, dtype=complex)
        return ZernikeDescriptor(
            moments=moments_arr,
            magnitudes=np.zeros(n_moments),
            order=order,
        )

    # Resample
    pts = _resample_contour(contour, n_points)

    # Center and map to unit disk
    center = pts.mean(axis=0)
    pts = pts - center

    # Normalise by the maximum radius so all points fall within the unit disk
    radii = np.hypot(pts[:, 0], pts[:, 1])
    max_r = radii.max()

    if max_r == 0.0:
        # Degenerate: all points at the origin
        moments_arr = np.zeros(n_moments, dtype=complex)
        return ZernikeDescriptor(
            moments=moments_arr,
            magnitudes=np.zeros(n_moments),
            order=order,
        )

    pts = pts / max_r
    rho = np.hypot(pts[:, 0], pts[:, 1])          # normalised radii in [0,1]
    theta = np.arctan2(pts[:, 1], pts[:, 0])       # angles

    # Compute moments
    # Zernike moment: A_nm = (n+1)/pi * sum_k conj(V_nm(rho_k, theta_k)) * dA
    # For a discrete contour, approximate area element as 1/n_points.
    scale = (1.0 / n_points)

    moments_arr = np.empty(n_moments, dtype=complex)
    for idx, (n, m) in enumerate(pairs):
        basis = _zernike_basis(n, m, rho, theta)
        moment = (n + 1) / np.pi * np.sum(np.conj(basis)) * scale
        moments_arr[idx] = moment

    magnitudes = np.abs(moments_arr)

    return ZernikeDescriptor(
        moments=moments_arr,
        magnitudes=magnitudes,
        order=order,
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def zernike_similarity(
    desc_a: ZernikeDescriptor,
    desc_b: ZernikeDescriptor,
) -> float:
    """
    Normalised correlation similarity between two Zernike descriptors.

    Uses the magnitude vectors so the result is rotation invariant.

    Args:
        desc_a: First ZernikeDescriptor.
        desc_b: Second ZernikeDescriptor.

    Returns:
        Similarity score in [0, 1].  Returns 0.0 for degenerate inputs.
    """
    a = desc_a.magnitudes.astype(float)
    b = desc_b.magnitudes.astype(float)

    # Align lengths
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na == 0.0 or nb == 0.0:
        return 0.0

    dot = float(np.dot(a, b))
    return float(np.clip(dot / (na * nb), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

def zernike_to_feature_vector(desc: ZernikeDescriptor) -> np.ndarray:
    """
    Convert a ZernikeDescriptor to an L2-normalised feature vector.

    The feature vector consists of the magnitude values, normalised to unit
    L2 norm.  If all magnitudes are zero, a zero vector is returned.

    Args:
        desc: ZernikeDescriptor.

    Returns:
        1-D float array of length equal to the number of moments.
    """
    vec = desc.magnitudes.astype(float).copy()
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec = vec / norm
    return vec
