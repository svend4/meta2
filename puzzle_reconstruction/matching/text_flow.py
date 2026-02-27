"""
Text-flow matcher for document fragments.

When fragments contain printed text, the baseline direction and text-line
continuity across edges provides strong matching signal beyond just shape.

Algorithm:
    1. Detect dominant text-baseline angle in each fragment using gradient
       analysis (text lines create strong horizontal gradient patterns).
    2. Compare baseline angles between adjacent edge candidates.
    3. Score the match: angle agreement + spatial alignment of line endings.

This module is intentionally lightweight — it works with pre-computed
gradient maps and does not require OCR or heavy text detection.

Typical usage:
    scorer = TextFlowScorer()
    score = scorer.score(frag_a_gradients, frag_b_gradients, edge_a, edge_b)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TextLineProfile:
    """Compact representation of text-line structure near an edge."""
    angle_deg: float          # dominant baseline angle in degrees
    line_positions: np.ndarray  # normalised y-positions of detected baselines ∈ [0, 1]
    confidence: float         # detection confidence ∈ [0, 1]
    n_lines: int              # number of detected text lines


@dataclass
class TextFlowMatch:
    """Result of text-flow matching between two edge profiles."""
    score: float              # overall match score ∈ [0, 1]
    angle_score: float        # agreement on baseline angle ∈ [0, 1]
    alignment_score: float    # spatial alignment of line endings ∈ [0, 1]
    angle_diff_deg: float     # absolute difference in angles


@dataclass
class TextFlowConfig:
    """Configuration for TextFlowScorer."""
    angle_tolerance_deg: float = 5.0   # tolerance for baseline angle matching
    max_angle_diff_deg: float  = 30.0  # angles more different → score=0
    line_align_tolerance: float = 0.05  # tolerance in normalised coords
    min_confidence: float = 0.1        # min confidence to use profile
    angle_weight: float = 0.5          # weight of angle component in final score
    align_weight: float = 0.5          # weight of alignment component


# ---------------------------------------------------------------------------
# Baseline detection
# ---------------------------------------------------------------------------

def detect_text_baseline_angle(
    gradient_y: np.ndarray,
    n_bins: int = 180,
) -> Tuple[float, float]:
    """
    Estimate dominant text-baseline angle from the vertical gradient map.

    Text baselines produce strong horizontal gradient responses.  We use
    a Hough-like accumulation over row-wise gradient energy to find the
    dominant angle.

    Args:
        gradient_y: (H, W) vertical gradient magnitude (e.g., Sobel-Y).
        n_bins:     Angular resolution in degrees.

    Returns:
        (angle_deg, confidence) where angle_deg ∈ [-90, 90) and
        confidence ∈ [0, 1] based on peak-to-mean ratio of the
        angular energy spectrum.
    """
    if gradient_y.size == 0:
        return 0.0, 0.0

    h, w = gradient_y.shape
    if h < 2 or w < 2:
        return 0.0, 0.0

    angles = np.linspace(-90.0, 90.0, n_bins, endpoint=False)
    energy = np.zeros(n_bins)

    # For each angle, project gradient energy onto that direction
    for k, angle in enumerate(angles):
        rad = np.deg2rad(angle)
        # Shear the image by tan(angle) and sum columns
        tan_a = np.tan(rad)
        col_energy = 0.0
        for col in range(w):
            # Vertical offset for this column
            row_shift = int(round(col * tan_a))
            if row_shift >= 0:
                g = gradient_y[row_shift:, col]
            else:
                g = gradient_y[:h + row_shift, col]
            col_energy += float(np.sum(g ** 2)) if len(g) > 0 else 0.0
        energy[k] = col_energy

    if energy.sum() == 0:
        return 0.0, 0.0

    best_idx = int(np.argmax(energy))
    best_angle = float(angles[best_idx])
    mean_energy = float(energy.mean())
    peak_energy = float(energy[best_idx])
    confidence = float(np.clip((peak_energy - mean_energy) / (peak_energy + 1e-9), 0.0, 1.0))

    return best_angle, confidence


def detect_text_line_positions(
    gradient_y: np.ndarray,
    min_peak_ratio: float = 0.3,
) -> np.ndarray:
    """
    Detect normalised positions of text baselines by finding peaks in the
    row-wise gradient energy profile.

    Args:
        gradient_y:     (H, W) vertical gradient magnitude.
        min_peak_ratio: Minimum peak height relative to max peak.

    Returns:
        1-D array of normalised positions ∈ [0, 1].
    """
    if gradient_y.size == 0:
        return np.array([])

    row_energy = np.sum(gradient_y ** 2, axis=1)  # (H,)
    if row_energy.max() == 0:
        return np.array([])

    h = len(row_energy)
    # Simple peak detection: find local maxima above threshold
    threshold = row_energy.max() * min_peak_ratio
    peaks = []
    for i in range(1, h - 1):
        if row_energy[i] >= threshold:
            if row_energy[i] >= row_energy[i - 1] and row_energy[i] >= row_energy[i + 1]:
                peaks.append(i / (h - 1))  # normalise to [0, 1]

    return np.array(peaks, dtype=float)


def build_text_line_profile(
    gradient_y: np.ndarray,
) -> TextLineProfile:
    """
    Build a TextLineProfile from a vertical gradient map.

    Args:
        gradient_y: (H, W) vertical gradient magnitude near an edge.

    Returns:
        TextLineProfile.
    """
    angle, conf = detect_text_baseline_angle(gradient_y)
    positions  = detect_text_line_positions(gradient_y)
    return TextLineProfile(
        angle_deg=angle,
        line_positions=positions,
        confidence=conf,
        n_lines=len(positions),
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def compare_baseline_angles(
    angle_a: float,
    angle_b: float,
    tolerance_deg: float = 5.0,
    max_diff_deg: float = 30.0,
) -> float:
    """
    Score angle agreement between two baseline angles.

    Returns 1.0 when |diff| ≤ tolerance, decreasing linearly to 0.0
    at max_diff_deg.

    Args:
        angle_a, angle_b:  Baseline angles in degrees.
        tolerance_deg:     Zero-penalty zone.
        max_diff_deg:      Angle at which score → 0.

    Returns:
        Score ∈ [0, 1].
    """
    diff = abs(angle_a - angle_b)
    # Handle wrap-around (e.g. 89 vs -89 = diff 2, not 178)
    diff = min(diff, 180.0 - diff)
    if diff <= tolerance_deg:
        return 1.0
    if diff >= max_diff_deg:
        return 0.0
    return float(1.0 - (diff - tolerance_deg) / (max_diff_deg - tolerance_deg))


def align_line_positions(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    tolerance: float = 0.05,
) -> float:
    """
    Score spatial alignment between two sets of baseline positions.

    Uses greedy nearest-neighbour matching: each position in A is matched
    to the nearest position in B within tolerance.

    Args:
        positions_a:  Normalised positions of text baselines ∈ [0, 1].
        positions_b:  Normalised positions (other fragment edge).
        tolerance:    Match window in normalised coordinates.

    Returns:
        Alignment score ∈ [0, 1].  0.0 if either array is empty.
    """
    if len(positions_a) == 0 or len(positions_b) == 0:
        return 0.0

    matched = 0
    used = set()
    for pa in positions_a:
        best_dist = float("inf")
        best_j = -1
        for j, pb in enumerate(positions_b):
            if j in used:
                continue
            d = abs(pa - pb)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_dist <= tolerance and best_j >= 0:
            matched += 1
            used.add(best_j)

    total = max(len(positions_a), len(positions_b))
    return matched / total


def match_text_flow(
    profile_a: TextLineProfile,
    profile_b: TextLineProfile,
    cfg: Optional[TextFlowConfig] = None,
) -> TextFlowMatch:
    """
    Compute text-flow match score between two edge profiles.

    Args:
        profile_a: TextLineProfile from fragment A's edge.
        profile_b: TextLineProfile from fragment B's edge.
        cfg:       Configuration (uses defaults if None).

    Returns:
        TextFlowMatch with per-component scores.
    """
    if cfg is None:
        cfg = TextFlowConfig()

    # If either profile has low confidence, return uncertain score
    if (profile_a.confidence < cfg.min_confidence or
            profile_b.confidence < cfg.min_confidence):
        return TextFlowMatch(
            score=0.5,
            angle_score=0.5,
            alignment_score=0.5,
            angle_diff_deg=abs(profile_a.angle_deg - profile_b.angle_deg),
        )

    angle_score = compare_baseline_angles(
        profile_a.angle_deg,
        profile_b.angle_deg,
        tolerance_deg=cfg.angle_tolerance_deg,
        max_diff_deg=cfg.max_angle_diff_deg,
    )

    align_score = align_line_positions(
        profile_a.line_positions,
        profile_b.line_positions,
        tolerance=cfg.line_align_tolerance,
    )

    score = (cfg.angle_weight * angle_score +
             cfg.align_weight * align_score)
    score = float(np.clip(score, 0.0, 1.0))

    return TextFlowMatch(
        score=score,
        angle_score=angle_score,
        alignment_score=align_score,
        angle_diff_deg=abs(profile_a.angle_deg - profile_b.angle_deg),
    )


# ---------------------------------------------------------------------------
# High-level scorer
# ---------------------------------------------------------------------------

class TextFlowScorer:
    """
    High-level scorer: takes raw gradient maps, builds profiles internally,
    and returns match scores.
    """

    def __init__(self, cfg: Optional[TextFlowConfig] = None) -> None:
        self.cfg = cfg or TextFlowConfig()

    def build_profile(self, gradient_y: np.ndarray) -> TextLineProfile:
        """Build a TextLineProfile from a gradient map."""
        return build_text_line_profile(gradient_y)

    def score(
        self,
        gradient_a: np.ndarray,
        gradient_b: np.ndarray,
    ) -> TextFlowMatch:
        """
        Score text-flow compatibility between two edge gradient maps.

        Args:
            gradient_a: (H_a, W_a) vertical gradient near edge of fragment A.
            gradient_b: (H_b, W_b) vertical gradient near edge of fragment B.

        Returns:
            TextFlowMatch.
        """
        pa = build_text_line_profile(gradient_a)
        pb = build_text_line_profile(gradient_b)
        return match_text_flow(pa, pb, self.cfg)

    def score_batch(
        self,
        gradient_a: np.ndarray,
        gradients_b: List[np.ndarray],
    ) -> List[TextFlowMatch]:
        """
        Score text-flow against multiple candidate edges.

        Args:
            gradient_a:   Reference edge gradient map.
            gradients_b:  List of candidate gradient maps.

        Returns:
            List of TextFlowMatch results.
        """
        pa = build_text_line_profile(gradient_a)
        return [
            match_text_flow(pa, build_text_line_profile(gb), self.cfg)
            for gb in gradients_b
        ]
