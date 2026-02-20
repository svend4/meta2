"""
Геометрические алгоритмы описания формы (танграм-подход).

fit_tangram, extract_tangram_edge, convex_hull, rdp_simplify,
normalize_polygon, classify_shape, compute_interior_angles
"""
from .inscriber import fit_tangram, extract_tangram_edge
from .hull import convex_hull, rdp_simplify, normalize_polygon
from .classifier import classify_shape, compute_interior_angles

__all__ = [
    "fit_tangram", "extract_tangram_edge",
    "convex_hull", "rdp_simplify", "normalize_polygon",
    "classify_shape", "compute_interior_angles",
]
