"""
Алгоритмы описания и синтеза подписей краёв фрагментов.

Подпакеты:
    tangram/   — геометрическое описание внутреннего многоугольника
    fractal/   — фрактальные характеристики края (Box, Divider, IFS, CSS)

Модули:
    synthesis  — синтез EdgeSignature из танграма и фрактала
"""
from .synthesis import compute_fractal_signature, build_edge_signatures

from .tangram.hull import convex_hull, rdp_simplify, normalize_polygon
from .tangram.classifier import classify_shape, compute_interior_angles
from .tangram.inscriber import fit_tangram

from .fractal.box_counting import box_counting_fd
from .fractal.divider import divider_fd
from .fractal.css import css_similarity_mirror
from .fractal.ifs import fit_ifs

__all__ = [
    # Синтез
    "compute_fractal_signature",
    "build_edge_signatures",
    # Танграм
    "convex_hull",
    "rdp_simplify",
    "normalize_polygon",
    "classify_shape",
    "compute_interior_angles",
    "fit_tangram",
    # Фрактал
    "box_counting_fd",
    "divider_fd",
    "css_similarity_mirror",
    "fit_ifs",
]
