"""
Фрактальные алгоритмы описания краёв фрагментов.

box_counting_fd, divider_fd, curvature_scale_space,
css_similarity_mirror, freeman_chain_code, fit_ifs_coefficients,
reconstruct_from_ifs, ifs_distance
"""
from .box_counting import box_counting_fd, box_counting_curve
from .divider import divider_fd, divider_curve
from .css import (
    curvature_scale_space,
    css_to_feature_vector,
    css_similarity,
    css_similarity_mirror,
    freeman_chain_code,
)
from .ifs import fit_ifs_coefficients, reconstruct_from_ifs, ifs_distance

fit_ifs = fit_ifs_coefficients

__all__ = [
    "box_counting_fd", "box_counting_curve",
    "divider_fd", "divider_curve",
    "curvature_scale_space", "css_to_feature_vector",
    "css_similarity", "css_similarity_mirror",
    "freeman_chain_code",
    "fit_ifs_coefficients", "fit_ifs",
    "reconstruct_from_ifs", "ifs_distance",
]
