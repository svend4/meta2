"""
Сопоставление краёв фрагментов.

Модули:
    dtw           — Dynamic Time Warping для кривых краёв
    pairwise      — попарное сравнение двух краёв (CSS + DTW + FD + Text)
    compat_matrix — построение полной матрицы совместимости
    icp           — Iterative Closest Point для точного выравнивания контуров
"""
from .dtw import dtw_distance, dtw_distance_mirror
from .pairwise import match_score
from .compat_matrix import build_compat_matrix
from .icp import icp_align, contour_icp, align_fragment_edge, ICPResult

__all__ = [
    "dtw_distance",
    "dtw_distance_mirror",
    "match_score",
    "build_compat_matrix",
    "icp_align",
    "contour_icp",
    "align_fragment_edge",
    "ICPResult",
]
