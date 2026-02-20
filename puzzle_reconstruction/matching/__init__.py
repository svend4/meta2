"""
Сопоставление краёв фрагментов.

Модули:
    dtw           — Dynamic Time Warping для кривых краёв
    pairwise      — попарное сравнение двух краёв (CSS + DTW + FD + Text)
    compat_matrix — построение полной матрицы совместимости
"""
from .dtw import dtw_distance, dtw_distance_mirror
from .pairwise import match_score
from .compat_matrix import build_compat_matrix

__all__ = [
    "dtw_distance",
    "dtw_distance_mirror",
    "match_score",
    "build_compat_matrix",
]
