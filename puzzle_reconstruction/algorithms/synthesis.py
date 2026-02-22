"""
Синтез EdgeSignature — «золотая середина» танграма и фрактальной кромки.

Идея:
    B_virtual(t) = α · B_tangram(t) + (1-α) · B_fractal(t)

Танграм даёт стабильную крупную геометрию,
Фрактальная кривая даёт точное воспроизведение мелких неровностей.
Их наложение — уникальная «береговая подпись» каждого края.
"""
import numpy as np
from typing import List

from ..models import (Fragment, EdgeSignature, EdgeSide,
                      TangramSignature, FractalSignature)
from .tangram.inscriber import extract_tangram_edge
from .fractal.css import css_to_feature_vector
from ..preprocessing.contour import resample_curve, split_contour_to_edges


def compute_fractal_signature(contour: np.ndarray) -> FractalSignature:
    """
    Вычисляет полную фрактальную подпись контура.
    """
    from .fractal.box_counting import box_counting_fd
    from .fractal.divider import divider_fd
    from .fractal.ifs import fit_ifs_coefficients
    from .fractal.css import curvature_scale_space, css_to_feature_vector, freeman_chain_code

    curve = resample_curve(contour, n_points=256)

    fd_box     = box_counting_fd(contour)
    fd_divider = divider_fd(contour)
    ifs_coeffs = fit_ifs_coefficients(curve, n_transforms=8)
    css_image  = curvature_scale_space(curve)
    chain_code = freeman_chain_code(contour[:64] if len(contour) > 64 else contour)

    return FractalSignature(
        fd_box=fd_box,
        fd_divider=fd_divider,
        ifs_coeffs=ifs_coeffs,
        css_image=css_image,
        chain_code=chain_code,
        curve=curve,
    )


def build_edge_signatures(fragment: Fragment,
                          alpha: float = 0.5,
                          n_sides: int = 4,
                          n_points: int = 128) -> List[EdgeSignature]:
    """
    Строит список EdgeSignature для всех краёв фрагмента.

    Args:
        fragment: Фрагмент с заполненными полями tangram и fractal.
        alpha:    Вес танграма в синтезе (0 = только фрактал, 1 = только танграм).
        n_sides:  Ожидаемое число краёв.
        n_points: Число точек в кривой края.

    Returns:
        Список EdgeSignature.
    """
    assert fragment.tangram  is not None, "Сначала вызовите fit_tangram()"
    assert fragment.fractal  is not None, "Сначала вызовите compute_fractal_signature()"

    # Разбиваем контур на логические края
    raw_edges = split_contour_to_edges(fragment.contour, n_sides=n_sides)

    signatures = []
    for idx, (edge_pts, side) in enumerate(raw_edges):
        sig = _build_one_edge(
            edge_pts=edge_pts,
            side=side,
            edge_id=idx,
            tangram=fragment.tangram,
            fractal=fragment.fractal,
            tangram_edge_index=idx,
            alpha=alpha,
            n_points=n_points,
        )
        signatures.append(sig)

    return signatures


def _build_one_edge(edge_pts: np.ndarray,
                    side: EdgeSide,
                    edge_id: int,
                    tangram: TangramSignature,
                    fractal: FractalSignature,
                    tangram_edge_index: int,
                    alpha: float,
                    n_points: int) -> EdgeSignature:
    """Строит EdgeSignature для одного края."""
    from .fractal.css import curvature_scale_space, css_to_feature_vector

    # Фрактальная кривая края (из реального контура)
    frac_curve = resample_curve(edge_pts, n_points=n_points)

    # Танграм кривая соответствующей стороны (геометрически правильная)
    tang_curve = extract_tangram_edge(tangram, tangram_edge_index, n_points=n_points)

    # Приводим к одному масштабу (оба уже нормализованы по-своему)
    frac_c = _normalize_curve(frac_curve)
    tang_c = _normalize_curve(tang_curve)

    # Синтез: «золотая середина»
    virtual_curve = alpha * tang_c + (1.0 - alpha) * frac_c

    # Средняя FD
    fd = (fractal.fd_box + fractal.fd_divider) / 2.0

    # CSS-вектор для этого края (вычисляем отдельно по краю, не всему контуру)
    css_edge = curvature_scale_space(frac_curve, n_sigmas=5)
    css_vec  = css_to_feature_vector(css_edge, n_bins=32)

    # Физическая длина края
    diffs = np.diff(edge_pts, axis=0)
    length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())

    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=virtual_curve,
        fd=fd,
        css_vec=css_vec,
        ifs_coeffs=fractal.ifs_coeffs,
        length=length,
    )


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Нормализует кривую: центроид в 0, масштаб = 1."""
    c = curve - curve.mean(axis=0)
    scale = np.abs(c).max()
    if scale > 0:
        c = c / scale
    return c
