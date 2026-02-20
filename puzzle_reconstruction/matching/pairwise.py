"""
Попарная оценка совместимости двух краёв фрагментов.

MatchScore = w1·CSS_sim + w2·(1/(1+DTW)) + w3·(1/(1+FD_diff)) + w4·Text_coh
"""
import numpy as np
from ..models import EdgeSignature, CompatEntry
from .dtw import dtw_distance_mirror
from ..algorithms.fractal.css import css_similarity_mirror


# Веса компонент оценки (сумма = 1.0)
W_CSS  = 0.35
W_DTW  = 0.30
W_FD   = 0.20
W_TEXT = 0.15


def match_score(e_i: EdgeSignature,
                e_j: EdgeSignature,
                text_score: float = 0.0) -> CompatEntry:
    """
    Вычисляет полную оценку совместимости двух краёв.

    Args:
        e_i, e_j:    Края двух разных фрагментов.
        text_score:  Внешняя оценка связности текста (0..1), от OCR-модуля.

    Returns:
        CompatEntry с полной информацией о совместимости.
    """
    # 1. CSS-сходство (с учётом зеркальности)
    css_sim = css_similarity_mirror(e_i.css_vec, e_j.css_vec)

    # 2. DTW между виртуальными кривыми (с зеркалированием)
    dtw_dist = dtw_distance_mirror(e_i.virtual_curve, e_j.virtual_curve)
    dtw_score = 1.0 / (1.0 + dtw_dist)

    # 3. Разница фрактальных размерностей
    fd_diff  = abs(e_i.fd - e_j.fd)
    fd_score = 1.0 / (1.0 + fd_diff)

    # 4. IFS-расстояние как дополнительный сигнал (нормализованное)
    ifs_dist  = _ifs_distance_norm(e_i.ifs_coeffs, e_j.ifs_coeffs)
    ifs_score = 1.0 / (1.0 + ifs_dist)

    # Итоговый скор (IFS учитываем как часть DTW-веса)
    score = (W_CSS  * css_sim
           + W_DTW  * (0.7 * dtw_score + 0.3 * ifs_score)
           + W_FD   * fd_score
           + W_TEXT * text_score)

    # Штраф за сильно разные длины краёв
    len_ratio = min(e_i.length, e_j.length) / (max(e_i.length, e_j.length) + 1e-5)
    if len_ratio < 0.5:
        score *= len_ratio  # Серьёзно штрафуем за несовместимые длины

    score = float(np.clip(score, 0.0, 1.0))

    return CompatEntry(
        edge_i=e_i,
        edge_j=e_j,
        score=score,
        dtw_dist=float(dtw_dist),
        css_sim=float(css_sim),
        fd_diff=float(fd_diff),
        text_score=float(text_score),
    )


def _ifs_distance_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Нормализованное расстояние между IFS-коэффициентами."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = a[:n] - b[:n]
    return float(np.linalg.norm(diff)) / (np.sqrt(n) + 1e-10)
