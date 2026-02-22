"""
Вписывание танграм-полигона ВНУТРЬ фрагмента.

Ключевая идея: берём выпуклую оболочку фрагмента и «сдвигаем» её внутрь
так, чтобы танграм-фигура целиком лежала в пределах маски фрагмента.
Это отделяет «геометрический скелет» фрагмента от «берегового» края.
"""
import numpy as np
from .hull import convex_hull, rdp_simplify, normalize_polygon
from .classifier import classify_shape
from ...models import TangramSignature


def fit_tangram(contour: np.ndarray,
                mask: np.ndarray = None,
                epsilon_ratio: float = 0.02,
                inset_ratio: float = 0.05) -> TangramSignature:
    """
    Вписывает танграм-полигон внутрь фрагмента.

    Args:
        contour:      (N, 2) внешний контур фрагмента.
        mask:         (H, W) uint8 маска (опционально, для проверки вхождения).
        epsilon_ratio: Параметр RDP-упрощения.
        inset_ratio:  Доля «отступа» от внешнего контура вовнутрь.

    Returns:
        TangramSignature
    """
    # 1. Выпуклая оболочка
    hull = convex_hull(contour)

    # 2. Упрощаем до танграм-подобного полигона
    simplified = rdp_simplify(hull, epsilon_ratio=epsilon_ratio)

    # 3. Сдвигаем вовнутрь (inset) — уменьшаем каждую сторону
    inset_poly = _inset_polygon(simplified, inset_ratio)

    # 4. Нормализуем
    normalized, centroid, scale, angle = normalize_polygon(inset_poly)

    # 5. Классифицируем форму
    shape_class = classify_shape(normalized)

    # 6. Площадь
    area = _polygon_area(normalized)

    return TangramSignature(
        polygon=normalized,
        shape_class=shape_class,
        centroid=centroid,
        angle=angle,
        scale=scale,
        area=area,
    )


def extract_tangram_edge(tangram: TangramSignature,
                         edge_index: int,
                         n_points: int = 128) -> np.ndarray:
    """
    Извлекает один край танграм-полигона как параметрическую кривую.

    Args:
        tangram:    TangramSignature.
        edge_index: Индекс стороны (0 .. K-1).
        n_points:   Число точек дискретизации.

    Returns:
        edge_curve: (n_points, 2)
    """
    poly = tangram.polygon
    n = len(poly)
    p0 = poly[edge_index % n]
    p1 = poly[(edge_index + 1) % n]
    t = np.linspace(0, 1, n_points)
    curve = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    return curve


# ---------------------------------------------------------------------------

def _inset_polygon(polygon: np.ndarray, ratio: float) -> np.ndarray:
    """
    «Схлопывает» полигон к центроиду на ratio от его масштаба.
    """
    centroid = polygon.mean(axis=0)
    inset = centroid + (1.0 - ratio) * (polygon - centroid)
    return inset


def _polygon_area(polygon: np.ndarray) -> float:
    """Площадь полигона через формулу Гаусса (Shoelace)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
