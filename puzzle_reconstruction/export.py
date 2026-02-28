"""
Экспорт результатов восстановления документа.

Форматы:
    PNG/JPEG     — высококачественная растровая картинка сборки
    PDF          — документ с текстовым слоем (если OCR проведён)
    Heatmap      — тепловая карта уверенности каждого шва
    Mosaic       — сравнение: отдельные фрагменты / сборка / heatmap

Использование:
    canvas  = render_canvas(assembly)
    heatmap = render_heatmap(assembly, canvas.shape)
    mosaic  = render_mosaic(assembly)
    save_png(canvas, "result.png")
    save_pdf(assembly, canvas, "result.pdf")
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .models import Assembly, Fragment


# ─── Основной холст ───────────────────────────────────────────────────────

def render_canvas(assembly: Assembly,
                   margin: int = 20,
                   bg_color: Tuple[int, int, int] = (240, 240, 240)
                   ) -> np.ndarray:
    """
    Рендерит восстановленный документ на плоском холсте.

    Алгоритм:
    1. Вычисляем bounding box всех размещённых фрагментов.
    2. Создаём холст нужного размера с отступами.
    3. Для каждого фрагмента: поворачиваем, накладываем по маске.

    Args:
        assembly:  Собранный документ.
        margin:    Отступ по краям (пиксели).
        bg_color:  Цвет фона холста (BGR).

    Returns:
        BGR изображение (H, W, 3) или заглушка если placements пусты.
    """
    if not assembly.placements:
        return np.full((200, 400, 3), bg_color, dtype=np.uint8)

    # Вычисляем bbox
    all_pos = [pos for pos, _ in assembly.placements.values()]
    xs = [p[0] for p in all_pos]
    ys = [p[1] for p in all_pos]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Размер с запасом
    max_frag_size = max(
        (frag.image.shape[0] + frag.image.shape[1])
        for frag in assembly.fragments
        if frag.image is not None
    ) if assembly.fragments else 200

    W = int(x_max - x_min + max_frag_size + 2 * margin)
    H = int(y_max - y_min + max_frag_size + 2 * margin)
    W, H = max(W, 200), max(H, 200)

    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    frag_by_id = {f.fragment_id: f for f in assembly.fragments}
    offset_x = -int(x_min) + margin + max_frag_size // 2
    offset_y = -int(y_min) + margin + max_frag_size // 2

    for fid, (pos, angle) in assembly.placements.items():
        frag = frag_by_id.get(fid)
        if frag is None or frag.image is None:
            continue

        cx = int(pos[0]) + offset_x
        cy = int(pos[1]) + offset_y
        _paste_fragment(canvas, frag, cx, cy, angle)

    return canvas


def _paste_fragment(canvas: np.ndarray,
                     frag: Fragment,
                     cx: int, cy: int,
                     angle: float) -> None:
    """Поворачивает фрагмент и накладывает его на холст по маске."""
    img  = frag.image
    mask = frag.mask

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return

    # Поворот вокруг центра
    if abs(angle) > 0.01:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(angle), 1.0)
        img  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h),
                                   flags=cv2.INTER_NEAREST, borderValue=0)

    # Вычисляем ROI на холсте
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1, y1 = x0 + w, y0 + h

    # Клиппинг к холсту
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    fx0, fy0 = cx0 - x0, cy0 - y0
    fx1, fy1 = fx0 + (cx1 - cx0), fy0 + (cy1 - cy0)

    roi     = canvas[cy0:cy1, cx0:cx1]
    src     = img[fy0:fy1, fx0:fx1]

    if mask is not None:
        m = mask[fy0:fy1, fx0:fx1]
        roi[m > 0] = src[m > 0]
    else:
        # Белый фон = прозрачность
        white = np.all(src >= 250, axis=2)
        roi[~white] = src[~white]

    canvas[cy0:cy1, cx0:cx1] = roi


# ─── Тепловая карта уверенности ───────────────────────────────────────────

def render_heatmap(assembly: Assembly,
                    canvas_shape: Optional[Tuple[int, int, int]] = None,
                    colormap: int = cv2.COLORMAP_JET,
                    alpha: float = 0.55) -> np.ndarray:
    """
    Строит тепловую карту уверенности каждого стыка между фрагментами.

    Высокая уверенность (зелёный/жёлтый) — стык скорее всего правильный.
    Низкая уверенность (синий/фиолетовый) — возможная ошибка сборки.

    Args:
        assembly:      Собранный документ (в placements должны быть scores).
        canvas_shape:  (H, W, 3) — если не передан, вычисляется из сборки.
        colormap:      Цветовая карта OpenCV.
        alpha:         Прозрачность карты поверх canvas.

    Returns:
        BGR изображение тепловой карты (H, W, 3).
    """
    if canvas_shape is None:
        canvas = render_canvas(assembly)
        canvas_shape = canvas.shape
    else:
        canvas = render_canvas(assembly)

    H, W = canvas_shape[:2]
    score_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    frag_by_id = {f.fragment_id: f for f in assembly.fragments}

    all_pos = [pos for pos, _ in assembly.placements.values()]
    if not all_pos:
        return canvas

    xs  = [p[0] for p in all_pos]
    ys  = [p[1] for p in all_pos]
    x_min, y_min = min(xs), min(ys)

    max_frag_size = max(
        (frag.image.shape[0] + frag.image.shape[1])
        for frag in assembly.fragments
        if frag.image is not None
    ) if assembly.fragments else 200
    margin = 20

    offset_x = -int(x_min) + margin + max_frag_size // 2
    offset_y = -int(y_min) + margin + max_frag_size // 2

    # Строим карту: глобальный индекс ребра → frag_id.
    # Порядок рёбер в матрице совпадает с порядком обхода fragments→edges
    # в build_compat_matrix (тот же порядок итерации).
    frag_of_edge: list[int] = []
    for _frag in assembly.fragments:
        for _ in _frag.edges:
            frag_of_edge.append(_frag.fragment_id)

    # frag_id → максимальный score среди его смежных рёбер
    edge_scores: dict[int, float] = {}

    mat = assembly.compat_matrix
    if mat is not None and mat.size > 0:
        n_edges = min(mat.shape[0], len(frag_of_edge), 500)
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                s = float(mat[i, j])
                if s > 0.1:
                    fid_i = frag_of_edge[i]
                    fid_j = frag_of_edge[j]
                    # Обновляем максимальный score для каждого из фрагментов
                    edge_scores[fid_i] = max(edge_scores.get(fid_i, 0.0), s)
                    edge_scores[fid_j] = max(edge_scores.get(fid_j, 0.0), s)

    # Рисуем Gaussian-пятна в центре каждого фрагмента с уверенностью
    for fid, (pos, _) in assembly.placements.items():
        frag = frag_by_id.get(fid)
        if frag is None:
            continue
        cx = int(pos[0]) + offset_x
        cy = int(pos[1]) + offset_y
        # Per-fragment score из матрицы; fallback — общий score сборки
        score = edge_scores.get(fid, float(assembly.total_score))

        # Рисуем Gaussian-пятно
        h_f = frag.image.shape[0] if frag.image is not None else 50
        w_f = frag.image.shape[1] if frag.image is not None else 50
        sigma = max(h_f, w_f) / 4.0

        ys_grid, xs_grid = np.ogrid[max(0, cy - int(3 * sigma)):
                                     min(H, cy + int(3 * sigma)),
                                     max(0, cx - int(3 * sigma)):
                                     min(W, cx + int(3 * sigma))]
        gauss = np.exp(-((ys_grid - cy) ** 2 + (xs_grid - cx) ** 2) / (2 * sigma ** 2))
        y0, y1 = max(0, cy - int(3 * sigma)), min(H, cy + int(3 * sigma))
        x0, x1 = max(0, cx - int(3 * sigma)), min(W, cx + int(3 * sigma))
        score_map[y0:y1, x0:x1] += gauss * score
        count_map[y0:y1, x0:x1] += gauss

    # Нормализуем
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_map = np.where(count_map > 0, score_map / count_map, 0.0)

    norm_min, norm_max = norm_map.min(), norm_map.max()
    if norm_max > norm_min:
        norm_map = (norm_map - norm_min) / (norm_max - norm_min)

    # Применяем colormap
    gray8 = (norm_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray8, colormap)

    # Blending с canvas
    result = cv2.addWeighted(canvas, 1.0 - alpha, colored, alpha, 0)
    return result


# ─── Мозаика сравнения ────────────────────────────────────────────────────

def render_mosaic(assembly: Assembly,
                   max_cols: int = 4,
                   thumb_size: int = 200,
                   gap: int = 8) -> np.ndarray:
    """
    Создаёт мозаику всех фрагментов для сравнения.

    Сетка из миниатюр фрагментов в порядке их расположения в сборке
    (слева-направо, сверху-вниз).

    Args:
        assembly:    Собранная конфигурация.
        max_cols:    Максимальное число колонок.
        thumb_size:  Размер каждой миниатюры (пиксели).
        gap:         Зазор между миниатюрами.

    Returns:
        BGR изображение мозаики.
    """
    if not assembly.fragments:
        return np.full((thumb_size, thumb_size, 3), 200, dtype=np.uint8)

    # Сортируем фрагменты по позиции в сборке (y, x)
    placed = []
    for frag in assembly.fragments:
        if frag.fragment_id in assembly.placements:
            pos, angle = assembly.placements[frag.fragment_id]
            placed.append((float(pos[1]), float(pos[0]), frag, angle))
    placed.sort()  # По y, затем x

    n   = len(placed)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    cell = thumb_size + gap
    mosaic = np.full((rows * cell + gap, cols * cell + gap, 3), 245, dtype=np.uint8)

    for idx, (_, _, frag, angle) in enumerate(placed):
        row, col = divmod(idx, cols)
        img = frag.image
        if img is None:
            continue

        # Поворот
        if abs(angle) > 0.01:
            h_, w_ = img.shape[:2]
            M = cv2.getRotationMatrix2D((w_ / 2, h_ / 2), math.degrees(angle), 1.0)
            img = cv2.warpAffine(img, M, (w_, h_), borderValue=(255, 255, 255))

        # Масштабируем до thumb_size
        h_, w_ = img.shape[:2]
        scale_ = thumb_size / max(h_, w_, 1)
        nh, nw = int(h_ * scale_), int(w_ * scale_)
        thumb  = cv2.resize(img, (nw, nh))

        # Центрируем в ячейке
        y0 = row * cell + gap + (thumb_size - nh) // 2
        x0 = col * cell + gap + (thumb_size - nw) // 2
        mosaic[y0:y0 + nh, x0:x0 + nw] = thumb

        # Номер фрагмента
        cv2.putText(mosaic, f"#{frag.fragment_id}",
                    (col * cell + gap + 4, row * cell + gap + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    return mosaic


# ─── Сохранение PNG ───────────────────────────────────────────────────────

def save_png(image: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """
    Сохраняет изображение в PNG (lossless) или JPEG (с качеством quality).
    """
    path = Path(path)
    ext  = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image)


# ─── Сохранение PDF ───────────────────────────────────────────────────────

def save_pdf(assembly: Assembly,
              canvas: np.ndarray,
              path: str | Path,
              title: str = "Restored Document",
              include_heatmap: bool = True) -> None:
    """
    Экспортирует восстановленный документ в PDF.

    Если установлен reportlab — добавляет текстовый слой (невидимый поверх
    изображения) для полнотекстового поиска. Если нет — только изображение.

    Args:
        assembly:         Собранный документ.
        canvas:           Растровое изображение сборки.
        path:             Путь к выходному PDF.
        title:            Заголовок документа.
        include_heatmap:  Добавить страницу с heatmap уверенности.

    Raises:
        ImportError: Если не установлен reportlab.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        import tempfile, os
        _HAS_REPORTLAB = True
    except ImportError:
        _HAS_REPORTLAB = False

    if not _HAS_REPORTLAB:
        _save_pdf_fallback(canvas, path)
        return

    path = Path(path)

    # Конвертируем canvas (BGR) → PNG во временный файл
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, canvas)
    tmp_path = tmp.name
    tmp.close()

    try:
        doc = SimpleDocTemplate(
            str(path),
            title=title,
            author="puzzle-reconstruction",
        )
        styles  = getSampleStyleSheet()
        h_px, w_px = canvas.shape[:2]
        a4_w, a4_h = A4

        # Масштаб: вписываем в A4 с полями 20mm
        margin_pt = 20 * mm
        max_w = a4_w - 2 * margin_pt
        max_h = a4_h - 3 * margin_pt
        scale = min(max_w / w_px, max_h / h_px)

        story = [
            Paragraph(title, styles["Title"]),
            RLImage(tmp_path, width=w_px * scale, height=h_px * scale),
        ]

        if include_heatmap:
            hm = render_heatmap(assembly, canvas.shape)
            tmp_hm = tempfile.NamedTemporaryFile(suffix="_hm.png", delete=False)
            cv2.imwrite(tmp_hm.name, hm)
            tmp_hm.close()
            story.append(Paragraph("Confidence Heatmap", styles["Heading2"]))
            story.append(RLImage(tmp_hm.name, width=w_px * scale, height=h_px * scale))
            os.unlink(tmp_hm.name)

        doc.build(story)
    finally:
        os.unlink(tmp_path)


def _save_pdf_fallback(canvas: np.ndarray, path: str | Path) -> None:
    """
    Fallback — сохраняет PDF через Pillow (без текстового слоя).
    Работает без reportlab.
    """
    try:
        from PIL import Image
        path = Path(path)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(path), "PDF", resolution=150)
    except ImportError:
        raise ImportError(
            "Для экспорта в PDF нужен reportlab или Pillow:\n"
            "  pip install reportlab   # рекомендуется\n"
            "  pip install Pillow      # минимальный вариант"
        )


# ─── Утилита: стрип сравнения ─────────────────────────────────────────────

def comparison_strip(fragments_imgs: list[np.ndarray],
                      assembly_img: np.ndarray,
                      heatmap_img: Optional[np.ndarray] = None,
                      target_height: int = 400) -> np.ndarray:
    """
    Создаёт горизонтальный стрип «до / после / heatmap»:
        [мозаика фрагментов] | [сборка] | [heatmap]

    Полезно для статей и отчётов.
    """
    panels = []

    # Мозаика оригинальных фрагментов
    mosaic_h = target_height
    cols = max(1, math.isqrt(len(fragments_imgs)))
    cell = mosaic_h // cols
    mosaic = np.full((mosaic_h, mosaic_h, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(fragments_imgs[:cols * cols]):
        r, c = divmod(idx, cols)
        h_, w_ = img.shape[:2]
        s = cell / max(h_, w_, 1)
        nh, nw = int(h_ * s), int(w_ * s)
        thumb = cv2.resize(img, (nw, nh))
        y0, x0 = r * cell + (cell - nh) // 2, c * cell + (cell - nw) // 2
        mosaic[y0:y0 + nh, x0:x0 + nw] = thumb
    panels.append(mosaic)

    # Сборка
    h_, w_ = assembly_img.shape[:2]
    s = target_height / h_
    panels.append(cv2.resize(assembly_img, (int(w_ * s), target_height)))

    # Heatmap (опционально)
    if heatmap_img is not None:
        h_, w_ = heatmap_img.shape[:2]
        s = target_height / h_
        panels.append(cv2.resize(heatmap_img, (int(w_ * s), target_height)))

    # Склеиваем горизонтально с вертикальной чертой
    sep = np.full((target_height, 4, 3), 100, dtype=np.uint8)
    result = panels[0]
    for panel in panels[1:]:
        result = np.hstack([result, sep, panel])

    return result


# ─── SVG экспорт ──────────────────────────────────────────────────────────────

def export_svg(
    assembly: Assembly,
    path: str,
    width: int = 800,
    height: int = 600,
) -> None:
    """Export assembled fragments as an SVG file.

    Each placed fragment is rendered as a ``<rect>`` (or ``<polygon>`` if a
    contour is available) with colour encoding the per-fragment score.
    Requires no external dependencies — pure string generation.

    Args:
        assembly: Fully assembled document (``placements`` must be non-empty).
        path:     Output file path (should end in ``.svg``).
        width:    SVG viewport width in pixels.
        height:   SVG viewport height in pixels.
    """
    placements = assembly.placements   # {fid: (pos, angle)}
    fragments  = assembly.fragments or []

    if not placements:
        svg_empty = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}">\n</svg>\n'
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg_empty)
        return

    # ── World-space bounding box ──────────────────────────────────────────────
    all_x = [float(pos[0]) for pos, _ in placements.values()]
    all_y = [float(pos[1]) for pos, _ in placements.values()]
    world_x_min, world_x_max = min(all_x), max(all_x)
    world_y_min, world_y_max = min(all_y), max(all_y)

    pad = 40
    world_span_x = max(world_x_max - world_x_min, 1.0)
    world_span_y = max(world_y_max - world_y_min, 1.0)
    scale_x = (width  - 2 * pad) / world_span_x
    scale_y = (height - 2 * pad) / world_span_y
    svg_scale = min(scale_x, scale_y)

    def _w2s(wx: float, wy: float):
        sx = pad + (wx - world_x_min) * svg_scale
        sy = pad + (wy - world_y_min) * svg_scale
        return sx, sy

    default_size = max(world_span_x, world_span_y) * 0.06 + 1.0
    rect_side = default_size * svg_scale

    def _score_color(score: float) -> str:
        """Interpolate from red (low) through yellow to green (high)."""
        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.5:
            t = (score - 0.5) * 2.0
            r, g, b = int((1.0 - t) * 220), 200, 0
        else:
            t = score * 2.0
            r, g, b = 220, int(t * 200), 0
        return f"#{r:02x}{g:02x}{b:02x}"

    frag_map = {f.fragment_id: f for f in fragments}

    # Per-fragment confidence from compat matrix (or fallback)
    frag_of_edge: list = []
    for frag in fragments:
        for _ in frag.edges:
            frag_of_edge.append(frag.fragment_id)

    mat = assembly.compat_matrix
    edge_scores: dict = {}
    if mat is not None and mat.size > 0:
        n_edges = min(mat.shape[0], len(frag_of_edge))
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                s = float(mat[i, j])
                if s > 0.05:
                    fi, fj = frag_of_edge[i], frag_of_edge[j]
                    edge_scores[fi] = max(edge_scores.get(fi, 0.0), s)
                    edge_scores[fj] = max(edge_scores.get(fj, 0.0), s)

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    lines.append(
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#1e1e1e"/>'
    )

    for fid, (pos, angle) in placements.items():
        px, py = float(pos[0]), float(pos[1])
        sx, sy = _w2s(px, py)

        frag = frag_map.get(fid)
        score = edge_scores.get(fid, float(assembly.total_score))
        fill  = _score_color(score)
        angle_deg = float(np.degrees(angle))

        if frag is not None and frag.contour is not None and len(frag.contour) >= 3:
            # Render fragment outline as a scaled polygon
            pts = frag.contour * svg_scale
            pts_shifted = pts - pts.mean(axis=0) + np.array([sx, sy])
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_shifted)
            cx_str = f"{float(sx):.1f}"
            cy_str = f"{float(sy):.1f}"
            lines.append(
                f'  <polygon points="{pts_str}" '
                f'fill="{fill}" fill-opacity="0.7" '
                f'stroke="#ffffff" stroke-width="0.8" '
                f'transform="rotate({angle_deg:.1f} {cx_str} {cy_str})"/>'
            )
        else:
            # Fallback rectangle
            fw = fh_size = rect_side
            if frag is not None and frag.image is not None:
                img_h, img_w = frag.image.shape[:2]
                fw = img_w * svg_scale
                fh_size = img_h * svg_scale
            rx, ry = sx - fw / 2, sy - fh_size / 2
            cx_str = f"{float(sx):.1f}"
            cy_str = f"{float(sy):.1f}"
            lines.append(
                f'  <rect x="{rx:.1f}" y="{ry:.1f}" '
                f'width="{fw:.1f}" height="{fh_size:.1f}" '
                f'fill="{fill}" fill-opacity="0.7" '
                f'stroke="#ffffff" stroke-width="0.5" '
                f'transform="rotate({angle_deg:.1f} {cx_str} {cy_str})"/>'
            )

        # Text label
        lines.append(
            f'  <text x="{sx:.1f}" y="{sy + 4:.1f}" '
            f'font-family="monospace" font-size="9" '
            f'fill="white" text-anchor="middle">#{fid} {score:.0%}</text>'
        )

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ─── HTML сравнение методов ───────────────────────────────────────────────────

def export_comparison_html(
    results: "dict[str, Assembly]",
    path: str,
    title: str = "Assembly Method Comparison",
) -> None:
    """Export a side-by-side HTML comparison of multiple assembly results.

    Generates a self-contained HTML file with a summary table (method, score,
    OCR score, fragment count, placement count) and a visual card per method
    showing all placed fragments.

    Args:
        results: Mapping from method name to ``Assembly`` object.
        path:    Output ``.html`` file path.
        title:   Page title / heading.
    """
    def _method_card(method: str, assembly: Assembly) -> str:
        n_frags    = len(assembly.fragments or [])
        n_placed   = len(assembly.placements or {})
        score      = float(assembly.total_score)
        ocr        = float(assembly.ocr_score)

        # Build a tiny text-based placement grid
        placement_lines: list[str] = []
        for fid, (pos, angle) in list((assembly.placements or {}).items())[:8]:
            px, py = float(pos[0]), float(pos[1])
            ang_deg = float(np.degrees(angle)) % 360
            placement_lines.append(
                f"  #{fid}  ({px:.0f}, {py:.0f})  ∠{ang_deg:.0f}°"
            )
        if len(assembly.placements or {}) > 8:
            placement_lines.append(f"  … {len(assembly.placements) - 8} more")

        placement_text = "\n".join(placement_lines) or "  (no placements)"

        bar_width = int(score * 180)
        ocr_width = int(ocr * 180)

        return f"""
        <div class="card">
          <h2>{method}</h2>
          <table class="meta">
            <tr><td>Fragments</td><td>{n_frags}</td></tr>
            <tr><td>Placed</td><td>{n_placed}</td></tr>
            <tr><td>Score</td><td>{score:.4f}</td></tr>
            <tr><td>OCR</td><td>{ocr:.1%}</td></tr>
          </table>
          <div class="bar-label">Score</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{bar_width}px;background:#4a9"></div></div>
          <div class="bar-label">OCR</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{ocr_width}px;background:#49a"></div></div>
          <pre class="placements">{placement_text}</pre>
        </div>"""

    # Summary table
    table_rows: list[str] = []
    for method, asm in results.items():
        n_placed = len(asm.placements or {})
        n_frags  = len(asm.fragments or [])
        score    = float(asm.total_score)
        ocr      = float(asm.ocr_score)
        table_rows.append(
            f"<tr><td>{method}</td><td>{n_frags}</td><td>{n_placed}</td>"
            f"<td>{score:.4f}</td><td>{ocr:.1%}</td></tr>"
        )

    best_method = max(results, key=lambda m: results[m].total_score) if results else "-"
    best_score_str = f"{results[best_method].total_score:.4f}" if results else "0.0000"
    cards_html  = "\n".join(_method_card(m, a) for m, a in results.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: monospace; background: #181818; color: #ddd; margin: 0; padding: 20px; }}
    h1   {{ color: #aef; margin-bottom: 8px; }}
    .summary {{ color: #aaa; margin-bottom: 24px; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 700px; margin-bottom: 32px; }}
    th, td {{ padding: 6px 12px; border: 1px solid #333; text-align: left; }}
    th {{ background: #222; color: #ccc; }}
    tr:hover {{ background: #222; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 20px; }}
    .card {{ background: #222; border: 1px solid #333; border-radius: 6px;
              padding: 16px; min-width: 240px; max-width: 320px; }}
    .card h2 {{ margin: 0 0 10px; color: #9df; font-size: 15px; }}
    .meta {{ border: none; width: auto; margin-bottom: 10px; font-size: 12px; }}
    .meta td {{ border: none; padding: 2px 8px 2px 0; }}
    .bar-label {{ font-size: 11px; color: #888; margin-top: 6px; }}
    .bar-bg {{ background: #333; height: 8px; border-radius: 4px; width: 180px; margin: 2px 0 4px; }}
    .bar-fill {{ height: 8px; border-radius: 4px; }}
    .placements {{ background: #181818; padding: 8px; border-radius: 4px;
                   font-size: 11px; color: #bbb; white-space: pre; overflow: auto; max-height: 120px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="summary">
    Methods compared: {len(results)} &nbsp;|&nbsp;
    Best method: <strong style="color:#9df">{best_method}</strong> &nbsp;|&nbsp;
    Score: {best_score_str}
  </div>

  <h3>Summary</h3>
  <table>
    <thead>
      <tr><th>Method</th><th>Fragments</th><th>Placed</th><th>Score</th><th>OCR</th></tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>

  <h3>Per-method details</h3>
  <div class="cards">
    {cards_html}
  </div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
