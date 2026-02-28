"""
Интерактивный просмотрщик сборки пазла.

Интерфейс в стиле «Особое мнение»: цветовая карта уверенности стыков,
возможность перетаскивать фрагменты вручную, экспорт.

Управление:
    Мышь (ЛКМ)    — выбрать/перетащить фрагмент
    R              — повернуть выбранный фрагмент на 90°
    A              — авто-сборка (запустить SA)
    S              — сохранить результат
    Z              — отменить последнее перемещение
    +/-            — масштаб
    Esc / Q        — выход
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from ..models import Assembly, Fragment

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# Цвета по уровню уверенности стыка (BGR)
COLOR_HIGH   = (0,   200,  0)    # Зелёный  > 0.85
COLOR_MED    = (0,   180, 220)   # Жёлтый   0.65–0.85
COLOR_LOW    = (0,    80, 220)   # Красный  < 0.65
COLOR_SELECT = (255, 160,  0)    # Голубой  — выбранный фрагмент


class AssemblyViewer:
    """
    Интерактивный просмотрщик и редактор сборки.

    Использование:
        viewer = AssemblyViewer(assembly)
        viewer.run()          # Блокирующий цикл
        result = viewer.assembly  # Финальная сборка после редактирования
    """

    WINDOW = "Puzzle Reconstruction  |  LMB=drag  R=rotate  S=save  Q=exit"

    def __init__(self,
                 assembly: Assembly,
                 scale: float = 1.0,
                 canvas_size: Tuple[int, int] = (1400, 900),
                 output_path: str = "result_edited.png"):
        if not _CV2:
            raise ImportError("opencv-python не установлен: pip install opencv-python")

        self.assembly     = assembly
        self.scale        = scale
        self.canvas_w, self.canvas_h = canvas_size
        self.output_path  = output_path

        # Смещение «камеры» (для панорамирования)
        self.offset       = np.array([50.0, 50.0])

        # Состояние мыши
        self._drag_fid:   Optional[int]       = None
        self._drag_start: Optional[np.ndarray] = None
        self._frag_start: Optional[np.ndarray] = None

        # История для отмены (undo)
        self._history:    List[Dict] = []

        # Кэш рендеров фрагментов (повёрнутые изображения)
        self._cache:      Dict[int, np.ndarray] = {}

        self._running = False

    # -------------------------------------------------------------------------
    # Публичный API
    # -------------------------------------------------------------------------

    def run(self) -> Assembly:
        """
        Запускает интерактивный цикл. Блокирует до закрытия окна.
        Возвращает финальную Assembly.
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        self._running = True
        while self._running:
            frame = self._render_frame()
            cv2.imshow(self.WINDOW, frame)

            key = cv2.waitKey(30) & 0xFF
            self._handle_key(key)

        cv2.destroyWindow(self.WINDOW)
        return self.assembly

    def save(self, path: Optional[str] = None) -> None:
        """Сохраняет текущий вид в файл."""
        from ..verification.ocr import render_assembly_image
        canvas = render_assembly_image(self.assembly)
        if canvas is not None:
            out = path or self.output_path
            cv2.imwrite(out, canvas)
            print(f"Сохранено: {out}")

    # -------------------------------------------------------------------------
    # Рендеринг
    # -------------------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        """Рисует текущее состояние сборки на холсте."""
        frame = np.full((self.canvas_h, self.canvas_w, 3), 40, dtype=np.uint8)

        for frag in self.assembly.fragments:
            fid = frag.fragment_id
            if fid not in self.assembly.placements:
                continue
            pos, angle = self.assembly.placements[fid]
            screen_pos = self._world_to_screen(pos)

            # Рисуем изображение фрагмента
            img = self._get_rotated(frag, angle)
            if img is not None:
                self._blit(frame, img, screen_pos)

            # Рисуем контур с цветом уверенности
            score = self._fragment_confidence(fid)
            color = self._confidence_color(score)
            if fid == self._drag_fid:
                color = COLOR_SELECT

            self._draw_fragment_border(frame, frag, screen_pos, angle, color)

            # Подпись: ID и уверенность
            label = f"#{fid}  {score:.0%}"
            cv2.putText(frame, label,
                        (int(screen_pos[0]) + 5, int(screen_pos[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # HUD: статистика сборки
        self._draw_hud(frame)
        return frame

    def _blit(self, frame: np.ndarray, img: np.ndarray, pos: np.ndarray) -> None:
        """Накладывает img на frame с позицией pos (верхний-левый угол)."""
        h, w = img.shape[:2]
        x, y = int(pos[0]), int(pos[1])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.canvas_w, x + w), min(self.canvas_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        rx1, ry1 = x1 - x, y1 - y
        rx2, ry2 = rx1 + (x2 - x1), ry1 + (y2 - y1)
        roi = frame[y1:y2, x1:x2]
        src = img[ry1:ry2, rx1:rx2]
        # Прозрачность: только не-белые пиксели
        mask = np.any(src < 240, axis=2)
        roi[mask] = src[mask]

    def _draw_fragment_border(self, frame, frag, screen_pos, angle, color):
        """Рисует контур фрагмента."""
        if frag.contour is None or len(frag.contour) < 3:
            h = w = 80
            x, y = int(screen_pos[0]), int(screen_pos[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            return

        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        pts_rot = (R @ frag.contour.T).T
        pts_screen = pts_rot * self.scale + screen_pos
        pts_int = pts_screen.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_int], True, color, 2, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Рисует HUD с информацией о сборке."""
        n_total  = len(self.assembly.fragments)
        n_placed = len(self.assembly.placements)
        score    = self.assembly.total_score

        lines = [
            f"Фрагменты: {n_placed}/{n_total}",
            f"Score:     {score:.4f}",
            f"OCR:       {self.assembly.ocr_score:.1%}",
            "",
            "LMB=drag  R=rotate",
            "A=autofix  S=save",
            "+/-=zoom  Z=undo  Q=exit",
        ]
        y = 20
        for line in lines:
            cv2.putText(frame, line, (self.canvas_w - 220, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 18

    # -------------------------------------------------------------------------
    # Ввод
    # -------------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        mouse_world = self._screen_to_world(np.array([x, y]))

        if event == cv2.EVENT_LBUTTONDOWN:
            fid = self._pick_fragment(mouse_world)
            if fid is not None:
                self._drag_fid   = fid
                self._drag_start = mouse_world.copy()
                pos, angle = self.assembly.placements[fid]
                self._frag_start = np.asarray(pos).copy()
                self._push_history()

        elif event == cv2.EVENT_MOUSEMOVE and self._drag_fid is not None:
            delta = mouse_world - self._drag_start
            pos, angle = self.assembly.placements[self._drag_fid]
            new_pos = self._frag_start + delta
            self.assembly.placements[self._drag_fid] = (new_pos, angle)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_fid = None

    def _handle_key(self, key: int) -> None:
        if key in (ord('q'), ord('Q'), 27):
            self._running = False

        elif key in (ord('r'), ord('R')) and self._drag_fid is not None:
            pos, angle = self.assembly.placements[self._drag_fid]
            self.assembly.placements[self._drag_fid] = (pos, angle + np.pi / 2)
            self._cache.pop(self._drag_fid, None)

        elif key in (ord('s'), ord('S')):
            self.save()

        elif key == ord('+') or key == ord('='):
            self.scale = min(4.0, self.scale * 1.2)
            self._cache.clear()

        elif key == ord('-'):
            self.scale = max(0.1, self.scale / 1.2)
            self._cache.clear()

        elif key in (ord('z'), ord('Z')):
            self._undo()

        elif key in (ord('a'), ord('A')):
            self._auto_refine()

    # -------------------------------------------------------------------------
    # Вспомогательные методы
    # -------------------------------------------------------------------------

    def _world_to_screen(self, pos: np.ndarray) -> np.ndarray:
        return pos * self.scale + self.offset

    def _screen_to_world(self, pos: np.ndarray) -> np.ndarray:
        return (pos - self.offset) / self.scale

    def _pick_fragment(self, world_pos: np.ndarray) -> Optional[int]:
        """Возвращает fragment_id, если мировая точка попадает в фрагмент."""
        for frag in reversed(self.assembly.fragments):
            fid = frag.fragment_id
            if fid not in self.assembly.placements:
                continue
            pos, angle = self.assembly.placements[fid]
            pos = np.asarray(pos)
            # Грубая проверка: попадание в bounding box
            if frag.contour is not None and len(frag.contour) > 0:
                c, s = np.cos(-angle), np.sin(-angle)
                R = np.array([[c, -s], [s, c]])
                local = R @ (world_pos - pos)
                bbox_min = frag.contour.min(axis=0)
                bbox_max = frag.contour.max(axis=0)
                if (bbox_min[0] <= local[0] <= bbox_max[0] and
                        bbox_min[1] <= local[1] <= bbox_max[1]):
                    return fid
            else:
                if np.linalg.norm(world_pos - pos) < 60:
                    return fid
        return None

    def _fragment_confidence(self, frag_id: int) -> float:
        """
        Средняя оценка совместимости краёв фрагмента с соседями.

        Использует матрицу совместимости: для каждого ребра фрагмента берёт
        максимальный score среди всех пар с рёбрами других фрагментов, затем
        усредняет по рёбрам. Fallback — нормированный общий score сборки.
        """
        frags = self.assembly.fragments or []
        # Проверяем что фрагмент существует
        target = next((f for f in frags if f.fragment_id == frag_id), None)
        if target is None or not target.edges:
            return 0.5

        mat = self.assembly.compat_matrix
        if mat is None or mat.size == 0:
            return 0.5

        # Строим карту: глобальный индекс ребра → frag_id (тот же порядок, что
        # в build_compat_matrix: итерация по fragments→edges)
        frag_of_edge: list = []
        for f in frags:
            for _ in f.edges:
                frag_of_edge.append(f.fragment_id)

        # Собираем индексы рёбер, принадлежащих запрошенному фрагменту
        own_indices = [idx for idx, fid in enumerate(frag_of_edge)
                       if fid == frag_id and idx < mat.shape[0]]

        if not own_indices:
            return 0.5

        # Для каждого ребра фрагмента — лучший score с чужим ребром
        best_scores = []
        for idx in own_indices:
            row = mat[idx].copy()
            # Обнуляем стыки с рёбрами того же фрагмента
            for other_idx, fid in enumerate(frag_of_edge):
                if fid == frag_id and other_idx < len(row):
                    row[other_idx] = 0.0
            best = float(np.max(row))
            if best > 0.0:
                best_scores.append(best)

        if best_scores:
            return float(np.mean(best_scores))
        return self.assembly.total_score / max(1, len(frags))

    @staticmethod
    def _confidence_color(score: float) -> Tuple[int, int, int]:
        if score > 0.85:
            return COLOR_HIGH
        elif score > 0.65:
            return COLOR_MED
        return COLOR_LOW

    def _get_rotated(self, frag: Fragment, angle: float) -> Optional[np.ndarray]:
        """Возвращает повёрнутое изображение фрагмента (с кэшированием)."""
        if frag.image is None:
            return None
        cache_key = frag.fragment_id
        # Проверяем кэш (упрощённо: не проверяем изменение угла)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        img = frag.image
        h, w = img.shape[:2]
        # Масштабирование
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Поворот
        angle_deg = -np.degrees(angle)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(scaled, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        self._cache[cache_key] = rotated
        return rotated

    def _push_history(self) -> None:
        """Сохраняет текущие позиции для возможности отмены."""
        snapshot = {fid: (pos.copy(), angle)
                    for fid, (pos, angle) in self.assembly.placements.items()}
        self._history.append(snapshot)
        if len(self._history) > 20:
            self._history.pop(0)

    def _undo(self) -> None:
        """Отменяет последнее действие."""
        if not self._history:
            return
        snapshot = self._history.pop()
        self.assembly.placements = snapshot
        self._cache.clear()

    def _auto_refine(self) -> None:
        """Запускает быстрый SA для уточнения текущей сборки."""
        print("Авто-уточнение (SA)...")
        from ..assembly.annealing import simulated_annealing
        # Быстрый прогон с малым числом итераций
        refined = simulated_annealing(self.assembly, [],
                                       T_max=200.0, max_iter=1000)
        self.assembly.placements = refined.placements
        self.assembly.total_score = refined.total_score
        self._cache.clear()
        print(f"  Score после уточнения: {refined.total_score:.4f}")


def show(assembly: Assembly, **kwargs) -> Assembly:
    """
    Удобная функция-обёртка для запуска просмотрщика.

    Пример:
        final = show(assembly, scale=0.8, output_path="out.png")
    """
    viewer = AssemblyViewer(assembly, **kwargs)
    return viewer.run()


# =============================================================================
# Standalone visualization utilities (pure numpy, no matplotlib / PIL)
# =============================================================================

def compat_matrix_heatmap(
    matrix: np.ndarray,
    colormap: str = "viridis",
    cell_size: int = 16,
) -> np.ndarray:
    """Convert an N×N compatibility matrix to an RGB heatmap image.

    Parameters
    ----------
    matrix:
        2-D float array of shape (N, N).  Values need not be pre-normalised.
    colormap:
        ``"viridis"`` (blue→green→yellow) or ``"hot"`` (black→red→white).
    cell_size:
        Side length in pixels for each matrix cell.

    Returns
    -------
    uint8 RGB array of shape ``(N*cell_size, N*cell_size, 3)``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")

    n = matrix.shape[0]
    m = matrix.shape[1]

    # Normalise to [0, 1]
    vmin, vmax = float(matrix.min()), float(matrix.max())
    if vmax - vmin < 1e-12:
        normalised = np.zeros_like(matrix, dtype=np.float64)
    else:
        normalised = (matrix - vmin) / (vmax - vmin)

    # Build colour lookup tables (256 entries)
    lut = np.zeros((256, 3), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, 256)

    if colormap == "hot":
        # black → red → orange → yellow → white
        r = np.clip(t * 3.0,       0.0, 1.0)
        g = np.clip(t * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(t * 3.0 - 2.0, 0.0, 1.0)
    else:
        # "viridis"-like: deep-blue → teal → green → yellow-green
        # Key anchor colours (R, G, B) at t = 0, 0.25, 0.5, 0.75, 1.0
        anchors_t   = np.array([0.0,   0.25,  0.5,   0.75,  1.0])
        anchors_r   = np.array([0.267, 0.128, 0.134, 0.477, 0.993])
        anchors_g   = np.array([0.005, 0.563, 0.658, 0.821, 0.906])
        anchors_b   = np.array([0.329, 0.551, 0.391, 0.318, 0.144])
        r = np.interp(t, anchors_t, anchors_r)
        g = np.interp(t, anchors_t, anchors_g)
        b = np.interp(t, anchors_t, anchors_b)

    lut[:, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
    lut[:, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    lut[:, 2] = np.clip(b * 255, 0, 255).astype(np.uint8)

    # Map normalised values → colour indices
    indices = np.clip((normalised * 255).astype(int), 0, 255)  # (N, M)

    # Build output image by repeating each cell
    out_h = n * cell_size
    out_w = m * cell_size
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for row in range(n):
        for col in range(m):
            colour = lut[indices[row, col]]
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            img[y0:y1, x0:x1] = colour

    return img


def compare_edges(
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    height: int = 100,
    width: int = 200,
) -> np.ndarray:
    """Create a side-by-side comparison image of two 1-D edge profiles.

    The left half shows *edge_a* drawn in blue; the right half shows *edge_b*
    drawn in red.  A white vertical divider separates the two halves.

    Parameters
    ----------
    edge_a, edge_b:
        1-D (or ravelled) float arrays representing edge intensity profiles.
    height, width:
        Output image dimensions in pixels.

    Returns
    -------
    uint8 RGB array of shape ``(height, width, 3)``.
    """
    edge_a = np.asarray(edge_a, dtype=np.float64).ravel()
    edge_b = np.asarray(edge_b, dtype=np.float64).ravel()

    img = np.full((height, width, 3), 255, dtype=np.uint8)

    half_w = width // 2

    def _draw_curve(data: np.ndarray, x_offset: int, panel_w: int,
                    colour: Tuple[int, int, int]) -> None:
        """Draw a polyline for *data* within a panel starting at x_offset."""
        n = len(data)
        if n == 0:
            return
        vmin, vmax = float(data.min()), float(data.max())
        span = vmax - vmin if (vmax - vmin) > 1e-12 else 1.0

        # Normalised y values in [0, 1], then mapped to pixel rows
        norm = (data - vmin) / span          # 0 = min, 1 = max
        # Flip so that large values appear near top (row 0)
        row_vals = ((1.0 - norm) * (height - 1)).astype(int)
        row_vals = np.clip(row_vals, 0, height - 1)

        # x positions spread across panel_w
        x_vals = np.linspace(0, panel_w - 1, n).astype(int)
        x_vals = np.clip(x_vals, 0, panel_w - 1)

        # Draw line segments between consecutive points
        for i in range(len(x_vals) - 1):
            x0, y0 = int(x_vals[i]) + x_offset, int(row_vals[i])
            x1, y1 = int(x_vals[i + 1]) + x_offset, int(row_vals[i + 1])
            _draw_line(img, x0, y0, x1, y1, colour)

        # Also plot individual pixels for single-point curves
        if len(x_vals) == 1:
            r, c = int(row_vals[0]), int(x_vals[0]) + x_offset
            if 0 <= r < height and 0 <= c < width:
                img[r, c] = colour

    _draw_curve(edge_a, x_offset=0,      panel_w=half_w, colour=(0,   0,   220))
    _draw_curve(edge_b, x_offset=half_w, panel_w=half_w, colour=(220, 0,   0))

    # Vertical divider (white line, 1-pixel wide — already white background,
    # so draw a slightly darker grey divider for visibility)
    divider_col = half_w
    if 0 <= divider_col < width:
        img[:, divider_col] = (180, 180, 180)

    return img


def assembly_score_history(
    scores: List[float],
    width: int = 400,
    height: int = 200,
) -> np.ndarray:
    """Render a score-vs-iteration line chart as a uint8 RGB image.

    Draws labelled axes, a blue score line, and a green marker at the best
    (maximum) score.  Uses only numpy array operations.

    Parameters
    ----------
    scores:
        Sequence of scalar score values (one per iteration).
    width, height:
        Output image dimensions in pixels.

    Returns
    -------
    uint8 RGB array of shape ``(height, width, 3)``.
    """
    scores = [float(s) for s in scores]

    # Background: light grey
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # Margins for axes
    margin_left   = 40
    margin_bottom = 30
    margin_top    = 15
    margin_right  = 15

    plot_w = width  - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top

    if plot_w < 2 or plot_h < 2:
        return img

    # Draw axis lines (dark grey)
    axis_colour = (80, 80, 80)
    # Y-axis
    for row in range(margin_top, height - margin_bottom + 1):
        r, c = row, margin_left
        if 0 <= r < height and 0 <= c < width:
            img[r, c] = axis_colour
    # X-axis
    for col in range(margin_left, width - margin_right + 1):
        r, c = height - margin_bottom, col
        if 0 <= r < height and 0 <= c < width:
            img[r, c] = axis_colour

    if len(scores) == 0:
        return img

    # Score range
    vmin = min(scores)
    vmax = max(scores)
    span = (vmax - vmin) if abs(vmax - vmin) > 1e-12 else 1.0

    n = len(scores)

    def _score_to_pixel(idx: int, val: float) -> Tuple[int, int]:
        """Map (iteration index, score value) → (col, row) in image coords."""
        if n > 1:
            frac_x = idx / (n - 1)
        else:
            frac_x = 0.5
        col = margin_left + int(frac_x * (plot_w - 1))
        frac_y = (val - vmin) / span          # 0 = vmin, 1 = vmax
        row = (height - margin_bottom) - int(frac_y * (plot_h - 1))
        col = int(np.clip(col, margin_left, width - margin_right - 1))
        row = int(np.clip(row, margin_top,  height - margin_bottom - 1))
        return col, row

    # Draw score line (blue)
    line_colour = (30, 120, 220)
    pixels = [_score_to_pixel(i, s) for i, s in enumerate(scores)]
    for i in range(len(pixels) - 1):
        x0, y0 = pixels[i]
        x1, y1 = pixels[i + 1]
        _draw_line(img, x0, y0, x1, y1, line_colour)

    # Mark best point (green filled square, 5×5)
    best_idx = int(np.argmax(scores))
    bx, by = pixels[best_idx]
    best_colour = (0, 180, 0)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r, c = by + dr, bx + dc
            if 0 <= r < height and 0 <= c < width:
                img[r, c] = best_colour

    return img


# ---------------------------------------------------------------------------
# AssemblyViewer.export_svg  (method added outside class definition via monkey-
# patching approach is fragile; we add it as a proper method through direct
# extension of the source module instead — see below for the implementation
# that is injected into the class body above via the class-level assignment)
# ---------------------------------------------------------------------------

def _export_svg_impl(
    self: "AssemblyViewer",
    path: str,
    width: int = 800,
    height: int = 600,
) -> None:
    """Export current assembly layout as an SVG file.

    Each placed fragment is rendered as a coloured ``<rect>`` at its world-
    space position scaled to fit the SVG canvas.  Colour encodes the per-
    fragment confidence score: green for high confidence, red for low.
    A fragment-ID text label is drawn inside each rectangle.

    Parameters
    ----------
    path:
        File-system path for the output ``.svg`` file.
    width, height:
        SVG viewport dimensions in pixels.
    """
    placements = self.assembly.placements   # {fid: (pos, angle)}
    fragments  = self.assembly.fragments    # list[Fragment]

    if not placements:
        # Write an empty SVG
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}">\n</svg>\n'
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg)
        return

    # Compute world-space bounding box to fit into SVG canvas
    all_x, all_y = [], []
    for fid, (pos, _angle) in placements.items():
        px, py = float(pos[0]), float(pos[1])
        all_x.append(px)
        all_y.append(py)

    world_x_min, world_x_max = min(all_x), max(all_x)
    world_y_min, world_y_max = min(all_y), max(all_y)

    pad = 40  # padding in SVG pixels
    world_span_x = max(world_x_max - world_x_min, 1.0)
    world_span_y = max(world_y_max - world_y_min, 1.0)
    scale_x = (width  - 2 * pad) / world_span_x
    scale_y = (height - 2 * pad) / world_span_y
    svg_scale = min(scale_x, scale_y)

    def _world_to_svg(wx: float, wy: float) -> Tuple[float, float]:
        sx = pad + (wx - world_x_min) * svg_scale
        sy = pad + (wy - world_y_min) * svg_scale
        return sx, sy

    # Default fragment size in world space (fallback when no image is present)
    default_size = max(world_span_x, world_span_y) * 0.05 + 1.0
    rect_w_svg = default_size * svg_scale
    rect_h_svg = default_size * svg_scale

    def _confidence_rgb(score: float) -> str:
        """Return hex colour string interpolated red→yellow→green."""
        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.5:
            # yellow → green
            t = (score - 0.5) * 2.0
            r = int((1.0 - t) * 220)
            g = 200
            b = 0
        else:
            # red → yellow
            t = score * 2.0
            r = 220
            g = int(t * 200)
            b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    frag_map: Dict[int, "Fragment"] = {f.fragment_id: f for f in fragments}

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    # Background
    lines.append(
        f'  <rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="#282828"/>'
    )
    # Assembly outline (bounding box)
    ol_x, ol_y = _world_to_svg(world_x_min, world_y_min)
    ol_w = world_span_x * svg_scale
    ol_h = world_span_y * svg_scale
    lines.append(
        f'  <rect x="{ol_x:.1f}" y="{ol_y:.1f}" '
        f'width="{ol_w:.1f}" height="{ol_h:.1f}" '
        f'fill="none" stroke="#888888" stroke-width="1"/>'
    )

    for fid, (pos, angle) in placements.items():
        px, py = float(pos[0]), float(pos[1])
        sx, sy = _world_to_svg(px, py)

        frag = frag_map.get(fid)
        if frag is not None and frag.image is not None:
            img_h, img_w = frag.image.shape[:2]
            fw = img_w * svg_scale
            fh = img_h * svg_scale
        else:
            fw, fh = rect_w_svg, rect_h_svg

        score = self._fragment_confidence(fid)
        fill  = _confidence_rgb(score)
        angle_deg = float(np.degrees(angle))

        cx, cy = sx + fw / 2, sy + fh / 2
        transform = (
            f'transform="rotate({angle_deg:.1f} {cx:.1f} {cy:.1f})"'
        )
        lines.append(
            f'  <rect x="{sx:.1f}" y="{sy:.1f}" '
            f'width="{fw:.1f}" height="{fh:.1f}" '
            f'fill="{fill}" fill-opacity="0.75" '
            f'stroke="#ffffff" stroke-width="0.5" '
            f'{transform}/>'
        )
        # Text label
        label_x, label_y = cx, cy + 4
        lines.append(
            f'  <text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'font-family="monospace" font-size="10" '
            f'fill="white" text-anchor="middle" '
            f'{transform}>#{fid} {score:.0%}</text>'
        )

    lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# Attach export_svg to the AssemblyViewer class
AssemblyViewer.export_svg = _export_svg_impl


# ---------------------------------------------------------------------------
# Internal drawing primitive shared by compare_edges / assembly_score_history
# ---------------------------------------------------------------------------

def _draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    colour: Tuple[int, int, int],
) -> None:
    """Draw a 1-pixel Bresenham line on *img* (modified in-place).

    Parameters
    ----------
    img:    uint8 RGB image array (H, W, 3).
    x0, y0: Start pixel (column, row).
    x1, y1: End pixel (column, row).
    colour: (R, G, B) tuple.
    """
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            img[y0, x0] = colour
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


# ---------------------------------------------------------------------------
# Assembly animation
# ---------------------------------------------------------------------------

def _show_assembly_animated_impl(
    self: "AssemblyViewer",
    history: List[Assembly],
    fps: int = 10,
    output: Optional[str] = None,
) -> None:
    """Play back assembly evolution as a frame-by-frame animation.

    Loops through each Assembly snapshot in *history*, renders it with the
    viewer's own ``_render_frame`` logic (temporarily swapping
    ``self.assembly``), and either shows the result in a cv2 window or writes
    it to a video file — or both.

    A green progress bar is drawn along the bottom edge of each frame, and a
    ``Step N/M  score=…`` overlay is printed in the bottom-left corner.

    Parameters
    ----------
    history:
        Ordered list of Assembly snapshots (e.g. one per SA/genetic iteration).
        Each entry must have ``.placements`` and ``.fragments`` populated.
    fps:
        Display / recording frame rate.  Sensible range: 1–60.
    output:
        Optional path for a video file (``.avi`` or ``.mp4``).  When given the
        animation is written to disk; the cv2 preview window is still shown.
        Pass ``output="animation.mp4"`` to save an MP4.
    """
    if not history:
        return

    saved_assembly = self.assembly
    delay = max(1, int(1000 / fps))          # milliseconds per frame

    win_title = "Assembly Animation  |  Space=pause  Q=stop"
    writer: Optional[object] = None

    try:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, self.canvas_w, self.canvas_h)

        if output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output, fourcc, float(fps),
                (self.canvas_w, self.canvas_h),
            )

        paused = False
        idx = 0
        total = len(history)

        while idx < total:
            self.assembly = history[idx]
            self._cache.clear()
            frame = self._render_frame()

            # Progress bar (green, bottom 6 px)
            bar_w = max(1, int(self.canvas_w * (idx + 1) / total))
            cv2.rectangle(
                frame,
                (0, self.canvas_h - 6),
                (bar_w, self.canvas_h),
                (0, 200, 80), -1,
            )
            # Step / score overlay
            label = (
                f"Step {idx + 1}/{total}"
                f"  score={history[idx].total_score:.4f}"
            )
            cv2.putText(
                frame, label,
                (10, self.canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (220, 220, 220), 1, cv2.LINE_AA,
            )
            if paused:
                cv2.putText(
                    frame, "PAUSED",
                    (self.canvas_w // 2 - 40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow(win_title, frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(delay if not paused else 50) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord(' '):
                paused = not paused

            if not paused:
                idx += 1

    finally:
        self.assembly = saved_assembly
        self._cache.clear()
        if writer is not None:
            writer.release()
        cv2.destroyWindow(win_title)


# Attach show_assembly_animated to AssemblyViewer
AssemblyViewer.show_assembly_animated = _show_assembly_animated_impl


def animate_assembly(
    history: List[Assembly],
    fps: int = 10,
    canvas_size: Tuple[int, int] = (1400, 900),
    output: Optional[str] = None,
) -> None:
    """Convenience wrapper: animate an assembly history without a pre-built viewer.

    Creates a temporary :class:`AssemblyViewer` and delegates to
    :meth:`AssemblyViewer.show_assembly_animated`.

    Parameters
    ----------
    history:
        Ordered list of Assembly snapshots (e.g. from an SA/genetic run).
    fps:
        Playback frame rate.
    canvas_size:
        ``(width, height)`` of the display window in pixels.
    output:
        Optional path for saving the animation as a video file.

    Example
    -------
    >>> from puzzle_reconstruction.ui.viewer import animate_assembly
    >>> animate_assembly(sa_history, fps=15, output="sa_evolution.mp4")
    """
    if not history:
        return
    viewer = AssemblyViewer(history[0], canvas_size=canvas_size)
    viewer.show_assembly_animated(history, fps=fps, output=output)
