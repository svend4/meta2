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
