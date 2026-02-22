"""
Утилиты анализа и обработки текста для реконструкции документов.

Предоставляет инструменты для постобработки OCR-результатов, оценки
плотности текста, выравнивания текстовых блоков и сравнения текстовых
фрагментов в контексте пазловой реконструкции документов.

Экспортирует:
    TextConfig         — параметры анализа текста
    TextBlock          — один текстовый блок с метаданными
    clean_ocr_text     — очистка и нормализация OCR-строки
    estimate_text_density — оценить долю текстовых пикселей в регионе
    find_text_lines    — найти горизонтальные текстовые строки (проекционный профиль)
    segment_words      — сегментировать слова в бинарной строке
    compute_text_score — оценить «текстовость» бинарного изображения
    compare_text_blocks — сравнить два TextBlock по символьному сходству
    align_text_blocks  — упорядочить блоки по позиции (сверху-вниз, слева-направо)
    batch_clean_text   — пакетная очистка OCR-строк
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── TextConfig ───────────────────────────────────────────────────────────────

@dataclass
class TextConfig:
    """Параметры анализа текста.

    Attributes:
        min_word_gap:    Минимальный зазор (пикс.) между словами при сегментации.
        line_threshold:  Минимальная доля заполненных пикселей в строке (0–1),
                         чтобы строка считалась текстовой.
        strip_punct:     Удалять ли знаки пунктуации при очистке OCR.
        lowercase:       Приводить ли к нижнему регистру при очистке.
        min_line_height: Минимальная высота (пикс.) текстовой строки.
    """
    min_word_gap:    int   = 4
    line_threshold:  float = 0.05
    strip_punct:     bool  = False
    lowercase:       bool  = False
    min_line_height: int   = 4

    def __post_init__(self) -> None:
        if self.min_word_gap < 0:
            raise ValueError(
                f"min_word_gap must be >= 0, got {self.min_word_gap}"
            )
        if not (0.0 <= self.line_threshold <= 1.0):
            raise ValueError(
                f"line_threshold must be in [0, 1], got {self.line_threshold}"
            )
        if self.min_line_height < 1:
            raise ValueError(
                f"min_line_height must be >= 1, got {self.min_line_height}"
            )


# ─── TextBlock ────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """Один текстовый блок с позицией и содержимым.

    Attributes:
        text:      OCR-распознанный текст блока.
        x:         Левый край (пикс.).
        y:         Верхний край (пикс.).
        w:         Ширина (пикс.).
        h:         Высота (пикс.).
        confidence: Уверенность OCR-движка (0–1, -1 если неизвестно).
        source_id:  Идентификатор исходного фрагмента (опционально).
    """
    text:       str
    x:          int
    y:          int
    w:          int
    h:          int
    confidence: float = -1.0
    source_id:  Optional[int] = None

    def __post_init__(self) -> None:
        if self.w < 0:
            raise ValueError(f"w must be >= 0, got {self.w}")
        if self.h < 0:
            raise ValueError(f"h must be >= 0, got {self.h}")
        if self.confidence != -1.0 and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1] or -1, got {self.confidence}"
            )

    @property
    def area(self) -> int:
        """Площадь блока (пикс²)."""
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Центр блока (cx, cy)."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def n_chars(self) -> int:
        """Количество непробельных символов в тексте."""
        return sum(1 for c in self.text if not c.isspace())


# ─── clean_ocr_text ───────────────────────────────────────────────────────────

def clean_ocr_text(
    text: str,
    cfg: Optional[TextConfig] = None,
) -> str:
    """Очистить и нормализовать OCR-строку.

    Выполняет:
      1. Unicode NFC-нормализацию.
      2. Удаление управляющих символов (кроме \\n, \\t).
      3. Свёртку множественных пробелов в один.
      4. Опционально: удаление пунктуации (cfg.strip_punct).
      5. Опционально: приведение к нижнему регистру (cfg.lowercase).

    Args:
        text: Исходная строка OCR.
        cfg:  Конфигурация. None → TextConfig().

    Returns:
        Очищенная строка.
    """
    if cfg is None:
        cfg = TextConfig()

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Убрать управляющие символы (кроме \n и \t)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C")
        or ch in ("\n", "\t")
    )

    # 3. Пунктуация
    if cfg.strip_punct:
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # 4. Нижний регистр
    if cfg.lowercase:
        text = text.lower()

    # 5. Свернуть пробелы (оставить \n)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text


# ─── estimate_text_density ────────────────────────────────────────────────────

def estimate_text_density(
    binary: np.ndarray,
) -> float:
    """Оценить долю текстовых (ненулевых) пикселей в бинарном изображении.

    Args:
        binary: Бинарное изображение (H, W), dtype uint8; текст — ненулевые пиксели.

    Returns:
        Доля ненулевых пикселей ∈ [0, 1].

    Raises:
        ValueError: Если binary не 2-D.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f"binary must be 2-D, got ndim={binary.ndim}")
    total = binary.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(binary)) / float(total)


# ─── find_text_lines ──────────────────────────────────────────────────────────

def find_text_lines(
    binary: np.ndarray,
    cfg: Optional[TextConfig] = None,
) -> List[Tuple[int, int]]:
    """Найти горизонтальные текстовые строки методом горизонтальной проекции.

    Строит горизонтальный профиль суммы ненулевых пикселей по строкам,
    затем находит непрерывные отрезки, где доля заполнения >= cfg.line_threshold.

    Args:
        binary: Бинарное изображение (H, W), dtype uint8.
        cfg:    Конфигурация. None → TextConfig().

    Returns:
        Список кортежей (y_start, y_end) — начало и конец каждой текстовой строки.

    Raises:
        ValueError: Если binary не 2-D.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f"binary must be 2-D, got ndim={binary.ndim}")
    if cfg is None:
        cfg = TextConfig()

    h, w = binary.shape
    if w == 0 or h == 0:
        return []

    # Горизонтальная проекция: доля ненулевых пикселей по строке
    row_sums = (binary > 0).sum(axis=1).astype(np.float32) / float(w)
    is_text = row_sums >= cfg.line_threshold

    lines: List[Tuple[int, int]] = []
    in_line = False
    y0 = 0
    for y, flag in enumerate(is_text):
        if flag and not in_line:
            y0 = y
            in_line = True
        elif not flag and in_line:
            if (y - y0) >= cfg.min_line_height:
                lines.append((y0, y))
            in_line = False
    if in_line and (h - y0) >= cfg.min_line_height:
        lines.append((y0, h))

    return lines


# ─── segment_words ────────────────────────────────────────────────────────────

def segment_words(
    binary_line: np.ndarray,
    cfg: Optional[TextConfig] = None,
) -> List[Tuple[int, int]]:
    """Сегментировать слова в одной текстовой строке.

    Анализирует вертикальную проекцию строки и находит пробелы
    шириной >= cfg.min_word_gap, разделяя их как границы слов.

    Args:
        binary_line: Бинарная строка изображения (H, W), dtype uint8.
        cfg:         Конфигурация. None → TextConfig().

    Returns:
        Список кортежей (x_start, x_end) — границы каждого слова.

    Raises:
        ValueError: Если binary_line не 2-D.
    """
    binary_line = np.asarray(binary_line)
    if binary_line.ndim != 2:
        raise ValueError(f"binary_line must be 2-D, got ndim={binary_line.ndim}")
    if cfg is None:
        cfg = TextConfig()

    w = binary_line.shape[1]
    if w == 0:
        return []

    # Вертикальная проекция: есть ли хоть один ненулевой пиксель в столбце
    col_has_ink = (binary_line > 0).any(axis=0)

    words: List[Tuple[int, int]] = []
    in_word = False
    x0 = 0
    gap_start = 0

    for x in range(w):
        if col_has_ink[x]:
            if not in_word:
                # Проверить ширину предшествующего пробела
                if in_word is False and x > 0:
                    gap = x - gap_start
                    if gap >= cfg.min_word_gap and words:
                        # Закрыть предыдущее слово уже закрыто; начинаем новое
                        pass
                x0 = x
                in_word = True
        else:
            if in_word:
                words.append((x0, x))
                in_word = False
                gap_start = x

    if in_word:
        words.append((x0, w))

    return words


# ─── compute_text_score ───────────────────────────────────────────────────────

def compute_text_score(
    binary: np.ndarray,
    cfg: Optional[TextConfig] = None,
) -> float:
    """Оценить «текстовость» бинарного изображения.

    Объединяет:
      - общую плотность текста,
      - количество обнаруженных строк,
      - равномерность вертикального распределения.

    Args:
        binary: Бинарное изображение (H, W), dtype uint8.
        cfg:    Конфигурация. None → TextConfig().

    Returns:
        Оценка ∈ [0, 1]; чем выше — тем «текстовее» изображение.

    Raises:
        ValueError: Если binary не 2-D.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f"binary must be 2-D, got ndim={binary.ndim}")
    if cfg is None:
        cfg = TextConfig()

    density = estimate_text_density(binary)
    if density == 0.0:
        return 0.0

    lines = find_text_lines(binary, cfg=cfg)
    h = binary.shape[0]
    if h == 0:
        return 0.0

    # Нормированное количество строк (грубая верхняя граница)
    max_lines = max(1, h // max(1, cfg.min_line_height * 2))
    line_score = min(1.0, len(lines) / float(max_lines))

    # Равномерность: среднеквадратичное отклонение строковой проекции
    row_sums = (binary > 0).sum(axis=1).astype(np.float32)
    if row_sums.max() > 0:
        row_norm = row_sums / row_sums.max()
        uniformity = 1.0 - float(row_norm.std())
        uniformity = max(0.0, uniformity)
    else:
        uniformity = 0.0

    score = 0.4 * density + 0.4 * line_score + 0.2 * uniformity
    return float(np.clip(score, 0.0, 1.0))


# ─── compare_text_blocks ──────────────────────────────────────────────────────

def compare_text_blocks(
    block_a: TextBlock,
    block_b: TextBlock,
) -> float:
    """Сравнить два TextBlock по нормированному расстоянию редактирования.

    Использует символьную схожесть (1 - edit_distance / max_len).

    Args:
        block_a: Первый блок.
        block_b: Второй блок.

    Returns:
        Сходство ∈ [0, 1]; 1.0 — идентичные тексты.
    """
    a = block_a.text.strip()
    b = block_b.text.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _edit_distance(a, b)
    max_len = max(len(a), len(b))
    return float(1.0 - dist / max_len)


def _edit_distance(s: str, t: str) -> int:
    """Расстояние Левенштейна между строками s и t."""
    m, n = len(s), len(t)
    # dp[j] = edit distance between s[:i] and t[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s[i - 1] == t[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ─── align_text_blocks ────────────────────────────────────────────────────────

def align_text_blocks(
    blocks: List[TextBlock],
    primary: str = "top-to-bottom",
) -> List[TextBlock]:
    """Упорядочить текстовые блоки по позиции.

    Args:
        blocks:  Список TextBlock.
        primary: Порядок сортировки:
                 "top-to-bottom"   — сначала по Y (↓), затем по X (→),
                 "left-to-right"   — сначала по X (→), затем по Y (↓),
                 "reading-order"   — строки сверху-вниз, внутри строки — слева-направо
                                     (группировка по Y с допуском ½ высоты блока).

    Returns:
        Новый список блоков в указанном порядке.

    Raises:
        ValueError: Если primary не распознан.
    """
    if primary == "top-to-bottom":
        return sorted(blocks, key=lambda b: (b.y, b.x))
    elif primary == "left-to-right":
        return sorted(blocks, key=lambda b: (b.x, b.y))
    elif primary == "reading-order":
        if not blocks:
            return []
        # Группировать блоки в строки по Y-перекрытию
        sorted_y = sorted(blocks, key=lambda b: b.y)
        lines_groups: List[List[TextBlock]] = []
        current_line: List[TextBlock] = [sorted_y[0]]
        for blk in sorted_y[1:]:
            ref = current_line[-1]
            tolerance = max(1, ref.h // 2)
            if blk.y <= ref.y + tolerance:
                current_line.append(blk)
            else:
                lines_groups.append(sorted(current_line, key=lambda b: b.x))
                current_line = [blk]
        lines_groups.append(sorted(current_line, key=lambda b: b.x))
        return [b for line in lines_groups for b in line]
    else:
        raise ValueError(
            f"Unknown primary order: '{primary}'. "
            "Use 'top-to-bottom', 'left-to-right', or 'reading-order'."
        )


# ─── batch_clean_text ─────────────────────────────────────────────────────────

def batch_clean_text(
    texts: List[str],
    cfg: Optional[TextConfig] = None,
) -> List[str]:
    """Пакетная очистка списка OCR-строк.

    Args:
        texts: Список строк.
        cfg:   Конфигурация. None → TextConfig().

    Returns:
        Список очищенных строк (той же длины).
    """
    if cfg is None:
        cfg = TextConfig()
    return [clean_ocr_text(t, cfg=cfg) for t in texts]
