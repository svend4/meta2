"""
Верификация сборки через OCR: насколько текст «читается» через стыки.

Если два фрагмента правильно собраны, текст в области стыка должен
быть связным и иметь малое число ошибок OCR.
"""
import numpy as np
from typing import Optional

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def text_coherence_score(frag_a_image: np.ndarray,
                          frag_b_image: np.ndarray,
                          edge_a: "EdgeSignature",   # noqa: F821
                          edge_b: "EdgeSignature",   # noqa: F821
                          strip_width: int = 40) -> float:
    """
    Оценивает связность текста в области стыка двух фрагментов.

    Вырезает полосу пикселей вдоль края каждого фрагмента,
    объединяет их и запускает OCR. Оценивает долю «чистых» слов.

    Args:
        frag_a_image: BGR изображение фрагмента A.
        frag_b_image: BGR изображение фрагмента B.
        edge_a:       EdgeSignature края A.
        edge_b:       EdgeSignature края B.
        strip_width:  Ширина полосы вдоль края в пикселях.

    Returns:
        score ∈ [0, 1] — оценка связности (1 = идеальный текст).
    """
    if not _TESSERACT_AVAILABLE or not _CV2_AVAILABLE:
        return 0.5  # Нейтральная оценка если OCR недоступен

    try:
        strip_a = _extract_edge_strip(frag_a_image, edge_a, strip_width, side="right")
        strip_b = _extract_edge_strip(frag_b_image, edge_b, strip_width, side="left")

        # Склеиваем полосы
        if strip_a is None or strip_b is None:
            return 0.5

        combined = np.hstack([strip_a, strip_b])
        text = pytesseract.image_to_string(combined, lang="rus+eng")

        return _score_text_quality(text)

    except Exception:
        return 0.5


def verify_full_assembly(assembly: "Assembly",  # noqa: F821
                          lang: str = "rus+eng") -> float:
    """
    Запускает OCR на всей собранной картинке и возвращает оценку качества.
    """
    if not _TESSERACT_AVAILABLE:
        return 0.5

    canvas = render_assembly_image(assembly)
    if canvas is None:
        return 0.5

    try:
        text = pytesseract.image_to_string(canvas, lang=lang)
        return _score_text_quality(text)
    except Exception:
        return 0.5


def render_assembly_image(assembly: "Assembly") -> Optional[np.ndarray]:  # noqa: F821
    """
    Рендерит все фрагменты на общий холст согласно их позициям.
    Используется для OCR-верификации и финального экспорта.
    """
    if not _CV2_AVAILABLE:
        return None

    if not assembly.placements:
        return None

    # Определяем размер холста
    all_positions = [pos for pos, _ in assembly.placements.values()]
    all_pos = np.array(all_positions)
    min_xy = all_pos.min(axis=0) - 50
    max_xy = all_pos.max(axis=0) + 500  # Запас на размер фрагментов

    canvas_w = int(max_xy[0] - min_xy[0])
    canvas_h = int(max_xy[1] - min_xy[1])
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for frag in assembly.fragments:
        fid = frag.fragment_id
        if fid not in assembly.placements:
            continue
        pos, angle = assembly.placements[fid]
        x = int(pos[0] - min_xy[0])
        y = int(pos[1] - min_xy[1])

        img = frag.image
        if img is None:
            continue

        # Поворачиваем и вставляем
        rotated = _rotate_image(img, angle)
        h, w = rotated.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(canvas_w, x + w), min(canvas_h, y + h)
        rx1, ry1 = x1 - x, y1 - y
        rx2, ry2 = rx1 + (x2 - x1), ry1 + (y2 - y1)
        if rx2 > rx1 and ry2 > ry1:
            canvas[y1:y2, x1:x2] = rotated[ry1:ry2, rx1:rx2]

    return canvas


# ---------------------------------------------------------------------------

def _extract_edge_strip(image: np.ndarray, edge, width: int, side: str) -> Optional[np.ndarray]:
    """Вырезает полосу пикселей вдоль края фрагмента."""
    h, w = image.shape[:2]
    if side == "right":
        return image[:, max(0, w - width):]
    else:
        return image[:, :min(w, width)]


def _score_text_quality(text: str) -> float:
    """Оценивает качество OCR-текста: доля нераспознанных символов."""
    if not text or not text.strip():
        return 0.0
    total = len(text)
    garbage = sum(1 for c in text if not (c.isalpha() or c.isspace() or c in ".,!?-"))
    clean_ratio = 1.0 - garbage / total
    # Дополнительно: бонус за длинные слова (они с меньшей вероятностью случайны)
    words = text.split()
    long_words = sum(1 for w in words if len(w) >= 4)
    word_bonus = min(0.2, long_words / (len(words) + 1) * 0.3)
    return float(np.clip(clean_ratio + word_bonus, 0.0, 1.0))


def _rotate_image(image: np.ndarray, angle_rad: float) -> np.ndarray:
    if not _CV2_AVAILABLE:
        return image
    angle_deg = np.degrees(angle_rad)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
