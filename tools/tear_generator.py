#!/usr/bin/env python3
"""
Генератор синтетических тестовых данных: «рвём» документ.

Берёт исходное изображение (или генерирует тестовый лист с текстом),
«рвёт» его на N фрагментов с реалистичными рваными краями,
сохраняет каждый фрагмент отдельным файлом.

Использование:
    python tools/tear_generator.py --input doc.png --output scans/ --pieces 6
    python tools/tear_generator.py --generate --output scans/ --pieces 4
    python tools/tear_generator.py --generate --output scans/ --pieces 9 --noise 0.8
"""
import argparse
import sys
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python")
    sys.exit(1)


def tear_document(image: np.ndarray,
                  n_pieces: int = 4,
                  noise_level: float = 0.5,
                  seed: int = 42) -> list[np.ndarray]:
    """
    «Рвёт» изображение на n_pieces фрагментов.

    Алгоритм:
    1. Делим изображение на сетку (rows × cols ≈ n_pieces).
    2. Для каждой границы генерируем «рваную» линию разрыва —
       Perlin-подобный шум или фрактальный сдвиг.
    3. Для каждого фрагмента создаём маску и вырезаем его.

    Args:
        image:       BGR изображение.
        n_pieces:    Желаемое число фрагментов.
        noise_level: Интенсивность рваного края (0 = ровно, 1 = сильно рвано).
        seed:        Random seed.

    Returns:
        Список BGR изображений фрагментов (с белым фоном вне маски).
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Определяем сетку разбиения
    cols, rows = _grid_shape(n_pieces)
    col_bounds = _divide_with_jitter(w, cols, rng, jitter=0.15)
    row_bounds = _divide_with_jitter(h, rows, rng, jitter=0.15)

    fragments = []

    for row in range(rows):
        for col in range(cols):
            x0, x1 = col_bounds[col], col_bounds[col + 1]
            y0, y1 = row_bounds[row], row_bounds[row + 1]

            # Создаём маску с рваными краями
            mask = _torn_mask(h, w, x0, x1, y0, y1,
                               noise_level=noise_level, rng=rng)

            # Вырезаем фрагмент: белый фон везде вне маски
            fragment = np.full_like(image, 255)
            fragment[mask > 0] = image[mask > 0]

            # Обрезаем по bounding box фрагмента
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            pad = 10
            x_lo = max(0, xs.min() - pad)
            x_hi = min(w, xs.max() + pad)
            y_lo = max(0, ys.min() - pad)
            y_hi = min(h, ys.max() + pad)
            fragment = fragment[y_lo:y_hi, x_lo:x_hi]

            fragments.append(fragment)

    return fragments


def generate_test_document(width: int = 800,
                            height: int = 1000,
                            font_scale: float = 0.55,
                            seed: int = 0) -> np.ndarray:
    """
    Генерирует синтетическую «страницу» с текстом для тестирования.
    """
    rng = np.random.RandomState(seed)
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Заголовок
    cv2.putText(img, "ТЕСТ ВОССТАНОВЛЕНИЯ ДОКУМЕНТА",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 2, cv2.LINE_AA)

    # Имитация строк текста случайными прямоугольниками
    y = 90
    while y < height - 40:
        x = 40
        row_h = int(rng.uniform(16, 20))
        while x < width - 40:
            word_w = int(rng.uniform(20, 90))
            darkness = int(rng.uniform(0, 60))
            cv2.rectangle(img, (x, y), (x + word_w, y + row_h),
                           (darkness, darkness, darkness), -1)
            x += word_w + int(rng.uniform(4, 12))
        y += row_h + int(rng.uniform(4, 8))

    # Горизонтальные «правила» (линии)
    for ly in range(80, height - 20, 25):
        alpha = rng.uniform(0.0, 0.15)
        color = int(255 * (1 - alpha))
        cv2.line(img, (40, ly), (width - 40, ly), (color, color, color), 1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Рваный край: генерация маски через фрактальный шум
# ─────────────────────────────────────────────────────────────────────────────

def _torn_mask(h: int, w: int,
               x0: int, x1: int, y0: int, y1: int,
               noise_level: float,
               rng: np.random.RandomState) -> np.ndarray:
    """
    Создаёт маску прямоугольного фрагмента с «рваными» краями.
    Каждая граница смещается на величину фрактального шума.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Базовый прямоугольник
    mask[y0:y1, x0:x1] = 255

    # Рваем каждый из 4 краёв
    amplitude = int(max(2, min(w, h) * noise_level * 0.06))

    # Верхний и нижний края (по оси X)
    for y_edge, is_top in [(y0, True), (y1 - 1, False)]:
        profile = _fractal_profile(x1 - x0, amplitude, rng)
        for dx, deviation in enumerate(profile):
            x = x0 + dx
            if 0 <= x < w:
                dev = int(round(deviation))
                if is_top:
                    # Рвём сверху: убираем пиксели выше (y_edge + dev)
                    y_cut = max(0, y_edge + dev)
                    mask[:y_cut, x] = 0
                else:
                    # Рвём снизу: убираем пиксели ниже (y_edge + dev)
                    y_cut = min(h, y_edge + dev)
                    mask[y_cut:, x] = 0

    # Левый и правый края (по оси Y)
    for x_edge, is_left in [(x0, True), (x1 - 1, False)]:
        profile = _fractal_profile(y1 - y0, amplitude, rng)
        for dy, deviation in enumerate(profile):
            y = y0 + dy
            if 0 <= y < h:
                dev = int(round(deviation))
                if is_left:
                    x_cut = max(0, x_edge + dev)
                    mask[y, :x_cut] = 0
                else:
                    x_cut = min(w, x_edge + dev)
                    mask[y, x_cut:] = 0

    return mask


def _fractal_profile(length: int,
                     amplitude: int,
                     rng: np.random.RandomState,
                     octaves: int = 5) -> np.ndarray:
    """
    Генерирует профиль рваного края через многооктавный шум (fBm).

    Сумма синусоид с убывающей амплитудой и возрастающей частотой
    имитирует фрактальную структуру линии разрыва.
    """
    profile = np.zeros(length)
    freq = 1.0
    amp  = float(amplitude)

    for _ in range(octaves):
        n_knots = max(2, int(length * freq / length * 4))
        knots   = rng.uniform(-amp, amp, n_knots)
        xs      = np.linspace(0, length - 1, n_knots)
        xnew    = np.arange(length)
        profile += np.interp(xnew, xs, knots)
        freq *= 2.0
        amp  *= 0.55

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _grid_shape(n: int) -> tuple[int, int]:
    """Возвращает (cols, rows) для ~n фрагментов, близко к квадрату."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return cols, rows


def _divide_with_jitter(total: int, n: int,
                         rng: np.random.RandomState,
                         jitter: float = 0.1) -> list[int]:
    """Делит [0, total] на n частей с небольшим случайным смещением границ."""
    step = total / n
    bounds = [0]
    for i in range(1, n):
        base = i * step
        shift = rng.uniform(-jitter * step, jitter * step)
        bounds.append(int(np.clip(base + shift, bounds[-1] + 20, total - 20)))
    bounds.append(total)
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Генератор синтетических тестовых данных (рваные фрагменты)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",    "-i",
                      help="Входное изображение документа")
    src.add_argument("--generate", "-g", action="store_true",
                      help="Сгенерировать тестовый документ с текстом")

    parser.add_argument("--output",  "-o", default="scans",
                         help="Директория для сохранения фрагментов")
    parser.add_argument("--pieces",  "-n", type=int, default=6,
                         help="Число фрагментов")
    parser.add_argument("--noise",   type=float, default=0.5,
                         help="Интенсивность рваного края [0..1]")
    parser.add_argument("--seed",    type=int, default=42,
                         help="Random seed")
    parser.add_argument("--width",   type=int, default=800,
                         help="Ширина генерируемого документа (только с --generate)")
    parser.add_argument("--height",  type=int, default=1000,
                         help="Высота генерируемого документа (только с --generate)")
    parser.add_argument("--save-original", action="store_true",
                         help="Сохранить оригинал для сравнения")

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем или генерируем исходный документ
    if args.generate:
        print(f"Генерирую тестовый документ {args.width}×{args.height}...")
        document = generate_test_document(args.width, args.height, seed=args.seed)
    else:
        document = cv2.imread(args.input)
        if document is None:
            print(f"Не удалось загрузить {args.input}")
            sys.exit(1)
        print(f"Загружен документ: {args.input}  ({document.shape[1]}×{document.shape[0]})")

    # Сохраняем оригинал
    if args.save_original:
        orig_path = out_dir / "original.png"
        cv2.imwrite(str(orig_path), document)
        print(f"Оригинал сохранён: {orig_path}")

    # Рвём документ
    print(f"Рву на {args.pieces} фрагментов (noise={args.noise})...")
    fragments = tear_document(document, n_pieces=args.pieces,
                               noise_level=args.noise, seed=args.seed)

    # Перемешиваем (имитация реальной ситуации)
    rng = np.random.RandomState(args.seed + 100)
    order = rng.permutation(len(fragments))

    saved = 0
    for new_idx, old_idx in enumerate(order):
        frag = fragments[old_idx]
        path = out_dir / f"fragment_{new_idx:03d}.png"
        cv2.imwrite(str(path), frag)
        h, w = frag.shape[:2]
        print(f"  {path.name}  ({w}×{h}px)")
        saved += 1

    print(f"\nСохранено {saved} фрагментов в {out_dir}/")
    print(f"\nТеперь запустите:")
    print(f"  python main.py --input {out_dir}/ --output result.png --visualize")


if __name__ == "__main__":
    main()
