#!/usr/bin/env python3
"""
Восстановление разорванного документа из отсканированных фрагментов.

Использование:
    python main.py --input scans/ --output result.png
    python main.py --input scans/ --output result.png --alpha 0.5 --n-sides 4
    python main.py --input scans/ --output result.png --visualize

Алгоритм:
    1. Сегментация каждого фрагмента
    2. Описание краёв: Танграм (изнутри) + Фрактальная кромка (снаружи)
    3. Синтез EdgeSignature (виртуальная линия пересечения)
    4. Матрица совместимости всех краёв
    5. Жадная начальная сборка + Имитация отжига
    6. OCR-верификация
    7. Экспорт результата
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Ошибка: opencv-python не установлен. Запустите: pip install opencv-python")
    sys.exit(1)

from puzzle_reconstruction.models import Fragment
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.preprocessing.orientation import estimate_orientation
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.verification.ocr import (
    verify_full_assembly, render_assembly_image
)


def load_fragments(input_dir: Path) -> list[Fragment]:
    """Загружает все изображения из директории как фрагменты."""
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    if not paths:
        print(f"Нет изображений в {input_dir}")
        sys.exit(1)

    print(f"Найдено {len(paths)} фрагментов")
    fragments = []

    for idx, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [!] Не удалось загрузить {path.name}, пропускаю")
            continue
        frag = Fragment(fragment_id=idx, image=img, mask=None, contour=None)
        fragments.append(frag)
        print(f"  [{idx:3d}] {path.name}  ({img.shape[1]}×{img.shape[0]})")

    return fragments


def process_fragment(frag: Fragment,
                     seg_method: str = "otsu",
                     alpha: float = 0.5,
                     n_sides: int = 4) -> Fragment:
    """
    Полная обработка одного фрагмента:
    сегментация → контур → танграм → фрактал → подписи краёв.
    """
    # 1. Сегментация
    frag.mask = segment_fragment(frag.image, method=seg_method)

    # 2. Ориентация
    frag.image = _correct_orientation(frag.image, frag.mask)

    # 3. Контур
    frag.contour = extract_contour(frag.mask)

    # 4. Танграм-аппроксимация (внутренний многоугольник)
    frag.tangram = fit_tangram(frag.contour, mask=frag.mask)

    # 5. Фрактальная подпись (внешняя кромка)
    frag.fractal = compute_fractal_signature(frag.contour)

    # 6. Синтез EdgeSignature для каждого края
    frag.edges = build_edge_signatures(frag, alpha=alpha, n_sides=n_sides)

    return frag


def _correct_orientation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Поворачивает изображение в правильную ориентацию по тексту."""
    from puzzle_reconstruction.preprocessing.orientation import (
        estimate_orientation, rotate_to_upright
    )
    angle = estimate_orientation(image, mask)
    if abs(angle) > 0.05:  # Поворачиваем только если наклон значительный
        return rotate_to_upright(image, angle)
    return image


def run(input_dir: str,
        output_path: str,
        alpha: float = 0.5,
        n_sides: int = 4,
        seg_method: str = "otsu",
        match_threshold: float = 0.3,
        sa_iter: int = 5000,
        visualize: bool = False) -> None:

    t_start = time.time()
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # ── ЭТАП 1: Загрузка ──────────────────────────────────────────────────
    print("\n[1/6] Загрузка фрагментов...")
    fragments = load_fragments(input_dir)

    # ── ЭТАП 2: Обработка каждого фрагмента ───────────────────────────────
    print(f"\n[2/6] Обработка {len(fragments)} фрагментов...")
    processed = []
    for i, frag in enumerate(fragments):
        print(f"  Фрагмент {i + 1}/{len(fragments)}: "
              f"танграм + фрактал + CSS...", end=" ", flush=True)
        try:
            frag = process_fragment(frag, seg_method=seg_method,
                                    alpha=alpha, n_sides=n_sides)
            processed.append(frag)
            fd = (frag.fractal.fd_box + frag.fractal.fd_divider) / 2
            print(f"OK  (FD={fd:.3f}, "
                  f"форма={frag.tangram.shape_class.value}, "
                  f"краёв={len(frag.edges)})")
        except Exception as e:
            print(f"ОШИБКА: {e}")

    if not processed:
        print("Ни один фрагмент не обработан.")
        sys.exit(1)

    # ── ЭТАП 3: Матрица совместимости ─────────────────────────────────────
    print(f"\n[3/6] Построение матрицы совместимости...")
    compat_matrix, entries = build_compat_matrix(processed, threshold=match_threshold)
    print(f"  Всего пар: {len(entries)} (порог={match_threshold})")
    if entries:
        print(f"  Лучшая пара: score={entries[0].score:.4f}")

    # ── ЭТАП 4: Начальная сборка ───────────────────────────────────────────
    print(f"\n[4/6] Жадная начальная сборка...")
    assembly = greedy_assembly(processed, entries)
    assembly.compat_matrix = compat_matrix
    print(f"  Начальный score: {assembly.total_score:.4f}")

    # ── ЭТАП 5: Оптимизация ────────────────────────────────────────────────
    print(f"\n[5/6] Оптимизация (SA, {sa_iter} итераций)...")
    assembly = simulated_annealing(assembly, entries, max_iter=sa_iter)
    print(f"  Итоговый score: {assembly.total_score:.4f}")

    # ── ЭТАП 6: Верификация и экспорт ─────────────────────────────────────
    print(f"\n[6/6] Верификация и экспорт...")
    ocr_score = verify_full_assembly(assembly)
    assembly.ocr_score = ocr_score
    print(f"  OCR coherence: {ocr_score:.3f}")

    canvas = render_assembly_image(assembly)
    if canvas is not None:
        cv2.imwrite(str(output_path), canvas)
        print(f"  Сохранено: {output_path}")

        if visualize:
            _show_result(canvas, assembly)
    else:
        print("  [!] Не удалось создать изображение результата")

    elapsed = time.time() - t_start
    print(f"\nГотово за {elapsed:.1f} сек.")
    print(f"Уверенность сборки: {assembly.total_score:.1%}")
    print(f"Связность текста:   {ocr_score:.1%}")


def _show_result(canvas: np.ndarray, assembly) -> None:
    """Интерактивный просмотр результата (нажмите Q для выхода)."""
    import cv2
    h, w = canvas.shape[:2]
    scale = min(1.0, 1200 / w, 900 / h)
    preview = cv2.resize(canvas, (int(w * scale), int(h * scale)))

    # Рисуем стыки
    for frag in assembly.fragments:
        fid = frag.fragment_id
        if fid not in assembly.placements:
            continue
        pos, _ = assembly.placements[fid]
        x, y = int(pos[0] * scale), int(pos[1] * scale)
        cv2.circle(preview, (x, y), 5, (0, 200, 0), -1)
        cv2.putText(preview, str(fid), (x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 255), 1)

    cv2.imshow("Результат сборки (Q — выход)", preview)
    while True:
        if cv2.waitKey(100) & 0xFF in (ord('q'), ord('Q'), 27):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Восстановление разорванного документа",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Директория с отсканированными фрагментами")
    parser.add_argument("--output", "-o", default="result.png",
                        help="Путь для сохранения результата")
    parser.add_argument("--alpha",  type=float, default=0.5,
                        help="Вес танграма в синтезе EdgeSignature (0..1)")
    parser.add_argument("--n-sides", type=int, default=4,
                        help="Ожидаемое число краёв на фрагмент")
    parser.add_argument("--seg-method", default="otsu",
                        choices=["otsu", "adaptive", "grabcut"],
                        help="Метод сегментации")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Минимальная оценка совместимости")
    parser.add_argument("--sa-iter", type=int, default=5000,
                        help="Число итераций имитации отжига")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Показать результат в окне OpenCV")

    args = parser.parse_args()

    run(
        input_dir=args.input,
        output_path=args.output,
        alpha=args.alpha,
        n_sides=args.n_sides,
        seg_method=args.seg_method,
        match_threshold=args.threshold,
        sa_iter=args.sa_iter,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
