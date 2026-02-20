#!/usr/bin/env python3
"""
Генератор смешанных фрагментов из нескольких документов.

Типичный сценарий для тестирования кластеризации:
  - N₁ фрагментов документа A перемешаны с N₂ фрагментами документа B.
  - Задача алгоритма: найти, какие фрагменты принадлежат одному документу.

Использование:
    # Смешать два существующих документа
    python tools/mix_documents.py \\
        --docs doc_a/ doc_b/ \\
        --output mixed/ \\
        --shuffle

    # Сгенерировать K тестовых документов и смешать их
    python tools/mix_documents.py \\
        --generate \\
        --n-docs 3 \\
        --pieces 4 \\
        --output mixed/ \\
        --ground-truth gt.json

Выходные данные:
    mixed/frag_000_doc0.png   — фрагмент 0 из документа 0
    mixed/frag_001_doc1.png   — фрагмент 1 из документа 1
    ...
    gt.json                   — {filename: doc_id} для оценки кластеризации
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document


# ─── Главные функции ──────────────────────────────────────────────────────

def mix_from_dirs(doc_dirs: List[Path],
                   output_dir: Path,
                   shuffle: bool = True,
                   seed: int = 42) -> Dict[str, int]:
    """
    Перемешивает фрагменты из нескольких директорий.

    Args:
        doc_dirs:   Список директорий с фрагментами (один документ = одна директория).
        output_dir: Выходная директория.
        shuffle:    Перемешать порядок файлов.
        seed:       Random seed для воспроизводимости.

    Returns:
        Ground-truth словарь {filename: doc_id}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    all_items: List[Tuple[Path, int]] = []
    for doc_id, d in enumerate(doc_dirs):
        files = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in exts)
        all_items.extend((f, doc_id) for f in files)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_items)

    ground_truth: Dict[str, int] = {}
    for idx, (src_path, doc_id) in enumerate(all_items):
        dst_name = f"frag_{idx:04d}_doc{doc_id}{src_path.suffix}"
        dst_path = output_dir / dst_name
        # Копируем файл
        import shutil
        shutil.copy2(src_path, dst_path)
        ground_truth[dst_name] = doc_id

    return ground_truth


def mix_from_generated(n_docs: int,
                         n_pieces: int,
                         output_dir: Path,
                         noise_level: float = 0.5,
                         shuffle: bool = True,
                         base_seed: int = 42) -> Dict[str, int]:
    """
    Генерирует n_docs синтетических документов, рвёт каждый на n_pieces фрагментов,
    перемешивает и сохраняет.

    Args:
        n_docs:      Число документов.
        n_pieces:    Число фрагментов каждого документа.
        output_dir:  Выходная директория.
        noise_level: Интенсивность рваного края.
        shuffle:     Перемешать порядок файлов.
        base_seed:   Базовый random seed.

    Returns:
        Ground-truth словарь {filename: doc_id}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_items: List[Tuple[np.ndarray, int, int]] = []  # (image, doc_id, frag_idx)

    for doc_id in range(n_docs):
        doc_seed  = base_seed + doc_id * 1000
        tear_seed = base_seed + doc_id * 100

        print(f"  Генерация документа {doc_id + 1}/{n_docs}...")
        document  = generate_test_document(800, 1000, seed=doc_seed)
        fragments = tear_document(document, n_pieces=n_pieces,
                                   noise_level=noise_level, seed=tear_seed)

        for frag_idx, frag in enumerate(fragments):
            all_items.append((frag, doc_id, frag_idx))

    if shuffle:
        rng = random.Random(base_seed)
        rng.shuffle(all_items)

    ground_truth: Dict[str, int] = {}
    for mixed_idx, (frag_img, doc_id, _) in enumerate(all_items):
        fname = f"frag_{mixed_idx:04d}_doc{doc_id}.png"
        fpath = output_dir / fname
        cv2.imwrite(str(fpath), frag_img)
        ground_truth[fname] = doc_id

    return ground_truth


# ─── Оценка кластеризации по GT ──────────────────────────────────────────

def evaluate_clustering(predicted: Dict[str, int],
                         ground_truth: Dict[str, int]) -> Dict[str, float]:
    """
    Оценивает качество кластеризации по ground-truth меткам.

    Метрики:
        purity      — доля фрагментов, правильно назначенных (arg max класса в кластере)
        rand_index  — доля правильных попарных решений (вместе / врозь)
        n_match     — сопоставление по Венгерскому алгоритму (если sklearn доступен)

    Args:
        predicted:    {filename: cluster_id}
        ground_truth: {filename: doc_id}

    Returns:
        Словарь метрик.
    """
    common = sorted(set(predicted) & set(ground_truth))
    if not common:
        return {"purity": 0.0, "rand_index": 0.0, "n_samples": 0}

    y_pred = np.array([predicted[f]    for f in common])
    y_true = np.array([ground_truth[f] for f in common])
    n      = len(common)

    # ── Purity ────────────────────────────────────────────────────────────
    from collections import Counter
    cluster_ids = set(y_pred)
    total_correct = 0
    for c in cluster_ids:
        mask = y_pred == c
        true_in_cluster = y_true[mask]
        most_common = Counter(true_in_cluster).most_common(1)[0][1]
        total_correct += most_common
    purity = total_correct / n

    # ── Rand Index ────────────────────────────────────────────────────────
    agree = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = (y_pred[i] == y_pred[j])
            same_true = (y_true[i] == y_true[j])
            if same_pred == same_true:
                agree += 1
    rand_index = agree / max(1, total_pairs)

    # ── Adjusted Rand Index (если sklearn) ────────────────────────────────
    ari = 0.0
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = float(adjusted_rand_score(y_true, y_pred))
    except ImportError:
        pass

    return {
        "purity":           purity,
        "rand_index":       rand_index,
        "adjusted_rand":    ari,
        "n_samples":        n,
        "n_clusters_pred":  int(len(set(y_pred))),
        "n_clusters_true":  int(len(set(y_true))),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Генерация смешанных фрагментов для тестирования кластеризации",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--docs", nargs="+",
                       help="Директории с уже нарезанными фрагментами")
    mode.add_argument("--generate", action="store_true",
                       help="Сгенерировать синтетические документы")

    parser.add_argument("--output",  "-o", required=True,
                         help="Директория для смешанных фрагментов")
    parser.add_argument("--n-docs",  type=int, default=3,
                         help="Число документов (только для --generate)")
    parser.add_argument("--pieces",  type=int, default=4,
                         help="Фрагментов на документ (только для --generate)")
    parser.add_argument("--noise",   type=float, default=0.5,
                         help="Интенсивность рваного края")
    parser.add_argument("--no-shuffle", action="store_true",
                         help="Не перемешивать фрагменты")
    parser.add_argument("--ground-truth", default=None,
                         help="Сохранить GT-словарь в JSON-файл")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    shuffle    = not args.no_shuffle

    print(f"Выходная директория: {output_dir}")

    if args.generate:
        print(f"Генерация {args.n_docs} документов × {args.pieces} фрагментов...")
        gt = mix_from_generated(
            n_docs=args.n_docs,
            n_pieces=args.pieces,
            output_dir=output_dir,
            noise_level=args.noise,
            shuffle=shuffle,
            base_seed=args.seed,
        )
    else:
        docs = [Path(d) for d in args.docs]
        print(f"Смешивание {len(docs)} директорий...")
        gt = mix_from_dirs(docs, output_dir, shuffle=shuffle, seed=args.seed)

    total = sum(1 for d in output_dir.iterdir() if d.suffix == ".png")
    print(f"Создано: {total} фрагментов из {len(set(gt.values()))} документов")

    if args.ground_truth:
        with open(args.ground_truth, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        print(f"Ground truth сохранён: {args.ground_truth}")

    return gt


if __name__ == "__main__":
    main()
