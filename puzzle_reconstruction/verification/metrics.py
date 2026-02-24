"""
Метрики качества восстановления документа.

Используются в бенчмарке, где известна ground-truth конфигурация
(получается из генератора синтетических данных).

Метрики:
    - Neighbor Accuracy (NA):  доля правильно определённых смежных пар
    - Direct Comparison (DC):  доля фрагментов на правильных позициях
    - Perfect Reconstruction:  True если NA == 1.0
    - Positional RMSE:         СКО ошибки позиционирования
    - Angular Error:           Средняя угловая ошибка в градусах
    - Edge Match Rate:         Доля краёв с правильным партнёром
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ReconstructionMetrics:
    """Результаты оценки качества сборки."""

    # Основные метрики
    neighbor_accuracy:  float   # [0,1] — доля правильных смежных пар
    direct_comparison:  float   # [0,1] — доля фрагментов на правильных позициях
    perfect:            bool    # True если NA = 1.0

    # Позиционные ошибки
    position_rmse:      float   # Пиксели (после нормализации)
    angular_error_deg:  float   # Средняя угловая ошибка, градусы

    # Дополнительные
    n_fragments:        int
    n_correct_pairs:    int
    n_total_pairs:      int
    edge_match_rate:    float   # [0,1]

    def summary(self) -> str:
        lines = [
            f"=== Метрики восстановления ({self.n_fragments} фрагментов) ===",
            f"  Neighbor Accuracy:   {self.neighbor_accuracy:.1%}",
            f"  Direct Comparison:   {self.direct_comparison:.1%}",
            f"  Perfect:             {'ДА ✓' if self.perfect else 'нет'}",
            f"  Position RMSE:       {self.position_rmse:.1f} px",
            f"  Angular Error:       {self.angular_error_deg:.1f}°",
            f"  Edge Match Rate:     {self.edge_match_rate:.1%}",
            f"  Correct pairs:       {self.n_correct_pairs}/{self.n_total_pairs}",
        ]
        return "\n".join(lines)


def evaluate_reconstruction(predicted: Dict[int, Tuple[np.ndarray, float]],
                             ground_truth: Dict[int, Tuple[np.ndarray, float]],
                             adjacency: Optional[List[Tuple[int, int]]] = None,
                             position_tolerance: float = 30.0,
                             angle_tolerance_deg: float = 10.0) -> ReconstructionMetrics:
    """
    Оценивает качество сборки относительно ground-truth.

    Args:
        predicted:          {frag_id: (pos, angle)} — предсказанная сборка.
        ground_truth:       {frag_id: (pos, angle)} — эталон.
        adjacency:          Список смежных пар (frag_id_a, frag_id_b).
                            Если None — смежность вычисляется из GT позиций.
        position_tolerance: Допуск на позицию (пиксели).
        angle_tolerance_deg: Допуск на угол поворота (градусы).

    Returns:
        ReconstructionMetrics
    """
    frag_ids = sorted(set(predicted.keys()) & set(ground_truth.keys()))
    n = len(frag_ids)

    if n == 0:
        return _zero_metrics(0)

    # ── Нормализуем обе конфигурации относительно первого фрагмента ──────
    pred_norm = _normalize_config(predicted, frag_ids)
    gt_norm   = _normalize_config(ground_truth, frag_ids)

    # ── 1. Direct Comparison ─────────────────────────────────────────────
    n_correct_pos = 0
    pos_errors    = []
    angle_errors  = []

    for fid in frag_ids:
        pred_pos,   pred_angle = pred_norm[fid]
        gt_pos,     gt_angle   = gt_norm[fid]

        pos_err = float(np.linalg.norm(pred_pos - gt_pos))
        pos_errors.append(pos_err)

        # Угловая ошибка с учётом симметрии (через 360°)
        ang_err = _angle_diff_deg(np.degrees(pred_angle), np.degrees(gt_angle))
        angle_errors.append(ang_err)

        if pos_err < position_tolerance and ang_err < angle_tolerance_deg:
            n_correct_pos += 1

    dc   = n_correct_pos / n
    rmse = float(np.sqrt(np.mean(np.array(pos_errors) ** 2)))
    ang_mean = float(np.mean(angle_errors))

    # ── 2. Neighbor Accuracy ─────────────────────────────────────────────
    if adjacency is None:
        adjacency = _compute_adjacency(gt_norm, threshold=position_tolerance * 3)

    n_total   = len(adjacency)
    n_correct = 0

    for (fid_a, fid_b) in adjacency:
        if fid_a not in pred_norm or fid_b not in pred_norm:
            continue
        # Смежные — если в предсказанной конфигурации они близко
        pred_pos_a = pred_norm[fid_a][0]
        pred_pos_b = pred_norm[fid_b][0]
        dist = float(np.linalg.norm(pred_pos_a - pred_pos_b))
        if dist < position_tolerance * 4:  # Мягкий критерий смежности
            n_correct += 1

    na = 1.0 if n_total == 0 else n_correct / n_total

    # ── 3. Edge Match Rate ────────────────────────────────────────────────
    emr = _compute_edge_match_rate(pred_norm, gt_norm, frag_ids, position_tolerance)

    return ReconstructionMetrics(
        neighbor_accuracy=na,
        direct_comparison=dc,
        perfect=(na == 1.0),
        position_rmse=rmse,
        angular_error_deg=ang_mean,
        n_fragments=n,
        n_correct_pairs=n_correct,
        n_total_pairs=n_total,
        edge_match_rate=emr,
    )


# ─── Функции сравнения нескольких методов ─────────────────────────────────

@dataclass
class BenchmarkResult:
    method:  str
    metrics: ReconstructionMetrics
    runtime_sec: float


def compare_methods(results: List[BenchmarkResult]) -> str:
    """Форматирует таблицу сравнения методов."""
    header = f"{'Метод':<12} {'NA':>7} {'DC':>7} {'RMSE':>8} {'Angle':>7} {'Perfect':>8} {'t,с':>6}"
    sep    = "─" * len(header)
    rows   = [header, sep]

    for r in sorted(results, key=lambda x: x.metrics.neighbor_accuracy, reverse=True):
        m = r.metrics
        perf = "ДА" if m.perfect else "нет"
        rows.append(
            f"{r.method:<12} "
            f"{m.neighbor_accuracy:>6.1%} "
            f"{m.direct_comparison:>6.1%} "
            f"{m.position_rmse:>7.1f} "
            f"{m.angular_error_deg:>6.1f}° "
            f"{perf:>8} "
            f"{r.runtime_sec:>5.1f}"
        )
    rows.append(sep)
    return "\n".join(rows)


# ─── Вспомогательные функции ──────────────────────────────────────────────

def _normalize_config(config: Dict[int, Tuple],
                       frag_ids: List[int]) -> Dict[int, Tuple[np.ndarray, float]]:
    """
    Нормализует конфигурацию: первый фрагмент помещается в начало координат,
    его угол = 0. Это позволяет сравнивать конфигурации независимо от
    абсолютного положения и ориентации.
    """
    if not frag_ids:
        return {}
    ref_id = frag_ids[0]
    ref_pos, ref_angle = config[ref_id]
    ref_pos   = np.asarray(ref_pos, dtype=float)
    ref_angle = float(ref_angle)

    c, s = np.cos(-ref_angle), np.sin(-ref_angle)
    R = np.array([[c, -s], [s, c]])

    result = {}
    for fid in frag_ids:
        if fid not in config:
            continue
        pos, angle = config[fid]
        pos = np.asarray(pos, dtype=float)
        # Поворачиваем и сдвигаем
        norm_pos   = R @ (pos - ref_pos)
        norm_angle = float(angle) - ref_angle
        result[fid] = (norm_pos, norm_angle)

    return result


def _compute_adjacency(gt_config: Dict[int, Tuple[np.ndarray, float]],
                        threshold: float) -> List[Tuple[int, int]]:
    """Определяет смежные пары по близости позиций в GT."""
    frag_ids = list(gt_config.keys())
    pairs    = []
    for i in range(len(frag_ids)):
        for j in range(i + 1, len(frag_ids)):
            pos_a = gt_config[frag_ids[i]][0]
            pos_b = gt_config[frag_ids[j]][0]
            if np.linalg.norm(pos_a - pos_b) < threshold:
                pairs.append((frag_ids[i], frag_ids[j]))
    return pairs


def _compute_edge_match_rate(pred: Dict, gt: Dict,
                              frag_ids: List[int],
                              tolerance: float) -> float:
    """
    Оценивает долю краёв, для которых правильный партнёр определён верно.
    Использует близость позиций для определения «правильного» стыка.
    """
    total, correct = 0, 0
    for fid in frag_ids:
        if fid not in pred or fid not in gt:
            continue
        pred_pos = pred[fid][0]
        gt_pos   = gt[fid][0]
        err = float(np.linalg.norm(pred_pos - gt_pos))
        total += 1
        if err < tolerance:
            correct += 1
    return correct / max(1, total)


def _angle_diff_deg(a: float, b: float) -> float:
    """Минимальная угловая разность с учётом симметрии (результат в [0, 180])."""
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def _zero_metrics(n: int) -> ReconstructionMetrics:
    return ReconstructionMetrics(
        neighbor_accuracy=0.0, direct_comparison=0.0, perfect=False,
        position_rmse=0.0, angular_error_deg=0.0,
        n_fragments=n, n_correct_pairs=0, n_total_pairs=0, edge_match_rate=0.0,
    )
