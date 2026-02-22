"""Агрегация доказательных сигналов для оценки совместимости пар фрагментов.

Модуль объединяет разнородные оценки совместимости (форма, цвет, текстура,
геометрия) в единое значение уверенности размещения.

Экспортирует:
    EvidenceConfig     — конфигурация агрегации (веса, пороги)
    EvidenceScore      — результат агрегации для одной пары
    aggregate_evidence — объединить словарь оценок в EvidenceScore
    weight_evidence    — применить веса к словарю оценок
    threshold_evidence — обнулить оценки ниже порога
    compute_confidence — вычислить уверенность из взвешенного среднего
    rank_by_evidence   — отсортировать пары по убыванию confidence
    batch_aggregate    — агрегировать список словарей оценок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── EvidenceConfig ───────────────────────────────────────────────────────────

@dataclass
class EvidenceConfig:
    """Конфигурация агрегации доказательных сигналов.

    Атрибуты:
        weights:        Словарь {channel: weight}. Отсутствующие каналы
                        получают вес 1.0.
        min_threshold:  Оценки ниже этого порога считаются нулевыми. ∈ [0, 1].
        require_all:    Если True — для успешного результата должны присутствовать
                        все каналы из weights.
        confidence_threshold: Минимальная confidence для is_confident. ∈ [0, 1].
    """
    weights: Dict[str, float] = field(default_factory=dict)
    min_threshold: float = 0.0
    require_all: bool = False
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_threshold <= 1.0):
            raise ValueError(
                f"min_threshold должен быть в [0, 1], получено {self.min_threshold}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold должен быть в [0, 1], "
                f"получено {self.confidence_threshold}"
            )
        for ch, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес канала '{ch}' должен быть >= 0, получено {w}"
                )


# ─── EvidenceScore ────────────────────────────────────────────────────────────

@dataclass
class EvidenceScore:
    """Результат агрегации доказательных сигналов.

    Атрибуты:
        pair_id:       Идентификатор пары (fragment_a, fragment_b).
        confidence:    Итоговая уверенность ∈ [0, 1].
        channel_scores: Исходные оценки по каналам после порогования.
        weighted_scores: Взвешенные оценки по каналам.
        n_channels:    Количество каналов, участвовавших в агрегации.
    """
    pair_id: Tuple[int, int]
    confidence: float
    channel_scores: Dict[str, float] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    n_channels: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence должна быть в [0, 1], получено {self.confidence}"
            )
        if self.n_channels < 0:
            raise ValueError(
                f"n_channels должен быть >= 0, получено {self.n_channels}"
            )
        a, b = self.pair_id
        if a < 0 or b < 0:
            raise ValueError(
                f"pair_id должен содержать неотрицательные индексы, получено {self.pair_id}"
            )

    @property
    def is_confident(self) -> bool:
        """True если confidence >= 0.5."""
        return self.confidence >= 0.5

    @property
    def dominant_channel(self) -> Optional[str]:
        """Канал с наибольшей взвешенной оценкой, или None если нет каналов."""
        if not self.weighted_scores:
            return None
        return max(self.weighted_scores, key=lambda k: self.weighted_scores[k])

    def summary(self) -> str:
        return (
            f"EvidenceScore(pair={self.pair_id}, confidence={self.confidence:.3f}, "
            f"channels={self.n_channels})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def weight_evidence(
    scores: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Применить веса к словарю оценок.

    Оценки умножаются на соответствующий вес. Каналы, отсутствующие в
    weights, получают вес 1.0.

    Args:
        scores:  Словарь {channel: score ∈ [0, 1]}.
        weights: Словарь {channel: weight >= 0}.

    Returns:
        Словарь {channel: score * weight}.

    Raises:
        ValueError: Если любое значение score не в [0, 1].
        ValueError: Если любой вес < 0.
    """
    for ch, s in scores.items():
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"Оценка канала '{ch}' должна быть в [0, 1], получено {s}"
            )
    for ch, w in weights.items():
        if w < 0.0:
            raise ValueError(
                f"Вес канала '{ch}' должен быть >= 0, получено {w}"
            )
    return {ch: s * weights.get(ch, 1.0) for ch, s in scores.items()}


def threshold_evidence(
    scores: Dict[str, float],
    min_threshold: float,
) -> Dict[str, float]:
    """Обнулить оценки ниже порога.

    Args:
        scores:        Словарь {channel: score}.
        min_threshold: Порог ∈ [0, 1]. Оценки < порога → 0.0.

    Returns:
        Словарь с обнулёнными низкими оценками.

    Raises:
        ValueError: Если min_threshold не в [0, 1].
    """
    if not (0.0 <= min_threshold <= 1.0):
        raise ValueError(
            f"min_threshold должен быть в [0, 1], получено {min_threshold}"
        )
    return {ch: (s if s >= min_threshold else 0.0) for ch, s in scores.items()}


def compute_confidence(
    weighted_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Вычислить уверенность как взвешенное среднее.

    Если нет каналов — возвращает 0.0.
    Если сумма весов равна 0 — возвращает 0.0.

    Args:
        weighted_scores: Словарь {channel: взвешенная оценка}.
        weights:         Словарь весов {channel: weight}.

    Returns:
        Уверенность ∈ [0, 1].
    """
    if not weighted_scores:
        return 0.0
    total_weight = sum(weights.get(ch, 1.0) for ch in weighted_scores)
    if total_weight == 0.0:
        return 0.0
    total = sum(weighted_scores.values())
    raw = total / total_weight
    return float(min(1.0, max(0.0, raw)))


def aggregate_evidence(
    scores: Dict[str, float],
    pair_id: Tuple[int, int] = (0, 0),
    cfg: Optional[EvidenceConfig] = None,
) -> EvidenceScore:
    """Объединить словарь оценок в EvidenceScore.

    Порядок обработки:
        1. threshold_evidence (если cfg.min_threshold > 0)
        2. weight_evidence
        3. compute_confidence
        4. Проверка require_all

    Args:
        scores:  Словарь {channel: score ∈ [0, 1]}.
        pair_id: Идентификатор пары (a, b).
        cfg:     Конфигурация (None → EvidenceConfig()).

    Returns:
        EvidenceScore с заполненными полями.

    Raises:
        ValueError: Если cfg.require_all и отсутствует хотя бы один канал
                    из cfg.weights.
    """
    if cfg is None:
        cfg = EvidenceConfig()

    # 1. Порогование
    thresholded = threshold_evidence(scores, cfg.min_threshold)

    # 2. Проверка require_all
    if cfg.require_all and cfg.weights:
        missing = [ch for ch in cfg.weights if ch not in thresholded or thresholded[ch] == 0.0]
        if missing:
            raise ValueError(
                f"require_all=True: отсутствуют каналы {missing}"
            )

    # 3. Взвешивание
    weighted = weight_evidence(thresholded, cfg.weights)

    # 4. Уверенность
    confidence = compute_confidence(weighted, cfg.weights)

    return EvidenceScore(
        pair_id=pair_id,
        confidence=confidence,
        channel_scores=dict(thresholded),
        weighted_scores=dict(weighted),
        n_channels=len(weighted),
    )


def rank_by_evidence(
    evidence_scores: List[EvidenceScore],
) -> List[EvidenceScore]:
    """Отсортировать список EvidenceScore по убыванию confidence.

    Args:
        evidence_scores: Список EvidenceScore.

    Returns:
        Отсортированный список (не изменяет исходный).
    """
    return sorted(evidence_scores, key=lambda e: e.confidence, reverse=True)


def batch_aggregate(
    batch: List[Dict[str, float]],
    pair_ids: Optional[List[Tuple[int, int]]] = None,
    cfg: Optional[EvidenceConfig] = None,
) -> List[EvidenceScore]:
    """Агрегировать список словарей оценок.

    Args:
        batch:    Список словарей {channel: score}.
        pair_ids: Список пар-идентификаторов. Если None — используются (i, i+1).
        cfg:      Конфигурация.

    Returns:
        Список EvidenceScore той же длины.

    Raises:
        ValueError: Если pair_ids задан и его длина не совпадает с batch.
    """
    if pair_ids is not None and len(pair_ids) != len(batch):
        raise ValueError(
            f"Длина pair_ids ({len(pair_ids)}) != длина batch ({len(batch)})"
        )
    results: List[EvidenceScore] = []
    for i, scores in enumerate(batch):
        pid = pair_ids[i] if pair_ids is not None else (i, i + 1)
        results.append(aggregate_evidence(scores, pair_id=pid, cfg=cfg))
    return results
