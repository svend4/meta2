"""Оценка качества совпадения пар фрагментов.

Модуль агрегирует различные сигналы (геометрия, текстура, частота, OCR)
в единую оценку совместимости двух фрагментов на уровне «пара фрагментов»
(в отличие от попарной оценки краёв).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── ScorerConfig ─────────────────────────────────────────────────────────────

@dataclass
class ScorerConfig:
    """Параметры оценщика совместимости.

    Атрибуты:
        weights:         Словарь {канал: вес} (все >= 0).
        min_score:       Нижний порог оценки (>= 0).
        max_score:       Верхний порог оценки (<= 1).
        normalize_input: Нормировать каждый канал перед агрегацией.
    """

    weights: Dict[str, float] = field(default_factory=lambda: {
        "geometry": 0.35,
        "texture": 0.25,
        "frequency": 0.20,
        "color": 0.20,
    })
    min_score: float = 0.0
    max_score: float = 1.0
    normalize_input: bool = True

    def __post_init__(self) -> None:
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес канала '{name}' должен быть >= 0, получено {w}"
                )
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )
        if self.max_score > 1.0:
            raise ValueError(
                f"max_score должен быть <= 1, получено {self.max_score}"
            )
        if self.min_score > self.max_score:
            raise ValueError(
                f"min_score ({self.min_score}) > max_score ({self.max_score})"
            )

    @property
    def total_weight(self) -> float:
        """Суммарный вес."""
        return float(sum(self.weights.values()))

    def normalized_weight(self, channel: str) -> float:
        """Нормированный вес канала [0, 1]."""
        tw = self.total_weight
        if tw < 1e-12:
            return 0.0
        return float(self.weights.get(channel, 0.0) / tw)


# ─── ChannelScore ─────────────────────────────────────────────────────────────

@dataclass
class ChannelScore:
    """Оценка одного канала сигнала.

    Атрибуты:
        channel: Имя канала.
        raw:     Исходная оценка (до нормализации).
        norm:    Нормированная оценка [0, 1].
        weight:  Вес канала.
    """

    channel: str
    raw: float
    norm: float
    weight: float

    def __post_init__(self) -> None:
        if not self.channel:
            raise ValueError("channel не должен быть пустым")
        if not (0.0 <= self.norm <= 1.0):
            raise ValueError(
                f"norm должен быть в [0, 1], получено {self.norm}"
            )
        if self.weight < 0.0:
            raise ValueError(
                f"weight должен быть >= 0, получено {self.weight}"
            )

    @property
    def contribution(self) -> float:
        """Взвешенный вклад в итоговую оценку."""
        return float(self.norm * self.weight)


# ─── MatchScore ───────────────────────────────────────────────────────────────

@dataclass
class MatchScore:
    """Итоговая оценка совместимости пары фрагментов.

    Атрибуты:
        id_a:      ID первого фрагмента.
        id_b:      ID второго фрагмента.
        score:     Итоговая оценка [0, 1].
        channels:  Словарь {канал: ChannelScore}.
        confident: True если score >= 0.7.
    """

    id_a: int
    id_b: int
    score: float
    channels: Dict[str, ChannelScore] = field(default_factory=dict)
    confident: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Упорядоченная пара (min, max)."""
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))

    @property
    def dominant_channel(self) -> Optional[str]:
        """Канал с наибольшим вкладом или None."""
        if not self.channels:
            return None
        return max(self.channels.values(),
                   key=lambda c: c.contribution).channel

    @property
    def n_channels(self) -> int:
        """Число активных каналов."""
        return len(self.channels)


# ─── _clip_score ──────────────────────────────────────────────────────────────

def _clip_score(s: float, cfg: ScorerConfig) -> float:
    """Обрезать оценку по [min_score, max_score]."""
    return float(np.clip(s, cfg.min_score, cfg.max_score))


# ─── score_channel ────────────────────────────────────────────────────────────

def score_channel(
    channel: str,
    raw_value: float,
    weight: float,
    normalize: bool = True,
) -> ChannelScore:
    """Создать ChannelScore для одного канала.

    Аргументы:
        channel:   Имя канала.
        raw_value: Исходное значение (любое).
        weight:    Вес (>= 0).
        normalize: Клипировать raw в [0, 1].

    Возвращает:
        ChannelScore.

    Исключения:
        ValueError: При weight < 0 или пустом channel.
    """
    if not channel:
        raise ValueError("channel не должен быть пустым")
    if weight < 0.0:
        raise ValueError(f"weight должен быть >= 0, получено {weight}")
    norm = float(np.clip(raw_value, 0.0, 1.0)) if normalize else float(raw_value)
    norm = float(np.clip(norm, 0.0, 1.0))
    return ChannelScore(channel=channel, raw=float(raw_value),
                        norm=norm, weight=weight)


# ─── compute_match_score ──────────────────────────────────────────────────────

def compute_match_score(
    id_a: int,
    id_b: int,
    channel_values: Dict[str, float],
    cfg: Optional[ScorerConfig] = None,
) -> MatchScore:
    """Вычислить итоговую оценку совместимости пары фрагментов.

    Аргументы:
        id_a:           ID первого фрагмента.
        id_b:           ID второго фрагмента.
        channel_values: Словарь {канал: значение_в_[0,1]}.
        cfg:            Параметры.

    Возвращает:
        MatchScore.

    Исключения:
        ValueError: Если channel_values пуст.
    """
    if cfg is None:
        cfg = ScorerConfig()
    if not channel_values:
        raise ValueError("channel_values не должен быть пустым")

    channels: Dict[str, ChannelScore] = {}
    weighted_sum = 0.0
    total_w = 0.0

    for ch, raw in channel_values.items():
        w = cfg.weights.get(ch, 0.0)
        cs = score_channel(ch, raw, w, cfg.normalize_input)
        channels[ch] = cs
        weighted_sum += cs.contribution
        total_w += w

    score = weighted_sum / (total_w + 1e-12)
    score = _clip_score(score, cfg)

    return MatchScore(
        id_a=id_a,
        id_b=id_b,
        score=score,
        channels=channels,
        confident=score >= 0.7,
    )


# ─── aggregate_match_scores ───────────────────────────────────────────────────

def aggregate_match_scores(
    scores: List[MatchScore],
) -> Optional[MatchScore]:
    """Объединить несколько MatchScore для одной пары в среднюю оценку.

    Аргументы:
        scores: Список MatchScore для одной и той же пары (id_a, id_b).

    Возвращает:
        Усреднённый MatchScore или None при пустом списке.
    """
    if not scores:
        return None

    id_a = scores[0].id_a
    id_b = scores[0].id_b
    mean_score = float(np.mean([s.score for s in scores]))
    mean_score = float(np.clip(mean_score, 0.0, 1.0))

    return MatchScore(id_a=id_a, id_b=id_b, score=mean_score,
                      confident=mean_score >= 0.7)


# ─── build_score_table ────────────────────────────────────────────────────────

def build_score_table(
    pairs: List[Tuple[int, int]],
    channel_values_map: Dict[Tuple[int, int], Dict[str, float]],
    cfg: Optional[ScorerConfig] = None,
) -> Dict[Tuple[int, int], MatchScore]:
    """Построить таблицу оценок для списка пар.

    Аргументы:
        pairs:               Список пар (id_a, id_b).
        channel_values_map:  Словарь {(id_a, id_b): {канал: значение}}.
        cfg:                 Параметры.

    Возвращает:
        Словарь {(min_id, max_id): MatchScore}.
    """
    if cfg is None:
        cfg = ScorerConfig()

    result: Dict[Tuple[int, int], MatchScore] = {}
    for a, b in pairs:
        key = (min(a, b), max(a, b))
        rev = (b, a)
        values = channel_values_map.get((a, b),
                  channel_values_map.get(rev, {}))
        if not values:
            ms = MatchScore(id_a=a, id_b=b, score=0.0, confident=False)
        else:
            ms = compute_match_score(a, b, values, cfg)
        result[key] = ms
    return result


# ─── filter_confident_pairs ───────────────────────────────────────────────────

def filter_confident_pairs(
    score_table: Dict[Tuple[int, int], MatchScore],
    threshold: float = 0.7,
) -> List[Tuple[int, int]]:
    """Вернуть пары с оценкой >= threshold.

    Аргументы:
        score_table: Таблица оценок.
        threshold:   Порог [0, 1].

    Возвращает:
        Список пар (min_id, max_id), отсортированный по убыванию score.

    Исключения:
        ValueError: Если threshold вне [0, 1].
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold должен быть в [0, 1], получено {threshold}"
        )
    filtered = [(k, v) for k, v in score_table.items()
                if v.score >= threshold]
    filtered.sort(key=lambda x: x[1].score, reverse=True)
    return [k for k, _ in filtered]
