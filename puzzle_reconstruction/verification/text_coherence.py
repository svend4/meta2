"""
Лингвистическая когерентность текста в собранном документе.

В отличие от OCR-верификации (ocr.py), которая оценивает «читаемость» символов,
данный модуль анализирует *языковую связность*: насколько слова и n-граммы
по обе стороны каждого стыка соответствуют статистике естественного языка.

Алгоритм:
    1. OCR-обработка отдельных полос вдоль каждого стыка.
    2. Построение биграммных / триграммных частот из прочитанного текста.
    3. Оценка log-вероятности биграмм на стыке (score = 1 - perplexity_norm).
    4. Опциональная оценка на основе разрывов слов (частичные слова у края).

Классы:
    NGramModel          — простая n-граммная модель (Laplace smoothing)
    TextCoherenceScorer — вычисляет seam_score для пар фрагментов

Функции:
    score_assembly_coherence — средний балл стыков всей Assembly
    seam_bigram_score        — быстрый bigram-score двух строк текста
    word_boundary_score      — оценка полноты слов у края стыка
    build_ngram_model        — строит NGramModel из корпуса строк
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pytesseract
    _TESSERACT = True
except ImportError:
    _TESSERACT = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ─── NGramModel ───────────────────────────────────────────────────────────────

@dataclass
class NGramModel:
    """
    Простая n-граммная языковая модель с Laplace сглаживанием.

    Attributes:
        n:         Порядок (2 = биграммы, 3 = триграммы).
        alpha:     Параметр сглаживания (0 < alpha ≤ 1).
        counts:    {(w_1, …, w_{n-1}): Counter({w_n: count})} — счётчики.
        vocab:     Словарь всех токенов.
        total:     Общее число n-грамм в обучающем корпусе.
    """
    n:      int
    alpha:  float = 0.01
    counts: Dict[tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    vocab:  set = field(default_factory=set)
    total:  int = 0

    # ── Обучение ──────────────────────────────────────────────────────────

    def fit(self, sentences: Sequence[str]) -> "NGramModel":
        """
        Обучает модель на списке предложений (строк).

        Args:
            sentences: Список предложений / строк текста.

        Returns:
            self (для цепочки вызовов).
        """
        for sent in sentences:
            tokens = _tokenize(sent)
            if len(tokens) < self.n:
                continue
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                ctx = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.counts[ctx][word] += 1
                self.total += 1

        return self

    # ── Вероятности ───────────────────────────────────────────────────────

    def log_prob(self, context: tuple, word: str) -> float:
        """
        Логарифмическая вероятность P(word | context) с Laplace сглаживанием.

        Args:
            context: Кортеж из n-1 предыдущих токенов.
            word:    Следующий токен.

        Returns:
            log P(word | context) ∈ (-∞, 0].
        """
        ctx_counts = self.counts.get(context, Counter())
        count_cw   = ctx_counts.get(word, 0)
        count_c    = sum(ctx_counts.values())
        V          = max(len(self.vocab), 1)
        prob       = (count_cw + self.alpha) / (count_c + self.alpha * V)
        return math.log(prob)

    def perplexity(self, sentence: str) -> float:
        """
        Вычисляет перплексию предложения.

        PP = exp(-1/N * Σ log P(w_i | context)).

        Args:
            sentence: Строка текста.

        Returns:
            Перплексия ≥ 1 (меньше = лучше). Возвращает inf при N=0.
        """
        tokens = _tokenize(sentence)
        if len(tokens) < self.n:
            return float("inf")

        log_prob = 0.0
        n_grams  = 0
        for i in range(len(tokens) - self.n + 1):
            ctx  = tuple(tokens[i:i + self.n - 1])
            word = tokens[i + self.n - 1]
            log_prob += self.log_prob(ctx, word)
            n_grams  += 1

        if n_grams == 0:
            return float("inf")
        return math.exp(-log_prob / n_grams)

    def sentence_score(self, sentence: str) -> float:
        """
        Нормированная оценка ∈ [0, 1] (1 = хорошая когерентность).

        Перплексия → score через: score = 1 / (1 + log(PP) / log(V + 1)).
        """
        pp = self.perplexity(sentence)
        if not math.isfinite(pp):
            return 0.0
        V   = max(len(self.vocab), 2)
        raw = math.log(pp) / math.log(V + 1)
        return float(np.clip(1.0 - raw, 0.0, 1.0))

    def __repr__(self) -> str:
        return (f"NGramModel(n={self.n}, alpha={self.alpha}, "
                f"vocab={len(self.vocab)}, total={self.total})")


# ─── TextCoherenceScorer ──────────────────────────────────────────────────────

class TextCoherenceScorer:
    """
    Оценивает лингвистическую когерентность на стыках фрагментов.

    Использует NGramModel, обученную на корпусе (или на тексте документа),
    и оценивает биграммы, пересекающие каждый стык.

    Usage::

        scorer = TextCoherenceScorer(n=2, lang="rus+eng")
        scorer.train(corpus_sentences)
        score = scorer.seam_score(img_left, img_right, strip_width=40)
    """

    def __init__(self,
                 n:           int   = 2,
                 alpha:       float = 0.01,
                 lang:        str   = "rus+eng",
                 strip_width: int   = 40) -> None:
        self.model       = NGramModel(n=n, alpha=alpha)
        self.lang        = lang
        self.strip_width = strip_width
        self._trained    = False

    # ── Обучение ──────────────────────────────────────────────────────────

    def train(self, sentences: Sequence[str]) -> "TextCoherenceScorer":
        """Обучает внутреннюю NGramModel на переданных предложениях."""
        self.model.fit(sentences)
        self._trained = bool(self.model.total)
        return self

    def train_from_assembly(self,
                             assembly: "Assembly",  # noqa: F821
                             lang: Optional[str] = None) -> "TextCoherenceScorer":
        """
        OCR-сканирует всю Assembly и обучает модель на полученном тексте.
        """
        if not _TESSERACT:
            return self
        from .ocr import render_assembly_image
        canvas = render_assembly_image(assembly)
        if canvas is None:
            return self
        try:
            text = pytesseract.image_to_string(canvas, lang=lang or self.lang)
            sentences = [line for line in text.splitlines() if line.strip()]
            self.train(sentences)
        except Exception:
            pass
        return self

    # ── Оценка стыков ─────────────────────────────────────────────────────

    def seam_score(self,
                   img_left:    np.ndarray,
                   img_right:   np.ndarray,
                   strip_width: Optional[int] = None) -> float:
        """
        Оценивает лингвистическую связность стыка двух фрагментов.

        Вырезает правую полосу левого фрагмента и левую полосу правого,
        делает OCR каждой, затем вычисляет seam_bigram_score.

        Args:
            img_left:    BGR изображение левого фрагмента.
            img_right:   BGR изображение правого фрагмента.
            strip_width: Ширина полосы (пиксели). None → self.strip_width.

        Returns:
            score ∈ [0, 1] (0.5 если OCR недоступен).
        """
        if not _TESSERACT or not _CV2:
            return 0.5

        width = strip_width or self.strip_width
        try:
            text_left  = self._ocr_strip(img_left,  side="right", width=width)
            text_right = self._ocr_strip(img_right, side="left",  width=width)
            return seam_bigram_score(text_left, text_right, self.model)
        except Exception:
            return 0.5

    def batch_seam_scores(self,
                           pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
        """Параллельная (последовательная) оценка списка стыков."""
        return [self.seam_score(a, b) for a, b in pairs]

    # ── Вспомогательные методы ────────────────────────────────────────────

    def _ocr_strip(self, img: np.ndarray, side: str, width: int) -> str:
        """Вырезает полосу изображения и возвращает её OCR-текст."""
        h, w = img.shape[:2]
        if side == "right":
            strip = img[:, max(0, w - width):]
        else:
            strip = img[:, :min(w, width)]
        try:
            return pytesseract.image_to_string(strip, lang=self.lang)
        except Exception:
            return ""

    def __repr__(self) -> str:
        return (f"TextCoherenceScorer(n={self.model.n}, "
                f"trained={self._trained}, lang={self.lang!r})")


# ─── Функции модуля ───────────────────────────────────────────────────────────

def score_assembly_coherence(assembly:    "Assembly",   # noqa: F821
                              scorer:     TextCoherenceScorer,
                              threshold:  float = 150.0) -> float:
    """
    Вычисляет средний балл когерентности всех соседних стыков.

    Args:
        assembly:   Собранный документ.
        scorer:     Обученный TextCoherenceScorer.
        threshold:  Максимальное расстояние (px) для определения соседства.

    Returns:
        Средний score ∈ [0, 1] (0.5 если стыков нет или OCR недоступен).
    """
    if not _TESSERACT:
        return 0.5

    placed = list(assembly.placements.items())
    scores = []

    for i, (fid_i, (pos_i, _)) in enumerate(placed):
        frag_i = _get_fragment(assembly, fid_i)
        if frag_i is None or frag_i.image is None:
            continue
        for fid_j, (pos_j, _) in placed[i + 1:]:
            dist = float(np.linalg.norm(pos_i - pos_j))
            if dist > threshold:
                continue
            frag_j = _get_fragment(assembly, fid_j)
            if frag_j is None or frag_j.image is None:
                continue
            # Определяем, кто «левее»
            if pos_i[0] <= pos_j[0]:
                score = scorer.seam_score(frag_i.image, frag_j.image)
            else:
                score = scorer.seam_score(frag_j.image, frag_i.image)
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.5


def seam_bigram_score(text_left:  str,
                       text_right: str,
                       model:      Optional[NGramModel] = None) -> float:
    """
    Оценивает связность стыка по биграммам на границе двух текстов.

    Берёт последнее слово left и первое слово right и оценивает P(right|left).

    Args:
        text_left:  Текст с левой стороны стыка.
        text_right: Текст с правой стороны стыка.
        model:      NGramModel. None → вернуть 0.5 (нейтрально).

    Returns:
        score ∈ [0, 1].
    """
    toks_left  = _tokenize(text_left)
    toks_right = _tokenize(text_right)

    if not toks_left or not toks_right:
        return 0.0

    if model is None or not model.vocab:
        return 0.5

    # Последнее слово слева + первое слово справа
    boundary_text = " ".join(toks_left[-2:] + toks_right[:2])
    return model.sentence_score(boundary_text)


def word_boundary_score(text_left: str, text_right: str) -> float:
    """
    Оценивает, насколько целыми являются слова у края стыка.

    Слово считается «неполным», если оно начинается/кончается на символ
    без пробела рядом (типичный артефакт разрезанного слова).

    Возвращает:
        1.0 — оба края содержат только полные слова.
        0.0 — явно видны разорванные слова.
        0.5 — нейтрально (нет текста).
    """
    if not text_left.strip() or not text_right.strip():
        return 0.5

    left_last  = text_left.rstrip().split()[-1] if text_left.strip() else ""
    right_first = text_right.lstrip().split()[0] if text_right.strip() else ""

    # Признаки разрыва: строчные без пробела, дефис на краю
    left_broken  = bool(re.match(r"^[a-zа-яёA-ZА-ЯЁ]{1,3}$", left_last))
    right_broken = bool(re.match(r"^[a-zа-яёA-ZА-ЯЁ]{1,3}$", right_first))

    if left_broken and right_broken:
        return 0.2
    if left_broken or right_broken:
        return 0.5
    return 1.0


def build_ngram_model(corpus: Sequence[str],
                       n:     int   = 2,
                       alpha: float = 0.01) -> NGramModel:
    """
    Удобная фабричная функция для построения NGramModel.

    Args:
        corpus: Список строк (предложений).
        n:      Порядок модели.
        alpha:  Параметр сглаживания.

    Returns:
        Обученная NGramModel.
    """
    return NGramModel(n=n, alpha=alpha).fit(corpus)


# ─── Внутренние утилиты ───────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Делит текст на токены: слова в нижнем регистре, без знаков препинания."""
    pattern = re.compile(r"[a-zA-Zа-яёА-ЯЁ]+", re.UNICODE)
    return [w.lower() for w in pattern.findall(text) if len(w) >= 2]


def _get_fragment(assembly: "Assembly",  # noqa: F821
                   fid: int) -> Optional["Fragment"]:  # noqa: F821
    """Ищет фрагмент по fragment_id в Assembly.fragments."""
    for f in assembly.fragments:
        if f.fragment_id == fid:
            return f
    return None
