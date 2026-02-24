"""Extra tests for puzzle_reconstruction/verification/text_coherence.py."""
from __future__ import annotations

import math

import pytest

from puzzle_reconstruction.verification.text_coherence import (
    NGramModel,
    TextCoherenceScorer,
    seam_bigram_score,
    word_boundary_score,
    build_ngram_model,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown fox ran across the field",
    "the lazy dog slept under the tree",
    "brown fox and lazy dog are friends",
]


# ─── NGramModel ──────────────────────────────────────────────────────────────

class TestNGramModelExtra:
    def test_fit(self):
        m = NGramModel(n=2).fit(_CORPUS)
        assert m.total > 0
        assert len(m.vocab) > 0

    def test_log_prob(self):
        m = NGramModel(n=2).fit(_CORPUS)
        lp = m.log_prob(("the",), "quick")
        assert lp < 0  # log probability is negative

    def test_log_prob_unseen(self):
        m = NGramModel(n=2).fit(_CORPUS)
        lp = m.log_prob(("zzz",), "yyy")
        assert lp < 0  # Still has a probability via Laplace smoothing

    def test_perplexity(self):
        m = NGramModel(n=2).fit(_CORPUS)
        pp = m.perplexity("the quick brown fox")
        assert pp >= 1.0
        assert math.isfinite(pp)

    def test_perplexity_empty(self):
        m = NGramModel(n=2).fit(_CORPUS)
        pp = m.perplexity("")
        assert pp == float("inf")

    def test_perplexity_short(self):
        m = NGramModel(n=2).fit(_CORPUS)
        pp = m.perplexity("fox")  # single word, can't form bigram
        assert pp == float("inf")

    def test_sentence_score(self):
        m = NGramModel(n=2).fit(_CORPUS)
        s = m.sentence_score("the quick brown fox")
        assert 0.0 <= s <= 1.0

    def test_sentence_score_empty(self):
        m = NGramModel(n=2).fit(_CORPUS)
        assert m.sentence_score("") == 0.0

    def test_trigram(self):
        m = NGramModel(n=3).fit(_CORPUS)
        assert m.total > 0
        pp = m.perplexity("the quick brown fox")
        assert pp >= 1.0

    def test_repr(self):
        m = NGramModel(n=2).fit(_CORPUS)
        s = repr(m)
        assert "n=2" in s
        assert "vocab=" in s

    def test_unfitted(self):
        m = NGramModel(n=2)
        assert m.total == 0
        pp = m.perplexity("hello world test")
        assert pp >= 1.0


# ─── TextCoherenceScorer ────────────────────────────────────────────────────

class TestTextCoherenceScorerExtra:
    def test_creation(self):
        s = TextCoherenceScorer(n=2, alpha=0.01)
        assert s._trained is False

    def test_train(self):
        s = TextCoherenceScorer(n=2)
        s.train(_CORPUS)
        assert s._trained is True

    def test_repr(self):
        s = TextCoherenceScorer(n=2, lang="eng")
        r = repr(s)
        assert "n=2" in r
        assert "eng" in r

    def test_batch_seam_scores_without_tesseract(self):
        s = TextCoherenceScorer(n=2)
        s.train(_CORPUS)
        # Without tesseract, seam_score returns 0.5
        import puzzle_reconstruction.verification.text_coherence as tc
        if not tc._TESSERACT:
            import numpy as np
            img = np.full((64, 64, 3), 128, dtype=np.uint8)
            scores = s.batch_seam_scores([(img, img)])
            assert len(scores) == 1
            assert scores[0] == pytest.approx(0.5)


# ─── seam_bigram_score ──────────────────────────────────────────────────────

class TestSeamBigramScoreExtra:
    def test_no_model(self):
        score = seam_bigram_score("hello world", "good morning", model=None)
        assert score == pytest.approx(0.5)

    def test_empty_left(self):
        score = seam_bigram_score("", "hello world")
        assert score == pytest.approx(0.0)

    def test_empty_right(self):
        score = seam_bigram_score("hello world", "")
        assert score == pytest.approx(0.0)

    def test_with_model(self):
        m = build_ngram_model(_CORPUS, n=2)
        score = seam_bigram_score("the quick", "brown fox", model=m)
        assert 0.0 <= score <= 1.0

    def test_coherent_vs_incoherent(self):
        m = build_ngram_model(_CORPUS, n=2)
        good = seam_bigram_score("the quick", "brown fox", model=m)
        bad = seam_bigram_score("zzz yyy", "xxx www", model=m)
        # Coherent text should generally score higher
        assert good >= bad


# ─── word_boundary_score ─────────────────────────────────────────────────────

class TestWordBoundaryScoreExtra:
    def test_full_words(self):
        score = word_boundary_score("hello world", "good morning")
        assert score == pytest.approx(1.0)

    def test_empty_left(self):
        score = word_boundary_score("", "hello")
        assert score == pytest.approx(0.5)

    def test_empty_right(self):
        score = word_boundary_score("hello", "")
        assert score == pytest.approx(0.5)

    def test_both_broken(self):
        score = word_boundary_score("the qu", "ck fox")
        # "qu" (2 chars) and "ck" (2 chars) look broken
        assert score < 1.0

    def test_one_broken(self):
        score = word_boundary_score("the qu", "brown fox jumps")
        assert score <= 0.5


# ─── build_ngram_model ──────────────────────────────────────────────────────

class TestBuildNgramModelExtra:
    def test_bigram(self):
        m = build_ngram_model(_CORPUS, n=2)
        assert isinstance(m, NGramModel)
        assert m.n == 2
        assert m.total > 0

    def test_trigram(self):
        m = build_ngram_model(_CORPUS, n=3)
        assert m.n == 3

    def test_custom_alpha(self):
        m = build_ngram_model(_CORPUS, n=2, alpha=0.1)
        assert m.alpha == pytest.approx(0.1)

    def test_empty_corpus(self):
        m = build_ngram_model([], n=2)
        assert m.total == 0
