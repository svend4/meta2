"""Tests for puzzle_reconstruction.verification.text_coherence"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.text_coherence import (
    NGramModel,
    TextCoherenceScorer,
    seam_bigram_score,
    word_boundary_score,
    build_ngram_model,
    _tokenize,
)


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a test sentence",
    "puzzle reconstruction document analysis",
    "fragments are placed in the correct order",
    "text coherence is important for reconstruction",
    "natural language processing helps identify seams",
]


# ─── _tokenize ────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = _tokenize("Hello World Test")
    assert tokens == ["hello", "world", "test"]


def test_tokenize_lowercase():
    tokens = _tokenize("ABC")
    assert tokens == ["abc"]


def test_tokenize_removes_short():
    tokens = _tokenize("a is of it the")
    # Only words >= 2 chars
    assert "a" not in tokens
    assert "is" in tokens


def test_tokenize_punctuation():
    tokens = _tokenize("hello, world!")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_numbers():
    # Numbers should be excluded
    tokens = _tokenize("abc 123 def")
    assert "abc" in tokens
    assert "def" in tokens


# ─── NGramModel ───────────────────────────────────────────────────────────────

def test_ngram_model_init():
    model = NGramModel(n=2, alpha=0.01)
    assert model.n == 2
    assert model.alpha == 0.01
    assert model.total == 0


def test_ngram_model_fit():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    assert model.total > 0
    assert len(model.vocab) > 0


def test_ngram_model_fit_returns_self():
    model = NGramModel(n=2)
    result = model.fit(CORPUS)
    assert result is model


def test_ngram_model_log_prob():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    lp = model.log_prob(("the",), "quick")
    assert lp < 0  # log probability is negative


def test_ngram_model_log_prob_unseen():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    # Unseen word with Laplace smoothing should still return finite value
    lp = model.log_prob(("xyzabc",), "qwerty")
    assert np.isfinite(lp)


def test_ngram_model_perplexity_basic():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    pp = model.perplexity("the quick brown fox")
    assert pp > 0
    assert np.isfinite(pp)


def test_ngram_model_perplexity_short():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    pp = model.perplexity("hi")
    assert pp == float("inf")


def test_ngram_model_sentence_score_range():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    score = model.sentence_score("the quick brown fox jumps")
    assert 0.0 <= score <= 1.0


def test_ngram_model_sentence_score_infinite_perplexity():
    model = NGramModel(n=2)
    # No training → perplexity will be high
    score = model.sentence_score("single")
    assert score == 0.0


def test_ngram_model_repr():
    model = NGramModel(n=2)
    model.fit(CORPUS)
    r = repr(model)
    assert "n=2" in r


def test_ngram_model_trigram():
    model = NGramModel(n=3)
    model.fit(CORPUS)
    assert model.n == 3
    lp = model.log_prob(("the", "quick"), "brown")
    assert np.isfinite(lp)


# ─── build_ngram_model ────────────────────────────────────────────────────────

def test_build_ngram_model_basic():
    model = build_ngram_model(CORPUS, n=2)
    assert isinstance(model, NGramModel)
    assert model.total > 0


def test_build_ngram_model_trigram():
    model = build_ngram_model(CORPUS, n=3, alpha=0.1)
    assert model.n == 3
    assert model.alpha == 0.1


def test_build_ngram_model_empty_corpus():
    model = build_ngram_model([], n=2)
    assert model.total == 0


# ─── seam_bigram_score ────────────────────────────────────────────────────────

def test_seam_bigram_score_no_model():
    score = seam_bigram_score("hello world", "test phrase", model=None)
    assert score == 0.5


def test_seam_bigram_score_empty_texts():
    model = build_ngram_model(CORPUS)
    score = seam_bigram_score("", "", model=model)
    assert score == 0.0


def test_seam_bigram_score_with_model():
    model = build_ngram_model(CORPUS)
    score = seam_bigram_score("the quick brown", "fox jumps over", model=model)
    assert 0.0 <= score <= 1.0


def test_seam_bigram_score_empty_model():
    model = NGramModel(n=2)  # untrained
    score = seam_bigram_score("hello world", "test phrase", model=model)
    assert score == 0.5


def test_seam_bigram_score_left_empty():
    model = build_ngram_model(CORPUS)
    score = seam_bigram_score("", "hello world", model=model)
    assert score == 0.0


# ─── word_boundary_score ──────────────────────────────────────────────────────

def test_word_boundary_score_full_words():
    # Use words with > 3 characters on both boundary edges
    # last word of left = "sentence" (8 chars), first of right = "complete" (8 chars)
    score = word_boundary_score("this is complete sentence", "complete another sentence here")
    # Both boundary words are >3 chars, so not broken
    assert score == 1.0


def test_word_boundary_score_empty():
    score = word_boundary_score("", "")
    assert score == 0.5


def test_word_boundary_score_short_word_both():
    # Short words on both sides (possibly broken)
    score = word_boundary_score("fo", "ox")
    assert score == 0.2


def test_word_boundary_score_short_word_one():
    score = word_boundary_score("fo", "complete sentence here")
    assert score == 0.5


def test_word_boundary_score_range():
    score = word_boundary_score("hello world", "test phrase")
    assert 0.0 <= score <= 1.0


# ─── TextCoherenceScorer ──────────────────────────────────────────────────────

def test_text_coherence_scorer_init():
    scorer = TextCoherenceScorer(n=2, alpha=0.01)
    assert scorer.model.n == 2
    assert not scorer._trained


def test_text_coherence_scorer_train():
    scorer = TextCoherenceScorer()
    result = scorer.train(CORPUS)
    assert result is scorer  # returns self
    assert scorer._trained


def test_text_coherence_scorer_train_empty():
    scorer = TextCoherenceScorer()
    scorer.train([])
    assert not scorer._trained


def test_text_coherence_scorer_seam_score_without_ocr():
    scorer = TextCoherenceScorer()
    scorer.train(CORPUS)
    img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    # Without tesseract → returns 0.5
    score = scorer.seam_score(img1, img2)
    assert 0.0 <= score <= 1.0


def test_text_coherence_scorer_batch_seam_scores():
    scorer = TextCoherenceScorer()
    scorer.train(CORPUS)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    pairs = [(img, img), (img, img)]
    scores = scorer.batch_seam_scores(pairs)
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_text_coherence_scorer_repr():
    scorer = TextCoherenceScorer(n=2)
    r = repr(scorer)
    assert "n=2" in r


def test_text_coherence_scorer_train_returns_self():
    scorer = TextCoherenceScorer()
    result = scorer.train(CORPUS)
    assert result is scorer
