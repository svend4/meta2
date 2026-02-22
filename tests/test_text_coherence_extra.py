"""Extra tests for puzzle_reconstruction.verification.text_coherence."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.verification.text_coherence import (
    NGramModel,
    TextCoherenceScorer,
    _tokenize,
    build_ngram_model,
    seam_bigram_score,
    word_boundary_score,
)


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barked at the fox",
    "a quick fox and a lazy dog",
]


# ─── TestTokenizeExtra ────────────────────────────────────────────────────────

class TestTokenizeExtra:
    def test_empty(self):
        assert _tokenize("") == []

    def test_whitespace(self):
        assert _tokenize("  \t\n ") == []

    def test_ascii(self):
        assert _tokenize("Hello world") == ["hello", "world"]

    def test_cyrillic(self):
        assert _tokenize("Привет мир") == ["привет", "мир"]

    def test_mixed(self):
        toks = _tokenize("Hello Привет 123 world мир")
        assert "hello" in toks
        assert "мир" in toks
        assert "123" not in toks

    def test_punctuation_stripped(self):
        toks = _tokenize("word, another! third.")
        assert all(c not in t for t in toks for c in ",.!")

    def test_short_filtered(self):
        toks = _tokenize("a b c word")
        assert "a" not in toks
        assert "word" in toks

    def test_lowercase(self):
        toks = _tokenize("THE QUICK")
        assert all(t == t.lower() for t in toks)

    def test_numbers_excluded(self):
        toks = _tokenize("test 42 hello 99")
        assert "42" not in toks
        assert "99" not in toks


# ─── TestNGramModelExtra ──────────────────────────────────────────────────────

class TestNGramModelExtra:
    def test_fit_empty(self):
        m = NGramModel(n=2)
        m.fit([])
        assert len(m.vocab) == 0
        assert m.total == 0

    def test_fit_builds_vocab(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert "fox" in m.vocab

    def test_fit_total_positive(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert m.total > 0

    def test_fit_chaining(self):
        m = NGramModel(n=2)
        assert m.fit(CORPUS) is m

    def test_fit_accumulates(self):
        m = NGramModel(n=2)
        m.fit(CORPUS)
        t1 = m.total
        m.fit(CORPUS)
        assert m.total == t1 * 2

    def test_log_prob_negative(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert m.log_prob(("the",), "fox") < 0.0

    def test_log_prob_seen_gt_unseen(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert m.log_prob(("the",), "fox") > m.log_prob(("the",), "xyzzy")

    def test_log_prob_without_fit(self):
        m = NGramModel(n=2, alpha=1.0)
        assert math.isfinite(m.log_prob(("foo",), "bar"))

    def test_perplexity_finite(self):
        m = NGramModel(n=2).fit(CORPUS)
        pp = m.perplexity("the fox barked")
        assert math.isfinite(pp) and pp > 0

    def test_perplexity_short_inf(self):
        m = NGramModel(n=3).fit(CORPUS)
        assert math.isinf(m.perplexity("one"))

    def test_sentence_score_range(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert 0.0 <= m.sentence_score("the quick fox") <= 1.0

    def test_sentence_score_known_ge_unknown(self):
        m = NGramModel(n=2).fit(CORPUS)
        assert m.sentence_score("the quick fox") >= m.sentence_score("xyzzy qqqq www")

    def test_repr(self):
        r = repr(NGramModel(n=2, alpha=0.05))
        assert "NGramModel" in r
        assert "n=2" in r


# ─── TestSeamBigramScoreExtra ─────────────────────────────────────────────────

class TestSeamBigramScoreExtra:
    @pytest.fixture
    def model(self):
        return NGramModel(n=2).fit(CORPUS * 10)

    def test_both_empty_zero(self, model):
        assert math.isclose(seam_bigram_score("", "", model), 0.0)

    def test_left_empty_zero(self, model):
        assert math.isclose(seam_bigram_score("", "some text", model), 0.0)

    def test_right_empty_zero(self, model):
        assert math.isclose(seam_bigram_score("some text", "", model), 0.0)

    def test_no_model_05(self):
        assert math.isclose(seam_bigram_score("hello", "world", model=None), 0.5)

    def test_empty_model_05(self):
        m = NGramModel(n=2)
        assert math.isclose(seam_bigram_score("hello", "world", model=m), 0.5)

    def test_with_model_in_range(self, model):
        score = seam_bigram_score("the quick", "fox jumps", model)
        assert 0.0 <= score <= 1.0

    def test_finite(self, model):
        s = seam_bigram_score("the quick brown", "fox jumps over", model)
        assert math.isfinite(s)

    def test_garbage_finite(self, model):
        s = seam_bigram_score("xyzzy qqqq", "wwww vvvv", model)
        assert math.isfinite(s)


# ─── TestWordBoundaryScoreExtra ───────────────────────────────────────────────

class TestWordBoundaryScoreExtra:
    def test_empty_empty_05(self):
        assert math.isclose(word_boundary_score("", ""), 0.5)

    def test_left_empty_05(self):
        assert math.isclose(word_boundary_score("", "hello world"), 0.5)

    def test_right_empty_05(self):
        assert math.isclose(word_boundary_score("hello world", ""), 0.5)

    def test_full_words_one(self):
        assert math.isclose(word_boundary_score("document text here",
                                                "another section here"), 1.0)

    def test_short_last_penalized(self):
        assert word_boundary_score("fragment te", "xt continues here") < 1.0

    def test_short_first_penalized(self):
        assert word_boundary_score("text continues here", "xt more text") < 1.0

    def test_both_short_lower(self):
        s_both = word_boundary_score("fragment te", "xt another")
        s_one = word_boundary_score("fragment te", "longer word here")
        assert s_both <= s_one

    def test_range(self):
        for left, right in [("hello world", "foo bar"), ("te", "xt"), ("", "")]:
            assert 0.0 <= word_boundary_score(left, right) <= 1.0


# ─── TestBuildNgramModelExtra ─────────────────────────────────────────────────

class TestBuildNgramModelExtra:
    def test_returns_ngram_model(self):
        assert isinstance(build_ngram_model(["hello world"], n=2), NGramModel)

    def test_n_stored(self):
        assert build_ngram_model(["hello world"], n=3).n == 3

    def test_alpha_stored(self):
        assert math.isclose(build_ngram_model(["hello world"], alpha=0.5).alpha, 0.5)

    def test_empty_corpus(self):
        m = build_ngram_model([], n=2)
        assert m.total == 0

    def test_has_vocab(self):
        m = build_ngram_model(CORPUS * 5, n=2)
        assert len(m.vocab) > 0


# ─── TestTextCoherenceScorerExtra ─────────────────────────────────────────────

class TestTextCoherenceScorerExtra:
    def test_repr(self):
        r = repr(TextCoherenceScorer(n=2, lang="rus+eng"))
        assert "TextCoherenceScorer" in r

    def test_not_trained(self):
        assert not TextCoherenceScorer()._trained

    def test_train_sets_trained(self):
        s = TextCoherenceScorer()
        s.train(["hello world foo bar"] * 10)
        assert s._trained

    def test_train_empty_not_trained(self):
        s = TextCoherenceScorer()
        s.train([])
        assert not s._trained

    def test_seam_score_no_ocr_05(self):
        import puzzle_reconstruction.verification.text_coherence as tc
        orig_tes = tc._TESSERACT
        orig_cv2 = tc._CV2
        try:
            tc._TESSERACT = False
            tc._CV2 = False
            s = TextCoherenceScorer()
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            assert math.isclose(s.seam_score(img, img), 0.5)
        finally:
            tc._TESSERACT = orig_tes
            tc._CV2 = orig_cv2

    def test_batch_seam_scores_length(self):
        import puzzle_reconstruction.verification.text_coherence as tc
        orig_tes = tc._TESSERACT
        try:
            tc._TESSERACT = False
            s = TextCoherenceScorer()
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            scores = s.batch_seam_scores([(img, img), (img, img)])
            assert len(scores) == 2
        finally:
            tc._TESSERACT = orig_tes
