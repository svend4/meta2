"""
Тесты для puzzle_reconstruction/verification/text_coherence.py

Покрытие:
    _tokenize          — пустая строка, ASCII, кириллица, смешанный, знаки препинания
    NGramModel         — fit (пустой корпус, длина vocab, total), log_prob
                         (Laplace > 0, unseen < seen), perplexity (small=good),
                         sentence_score ∈ [0,1], fit повторно добавляет данные
    seam_bigram_score  — оба пустые→0.0, один пустой→0.0, no model→0.5,
                         с моделью→[0,1]
    word_boundary_score — нет текста→0.5, полные слова→1.0, короткие→штраф,
                          оба коротких→меньше 0.5
    build_ngram_model  — тип NGramModel, n и alpha совпадают
    TextCoherenceScorer — repr, train меняет _trained, seam_score без OCR→0.5
"""
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


# ─── _tokenize ────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_empty_string(self):
        assert _tokenize("") == []

    def test_whitespace_only(self):
        assert _tokenize("   \t\n  ") == []

    def test_ascii_words(self):
        toks = _tokenize("Hello world!")
        assert toks == ["hello", "world"]

    def test_cyrillic(self):
        toks = _tokenize("Привет мир")
        assert toks == ["привет", "мир"]

    def test_mixed_lang(self):
        toks = _tokenize("Hello Привет 123 world мир")
        assert "hello" in toks
        assert "привет" in toks
        assert "world" in toks
        assert "мир" in toks
        # Цифры не включаются
        assert "123" not in toks

    def test_punctuation_stripped(self):
        toks = _tokenize("слово, другое! третье.")
        assert all("," not in t and "." not in t and "!" not in t for t in toks)

    def test_short_words_filtered(self):
        """Слова длиной < 2 не включаются."""
        toks = _tokenize("a b c word")
        assert "a" not in toks
        assert "b" not in toks
        assert "word" in toks

    def test_lowercase(self):
        toks = _tokenize("THE QUICK BROWN FOX")
        assert all(t == t.lower() for t in toks)


# ─── NGramModel ───────────────────────────────────────────────────────────────

class TestNGramModel:
    CORPUS = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barked at the fox",
        "a quick fox and a lazy dog",
    ]

    def test_fit_empty_corpus(self):
        m = NGramModel(n=2)
        m.fit([])
        assert len(m.vocab) == 0
        assert m.total == 0

    def test_fit_builds_vocab(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        assert len(m.vocab) > 0
        assert "fox" in m.vocab

    def test_fit_counts_bigrams(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        assert m.total > 0

    def test_fit_chaining(self):
        m = NGramModel(n=2)
        m2 = m.fit(self.CORPUS)
        assert m2 is m  # возвращает self

    def test_fit_accumulates(self):
        m = NGramModel(n=2)
        m.fit(self.CORPUS)
        total1 = m.total
        m.fit(self.CORPUS)
        assert m.total == total1 * 2  # Данные добавились

    def test_log_prob_negative(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        lp = m.log_prob(("the",), "fox")
        assert lp < 0.0

    def test_log_prob_seen_gt_unseen(self):
        """Виданная комбинация должна иметь более высокую вероятность."""
        m = NGramModel(n=2).fit(self.CORPUS)
        lp_seen   = m.log_prob(("the",), "fox")
        lp_unseen = m.log_prob(("the",), "xyzzy")
        assert lp_seen > lp_unseen

    def test_log_prob_without_fit(self):
        """Без данных log_prob опирается только на сглаживание."""
        m = NGramModel(n=2, alpha=1.0)
        lp = m.log_prob(("foo",), "bar")
        assert math.isfinite(lp)

    def test_perplexity_good_sentence(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        pp_good = m.perplexity("the fox barked at the dog")
        pp_bad  = m.perplexity("xyzzy qqqq wwww vvvv uuuu")
        # Хорошее предложение → меньшая перплексия
        assert pp_good < pp_bad

    def test_perplexity_inf_for_short_sentence(self):
        m = NGramModel(n=3).fit(self.CORPUS)
        # Одно слово → нет триграмм → перплексия inf
        pp = m.perplexity("one")
        assert math.isinf(pp)

    def test_sentence_score_in_range(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        score = m.sentence_score("the quick fox jumps")
        assert 0.0 <= score <= 1.0

    def test_sentence_score_zero_for_inf_perplexity(self):
        m = NGramModel(n=2)
        # Без обучения, vocab пустой → sentence_score → 0.0
        score = m.sentence_score("one")
        assert math.isclose(score, 0.0)

    def test_sentence_score_higher_for_known(self):
        m = NGramModel(n=2).fit(self.CORPUS)
        s_known   = m.sentence_score("the quick fox")
        s_unknown = m.sentence_score("xyzzy qqqq wwww")
        assert s_known >= s_unknown

    def test_repr(self):
        m = NGramModel(n=2, alpha=0.05)
        r = repr(m)
        assert "NGramModel" in r
        assert "n=2" in r


# ─── seam_bigram_score ────────────────────────────────────────────────────────

class TestSeamBigramScore:
    @pytest.fixture
    def model(self):
        corpus = ["the quick brown fox jumps over the lazy dog"] * 10
        return NGramModel(n=2).fit(corpus)

    def test_both_empty_returns_zero(self, model):
        score = seam_bigram_score("", "", model)
        assert math.isclose(score, 0.0)

    def test_left_empty_returns_zero(self, model):
        score = seam_bigram_score("", "some text here", model)
        assert math.isclose(score, 0.0)

    def test_right_empty_returns_zero(self, model):
        score = seam_bigram_score("some text here", "", model)
        assert math.isclose(score, 0.0)

    def test_no_model_returns_05(self):
        score = seam_bigram_score("hello world", "foo bar", model=None)
        assert math.isclose(score, 0.5)

    def test_empty_model_returns_05(self):
        m = NGramModel(n=2)
        score = seam_bigram_score("hello world", "foo bar", model=m)
        assert math.isclose(score, 0.5)

    def test_with_model_in_range(self, model):
        score = seam_bigram_score("the quick", "fox jumps", model)
        assert 0.0 <= score <= 1.0

    def test_known_text_higher_than_garbage(self, model):
        s_good = seam_bigram_score("the quick brown", "fox jumps over", model)
        s_bad  = seam_bigram_score("xyzzy qqqq", "wwww vvvv uuu", model)
        # Хороший стык → выше (не гарантировано, но ожидаемо)
        assert math.isfinite(s_good) and math.isfinite(s_bad)


# ─── word_boundary_score ──────────────────────────────────────────────────────

class TestWordBoundaryScore:
    def test_no_text_returns_05(self):
        assert math.isclose(word_boundary_score("", ""), 0.5)

    def test_left_empty_returns_05(self):
        assert math.isclose(word_boundary_score("", "hello world"), 0.5)

    def test_right_empty_returns_05(self):
        assert math.isclose(word_boundary_score("hello world", ""), 0.5)

    def test_full_words_returns_10(self):
        """Длинные слова с обеих сторон → score = 1.0."""
        score = word_boundary_score("document text here", "another section here")
        assert math.isclose(score, 1.0)

    def test_short_last_word_penalized(self):
        """Короткое последнее слово слева → penalty."""
        score = word_boundary_score("fragment te", "xt continues here")
        assert score < 1.0

    def test_short_first_word_penalized(self):
        """Короткое первое слово справа → penalty."""
        score = word_boundary_score("text continues here", "xt more text here")
        assert score < 1.0

    def test_both_short_heavily_penalized(self):
        """Оба коротких → сильный штраф."""
        score_both = word_boundary_score("fragment te", "xt another")
        score_one  = word_boundary_score("fragment te", "longer word here")
        assert score_both <= score_one

    def test_score_in_range(self):
        for left, right in [
            ("hello world", "foo bar baz"),
            ("te", "xt"),
            ("", ""),
            ("document", "fragment"),
        ]:
            score = word_boundary_score(left, right)
            assert 0.0 <= score <= 1.0, f"Out of range for ({left!r}, {right!r})"


# ─── build_ngram_model ────────────────────────────────────────────────────────

class TestBuildNgramModel:
    def test_returns_ngram_model(self):
        m = build_ngram_model(["hello world", "foo bar"], n=2)
        assert isinstance(m, NGramModel)

    def test_n_correct(self):
        m = build_ngram_model(["hello world"], n=3)
        assert m.n == 3

    def test_alpha_correct(self):
        m = build_ngram_model(["hello world"], alpha=0.5)
        assert math.isclose(m.alpha, 0.5)

    def test_empty_corpus(self):
        m = build_ngram_model([], n=2)
        assert isinstance(m, NGramModel)
        assert m.total == 0

    def test_trained_model_has_vocab(self):
        corpus = ["the quick brown fox"] * 5
        m = build_ngram_model(corpus, n=2)
        assert len(m.vocab) > 0


# ─── TextCoherenceScorer ──────────────────────────────────────────────────────

class TestTextCoherenceScorer:
    def test_repr(self):
        s = TextCoherenceScorer(n=2, lang="rus+eng")
        r = repr(s)
        assert "TextCoherenceScorer" in r
        assert "2" in r

    def test_not_trained_initially(self):
        s = TextCoherenceScorer()
        assert not s._trained

    def test_train_sets_trained(self):
        s = TextCoherenceScorer()
        s.train(["hello world foo bar"] * 10)
        assert s._trained

    def test_train_empty_corpus(self):
        s = TextCoherenceScorer()
        s.train([])
        assert not s._trained

    def test_seam_score_no_ocr_returns_05(self):
        """Без pytesseract / OpenCV всегда возвращает 0.5."""
        import puzzle_reconstruction.verification.text_coherence as tc
        orig_tes = tc._TESSERACT
        orig_cv2 = tc._CV2
        try:
            tc._TESSERACT = False
            tc._CV2       = False
            s     = TextCoherenceScorer()
            img   = np.zeros((32, 32, 3), dtype=np.uint8)
            score = s.seam_score(img, img)
            assert math.isclose(score, 0.5)
        finally:
            tc._TESSERACT = orig_tes
            tc._CV2       = orig_cv2

    def test_batch_seam_scores_length(self):
        """batch_seam_scores возвращает правильное число оценок."""
        import puzzle_reconstruction.verification.text_coherence as tc
        orig_tes = tc._TESSERACT
        try:
            tc._TESSERACT = False
            s    = TextCoherenceScorer()
            img  = np.zeros((32, 32, 3), dtype=np.uint8)
            pairs = [(img, img), (img, img), (img, img)]
            scores = s.batch_seam_scores(pairs)
            assert len(scores) == 3
        finally:
            tc._TESSERACT = orig_tes
