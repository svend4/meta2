"""Tests for puzzle_reconstruction.utils.annealing_score_utils."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.annealing_score_utils import (
    AnnealingScoreConfig,
    AnnealingScoreEntry,
    AnnealingSummary,
    make_annealing_entry,
    entries_from_log,
    summarise_annealing,
    filter_accepted,
    filter_rejected,
    filter_by_min_score,
    filter_by_temperature_range,
    top_k_entries,
    annealing_score_stats,
    best_entry,
    compare_summaries,
    batch_summarise,
)

np.random.seed(42)


def _make_entries(n=10):
    entries = []
    best = 0.0
    for i in range(n):
        score = float(np.random.uniform(0.0, 1.0))
        best = max(best, score)
        accepted = bool(np.random.randint(0, 2))
        entries.append(AnnealingScoreEntry(
            iteration=i,
            temperature=max(0.0, 1.0 - i * 0.1),
            current_score=score,
            best_score=best,
            accepted=accepted,
        ))
    return entries


# ── 1. AnnealingScoreConfig defaults ─────────────────────────────────────────
def test_config_defaults():
    cfg = AnnealingScoreConfig()
    assert cfg.min_score == 0.0
    assert cfg.convergence_window == 10
    assert cfg.improvement_threshold == 1e-4
    assert cfg.prefer_high_score is True


# ── 2. AnnealingScoreConfig validation ───────────────────────────────────────
def test_config_invalid_window():
    with pytest.raises(ValueError):
        AnnealingScoreConfig(convergence_window=0)


def test_config_invalid_threshold():
    with pytest.raises(ValueError):
        AnnealingScoreConfig(improvement_threshold=-0.1)


# ── 3. AnnealingScoreEntry basic ─────────────────────────────────────────────
def test_entry_basic():
    e = AnnealingScoreEntry(iteration=0, temperature=1.0,
                             current_score=0.5, best_score=0.5, accepted=True)
    assert e.iteration == 0
    assert e.temperature == 1.0
    assert e.current_score == 0.5
    assert e.best_score == 0.5
    assert e.accepted is True


def test_entry_invalid_iteration():
    with pytest.raises(ValueError):
        AnnealingScoreEntry(iteration=-1, temperature=1.0,
                             current_score=0.5, best_score=0.5, accepted=True)


def test_entry_invalid_temperature():
    with pytest.raises(ValueError):
        AnnealingScoreEntry(iteration=0, temperature=-0.1,
                             current_score=0.5, best_score=0.5, accepted=True)


# ── 4. make_annealing_entry ──────────────────────────────────────────────────
def test_make_annealing_entry():
    e = make_annealing_entry(5, 0.5, 0.8, 0.9, True, meta={"foo": 1})
    assert e.iteration == 5
    assert e.temperature == 0.5
    assert e.current_score == 0.8
    assert e.best_score == 0.9
    assert e.accepted is True
    assert e.meta == {"foo": 1}


def test_make_annealing_entry_no_meta():
    e = make_annealing_entry(0, 1.0, 0.3, 0.3, False)
    assert e.meta == {}


# ── 5. entries_from_log ──────────────────────────────────────────────────────
def test_entries_from_log_basic():
    log = [
        {"iteration": 0, "temperature": 1.0, "current_score": 0.5,
         "best_score": 0.5, "accepted": True, "extra": "x"},
        {"iteration": 1, "temperature": 0.9, "current_score": 0.6,
         "best_score": 0.6, "accepted": False},
    ]
    entries = entries_from_log(log)
    assert len(entries) == 2
    assert entries[0].iteration == 0
    assert entries[0].meta == {"extra": "x"}
    assert entries[1].accepted is False


def test_entries_from_log_missing_keys():
    log = [{}]
    entries = entries_from_log(log)
    assert len(entries) == 1
    assert entries[0].iteration == 0
    assert entries[0].temperature == 0.0
    assert entries[0].accepted is False


# ── 6. summarise_annealing empty ─────────────────────────────────────────────
def test_summarise_annealing_empty():
    s = summarise_annealing([])
    assert s.n_iterations == 0
    assert s.acceptance_rate == 0.0
    assert s.converged is False


# ── 7. summarise_annealing non-empty ─────────────────────────────────────────
def test_summarise_annealing_nonempty():
    entries = _make_entries(20)
    s = summarise_annealing(entries)
    assert s.n_iterations == 20
    assert 0.0 <= s.acceptance_rate <= 1.0
    assert s.best_score >= s.final_score or s.best_score >= 0.0
    assert isinstance(s.converged, bool)


# ── 8. filter_accepted / filter_rejected ────────────────────────────────────
def test_filter_accepted():
    entries = _make_entries(20)
    acc = filter_accepted(entries)
    assert all(e.accepted for e in acc)


def test_filter_rejected():
    entries = _make_entries(20)
    rej = filter_rejected(entries)
    assert all(not e.accepted for e in rej)
    assert len(filter_accepted(entries)) + len(rej) == 20


# ── 9. filter_by_min_score ───────────────────────────────────────────────────
def test_filter_by_min_score():
    entries = _make_entries(20)
    high = filter_by_min_score(entries, min_score=0.5)
    assert all(e.current_score >= 0.5 for e in high)


# ── 10. filter_by_temperature_range ─────────────────────────────────────────
def test_filter_by_temperature_range():
    entries = _make_entries(10)
    filtered = filter_by_temperature_range(entries, t_min=0.3, t_max=0.7)
    assert all(0.3 <= e.temperature <= 0.7 for e in filtered)


# ── 11. top_k_entries ────────────────────────────────────────────────────────
def test_top_k_entries():
    entries = _make_entries(20)
    top5 = top_k_entries(entries, 5)
    assert len(top5) == 5
    scores = [e.current_score for e in top5]
    assert scores == sorted(scores, reverse=True)


def test_top_k_entries_zero():
    entries = _make_entries(5)
    assert top_k_entries(entries, 0) == []


# ── 12. annealing_score_stats ────────────────────────────────────────────────
def test_annealing_score_stats_empty():
    stats = annealing_score_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_annealing_score_stats_nonempty():
    entries = _make_entries(10)
    stats = annealing_score_stats(entries)
    assert stats["count"] == 10
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std"] >= 0.0
    assert 0.0 <= stats["acceptance_rate"] <= 1.0


# ── 13. best_entry ───────────────────────────────────────────────────────────
def test_best_entry_empty():
    assert best_entry([]) is None


def test_best_entry_nonempty():
    entries = _make_entries(10)
    best = best_entry(entries)
    assert best is not None
    max_score = max(e.current_score for e in entries)
    assert best.current_score == max_score


# ── 14. compare_summaries ────────────────────────────────────────────────────
def test_compare_summaries():
    ea = _make_entries(20)
    eb = _make_entries(15)
    sa = summarise_annealing(ea)
    sb = summarise_annealing(eb)
    delta = compare_summaries(sa, sb)
    assert "best_score_delta" in delta
    assert "final_score_delta" in delta
    assert "acceptance_rate_delta" in delta
    assert delta["n_iter_delta"] == 5


# ── 15. batch_summarise ──────────────────────────────────────────────────────
def test_batch_summarise():
    logs = [
        [{"iteration": i, "temperature": 1.0 - i * 0.1,
          "current_score": float(i) / 10, "best_score": float(i) / 10,
          "accepted": True} for i in range(10)],
        [{"iteration": i, "temperature": 0.5,
          "current_score": 0.5, "best_score": 0.5,
          "accepted": False} for i in range(5)],
    ]
    summaries = batch_summarise(logs)
    assert len(summaries) == 2
    assert summaries[0].n_iterations == 10
    assert summaries[1].n_iterations == 5


# ── 16. AnnealingSummary repr ────────────────────────────────────────────────
def test_summary_repr():
    entries = _make_entries(10)
    s = summarise_annealing(entries)
    r = repr(s)
    assert "AnnealingSummary" in r
    assert "n_iter" in r


# ── 17. convergence detection ─────────────────────────────────────────────────
def test_convergence_flat_tail():
    entries = [
        AnnealingScoreEntry(iteration=i, temperature=0.1,
                             current_score=0.9, best_score=0.9, accepted=True)
        for i in range(20)
    ]
    cfg = AnnealingScoreConfig(convergence_window=5, improvement_threshold=1e-4)
    s = summarise_annealing(entries, cfg)
    assert s.converged is True
