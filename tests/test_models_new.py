"""
Tests for the new MatchingState and AssemblySession dataclasses
added to puzzle_reconstruction.models.
"""
import numpy as np
import pytest

from puzzle_reconstruction.models import MatchingState, AssemblySession


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_matching_state(**overrides):
    """Return a MatchingState with sensible defaults."""
    defaults = dict(
        compat_matrix=np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
        entries=[object(), object()],
        threshold=0.5,
        n_fragments=4,
        timestamp="2026-01-01T00:00:00",
        config_dict={"alpha": 0.3},
        method="otsu",
    )
    defaults.update(overrides)
    return MatchingState(**defaults)


def make_assembly_session(**overrides):
    """Return an AssemblySession with sensible defaults."""
    defaults = dict(
        method="simulated_annealing",
        iteration=50,
        best_score=0.87,
        score_history=[float(i) * 0.01 for i in range(50)],
        best_placement={"1": [10, 20, 90.0], "2": [30, 40, 0.0]},
        n_fragments=5,
        config_dict={"temp": 1000},
        random_seed=7,
    )
    defaults.update(overrides)
    return AssemblySession(**defaults)


# ===========================================================================
# MatchingState tests
# ===========================================================================

class TestMatchingStateConstruction:
    def test_construction_with_all_fields(self):
        state = make_matching_state()
        assert state.threshold == 0.5
        assert state.n_fragments == 4
        assert state.method == "otsu"

    def test_method_defaults_to_auto(self):
        state = MatchingState(
            compat_matrix=np.zeros((2, 2), dtype=np.float32),
            entries=[],
            threshold=0.4,
            n_fragments=2,
            timestamp="2026-01-01T00:00:00",
            config_dict={},
        )
        assert state.method == "auto"

    def test_timestamp_is_string(self):
        state = make_matching_state()
        assert isinstance(state.timestamp, str)

    def test_compat_matrix_stored(self):
        mat = np.eye(3, dtype=np.float32)
        state = make_matching_state(compat_matrix=mat)
        assert state.compat_matrix.shape == (3, 3)


class TestMatchingStateToDict:
    def test_to_dict_returns_dict(self):
        result = make_matching_state().to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_expected_keys(self):
        keys = {"compat_matrix", "threshold", "n_fragments", "timestamp",
                "config_dict", "method", "n_entries"}
        result = make_matching_state().to_dict()
        assert keys == set(result.keys())

    def test_to_dict_threshold_value(self):
        result = make_matching_state(threshold=0.75).to_dict()
        assert result["threshold"] == pytest.approx(0.75)

    def test_to_dict_n_entries_reflects_entries_length(self):
        state = make_matching_state(entries=["a", "b", "c"])
        assert state.to_dict()["n_entries"] == 3

    def test_to_dict_compat_matrix_is_list(self):
        result = make_matching_state().to_dict()
        assert isinstance(result["compat_matrix"], list)


class TestMatchingStateFromDict:
    def test_from_dict_roundtrip_preserves_threshold(self):
        state = make_matching_state(threshold=0.33)
        restored = MatchingState.from_dict(state.to_dict())
        assert restored.threshold == pytest.approx(0.33)

    def test_from_dict_reconstructs_compat_matrix_correctly(self):
        mat = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        state = make_matching_state(compat_matrix=mat)
        restored = MatchingState.from_dict(state.to_dict())
        np.testing.assert_allclose(restored.compat_matrix, mat)

    def test_from_dict_compat_matrix_is_float32(self):
        state = make_matching_state()
        restored = MatchingState.from_dict(state.to_dict())
        assert restored.compat_matrix.dtype == np.float32

    def test_from_dict_entries_is_empty_list(self):
        state = make_matching_state(entries=["x", "y", "z"])
        restored = MatchingState.from_dict(state.to_dict())
        assert restored.entries == []

    def test_from_dict_with_empty_config_dict(self):
        d = make_matching_state().to_dict()
        d["config_dict"] = {}
        restored = MatchingState.from_dict(d)
        assert restored.config_dict == {}

    def test_from_dict_missing_config_dict_defaults_to_empty(self):
        d = make_matching_state().to_dict()
        del d["config_dict"]
        restored = MatchingState.from_dict(d)
        assert restored.config_dict == {}

    def test_from_dict_missing_method_defaults_to_auto(self):
        d = make_matching_state().to_dict()
        del d["method"]
        restored = MatchingState.from_dict(d)
        assert restored.method == "auto"

    def test_from_dict_preserves_n_fragments(self):
        state = make_matching_state(n_fragments=10)
        restored = MatchingState.from_dict(state.to_dict())
        assert restored.n_fragments == 10

    def test_from_dict_preserves_timestamp(self):
        ts = "2025-06-15T12:30:00"
        state = make_matching_state(timestamp=ts)
        restored = MatchingState.from_dict(state.to_dict())
        assert restored.timestamp == ts


class TestMatchingStateSaveLoad:
    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        make_matching_state().save(path)
        import pathlib
        assert pathlib.Path(path).exists()

    def test_load_after_save_roundtrip_preserves_threshold(self, tmp_path):
        path = str(tmp_path / "state.json")
        make_matching_state(threshold=0.61).save(path)
        restored = MatchingState.load(path)
        assert restored.threshold == pytest.approx(0.61)

    def test_load_after_save_preserves_n_fragments(self, tmp_path):
        path = str(tmp_path / "state.json")
        make_matching_state(n_fragments=7).save(path)
        restored = MatchingState.load(path)
        assert restored.n_fragments == 7

    def test_load_after_save_preserves_method(self, tmp_path):
        path = str(tmp_path / "state.json")
        make_matching_state(method="kmeans").save(path)
        restored = MatchingState.load(path)
        assert restored.method == "kmeans"

    def test_load_after_save_compat_matrix_float32(self, tmp_path):
        path = str(tmp_path / "state.json")
        make_matching_state().save(path)
        restored = MatchingState.load(path)
        assert restored.compat_matrix.dtype == np.float32

    def test_load_after_save_compat_matrix_values(self, tmp_path):
        path = str(tmp_path / "state.json")
        mat = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        make_matching_state(compat_matrix=mat).save(path)
        restored = MatchingState.load(path)
        np.testing.assert_allclose(restored.compat_matrix, mat, atol=1e-6)


# ===========================================================================
# AssemblySession tests
# ===========================================================================

class TestAssemblySessionConstruction:
    def test_construction_with_required_fields(self):
        session = make_assembly_session()
        assert session.method == "simulated_annealing"
        assert session.iteration == 50
        assert session.best_score == pytest.approx(0.87)

    def test_config_dict_defaults_to_empty_dict(self):
        session = AssemblySession(
            method="genetic",
            iteration=0,
            best_score=0.0,
            score_history=[],
            best_placement={},
            n_fragments=3,
        )
        assert session.config_dict == {}

    def test_random_seed_defaults_to_42(self):
        session = AssemblySession(
            method="genetic",
            iteration=0,
            best_score=0.0,
            score_history=[],
            best_placement={},
            n_fragments=3,
        )
        assert session.random_seed == 42

    def test_score_history_is_list(self):
        session = make_assembly_session()
        assert isinstance(session.score_history, list)

    def test_best_placement_is_dict(self):
        session = make_assembly_session()
        assert isinstance(session.best_placement, dict)


class TestAssemblySessionToDict:
    def test_to_dict_returns_dict(self):
        result = make_assembly_session().to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_keys(self):
        expected = {"method", "iteration", "best_score", "score_history",
                    "best_placement", "n_fragments", "config_dict", "random_seed"}
        assert expected == set(make_assembly_session().to_dict().keys())

    def test_to_dict_method_value(self):
        result = make_assembly_session(method="beam").to_dict()
        assert result["method"] == "beam"

    def test_to_dict_random_seed_preserved(self):
        result = make_assembly_session(random_seed=99).to_dict()
        assert result["random_seed"] == 99


class TestAssemblySessionFromDict:
    def test_from_dict_roundtrip_preserves_method(self):
        session = make_assembly_session(method="genetic")
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.method == "genetic"

    def test_from_dict_roundtrip_preserves_iteration(self):
        session = make_assembly_session(iteration=200)
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.iteration == 200

    def test_from_dict_roundtrip_preserves_best_score(self):
        session = make_assembly_session(best_score=0.999)
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.best_score == pytest.approx(0.999)

    def test_from_dict_roundtrip_preserves_score_history(self):
        history = [0.1, 0.3, 0.5, 0.7]
        session = make_assembly_session(score_history=history)
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.score_history == history

    def test_from_dict_roundtrip_preserves_n_fragments(self):
        session = make_assembly_session(n_fragments=12)
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.n_fragments == 12

    def test_from_dict_roundtrip_preserves_random_seed(self):
        session = make_assembly_session(random_seed=123)
        restored = AssemblySession.from_dict(session.to_dict())
        assert restored.random_seed == 123


class TestAssemblySessionCheckpointResume:
    def test_checkpoint_creates_file(self, tmp_path):
        path = str(tmp_path / "session.json")
        make_assembly_session().checkpoint(path)
        import pathlib
        assert pathlib.Path(path).exists()

    def test_resume_after_checkpoint_preserves_method(self, tmp_path):
        path = str(tmp_path / "session.json")
        make_assembly_session(method="mcts").checkpoint(path)
        restored = AssemblySession.resume(path)
        assert restored.method == "mcts"

    def test_resume_after_checkpoint_preserves_best_score(self, tmp_path):
        path = str(tmp_path / "session.json")
        make_assembly_session(best_score=0.123).checkpoint(path)
        restored = AssemblySession.resume(path)
        assert restored.best_score == pytest.approx(0.123)

    def test_resume_after_checkpoint_with_best_placement_dict(self, tmp_path):
        path = str(tmp_path / "session.json")
        placement = {"0": [5, 10, 45.0], "1": [15, 25, 180.0]}
        make_assembly_session(best_placement=placement).checkpoint(path)
        restored = AssemblySession.resume(path)
        assert restored.best_placement == placement


class TestAssemblySessionIsConverged:
    def test_is_converged_false_when_history_less_than_100(self):
        session = make_assembly_session(
            score_history=[0.5] * 50,
            best_score=0.5,
        )
        assert session.is_converged is False

    def test_is_converged_false_when_exactly_99_items(self):
        session = make_assembly_session(
            score_history=[0.5] * 99,
            best_score=0.5,
        )
        assert session.is_converged is False

    def test_is_converged_false_when_still_improving(self):
        # Last 100 values contain one that exceeds best_score + 1e-6
        history = [0.5] * 200
        history[-1] = 0.99          # most recent greatly exceeds best_score
        session = make_assembly_session(
            score_history=history,
            best_score=0.5,
        )
        assert session.is_converged is False

    def test_is_converged_true_when_plateau_for_100_iterations(self):
        # All of the last 100 scores are at or below best_score + 1e-6
        plateau_value = 0.5
        history = [0.1] * 50 + [plateau_value] * 100
        session = make_assembly_session(
            score_history=history,
            best_score=plateau_value,
        )
        assert session.is_converged is True

    def test_is_converged_true_with_exactly_100_items_all_plateau(self):
        session = make_assembly_session(
            score_history=[0.7] * 100,
            best_score=0.7,
        )
        assert session.is_converged is True
