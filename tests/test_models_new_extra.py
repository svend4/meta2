"""Extra tests for puzzle_reconstruction/models.py"""
import json
import numpy as np
import pytest
import tempfile
import pathlib

from puzzle_reconstruction.models import (
    ShapeClass,
    EdgeSide,
    FractalSignature,
    TangramSignature,
    EdgeSignature,
    Edge,
    Placement,
    Fragment,
    CompatEntry,
    Assembly,
    MatchingState,
    AssemblySession,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(**kwargs):
    defaults = dict(
        compat_matrix=np.eye(3, dtype=np.float32),
        entries=[],
        threshold=0.5,
        n_fragments=3,
        timestamp="2026-01-01T00:00:00",
        config_dict={},
        method="auto",
    )
    defaults.update(kwargs)
    return MatchingState(**defaults)


def _make_session(**kwargs):
    defaults = dict(
        method="simulated_annealing",
        iteration=10,
        best_score=0.5,
        score_history=[0.1 * i for i in range(10)],
        best_placement={"0": [0, 0, 0.0]},
        n_fragments=2,
    )
    defaults.update(kwargs)
    return AssemblySession(**defaults)


# ─── ShapeClass enum ─────────────────────────────────────────────────────────

def test_shape_class_all_values_present():
    expected = {"triangle", "rectangle", "trapezoid", "parallelogram",
                "pentagon", "hexagon", "polygon"}
    actual = {sc.value for sc in ShapeClass}
    assert expected == actual


def test_shape_class_is_string_enum():
    assert ShapeClass.TRIANGLE == "triangle"
    assert ShapeClass.HEXAGON == "hexagon"


def test_shape_class_from_string():
    sc = ShapeClass("rectangle")
    assert sc is ShapeClass.RECTANGLE


def test_shape_class_invalid_raises():
    with pytest.raises(ValueError):
        ShapeClass("octagon")


# ─── EdgeSide enum ────────────────────────────────────────────────────────────

def test_edge_side_all_values():
    expected = {"top", "bottom", "left", "right", "unknown"}
    actual = {es.value for es in EdgeSide}
    assert expected == actual


def test_edge_side_unknown_value():
    assert EdgeSide.UNKNOWN == "unknown"


# ─── FractalSignature ─────────────────────────────────────────────────────────

def test_fractal_signature_fields():
    sig = FractalSignature(
        fd_box=1.3,
        fd_divider=1.4,
        ifs_coeffs=np.array([0.1, 0.2]),
        css_image=[(1.0, np.array([0.5]))],
        chain_code="01234567",
        curve=np.zeros((10, 2)),
    )
    assert sig.fd_box == pytest.approx(1.3)
    assert sig.fd_divider == pytest.approx(1.4)
    assert len(sig.ifs_coeffs) == 2
    assert sig.chain_code == "01234567"


def test_fractal_signature_curve_shape():
    curve = np.random.default_rng(0).random((50, 2))
    sig = FractalSignature(
        fd_box=1.5, fd_divider=1.5,
        ifs_coeffs=np.zeros(4),
        css_image=[],
        chain_code="",
        curve=curve,
    )
    assert sig.curve.shape == (50, 2)


# ─── TangramSignature ─────────────────────────────────────────────────────────

def test_tangram_signature_fields():
    ts = TangramSignature(
        polygon=np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float),
        shape_class=ShapeClass.TRIANGLE,
        centroid=np.array([0.5, 0.33]),
        angle=0.0,
        scale=1.0,
        area=0.5,
    )
    assert ts.shape_class is ShapeClass.TRIANGLE
    assert ts.area == pytest.approx(0.5)


# ─── EdgeSignature ────────────────────────────────────────────────────────────

def test_edge_signature_fields():
    es = EdgeSignature(
        edge_id=42,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.2,
        css_vec=np.ones(16),
        ifs_coeffs=np.zeros(8),
        length=100.0,
    )
    assert es.edge_id == 42
    assert es.side is EdgeSide.TOP
    assert es.length == pytest.approx(100.0)


# ─── Edge ─────────────────────────────────────────────────────────────────────

def test_edge_defaults():
    e = Edge(edge_id=1, contour=np.zeros((10, 2)))
    assert e.text_hint == ""
    assert e.edge_id == 1


def test_edge_with_text_hint():
    e = Edge(edge_id=7, contour=np.zeros((5, 2)), text_hint="lorem")
    assert e.text_hint == "lorem"


def test_edge_contour_stored():
    c = np.random.default_rng(0).random((20, 2))
    e = Edge(edge_id=3, contour=c)
    np.testing.assert_array_equal(e.contour, c)


# ─── Placement ────────────────────────────────────────────────────────────────

def test_placement_default_rotation():
    p = Placement(fragment_id=1, position=(10.0, 20.0))
    assert p.rotation == pytest.approx(0.0)


def test_placement_with_rotation():
    p = Placement(fragment_id=2, position=(5.0, 5.0), rotation=90.0)
    assert p.rotation == pytest.approx(90.0)


def test_placement_position_stored():
    p = Placement(fragment_id=0, position=(1.5, 2.5))
    assert p.position == (1.5, 2.5)


# ─── Fragment ─────────────────────────────────────────────────────────────────

def test_fragment_defaults():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    f = Fragment(fragment_id=0, image=img)
    assert f.mask is None
    assert f.contour is None
    assert f.tangram is None
    assert f.fractal is None
    assert f.edges == []
    assert f.placed is False
    assert f.position is None
    assert f.rotation == pytest.approx(0.0)
    assert f.bounding_box is None


def test_fragment_image_stored():
    img = np.ones((16, 16, 3), dtype=np.uint8) * 128
    f = Fragment(fragment_id=1, image=img)
    np.testing.assert_array_equal(f.image, img)


def test_fragment_placed_flag():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    f = Fragment(fragment_id=2, image=img, placed=True)
    assert f.placed is True


def test_fragment_bounding_box():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    f = Fragment(fragment_id=3, image=img, bounding_box=(0, 0, 8, 8))
    assert f.bounding_box == (0, 0, 8, 8)


# ─── CompatEntry ─────────────────────────────────────────────────────────────

def test_compat_entry_defaults():
    e1 = Edge(edge_id=0, contour=np.zeros((5, 2)))
    e2 = Edge(edge_id=1, contour=np.zeros((5, 2)))
    c = CompatEntry(edge_i=e1, edge_j=e2, score=0.7)
    assert c.dtw_dist == pytest.approx(0.0)
    assert c.css_sim == pytest.approx(0.0)
    assert c.fd_diff == pytest.approx(0.0)
    assert c.text_score == pytest.approx(0.0)


def test_compat_entry_fields():
    c = CompatEntry(
        edge_i=1, edge_j=2, score=0.8,
        dtw_dist=0.3, css_sim=0.7, fd_diff=0.1, text_score=0.5,
    )
    assert c.score == pytest.approx(0.8)
    assert c.dtw_dist == pytest.approx(0.3)
    assert c.css_sim == pytest.approx(0.7)


# ─── Assembly ────────────────────────────────────────────────────────────────

def test_assembly_defaults():
    a = Assembly()
    assert a.placements == []
    assert a.fragments is None
    assert a.compat_matrix is None
    assert a.total_score == pytest.approx(0.0)
    assert a.ocr_score == pytest.approx(0.0)
    assert a.method == ""


def test_assembly_with_values():
    a = Assembly(total_score=0.9, method="greedy")
    assert a.total_score == pytest.approx(0.9)
    assert a.method == "greedy"


# ─── MatchingState edge cases ─────────────────────────────────────────────────

def test_matching_state_large_matrix():
    mat = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    state = _make_state(compat_matrix=mat, n_fragments=100)
    d = state.to_dict()
    restored = MatchingState.from_dict(d)
    assert restored.compat_matrix.shape == (100, 100)


def test_matching_state_zero_fragments():
    state = _make_state(n_fragments=0, compat_matrix=np.zeros((0, 0), dtype=np.float32))
    assert state.to_dict()["n_fragments"] == 0


def test_matching_state_complex_config_dict():
    cfg = {"nested": {"a": 1, "b": [1, 2, 3]}, "c": "hello"}
    state = _make_state(config_dict=cfg)
    restored = MatchingState.from_dict(state.to_dict())
    assert restored.config_dict == cfg


def test_matching_state_entries_many_items():
    entries = list(range(1000))
    state = _make_state(entries=entries)
    assert state.to_dict()["n_entries"] == 1000


def test_matching_state_method_empty_string():
    state = _make_state(method="")
    d = state.to_dict()
    restored = MatchingState.from_dict(d)
    assert restored.method == ""


def test_matching_state_save_load_large_matrix(tmp_path):
    mat = np.random.default_rng(7).random((50, 50)).astype(np.float32)
    state = _make_state(compat_matrix=mat, n_fragments=50)
    path = str(tmp_path / "state_large.json")
    state.save(path)
    restored = MatchingState.load(path)
    np.testing.assert_allclose(restored.compat_matrix, mat, atol=1e-5)


def test_matching_state_from_dict_entries_always_empty():
    state = _make_state(entries=["a", "b"])
    restored = MatchingState.from_dict(state.to_dict())
    assert restored.entries == []


def test_matching_state_json_serializable(tmp_path):
    state = _make_state()
    path = str(tmp_path / "state.json")
    state.save(path)
    raw = json.loads(pathlib.Path(path).read_text())
    assert isinstance(raw, dict)


# ─── AssemblySession edge cases ───────────────────────────────────────────────

def test_assembly_session_empty_history():
    s = _make_session(score_history=[], best_score=0.0)
    assert s.is_converged is False


def test_assembly_session_99_history():
    s = _make_session(score_history=[0.5] * 99, best_score=0.5)
    assert s.is_converged is False


def test_assembly_session_100_identical():
    s = _make_session(score_history=[0.5] * 100, best_score=0.5)
    assert s.is_converged is True


def test_assembly_session_101_scores_check_last_100():
    # Only last 100 matter; first one is bad but recent are plateau
    s = _make_session(score_history=[10.0] + [0.5] * 100, best_score=0.5)
    assert s.is_converged is True


def test_assembly_session_converged_then_improved():
    history = [0.5] * 100 + [0.9]
    s = _make_session(score_history=history, best_score=0.5)
    assert s.is_converged is False


def test_assembly_session_to_dict_score_history_type():
    s = _make_session()
    d = s.to_dict()
    assert isinstance(d["score_history"], list)


def test_assembly_session_to_dict_placement_type():
    s = _make_session()
    d = s.to_dict()
    assert isinstance(d["best_placement"], dict)


def test_assembly_session_empty_placement():
    s = _make_session(best_placement={})
    d = s.to_dict()
    restored = AssemblySession.from_dict(d)
    assert restored.best_placement == {}


def test_assembly_session_checkpoint_valid_json(tmp_path):
    s = _make_session()
    path = str(tmp_path / "session.json")
    s.checkpoint(path)
    raw = json.loads(pathlib.Path(path).read_text())
    assert "iteration" in raw
    assert "best_score" in raw


def test_assembly_session_resume_score_history(tmp_path):
    hist = [float(i) * 0.1 for i in range(20)]
    s = _make_session(score_history=hist)
    path = str(tmp_path / "s.json")
    s.checkpoint(path)
    restored = AssemblySession.resume(path)
    assert restored.score_history == hist


def test_assembly_session_zero_iteration():
    s = _make_session(iteration=0, score_history=[], best_score=0.0)
    assert s.iteration == 0
    assert s.is_converged is False


def test_assembly_session_very_long_score_history():
    hist = [float(i) / 1000.0 for i in range(1000)]
    s = _make_session(score_history=hist, best_score=0.999)
    # Last 100 values are 0.9–0.999; best_score=0.999 → converged
    assert s.is_converged is True
