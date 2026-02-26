"""Tests for puzzle_reconstruction.utils.config_utils"""
import json
import os
import tempfile
import pytest
from pathlib import Path
from puzzle_reconstruction.utils.config_utils import (
    validate_section,
    validate_range,
    merge_dicts,
    flatten_dict,
    unflatten_dict,
    overrides_from_env,
    ConfigProfile,
    PROFILES,
    apply_profile,
    list_profiles,
    load_json_config,
    save_json_config,
    diff_configs,
)


# ─── validate_section ────────────────────────────────────────────────────────

def test_validate_section_dict_ok():
    schema = {"lr": float, "epochs": int}
    errors = validate_section({"lr": 0.01, "epochs": 10}, schema)
    assert errors == []


def test_validate_section_missing_field():
    schema = {"lr": float}
    errors = validate_section({}, schema)
    assert len(errors) == 1
    assert "lr" in errors[0]


def test_validate_section_wrong_type():
    schema = {"lr": float}
    errors = validate_section({"lr": "bad"}, schema)
    assert len(errors) == 1


def test_validate_section_unsupported_type():
    errors = validate_section(42, {"x": int})
    assert len(errors) == 1
    assert "Unsupported" in errors[0]


def test_validate_section_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        lr: float = 0.01
        epochs: int = 10

    schema = {"lr": float, "epochs": int}
    errors = validate_section(Cfg(), schema)
    assert errors == []


# ─── validate_range ──────────────────────────────────────────────────────────

def test_validate_range_valid():
    result = validate_range(0.5, 0.0, 1.0)
    assert result is None


def test_validate_range_at_boundary():
    assert validate_range(0.0, 0.0, 1.0) is None
    assert validate_range(1.0, 0.0, 1.0) is None


def test_validate_range_below():
    result = validate_range(-0.1, 0.0, 1.0, name="score")
    assert result is not None
    assert "score" in result


def test_validate_range_above():
    result = validate_range(1.5, 0.0, 1.0)
    assert result is not None


# ─── merge_dicts ─────────────────────────────────────────────────────────────

def test_merge_dicts_simple():
    result = merge_dicts({"a": 1, "b": 2}, {"b": 99, "c": 3})
    assert result == {"a": 1, "b": 99, "c": 3}


def test_merge_dicts_deep_merge():
    base = {"section": {"a": 1, "b": 2}}
    override = {"section": {"b": 99}}
    result = merge_dicts(base, override)
    assert result["section"]["a"] == 1
    assert result["section"]["b"] == 99


def test_merge_dicts_does_not_mutate_base():
    base = {"a": 1}
    merge_dicts(base, {"a": 2})
    assert base["a"] == 1


# ─── flatten_dict ────────────────────────────────────────────────────────────

def test_flatten_dict_basic():
    d = {"a": {"b": 1, "c": 2}}
    flat = flatten_dict(d)
    assert flat == {"a.b": 1, "a.c": 2}


def test_flatten_dict_already_flat():
    d = {"x": 1, "y": 2}
    flat = flatten_dict(d)
    assert flat == {"x": 1, "y": 2}


def test_flatten_dict_nested():
    d = {"a": {"b": {"c": 42}}}
    flat = flatten_dict(d)
    assert flat == {"a.b.c": 42}


def test_flatten_dict_custom_sep():
    d = {"a": {"b": 1}}
    flat = flatten_dict(d, sep="/")
    assert flat == {"a/b": 1}


# ─── unflatten_dict ──────────────────────────────────────────────────────────

def test_unflatten_dict_basic():
    flat = {"a.b": 1, "a.c": 2}
    result = unflatten_dict(flat)
    assert result == {"a": {"b": 1, "c": 2}}


def test_unflatten_roundtrip():
    original = {"outer": {"inner": {"value": 42}}}
    flat = flatten_dict(original)
    recovered = unflatten_dict(flat)
    assert recovered == original


def test_unflatten_dict_flat_keys():
    flat = {"x": 10, "y": 20}
    result = unflatten_dict(flat)
    assert result == {"x": 10, "y": 20}


# ─── overrides_from_env ──────────────────────────────────────────────────────

def test_overrides_from_env_basic():
    os.environ["PUZZLE_TESTKEY"] = "42"
    result = overrides_from_env(prefix="PUZZLE_")
    assert "testkey" in result
    assert result["testkey"] == 42
    del os.environ["PUZZLE_TESTKEY"]


def test_overrides_from_env_bool_true():
    os.environ["PUZZLE_FLAG"] = "true"
    result = overrides_from_env(prefix="PUZZLE_")
    assert result.get("flag") is True
    del os.environ["PUZZLE_FLAG"]


def test_overrides_from_env_no_matching():
    result = overrides_from_env(prefix="NONEXISTENT_XYZ123_")
    assert result == {}


# ─── ConfigProfile ───────────────────────────────────────────────────────────

def test_config_profile_apply_to():
    profile = ConfigProfile(
        name="test",
        description="test profile",
        overrides={"lr": 0.001},
    )
    base = {"lr": 0.01, "epochs": 10}
    result = profile.apply_to(base)
    assert result["lr"] == pytest.approx(0.001)
    assert result["epochs"] == 10


# ─── PROFILES / apply_profile ────────────────────────────────────────────────

def test_profiles_exist():
    assert "fast" in PROFILES
    assert "accurate" in PROFILES
    assert "debug" in PROFILES


def test_apply_profile_fast():
    cfg = {"fractal": {"ifs_transforms": 8}, "assembly": {"method": "beam"}}
    result = apply_profile(cfg, "fast")
    assert result["assembly"]["method"] == "greedy"


def test_apply_profile_unknown():
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile({}, "nonexistent")


def test_list_profiles():
    profiles = list_profiles()
    assert len(profiles) >= 3
    names = [p[0] for p in profiles]
    assert "fast" in names
    assert "accurate" in names


# ─── load_json_config / save_json_config ──────────────────────────────────────

def test_save_and_load_json_config():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        data = {"lr": 0.01, "epochs": 10}
        save_json_config(data, path)
        loaded = load_json_config(path)
        assert loaded == data


def test_load_json_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_json_config(Path("/nonexistent/path/config.json"))


def test_save_json_config_creates_parents():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subdir" / "cfg.json"
        save_json_config({"x": 1}, path)
        assert path.exists()


# ─── diff_configs ────────────────────────────────────────────────────────────

def test_diff_configs_detects_changes():
    a = {"lr": 0.01, "epochs": 10}
    b = {"lr": 0.001, "epochs": 10}
    diff = diff_configs(a, b)
    assert "lr" in diff
    assert diff["lr"] == (0.01, 0.001)
    assert "epochs" not in diff


def test_diff_configs_nested():
    a = {"section": {"key": 1}}
    b = {"section": {"key": 2}}
    diff = diff_configs(a, b)
    assert "section.key" in diff


def test_diff_configs_empty():
    diff = diff_configs({}, {})
    assert diff == {}
