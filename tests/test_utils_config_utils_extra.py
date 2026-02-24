"""Extra tests for puzzle_reconstruction/utils/config_utils.py."""
from __future__ import annotations

import json
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


# ─── validate_section ─────────────────────────────────────────────────────────

class TestValidateSectionExtra:
    def test_valid_dict_no_errors(self):
        errors = validate_section({"x": 5}, {"x": int})
        assert errors == []

    def test_missing_field_gives_error(self):
        errors = validate_section({}, {"x": int})
        assert len(errors) > 0

    def test_wrong_type_gives_error(self):
        errors = validate_section({"x": "five"}, {"x": int})
        assert len(errors) > 0

    def test_none_value_passes(self):
        errors = validate_section({"x": None}, {"x": int})
        assert errors == []

    def test_unsupported_type_returns_error(self):
        errors = validate_section(42, {"x": int})
        assert len(errors) > 0


# ─── validate_range ───────────────────────────────────────────────────────────

class TestValidateRangeExtra:
    def test_in_range_returns_none(self):
        assert validate_range(0.5, 0.0, 1.0) is None

    def test_at_lo_returns_none(self):
        assert validate_range(0.0, 0.0, 1.0) is None

    def test_at_hi_returns_none(self):
        assert validate_range(1.0, 0.0, 1.0) is None

    def test_below_lo_returns_string(self):
        result = validate_range(-0.1, 0.0, 1.0)
        assert isinstance(result, str)

    def test_above_hi_returns_string(self):
        result = validate_range(1.5, 0.0, 1.0)
        assert isinstance(result, str)

    def test_name_in_error_message(self):
        result = validate_range(-1, 0, 10, name="my_param")
        assert "my_param" in result


# ─── merge_dicts ──────────────────────────────────────────────────────────────

class TestMergeDictsExtra:
    def test_returns_dict(self):
        assert isinstance(merge_dicts({"a": 1}, {}), dict)

    def test_base_not_mutated(self):
        base = {"a": 1}
        merge_dicts(base, {"a": 2})
        assert base["a"] == 1

    def test_override_wins(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_deep_merge(self):
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 99}}
        result = merge_dicts(base, override)
        assert result["nested"]["x"] == 1
        assert result["nested"]["y"] == 99

    def test_new_key_added(self):
        result = merge_dicts({"a": 1}, {"b": 2})
        assert "a" in result and "b" in result


# ─── flatten_dict ─────────────────────────────────────────────────────────────

class TestFlattenDictExtra:
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == d

    def test_nested_flattened(self):
        d = {"a": {"b": 1}}
        result = flatten_dict(d)
        assert "a.b" in result
        assert result["a.b"] == 1

    def test_custom_sep(self):
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep="/")
        assert "a/b" in result

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_deeply_nested(self):
        d = {"a": {"b": {"c": 42}}}
        result = flatten_dict(d)
        assert result["a.b.c"] == 42


# ─── unflatten_dict ───────────────────────────────────────────────────────────

class TestUnflattenDictExtra:
    def test_flat_unchanged(self):
        d = {"a": 1, "b": 2}
        assert unflatten_dict(d) == d

    def test_nested_reconstructed(self):
        flat = {"a.b": 1}
        result = unflatten_dict(flat)
        assert result["a"]["b"] == 1

    def test_roundtrip(self):
        original = {"a": {"b": 1, "c": 2}}
        result = unflatten_dict(flatten_dict(original))
        assert result == original

    def test_empty_dict(self):
        assert unflatten_dict({}) == {}


# ─── overrides_from_env ───────────────────────────────────────────────────────

class TestOverridesFromEnvExtra:
    def test_returns_dict(self, monkeypatch):
        monkeypatch.delenv("PUZZLE_TEST_KEY", raising=False)
        result = overrides_from_env(prefix="PUZZLE_TEST_")
        assert isinstance(result, dict)

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("PUZZLE_TEST_MY_VAR", "42")
        result = overrides_from_env(prefix="PUZZLE_TEST_")
        assert result.get("my_var") == 42

    def test_bool_cast_true(self, monkeypatch):
        monkeypatch.setenv("PUZZLE_TEST_FLAG", "true")
        result = overrides_from_env(prefix="PUZZLE_TEST_")
        assert result.get("flag") is True

    def test_section_separator(self, monkeypatch):
        monkeypatch.setenv("PUZZLE_TEST_A__B", "5")
        result = overrides_from_env(prefix="PUZZLE_TEST_")
        assert result.get("a", {}).get("b") == 5


# ─── ConfigProfile ────────────────────────────────────────────────────────────

class TestConfigProfileExtra:
    def test_stores_name(self):
        p = ConfigProfile(name="fast", description="desc", overrides={})
        assert p.name == "fast"

    def test_apply_to_merges(self):
        p = ConfigProfile(name="p", description="d", overrides={"x": 99})
        result = p.apply_to({"x": 1, "y": 2})
        assert result["x"] == 99
        assert result["y"] == 2


# ─── PROFILES and apply_profile ───────────────────────────────────────────────

class TestApplyProfileExtra:
    def test_profiles_exist(self):
        assert len(PROFILES) >= 3

    def test_fast_profile(self):
        result = apply_profile({}, "fast")
        assert isinstance(result, dict)

    def test_accurate_profile(self):
        result = apply_profile({}, "accurate")
        assert isinstance(result, dict)

    def test_debug_profile(self):
        result = apply_profile({}, "debug")
        assert isinstance(result, dict)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            apply_profile({}, "nonexistent")


# ─── list_profiles ────────────────────────────────────────────────────────────

class TestListProfilesExtra:
    def test_returns_list(self):
        assert isinstance(list_profiles(), list)

    def test_each_item_is_tuple(self):
        for item in list_profiles():
            assert isinstance(item, tuple) and len(item) == 2

    def test_names_are_strings(self):
        for name, desc in list_profiles():
            assert isinstance(name, str) and isinstance(desc, str)


# ─── load_json_config / save_json_config ──────────────────────────────────────

class TestJsonConfigExtra:
    def test_save_and_load(self, tmp_path):
        p = tmp_path / "cfg.json"
        data = {"alpha": 1, "beta": "test"}
        save_json_config(data, p)
        loaded = load_json_config(p)
        assert loaded == data

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json_config(tmp_path / "missing.json")

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "cfg.json"
        save_json_config({"x": 1}, p)
        assert p.exists()


# ─── diff_configs ─────────────────────────────────────────────────────────────

class TestDiffConfigsExtra:
    def test_identical_empty(self):
        assert diff_configs({"a": 1}, {"a": 1}) == {}

    def test_different_values(self):
        d = diff_configs({"a": 1}, {"a": 2})
        assert "a" in d

    def test_missing_key_in_b(self):
        d = diff_configs({"a": 1}, {})
        assert "a" in d

    def test_nested_diff(self):
        a = {"x": {"y": 1}}
        b = {"x": {"y": 2}}
        d = diff_configs(a, b)
        assert "x.y" in d
