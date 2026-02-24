"""Extra tests for puzzle_reconstruction/utils/config_manager.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.config_manager import (
    ConfigField,
    ConfigSpec,
    ConfigSnapshot,
    validate_field_type,
    validate_config,
    load_config,
    merge_configs,
    diff_configs,
    make_config_snapshot,
    batch_validate,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _field(name="x", default=0, required=False, type_name="int") -> ConfigField:
    return ConfigField(name=name, default=default, required=required, type_name=type_name)


def _spec(name="s", fields=None) -> ConfigSpec:
    return ConfigSpec(name=name, fields=fields or [])


def _snapshot(name="cfg", data=None) -> ConfigSnapshot:
    return ConfigSnapshot(name=name, data=data or {"a": 1})


# ─── ConfigField ──────────────────────────────────────────────────────────────

class TestConfigFieldExtra:
    def test_stores_name(self):
        assert _field(name="lr").name == "lr"

    def test_stores_default(self):
        assert _field(default=42).default == 42

    def test_stores_required(self):
        assert _field(required=True).required is True

    def test_stores_type_name(self):
        assert _field(type_name="float").type_name == "float"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigField(name="", default=0)

    def test_invalid_type_name_raises(self):
        with pytest.raises(ValueError):
            ConfigField(name="x", type_name="list")

    def test_valid_type_names(self):
        for t in ("int", "float", "str", "bool", "any"):
            f = ConfigField(name="x", type_name=t)
            assert f.type_name == t

    def test_default_type_name_any(self):
        f = ConfigField(name="x")
        assert f.type_name == "any"


# ─── ConfigSpec ───────────────────────────────────────────────────────────────

class TestConfigSpecExtra:
    def test_stores_name(self):
        assert _spec(name="myspec").name == "myspec"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigSpec(name="")

    def test_required_fields(self):
        spec = _spec(fields=[_field(required=True), _field(required=False)])
        assert len(spec.required_fields) == 1

    def test_optional_fields(self):
        spec = _spec(fields=[_field(required=True), _field(required=False)])
        assert len(spec.optional_fields) == 1

    def test_field_names(self):
        spec = _spec(fields=[_field(name="a"), _field(name="b")])
        assert spec.field_names() == ["a", "b"]

    def test_empty_fields(self):
        spec = _spec(fields=[])
        assert spec.field_names() == []


# ─── ConfigSnapshot ───────────────────────────────────────────────────────────

class TestConfigSnapshotExtra:
    def test_stores_name(self):
        assert _snapshot(name="test").name == "test"

    def test_stores_data(self):
        s = _snapshot(data={"k": 99})
        assert s.data["k"] == 99

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigSnapshot(name="", data={})

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            ConfigSnapshot(name="s", data={}, timestamp=-1.0)

    def test_get_existing_key(self):
        s = _snapshot(data={"x": 5})
        assert s.get("x") == 5

    def test_get_missing_returns_default(self):
        s = _snapshot(data={})
        assert s.get("missing", "default") == "default"

    def test_has_true(self):
        s = _snapshot(data={"y": 7})
        assert s.has("y") is True

    def test_has_false(self):
        s = _snapshot(data={})
        assert s.has("z") is False


# ─── validate_field_type ──────────────────────────────────────────────────────

class TestValidateFieldTypeExtra:
    def test_any_always_true(self):
        assert validate_field_type("anything", "any") is True

    def test_int_valid(self):
        assert validate_field_type(5, "int") is True

    def test_int_invalid(self):
        assert validate_field_type("five", "int") is False

    def test_float_int_is_valid(self):
        assert validate_field_type(3, "float") is True

    def test_float_invalid(self):
        assert validate_field_type("x", "float") is False

    def test_str_valid(self):
        assert validate_field_type("hello", "str") is True

    def test_bool_valid(self):
        assert validate_field_type(True, "bool") is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            validate_field_type(1, "list")


# ─── validate_config ──────────────────────────────────────────────────────────

class TestValidateConfigExtra:
    def test_valid_returns_empty(self):
        spec = _spec(fields=[_field(name="n", required=True, type_name="int")])
        assert validate_config({"n": 5}, spec) == []

    def test_missing_required_gives_error(self):
        spec = _spec(fields=[_field(name="n", required=True)])
        errors = validate_config({}, spec)
        assert len(errors) > 0

    def test_wrong_type_gives_error(self):
        spec = _spec(fields=[_field(name="n", required=True, type_name="int")])
        errors = validate_config({"n": "not_int"}, spec)
        assert len(errors) > 0

    def test_optional_missing_ok(self):
        spec = _spec(fields=[_field(name="opt", required=False)])
        assert validate_config({}, spec) == []

    def test_empty_spec_valid(self):
        assert validate_config({"extra": 1}, _spec(fields=[])) == []


# ─── load_config ──────────────────────────────────────────────────────────────

class TestLoadConfigExtra:
    def test_returns_dict(self):
        spec = _spec(fields=[_field(name="x", default=0)])
        result = load_config({}, spec)
        assert isinstance(result, dict)

    def test_fills_defaults(self):
        spec = _spec(fields=[_field(name="x", default=42)])
        result = load_config({}, spec)
        assert result["x"] == 42

    def test_overrides_default(self):
        spec = _spec(fields=[_field(name="x", default=0)])
        result = load_config({"x": 99}, spec)
        assert result["x"] == 99

    def test_missing_required_raises(self):
        spec = _spec(fields=[_field(name="req", required=True)])
        with pytest.raises(ValueError):
            load_config({}, spec)


# ─── merge_configs ────────────────────────────────────────────────────────────

class TestMergeConfigsExtra:
    def test_returns_dict(self):
        assert isinstance(merge_configs([{"a": 1}]), dict)

    def test_later_overrides_earlier(self):
        result = merge_configs([{"a": 1}, {"a": 2}])
        assert result["a"] == 2

    def test_combines_keys(self):
        result = merge_configs([{"a": 1}, {"b": 2}])
        assert result["a"] == 1 and result["b"] == 2

    def test_empty_list(self):
        assert merge_configs([]) == {}


# ─── diff_configs ─────────────────────────────────────────────────────────────

class TestDiffConfigsExtra:
    def test_identical_empty_diff(self):
        assert diff_configs({"a": 1}, {"a": 1}) == {}

    def test_different_values_in_diff(self):
        d = diff_configs({"a": 1}, {"a": 2})
        assert "a" in d

    def test_diff_tuple_values(self):
        d = diff_configs({"a": 1}, {"a": 2})
        assert d["a"] == (1, 2)

    def test_missing_key_in_diff(self):
        d = diff_configs({"a": 1}, {})
        assert "a" in d


# ─── make_config_snapshot ─────────────────────────────────────────────────────

class TestMakeConfigSnapshotExtra:
    def test_returns_snapshot(self):
        s = make_config_snapshot("test", {"x": 1})
        assert isinstance(s, ConfigSnapshot)

    def test_stores_name(self):
        s = make_config_snapshot("myname", {})
        assert s.name == "myname"

    def test_timestamp_positive(self):
        s = make_config_snapshot("s", {})
        assert s.timestamp >= 0.0

    def test_data_copied(self):
        d = {"k": 5}
        s = make_config_snapshot("s", d)
        d["k"] = 99
        assert s.data["k"] == 5  # original should not be affected


# ─── batch_validate ───────────────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_returns_list(self):
        spec = _spec(fields=[_field(name="x", required=True, type_name="int")])
        result = batch_validate([{"x": 1}], spec)
        assert isinstance(result, list)

    def test_length_matches(self):
        spec = _spec(fields=[])
        result = batch_validate([{}, {}, {}], spec)
        assert len(result) == 3

    def test_empty_list(self):
        assert batch_validate([], _spec(fields=[])) == []

    def test_errors_for_invalid(self):
        spec = _spec(fields=[_field(name="n", required=True)])
        result = batch_validate([{}, {"n": 1}], spec)
        assert len(result[0]) > 0
        assert result[1] == []
