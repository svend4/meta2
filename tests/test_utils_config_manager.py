"""Tests for puzzle_reconstruction.utils.config_manager"""
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


# ─── ConfigField ──────────────────────────────────────────────────────────────

def test_config_field_basic():
    f = ConfigField(name="lr", default=0.01, type_name="float")
    assert f.name == "lr"
    assert f.default == pytest.approx(0.01)
    assert f.type_name == "float"


def test_config_field_required():
    f = ConfigField(name="epochs", required=True, type_name="int")
    assert f.required is True


def test_config_field_invalid_name():
    with pytest.raises(ValueError):
        ConfigField(name="")


def test_config_field_invalid_type_name():
    with pytest.raises(ValueError):
        ConfigField(name="x", type_name="list")


def test_config_field_valid_type_names():
    for tn in ["int", "float", "str", "bool", "any"]:
        f = ConfigField(name="x", type_name=tn)
        assert f.type_name == tn


# ─── ConfigSpec ───────────────────────────────────────────────────────────────

def test_config_spec_basic():
    spec = ConfigSpec(name="myspec")
    assert spec.name == "myspec"
    assert spec.fields == []


def test_config_spec_invalid_name():
    with pytest.raises(ValueError):
        ConfigSpec(name="")


def test_config_spec_required_fields():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("a", required=True),
        ConfigField("b", required=False),
    ])
    assert len(spec.required_fields) == 1
    assert spec.required_fields[0].name == "a"


def test_config_spec_optional_fields():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("a", required=True),
        ConfigField("b", required=False),
        ConfigField("c", required=False),
    ])
    assert len(spec.optional_fields) == 2


def test_config_spec_field_names():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("x"),
        ConfigField("y"),
    ])
    assert spec.field_names() == ["x", "y"]


# ─── ConfigSnapshot ───────────────────────────────────────────────────────────

def test_config_snapshot_basic():
    snap = ConfigSnapshot(name="main", data={"lr": 0.01}, timestamp=1000.0)
    assert snap.name == "main"
    assert snap.get("lr") == pytest.approx(0.01)
    assert snap.has("lr") is True
    assert snap.has("missing") is False


def test_config_snapshot_get_default():
    snap = ConfigSnapshot(name="s", data={})
    assert snap.get("x", 99) == 99


def test_config_snapshot_invalid_name():
    with pytest.raises(ValueError):
        ConfigSnapshot(name="", data={})


def test_config_snapshot_invalid_timestamp():
    with pytest.raises(ValueError):
        ConfigSnapshot(name="s", data={}, timestamp=-1.0)


# ─── validate_field_type ──────────────────────────────────────────────────────

def test_validate_field_type_int():
    assert validate_field_type(5, "int") is True
    assert validate_field_type(5.0, "int") is False


def test_validate_field_type_float():
    assert validate_field_type(1.0, "float") is True
    assert validate_field_type(1, "float") is True  # int is valid float
    assert validate_field_type("x", "float") is False


def test_validate_field_type_str():
    assert validate_field_type("hello", "str") is True
    assert validate_field_type(123, "str") is False


def test_validate_field_type_bool():
    assert validate_field_type(True, "bool") is True
    assert validate_field_type(0, "bool") is False


def test_validate_field_type_any():
    assert validate_field_type(None, "any") is True
    assert validate_field_type([1, 2], "any") is True


def test_validate_field_type_unknown():
    with pytest.raises(ValueError):
        validate_field_type(1, "list")


# ─── validate_config ──────────────────────────────────────────────────────────

def test_validate_config_ok():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", required=True, type_name="float"),
    ])
    errors = validate_config({"lr": 0.01}, spec)
    assert errors == []


def test_validate_config_missing_required():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", required=True, type_name="float"),
    ])
    errors = validate_config({}, spec)
    assert len(errors) == 1
    assert "lr" in errors[0]


def test_validate_config_wrong_type():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", required=True, type_name="float"),
    ])
    errors = validate_config({"lr": "bad"}, spec)
    assert len(errors) == 1


# ─── load_config ──────────────────────────────────────────────────────────────

def test_load_config_fills_defaults():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", default=0.001, required=False, type_name="float"),
        ConfigField("epochs", default=10, required=False, type_name="int"),
    ])
    result = load_config({}, spec)
    assert result["lr"] == pytest.approx(0.001)
    assert result["epochs"] == 10


def test_load_config_required_missing_raises():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("batch_size", required=True, type_name="int"),
    ])
    with pytest.raises(ValueError):
        load_config({}, spec)


def test_load_config_provided_values_used():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", default=0.001, required=False, type_name="float"),
    ])
    result = load_config({"lr": 0.1}, spec)
    assert result["lr"] == pytest.approx(0.1)


# ─── merge_configs ────────────────────────────────────────────────────────────

def test_merge_configs_later_overrides():
    result = merge_configs([{"a": 1, "b": 2}, {"b": 99, "c": 3}])
    assert result == {"a": 1, "b": 99, "c": 3}


def test_merge_configs_empty():
    result = merge_configs([])
    assert result == {}


# ─── diff_configs ────────────────────────────────────────────────────────────

def test_diff_configs_basic():
    base = {"lr": 0.01, "epochs": 10}
    other = {"lr": 0.001, "epochs": 10}
    diff = diff_configs(base, other)
    assert "lr" in diff
    assert diff["lr"] == (0.01, 0.001)
    assert "epochs" not in diff


def test_diff_configs_key_only_in_one():
    base = {"a": 1}
    other = {"b": 2}
    diff = diff_configs(base, other)
    assert "a" in diff
    assert "b" in diff


# ─── make_config_snapshot ────────────────────────────────────────────────────

def test_make_config_snapshot_returns_snapshot():
    snap = make_config_snapshot("run1", {"lr": 0.01})
    assert isinstance(snap, ConfigSnapshot)
    assert snap.name == "run1"
    assert snap.get("lr") == pytest.approx(0.01)
    assert snap.timestamp >= 0


def test_make_config_snapshot_data_is_copy():
    data = {"x": 1}
    snap = make_config_snapshot("s", data)
    data["x"] = 999
    assert snap.get("x") == 1


# ─── batch_validate ──────────────────────────────────────────────────────────

def test_batch_validate():
    spec = ConfigSpec(name="s", fields=[
        ConfigField("lr", required=True, type_name="float"),
    ])
    data_list = [{"lr": 0.01}, {}, {"lr": "bad"}]
    results = batch_validate(data_list, spec)
    assert len(results) == 3
    assert results[0] == []
    assert len(results[1]) > 0
    assert len(results[2]) > 0
