"""Тесты для puzzle_reconstruction.utils.config_manager."""
import time

import pytest

from puzzle_reconstruction.utils.config_manager import (
    ConfigField,
    ConfigSnapshot,
    ConfigSpec,
    batch_validate,
    diff_configs,
    load_config,
    make_config_snapshot,
    merge_configs,
    validate_config,
    validate_field_type,
)


# ─── TestConfigField ──────────────────────────────────────────────────────────

class TestConfigField:
    def test_basic_construction(self):
        f = ConfigField(name="lr", default=0.01, type_name="float")
        assert f.name == "lr"
        assert f.default == 0.01
        assert f.type_name == "float"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigField(name="")

    def test_invalid_type_name_raises(self):
        with pytest.raises(ValueError):
            ConfigField(name="x", type_name="list")

    def test_all_valid_type_names(self):
        for t in ("int", "float", "str", "bool", "any"):
            f = ConfigField(name="x", type_name=t)
            assert f.type_name == t

    def test_defaults(self):
        f = ConfigField(name="x")
        assert f.required is False
        assert f.type_name == "any"
        assert f.description == ""

    def test_required_field(self):
        f = ConfigField(name="batch_size", required=True, type_name="int")
        assert f.required is True

    def test_description_stored(self):
        f = ConfigField(name="x", description="Learning rate")
        assert f.description == "Learning rate"


# ─── TestConfigSpec ───────────────────────────────────────────────────────────

class TestConfigSpec:
    def _spec(self):
        return ConfigSpec(
            name="model",
            fields=[
                ConfigField(name="lr", required=True, type_name="float"),
                ConfigField(name="epochs", required=True, type_name="int"),
                ConfigField(name="dropout", default=0.1, type_name="float"),
                ConfigField(name="verbose", default=False, type_name="bool"),
            ],
        )

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigSpec(name="")

    def test_required_fields(self):
        spec = self._spec()
        req = spec.required_fields
        assert {f.name for f in req} == {"lr", "epochs"}

    def test_optional_fields(self):
        spec = self._spec()
        opt = spec.optional_fields
        assert {f.name for f in opt} == {"dropout", "verbose"}

    def test_field_names(self):
        spec = self._spec()
        assert set(spec.field_names()) == {"lr", "epochs", "dropout", "verbose"}

    def test_empty_fields(self):
        spec = ConfigSpec(name="empty")
        assert spec.required_fields == []
        assert spec.optional_fields == []

    def test_no_required_fields(self):
        spec = ConfigSpec(
            name="s",
            fields=[ConfigField(name="a"), ConfigField(name="b")],
        )
        assert spec.required_fields == []
        assert len(spec.optional_fields) == 2


# ─── TestConfigSnapshot ───────────────────────────────────────────────────────

class TestConfigSnapshot:
    def test_basic_construction(self):
        s = ConfigSnapshot(name="run1", data={"lr": 0.01}, timestamp=1000.0)
        assert s.name == "run1"
        assert s.timestamp == 1000.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ConfigSnapshot(name="", data={})

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            ConfigSnapshot(name="x", data={}, timestamp=-1.0)

    def test_zero_timestamp_ok(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=0.0)
        assert s.timestamp == 0.0

    def test_get_existing_key(self):
        s = ConfigSnapshot(name="x", data={"k": 42}, timestamp=0.0)
        assert s.get("k") == 42

    def test_get_missing_key_returns_default(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=0.0)
        assert s.get("missing", "fallback") == "fallback"

    def test_get_missing_key_none_by_default(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=0.0)
        assert s.get("missing") is None

    def test_has_existing_key(self):
        s = ConfigSnapshot(name="x", data={"k": 1}, timestamp=0.0)
        assert s.has("k") is True

    def test_has_missing_key(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=0.0)
        assert s.has("k") is False


# ─── TestValidateFieldType ────────────────────────────────────────────────────

class TestValidateFieldType:
    def test_any_always_true(self):
        for v in (1, 1.5, "s", True, None, []):
            assert validate_field_type(v, "any") is True

    def test_int_pass(self):
        assert validate_field_type(5, "int") is True

    def test_int_fail(self):
        assert validate_field_type(3.14, "int") is False

    def test_float_pass_int_value(self):
        # int is acceptable for float
        assert validate_field_type(3, "float") is True

    def test_float_pass_float_value(self):
        assert validate_field_type(3.14, "float") is True

    def test_float_fail_str(self):
        assert validate_field_type("3.14", "float") is False

    def test_str_pass(self):
        assert validate_field_type("hello", "str") is True

    def test_str_fail(self):
        assert validate_field_type(1, "str") is False

    def test_bool_pass(self):
        assert validate_field_type(True, "bool") is True

    def test_bool_fail(self):
        assert validate_field_type(1, "bool") is False

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            validate_field_type(1, "list")


# ─── TestValidateConfig ───────────────────────────────────────────────────────

class TestValidateConfig:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[
                ConfigField(name="lr", required=True, type_name="float"),
                ConfigField(name="epochs", required=True, type_name="int"),
                ConfigField(name="name", default="model", type_name="str"),
            ],
        )

    def test_valid_full_config(self):
        data = {"lr": 0.01, "epochs": 10, "name": "net"}
        errors = validate_config(data, self._spec())
        assert errors == []

    def test_missing_required_field(self):
        data = {"lr": 0.01}
        errors = validate_config(data, self._spec())
        assert any("epochs" in e for e in errors)

    def test_both_required_missing(self):
        errors = validate_config({}, self._spec())
        assert len(errors) >= 2

    def test_wrong_type_required_field(self):
        data = {"lr": "high", "epochs": 10}
        errors = validate_config(data, self._spec())
        assert any("lr" in e for e in errors)

    def test_wrong_type_optional_field(self):
        data = {"lr": 0.01, "epochs": 10, "name": 99}
        errors = validate_config(data, self._spec())
        assert any("name" in e for e in errors)

    def test_optional_field_absent_no_error(self):
        data = {"lr": 0.01, "epochs": 10}
        errors = validate_config(data, self._spec())
        assert errors == []

    def test_empty_spec_always_valid(self):
        spec = ConfigSpec(name="empty")
        errors = validate_config({"anything": 1}, spec)
        assert errors == []


# ─── TestLoadConfig ───────────────────────────────────────────────────────────

class TestLoadConfig:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[
                ConfigField(name="lr", required=True, type_name="float"),
                ConfigField(name="epochs", default=10, type_name="int"),
                ConfigField(name="verbose", default=False, type_name="bool"),
            ],
        )

    def test_fills_defaults(self):
        cfg = load_config({"lr": 0.01}, self._spec())
        assert cfg["epochs"] == 10
        assert cfg["verbose"] is False

    def test_provided_values_override_defaults(self):
        cfg = load_config({"lr": 0.001, "epochs": 20}, self._spec())
        assert cfg["epochs"] == 20

    def test_missing_required_raises(self):
        with pytest.raises(ValueError):
            load_config({}, self._spec())

    def test_all_provided(self):
        data = {"lr": 0.01, "epochs": 5, "verbose": True}
        cfg = load_config(data, self._spec())
        assert cfg == data

    def test_extra_keys_ignored(self):
        data = {"lr": 0.01, "unknown": "x"}
        cfg = load_config(data, self._spec())
        assert "unknown" not in cfg


# ─── TestMergeConfigs ─────────────────────────────────────────────────────────

class TestMergeConfigs:
    def test_single_dict(self):
        result = merge_configs([{"a": 1}])
        assert result == {"a": 1}

    def test_later_overrides_earlier(self):
        result = merge_configs([{"a": 1, "b": 2}, {"b": 99}])
        assert result["b"] == 99
        assert result["a"] == 1

    def test_three_way_merge(self):
        result = merge_configs([{"a": 1}, {"b": 2}, {"a": 10, "c": 3}])
        assert result == {"a": 10, "b": 2, "c": 3}

    def test_empty_list_returns_empty(self):
        assert merge_configs([]) == {}

    def test_all_empty_dicts(self):
        assert merge_configs([{}, {}, {}]) == {}

    def test_does_not_mutate_inputs(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        merge_configs([d1, d2])
        assert d1["a"] == 1


# ─── TestDiffConfigs ──────────────────────────────────────────────────────────

class TestDiffConfigs:
    def test_identical_configs_empty_diff(self):
        d = {"a": 1, "b": "x"}
        assert diff_configs(d, d.copy()) == {}

    def test_changed_value(self):
        diff = diff_configs({"a": 1}, {"a": 2})
        assert "a" in diff
        assert diff["a"] == (1, 2)

    def test_key_only_in_base(self):
        diff = diff_configs({"a": 1, "b": 2}, {"a": 1})
        assert "b" in diff
        assert diff["b"][0] == 2
        assert diff["b"][1] is None

    def test_key_only_in_other(self):
        diff = diff_configs({"a": 1}, {"a": 1, "b": 2})
        assert "b" in diff
        assert diff["b"][0] is None
        assert diff["b"][1] == 2

    def test_multiple_changes(self):
        diff = diff_configs({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 99, "d": 4})
        assert "b" in diff
        assert "c" in diff
        assert "d" in diff
        assert "a" not in diff

    def test_empty_dicts_no_diff(self):
        assert diff_configs({}, {}) == {}


# ─── TestMakeConfigSnapshot ───────────────────────────────────────────────────

class TestMakeConfigSnapshot:
    def test_name_preserved(self):
        s = make_config_snapshot("run", {"lr": 0.01})
        assert s.name == "run"

    def test_data_preserved(self):
        data = {"lr": 0.01, "epochs": 5}
        s = make_config_snapshot("run", data)
        assert s.data == data

    def test_data_is_copy(self):
        data = {"lr": 0.01}
        s = make_config_snapshot("run", data)
        data["lr"] = 99
        assert s.data["lr"] == 0.01

    def test_timestamp_is_positive(self):
        before = time.time()
        s = make_config_snapshot("run", {})
        after = time.time()
        assert before <= s.timestamp <= after

    def test_returns_snapshot_type(self):
        s = make_config_snapshot("run", {})
        assert isinstance(s, ConfigSnapshot)


# ─── TestBatchValidate ────────────────────────────────────────────────────────

class TestBatchValidate:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[ConfigField(name="x", required=True, type_name="int")],
        )

    def test_empty_list(self):
        assert batch_validate([], self._spec()) == []

    def test_all_valid(self):
        results = batch_validate([{"x": 1}, {"x": 2}], self._spec())
        assert all(r == [] for r in results)

    def test_one_invalid(self):
        results = batch_validate([{"x": 1}, {}], self._spec())
        assert results[0] == []
        assert len(results[1]) > 0

    def test_length_matches_input(self):
        data_list = [{"x": i} for i in range(5)]
        results = batch_validate(data_list, self._spec())
        assert len(results) == 5
