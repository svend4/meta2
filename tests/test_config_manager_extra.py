"""Extra tests for puzzle_reconstruction/utils/config_manager.py"""
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


# ─── TestConfigFieldExtra ─────────────────────────────────────────────────────

class TestConfigFieldExtra:
    def test_none_default_valid(self):
        f = ConfigField(name="x", default=None)
        assert f.default is None

    def test_negative_default_valid(self):
        f = ConfigField(name="steps", default=-1)
        assert f.default == -1

    def test_bool_default_valid(self):
        f = ConfigField(name="flag", default=False)
        assert f.default is False

    def test_string_default_valid(self):
        f = ConfigField(name="model", default="resnet")
        assert f.default == "resnet"

    def test_required_false_by_default(self):
        f = ConfigField(name="x")
        assert f.required is False

    def test_required_true_stored(self):
        f = ConfigField(name="x", required=True)
        assert f.required is True

    def test_description_empty_by_default(self):
        f = ConfigField(name="x")
        assert f.description == ""

    def test_type_name_int(self):
        f = ConfigField(name="n", type_name="int")
        assert f.type_name == "int"


# ─── TestConfigSpecExtra ─────────────────────────────────────────────────────

class TestConfigSpecExtra:
    def _spec_all_required(self):
        return ConfigSpec(
            name="s",
            fields=[
                ConfigField(name="a", required=True, type_name="int"),
                ConfigField(name="b", required=True, type_name="float"),
                ConfigField(name="c", required=True, type_name="str"),
            ],
        )

    def test_three_required_fields(self):
        spec = self._spec_all_required()
        assert len(spec.required_fields) == 3

    def test_no_optional_when_all_required(self):
        spec = self._spec_all_required()
        assert spec.optional_fields == []

    def test_field_names_returns_all(self):
        spec = self._spec_all_required()
        assert set(spec.field_names()) == {"a", "b", "c"}

    def test_field_names_is_list(self):
        spec = ConfigSpec(name="s", fields=[ConfigField(name="x")])
        assert isinstance(spec.field_names(), list)

    def test_many_optional_fields(self):
        spec = ConfigSpec(
            name="s",
            fields=[ConfigField(name=f"f{i}") for i in range(10)],
        )
        assert len(spec.optional_fields) == 10

    def test_spec_name_stored(self):
        spec = ConfigSpec(name="my_spec")
        assert spec.name == "my_spec"


# ─── TestConfigSnapshotExtra ──────────────────────────────────────────────────

class TestConfigSnapshotExtra:
    def test_large_data_dict(self):
        data = {f"key_{i}": i for i in range(100)}
        s = ConfigSnapshot(name="big", data=data, timestamp=1.0)
        assert s.get("key_50") == 50

    def test_large_timestamp(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=1e12)
        assert s.timestamp == pytest.approx(1e12)

    def test_has_returns_false_for_absent(self):
        s = ConfigSnapshot(name="x", data={"a": 1}, timestamp=1.0)
        assert s.has("b") is False

    def test_get_with_none_value(self):
        s = ConfigSnapshot(name="x", data={"key": None}, timestamp=1.0)
        assert s.get("key") is None

    def test_get_with_custom_default(self):
        s = ConfigSnapshot(name="x", data={}, timestamp=1.0)
        assert s.get("missing", 42) == 42

    def test_data_stored_correctly(self):
        data = {"lr": 0.001, "batch": 32}
        s = ConfigSnapshot(name="train", data=data, timestamp=5.0)
        assert s.data["lr"] == pytest.approx(0.001)
        assert s.data["batch"] == 32


# ─── TestValidateFieldTypeExtra ───────────────────────────────────────────────

class TestValidateFieldTypeExtra:
    def test_none_with_any(self):
        assert validate_field_type(None, "any") is True

    def test_false_with_bool(self):
        assert validate_field_type(False, "bool") is True

    def test_true_with_bool(self):
        assert validate_field_type(True, "bool") is True

    def test_zero_with_int(self):
        assert validate_field_type(0, "int") is True

    def test_zero_float_with_float(self):
        assert validate_field_type(0.0, "float") is True

    def test_empty_string_with_str(self):
        assert validate_field_type("", "str") is True

    def test_list_with_any(self):
        assert validate_field_type([1, 2, 3], "any") is True

    def test_none_with_int_fails(self):
        assert validate_field_type(None, "int") is False


# ─── TestValidateConfigExtra ──────────────────────────────────────────────────

class TestValidateConfigExtra:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[
                ConfigField(name="lr", required=True, type_name="float"),
                ConfigField(name="name", default="model", type_name="str"),
            ],
        )

    def test_extra_keys_no_error(self):
        data = {"lr": 0.01, "extra_key": 999}
        errors = validate_config(data, self._spec())
        assert errors == []

    def test_all_correct_types(self):
        data = {"lr": 0.01, "name": "net"}
        errors = validate_config(data, self._spec())
        assert errors == []

    def test_type_mismatch_required(self):
        data = {"lr": "not_a_float", "name": "net"}
        errors = validate_config(data, self._spec())
        assert any("lr" in e for e in errors)

    def test_returns_list(self):
        errors = validate_config({"lr": 0.01}, self._spec())
        assert isinstance(errors, list)

    def test_all_optional_absent_no_error(self):
        spec = ConfigSpec(name="s", fields=[
            ConfigField(name="a", default=1, type_name="int"),
            ConfigField(name="b", default=2, type_name="int"),
        ])
        errors = validate_config({}, spec)
        assert errors == []


# ─── TestLoadConfigExtra ──────────────────────────────────────────────────────

class TestLoadConfigExtra:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[
                ConfigField(name="lr", required=True, type_name="float"),
                ConfigField(name="verbose", default=False, type_name="bool"),
                ConfigField(name="name", default="model", type_name="str"),
            ],
        )

    def test_boolean_default_loaded(self):
        cfg = load_config({"lr": 0.01}, self._spec())
        assert cfg["verbose"] is False

    def test_string_default_loaded(self):
        cfg = load_config({"lr": 0.01}, self._spec())
        assert cfg["name"] == "model"

    def test_extra_keys_not_in_result(self):
        cfg = load_config({"lr": 0.01, "foo": "bar"}, self._spec())
        assert "foo" not in cfg

    def test_empty_spec_empty_result(self):
        cfg = load_config({"any": 1}, ConfigSpec(name="s"))
        assert cfg == {}

    def test_all_defaults_used(self):
        spec = ConfigSpec(name="s", fields=[
            ConfigField(name="a", default=10, type_name="int"),
            ConfigField(name="b", default=20, type_name="int"),
        ])
        cfg = load_config({}, spec)
        assert cfg["a"] == 10
        assert cfg["b"] == 20


# ─── TestMergeConfigsExtra ────────────────────────────────────────────────────

class TestMergeConfigsExtra:
    def test_same_keys_all_overridden(self):
        result = merge_configs([{"a": 1}, {"a": 2}, {"a": 3}])
        assert result["a"] == 3

    def test_no_mutation_of_second(self):
        d1 = {"a": 1}
        d2 = {"b": 2}
        merge_configs([d1, d2])
        assert d2 == {"b": 2}

    def test_five_dicts(self):
        dicts = [{"k": i} for i in range(5)]
        result = merge_configs(dicts)
        assert result["k"] == 4

    def test_disjoint_keys_all_present(self):
        result = merge_configs([{"a": 1}, {"b": 2}, {"c": 3}])
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_returns_dict(self):
        result = merge_configs([{"a": 1}])
        assert isinstance(result, dict)


# ─── TestDiffConfigsExtra ─────────────────────────────────────────────────────

class TestDiffConfigsExtra:
    def test_different_types_same_key(self):
        diff = diff_configs({"a": 1}, {"a": "1"})
        assert "a" in diff

    def test_none_value_diff(self):
        diff = diff_configs({"a": None}, {"a": 1})
        assert "a" in diff

    def test_bool_to_int_diff(self):
        diff = diff_configs({"flag": True}, {"flag": False})
        assert "flag" in diff

    def test_empty_base_all_new(self):
        diff = diff_configs({}, {"a": 1, "b": 2})
        assert "a" in diff
        assert "b" in diff

    def test_empty_other_all_removed(self):
        diff = diff_configs({"a": 1, "b": 2}, {})
        assert "a" in diff
        assert "b" in diff

    def test_returns_dict(self):
        assert isinstance(diff_configs({}, {}), dict)


# ─── TestMakeConfigSnapshotExtra ─────────────────────────────────────────────

class TestMakeConfigSnapshotExtra:
    def test_various_data_types(self):
        data = {"int": 1, "float": 3.14, "str": "x", "bool": True}
        s = make_config_snapshot("run", data)
        assert s.data["int"] == 1
        assert s.data["bool"] is True

    def test_empty_data(self):
        s = make_config_snapshot("run", {})
        assert s.data == {}

    def test_two_snapshots_different_timestamps(self):
        s1 = make_config_snapshot("r1", {})
        time.sleep(0.01)
        s2 = make_config_snapshot("r2", {})
        assert s2.timestamp >= s1.timestamp

    def test_name_with_spaces(self):
        s = make_config_snapshot("my run 1", {})
        assert s.name == "my run 1"


# ─── TestBatchValidateExtra ───────────────────────────────────────────────────

class TestBatchValidateExtra:
    def _spec(self):
        return ConfigSpec(
            name="s",
            fields=[ConfigField(name="x", required=True, type_name="int")],
        )

    def test_all_invalid(self):
        results = batch_validate([{}, {}, {}], self._spec())
        assert all(len(r) > 0 for r in results)

    def test_five_valid(self):
        data = [{"x": i} for i in range(5)]
        results = batch_validate(data, self._spec())
        assert all(r == [] for r in results)

    def test_returns_list_of_lists(self):
        results = batch_validate([{"x": 1}], self._spec())
        assert isinstance(results, list)
        assert isinstance(results[0], list)

    def test_mixed_valid_invalid(self):
        data = [{"x": 1}, {}, {"x": 3}, {}]
        results = batch_validate(data, self._spec())
        assert results[0] == []
        assert len(results[1]) > 0
        assert results[2] == []
        assert len(results[3]) > 0
