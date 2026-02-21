"""Тесты для puzzle_reconstruction.io.metadata_writer."""
import json
import pytest
from puzzle_reconstruction.io.metadata_writer import (
    WriterConfig,
    MetadataRecord,
    MetadataCollection,
    write_json,
    read_json,
    write_csv,
    write_summary,
    filter_by_score,
    merge_collections,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rec(fid: int = 0, x: float = 0.0, y: float = 0.0,
         rot: float = 0.0, score: float = 0.8,
         extra: dict = None) -> MetadataRecord:
    return MetadataRecord(fragment_id=fid, position=(x, y),
                          rotation_deg=rot, score=score,
                          extra=extra or {})


def _col(*recs) -> MetadataCollection:
    if not recs:
        recs = (_rec(0), _rec(1, x=10.0, score=0.6), _rec(2, x=20.0, score=0.4))
    return MetadataCollection(records=list(recs))


def _col_with_meta() -> MetadataCollection:
    col = _col()
    col.global_meta = {"title": "test", "version": 1}
    return col


# ─── TestWriterConfig ─────────────────────────────────────────────────────────

class TestWriterConfig:
    def test_defaults(self):
        cfg = WriterConfig()
        assert cfg.indent == 2
        assert cfg.sort_keys is True
        assert cfg.csv_dialect == "excel"
        assert cfg.float_prec == 6

    def test_indent_zero_ok(self):
        cfg = WriterConfig(indent=0)
        assert cfg.indent == 0

    def test_indent_none_ok(self):
        cfg = WriterConfig(indent=None)
        assert cfg.indent is None

    def test_indent_neg_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(indent=-1)

    def test_float_prec_zero_ok(self):
        cfg = WriterConfig(float_prec=0)
        assert cfg.float_prec == 0

    def test_float_prec_neg_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(float_prec=-1)

    def test_csv_dialect_empty_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(csv_dialect="")

    def test_custom(self):
        cfg = WriterConfig(indent=4, sort_keys=False,
                           csv_dialect="unix", float_prec=3)
        assert cfg.indent == 4
        assert cfg.sort_keys is False


# ─── TestMetadataRecord ───────────────────────────────────────────────────────

class TestMetadataRecord:
    def test_basic(self):
        r = _rec(1, 5.0, 3.0, 45.0, 0.9)
        assert r.fragment_id == 1
        assert r.position == (5.0, 3.0)
        assert r.rotation_deg == pytest.approx(45.0)
        assert r.score == pytest.approx(0.9)

    def test_score_zero_ok(self):
        r = _rec(score=0.0)
        assert r.score == 0.0

    def test_score_one_ok(self):
        r = _rec(score=1.0)
        assert r.score == 1.0

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(0.0, 0.0), score=-0.1)

    def test_score_above_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(0.0, 0.0), score=1.1)

    def test_position_wrong_length_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(1.0,), score=0.5)

    def test_position_3_values_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(1.0, 2.0, 3.0), score=0.5)

    def test_to_dict_keys(self):
        r = _rec(1, 5.0, 3.0)
        d = r.to_dict()
        assert "fragment_id" in d
        assert "x" in d
        assert "y" in d
        assert "rotation_deg" in d
        assert "score" in d

    def test_to_dict_values(self):
        r = _rec(7, 2.5, 3.5, 90.0, 0.75)
        d = r.to_dict()
        assert d["fragment_id"] == 7
        assert d["x"] == pytest.approx(2.5)
        assert d["y"] == pytest.approx(3.5)

    def test_to_dict_extra_included(self):
        r = _rec(extra={"foo": "bar", "count": 3})
        d = r.to_dict()
        assert d["foo"] == "bar"
        assert d["count"] == 3

    def test_to_dict_float_prec(self):
        r = _rec(score=0.123456789)
        d2 = r.to_dict(float_prec=2)
        assert d2["score"] == pytest.approx(0.12, abs=0.01)

    def test_negative_rotation_ok(self):
        r = _rec(rot=-45.0)
        assert r.rotation_deg == pytest.approx(-45.0)

    def test_zero_position_ok(self):
        r = _rec(x=0.0, y=0.0)
        assert r.position == (0.0, 0.0)


# ─── TestMetadataCollection ───────────────────────────────────────────────────

class TestMetadataCollection:
    def test_n_fragments(self):
        col = _col()
        assert col.n_fragments == 3

    def test_n_fragments_empty(self):
        col = MetadataCollection()
        assert col.n_fragments == 0

    def test_fragment_ids(self):
        col = _col()
        assert col.fragment_ids == [0, 1, 2]

    def test_mean_score(self):
        col = _col()
        expected = (0.8 + 0.6 + 0.4) / 3.0
        assert col.mean_score == pytest.approx(expected)

    def test_mean_score_empty(self):
        col = MetadataCollection()
        assert col.mean_score == pytest.approx(0.0)

    def test_get_record_found(self):
        col = _col()
        r = col.get_record(1)
        assert r is not None
        assert r.fragment_id == 1

    def test_get_record_not_found(self):
        col = _col()
        assert col.get_record(99) is None

    def test_add(self):
        col = MetadataCollection()
        col.add(_rec(5))
        assert col.n_fragments == 1
        assert col.get_record(5) is not None

    def test_to_list_length(self):
        col = _col()
        lst = col.to_list()
        assert len(lst) == 3

    def test_to_list_dicts(self):
        col = _col()
        for item in col.to_list():
            assert isinstance(item, dict)
            assert "fragment_id" in item

    def test_global_meta_stored(self):
        col = _col_with_meta()
        assert col.global_meta["title"] == "test"


# ─── TestWriteJson ────────────────────────────────────────────────────────────

class TestWriteJson:
    def test_returns_string(self):
        result = write_json(_col())
        assert isinstance(result, str)

    def test_valid_json(self):
        result = write_json(_col())
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_has_fragments_key(self):
        data = json.loads(write_json(_col()))
        assert "fragments" in data

    def test_fragments_count(self):
        data = json.loads(write_json(_col()))
        assert len(data["fragments"]) == 3

    def test_has_n_fragments(self):
        data = json.loads(write_json(_col()))
        assert data["n_fragments"] == 3

    def test_has_mean_score(self):
        data = json.loads(write_json(_col()))
        assert "mean_score" in data

    def test_has_meta_key(self):
        data = json.loads(write_json(_col()))
        assert "meta" in data

    def test_global_meta_serialized(self):
        col = _col_with_meta()
        data = json.loads(write_json(col))
        assert data["meta"]["title"] == "test"

    def test_compact_json_no_indent(self):
        cfg = WriterConfig(indent=None)
        result = write_json(_col(), cfg)
        assert "\n" not in result

    def test_sort_keys_false(self):
        cfg = WriterConfig(sort_keys=False)
        result = write_json(_col(), cfg)
        assert isinstance(json.loads(result), dict)

    def test_empty_collection(self):
        col = MetadataCollection()
        data = json.loads(write_json(col))
        assert data["n_fragments"] == 0
        assert data["fragments"] == []


# ─── TestReadJson ─────────────────────────────────────────────────────────────

class TestReadJson:
    def _roundtrip(self, col=None):
        if col is None:
            col = _col()
        return read_json(write_json(col))

    def test_roundtrip_n_fragments(self):
        col = self._roundtrip()
        assert col.n_fragments == 3

    def test_roundtrip_fragment_ids(self):
        col = self._roundtrip()
        assert set(col.fragment_ids) == {0, 1, 2}

    def test_roundtrip_positions(self):
        col = self._roundtrip()
        r0 = col.get_record(0)
        assert r0.position[0] == pytest.approx(0.0)

    def test_roundtrip_scores(self):
        col = self._roundtrip()
        r0 = col.get_record(0)
        assert r0.score == pytest.approx(0.8, abs=1e-4)

    def test_roundtrip_global_meta(self):
        col = self._roundtrip(_col_with_meta())
        assert col.global_meta["title"] == "test"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            read_json("not json {")

    def test_missing_fragment_id_raises(self):
        bad = json.dumps({"fragments": [{"x": 1.0, "y": 2.0}]})
        with pytest.raises(ValueError):
            read_json(bad)

    def test_empty_collection_roundtrip(self):
        col = MetadataCollection()
        restored = read_json(write_json(col))
        assert restored.n_fragments == 0

    def test_extra_fields_preserved(self):
        r = _rec(extra={"label": "A"})
        col = MetadataCollection(records=[r])
        restored = read_json(write_json(col))
        r0 = restored.get_record(0)
        assert r0.extra.get("label") == "A"

    def test_returns_metadata_collection(self):
        result = read_json(write_json(_col()))
        assert isinstance(result, MetadataCollection)


# ─── TestWriteCsv ─────────────────────────────────────────────────────────────

class TestWriteCsv:
    def test_returns_string(self):
        assert isinstance(write_csv(_col()), str)

    def test_has_header(self):
        result = write_csv(_col())
        first_line = result.strip().split("\n")[0]
        assert "fragment_id" in first_line

    def test_row_count(self):
        result = write_csv(_col())
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 4  # 1 header + 3 data rows

    def test_empty_collection(self):
        result = write_csv(MetadataCollection())
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 1  # header only

    def test_extra_columns(self):
        r = _rec(extra={"label": "X"})
        col = MetadataCollection(records=[r])
        result = write_csv(col, extra_columns=["label"])
        assert "label" in result.split("\n")[0]

    def test_score_column_present(self):
        result = write_csv(_col())
        assert "score" in result.split("\n")[0]

    def test_rotation_column_present(self):
        result = write_csv(_col())
        assert "rotation_deg" in result.split("\n")[0]


# ─── TestWriteSummary ─────────────────────────────────────────────────────────

class TestWriteSummary:
    def test_returns_string(self):
        assert isinstance(write_summary(_col()), str)

    def test_contains_fragment_count(self):
        result = write_summary(_col())
        assert "3" in result

    def test_contains_mean_score(self):
        result = write_summary(_col())
        assert "score" in result.lower() or "Score" in result

    def test_contains_fragment_ids(self):
        result = write_summary(_col())
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_global_meta_included(self):
        col = _col_with_meta()
        result = write_summary(col)
        assert "title" in result

    def test_empty_collection(self):
        result = write_summary(MetadataCollection())
        assert isinstance(result, str)

    def test_multiline(self):
        result = write_summary(_col())
        assert "\n" in result


# ─── TestFilterByScore ────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_all_pass(self):
        col = filter_by_score(_col(), min_score=0.0)
        assert col.n_fragments == 3

    def test_none_pass(self):
        col = filter_by_score(_col(), min_score=1.0)
        assert col.n_fragments == 0

    def test_partial(self):
        col = filter_by_score(_col(), min_score=0.5)
        # scores: 0.8, 0.6, 0.4 → 2 pass
        assert col.n_fragments == 2

    def test_exact_boundary_included(self):
        col = filter_by_score(_col(), min_score=0.6)
        # score=0.6 counts
        assert col.n_fragments == 2

    def test_returns_collection(self):
        assert isinstance(filter_by_score(_col(), 0.0), MetadataCollection)

    def test_does_not_mutate_original(self):
        original = _col()
        filter_by_score(original, 0.9)
        assert original.n_fragments == 3

    def test_neg_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_col(), -0.1)

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_col(), 1.1)

    def test_global_meta_preserved(self):
        col = _col_with_meta()
        filtered = filter_by_score(col, 0.0)
        assert filtered.global_meta["title"] == "test"


# ─── TestMergeCollections ─────────────────────────────────────────────────────

class TestMergeCollections:
    def test_basic_merge(self):
        col1 = MetadataCollection(records=[_rec(0), _rec(1)])
        col2 = MetadataCollection(records=[_rec(2), _rec(3)])
        merged = merge_collections(col1, col2)
        assert merged.n_fragments == 4

    def test_duplicate_id_later_wins(self):
        col1 = MetadataCollection(records=[_rec(0, score=0.3)])
        col2 = MetadataCollection(records=[_rec(0, score=0.9)])
        merged = merge_collections(col1, col2)
        assert merged.n_fragments == 1
        assert merged.get_record(0).score == pytest.approx(0.9)

    def test_single_collection(self):
        col = _col()
        merged = merge_collections(col)
        assert merged.n_fragments == 3

    def test_empty_collections(self):
        col1 = MetadataCollection()
        col2 = MetadataCollection()
        merged = merge_collections(col1, col2)
        assert merged.n_fragments == 0

    def test_global_meta_merged(self):
        col1 = MetadataCollection(global_meta={"a": 1})
        col2 = MetadataCollection(global_meta={"b": 2})
        merged = merge_collections(col1, col2)
        assert merged.global_meta["a"] == 1
        assert merged.global_meta["b"] == 2

    def test_global_meta_later_overwrites(self):
        col1 = MetadataCollection(global_meta={"key": "old"})
        col2 = MetadataCollection(global_meta={"key": "new"})
        merged = merge_collections(col1, col2)
        assert merged.global_meta["key"] == "new"

    def test_three_collections(self):
        col1 = MetadataCollection(records=[_rec(0)])
        col2 = MetadataCollection(records=[_rec(1)])
        col3 = MetadataCollection(records=[_rec(2)])
        merged = merge_collections(col1, col2, col3)
        assert merged.n_fragments == 3

    def test_returns_metadata_collection(self):
        result = merge_collections(_col())
        assert isinstance(result, MetadataCollection)
