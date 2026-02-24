"""Extra tests for puzzle_reconstruction/io/metadata_writer.py."""
from __future__ import annotations

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

def _rec(fid=0, x=10.0, y=20.0, rot=0.0, score=0.8):
    return MetadataRecord(fragment_id=fid, position=(x, y),
                          rotation_deg=rot, score=score)


def _col(records=None, meta=None):
    return MetadataCollection(
        records=records or [],
        global_meta=meta or {},
    )


# ─── WriterConfig ───────────────────────────────────────────────────────────

class TestWriterConfigExtra:
    def test_defaults(self):
        cfg = WriterConfig()
        assert cfg.indent == 2
        assert cfg.sort_keys is True
        assert cfg.csv_dialect == "excel"
        assert cfg.float_prec == 6

    def test_none_indent_ok(self):
        cfg = WriterConfig(indent=None)
        assert cfg.indent is None

    def test_negative_indent_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(indent=-1)

    def test_negative_float_prec_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(float_prec=-1)

    def test_empty_csv_dialect_raises(self):
        with pytest.raises(ValueError):
            WriterConfig(csv_dialect="")


# ─── MetadataRecord ─────────────────────────────────────────────────────────

class TestMetadataRecordExtra:
    def test_fields_stored(self):
        r = _rec(fid=5, x=1.0, y=2.0, rot=45.0, score=0.9)
        assert r.fragment_id == 5
        assert r.position == (1.0, 2.0)
        assert r.rotation_deg == pytest.approx(45.0)
        assert r.score == pytest.approx(0.9)

    def test_bad_position_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(1.0,))

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _rec(score=1.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            _rec(score=-0.1)

    def test_extra_default_empty(self):
        r = _rec()
        assert r.extra == {}

    def test_extra_stored(self):
        r = MetadataRecord(fragment_id=0, position=(0.0, 0.0),
                           score=0.5, extra={"key": "val"})
        assert r.extra["key"] == "val"

    def test_to_dict_keys(self):
        r = _rec()
        d = r.to_dict()
        assert "fragment_id" in d
        assert "x" in d and "y" in d
        assert "rotation_deg" in d
        assert "score" in d

    def test_to_dict_extra_included(self):
        r = MetadataRecord(fragment_id=0, position=(0.0, 0.0),
                           score=0.5, extra={"tag": "A"})
        d = r.to_dict()
        assert d["tag"] == "A"


# ─── MetadataCollection ─────────────────────────────────────────────────────

class TestMetadataCollectionExtra:
    def test_empty(self):
        col = _col()
        assert col.n_fragments == 0
        assert col.fragment_ids == []
        assert col.mean_score == pytest.approx(0.0)

    def test_n_fragments(self):
        col = _col([_rec(fid=0), _rec(fid=1)])
        assert col.n_fragments == 2

    def test_fragment_ids(self):
        col = _col([_rec(fid=3), _rec(fid=7)])
        assert col.fragment_ids == [3, 7]

    def test_mean_score(self):
        col = _col([_rec(score=0.4), _rec(score=0.8)])
        assert col.mean_score == pytest.approx(0.6)

    def test_get_record_found(self):
        col = _col([_rec(fid=5, score=0.9)])
        r = col.get_record(5)
        assert r is not None and r.score == pytest.approx(0.9)

    def test_get_record_not_found(self):
        col = _col([_rec(fid=5)])
        assert col.get_record(99) is None

    def test_add(self):
        col = _col()
        col.add(_rec(fid=1))
        assert col.n_fragments == 1

    def test_to_list(self):
        col = _col([_rec(fid=0), _rec(fid=1)])
        lst = col.to_list()
        assert len(lst) == 2 and isinstance(lst[0], dict)


# ─── write_json / read_json roundtrip ───────────────────────────────────────

class TestWriteReadJsonExtra:
    def test_roundtrip_empty(self):
        col = _col()
        s = write_json(col)
        col2 = read_json(s)
        assert col2.n_fragments == 0

    def test_roundtrip_with_records(self):
        col = _col([_rec(fid=0, score=0.7), _rec(fid=1, score=0.9)])
        s = write_json(col)
        col2 = read_json(s)
        assert col2.n_fragments == 2
        assert col2.get_record(0).score == pytest.approx(0.7)

    def test_json_is_valid(self):
        col = _col([_rec()])
        s = write_json(col)
        data = json.loads(s)
        assert "fragments" in data

    def test_global_meta_preserved(self):
        col = _col(meta={"project": "test"})
        s = write_json(col)
        col2 = read_json(s)
        assert col2.global_meta["project"] == "test"

    def test_read_invalid_json_raises(self):
        with pytest.raises(ValueError):
            read_json("{invalid")

    def test_read_missing_field_raises(self):
        s = json.dumps({"fragments": [{"x": 0, "y": 0}]})
        with pytest.raises(ValueError):
            read_json(s)

    def test_compact_json(self):
        cfg = WriterConfig(indent=None)
        col = _col([_rec()])
        s = write_json(col, cfg)
        assert "\n" not in s


# ─── write_csv ──────────────────────────────────────────────────────────────

class TestWriteCsvExtra:
    def test_header_present(self):
        col = _col([_rec()])
        s = write_csv(col)
        assert "fragment_id" in s.splitlines()[0]

    def test_row_count(self):
        col = _col([_rec(fid=0), _rec(fid=1)])
        s = write_csv(col)
        lines = [l for l in s.strip().splitlines() if l]
        assert len(lines) == 3  # header + 2 rows

    def test_extra_columns(self):
        r = MetadataRecord(fragment_id=0, position=(0.0, 0.0),
                           score=0.5, extra={"tag": "A"})
        col = _col([r])
        s = write_csv(col, extra_columns=["tag"])
        assert "tag" in s.splitlines()[0]

    def test_empty_collection(self):
        col = _col()
        s = write_csv(col)
        lines = [l for l in s.strip().splitlines() if l]
        assert len(lines) == 1  # header only


# ─── write_summary ──────────────────────────────────────────────────────────

class TestWriteSummaryExtra:
    def test_contains_header(self):
        col = _col()
        s = write_summary(col)
        assert "Metadata Summary" in s

    def test_fragment_count(self):
        col = _col([_rec(), _rec(fid=1)])
        s = write_summary(col)
        assert "Fragments : 2" in s

    def test_global_meta_shown(self):
        col = _col(meta={"project": "demo"})
        s = write_summary(col)
        assert "project" in s


# ─── filter_by_score ────────────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_basic(self):
        col = _col([_rec(score=0.3), _rec(score=0.7, fid=1)])
        filtered = filter_by_score(col, 0.5)
        assert filtered.n_fragments == 1

    def test_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_col(), 1.5)

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_col(), -0.1)

    def test_preserves_global_meta(self):
        col = _col(meta={"k": "v"})
        filtered = filter_by_score(col, 0.0)
        assert filtered.global_meta["k"] == "v"

    def test_zero_keeps_all(self):
        col = _col([_rec(score=0.0, fid=0), _rec(score=0.5, fid=1)])
        filtered = filter_by_score(col, 0.0)
        assert filtered.n_fragments == 2


# ─── merge_collections ──────────────────────────────────────────────────────

class TestMergeCollectionsExtra:
    def test_empty_merge(self):
        merged = merge_collections()
        assert merged.n_fragments == 0

    def test_single_collection(self):
        col = _col([_rec(fid=0)])
        merged = merge_collections(col)
        assert merged.n_fragments == 1

    def test_dedup_by_fragment_id(self):
        c1 = _col([_rec(fid=0, score=0.3)])
        c2 = _col([_rec(fid=0, score=0.9)])
        merged = merge_collections(c1, c2)
        assert merged.n_fragments == 1
        assert merged.get_record(0).score == pytest.approx(0.9)

    def test_global_meta_merged(self):
        c1 = _col(meta={"a": 1})
        c2 = _col(meta={"b": 2})
        merged = merge_collections(c1, c2)
        assert merged.global_meta["a"] == 1
        assert merged.global_meta["b"] == 2

    def test_three_collections(self):
        c1 = _col([_rec(fid=0)])
        c2 = _col([_rec(fid=1)])
        c3 = _col([_rec(fid=2)])
        merged = merge_collections(c1, c2, c3)
        assert merged.n_fragments == 3
