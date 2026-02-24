"""Extra tests for puzzle_reconstruction/io/metadata_writer.py."""
from __future__ import annotations

import json

import pytest

from puzzle_reconstruction.io.metadata_writer import (
    MetadataCollection,
    MetadataRecord,
    WriterConfig,
    filter_by_score,
    merge_collections,
    read_json,
    write_csv,
    write_json,
    write_summary,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _record(fid: int = 0, x: float = 0.0, y: float = 0.0,
            rot: float = 0.0, score: float = 0.5) -> MetadataRecord:
    return MetadataRecord(
        fragment_id=fid, position=(x, y),
        rotation_deg=rot, score=score,
    )


def _collection(n: int = 3, base_score: float = 0.5) -> MetadataCollection:
    records = [_record(fid=i, x=float(i), score=base_score) for i in range(n)]
    return MetadataCollection(records=records, global_meta={"test": True})


# ─── WriterConfig (extra) ─────────────────────────────────────────────────────

class TestWriterConfigExtra:
    def test_default_indent(self):
        assert WriterConfig().indent == 2

    def test_default_sort_keys(self):
        assert WriterConfig().sort_keys is True

    def test_default_csv_dialect(self):
        assert WriterConfig().csv_dialect == "excel"

    def test_default_float_prec(self):
        assert WriterConfig().float_prec == 6

    def test_custom_indent(self):
        cfg = WriterConfig(indent=4)
        assert cfg.indent == 4

    def test_indent_none_ok(self):
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

    def test_custom_float_prec(self):
        cfg = WriterConfig(float_prec=3)
        assert cfg.float_prec == 3

    def test_sort_keys_false(self):
        cfg = WriterConfig(sort_keys=False)
        assert cfg.sort_keys is False


# ─── MetadataRecord (extra) ───────────────────────────────────────────────────

class TestMetadataRecordExtra:
    def test_fragment_id_stored(self):
        r = _record(fid=7)
        assert r.fragment_id == 7

    def test_position_stored(self):
        r = _record(x=10.0, y=20.0)
        assert r.position == (10.0, 20.0)

    def test_rotation_stored(self):
        r = _record(rot=45.0)
        assert r.rotation_deg == pytest.approx(45.0)

    def test_score_stored(self):
        r = _record(score=0.75)
        assert r.score == pytest.approx(0.75)

    def test_wrong_position_length_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(1, 2, 3))

    def test_score_negative_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(0, 0), score=-0.1)

    def test_score_gt_one_raises(self):
        with pytest.raises(ValueError):
            MetadataRecord(fragment_id=0, position=(0, 0), score=1.1)

    def test_score_zero_ok(self):
        r = _record(score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        r = _record(score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_extra_stored(self):
        r = MetadataRecord(fragment_id=0, position=(0, 0), extra={"k": 42})
        assert r.extra["k"] == 42

    def test_to_dict_keys(self):
        r = _record()
        d = r.to_dict()
        for key in ("fragment_id", "x", "y", "rotation_deg", "score"):
            assert key in d

    def test_to_dict_values(self):
        r = _record(fid=3, x=5.0, y=7.0, rot=90.0, score=0.8)
        d = r.to_dict(float_prec=2)
        assert d["fragment_id"] == 3
        assert d["x"] == pytest.approx(5.0)
        assert d["y"] == pytest.approx(7.0)

    def test_to_dict_extra_included(self):
        r = MetadataRecord(fragment_id=0, position=(0, 0), extra={"label": "A"})
        d = r.to_dict()
        assert d["label"] == "A"


# ─── MetadataCollection (extra) ───────────────────────────────────────────────

class TestMetadataCollectionExtra:
    def test_n_fragments(self):
        col = _collection(3)
        assert col.n_fragments == 3

    def test_empty_n_fragments_zero(self):
        col = MetadataCollection()
        assert col.n_fragments == 0

    def test_fragment_ids(self):
        col = _collection(3)
        assert col.fragment_ids == [0, 1, 2]

    def test_mean_score_empty(self):
        col = MetadataCollection()
        assert col.mean_score == pytest.approx(0.0)

    def test_mean_score_computed(self):
        col = _collection(4, base_score=0.5)
        assert col.mean_score == pytest.approx(0.5)

    def test_get_record_existing(self):
        col = _collection(3)
        rec = col.get_record(1)
        assert rec is not None
        assert rec.fragment_id == 1

    def test_get_record_missing_returns_none(self):
        col = _collection(3)
        assert col.get_record(99) is None

    def test_add_record(self):
        col = MetadataCollection()
        col.add(_record(fid=5))
        assert col.n_fragments == 1

    def test_to_list(self):
        col = _collection(2)
        lst = col.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 2

    def test_to_list_elements_are_dicts(self):
        col = _collection(2)
        for item in col.to_list():
            assert isinstance(item, dict)


# ─── write_json (extra) ───────────────────────────────────────────────────────

class TestWriteJsonExtra:
    def test_returns_string(self):
        assert isinstance(write_json(_collection()), str)

    def test_valid_json(self):
        s = write_json(_collection())
        data = json.loads(s)  # must not raise
        assert isinstance(data, dict)

    def test_contains_n_fragments(self):
        s = write_json(_collection(3))
        data = json.loads(s)
        assert data["n_fragments"] == 3

    def test_contains_fragments_list(self):
        s = write_json(_collection(2))
        data = json.loads(s)
        assert isinstance(data["fragments"], list)
        assert len(data["fragments"]) == 2

    def test_contains_mean_score(self):
        s = write_json(_collection(2, base_score=0.5))
        data = json.loads(s)
        assert data["mean_score"] == pytest.approx(0.5)

    def test_global_meta_preserved(self):
        col = _collection(1)
        col.global_meta = {"source": "test"}
        s = write_json(col)
        data = json.loads(s)
        assert data["meta"]["source"] == "test"

    def test_none_cfg_uses_defaults(self):
        s = write_json(_collection(), cfg=None)
        assert isinstance(json.loads(s), dict)

    def test_compact_json_with_none_indent(self):
        cfg = WriterConfig(indent=None)
        s = write_json(_collection(2), cfg=cfg)
        assert "\n" not in s


# ─── read_json (extra) ────────────────────────────────────────────────────────

class TestReadJsonExtra:
    def test_returns_collection(self):
        s = write_json(_collection(2))
        result = read_json(s)
        assert isinstance(result, MetadataCollection)

    def test_roundtrip_n_fragments(self):
        col = _collection(3)
        result = read_json(write_json(col))
        assert result.n_fragments == 3

    def test_roundtrip_fragment_ids(self):
        col = _collection(3)
        result = read_json(write_json(col))
        assert set(result.fragment_ids) == {0, 1, 2}

    def test_roundtrip_scores(self):
        col = _collection(2, base_score=0.75)
        result = read_json(write_json(col))
        for rec in result.records:
            assert rec.score == pytest.approx(0.75, abs=1e-4)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            read_json("{not valid json}")

    def test_missing_fragment_id_raises(self):
        bad = json.dumps({"fragments": [{"x": 0, "y": 0}]})
        with pytest.raises(ValueError):
            read_json(bad)

    def test_empty_fragments(self):
        s = json.dumps({"fragments": [], "meta": {}})
        result = read_json(s)
        assert result.n_fragments == 0


# ─── write_csv (extra) ────────────────────────────────────────────────────────

class TestWriteCsvExtra:
    def test_returns_string(self):
        assert isinstance(write_csv(_collection(2)), str)

    def test_has_header(self):
        s = write_csv(_collection(2))
        assert "fragment_id" in s

    def test_n_lines_correct(self):
        s = write_csv(_collection(3))
        # header + 3 records + possibly trailing newline
        lines = [l for l in s.strip().splitlines() if l.strip()]
        assert len(lines) == 4  # header + 3 records

    def test_none_cfg_uses_defaults(self):
        s = write_csv(_collection(2), cfg=None)
        assert isinstance(s, str)

    def test_extra_columns_included(self):
        rec = MetadataRecord(fragment_id=0, position=(0, 0),
                             score=0.5, extra={"tag": "abc"})
        col = MetadataCollection(records=[rec])
        s = write_csv(col, extra_columns=["tag"])
        assert "tag" in s

    def test_empty_collection(self):
        s = write_csv(MetadataCollection())
        assert "fragment_id" in s

    def test_values_present(self):
        s = write_csv(_collection(1))
        assert "0" in s  # fragment_id=0 at least


# ─── write_summary (extra) ────────────────────────────────────────────────────

class TestWriteSummaryExtra:
    def test_returns_string(self):
        assert isinstance(write_summary(_collection(2)), str)

    def test_contains_fragments_count(self):
        s = write_summary(_collection(3))
        assert "3" in s

    def test_contains_mean_score(self):
        s = write_summary(_collection(2, base_score=0.5))
        assert "0.5" in s

    def test_contains_header(self):
        s = write_summary(_collection(1))
        assert "Summary" in s

    def test_global_meta_included(self):
        col = _collection(1)
        col.global_meta = {"project": "demo"}
        s = write_summary(col)
        assert "project" in s

    def test_none_cfg_uses_defaults(self):
        s = write_summary(_collection(2), cfg=None)
        assert isinstance(s, str)

    def test_empty_collection(self):
        s = write_summary(MetadataCollection())
        assert isinstance(s, str)


# ─── filter_by_score (extra) ──────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_returns_collection(self):
        result = filter_by_score(_collection(3), 0.0)
        assert isinstance(result, MetadataCollection)

    def test_all_pass(self):
        col = _collection(3, base_score=0.5)
        result = filter_by_score(col, 0.0)
        assert result.n_fragments == 3

    def test_none_pass(self):
        col = _collection(3, base_score=0.3)
        result = filter_by_score(col, 0.9)
        assert result.n_fragments == 0

    def test_partial_filter(self):
        records = [_record(fid=0, score=0.3), _record(fid=1, score=0.8)]
        col = MetadataCollection(records=records)
        result = filter_by_score(col, 0.5)
        assert result.n_fragments == 1
        assert result.records[0].fragment_id == 1

    def test_invalid_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_collection(2), -0.1)

    def test_invalid_min_score_gt_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_collection(2), 1.1)

    def test_exact_score_included(self):
        col = MetadataCollection(records=[_record(fid=0, score=0.5)])
        result = filter_by_score(col, 0.5)
        assert result.n_fragments == 1

    def test_global_meta_preserved(self):
        col = _collection(2)
        col.global_meta = {"x": 1}
        result = filter_by_score(col, 0.0)
        assert result.global_meta["x"] == 1


# ─── merge_collections (extra) ────────────────────────────────────────────────

class TestMergeCollectionsExtra:
    def test_returns_collection(self):
        result = merge_collections(_collection(2))
        assert isinstance(result, MetadataCollection)

    def test_single_collection(self):
        col = _collection(3)
        result = merge_collections(col)
        assert result.n_fragments == 3

    def test_two_disjoint_collections(self):
        c1 = MetadataCollection(records=[_record(fid=0)])
        c2 = MetadataCollection(records=[_record(fid=1)])
        result = merge_collections(c1, c2)
        assert result.n_fragments == 2

    def test_duplicate_fid_overwritten(self):
        c1 = MetadataCollection(records=[_record(fid=0, score=0.3)])
        c2 = MetadataCollection(records=[_record(fid=0, score=0.9)])
        result = merge_collections(c1, c2)
        assert result.n_fragments == 1
        assert result.records[0].score == pytest.approx(0.9)

    def test_global_meta_merged(self):
        c1 = MetadataCollection(global_meta={"a": 1})
        c2 = MetadataCollection(global_meta={"b": 2})
        result = merge_collections(c1, c2)
        assert result.global_meta["a"] == 1
        assert result.global_meta["b"] == 2

    def test_later_global_meta_overrides(self):
        c1 = MetadataCollection(global_meta={"k": "v1"})
        c2 = MetadataCollection(global_meta={"k": "v2"})
        result = merge_collections(c1, c2)
        assert result.global_meta["k"] == "v2"

    def test_three_collections(self):
        c1 = MetadataCollection(records=[_record(fid=0)])
        c2 = MetadataCollection(records=[_record(fid=1)])
        c3 = MetadataCollection(records=[_record(fid=2)])
        result = merge_collections(c1, c2, c3)
        assert result.n_fragments == 3
