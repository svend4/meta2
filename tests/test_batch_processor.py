"""Тесты для puzzle_reconstruction.utils.batch_processor."""
import pytest
from puzzle_reconstruction.utils.batch_processor import (
    ProcessConfig,
    ProcessItem,
    BatchSummary,
    make_processor,
    process_items,
    filter_successful,
    retry_failed_items,
    split_batch,
    merge_batch_results,
)


# ─── TestProcessConfig ────────────────────────────────────────────────────────

class TestProcessConfig:
    def test_defaults(self):
        cfg = ProcessConfig()
        assert cfg.batch_size == 32
        assert cfg.max_retries == 0
        assert cfg.stop_on_error is False
        assert cfg.verbose is False

    def test_valid_custom(self):
        cfg = ProcessConfig(batch_size=8, max_retries=3, stop_on_error=True)
        assert cfg.batch_size == 8
        assert cfg.max_retries == 3
        assert cfg.stop_on_error is True

    def test_invalid_batch_size_zero(self):
        with pytest.raises(ValueError):
            ProcessConfig(batch_size=0)

    def test_invalid_batch_size_neg(self):
        with pytest.raises(ValueError):
            ProcessConfig(batch_size=-1)

    def test_invalid_max_retries_neg(self):
        with pytest.raises(ValueError):
            ProcessConfig(max_retries=-1)


# ─── TestProcessItem ──────────────────────────────────────────────────────────

class TestProcessItem:
    def test_basic_success(self):
        pi = ProcessItem(index=0, success=True, result=42)
        assert pi.index == 0
        assert pi.success is True
        assert pi.result == 42
        assert pi.error is None
        assert pi.retries == 0

    def test_failure(self):
        pi = ProcessItem(index=1, success=False, error="oops")
        assert pi.success is False
        assert pi.error == "oops"

    def test_with_retries(self):
        pi = ProcessItem(index=2, success=True, retries=2)
        assert pi.retries == 2

    def test_invalid_index_neg(self):
        with pytest.raises(ValueError):
            ProcessItem(index=-1, success=True)

    def test_invalid_retries_neg(self):
        with pytest.raises(ValueError):
            ProcessItem(index=0, success=True, retries=-1)


# ─── TestBatchSummary ─────────────────────────────────────────────────────────

class TestBatchSummary:
    def _make_summary(self, n_total=4, n_success=3, n_failed=1, n_retried=0):
        items = [
            ProcessItem(i, i < n_success, result=i if i < n_success else None,
                        error=None if i < n_success else "err")
            for i in range(n_total)
        ]
        return BatchSummary(
            total=n_total,
            n_success=n_success,
            n_failed=n_failed,
            n_retried=n_retried,
            items=items,
        )

    def test_success_ratio(self):
        s = self._make_summary(4, 3, 1)
        assert abs(s.success_ratio - 0.75) < 1e-9

    def test_success_ratio_zero_total(self):
        s = BatchSummary(total=0, n_success=0, n_failed=0, n_retried=0, items=[])
        assert s.success_ratio == 0.0

    def test_failed_indices(self):
        s = self._make_summary(4, 3, 1)
        assert s.failed_indices == [3]

    def test_successful_results(self):
        s = self._make_summary(4, 3, 1)
        assert s.successful_results == [0, 1, 2]

    def test_successful_results_empty(self):
        s = BatchSummary(total=0, n_success=0, n_failed=0, n_retried=0, items=[])
        assert s.successful_results == []

    def test_invalid_total_neg(self):
        with pytest.raises(ValueError):
            BatchSummary(total=-1, n_success=0, n_failed=0, n_retried=0, items=[])

    def test_invalid_n_success_neg(self):
        with pytest.raises(ValueError):
            BatchSummary(total=0, n_success=-1, n_failed=0, n_retried=0, items=[])

    def test_invalid_n_failed_neg(self):
        with pytest.raises(ValueError):
            BatchSummary(total=0, n_success=0, n_failed=-1, n_retried=0, items=[])

    def test_invalid_n_retried_neg(self):
        with pytest.raises(ValueError):
            BatchSummary(total=0, n_success=0, n_failed=0, n_retried=-1, items=[])


# ─── TestMakeProcessor ────────────────────────────────────────────────────────

class TestMakeProcessor:
    def test_default(self):
        cfg = make_processor()
        assert isinstance(cfg, ProcessConfig)
        assert cfg.batch_size == 32

    def test_custom(self):
        cfg = make_processor(batch_size=4, max_retries=2, stop_on_error=True)
        assert cfg.batch_size == 4
        assert cfg.max_retries == 2
        assert cfg.stop_on_error is True


# ─── TestProcessItems ─────────────────────────────────────────────────────────

class TestProcessItems:
    def test_all_success(self):
        items = [1, 2, 3]
        summary = process_items(items, lambda x: x * 2)
        assert summary.total == 3
        assert summary.n_success == 3
        assert summary.n_failed == 0
        assert summary.successful_results == [2, 4, 6]

    def test_all_fail(self):
        def failing(x):
            raise ValueError("bad")
        summary = process_items([1, 2, 3], failing)
        assert summary.n_failed == 3
        assert summary.n_success == 0

    def test_partial_fail(self):
        def fn(x):
            if x == 2:
                raise RuntimeError("two is bad")
            return x
        summary = process_items([1, 2, 3], fn)
        assert summary.n_success == 2
        assert summary.n_failed == 1
        assert summary.failed_indices == [1]

    def test_error_message_captured(self):
        def fn(x):
            raise ValueError("test error")
        summary = process_items([1], fn)
        assert "ValueError" in summary.items[0].error

    def test_empty_items(self):
        summary = process_items([], lambda x: x)
        assert summary.total == 0
        assert summary.n_success == 0

    def test_stop_on_error(self):
        cfg = ProcessConfig(stop_on_error=True)
        calls = []
        def fn(x):
            calls.append(x)
            if x == 1:
                raise ValueError()
            return x
        process_items([0, 1, 2, 3], fn, cfg)
        # Should stop after failing on item 1
        assert 3 not in calls

    def test_with_retries_success_eventually(self):
        state = {"count": 0}
        def fn(x):
            state["count"] += 1
            if state["count"] < 3:
                raise RuntimeError("retry me")
            return x
        cfg = ProcessConfig(max_retries=3)
        summary = process_items([42], fn, cfg)
        assert summary.n_success == 1
        assert summary.items[0].retries >= 1

    def test_with_retries_exhausted(self):
        def fn(x):
            raise RuntimeError("always fails")
        cfg = ProcessConfig(max_retries=2)
        summary = process_items([1], fn, cfg)
        assert summary.n_failed == 1
        assert summary.items[0].retries == 2

    def test_default_config(self):
        summary = process_items([10], lambda x: x + 1)
        assert summary.items[0].result == 11

    def test_result_order_preserved(self):
        items = list(range(10))
        summary = process_items(items, lambda x: x)
        for i, item in enumerate(summary.items):
            assert item.index == i
            assert item.result == i


# ─── TestFilterSuccessful ─────────────────────────────────────────────────────

class TestFilterSuccessful:
    def test_basic(self):
        summary = process_items([1, 2, 3], lambda x: x * 3)
        ok = filter_successful(summary)
        assert len(ok) == 3
        assert all(item.success for item in ok)

    def test_with_failures(self):
        def fn(x):
            if x % 2 == 0:
                raise ValueError()
            return x
        summary = process_items([1, 2, 3, 4], fn)
        ok = filter_successful(summary)
        assert len(ok) == 2

    def test_all_failed(self):
        summary = process_items([1], lambda x: (_ for _ in ()).throw(ValueError()))
        ok = filter_successful(summary)
        assert ok == []


# ─── TestRetryFailedItems ────────────────────────────────────────────────────

class TestRetryFailedItems:
    def test_retry_succeeds(self):
        state = {"fixed": False}
        def fn(x):
            if not state["fixed"]:
                raise ValueError("not ready")
            return x * 10
        original_items = [5, 6]
        first_summary = process_items(original_items, fn)
        assert first_summary.n_failed == 2
        state["fixed"] = True
        retry_summary = retry_failed_items(original_items, first_summary, fn)
        assert retry_summary.n_success == 2

    def test_no_failed_empty_retry(self):
        original = [1, 2, 3]
        summary = process_items(original, lambda x: x)
        retry = retry_failed_items(original, summary, lambda x: x)
        assert retry.total == 0
        assert retry.n_success == 0

    def test_partial_retry(self):
        def fn(x):
            if x == 2:
                raise RuntimeError()
            return x
        original = [1, 2, 3]
        summary = process_items(original, fn)
        assert summary.n_failed == 1
        retry = retry_failed_items(original, summary, lambda x: x * 2)
        assert retry.n_success == 1


# ─── TestSplitBatch ───────────────────────────────────────────────────────────

class TestSplitBatch:
    def test_even_split(self):
        batches = split_batch(list(range(6)), 2)
        assert len(batches) == 3
        assert batches[0] == [0, 1]

    def test_last_batch_smaller(self):
        batches = split_batch(list(range(5)), 2)
        assert len(batches) == 3
        assert batches[-1] == [4]

    def test_batch_size_larger_than_list(self):
        batches = split_batch([1, 2], 10)
        assert len(batches) == 1
        assert batches[0] == [1, 2]

    def test_empty_list(self):
        assert split_batch([], 4) == []

    def test_invalid_batch_size_zero(self):
        with pytest.raises(ValueError):
            split_batch([1, 2], 0)

    def test_invalid_batch_size_neg(self):
        with pytest.raises(ValueError):
            split_batch([1, 2], -1)

    def test_single_element_batches(self):
        batches = split_batch([10, 20, 30], 1)
        assert batches == [[10], [20], [30]]


# ─── TestMergeBatchResults ────────────────────────────────────────────────────

class TestMergeBatchResults:
    def test_basic_merge(self):
        s1 = process_items([1, 2], lambda x: x)
        s2 = process_items([3, 4], lambda x: x)
        merged = merge_batch_results([s1, s2])
        assert merged.total == 4
        assert merged.n_success == 4

    def test_index_offset(self):
        s1 = process_items([10], lambda x: x)
        s2 = process_items([20], lambda x: x)
        merged = merge_batch_results([s1, s2])
        indices = [item.index for item in merged.items]
        assert indices == [0, 1]

    def test_failed_counts(self):
        def fn(x):
            if x > 5:
                raise ValueError()
            return x
        s1 = process_items([1, 2], fn)
        s2 = process_items([6, 7], fn)
        merged = merge_batch_results([s1, s2])
        assert merged.n_failed == 2
        assert merged.n_success == 2

    def test_empty_summaries_raises(self):
        with pytest.raises(ValueError):
            merge_batch_results([])

    def test_single_summary(self):
        s = process_items([1, 2, 3], lambda x: x)
        merged = merge_batch_results([s])
        assert merged.total == 3

    def test_retried_counts(self):
        state = {"calls": 0}
        def fn(x):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError()
            return x
        cfg = ProcessConfig(max_retries=1)
        s1 = process_items([42], fn, cfg)
        s2 = process_items([1], lambda x: x)
        merged = merge_batch_results([s1, s2])
        assert merged.n_retried >= 1
