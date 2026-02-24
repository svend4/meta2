"""Extra tests for puzzle_reconstruction/utils/batch_processor.py"""
import pytest
from puzzle_reconstruction.utils.batch_processor import (
    BatchSummary,
    ProcessConfig,
    ProcessItem,
    filter_successful,
    make_processor,
    merge_batch_results,
    process_items,
    retry_failed_items,
    split_batch,
)


# ─── TestProcessConfigExtra ───────────────────────────────────────────────────

class TestProcessConfigExtra:
    def test_batch_size_one_valid(self):
        cfg = ProcessConfig(batch_size=1)
        assert cfg.batch_size == 1

    def test_max_retries_large(self):
        cfg = ProcessConfig(max_retries=100)
        assert cfg.max_retries == 100

    def test_verbose_true(self):
        cfg = ProcessConfig(verbose=True)
        assert cfg.verbose is True

    def test_stop_on_error_false_default(self):
        cfg = ProcessConfig()
        assert cfg.stop_on_error is False

    def test_all_params_custom(self):
        cfg = ProcessConfig(batch_size=16, max_retries=5, stop_on_error=True, verbose=True)
        assert cfg.batch_size == 16
        assert cfg.max_retries == 5
        assert cfg.stop_on_error is True
        assert cfg.verbose is True


# ─── TestProcessItemExtra ─────────────────────────────────────────────────────

class TestProcessItemExtra:
    def test_success_none_result(self):
        pi = ProcessItem(index=0, success=True, result=None)
        assert pi.success is True
        assert pi.result is None

    def test_success_list_result(self):
        pi = ProcessItem(index=1, success=True, result=[1, 2, 3])
        assert pi.result == [1, 2, 3]

    def test_failure_no_result(self):
        pi = ProcessItem(index=2, success=False, error="timeout")
        assert pi.result is None
        assert pi.error == "timeout"

    def test_index_zero_valid(self):
        pi = ProcessItem(index=0, success=True)
        assert pi.index == 0

    def test_large_index(self):
        pi = ProcessItem(index=999999, success=True, result=42)
        assert pi.index == 999999

    def test_retries_zero_default(self):
        pi = ProcessItem(index=0, success=True)
        assert pi.retries == 0

    def test_retries_large(self):
        pi = ProcessItem(index=0, success=False, retries=50)
        assert pi.retries == 50


# ─── TestBatchSummaryExtra ────────────────────────────────────────────────────

class TestBatchSummaryExtra:
    def _make(self, n, n_success):
        items = [
            ProcessItem(i, i < n_success, result=i if i < n_success else None,
                        error=None if i < n_success else "err")
            for i in range(n)
        ]
        return BatchSummary(
            total=n, n_success=n_success, n_failed=n - n_success,
            n_retried=0, items=items,
        )

    def test_all_success_ratio(self):
        s = self._make(5, 5)
        assert s.success_ratio == pytest.approx(1.0)

    def test_all_failed_ratio(self):
        s = self._make(4, 0)
        assert s.success_ratio == pytest.approx(0.0)

    def test_failed_indices_order(self):
        s = self._make(5, 3)
        assert s.failed_indices == [3, 4]

    def test_successful_results_values(self):
        s = self._make(5, 3)
        assert s.successful_results == [0, 1, 2]

    def test_total_zero(self):
        s = BatchSummary(total=0, n_success=0, n_failed=0, n_retried=0, items=[])
        assert s.success_ratio == 0.0
        assert s.failed_indices == []
        assert s.successful_results == []

    def test_n_retried_stored(self):
        items = [ProcessItem(0, True, result=1, retries=3)]
        s = BatchSummary(total=1, n_success=1, n_failed=0, n_retried=3, items=items)
        assert s.n_retried == 3


# ─── TestMakeProcessorExtra ───────────────────────────────────────────────────

class TestMakeProcessorExtra:
    def test_returns_process_config(self):
        cfg = make_processor()
        assert isinstance(cfg, ProcessConfig)

    def test_batch_size_1(self):
        cfg = make_processor(batch_size=1)
        assert cfg.batch_size == 1

    def test_large_batch_size(self):
        cfg = make_processor(batch_size=1024)
        assert cfg.batch_size == 1024

    def test_stop_on_error_passed(self):
        cfg = make_processor(stop_on_error=True)
        assert cfg.stop_on_error is True


# ─── TestProcessItemsExtra ────────────────────────────────────────────────────

class TestProcessItemsExtra:
    def test_single_item(self):
        summary = process_items([42], lambda x: x * 2)
        assert summary.total == 1
        assert summary.n_success == 1
        assert summary.items[0].result == 84

    def test_large_batch(self):
        items = list(range(100))
        summary = process_items(items, lambda x: x + 1)
        assert summary.total == 100
        assert summary.n_success == 100
        assert summary.n_failed == 0

    def test_result_is_string(self):
        summary = process_items(["a", "b", "c"], lambda x: x.upper())
        assert summary.successful_results == ["A", "B", "C"]

    def test_result_is_none_for_failed(self):
        def fn(x):
            raise ValueError("fail")
        summary = process_items([1], fn)
        assert summary.items[0].result is None

    def test_error_string_contains_type(self):
        def fn(x):
            raise TypeError("bad type")
        summary = process_items([1], fn)
        assert "TypeError" in summary.items[0].error

    def test_indices_continuous(self):
        summary = process_items([10, 20, 30], lambda x: x)
        assert [item.index for item in summary.items] == [0, 1, 2]

    def test_stop_on_error_second_item(self):
        calls = []

        def fn(x):
            calls.append(x)
            if x == 5:
                raise ValueError("stop here")
            return x

        cfg = ProcessConfig(stop_on_error=True)
        process_items([1, 5, 10, 20], fn, cfg)
        assert 10 not in calls
        assert 20 not in calls

    def test_retries_counted(self):
        state = {"n": 0}

        def fn(x):
            state["n"] += 1
            if state["n"] < 4:
                raise RuntimeError("not yet")
            return x * 2

        cfg = ProcessConfig(max_retries=5)
        summary = process_items([7], fn, cfg)
        assert summary.n_success == 1
        assert summary.items[0].retries >= 1

    def test_summary_retried_count(self):
        state = {"count": 0}

        def fn(x):
            state["count"] += 1
            if state["count"] == 1:
                raise RuntimeError("once")
            return x

        cfg = ProcessConfig(max_retries=2)
        summary = process_items([99], fn, cfg)
        assert summary.n_retried >= 1


# ─── TestFilterSuccessfulExtra ────────────────────────────────────────────────

class TestFilterSuccessfulExtra:
    def test_empty_summary(self):
        s = BatchSummary(total=0, n_success=0, n_failed=0, n_retried=0, items=[])
        assert filter_successful(s) == []

    def test_all_successful(self):
        summary = process_items([1, 2, 3], lambda x: x)
        ok = filter_successful(summary)
        assert len(ok) == 3
        assert all(item.success for item in ok)

    def test_returns_process_item_list(self):
        summary = process_items([1, 2], lambda x: x)
        ok = filter_successful(summary)
        assert all(isinstance(item, ProcessItem) for item in ok)

    def test_mixed_results(self):
        def fn(x):
            if x % 3 == 0:
                raise ValueError()
            return x
        summary = process_items(list(range(9)), fn)
        ok = filter_successful(summary)
        assert all(item.result % 3 != 0 for item in ok)


# ─── TestRetryFailedItemsExtra ────────────────────────────────────────────────

class TestRetryFailedItemsExtra:
    def test_all_already_succeeded(self):
        original = [1, 2, 3]
        summary = process_items(original, lambda x: x)
        retry = retry_failed_items(original, summary, lambda x: x * 10)
        assert retry.total == 0
        assert retry.n_success == 0

    def test_retry_with_different_function(self):
        def fn(x):
            raise RuntimeError("fail")
        original = [1, 2]
        summary = process_items(original, fn)
        retry = retry_failed_items(original, summary, lambda x: x * 100)
        assert retry.n_success == 2

    def test_retry_count(self):
        def fn(x):
            raise RuntimeError("fail")
        original = list(range(5))
        summary = process_items(original, fn)
        retry = retry_failed_items(original, summary, lambda x: x)
        assert retry.total == 5


# ─── TestSplitBatchExtra ──────────────────────────────────────────────────────

class TestSplitBatchExtra:
    def test_batch_size_equals_list_length(self):
        batches = split_batch([1, 2, 3], 3)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_single_element_list(self):
        batches = split_batch([99], 1)
        assert batches == [[99]]

    def test_large_list_small_batch(self):
        batches = split_batch(list(range(100)), 7)
        total = sum(len(b) for b in batches)
        assert total == 100

    def test_all_batches_except_last_full(self):
        batches = split_batch(list(range(10)), 3)
        for batch in batches[:-1]:
            assert len(batch) == 3
        assert len(batches[-1]) == 1

    def test_no_items_lost(self):
        items = list(range(37))
        batches = split_batch(items, 5)
        recovered = [x for batch in batches for x in batch]
        assert recovered == items

    def test_batch_size_1_all_singles(self):
        batches = split_batch([10, 20, 30, 40], 1)
        assert all(len(b) == 1 for b in batches)
        assert len(batches) == 4


# ─── TestMergeBatchResultsExtra ───────────────────────────────────────────────

class TestMergeBatchResultsExtra:
    def test_three_summaries_merged(self):
        s1 = process_items([1, 2], lambda x: x)
        s2 = process_items([3, 4], lambda x: x)
        s3 = process_items([5, 6], lambda x: x)
        merged = merge_batch_results([s1, s2, s3])
        assert merged.total == 6
        assert merged.n_success == 6

    def test_merged_indices_0_to_n(self):
        s1 = process_items([10, 20], lambda x: x)
        s2 = process_items([30, 40], lambda x: x)
        merged = merge_batch_results([s1, s2])
        indices = [item.index for item in merged.items]
        assert indices == list(range(4))

    def test_merged_failure_counts(self):
        def fn(x):
            if x > 10:
                raise ValueError()
            return x
        s1 = process_items([1, 2, 3], fn)
        s2 = process_items([11, 12], fn)
        merged = merge_batch_results([s1, s2])
        assert merged.n_failed == 2
        assert merged.n_success == 3

    def test_merged_successful_results(self):
        s1 = process_items([1], lambda x: x * 2)
        s2 = process_items([3], lambda x: x * 2)
        merged = merge_batch_results([s1, s2])
        assert set(merged.successful_results) == {2, 6}

    def test_merged_success_ratio(self):
        s1 = process_items([1, 2], lambda x: x)
        s2 = process_items([3, 4], lambda x: x)
        merged = merge_batch_results([s1, s2])
        assert merged.success_ratio == pytest.approx(1.0)
