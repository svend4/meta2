"""Extra tests for puzzle_reconstruction.utils.batch_processor."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ok(x):
    return x * 2


def _fail(x):
    raise RuntimeError("intentional error")


def _flaky_fn():
    calls = [0]
    def fn(x):
        calls[0] += 1
        if calls[0] <= 1:
            raise ValueError("flaky")
        return x
    return fn


def _run(items, fn=_ok, **cfg_kwargs):
    cfg = ProcessConfig(**cfg_kwargs) if cfg_kwargs else None
    return process_items(items, fn, cfg)


# ─── TestProcessConfigExtra ──────────────────────────────────────────────────

class TestProcessConfigExtra:
    def test_defaults(self):
        cfg = ProcessConfig()
        assert cfg.batch_size == 32
        assert cfg.max_retries == 0
        assert cfg.stop_on_error is False

    def test_custom_batch_size(self):
        cfg = ProcessConfig(batch_size=64)
        assert cfg.batch_size == 64

    def test_custom_max_retries(self):
        cfg = ProcessConfig(max_retries=5)
        assert cfg.max_retries == 5

    def test_stop_on_error_true(self):
        cfg = ProcessConfig(stop_on_error=True)
        assert cfg.stop_on_error is True

    def test_verbose_true(self):
        cfg = ProcessConfig(verbose=True)
        assert cfg.verbose is True

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError):
            ProcessConfig(batch_size=0)

    def test_batch_size_neg_raises(self):
        with pytest.raises(ValueError):
            ProcessConfig(batch_size=-5)

    def test_max_retries_neg_raises(self):
        with pytest.raises(ValueError):
            ProcessConfig(max_retries=-1)

    def test_batch_size_one(self):
        cfg = ProcessConfig(batch_size=1)
        assert cfg.batch_size == 1


# ─── TestProcessItemExtra ────────────────────────────────────────────────────

class TestProcessItemExtra:
    def test_success_true_result(self):
        pi = ProcessItem(index=0, success=True, result="hello")
        assert pi.result == "hello"
        assert pi.error is None

    def test_success_false_error(self):
        pi = ProcessItem(index=1, success=False, error="boom")
        assert pi.error == "boom"
        assert pi.result is None

    def test_index_negative_raises(self):
        with pytest.raises(ValueError):
            ProcessItem(index=-1, success=True)

    def test_retries_negative_raises(self):
        with pytest.raises(ValueError):
            ProcessItem(index=0, success=True, retries=-1)

    def test_retries_default_zero(self):
        pi = ProcessItem(index=0, success=True)
        assert pi.retries == 0

    def test_retries_custom(self):
        pi = ProcessItem(index=0, success=True, retries=3)
        assert pi.retries == 3

    def test_index_stored(self):
        pi = ProcessItem(index=42, success=True)
        assert pi.index == 42


# ─── TestBatchSummaryExtra ───────────────────────────────────────────────────

class TestBatchSummaryExtra:
    def _make(self, items, n_success, n_failed, n_retried=0):
        return BatchSummary(
            total=len(items), n_success=n_success, n_failed=n_failed,
            n_retried=n_retried, items=items,
        )

    def test_success_ratio_all_success(self):
        items = [ProcessItem(i, True) for i in range(10)]
        s = self._make(items, 10, 0)
        assert s.success_ratio == pytest.approx(1.0)

    def test_success_ratio_all_fail(self):
        items = [ProcessItem(i, False, error="e") for i in range(5)]
        s = self._make(items, 0, 5)
        assert s.success_ratio == pytest.approx(0.0)

    def test_success_ratio_empty(self):
        s = self._make([], 0, 0)
        assert s.success_ratio == pytest.approx(0.0)

    def test_failed_indices(self):
        items = [
            ProcessItem(0, True),
            ProcessItem(1, False, error="e"),
            ProcessItem(2, False, error="e"),
        ]
        s = self._make(items, 1, 2)
        assert s.failed_indices == [1, 2]

    def test_successful_results(self):
        items = [
            ProcessItem(0, True, result=10),
            ProcessItem(1, False, error="e"),
            ProcessItem(2, True, result=30),
        ]
        s = self._make(items, 2, 1)
        assert s.successful_results == [10, 30]

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            BatchSummary(total=-1, n_success=0, n_failed=0,
                         n_retried=0, items=[])

    def test_n_retried_stored(self):
        items = [ProcessItem(0, True, retries=2)]
        s = self._make(items, 1, 0, n_retried=1)
        assert s.n_retried == 1


# ─── TestMakeProcessorExtra ──────────────────────────────────────────────────

class TestMakeProcessorExtra:
    def test_returns_process_config(self):
        assert isinstance(make_processor(), ProcessConfig)

    def test_custom_batch_size(self):
        assert make_processor(batch_size=8).batch_size == 8

    def test_custom_max_retries(self):
        assert make_processor(max_retries=3).max_retries == 3

    def test_stop_on_error_flag(self):
        assert make_processor(stop_on_error=True).stop_on_error is True


# ─── TestProcessItemsExtra ───────────────────────────────────────────────────

class TestProcessItemsExtra:
    def test_empty_input(self):
        s = _run([])
        assert s.total == 0

    def test_all_success(self):
        s = _run([1, 2, 3, 4])
        assert s.n_success == 4
        assert s.n_failed == 0

    def test_results_correct(self):
        s = _run([10, 20, 30])
        assert s.successful_results == [20, 40, 60]

    def test_all_fail(self):
        s = _run([1, 2], fn=_fail)
        assert s.n_failed == 2

    def test_indices_sequential(self):
        s = _run([10, 20, 30])
        indices = [item.index for item in s.items]
        assert indices == [0, 1, 2]

    def test_stop_on_error(self):
        s = process_items([1, 2, 3, 4, 5], _fail,
                          ProcessConfig(stop_on_error=True))
        assert len(s.items) < 5

    def test_no_stop_processes_all(self):
        s = process_items([1, 2, 3], _fail,
                          ProcessConfig(stop_on_error=False))
        assert len(s.items) == 3

    def test_retry_flaky(self):
        fn = _flaky_fn()
        s = process_items([1], fn, ProcessConfig(max_retries=2))
        assert s.n_success == 1
        assert s.items[0].retries >= 1

    def test_none_cfg_uses_default(self):
        s = process_items([1, 2], _ok, None)
        assert s.n_success == 2

    def test_total_matches_input(self):
        s = _run(list(range(10)))
        assert s.total == 10

    def test_failed_items_have_error(self):
        s = _run([1], fn=_fail)
        assert s.items[0].error is not None


# ─── TestFilterSuccessfulExtra ───────────────────────────────────────────────

class TestFilterSuccessfulExtra:
    def test_all_success(self):
        s = _run([1, 2, 3])
        assert len(filter_successful(s)) == 3

    def test_all_fail(self):
        s = process_items([1, 2], _fail)
        assert filter_successful(s) == []

    def test_mixed(self):
        items = [
            ProcessItem(0, True, result="a"),
            ProcessItem(1, False, error="e"),
            ProcessItem(2, True, result="c"),
        ]
        summary = BatchSummary(total=3, n_success=2, n_failed=1,
                               n_retried=0, items=items)
        ok = filter_successful(summary)
        assert len(ok) == 2

    def test_returns_process_items(self):
        s = _run([1, 2])
        for item in filter_successful(s):
            assert isinstance(item, ProcessItem)


# ─── TestRetryFailedItemsExtra ───────────────────────────────────────────────

class TestRetryFailedItemsExtra:
    def test_retry_all_failed(self):
        original = [1, 2, 3]
        s = process_items(original, _fail)
        s2 = retry_failed_items(original, s, _ok)
        assert s2.n_success == 3

    def test_no_failed_empty(self):
        original = [1, 2]
        s = _run(original)
        s2 = retry_failed_items(original, s, _ok)
        assert s2.total == 0

    def test_partial_retry(self):
        items = [
            ProcessItem(0, True, result=10),
            ProcessItem(1, False, error="e"),
        ]
        summary = BatchSummary(total=2, n_success=1, n_failed=1,
                               n_retried=0, items=items)
        s2 = retry_failed_items([5, 10], summary, _ok)
        assert s2.n_success == 1


# ─── TestSplitBatchExtra ─────────────────────────────────────────────────────

class TestSplitBatchExtra:
    def test_exact_split(self):
        assert split_batch([1, 2, 3, 4], batch_size=2) == [[1, 2], [3, 4]]

    def test_remainder(self):
        batches = split_batch([1, 2, 3, 4, 5], batch_size=2)
        assert len(batches) == 3
        assert batches[-1] == [5]

    def test_empty(self):
        assert split_batch([], batch_size=4) == []

    def test_batch_exceeds_list(self):
        assert split_batch([1, 2], batch_size=10) == [[1, 2]]

    def test_batch_size_one(self):
        assert split_batch([10, 20, 30], batch_size=1) == [[10], [20], [30]]

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError):
            split_batch([1], batch_size=0)

    def test_batch_size_equals_list(self):
        assert split_batch([1, 2, 3], batch_size=3) == [[1, 2, 3]]

    def test_single_element(self):
        assert split_batch([42], batch_size=5) == [[42]]


# ─── TestMergeBatchResultsExtra ──────────────────────────────────────────────

class TestMergeBatchResultsExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_batch_results([])

    def test_single_summary(self):
        s = _run([1, 2, 3])
        merged = merge_batch_results([s])
        assert merged.total == 3

    def test_two_summaries(self):
        s1 = _run([1, 2])
        s2 = _run([3, 4, 5])
        merged = merge_batch_results([s1, s2])
        assert merged.total == 5
        assert merged.n_success == 5

    def test_indices_offset(self):
        s1 = _run([10])
        s2 = _run([20, 30])
        merged = merge_batch_results([s1, s2])
        indices = [item.index for item in merged.items]
        assert indices == [0, 1, 2]

    def test_failures_summed(self):
        s1 = _run([1], fn=_fail)
        s2 = _run([2], fn=_fail)
        merged = merge_batch_results([s1, s2])
        assert merged.n_failed == 2

    def test_three_summaries(self):
        s1 = _run([1])
        s2 = _run([2])
        s3 = _run([3])
        merged = merge_batch_results([s1, s2, s3])
        assert merged.total == 3
        assert merged.n_success == 3
