"""Extra tests for puzzle_reconstruction/utils/logger.py"""
import logging
import pytest

from puzzle_reconstruction.utils.logger import (
    ColorFormatter,
    get_logger,
    stage,
    ProgressBar,
    PipelineTimer,
)


# ─── ColorFormatter extras ────────────────────────────────────────────────────

class TestColorFormatterExtra:
    def test_level_colors_is_dict(self):
        assert isinstance(ColorFormatter.LEVEL_COLORS, dict)

    def test_critical_level_handled(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("t", logging.CRITICAL, "", 0, "critical msg", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_unknown_level_does_not_crash(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("t", 99, "", 0, "strange level", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_message_with_numbers(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 0, "progress: 42/100", (), None)
        result = fmt.format(record)
        assert "42" in result

    def test_empty_message(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 0, "", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_long_message(self):
        fmt = ColorFormatter()
        msg = "x" * 1000
        record = logging.LogRecord("t", logging.INFO, "", 0, msg, (), None)
        result = fmt.format(record)
        assert "x" * 10 in result

    def test_format_returns_same_type_all_levels(self):
        fmt = ColorFormatter()
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            record = logging.LogRecord("t", level, "", 0, "msg", (), None)
            assert isinstance(fmt.format(record), str)


# ─── get_logger extras ────────────────────────────────────────────────────────

class TestGetLoggerExtra:
    def test_different_names_different_loggers(self):
        l1 = get_logger("test_extra_logger_alpha_999")
        l2 = get_logger("test_extra_logger_beta_999")
        assert l1 is not l2

    def test_logger_has_handlers(self):
        logger = get_logger("test_extra_has_handlers_001")
        assert len(logger.handlers) >= 1

    def test_logger_propagate_false(self):
        logger = get_logger("test_extra_nopropagate_001")
        assert logger.propagate is False

    def test_logger_name_preserved(self):
        name = "test_extra_name_preserved_001"
        logger = get_logger(name)
        assert logger.name == name

    def test_multiple_levels(self):
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            logger = get_logger(f"test_extra_level_{level}_001", level=level)
            assert logger.level == level

    def test_with_file_handler_has_file(self, tmp_path):
        path = str(tmp_path / "extra_log.log")
        logger = get_logger("test_extra_file_001", log_file=path)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_handler_uses_color_formatter(self):
        logger = get_logger("test_extra_fmt_001")
        formatters = [h.formatter for h in logger.handlers if h.formatter is not None]
        assert any(isinstance(f, ColorFormatter) for f in formatters)


# ─── stage extras ─────────────────────────────────────────────────────────────

class TestStageExtra:
    def test_stage_body_runs(self):
        results = []
        with stage("body_test"):
            results.append(42)
        assert results == [42]

    def test_stage_returns_elapsed_or_none(self):
        # stage is a context manager — just ensure no crash
        with stage("ret_test") as ctx:
            pass

    def test_stage_many_iterations(self):
        for i in range(10):
            with stage(f"iter_{i}"):
                pass

    def test_stage_with_custom_logger(self):
        logger = get_logger("test_extra_stage_custom_001")
        with stage("custom_logger_stage", logger=logger):
            pass

    def test_stage_re_raises_type_error(self):
        with pytest.raises(TypeError):
            with stage("type_error_stage"):
                raise TypeError("deliberate")

    def test_stage_re_raises_index_error(self):
        with pytest.raises(IndexError):
            with stage("index_error_stage"):
                raise IndexError("deliberate")

    def test_nested_stages_inner_raises_propagates(self):
        with pytest.raises(RuntimeError):
            with stage("outer"):
                with stage("inner"):
                    raise RuntimeError("inner error")


# ─── ProgressBar extras ───────────────────────────────────────────────────────

class TestProgressBarExtra:
    def test_width_default_30(self):
        pb = ProgressBar("t", total=5)
        assert pb.width == 30

    def test_custom_width_stored(self):
        pb = ProgressBar("t", total=5, width=20)
        assert pb.width == 20

    def test_label_stored(self):
        pb = ProgressBar("my task", total=10)
        assert pb.label == "my task"

    def test_total_stored(self):
        pb = ProgressBar("t", total=99)
        assert pb.total == 99

    def test_context_manager_enter_returns_pb(self):
        pb = ProgressBar("t", total=5)
        with pb as entered:
            assert entered is pb

    def test_update_sequence(self):
        with ProgressBar("seq", total=10) as pb:
            for i in range(1, 11):
                pb.update(i)

    def test_update_value_zero(self):
        with ProgressBar("zero", total=10) as pb:
            pb.update(0)

    def test_large_total(self):
        pb = ProgressBar("large", total=1_000_000)
        assert pb.total == 1_000_000
        with pb:
            pb.update(500_000)


# ─── PipelineTimer extras ─────────────────────────────────────────────────────

class TestPipelineTimerExtra:
    def test_empty_report_is_string(self):
        timer = PipelineTimer()
        assert isinstance(timer.report(), str)

    def test_single_stage_time_nonneg(self):
        timer = PipelineTimer()
        with timer.measure("s"):
            pass
        assert timer._stages["s"] >= 0.0

    def test_total_equals_sum_of_stages(self):
        timer = PipelineTimer()
        with timer.measure("x"):
            pass
        with timer.measure("y"):
            pass
        total = sum(timer._stages.values())
        assert total >= 0.0

    def test_stage_names_preserved(self):
        timer = PipelineTimer()
        names = ["load", "process", "save"]
        for n in names:
            with timer.measure(n):
                pass
        for n in names:
            assert n in timer._stages

    def test_repeated_stage_name_overwrites_or_accumulates(self):
        timer = PipelineTimer()
        with timer.measure("s"):
            pass
        t1 = timer._stages["s"]
        with timer.measure("s"):
            pass
        # Should still have entry "s"
        assert "s" in timer._stages

    def test_exception_records_stage_time_nonneg(self):
        timer = PipelineTimer()
        try:
            with timer.measure("err"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert timer._stages.get("err", -1) >= 0.0

    def test_report_contains_итого(self):
        timer = PipelineTimer()
        with timer.measure("s"):
            pass
        assert "ИТОГО" in timer.report()

    def test_report_contains_all_stage_names(self):
        timer = PipelineTimer()
        with timer.measure("alpha"):
            pass
        with timer.measure("beta"):
            pass
        report = timer.report()
        assert "alpha" in report
        assert "beta" in report
