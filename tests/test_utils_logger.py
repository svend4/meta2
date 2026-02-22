"""Tests for puzzle_reconstruction/utils/logger.py"""
import logging
import time
import pytest

from puzzle_reconstruction.utils.logger import (
    ColorFormatter,
    get_logger,
    stage,
    ProgressBar,
    PipelineTimer,
)


# ─── ColorFormatter ───────────────────────────────────────────────────────────

class TestColorFormatter:
    def test_format_returns_string(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_format_contains_message(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        result = fmt.format(record)
        assert "hello world" in result

    def test_format_info_level(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert "INFO"[:4] in result

    def test_format_debug_level(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "debug msg", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_format_warning_level(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.WARNING, "", 0, "warn msg", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_format_error_level(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("test", logging.ERROR, "", 0, "error msg", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_level_colors_covers_standard_levels(self):
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            assert level in ColorFormatter.LEVEL_COLORS

    def test_format_non_empty(self):
        fmt = ColorFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 0, "x", (), None)
        assert len(fmt.format(record)) > 0

    def test_format_truncates_level_to_four(self):
        """Level name is truncated to 4 chars (e.g., 'WARN')."""
        fmt = ColorFormatter()
        record = logging.LogRecord("t", logging.WARNING, "", 0, "w", (), None)
        result = fmt.format(record)
        # Should contain 'WARN' (4-char truncation of WARNING)
        assert "WARN" in result


# ─── get_logger ───────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_iter115_get_01")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("test_iter115_name_abc")
        assert logger.name == "test_iter115_name_abc"

    def test_logger_idempotent(self):
        """Calling get_logger twice with same name returns same object."""
        logger1 = get_logger("test_iter115_idempotent_xyz")
        n_handlers = len(logger1.handlers)
        logger2 = get_logger("test_iter115_idempotent_xyz")
        assert logger2 is logger1
        assert len(logger2.handlers) == n_handlers

    def test_logger_level_info(self):
        logger = get_logger("test_iter115_level_info_01", level=logging.INFO)
        assert logger.level == logging.INFO

    def test_logger_level_debug(self):
        logger = get_logger("test_iter115_level_debug_01", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_level_warning(self):
        logger = get_logger("test_iter115_level_warn_01", level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_logger_with_file(self, tmp_path):
        logfile = str(tmp_path / "test_iter115.log")
        logger = get_logger("test_iter115_with_file_01", log_file=logfile)
        # Should have at least 2 handlers: console + file
        assert len(logger.handlers) >= 2

    def test_logger_no_propagate(self):
        logger = get_logger("test_iter115_no_prop_01")
        assert logger.propagate is False

    def test_logger_has_console_handler(self):
        logger = get_logger("test_iter115_console_01")
        stream_handlers = [h for h in logger.handlers
                           if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) >= 1


# ─── stage ────────────────────────────────────────────────────────────────────

class TestStage:
    def test_stage_yields(self):
        logger = get_logger("test_iter115_stage_yields_01")
        ran = []
        with stage("TestStage", logger=logger):
            ran.append(True)
        assert ran == [True]

    def test_stage_reraises_exception(self):
        logger = get_logger("test_iter115_stage_reraise_01")
        with pytest.raises(ValueError, match="intentional"):
            with stage("FailStage", logger=logger):
                raise ValueError("intentional")

    def test_stage_without_logger_uses_default(self):
        ran = []
        with stage("DefaultLoggerStage"):
            ran.append(True)
        assert ran == [True]

    def test_stage_context_multiple_times(self):
        logger = get_logger("test_iter115_stage_multi_01")
        for i in range(3):
            with stage(f"stage_{i}", logger=logger):
                pass

    def test_stage_exception_type_preserved(self):
        logger = get_logger("test_iter115_stage_type_01")
        with pytest.raises(RuntimeError):
            with stage("RuntimeStage", logger=logger):
                raise RuntimeError("runtime error")

    def test_stage_nested(self):
        logger = get_logger("test_iter115_stage_nested_01")
        with stage("outer", logger=logger):
            with stage("inner", logger=logger):
                pass


# ─── ProgressBar ──────────────────────────────────────────────────────────────

class TestProgressBar:
    def test_context_manager_returns_self(self):
        logger = get_logger("test_iter115_pb_01")
        pb = ProgressBar("test", total=10, logger=logger)
        result = pb.__enter__()
        pb.__exit__(None, None, None)
        assert result is pb

    def test_update_does_not_raise(self):
        logger = get_logger("test_iter115_pb_update_01")
        with ProgressBar("update test", total=5, logger=logger) as pb:
            pb.update(1)
            pb.update(3)
            pb.update(5)

    def test_update_beyond_total(self):
        """update with value > total should not crash (clamps to 100%)."""
        logger = get_logger("test_iter115_pb_beyond_01")
        with ProgressBar("overflow test", total=5, logger=logger) as pb:
            pb.update(10)

    def test_update_at_zero_total(self):
        """ProgressBar with total=0 should not divide by zero."""
        logger = get_logger("test_iter115_pb_zero_01")
        with ProgressBar("zero total", total=0, logger=logger) as pb:
            pb.update(0)

    def test_default_width(self):
        logger = get_logger("test_iter115_pb_width_01")
        pb = ProgressBar("width", total=10, logger=logger)
        assert pb.width == 30

    def test_custom_width(self):
        logger = get_logger("test_iter115_pb_custom_01")
        pb = ProgressBar("custom", total=10, width=50, logger=logger)
        assert pb.width == 50

    def test_label_stored(self):
        logger = get_logger("test_iter115_pb_label_01")
        pb = ProgressBar("my label", total=20, logger=logger)
        assert pb.label == "my label"

    def test_total_stored(self):
        logger = get_logger("test_iter115_pb_total_01")
        pb = ProgressBar("x", total=42, logger=logger)
        assert pb.total == 42


# ─── PipelineTimer ────────────────────────────────────────────────────────────

class TestPipelineTimer:
    def test_report_empty(self):
        timer = PipelineTimer()
        result = timer.report()
        assert result == "(нет данных)"

    def test_measure_records_stage(self):
        timer = PipelineTimer()
        with timer.measure("stage1"):
            pass
        assert "stage1" in timer._stages

    def test_measure_time_positive(self):
        timer = PipelineTimer()
        with timer.measure("timed"):
            time.sleep(0.01)
        assert timer._stages["timed"] > 0.0

    def test_measure_time_non_negative(self):
        timer = PipelineTimer()
        with timer.measure("fast"):
            pass
        assert timer._stages["fast"] >= 0.0

    def test_report_contains_stage_name(self):
        timer = PipelineTimer()
        with timer.measure("alpha"):
            pass
        report = timer.report()
        assert "alpha" in report

    def test_multiple_stages(self):
        timer = PipelineTimer()
        with timer.measure("a"):
            pass
        with timer.measure("b"):
            pass
        assert len(timer._stages) == 2

    def test_report_multiple_stages(self):
        timer = PipelineTimer()
        with timer.measure("step1"):
            pass
        with timer.measure("step2"):
            pass
        report = timer.report()
        assert "step1" in report
        assert "step2" in report

    def test_report_contains_total(self):
        timer = PipelineTimer()
        with timer.measure("x"):
            pass
        report = timer.report()
        assert "ИТОГО" in report

    def test_measure_exception_still_records(self):
        """Exception inside measure should still record the time."""
        timer = PipelineTimer()
        try:
            with timer.measure("errored"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert "errored" in timer._stages

    def test_report_is_string(self):
        timer = PipelineTimer()
        with timer.measure("s"):
            pass
        assert isinstance(timer.report(), str)
