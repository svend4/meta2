"""Extra tests for main.py — build_parser() and assemble()."""
import argparse
import pytest
from unittest.mock import patch, MagicMock

from main import build_parser, assemble

_RUN_SELECTED = "main.run_selected"


def _mock_run_selected(mock_asm):
    return [MagicMock(success=True, assembly=mock_asm, error=None)]


def _make_cfg(method="greedy"):
    from puzzle_reconstruction.config import Config
    cfg = Config()
    cfg.assembly.method = method
    return cfg


# ─── build_parser extras ──────────────────────────────────────────────────────

class TestBuildParserExtra:
    def test_parser_prog_name_set(self):
        parser = build_parser()
        assert parser.prog is not None
        assert isinstance(parser.prog, str)

    def test_input_short_and_long_equivalent(self):
        parser = build_parser()
        a = parser.parse_args(["-i", "/dir"])
        b = parser.parse_args(["--input", "/dir"])
        assert a.input == b.input

    def test_output_short_and_long_equivalent(self):
        parser = build_parser()
        a = parser.parse_args(["-i", "/d", "-o", "a.png"])
        b = parser.parse_args(["-i", "/d", "--output", "a.png"])
        assert a.output == b.output

    def test_alpha_zero(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--alpha", "0.0"])
        assert args.alpha == pytest.approx(0.0)

    def test_alpha_one(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--alpha", "1.0"])
        assert args.alpha == pytest.approx(1.0)

    def test_n_sides_zero(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--n-sides", "0"])
        assert args.n_sides == 0

    def test_n_sides_large(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--n-sides", "100"])
        assert args.n_sides == 100

    def test_beam_width_one(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--beam-width", "1"])
        assert args.beam_width == 1

    def test_sa_iter_large(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--sa-iter", "100000"])
        assert args.sa_iter == 100000

    def test_threshold_zero(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--threshold", "0.0"])
        assert args.threshold == pytest.approx(0.0)

    def test_threshold_one(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--threshold", "1.0"])
        assert args.threshold == pytest.approx(1.0)

    def test_config_json_path(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--config", "my/config.json"])
        assert args.config == "my/config.json"

    def test_log_file_path_with_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--log-file", "/var/log/app.log"])
        assert args.log_file == "/var/log/app.log"

    def test_all_flags_combined(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input", "/x",
            "--output", "out.png",
            "--method", "beam",
            "--verbose",
            "--visualize",
            "--interactive",
            "--alpha", "0.5",
            "--n-sides", "4",
            "--beam-width", "10",
            "--sa-iter", "500",
            "--config", "cfg.json",
            "--seg-method", "otsu",
            "--threshold", "0.3",
            "--log-file", "run.log",
        ])
        assert args.input == "/x"
        assert args.output == "out.png"
        assert args.method == "beam"
        assert args.verbose is True
        assert args.visualize is True
        assert args.interactive is True
        assert args.alpha == pytest.approx(0.5)
        assert args.n_sides == 4
        assert args.beam_width == 10
        assert args.sa_iter == 500
        assert args.config == "cfg.json"
        assert args.seg_method == "otsu"
        assert args.threshold == pytest.approx(0.3)
        assert args.log_file == "run.log"

    def test_seg_method_default_not_invalid(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        # default value should be a valid string (not None), or None — either is acceptable
        assert args.seg_method is None or isinstance(args.seg_method, str)

    def test_method_choices_exhaustive(self):
        parser = build_parser()
        for m in ("greedy", "sa", "beam", "gamma"):
            args = parser.parse_args(["--input", "/dir", "--method", m])
            assert args.method == m


# ─── assemble extras ──────────────────────────────────────────────────────────

class TestAssembleExtra:
    def test_greedy_result_returned(self):
        cfg = _make_cfg("greedy")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)):
            result = assemble([], [], cfg, MagicMock())
        assert result is mock_asm

    def test_beam_result_returned(self):
        cfg = _make_cfg("beam")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)):
            result = assemble([], [], cfg, MagicMock())
        assert result is mock_asm

    def test_sa_result_returned(self):
        cfg = _make_cfg("sa")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)):
            result = assemble([], [], cfg, MagicMock())
        assert result is mock_asm

    def test_gamma_result_returned(self):
        cfg = _make_cfg("gamma")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)):
            result = assemble([], [], cfg, MagicMock())
        assert result is mock_asm

    def test_beam_called_once_only(self):
        cfg = _make_cfg("beam")
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble([], [], cfg, MagicMock())
        assert mock.call_count == 1

    def test_greedy_called_once_only(self):
        cfg = _make_cfg("greedy")
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble([], [], cfg, MagicMock())
        assert mock.call_count == 1

    def test_sa_simulated_annealing_called_once(self):
        cfg = _make_cfg("sa")
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble([], [], cfg, MagicMock())
        assert mock.call_count == 1

    def test_gamma_optimizer_called_once(self):
        cfg = _make_cfg("gamma")
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble([], [], cfg, MagicMock())
        assert mock.call_count == 1

    def test_unknown_method_raises_system_exit(self):
        cfg = _make_cfg("xxx_unknown")
        with pytest.raises(SystemExit):
            assemble([], [], cfg, MagicMock())

    def test_greedy_passes_correct_frags(self):
        cfg = _make_cfg("greedy")
        frags = [MagicMock(), MagicMock()]
        entries = []
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble(frags, entries, cfg, MagicMock())
        pos_args = mock.call_args[0]
        assert pos_args[0] is frags
