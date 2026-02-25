"""Tests for main.py — build_parser() and assemble() helper functions."""
import argparse
import sys
import pytest
from unittest.mock import patch, MagicMock

from main import build_parser, assemble


# ─── build_parser ─────────────────────────────────────────────────────────────

class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_input_argument_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # --input is required

    def test_input_argument_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/some/dir"])
        assert args.input == "/some/dir"

    def test_input_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-i", "/short"])
        assert args.input == "/short"

    def test_output_default(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.output == "result.png"

    def test_output_custom(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--output", "my.png"])
        assert args.output == "my.png"

    def test_output_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-i", "/dir", "-o", "short.png"])
        assert args.output == "short.png"

    def test_method_default(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.method == "beam"

    def test_method_greedy(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--method", "greedy"])
        assert args.method == "greedy"

    def test_method_sa(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--method", "sa"])
        assert args.method == "sa"

    def test_method_beam(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--method", "beam"])
        assert args.method == "beam"

    def test_method_gamma(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--method", "gamma"])
        assert args.method == "gamma"

    def test_method_invalid_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "/dir", "--method", "invalid"])

    def test_verbose_flag_sets_true(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--verbose"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.verbose is False

    def test_visualize_flag_sets_true(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--visualize"])
        assert args.visualize is True

    def test_visualize_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.visualize is False

    def test_interactive_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--interactive"])
        assert args.interactive is True

    def test_interactive_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.interactive is False

    def test_alpha_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--alpha", "0.7"])
        assert args.alpha == pytest.approx(0.7)

    def test_alpha_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.alpha is None

    def test_n_sides_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--n-sides", "3"])
        assert args.n_sides == 3

    def test_n_sides_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.n_sides is None

    def test_beam_width_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--beam-width", "15"])
        assert args.beam_width == 15

    def test_beam_width_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.beam_width is None

    def test_sa_iter_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--sa-iter", "3000"])
        assert args.sa_iter == 3000

    def test_sa_iter_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.sa_iter is None

    def test_config_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--config", "cfg.json"])
        assert args.config == "cfg.json"

    def test_config_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.config is None

    def test_seg_method_otsu(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--seg-method", "otsu"])
        assert args.seg_method == "otsu"

    def test_seg_method_adaptive(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--seg-method", "adaptive"])
        assert args.seg_method == "adaptive"

    def test_seg_method_invalid_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "/dir", "--seg-method", "invalid"])

    def test_threshold_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--threshold", "0.4"])
        assert args.threshold == pytest.approx(0.4)

    def test_log_file_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir", "--log-file", "app.log"])
        assert args.log_file == "app.log"

    def test_log_file_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "/dir"])
        assert args.log_file is None


# ─── assemble ─────────────────────────────────────────────────────────────────

def _make_cfg(method="greedy"):
    from puzzle_reconstruction.config import Config
    cfg = Config()
    cfg.assembly.method = method
    return cfg


_RUN_SELECTED = "main.run_selected"


def _mock_run_selected(mock_asm):
    """Returns a mock result list as run_selected() would return for a single method."""
    return [MagicMock(success=True, assembly=mock_asm, error=None)]


class TestAssemble:
    def test_greedy_calls_greedy_assembly(self):
        cfg = _make_cfg("greedy")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            result, _ = assemble([], [], cfg, MagicMock())
        mock.assert_called_once()
        assert result is mock_asm

    def test_beam_calls_beam_search(self):
        cfg = _make_cfg("beam")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            result, _ = assemble([], [], cfg, MagicMock())
        mock.assert_called_once()
        assert result is mock_asm

    def test_sa_calls_simulated_annealing(self):
        cfg = _make_cfg("sa")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            result, _ = assemble([], [], cfg, MagicMock())
        mock.assert_called_once()
        assert result is mock_asm

    def test_gamma_calls_gamma_optimizer(self):
        cfg = _make_cfg("gamma")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            result, _ = assemble([], [], cfg, MagicMock())
        mock.assert_called_once()
        assert result is mock_asm

    def test_sa_uses_greedy_as_init(self):
        """SA delegates to run_selected with method='sa'."""
        cfg = _make_cfg("sa")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            assemble([], [], cfg, MagicMock())
        _, kwargs = mock.call_args
        assert kwargs.get("methods") == ["sa"]

    def test_gamma_uses_greedy_as_init(self):
        """gamma delegates to run_selected with method='gamma'."""
        cfg = _make_cfg("gamma")
        mock_asm = MagicMock()
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(mock_asm)) as mock:
            assemble([], [], cfg, MagicMock())
        _, kwargs = mock.call_args
        assert kwargs.get("methods") == ["gamma"]

    def test_unknown_method_sys_exit(self):
        cfg = _make_cfg("unknown")
        with pytest.raises(SystemExit):
            assemble([], [], cfg, MagicMock())

    def test_greedy_passes_fragments_and_entries(self):
        cfg = _make_cfg("greedy")
        frags = [MagicMock()]
        entries = [MagicMock()]
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble(frags, entries, cfg, MagicMock())
        pos_args = mock.call_args[0]
        assert pos_args[0] is frags
        assert pos_args[1] is entries

    def test_beam_uses_beam_width_from_config(self):
        cfg = _make_cfg("beam")
        cfg.assembly.beam_width = 7
        with patch(_RUN_SELECTED, return_value=_mock_run_selected(MagicMock())) as mock:
            assemble([], [], cfg, MagicMock())
        _, kwargs = mock.call_args
        assert kwargs.get("beam_width") == 7
