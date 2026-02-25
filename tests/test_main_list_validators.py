"""
Tests for the --list-validators CLI option added in Phase 6 (v1.0.0).

--list-validators is special: it must work WITHOUT --input (which is normally
required) by intercepting sys.argv before argparse runs.

Tested behaviour:
  - Parser has the --list-validators flag defined
  - main() prints all 21 validator names when invoked with --list-validators
  - Output contains usage hint lines
  - main() returns without calling run() (no side-effects)
  - all_validator_names() is stable across multiple calls
"""
from __future__ import annotations

import importlib
import io
import sys
from unittest.mock import patch

import pytest

from puzzle_reconstruction.verification.suite import (
    all_validator_names,
    VerificationSuite,
)
from main import build_parser


# ─── Parser definition ────────────────────────────────────────────────────────

class TestListValidatorsParserFlag:

    def test_flag_is_registered(self):
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        assert "list_validators" in actions

    def test_flag_is_store_true(self):
        parser = build_parser()
        for action in parser._actions:
            if action.dest == "list_validators":
                import argparse
                assert isinstance(action, argparse._StoreTrueAction)
                break

    def test_flag_default_is_false(self):
        """--list-validators defaults to False when not supplied."""
        parser = build_parser()
        args = parser.parse_args(["--input", "/d"])
        assert args.list_validators is False

    def test_flag_set_to_true_when_supplied(self):
        parser = build_parser()
        # --list-validators alone would fail because --input is required,
        # so supply both; main() intercepts before argparse anyway
        args = parser.parse_args(["--input", "/d", "--list-validators"])
        assert args.list_validators is True


# ─── main() output when --list-validators present ─────────────────────────────

class TestListValidatorsOutput:
    """Uses sys.argv patching + capsys to capture output from main()."""

    def _run_list_validators(self, capsys):
        """Runs main() with --list-validators injected into sys.argv."""
        import main as main_module
        with patch.object(sys, "argv", ["main.py", "--list-validators"]):
            main_module.main()
        return capsys.readouterr().out

    def test_runs_without_error(self, capsys):
        self._run_list_validators(capsys)  # must not raise

    def test_shows_count_line(self, capsys):
        out = self._run_list_validators(capsys)
        assert "21" in out

    def test_shows_all_validator_names(self, capsys):
        out = self._run_list_validators(capsys)
        for name in all_validator_names():
            assert name in out, f"Validator '{name}' missing from output"

    def test_shows_numbered_list(self, capsys):
        out = self._run_list_validators(capsys)
        # Should have at least lines " 1. xxx" through "21. xxx"
        assert " 1." in out
        assert "21." in out

    def test_shows_usage_hint(self, capsys):
        out = self._run_list_validators(capsys)
        assert "--validators" in out

    def test_shows_all_keyword_example(self, capsys):
        out = self._run_list_validators(capsys)
        assert "all" in out

    def test_does_not_call_run(self, capsys):
        """main() should return early without calling run()."""
        import main as main_module
        with patch.object(sys, "argv", ["main.py", "--list-validators"]):
            with patch.object(main_module, "run") as mock_run:
                main_module.main()
        mock_run.assert_not_called()

    def test_output_has_exactly_21_validator_lines(self, capsys):
        out = self._run_list_validators(capsys)
        names = all_validator_names()
        count = sum(1 for name in names if name in out)
        assert count == 21


# ─── all_validator_names() contract ───────────────────────────────────────────

class TestAllValidatorNamesContract:

    def test_returns_list(self):
        assert isinstance(all_validator_names(), list)

    def test_length_is_21(self):
        assert len(all_validator_names()) == 21

    def test_all_strings(self):
        for name in all_validator_names():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_no_duplicates(self):
        names = all_validator_names()
        assert len(names) == len(set(names))

    def test_stable_across_calls(self):
        assert all_validator_names() == all_validator_names()

    def test_original_9_present(self):
        original = {"assembly_score", "layout", "completeness", "seam",
                    "overlap", "text_coherence", "confidence",
                    "consistency", "edge_quality"}
        names = set(all_validator_names())
        for n in original:
            assert n in names, f"Original validator '{n}' missing"

    def test_new_12_present(self):
        new_12 = {"boundary", "layout_verify", "overlap_validate", "spatial",
                  "placement", "layout_score", "fragment_valid", "quality_report",
                  "score_report", "full_report", "metrics", "overlap_area"}
        names = set(all_validator_names())
        for n in new_12:
            assert n in names, f"New validator '{n}' missing"

    def test_suite_accepts_all_names(self):
        """VerificationSuite should accept all 21 names without error."""
        names = all_validator_names()
        suite  = VerificationSuite(validators=names)
        assert suite is not None
