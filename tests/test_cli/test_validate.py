"""Tests for ``qr-sampler validate`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _mock_dep_check() -> object:
    """Mock dependency checks since vllm is not installed in test env."""
    with patch(
        "qr_sampler.profiles.compatibility.CompatibilityChecker.check_dependencies",
        return_value=[],
    ):
        yield


class TestValidateCommand:
    """Tests for the validate subcommand."""

    def test_known_working_stack_exits_zero(self, runner: CliRunner) -> None:
        """Known-working engine + model combination exits 0."""
        result = runner.invoke(
            cli,
            ["validate", "--engine", "vllm", "--model", "Qwen/Qwen2.5-1.5B-Instruct"],
        )
        assert result.exit_code == 0
        assert "ALL KNOWN-WORKING" in result.output

    def test_untested_model_exits_one(self, runner: CliRunner) -> None:
        """Untested model (not in known lists) exits 1 with warning."""
        result = runner.invoke(
            cli,
            ["validate", "--engine", "vllm", "--model", "some/unknown-model"],
        )
        assert result.exit_code == 1
        assert "UNTESTED" in result.output

    def test_incompatible_exits_two(self, runner: CliRunner) -> None:
        """Unknown engine profile exits 2 with error when model check triggers."""
        result = runner.invoke(
            cli,
            [
                "validate",
                "--engine",
                "nonexistent_engine",
                "--model",
                "some/model",
            ],
        )
        assert result.exit_code == 2

    def test_known_working_entropy_amplifier(self, runner: CliRunner) -> None:
        """Known-working entropy + amplifier exits 0."""
        result = runner.invoke(
            cli,
            [
                "validate",
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--entropy",
                "quantum_grpc",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0
        assert "ALL KNOWN-WORKING" in result.output

    def test_sampler_validation(self, runner: CliRunner) -> None:
        """Requesting a valid sampler includes it in the report."""
        result = runner.invoke(
            cli,
            [
                "validate",
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--sampler",
                "edt",
            ],
        )
        assert result.exit_code == 0
        assert "edt" in result.output

    def test_invalid_sampler_exits_two(self, runner: CliRunner) -> None:
        """Requesting a non-existent sampler exits 2."""
        result = runner.invoke(
            cli,
            [
                "validate",
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--sampler",
                "nonexistent",
            ],
        )
        assert result.exit_code == 2

    def test_config_file_support(self, runner: CliRunner, tmp_path: Path) -> None:
        """Validate command loads stack from a YAML config file."""
        config = tmp_path / "stack.yaml"
        config.write_text(
            "engine: vllm\n"
            "model: Qwen/Qwen2.5-1.5B-Instruct\n"
            "entropy_source: system\n"
            "amplifier: zscore_mean\n"
        )
        result = runner.invoke(
            cli,
            ["validate", "--config", str(config)],
        )
        assert result.exit_code == 0

    def test_engine_only_no_model(self, runner: CliRunner) -> None:
        """Validate with only engine specified (no model) exits 0."""
        result = runner.invoke(
            cli,
            ["validate", "--engine", "vllm"],
        )
        assert result.exit_code == 0

    def test_missing_deps_exit_two(self, runner: CliRunner) -> None:
        """When dependencies are missing, exit code is 2."""
        with patch(
            "qr_sampler.profiles.compatibility.CompatibilityChecker.check_dependencies",
            return_value=["vllm"],
        ):
            result = runner.invoke(
                cli,
                ["validate", "--engine", "vllm"],
            )
            assert result.exit_code == 2
            assert "Missing dependencies" in result.output
