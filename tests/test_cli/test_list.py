"""Tests for ``qr-sampler list`` command group."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestListEngines:
    """Tests for ``qr-sampler list engines``."""

    def test_lists_vllm(self, runner: CliRunner) -> None:
        """Lists the vllm engine profile."""
        result = runner.invoke(cli, ["list", "engines"])
        assert result.exit_code == 0
        assert "vllm" in result.output

    def test_lists_vllm_metal(self, runner: CliRunner) -> None:
        """Lists the vllm_metal engine profile."""
        result = runner.invoke(cli, ["list", "engines"])
        assert result.exit_code == 0
        assert "vllm_metal" in result.output


class TestListModels:
    """Tests for ``qr-sampler list models``."""

    def test_lists_models_for_engine(self, runner: CliRunner) -> None:
        """Lists known-working models for a specific engine."""
        result = runner.invoke(cli, ["list", "models", "--engine", "vllm"])
        assert result.exit_code == 0
        assert "Qwen/Qwen2.5-1.5B-Instruct" in result.output

    def test_lists_all_models(self, runner: CliRunner) -> None:
        """Lists models from all engines when no filter."""
        result = runner.invoke(cli, ["list", "models"])
        assert result.exit_code == 0
        assert "Qwen/Qwen2.5-1.5B-Instruct" in result.output

    def test_unknown_engine_error(self, runner: CliRunner) -> None:
        """Unknown engine produces error."""
        result = runner.invoke(cli, ["list", "models", "--engine", "nonexistent"])
        assert result.exit_code != 0


class TestListEntropySources:
    """Tests for ``qr-sampler list entropy-sources``."""

    def test_lists_sources(self, runner: CliRunner) -> None:
        """Lists available entropy source profiles."""
        result = runner.invoke(cli, ["list", "entropy-sources"])
        assert result.exit_code == 0
        assert "system" in result.output
        assert "quantum_grpc" in result.output


class TestListAmplifiers:
    """Tests for ``qr-sampler list amplifiers``."""

    def test_lists_amplifiers(self, runner: CliRunner) -> None:
        """Lists available amplifier profiles."""
        result = runner.invoke(cli, ["list", "amplifiers"])
        assert result.exit_code == 0
        assert "zscore_mean" in result.output

    def test_filters_by_entropy_source(self, runner: CliRunner) -> None:
        """Filters amplifiers by compatible entropy source."""
        result = runner.invoke(
            cli,
            ["list", "amplifiers", "--entropy", "quantum_grpc"],
        )
        assert result.exit_code == 0
        assert "zscore_mean" in result.output


class TestListSamplers:
    """Tests for ``qr-sampler list samplers``."""

    def test_lists_samplers(self, runner: CliRunner) -> None:
        """Lists available sampler profiles."""
        result = runner.invoke(cli, ["list", "samplers"])
        assert result.exit_code == 0
        assert "fixed" in result.output
        assert "edt" in result.output
