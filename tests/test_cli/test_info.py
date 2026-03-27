"""Tests for ``qr-sampler info`` command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestInfoEngine:
    """Tests for ``qr-sampler info engine``."""

    def test_shows_vllm_info(self, runner: CliRunner) -> None:
        """Shows detailed info for the vllm engine."""
        result = runner.invoke(cli, ["info", "engine", "vllm"])
        assert result.exit_code == 0
        assert "vLLM" in result.output
        assert "vllm" in result.output
        assert "Qwen/Qwen2.5-1.5B-Instruct" in result.output

    def test_shows_vllm_metal_info(self, runner: CliRunner) -> None:
        """Shows detailed info for the vllm_metal engine."""
        result = runner.invoke(cli, ["info", "engine", "vllm_metal"])
        assert result.exit_code == 0
        assert "Apple Silicon" in result.output

    def test_unknown_engine_error(self, runner: CliRunner) -> None:
        """Unknown engine name produces error."""
        result = runner.invoke(cli, ["info", "engine", "nonexistent"])
        assert result.exit_code != 0


class TestInfoEntropy:
    """Tests for ``qr-sampler info entropy``."""

    def test_shows_quantum_grpc_info(self, runner: CliRunner) -> None:
        """Shows detailed info for quantum_grpc source."""
        result = runner.invoke(cli, ["info", "entropy", "quantum_grpc"])
        assert result.exit_code == 0
        assert "Quantum gRPC" in result.output
        assert "grpc" in result.output

    def test_shows_system_info(self, runner: CliRunner) -> None:
        """Shows detailed info for system entropy source."""
        result = runner.invoke(cli, ["info", "entropy", "system"])
        assert result.exit_code == 0
        assert "System" in result.output


class TestInfoAmplifier:
    """Tests for ``qr-sampler info amplifier``."""

    def test_shows_zscore_info(self, runner: CliRunner) -> None:
        """Shows detailed info for zscore_mean amplifier."""
        result = runner.invoke(cli, ["info", "amplifier", "zscore_mean"])
        assert result.exit_code == 0
        assert "Z-Score" in result.output

    def test_shows_compatible_sources(self, runner: CliRunner) -> None:
        """Amplifier info includes compatible entropy sources."""
        result = runner.invoke(cli, ["info", "amplifier", "zscore_mean"])
        assert result.exit_code == 0
        assert "system" in result.output


class TestInfoSampler:
    """Tests for ``qr-sampler info sampler``."""

    def test_shows_edt_info(self, runner: CliRunner) -> None:
        """Shows detailed info for EDT sampler."""
        result = runner.invoke(cli, ["info", "sampler", "edt"])
        assert result.exit_code == 0
        assert "Entropy-Dependent Temperature" in result.output
        assert "requires_vocab_size" in result.output.lower() or "True" in result.output

    def test_shows_fixed_info(self, runner: CliRunner) -> None:
        """Shows detailed info for fixed sampler."""
        result = runner.invoke(cli, ["info", "sampler", "fixed"])
        assert result.exit_code == 0
        assert "Fixed" in result.output
