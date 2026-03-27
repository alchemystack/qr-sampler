"""``qr-sampler validate`` command.

Checks compatibility of a component combination and reports results.

Exit codes:
    0 -- All components are known-working.
    1 -- Some components are untested (warnings only).
    2 -- Errors found (incompatible or missing components).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml

from qr_sampler.profiles.compatibility import CompatibilityChecker, CompatibilityReport
from qr_sampler.profiles.loader import ProfileLoader


def _load_stack_config(config_path: Path) -> dict[str, Any]:
    """Load a stack configuration YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        click.BadParameter: If the file cannot be read or parsed.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise click.BadParameter(f"Cannot read config file {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise click.BadParameter(
            f"Expected YAML mapping in {config_path}, got {type(data).__name__}"
        )
    return data


def _print_report(report: CompatibilityReport) -> None:
    """Print a compatibility report to stdout."""
    # Pairwise checks.
    if report.checks:
        click.echo("Compatibility checks:")
        for check in report.checks:
            symbol = {
                "known_working": click.style("OK", fg="green"),
                "untested": click.style("WARN", fg="yellow"),
                "known_incompatible": click.style("FAIL", fg="red"),
            }[check.status]
            click.echo(f"  [{symbol}] {check.component_a} <-> {check.component_b}")
            if check.message:
                click.echo(f"         {check.message}")

    # Warnings.
    if report.warnings:
        click.echo()
        click.echo(click.style("Warnings:", fg="yellow"))
        for warn in report.warnings:
            click.echo(f"  - {warn}")

    # Errors.
    if report.errors:
        click.echo()
        click.echo(click.style("Errors:", fg="red"))
        for err in report.errors:
            click.echo(f"  - {err}")

    # Missing dependencies.
    if report.missing_dependencies:
        click.echo()
        click.echo(click.style("Missing dependencies:", fg="red"))
        for dep in report.missing_dependencies:
            click.echo(f"  - {dep}")

    # Available samplers.
    if report.available_samplers:
        click.echo()
        click.echo("Available samplers:")
        for s in sorted(report.available_samplers):
            click.echo(f"  - {s}")

    # Summary.
    click.echo()
    if report.errors or report.missing_dependencies:
        click.echo(click.style("Result: INCOMPATIBLE", fg="red", bold=True))
    elif report.warnings:
        click.echo(click.style("Result: UNTESTED (proceed with caution)", fg="yellow", bold=True))
    else:
        click.echo(click.style("Result: ALL KNOWN-WORKING", fg="green", bold=True))


@click.command()
@click.option(
    "--engine",
    default="vllm",
    show_default=True,
    help="Engine profile identifier.",
)
@click.option(
    "--model",
    default=None,
    help="Model identifier (e.g. Qwen/Qwen2.5-1.5B-Instruct).",
)
@click.option(
    "--entropy",
    "entropy_source",
    default=None,
    help="Entropy source profile identifier.",
)
@click.option(
    "--amplifier",
    default=None,
    help="Amplifier profile identifier.",
)
@click.option(
    "--sampler",
    default=None,
    help="Sampler profile identifier.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Stack configuration YAML file (overrides individual options).",
)
def validate(
    engine: str,
    model: str | None,
    entropy_source: str | None,
    amplifier: str | None,
    sampler: str | None,
    config_path: Path | None,
) -> None:
    """Validate compatibility of a stack configuration.

    Checks engine-model, entropy-amplifier, and dependency compatibility.
    Exits 0 (all OK), 1 (warnings), or 2 (errors).
    """
    # Load from config file if provided.
    if config_path is not None:
        cfg = _load_stack_config(config_path)
        engine = cfg.get("engine", engine)
        model = cfg.get("model", model)
        entropy_source = cfg.get("entropy_source", entropy_source)
        amplifier = cfg.get("amplifier", amplifier)
        sampler = cfg.get("sampler", sampler)

    loader = ProfileLoader()
    checker = CompatibilityChecker(loader)

    report = checker.check_stack(
        engine=engine,
        model=model,
        entropy_source=entropy_source,
        amplifier=amplifier,
        sampler=sampler,
    )

    _print_report(report)

    # Exit code: 0 = OK, 1 = warnings, 2 = errors.
    if report.errors or report.missing_dependencies:
        raise SystemExit(2)
    if report.warnings:
        raise SystemExit(1)
    raise SystemExit(0)
