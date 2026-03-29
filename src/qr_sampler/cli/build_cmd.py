"""``qr-sampler build`` command.

Generates Docker Compose files and environment templates from a stack
configuration. Uses Jinja2 templates shipped with the package.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field

from qr_sampler.profiles.compatibility import CompatibilityChecker
from qr_sampler.profiles.loader import ProfileLoader

if TYPE_CHECKING:
    from qr_sampler.profiles.schema import (
        AmplifierProfile,
        EngineProfile,
        EntropySourceProfile,
        SamplerProfile,
    )


class StackConfig(BaseModel):
    """User-facing stack configuration for the ``build`` command.

    Validated via pydantic. Can be loaded from a YAML config file
    or constructed from CLI arguments.
    """

    model_config = ConfigDict(frozen=True)

    engine: str = "vllm"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    entropy_source: str = "system"
    amplifier: str = "zscore_mean"
    sampler: str = "fixed"
    engine_config: dict[str, Any] = Field(default_factory=dict)
    entropy_config: dict[str, Any] = Field(default_factory=dict)


@dataclass
class BuildContext:
    """Resolved profile data for template rendering.

    Constructed after loading and validating all profiles. Passed
    to Jinja2 templates as the rendering context.
    """

    engine: EngineProfile
    model: str
    entropy_source: EntropySourceProfile
    amplifier: AmplifierProfile
    sampler: SamplerProfile
    engine_config: dict[str, Any] = field(default_factory=dict)
    entropy_config: dict[str, Any] = field(default_factory=dict)
    needs_entropy_server: bool = False
    project_name: str = "qr-sampler"


def _resolve_templates_dir() -> Path:
    """Locate the built-in templates directory via importlib.resources."""
    ref = importlib.resources.files("qr_sampler") / "templates"
    return Path(str(ref))


def _render_template(template_name: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template with the given context.

    Args:
        template_name: Name of the template file (e.g. ``'docker-compose.yml.j2'``).
        context: Template variables.

    Returns:
        Rendered string.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise click.ClickException(
            "The 'build' command requires Jinja2. Install with: pip install qr-sampler[cli]"
        ) from exc

    templates_dir = _resolve_templates_dir()
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)


def _build_context(stack: StackConfig, loader: ProfileLoader) -> BuildContext:
    """Load profiles and build the template rendering context.

    Args:
        stack: Validated stack configuration.
        loader: Profile loader instance.

    Returns:
        Fully resolved build context.

    Raises:
        click.ClickException: If a profile cannot be loaded.
    """
    try:
        engine = loader.load_engine(stack.engine)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        entropy_source = loader.load_entropy_source(stack.entropy_source)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        amplifier = loader.load_amplifier(stack.amplifier)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        sampler = loader.load_sampler(stack.sampler)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    needs_entropy_server = entropy_source.transport == "grpc"

    # Merge engine defaults with user overrides.
    merged_engine_config = dict(engine.defaults)
    merged_engine_config.update(stack.engine_config)

    # Merge entropy defaults with user overrides.
    merged_entropy_config = dict(entropy_source.defaults)
    merged_entropy_config.update(stack.entropy_config)

    return BuildContext(
        engine=engine,
        model=stack.model,
        entropy_source=entropy_source,
        amplifier=amplifier,
        sampler=sampler,
        engine_config=merged_engine_config,
        entropy_config=merged_entropy_config,
        needs_entropy_server=needs_entropy_server,
        project_name="qr-sampler",
    )


def _load_config_file(config_path: Path) -> dict[str, Any]:
    """Load a stack configuration YAML file."""
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


@click.command()
@click.option(
    "--engine",
    default=None,
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
    help="Stack configuration YAML file.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    show_default=True,
    help="Output directory for generated files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print generated files without writing to disk.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Bypass compatibility warnings (proceed despite untested combos).",
)
def build(
    engine: str | None,
    model: str | None,
    entropy_source: str | None,
    amplifier: str | None,
    sampler: str | None,
    config_path: Path | None,
    output_dir: Path,
    dry_run: bool,
    force: bool,
) -> None:
    """Generate Docker Compose files for a stack configuration.

    Validates the combination, then renders docker-compose.yml and .env
    files from templates. Use --dry-run to preview output without writing.
    """
    # Build stack config from CLI args + optional file.
    stack_data: dict[str, Any] = {}
    if config_path is not None:
        stack_data = _load_config_file(config_path)

    # CLI flags override file values (if non-None).
    if engine is not None:
        stack_data["engine"] = engine
    if model is not None:
        stack_data["model"] = model
    if entropy_source is not None:
        stack_data["entropy_source"] = entropy_source
    if amplifier is not None:
        stack_data["amplifier"] = amplifier
    if sampler is not None:
        stack_data["sampler"] = sampler

    stack = StackConfig.model_validate(stack_data)

    # Validate compatibility.
    loader = ProfileLoader()
    checker = CompatibilityChecker(loader)
    report = checker.check_stack(
        engine=stack.engine,
        model=stack.model,
        entropy_source=stack.entropy_source,
        amplifier=stack.amplifier,
        sampler=stack.sampler,
    )

    if report.errors:
        for err in report.errors:
            click.echo(click.style(f"Error: {err}", fg="red"), err=True)
        raise SystemExit(2)

    if report.warnings and not force:
        for warn in report.warnings:
            click.echo(click.style(f"Warning: {warn}", fg="yellow"), err=True)
        click.echo(
            click.style("Use --force to proceed despite warnings.", fg="yellow"),
            err=True,
        )
        raise SystemExit(1)

    # Build rendering context.
    ctx = _build_context(stack, loader)
    template_vars = {
        "engine": ctx.engine,
        "model": ctx.model,
        "entropy_source": ctx.entropy_source,
        "amplifier": ctx.amplifier,
        "sampler": ctx.sampler,
        "engine_config": ctx.engine_config,
        "entropy_config": ctx.entropy_config,
        "needs_entropy_server": ctx.needs_entropy_server,
        "project_name": ctx.project_name,
    }

    compose_content = _render_template("docker-compose.yml.j2", template_vars)
    env_content = _render_template("env.j2", template_vars)

    if dry_run:
        click.echo("# docker-compose.yml")
        click.echo(compose_content)
        click.echo()
        click.echo("# .env")
        click.echo(env_content)
        return

    # Write files.
    output_dir.mkdir(parents=True, exist_ok=True)

    compose_path = output_dir / "docker-compose.yml"
    compose_path.write_text(compose_content, encoding="utf-8")
    click.echo(f"Wrote {compose_path}")

    env_path = output_dir / ".env"
    env_path.write_text(env_content, encoding="utf-8")
    click.echo(f"Wrote {env_path}")

    env_example_path = output_dir / ".env.example"
    env_example_path.write_text(env_content, encoding="utf-8")
    click.echo(f"Wrote {env_example_path}")

    click.echo()
    click.echo(click.style("Build artifacts generated successfully.", fg="green", bold=True))
