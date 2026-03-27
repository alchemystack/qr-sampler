"""Top-level Click group for the qr-sampler CLI.

Entry point registered via ``[project.scripts]`` in ``pyproject.toml``.
Also accessible via ``python -m qr_sampler``.
"""

from __future__ import annotations

try:
    import click
except ImportError as exc:
    raise ImportError(
        "The qr-sampler CLI requires the 'cli' extra. Install with: pip install qr-sampler[cli]"
    ) from exc

from qr_sampler.cli.build_cmd import build
from qr_sampler.cli.info_cmd import info
from qr_sampler.cli.list_cmd import list_profiles
from qr_sampler.cli.validate_cmd import validate


@click.group()
@click.version_option(package_name="qr-sampler")
def cli() -> None:
    """qr-sampler: Plug any randomness source into LLM token sampling."""


cli.add_command(validate)
cli.add_command(build)
cli.add_command(list_profiles, name="list")
cli.add_command(info)
