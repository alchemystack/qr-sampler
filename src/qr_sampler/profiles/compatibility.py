"""Compatibility checker for stack component combinations.

Uses profile metadata to determine whether a given engine + model +
entropy source + amplifier + sampler combination is known-working,
untested, or known-incompatible. This is advisory validation only --
it never blocks runtime sampling behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qr_sampler.profiles.loader import ProfileLoader

logger = logging.getLogger("qr_sampler")


@dataclass(frozen=True, slots=True)
class CompatibilityStatus:
    """Result of a pairwise compatibility check.

    Attributes:
        component_a: First component identifier (e.g. ``'engine:vllm'``).
        component_b: Second component identifier (e.g. ``'model:Qwen/Qwen2.5-1.5B-Instruct'``).
        status: Tri-state result.
        message: Human-readable explanation.
    """

    component_a: str
    component_b: str
    status: Literal["known_working", "untested", "known_incompatible"]
    message: str = ""


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Full compatibility report for a stack configuration.

    Attributes:
        checks: Individual pairwise compatibility results.
        warnings: Non-fatal issues (e.g. untested combinations).
        errors: Fatal issues (e.g. known-incompatible pairs).
        available_samplers: Samplers compatible with this stack.
        missing_dependencies: Dependencies not available.
    """

    checks: list[CompatibilityStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    available_samplers: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors or missing dependencies were found."""
        return len(self.errors) == 0 and len(self.missing_dependencies) == 0


class CompatibilityChecker:
    """Check compatibility between stack components using profile data.

    All checks are advisory -- missing profiles produce warnings, not
    runtime failures. Profile data drives CLI validation and documentation.
    """

    def __init__(self, loader: ProfileLoader) -> None:
        self._loader = loader

    def check_engine_model(self, engine_id: str, model_id: str) -> CompatibilityStatus:
        """Check whether an engine is known to work with a model.

        Args:
            engine_id: Engine profile identifier.
            model_id: Model identifier string (e.g. HuggingFace repo path).

        Returns:
            Tri-state compatibility status.
        """
        engine = self._loader.load_engine(engine_id)

        if model_id in engine.known_incompatible_models:
            return CompatibilityStatus(
                component_a=f"engine:{engine_id}",
                component_b=f"model:{model_id}",
                status="known_incompatible",
                message=(f"Model {model_id!r} is listed as incompatible with engine {engine_id!r}"),
            )

        if model_id in engine.known_working_models:
            return CompatibilityStatus(
                component_a=f"engine:{engine_id}",
                component_b=f"model:{model_id}",
                status="known_working",
                message=(f"Model {model_id!r} is verified working with engine {engine_id!r}"),
            )

        return CompatibilityStatus(
            component_a=f"engine:{engine_id}",
            component_b=f"model:{model_id}",
            status="untested",
            message=(f"Model {model_id!r} has not been tested with engine {engine_id!r}"),
        )

    def check_entropy_amplifier(self, source_id: str, amplifier_id: str) -> CompatibilityStatus:
        """Check whether an entropy source is compatible with an amplifier.

        Performs a bidirectional check: the source's compatible/incompatible
        amplifier lists AND the amplifier's compatible/incompatible source
        lists are both consulted. Incompatibility on either side is decisive.

        Args:
            source_id: Entropy source profile identifier.
            amplifier_id: Amplifier profile identifier.

        Returns:
            Tri-state compatibility status.
        """
        source = self._loader.load_entropy_source(source_id)
        amplifier = self._loader.load_amplifier(amplifier_id)

        # Check incompatibility on either side first.
        if amplifier_id in source.known_incompatible_amplifiers:
            return CompatibilityStatus(
                component_a=f"entropy:{source_id}",
                component_b=f"amplifier:{amplifier_id}",
                status="known_incompatible",
                message=(
                    f"Amplifier {amplifier_id!r} is listed as incompatible "
                    f"with entropy source {source_id!r}"
                ),
            )

        if source_id in amplifier.known_incompatible_entropy_sources:
            return CompatibilityStatus(
                component_a=f"entropy:{source_id}",
                component_b=f"amplifier:{amplifier_id}",
                status="known_incompatible",
                message=(
                    f"Entropy source {source_id!r} is listed as incompatible "
                    f"with amplifier {amplifier_id!r}"
                ),
            )

        # Check known-working on either side.
        source_says_ok = amplifier_id in source.compatible_amplifiers
        amp_says_ok = source_id in amplifier.compatible_entropy_sources

        if source_says_ok or amp_says_ok:
            return CompatibilityStatus(
                component_a=f"entropy:{source_id}",
                component_b=f"amplifier:{amplifier_id}",
                status="known_working",
                message=(
                    f"Amplifier {amplifier_id!r} is verified working "
                    f"with entropy source {source_id!r}"
                ),
            )

        return CompatibilityStatus(
            component_a=f"entropy:{source_id}",
            component_b=f"amplifier:{amplifier_id}",
            status="untested",
            message=(
                f"Amplifier {amplifier_id!r} has not been tested with entropy source {source_id!r}"
            ),
        )

    def check_dependencies(self, engine_id: str) -> list[str]:
        """Return missing Python package dependencies for an engine.

        Args:
            engine_id: Engine profile identifier.

        Returns:
            List of package names that could not be imported.
        """
        engine = self._loader.load_engine(engine_id)
        missing: list[str] = []
        for dep in engine.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing

    def check_stack(
        self,
        engine: str,
        model: str | None = None,
        entropy_source: str | None = None,
        amplifier: str | None = None,
        sampler: str | None = None,
    ) -> CompatibilityReport:
        """Run all compatibility checks for a full stack configuration.

        Args:
            engine: Engine profile identifier (required).
            model: Model identifier (optional).
            entropy_source: Entropy source profile identifier (optional).
            amplifier: Amplifier profile identifier (optional).
            sampler: Sampler profile identifier (optional, for filtering).

        Returns:
            Full compatibility report with checks, warnings, and errors.
        """
        checks: list[CompatibilityStatus] = []
        warnings: list[str] = []
        errors: list[str] = []

        # Engine-model check.
        if model is not None:
            try:
                status = self.check_engine_model(engine, model)
                checks.append(status)
                if status.status == "known_incompatible":
                    errors.append(status.message)
                elif status.status == "untested":
                    warnings.append(status.message)
            except KeyError as exc:
                errors.append(str(exc))

        # Entropy-amplifier check.
        if entropy_source is not None and amplifier is not None:
            try:
                status = self.check_entropy_amplifier(entropy_source, amplifier)
                checks.append(status)
                if status.status == "known_incompatible":
                    errors.append(status.message)
                elif status.status == "untested":
                    warnings.append(status.message)
            except KeyError as exc:
                errors.append(str(exc))

        # Dependency check.
        try:
            missing = self.check_dependencies(engine)
        except KeyError as exc:
            errors.append(str(exc))
            missing = []

        # Collect available samplers.
        available_samplers: list[str] = []
        try:
            all_samplers = self._loader.list_samplers()
            if sampler is not None:
                # Validate the requested sampler exists.
                sampler_ids = [s.id for s in all_samplers]
                if sampler not in sampler_ids:
                    errors.append(
                        f"Unknown sampler profile: {sampler!r}. "
                        f"Available: {', '.join(sorted(sampler_ids))}"
                    )
                else:
                    available_samplers = [sampler]
            else:
                available_samplers = [s.id for s in all_samplers]
        except Exception:
            logger.warning("Failed to list sampler profiles", exc_info=True)

        return CompatibilityReport(
            checks=checks,
            warnings=warnings,
            errors=errors,
            available_samplers=available_samplers,
            missing_dependencies=missing,
        )
