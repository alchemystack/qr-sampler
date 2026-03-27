"""Abstract base class for inference engine adapters.

An engine adapter translates between an inference engine's hook
mechanism and the engine-agnostic ``SamplingPipeline``. Adapters are
thin wrappers -- they must NOT contain sampling logic.

Architecture invariant #13: Engine adapters are thin wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qr_sampler.core.pipeline import SamplingPipeline


class EngineAdapter(ABC):
    """Abstract base for inference engine adapters.

    An engine adapter translates between an inference engine's hook
    mechanism and the engine-agnostic SamplingPipeline. Adapters are
    thin wrappers -- they must NOT contain sampling logic.

    Subclasses must implement ``get_pipeline()`` to return the underlying
    pipeline. The default ``close()`` delegates to ``pipeline.close()``.
    """

    @abstractmethod
    def get_pipeline(self) -> SamplingPipeline:
        """Return the underlying SamplingPipeline.

        Returns:
            The engine-agnostic sampling pipeline used by this adapter.
        """

    def close(self) -> None:
        """Release resources. Default delegates to pipeline.close()."""
        self.get_pipeline().close()
