"""Engine adapter layer for inference engine integration.

Re-exports the core types needed by consumers and adapter authors.
"""

from qr_sampler.engines.base import EngineAdapter
from qr_sampler.engines.registry import EngineAdapterRegistry

__all__ = [
    "EngineAdapter",
    "EngineAdapterRegistry",
]
