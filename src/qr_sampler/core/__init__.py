"""Engine-agnostic core sampling pipeline.

Re-exports the main types and factory functions for use by engine adapters
and external consumers.
"""

from qr_sampler.core.pipeline import (
    SamplingPipeline,
    accepts_config,
    build_entropy_source,
    build_pipeline,
    config_hash,
)
from qr_sampler.core.types import SamplingResult

__all__ = [
    "SamplingPipeline",
    "SamplingResult",
    "accepts_config",
    "build_entropy_source",
    "build_pipeline",
    "config_hash",
]
