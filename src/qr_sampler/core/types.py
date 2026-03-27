"""Data types for the engine-agnostic sampling pipeline.

Defines ``SamplingResult``, the frozen dataclass returned by
``SamplingPipeline.sample_token()``. Contains the selected token,
a numpy one-hot array, and the full diagnostic record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.logging.types import TokenSamplingRecord


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Result of a single token sampling operation.

    Returned by ``SamplingPipeline.sample_token()``. The ``one_hot`` array
    is engine-agnostic (numpy); engine adapters convert it to their native
    tensor format.

    Attributes:
        token_id: Selected vocabulary index.
        one_hot: 1-D numpy array of shape ``(vocab_size,)`` with ``-inf``
            everywhere except ``0.0`` at ``token_id``.
        record: Full sampling record for logging and diagnostics.
    """

    token_id: int
    one_hot: np.ndarray
    record: TokenSamplingRecord
