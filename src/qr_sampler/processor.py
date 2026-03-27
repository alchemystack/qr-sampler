"""Backward-compatible re-export of QRSamplerLogitsProcessor.

The implementation has moved to ``qr_sampler.engines.vllm.VLLMAdapter``.
This module re-exports ``VLLMAdapter`` under the original name to preserve
the ``vllm.logits_processors`` entry point and any direct imports.
"""

from qr_sampler.engines.vllm import VLLMAdapter as QRSamplerLogitsProcessor

__all__ = ["QRSamplerLogitsProcessor"]
