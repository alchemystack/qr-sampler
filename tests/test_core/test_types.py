"""Tests for core/types.py — SamplingResult frozen dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.core.types import SamplingResult
from qr_sampler.logging.types import TokenSamplingRecord


def _make_record(**overrides: object) -> TokenSamplingRecord:
    """Create a TokenSamplingRecord with default values, allowing overrides."""
    defaults: dict[str, object] = {
        "timestamp_ns": 1000000,
        "entropy_fetch_ms": 0.5,
        "total_sampling_ms": 1.0,
        "entropy_source_used": "mock_uniform",
        "entropy_is_fallback": False,
        "sample_mean": 127.5,
        "z_score": 0.1,
        "u_value": 0.42,
        "temperature_strategy": "fixed",
        "shannon_entropy": 2.3,
        "temperature_used": 0.7,
        "token_id": 3,
        "token_rank": 0,
        "token_prob": 0.9,
        "num_candidates": 10,
        "config_hash": "abcdef1234567890",
    }
    defaults.update(overrides)
    return TokenSamplingRecord(**defaults)  # type: ignore[arg-type]


class TestSamplingResultFields:
    """Test SamplingResult field access."""

    def test_token_id(self) -> None:
        """SamplingResult exposes the selected token_id."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record()
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        assert result.token_id == 3

    def test_one_hot_array(self) -> None:
        """SamplingResult exposes the one-hot numpy array."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[5] = 0.0
        record = _make_record(token_id=5)
        result = SamplingResult(token_id=5, one_hot=one_hot, record=record)

        assert result.one_hot[5] == 0.0
        assert np.sum(np.isneginf(result.one_hot)) == 9
        assert result.one_hot.shape == (10,)

    def test_record_access(self) -> None:
        """SamplingResult exposes the full TokenSamplingRecord."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record(u_value=0.55)
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        assert result.record.u_value == 0.55
        assert result.record.entropy_source_used == "mock_uniform"


class TestSamplingResultImmutability:
    """Test that SamplingResult is frozen (immutable)."""

    def test_cannot_set_token_id(self) -> None:
        """Assigning to token_id raises FrozenInstanceError."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record()
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        with pytest.raises(AttributeError):
            result.token_id = 5  # type: ignore[misc]

    def test_cannot_set_one_hot(self) -> None:
        """Assigning to one_hot raises FrozenInstanceError."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record()
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        with pytest.raises(AttributeError):
            result.one_hot = np.zeros(10)  # type: ignore[misc]

    def test_cannot_set_record(self) -> None:
        """Assigning to record raises FrozenInstanceError."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record()
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        with pytest.raises(AttributeError):
            result.record = _make_record()  # type: ignore[misc]

    def test_has_slots(self) -> None:
        """SamplingResult uses __slots__ for memory efficiency."""
        one_hot = np.full(10, float("-inf"), dtype=np.float32)
        one_hot[3] = 0.0
        record = _make_record()
        result = SamplingResult(token_id=3, one_hot=one_hot, record=record)
        assert hasattr(result, "__slots__")
        # Frozen+slots dataclasses raise either AttributeError or TypeError
        # depending on Python version when setting arbitrary attributes.
        with pytest.raises((AttributeError, TypeError)):
            result.extra_field = "should fail"  # type: ignore[attr-defined]
