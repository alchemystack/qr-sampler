"""Registry for engine adapter implementations.

Uses a decorator pattern for registration and entry-point auto-discovery
for third-party adapters, following the same pattern as
``EntropySourceRegistry``.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.engines.base import EngineAdapter

logger = logging.getLogger("qr_sampler")

_ENTRY_POINT_GROUP = "qr_sampler.engine_adapters"


class EngineAdapterRegistry:
    """Registry for engine adapter classes.

    Discovery chain:

    1. Built-in adapters registered via ``@EngineAdapterRegistry.register()``
       decorator
    2. Third-party adapters discovered via ``qr_sampler.engine_adapters``
       entry points (loaded lazily on first ``get()`` call)
    """

    _registry: ClassVar[dict[str, type[EngineAdapter]]] = {}
    _entry_points_loaded: ClassVar[bool] = False

    @classmethod
    def register(cls, name: str) -> Callable[[type[EngineAdapter]], type[EngineAdapter]]:
        """Decorator to register an adapter class under a string key.

        Args:
            name: Unique identifier for the adapter (e.g., ``'vllm'``).

        Returns:
            The original class, unmodified.

        Example::

            @EngineAdapterRegistry.register("vllm")
            class VLLMAdapter(EngineAdapter):
                ...
        """

        def decorator(adapter_cls: type[EngineAdapter]) -> type[EngineAdapter]:
            cls._registry[name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EngineAdapter]:
        """Look up an adapter class by name.

        Loads entry points on the first call if not already loaded.

        Args:
            name: Registered identifier for the adapter.

        Returns:
            The engine adapter class (not an instance).

        Raises:
            KeyError: If *name* is not found after loading entry points.
        """
        if name in cls._registry:
            return cls._registry[name]

        # Lazy-load third-party entry points.
        if not cls._entry_points_loaded:
            cls._load_entry_points()
            if name in cls._registry:
                return cls._registry[name]

        available = ", ".join(sorted(cls._registry.keys())) or "(none)"
        raise KeyError(f"Unknown engine adapter: {name!r}. Available: {available}")

    @classmethod
    def list_available(cls) -> list[str]:
        """Return all registered adapter names.

        Triggers entry-point loading if not yet done.

        Returns:
            Sorted list of registered adapter identifiers.
        """
        if not cls._entry_points_loaded:
            cls._load_entry_points()
        return sorted(cls._registry.keys())

    @classmethod
    def _load_entry_points(cls) -> None:
        """Discover and register adapters from the entry-point group.

        Each entry point maps a name to a fully-qualified class path.
        Errors during individual entry-point loading are logged as warnings
        but do not prevent other adapters from loading.
        """
        cls._entry_points_loaded = True
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception:  # Intentional: must not crash on broken metadata
            logger.warning(
                "Failed to load entry points for %s",
                _ENTRY_POINT_GROUP,
                exc_info=True,
            )
            return

        for ep in eps:
            if ep.name in cls._registry:
                # Built-in decorator registration takes precedence.
                continue
            try:
                adapter_cls = ep.load()
                cls._registry[ep.name] = adapter_cls
                logger.debug("Loaded engine adapter %r from entry point", ep.name)
            except Exception:  # Intentional: one bad plugin must not block others
                logger.warning(
                    "Failed to load engine adapter entry point %r: %s",
                    ep.name,
                    ep.value,
                    exc_info=True,
                )

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state. **Test-only** -- not part of public API."""
        cls._registry.clear()
        cls._entry_points_loaded = False
