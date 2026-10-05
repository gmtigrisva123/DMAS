"""Backend selection + caching / fallback wrapper.

Reasoner is what the agents hold. It picks the configured backend, caches
responses and degrades instead of failing: if the live model is down or
returns garbage the offline reasoner answers and the telemetry records it,
so a report always says which engine produced it.
"""

from typing import Any, Dict, Optional

from ..config import LLMConfig
from ..errors import BackendError, ConfigurationError
from .base import Completion, LLMBackend, Task
from .cache import ResponseCache
from .claude import ClaudeBackend
from .openai_compatible import OpenAICompatibleBackend
from .offline import OfflineReasoner


def make_backend(config: LLMConfig) -> LLMBackend:
    """Turn config.backend into a backend instance."""
    choice = (config.backend or "auto").lower()
    if choice == "offline":
        return OfflineReasoner()
    if choice == "claude":
        backend = ClaudeBackend(config)
        if not backend.available():
            raise ConfigurationError("backend='claude' requires ANTHROPIC_API_KEY")
        return backend
    if choice == "openai":
        compatible = OpenAICompatibleBackend(config)
        if not compatible.available():
            raise ConfigurationError(
                "backend='openai' requires OPENAI_API_KEY (or a localhost base_url)"
            )
        return compatible
    if choice != "auto":
        raise ConfigurationError(f"unknown backend: {config.backend!r}")
    # auto = the paper's default (openai-compatible endpoint) when there is a
    # key or a local server, otherwise the offline reasoner
    live = OpenAICompatibleBackend(config)
    return live if live.available() else OfflineReasoner()


class Reasoner:
    """Caching, degrading wrapper around a backend."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[LLMBackend] = None,
        *,
        cache: Optional[ResponseCache] = None,
    ):
        self.config = config or LLMConfig()
        self.backend = backend or make_backend(self.config)
        self.fallback = OfflineReasoner()
        self.cache = cache if cache is not None else ResponseCache(self.config.cache_dir)
        self.calls = 0
        self.cache_hits = 0
        self.tokens = 0
        self.degradations: list = []

    @property
    def name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    @property
    def is_live(self) -> bool:
        return self.name in ("claude", "openai")

    def run(self, task: Task) -> Completion:
        # the key includes the backend and the model: the offline reasoner and a
        # live model must never share a cache entry
        key = self.cache.key(
            task, f"{self.name}:{self.config.model}:{self.config.effort}", self.config.temperature
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        try:
            completion = self.backend.run(task)
        except BackendError as exc:
            self.degradations.append(f"{task.name}: {exc}")
            completion = self.fallback.run(task)
            completion.degraded = True
            completion.note = str(exc)
        except Exception as exc:
            self.degradations.append(f"{task.name}: unexpected {type(exc).__name__}: {exc}")
            completion = self.fallback.run(task)
            completion.degraded = True
            completion.note = f"{type(exc).__name__}: {exc}"

        self.calls += 1
        self.tokens += completion.total_tokens
        if not completion.degraded:
            self.cache.put(key, completion)
        return completion
