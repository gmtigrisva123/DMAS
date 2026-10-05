"""Backend interface.

Agents never build prompt strings, they build a Task (name + payload +
expected output schema) and give it to a backend. That way the offline
reasoner can implement the same tasks without any prompt parsing (runs with
no network, no key, reproducible tests) and the live client owns prompt
rendering, validation and retry in one place.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    Protocol = object
    def runtime_checkable(cls):
        return cls


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Task:
    """A structured request to the reasoning engine."""

    name: str
    system: str
    instruction: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    # keys the response must have -> expected python type
    schema: Mapping[str, type] = field(default_factory=dict)
    # json schema for the reply, backends with constrained decoding enforce it
    # server side. `schema` above is still what the pipeline validates against.
    json_schema: Optional[Mapping[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def cache_key_parts(self) -> Tuple[Any, ...]:
        return (self.name, self.system, self.instruction, _stable(self.payload))


@dataclass
class Completion:
    """Raw backend response plus the telemetry we record."""

    text: str = ""
    reasoning: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    backend: str = "offline"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    degraded: bool = False
    note: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMBackend(Protocol):
    """What every backend has to provide."""

    name: str

    def run(self, task: Task) -> Completion:
        ...

    def available(self) -> bool:
        ...


def _stable(value: Any) -> Any:
    """Hashable, order independent projection used for cache keys."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _stable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_stable(v) for v in value))
    return value


def validate(data: Mapping[str, Any], schema: Mapping[str, type]) -> Tuple[bool, str]:
    """Check a parsed response against the task's expected shape."""
    for key, expected in schema.items():
        if key not in data:
            return False, f"missing key {key!r}"
        value = data[key]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if expected is list and isinstance(value, tuple):
            continue
        if not isinstance(value, expected):
            return False, f"key {key!r} should be {expected.__name__}, got {type(value).__name__}"
    return True, ""
