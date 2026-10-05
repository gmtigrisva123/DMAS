"""Backends: Claude, any OpenAI-compatible server, and the offline reasoner."""

from .base import Completion, LLMBackend, Message, Task, validate
from .cache import ResponseCache
from .claude import ClaudeBackend
from .offline import CONCEPT_PROBES, OfflineReasoner
from .openai_compatible import OpenAICompatibleBackend
from .parsing import find_json_object, parse_structured, split_reasoning
from .registry import Reasoner, make_backend
