"""Claude backend, talks to the Messages API with plain urllib (no SDK).

Three things that matter for a reasoning model:
- it thinks before answering, the thinking blocks are kept for the audit
  trail and never shown to a student (they usually contain the answer)
- every task has a json schema and we ask the API to constrain the reply to
  it (output_config.format). A reply that still fails validation is retried
  once with a repair instruction before we report the backend as degraded
- no sampling params are sent, current Claude models reject temperature /
  top_p, reproducibility comes from the response cache
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..config import ANTHROPIC_URL, GROQ_URL, LLMConfig
from ..errors import BackendError
from ..version import __version__
from .base import Completion, Task, validate
from .parsing import parse_structured

API_VERSION = "2023-06-01"
_USER_AGENT = f"dmas/{__version__} (python-urllib)"

_JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. Do not wrap it in prose. "
    "Required keys: {keys}."
)

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed as the required JSON object ({problem}). "
    "Reply again with only the JSON object."
)

# older model families without adaptive thinking / effort
_LEGACY_PREFIXES = ("claude-haiku-4-5", "claude-3", "claude-sonnet-4-5", "claude-opus-4-5")

# status codes worth retrying: rate limit, overload, transient server errors
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ClaudeBackend:
    """Messages API client."""

    name = "claude"

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        # the shared default base_url points at groq, which is not where the
        # Messages API lives; an explicit base_url is kept as is
        if self.config.base_url == GROQ_URL:
            self.config = replace(self.config, base_url=ANTHROPIC_URL)

    # liveness
    def available(self) -> bool:
        return bool(self.config.resolved_key("claude"))

    # rendering
    def render(self, task: Task, repair: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """System prompt + message list for one task."""
        keys = ", ".join(sorted(task.schema)) or "as described"
        payload = json.dumps(task.payload, ensure_ascii=False, indent=2, default=str, sort_keys=True)
        user = (
            f"{task.instruction}\n\n"
            f"### Structured input\n```json\n{payload}\n```\n\n"
            f"{_JSON_INSTRUCTION.format(keys=keys)}"
        )
        if repair:
            user += "\n\n" + _REPAIR_INSTRUCTION.format(problem=repair)
        return task.system, [{"role": "user", "content": user}]

    def body(self, task: Task, repair: Optional[str] = None, *, structured: bool = True) -> Dict[str, Any]:
        """Request body. structured=True asks the API to enforce the task schema."""
        system, messages = self.render(task, repair=repair)
        body: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": task.max_tokens or self.config.max_tokens,
            "system": system,
            "messages": messages,
        }
        legacy = self.config.model.startswith(_LEGACY_PREFIXES)
        output_config: Dict[str, Any] = {}
        if not legacy:
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.config.effort:
                output_config["effort"] = self.config.effort
        if structured and task.json_schema:
            output_config["format"] = {"type": "json_schema", "schema": dict(task.json_schema)}
        if output_config:
            body["output_config"] = output_config
        return body

    # calls
    def run(self, task: Task) -> Completion:
        repair: Optional[str] = None
        structured = True
        last_problem = "no response"
        for attempt in range(3):
            body = self.body(task, repair=repair, structured=structured)
            try:
                raw = self._post(body)
            except BackendError as exc:
                # if the endpoint refuses the constrained output request, retry once as a
                # plain request instead of failing
                if structured and exc.status == 400 and "output_config" in str(exc):
                    structured = False
                    continue
                raise
            text = raw["content"]
            data = parse_structured(text) or {}
            ok, problem = validate(data, task.schema) if data else (False, "no JSON object found")
            if ok:
                return Completion(
                    text=text,
                    reasoning=raw["reasoning"],
                    data=data,
                    model=raw.get("model") or self.config.model,
                    backend=self.name,
                    prompt_tokens=raw.get("prompt_tokens", 0),
                    completion_tokens=raw.get("completion_tokens", 0),
                )
            last_problem = problem
            repair = problem
            if attempt == 1:
                break
        raise BackendError(f"{self.config.model} returned unusable output: {last_problem}")

    def _post(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        url = self.config.base_url.rstrip("/")
        if not url.endswith("/v1/messages"):
            url = url + ("/messages" if url.endswith("/v1") else "/v1/messages")
        key = self.config.resolved_key("claude")
        if not key:
            raise BackendError("no API key for the Claude backend (set ANTHROPIC_API_KEY)")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": key,
            "anthropic-version": API_VERSION,
            "User-Agent": _USER_AGENT,
        }
        data = json.dumps(body).encode("utf-8")

        delay = 1.5
        last: Optional[BackendError] = None
        for attempt in range(max(1, self.config.max_retries)):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _unpack(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600] if hasattr(exc, "read") else ""
                last = BackendError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code)
                if exc.code not in _RETRYABLE:
                    break
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                if retry_after:
                    try:
                        delay = min(max(delay, float(retry_after) + 0.5), self.config.max_wait)
                    except ValueError:
                        pass
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = BackendError(f"transport error contacting {url}: {exc}")
            except json.JSONDecodeError as exc:
                last = BackendError(f"malformed JSON from {url}: {exc}")
            if attempt + 1 < self.config.max_retries:
                time.sleep(delay)
                delay = min(delay * 2.0, self.config.max_wait)
        raise last or BackendError("request failed for an unknown reason")


def _unpack(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise a Messages API response envelope."""
    stop = payload.get("stop_reason")
    if stop == "refusal":
        details = payload.get("stop_details") or {}
        raise BackendError(
            "the model declined the request"
            + (f" ({details.get('category')})" if details.get("category") else "")
        )
    texts: List[str] = []
    thoughts: List[str] = []
    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            texts.append(block.get("text") or "")
        elif kind == "thinking":
            thought = block.get("thinking") or block.get("summary") or ""
            if thought:
                thoughts.append(thought)
    if not texts:
        raise BackendError(f"response contained no text block (stop_reason={stop})")
    if stop == "max_tokens":
        raise BackendError("response was truncated at max_tokens")
    usage = payload.get("usage") or {}
    return {
        "content": "\n".join(texts),
        "reasoning": "\n".join(thoughts),
        "model": payload.get("model") or "",
        "prompt_tokens": int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
    }
