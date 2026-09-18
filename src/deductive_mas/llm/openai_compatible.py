"""OpenAI-compatible chat completions backend over urllib.

Lets the same tasks run against any endpoint speaking the OpenAI protocol,
e.g. a local open weight model (--backend openai --base-url
http://localhost:11434/v1 --model <name>). No SDK.

For reasoning models: some return the chain of thought in a separate
reasoning_content field, kept for the audit trail and never shown to the
student. They are also not strict about output format, so a response that
fails validation is retried once with a repair instruction.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..config import LLMConfig
from ..errors import BackendError
from ..version import __version__
from .base import Completion, Message, Task, validate
from .parsing import parse_structured, split_reasoning

_USER_AGENT = f"dmas/{__version__} (python-urllib)"

_JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. Do not wrap it in prose. "
    "Required keys: {keys}."
)

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed as the required JSON object ({problem}). "
    "Reply again with only the JSON object."
)


class OpenAICompatibleBackend:
    """Client for any OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._last_error: Optional[str] = None

    # liveness
    def available(self) -> bool:
        """True if there is an api key or the endpoint is a local server."""
        if self.config.resolved_key("openai"):
            return True
        host = self.config.base_url.lower()
        return "localhost" in host or "127.0.0.1" in host or "0.0.0.0" in host

    # rendering
    def render(self, task: Task, repair: Optional[str] = None) -> List[Message]:
        keys = ", ".join(sorted(task.schema)) or "as described"
        payload = json.dumps(task.payload, ensure_ascii=False, indent=2, default=str, sort_keys=True)
        user = (
            f"{task.instruction}\n\n"
            f"### Structured input\n```json\n{payload}\n```\n\n"
            f"{_JSON_INSTRUCTION.format(keys=keys)}"
        )
        messages = [Message("system", task.system), Message("user", user)]
        if repair:
            messages.append(Message("user", _REPAIR_INSTRUCTION.format(problem=repair)))
        return messages

    # calls
    def run(self, task: Task) -> Completion:
        repair: Optional[str] = None
        structured = True
        last_problem = "no response"
        for attempt in range(3):
            messages = self.render(task, repair=repair)
            try:
                raw = self._post(messages, task, structured=structured)
            except BackendError as exc:
                # an endpoint that rejects constrained output or the reasoning control is
                # retried once as a plain request
                if structured and exc.status == 400:
                    structured = False
                    continue
                raise
            reasoning, answer = split_reasoning(raw.get("content", ""))
            reasoning = raw.get("reasoning") or reasoning
            data = parse_structured(answer) or {}
            ok, problem = validate(data, task.schema) if data else (False, "no JSON object found")
            if ok:
                return Completion(
                    text=answer,
                    reasoning=reasoning,
                    data=data,
                    model=raw.get("model") or self.config.model,
                    backend=self.name,
                    prompt_tokens=raw.get("prompt_tokens", 0),
                    completion_tokens=raw.get("completion_tokens", 0),
                )
            last_problem = problem
            repair = problem
            if attempt >= 1:
                break
        raise BackendError(f"{self.config.model} returned unusable output: {last_problem}")

    def _post(self, messages: List[Message], task: Task, *, structured: bool = True) -> Dict[str, Any]:
        url = self.config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")

        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": task.temperature if task.temperature is not None else self.config.temperature,
            "max_tokens": task.max_tokens or self.config.max_tokens,
            "stream": False,
        }
        if structured:
            if self.config.effort:
                body["reasoning_effort"] = self.config.effort
            if task.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": task.name, "strict": True, "schema": dict(task.json_schema)},
                }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # some gateways reject the default urllib user agent
            "User-Agent": _USER_AGENT,
        }
        key = self.config.resolved_key("openai")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        delay = 1.0
        last: Optional[BackendError] = None
        for attempt in range(max(1, self.config.max_retries)):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _unpack(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400] if hasattr(exc, "read") else ""
                last = BackendError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code)
                if exc.code in (400, 401, 403, 404, 413, 422):
                    break                                    # not worth retrying
                # a rate limited endpoint says how long to wait, use that (within reason)
                # instead of the generic backoff
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
                delay = min(delay * 2.0, self.config.max_wait)  # exponential backoff
        raise last or BackendError("request failed for an unknown reason")


def _unpack(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise an OpenAI style response envelope."""
    choices = payload.get("choices") or []
    if not choices:
        raise BackendError("response contained no choices")
    message = choices[0].get("message") or {}
    usage = payload.get("usage") or {}
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }
