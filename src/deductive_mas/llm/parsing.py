"""Get structured data out of model output.

Reasoning models mix chain of thought with the answer and are sloppy with
json. So treat the output as untrusted text: split off the reasoning, find a
balanced json object by bracket matching (regex cannot do nesting) and
repair the few things that actually go wrong (code fences, trailing commas,
single quotes, python literals). If it still does not parse it is a backend
failure and we fall back to the offline reasoner.
"""

import ast
import json
import re
from typing import Any, Dict, Optional, Tuple

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|python)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def split_reasoning(text: str) -> Tuple[str, str]:
    """Split an explicit chain of thought from the answer."""
    reasoning_parts = _THINK_RE.findall(text or "")
    answer = _THINK_RE.sub("", text or "").strip()
    return "\n".join(part.strip() for part in reasoning_parts).strip(), answer


def find_json_object(text: str) -> Optional[str]:
    """First balanced {...} span, respecting strings and escapes."""
    if not text:
        return None
    depth = 0
    start = -1
    in_string = False
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in "\"'":
            in_string = True
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : index + 1]
    return None


def repair_json(blob: str) -> str:
    """The small safe repairs that cover what models actually do."""
    cleaned = blob.strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    return cleaned


def parse_structured(text: str) -> Optional[Dict[str, Any]]:
    """Best effort recovery of a json object from model output."""
    if not text:
        return None
    candidates = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    blob = find_json_object(text)
    if blob:
        candidates.append(blob)
    candidates.append(text)

    for candidate in candidates:
        for attempt in (candidate, repair_json(candidate)):
            attempt = attempt.strip()
            if not attempt:
                continue
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(attempt)
                except (ValueError, SyntaxError, MemoryError, RecursionError):
                    continue
            if isinstance(parsed, dict):
                return parsed
    return None


def clamp_unit(value: Any, default: float = 0.0) -> float:
    """Force a model supplied confidence into [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(1.0, number))


def as_str_list(value: Any, limit: int = 8) -> list:
    """Force a model supplied field into a list of non empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "hint", "value", "name", "id"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        out.append(item[key].strip())
                        break
            if len(out) >= limit:
                break
        return out
    return []
