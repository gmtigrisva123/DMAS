"""On disk cache for backend responses, keyed by content hash.
Re-running the same session must give the same guidance and we should not
pay twice for the same request. The key hashes everything that can change
the response (model, temperature, task, payload).
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..util.determinism import stable_hash
from .base import Completion, Task


class ResponseCache:
    """Flat directory of json blobs keyed by a 64 bit hash."""

    def __init__(self, directory: Optional[str], *, enabled: bool = True):
        self.enabled = bool(enabled and directory)
        self.directory = Path(directory) if directory else None
        if self.enabled and self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError:                      # read only environment
                self.enabled = False

    def key(self, task: Task, model: str, temperature: float) -> str:
        return "%016x" % stable_hash(model, round(temperature, 4), *task.cache_key_parts())

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[Completion]:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        completion = Completion(**{k: v for k, v in payload.items() if k in Completion.__annotations__})
        completion.cached = True
        return completion

    def put(self, key: str, completion: Completion):
        if not self.enabled:
            return
        path = self._path(key)
        payload = asdict(completion)
        payload["cached"] = False
        try:
            # atomic replace so a crash mid write cannot leave a corrupt entry
            handle = tempfile.NamedTemporaryFile(
                "w", dir=str(self.directory), delete=False, encoding="utf-8", suffix=".tmp"
            )
            with handle as stream:
                json.dump(payload, stream, ensure_ascii=False)
            os.replace(handle.name, path)
        except OSError:
            return

    def clear(self) -> int:
        if not self.enabled or self.directory is None:
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed
