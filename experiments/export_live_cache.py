"""Copy the live model completions from .dmas_cache into experiments/live_cache
(which ships with the artefact).

    python3 experiments/export_live_cache.py

With --cache-dir experiments/live_cache (and any value in OPENAI_API_KEY,
the backend must be configured even if every request comes from the cache)
the live studies rerun without network. Offline generator entries are not
exported, they are regenerated in milliseconds anyway.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".dmas_cache"
TARGET = ROOT / "experiments" / "live_cache"


def main():
    TARGET.mkdir(exist_ok=True)
    copied = 0
    for path in SOURCE.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") in ("openai", "claude"):
            shutil.copy2(path, TARGET / path.name)
            copied += 1
    print(f"exported {copied} live completions to {TARGET}")


if __name__ == "__main__":
    main()
