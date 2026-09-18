"""Run the live model studies to completion on a rate limited free tier.

Each job is a dmas command that writes json. A run is complete when the
json says degraded_rows == 0 (every row answered by the live backend, not
the offline stand in). Completions are cached by content hash so rerunning
only pays for the missing rows, the loop just retries until the report is
clean and pauses when the endpoint is exhausted for the day.

    python3 experiments/live_runner.py jobs.txt

jobs.txt: one job per line, '<output.json> <dmas arguments...>', lines
starting with # are ignored. The key is read from OPENAI_API_KEY, never
printed or stored.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAUSE_ON_EXHAUSTION = 20 * 60          # seconds between attempts once a run degrades
MAX_ROUNDS = 400


def run_job(output: Path, arguments: list) -> bool:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, "-m", "deductive_mas", *arguments, "--json", "--color", "never"]
    started = time.time()
    proc = subprocess.run(command, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! {output.name}: exit {proc.returncode}: {proc.stderr[-400:]}", flush=True)
        return False
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  ! {output.name}: not JSON: {proc.stdout[:200]}", flush=True)
        return False
    degraded = int(payload.get("degraded_rows", 0))
    output.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"  {output.name}: degraded_rows={degraded} ({time.time() - started:.0f}s)", flush=True)
    return degraded == 0


def main(path: str) -> int:
    jobs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        jobs.append((ROOT / "experiments" / "results" / parts[0], parts[1:]))
    pending = list(jobs)
    for round_index in range(MAX_ROUNDS):
        print(f"round {round_index + 1}: {len(pending)} job(s) pending", flush=True)
        still = []
        for output, arguments in pending:
            if not run_job(output, arguments):
                still.append((output, arguments))
        pending = still
        if not pending:
            print("all jobs complete", flush=True)
            return 0
        print(f"sleeping {PAUSE_ON_EXHAUSTION // 60} min before retrying", flush=True)
        time.sleep(PAUSE_ON_EXHAUSTION)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
