"""One-at-a-time sensitivity sweep over the author-set constants (config.py).
Offline backend on both banks; the model-source reliability is also run with the
cached gpt-oss-120b replay (the cache is keyed by request, so changing a reliability
does not change any request and every completion is a cache hit)."""
import csv, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CFG = HERE / "configs"; CFG.mkdir(exist_ok=True)
OUT = HERE / "sweep_results.csv"

GRID = {  # name -> (base value, values to try)
    "static_reliability":  (0.86, [0.70, 0.78, 0.86, 0.94, 0.99]),
    "dynamic_reliability": (0.92, [0.80, 0.86, 0.92, 0.96, 0.99]),
    "llm_reliability":     (0.62, [0.30, 0.45, 0.62, 0.80, 0.95]),
    "report_threshold":    (0.22, [0.10, 0.15, 0.22, 0.30, 0.40]),
    "primary_threshold":   (0.40, [0.25, 0.32, 0.40, 0.50, 0.60]),
    "dynamic_refutation":  (0.55, [0.35, 0.45, 0.55, 0.65, 0.75]),
}
BANKS = ["dev", "heldout"]
COLS = ["param", "value", "backend", "bank", "top1", "micro_f1", "kappa", "fpr", "localised", "degraded"]

def run(param, value, bank, backend):
    cfg = CFG / f"{param}_{value}.json"
    cfg.write_text(json.dumps({"diagnosis": {param: value}}))
    cmd = [sys.executable, "-m", "deductive_mas", "bench", "--bank", bank,
           "--config", str(cfg), "--no-narrative", "--json"]
    if backend == "offline":
        cmd += ["--backend", "offline", "--no-cache"]
    else:
        cmd += ["--backend", "openai", "--base-url", "https://api.groq.com/openai/v1",
                "--model", "openai/gpt-oss-120b", "--cache-dir", str(REPO / "experiments/live_cache")]
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(REPO / "src"))
    if backend != "offline":
        env["OPENAI_API_KEY"] = "replay-placeholder"
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"{param}={value} {bank} {backend}: {p.stderr[-400:]}")
    d = json.loads(p.stdout)
    return {"param": param, "value": value, "backend": backend, "bank": bank,
            "top1": d["top1_accuracy"], "micro_f1": d["micro_f1"], "kappa": d["cohen_kappa"],
            "fpr": d["false_positive_rate"], "localised": d["divergence_localisation_rate"],
            "degraded": d["degraded_rows"]}

rows = []
for param, (base, values) in GRID.items():
    for value in values:
        for bank in BANKS:
            rows.append(run(param, value, bank, "offline"))
        if param == "llm_reliability":
            for bank in BANKS:
                rows.append(run(param, value, bank, "gpt-oss-120b"))
        print(param, value, "done", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print("wrote", OUT, len(rows), "rows")
