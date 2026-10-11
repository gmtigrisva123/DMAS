# Sensitivity of the diagnosis to the author-set constants

One-at-a-time sweep over `config.py` values (all other constants at their defaults). Run with `run_sweep.py`; raw rows in `sweep_results.csv`. Base values are in bold.

Caveat: nothing here re-tunes the defaults. The held-out columns are post hoc sensitivity checks run after the held-out bank was frozen; they show how fragile each default is, not a better value.

| constant | values tried | dev top-1 | held-out top-1 | held-out micro-F1 | held-out FPR |
|---|---|---|---|---|---|
| static_reliability | 0.7, 0.78, **0.86**, 0.94, 0.99 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.76 / 0.76 / 0.76 / 0.76 / 0.76 | 0.724 / 0.724 / 0.712 / 0.712 / 0.712 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| dynamic_reliability | 0.8, 0.86, **0.92**, 0.96, 0.99 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.76 / 0.76 / 0.76 / 0.76 / 0.76 | 0.712 / 0.712 / 0.712 / 0.712 / 0.712 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| llm_reliability | 0.3, 0.45, **0.62**, 0.8, 0.95 | 0.87 / 0.87 / 1.00 / 1.00 / 1.00 | 0.72 / 0.72 / 0.76 / 0.76 / 0.76 | 0.690 / 0.690 / 0.712 / 0.712 / 0.712 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| report_threshold | 0.1, 0.15, **0.22**, 0.3, 0.4 | 1.00 / 1.00 / 1.00 / 0.83 / 0.70 | 0.76 / 0.76 / 0.76 / 0.64 / 0.44 | 0.689 / 0.712 / 0.712 / 0.654 / 0.500 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| primary_threshold | 0.25, 0.32, **0.4**, 0.5, 0.6 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.76 / 0.76 / 0.76 / 0.76 / 0.76 | 0.712 / 0.712 / 0.712 / 0.712 / 0.712 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| dynamic_refutation | 0.35, 0.45, **0.55**, 0.65, 0.75 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.76 / 0.76 / 0.76 / 0.76 / 0.72 | 0.712 / 0.712 / 0.712 / 0.712 / 0.690 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |

Model-source reliability, with the cached gpt-oss-120b replay (held-out, top-1 at 0.30 / 0.45 / 0.62 / 0.80 / 0.95):

0.88 / 0.88 / 0.88 / 0.88 / 0.84

Held-out cases with one degraded row (`cc_edge_scan`, a cache miss) are included as-is in the replay numbers.

## Reading

- Static and dynamic reliabilities: no change in top-1 across 0.70–0.99 (dev and held-out). We observed no change in top-1; we did not test why.
- `primary_threshold`: no change across 0.25–0.60. It only labels a diagnosis as settled.
- `report_threshold` is the sensitive one. Held-out top-1 is flat for 0.10–0.22, then falls to 0.64 at 0.30 and 0.44 at 0.40. The default 0.22 is at the edge of the flat region, so the paper should say so and report this table.
- `llm_reliability` matters only through the model source (offline, that is the greedy heuristic; with the model replayed, the reasoning model). Held-out with the replayed gpt-oss-120b: 0.88 at 0.30–0.80, 0.84 at 0.95.
- `dynamic_refutation`: unchanged on dev; on held-out it only changes at 0.75 (0.72).
