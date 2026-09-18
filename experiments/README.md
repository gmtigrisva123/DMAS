# Experiments

Everything the paper reports is produced by the `dmas` command line, written to
`results/` as JSON, and turned into figures and LaTeX macros by two scripts. No
number in the paper is typed by hand.

| study | command | output |
|---|---|---|
| RQ1, development bank | `dmas bench --bank dev --no-narrative --json` | `results/rq1_dev_full_offline.json` |
| RQ1, held-out bank | `dmas bench --bank heldout --no-narrative --json` | `results/rq1_heldout_full_offline.json` |
| RQ1 with a reasoning model | `dmas bench --bank <bank> --backend openai --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b --no-narrative --json` | `results/rq1_<bank>_full_gpt-oss-120b.json` |
| model-only baseline | add `--model-only` to the line above | `results/rq1_<bank>_modelonly_gpt-oss-120b.json` |
| evidence-source ablation | `dmas bench --bank <bank> --sources symbolic,cost ...` | `results/rq1_<bank>_sources_*.json` |
| fusion-rule ablation | `dmas bench --bank <bank> --fusion max|mean|noisy-or` | `results/rq1_<bank>_fusion_*.json` |
| external validation | `dmas refactory --data <refactory>/data --json` | `results/refactory_full_offline.json` |
| gate, adversarial benchmark | `dmas gate adversarial --json` | `results/gate_adversarial_offline.json` |
| gate, live audit | `dmas gate audit [--backend openai ...] --json` | `results/gate_audit_*.json` |
| RQ2, one cohort | `dmas experiment --students 180 --json` | `results/rq2_experiment_offline.json` |
| RQ2, design analysis | `dmas design --seeds 30 --null-seeds 200 --power-seeds 30 --json` | `results/rq2_design_offline.json` |
| RQ3, robustness | `dmas robustness --bank dev|heldout --json` | `results/rq3_robustness_*.json` |

Then:

```bash
python3 experiments/make_numbers.py      # -> paper/numbers.tex (LaTeX macros)
python3 experiments/make_tables.py       # -> paper/tables.tex (appendix tables)
python3 experiments/make_figures.py      # -> experiments/figures/*.pdf (needs matplotlib)
```

The external corpus is the Refactory dataset (Hu et al., ASE 2019,
https://github.com/githubhuyang/refactory, LGPL-3.0); unpack its `data.zip` and
point `--data` at the resulting `data` directory. It is not redistributed here.

`live_runner.py` drives the live-model studies to completion under a
rate-limited endpoint: a job is complete only when its JSON reports
`degraded_rows == 0`, and because every completion is cached by content hash
(`.dmas_cache/`, keyed by backend, model and reasoning effort), re-running a
job only pays for the rows that are still missing.
