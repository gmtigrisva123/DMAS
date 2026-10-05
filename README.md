<div align="center">

# DMAS

**A deductive multi-agent system for automated cognitive misconception
diagnosis and Socratic intervention in advanced computer science education**

*Diagnosing the belief, not the bug.*

`Python 3.9+` · zero dependencies · `MIT`

</div>

---

Contemporary AI tutors operate in a *direct response* paradigm: a student brings
a problem, the model returns the solution. That is efficient and pedagogically
inert. In cumulative disciplines a correct answer handed over does not repair the
belief that produced the wrong one, so the same cognitive error reappears in the
next, harder problem.

This system inverts the use of the model's reasoning. Instead of pointing it at
the *problem*, it points it at the *student*: three agents collaborate to find
the exact instant a learner's reasoning left the correct path, locate the
prerequisite concept that made that misstep possible, and ask the question that
exposes the contradiction — without ever revealing the answer.

```bash
./dmas session --file attempt.py
```

Runs immediately from a fresh checkout. No install, no API key, no network, no
third-party packages.

## The session

The tutor is interactive, because Socratic teaching is. It reveals **one rung at
a time** — a ladder printed all at once is just a worked answer delivered slowly
— and it remembers you between attempts, so after you edit the file it can
answer the question that matters: was the belief repaired, or the symptom patched?

```
dmas ▸ load attempt.py

  loaded  attempt.py  ·  10 lines  ·  task  lower_bound

   DIAGNOSIS   Mixed interval conventions  ·  0.77  ·  L4
  5/7 inputs pass

dmas ▸ hint

  hint 1/3  ·  orienting  ·  Loop invariants

  Take the input ([0], 1) and step through your own function by hand, writing
  down the values at line 4 each time you reach it. Stop at the first moment a
  value surprises you.

  sit with that one first — `hint` again when you are stuck

dmas ▸ hint

  hint 2/3  ·  contradiction  ·  Loop invariants

  At that point your `hi` holds 0, but for the answer to be reachable it would
  have to be 1. Which line last wrote `hi`, and what did you assume about the
  range it was allowed to take?

      … the student edits the file in their editor …

dmas ▸ retry

   NO FINDING   87/87 inputs pass; no structural warning sign either

── Attempt 2 vs 1 ─────────────────────────────────────────────────────────

  resolved  Mixed interval conventions
  checks    5/7 inputs pass → 87/87 inputs pass
  mastery   Loop invariants  0.39 ↑ 0.72
```

A partial fix that makes things worse is reported just as plainly — `persists`,
`new`, checks 5/7 → 0/7 — which is the honest thing and the useful thing.

| command | |
|---|---|
| `load <file>` | read a file; the task is detected from the function it defines, or falls back to structural findings for a file the tutor has no reference for |
| `hint` | the next rung, and only the next rung |
| `challenge` | a predict-then-run task built from the minimal failing input |
| `why` · `code` · `report` | the evidence, the marked source, the full report |
| `retry` | re-read the file after you edit it, and diff the two attempts |
| `mastery` · `progress` | what the session believes you have mastered; every attempt so far |
| `concept [<id>]` · `problems` · `problem <id>` | explore the curriculum; pin a task |

Tab completion and history work in a terminal; commands can also be piped in,
which is how the tests drive it.

---

## What it actually does

Given a flawed submission, the system produces this:

```
── Cognitive diagnosis ──────────────────────────────────────────────────────

  misconception                    name                        belief
  bs.interval-convention-mismatch  Mixed interval conventions  █████████░░░ 0.77  MAJOR  L4

  The flawed belief: "I think the upper bound is just 'the end of the array'
  — which end does not matter."

  why:
    • [dynamic] execution first diverges at line 4, the same line this rule flags
    • [static]  the guard `lo < hi` treats `hi` as an exclusive bound, but it is
                initialised to `len(a) - 1`, the last valid index

── Knowledge-graph alignment ────────────────────────────────────────────────

  root cause  Loop invariants  ·  loop-invariant, depth 4
  the deepest implicated concept whose own prerequisites are already secure
  learning path  Loop invariants → Half-open intervals → Boundary conditions

── Cognitive intervention ───────────────────────────────────────────────────

  target  Loop invariants  ·  success probability  61%  ·   IN ZPD

  1 · orienting
     Take the input ([0], 1) and step through your own function by hand,
     writing down the values at line 4 each time you reach it.

  2 · contradiction
     At that point your `hi` holds 0, but for the answer to be reachable it
     would have to be 1. Which line last wrote `hi`, and what did you assume
     about the range it was allowed to take?

  3 · conceptual
     Write down the property you believe holds at the top of every iteration.
     Does it still hold immediately after your update lines run?

  answer leakage  0.00  ·  grounding  0.55
```

Note what is *not* there: the fix. `hi = len(a)` never appears, and the gate that
guarantees that is a mechanism, not an instruction — see below.

## The three agents

**Socratic Evaluator.** Executes the submission and a reference implementation on
the same minimised failing input, matches their variables **by behaviour rather
than by name**, and reports the earliest checkpoint at which the two disagree.
Symbolic rules, trace divergence, cost analysis and the reasoning model are fused
with Dempster–Shafer belief functions — so agreement between independent evidence
types compounds, and disagreement is surfaced as *contested* rather than averaged
away. A leading hypothesis below the settlement threshold is labelled
**provisional** rather than presented as a verdict; on the development bank 16 of
23 diagnoses are settled and 7 are flagged provisional.

**Knowledge-Graph Alignment.** Propagates blame through a 65-concept prerequisite
DAG by personalised PageRank, folds it into a Bayesian knowledge-tracing model as
soft negative evidence, and extracts the **deficiency frontier**: the unmastered
concepts whose own prerequisites are already secure. Those are the only places
where teaching can succeed — Vygotsky's zone of proximal development, read off
the graph.

**Cognitive Intervention.** Selects a target inside that zone by item response
theory, grounds its language in retrieved evidence cards, and emits a three-rung
Socratic ladder plus a counter-factual challenge — *predict what your function
returns, then run it*.

## The guarantee

> **No guidance reaches a student without passing the answer-leakage gate.**

Enforced by replacement, not by warning. Three independent detection channels:
MOSS-style winnowing fingerprints catch a copied passage; AST structural matching
catches a statement of the reference the student has not written; edit-proximity
catches a *modified* version of a line the student has written. Prose repairs
("set `hi` to `len(a)`") are normalised into assignments first, because a model
told not to write code reaches for that phrasing immediately.

Quoting the student verbatim is explicitly allowed. Making someone look at what
they actually wrote is the method.

The test suite includes a backend that deliberately emits the full reference
solution; the assertion is that the student never sees it, and that the guidance
they see instead is still three usable rungs.

## Results

Every number below is regenerated by the commands in `experiments/` and quoted in
the paper through generated macros, so the README, the JSON and the paper cannot
disagree. The reasoning backends for the "+ model" rows are `openai/gpt-oss-120b`
(the paper's primary model) and `openai/gpt-oss-20b` (a comparison), open-weight
reasoning models served through an OpenAI-compatible endpoint; the other rows use
the deterministic offline generator and run with no network.

### RQ1 — diagnostic accuracy

Two labelled banks. The **development bank** (23 labelled + 6 correct) was built
alongside the rules and establishes internal validity only. The **held-out bank**
(25 labelled + 6 correct, six new tasks) was written after the rules were
frozen and labelled before the system ran; no rule or threshold was changed
afterwards (the protocol is stated in `data/heldout.py`).

| bank | system | top-1 | recall | micro-F1 | κ | FPR on clean code |
|---|---|---|---|---|---|---|
| development | deterministic | 1.00 | 1.00 | 0.81 | 1.00 | 0.00 |
| development | + gpt-oss-120b | 1.00 | 1.00 | 0.81 | 1.00 | 0.00 |
| development | + gpt-oss-20b | 0.96 | 1.00 | 0.79 | 0.95 | 0.00 |
| development | gpt-oss-120b only | 0.87 | 0.91 | 0.83 | 0.86 | 0.00 |
| held-out | deterministic | 0.76 | 0.76 | 0.71 | 0.74 | 0.00 |
| held-out | + gpt-oss-120b | 0.88 | 0.92 | 0.78 | 0.87 | 0.00 |
| held-out | + gpt-oss-20b | 0.96 | 1.00 | 0.82 | 0.96 | 0.17 |
| held-out | gpt-oss-120b only | 0.80 | 0.88 | 0.71 | 0.79 | 0.00 |

The six held-out misses are listed and explained in the paper's appendix: four
are symptom patterns the rules do not cover (the failing input is found and
localised but no rule names the belief), one is a wrong rather than missing base
case, one is recursion inside a nested helper. Fusing `gpt-oss-120b` closes three
of the six without adding a false positive; it proposed no belief on any of the
twelve clean programs. The smaller `gpt-oss-20b` names more (top-1 0.96 held-out)
but also accuses one correct program of an asymptotic gap, which the reliability
discount (0.62) and the clean run's opposing mass reduce to a provisional 0.37
without silencing — the case for leaving the verdict on clean code to the
executional sources.

### External validation — 4,225 real student programs

The Refactory corpus (Hu et al., ASE 2019) provides programs written by 361
students for five introductory assignments, each marked correct or wrong by the
course's tests. Those labels say nothing about *why*, so the corpus cannot score
diagnostic accuracy; it can score false accusations and detection.

| measure | value |
|---|---|
| course-accepted programs on which a *correctness* belief was named | 4.5% |
| … with only a cost or hygiene finding | 31.2% |
| course-rejected programs on which a failing input was found | 96.5% |
| … localised to a line | 99.6% |
| … on which a correctness belief was named | 50.3% |
| latency per file | 37 ms |

Of the rejected programs with no failing input, 59 use the built-in sort in an
assignment that forbade it. Of the accepted programs accused of a correctness
belief, 51 fail on an input inside the course's contract that the course's own
tests did not cover.

### The gate

On 274 manufactured leaks (a reference line verbatim or renamed, a prose repair
around a code span, the whole solution) the gate rejects
99.3%; on 416 legitimate items (the student's own line
quoted back, a question, a fallback rung) it rejects 0.0%. A repair
stated in plain words with no code span is beyond the gate's design; the
grounding channel happens to catch 50% of those, and the paper says so.

### RQ2 — design analysis with simulated learners

**These are simulated learners.** The response model encodes the scaffolding
assumption under test, so any single run reproduces it. What the simulation
establishes is narrower: under a null response model in which the arms cannot
differ, the family-wise rejection rate over 200 cohorts is
1.0% (the statistics do not manufacture an effect); under the
default model the retention effect is Hedges g = 0.56
[0.28, 0.73] and the recurring-error effect g = -0.59
[-0.89, -0.34] over 30 cohorts; and the one quantity that does not depend
on the response model — the **on-target rate**, the share of interventions aimed
at a concept the learner actually lacks — is 93% for the full system and
39% without the knowledge graph. Power curves for sizing a human study are
in `experiments/results/rq2_design_offline.json`.

### RQ3 — robustness

Semantics-preserving rewrites (rename every identifier, split tuple assignments,
expand augmented assignments, invert conditionals, inject dead code, all five at
once), each mutant verified behaviourally identical before scoring. Top-1
stability: 0.942 over 138 development mutants,
1.000 over 150 held-out mutants. Renaming costs nothing, because
variables are matched by behaviour.

## Install and use

```bash
git clone <repository> && cd deductive_multi_agent_system_\(MAS\)
./dmas doctor                      # verify the environment
./dmas session --file attempt.py   # the interactive tutor
./dmas demo                        # a non-interactive guided tour
```

`./dmas` is a two-line launcher that puts `src/` on the import path, so a fresh
checkout works with no install step. `pip install -e .` puts an equivalent `dmas`
command on your PATH; every example below works with either.

```bash
dmas session                                       # interactive; `load` a file inside
dmas diagnose --submission lb_inclusive_bound      # one-shot report, from the built-in bank
dmas diagnose --file attempt.py --problem lower_bound
dmas kg show binary-search                         # inspect the curriculum
dmas kg path binary-search sequence-indexing
dmas retrieve "why does my BFS revisit nodes" --full
dmas bench --json > rq1.json                       # the three studies
dmas experiment --students 180 --json > rq2.json
dmas robustness --json > rq3.json
```

Every command takes `--json`, and every command is deterministic given `--seed`.

### With a reasoning model

The paper's live results use `openai/gpt-oss-120b`, an open-weight reasoning
model, through Groq's OpenAI-compatible endpoint. That is the default, so with
a key in the environment the flags below are optional:

```bash
export OPENAI_API_KEY=...
dmas diagnose --submission mc_greedy --backend openai \
  --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b
```

A locally served model works the same way (`--base-url http://localhost:11434/v1
--model <name>`), and a Claude backend is available (`--backend claude --model
claude-opus-5`, reads `ANTHROPIC_API_KEY`). The backend is a swappable component behind a typed task
interface with a JSON schema per task, so running the offline generator and a
model and diffing the reports *is* the ablation that measures what the model
contributes. A model's chain of thought is captured for the audit trail and never
shown to a student — reasoning traces routinely contain the answer, which is
exactly what this system must not disclose. Every completion is cached by
content hash (keyed by backend and model), so a study can be resumed and
re-derived without network access.

### As a library

```python
# PYTHONPATH=src, or after `pip install -e .`
from deductive_mas import DeductiveOrchestrator, problem, submission

result = DeductiveOrchestrator().tutor(
    problem("lower_bound"), submission("lb_inclusive_bound")
)

print(result.diagnosis.primary.misconception_id)   # bs.interval-convention-mismatch
print(result.alignment.root_cause)                 # loop-invariant
for hint in result.intervention.hints:
    print(hint.level, hint.text)
assert result.intervention.leakage < 0.18          # the gate held
```

## Why no dependencies

Every numerical, retrieval and statistical routine is implemented here from first
principles: one-sided Jacobi SVD, randomised range finding, the Hungarian
algorithm, dynamic time warping, Okapi BM25, reciprocal rank fusion, winnowing
fingerprints, Dempster's rule, the regularised incomplete beta and gamma
functions, BCa bootstrap intervals.

That is a deliberate scientific choice, not asceticism. Every p-value in the
results section traces to code in this repository, the statistics are validated
against published reference tables in `tests/test_stats.py`, and the whole system
runs identically on any machine with a Python 3.9 interpreter — which is the
deployment setting the proposal targets.

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | layering, the blackboard, evidence flow, the compliance gate |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | every algorithm, why it was chosen, and the subtleties that bit |
| [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) | how each RQ is measured, and what the results do and do not establish |
| [`docs/PROPOSAL_TRACEABILITY.md`](docs/PROPOSAL_TRACEABILITY.md) | claim-by-claim mapping to the proposal, including the deviations |

## Limitations

Stated plainly, because a system that diagnoses overconfidence should not exhibit
it.

1. **Diagnostic accuracy is measured on a bank developed alongside the rules.**
   Internal validity only.
2. **The RQ2 cohort is simulated.** No claim is made about human learning.
3. **Grounding is support, not entailment.** It catches invented vocabulary and
   off-corpus claims; a fluent, on-topic, subtly false sentence the corpus does
   not contradict can pass.
4. **The guarded interpreter is not a security sandbox.** Three independent
   limits, appropriate for classroom submissions, unsuitable for adversarial code.
5. **Cost inference is an upper bound**, deliberately over-approximating
   data-dependent loops; every complexity claim shown to a student is corroborated
   dynamically first.
6. **Cross-algorithm comparison is coarse.** When a submission attacks the
   problem by a different method than the reference — a greedy scan against a
   dynamic-programming solution — the variable correspondence is weak by
   construction. Roles below a confidence bar are dropped and the divergence
   falls back to the output level, but the resulting contradiction is less
   pointed than it is for two implementations of the same algorithm.
7. **Python only.** The analysis layer is AST-specific; the knowledge, retrieval,
   belief and statistics layers are not.
8. **Files with no reference get structural findings only.** Differential
   testing, trace alignment and the complexity comparison all need an oracle;
   for an unknown task the session says so and reports what the rules, the
   dataflow facts and the cost analysis can still establish.

## Layout

```
dmas                                  run the CLI from a checkout, no install
src/deductive_mas/
  domain.py  config.py  pipeline.py     shared vocabulary, settings, composition
  util/          determinism, tokenisation, linear algebra
  analysis/      CFG, dataflow, cost, tracing, alignment, shrinking, leakage
  knowledge/     ontology, misconception taxonomy, mastery, ZPD
  retrieval/     BM25, LSA, fusion, corpus, MA-RAG
  llm/           Claude, OpenAI-compatible endpoints, offline reasoner, caching
  agents/        the three agents, the gate, the orchestrator
  simulation/    RQ1 benchmark, RQ2 experiment + design analysis, RQ3 robustness,
                 gate benchmark and audit, external validation (Refactory)
  stats/         special functions, tests, effect sizes, power, agreement
  ui/  cli/      terminal presentation, the dmas command, the interactive session
tests/           the test suite
experiments/     result files, figure and number scripts, the live-run driver
docs/
```

## Licence

MIT.
