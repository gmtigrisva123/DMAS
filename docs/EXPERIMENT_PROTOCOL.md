# Experimental protocol

This document specifies how the three research questions are measured, what the
current results establish, and — importantly — what they do not.

## RQ1 — Diagnostic accuracy

> To what extent can the Socratic Evaluator accurately identify and categorise
> semantic/logical misconceptions in student code compared to expert human
> assessments?

**Instrument.** `dmas bench` scores the evaluator against the gold labels in
`deductive_mas/data/submissions.py` exactly as a second human annotator would be
scored: multi-label precision/recall/F1, top-1 accuracy, and **chance-corrected**
agreement (Cohen's κ). Raw accuracy is the wrong instrument with 28 labels, most
of them rare — a system that always predicts the majority class would look good.

**Comparator.** A test runner: it can say *that* a submission is wrong but never
*why*, so its top-1 diagnostic accuracy is zero by construction. Stating that
explicitly is the point — it is what the diagnostic layer buys.

**False positives.** Six deliberately *correct* submissions are included. Without
them a diagnostic system's false-positive rate cannot be measured at all, and a
system that flags something on every submission would score perfectly on recall.

**What the current result establishes.** Internal validity: the pipeline recovers
the beliefs it is designed to name, the taxonomy is expressible, and the fusion
does not manufacture findings on clean code.

**What it does not establish.** Generalisation. The taxonomy, the detection rules
and this labelled bank were developed together. The caveat travels with the
numbers — it is a field of the result object, printed in every report and present
in the JSON.

**The held-out bank** (`data/heldout.py`, `dmas bench --bank heldout`) is the
first step past internal validity: six new tasks and 31 submissions written after
the rules were frozen and labelled before the system ran, under a protocol stated
in the module itself. Its labels are the authors' own.

**External validation** (`dmas refactory --data ...`) runs the evaluator over the
4,225 real student programs of the Refactory corpus, whose correct/wrong labels
are objective and third-party. It scores false accusations on accepted programs
and detection, localisation and explanation on rejected ones; it cannot score
diagnostic accuracy, because the corpus has no misconception labels.

**Ablations.** `--sources` drops evidence sources, `--fusion` swaps the
combination rule, `--model-only` runs the reasoning model on the code alone, and
`--backend openai --model ...` fuses a live reasoning model. The comparison of
the last two against the deterministic evaluator is the measurement of what the
model contributes.

**To establish generalisation fully** you need what this repository cannot
contain: a held-out corpus of real student submissions annotated independently
by at least two domain experts, with inter-annotator agreement reported *before*
the system is run on it. `DiagnosticBenchmark` accepts any sequence of
`Submission` objects.

## RQ2 — Intervention effectiveness

> Does prompting student self-correction via the Cognitive Intervention Agent
> lead to a statistically significant reduction in recurring algorithmic errors
> compared to direct solution generation?

**Design.** Three arms, which is the minimum needed to attribute an effect to the
right cause:

| arm | description |
|---|---|
| `control-direct` | the "vanilla LLM": explains the correct solution, diagnoses nothing |
| `treatment-mas` | the full deductive multi-agent system |
| `ablation-no-kg` | Socratic form retained, knowledge-graph alignment removed |

Without the ablation, any effect could be attributed to *asking questions* rather
than to asking the *right* question, and objective 2 of the proposal — the
knowledge graph — would go unevaluated.

**Randomisation.** Stratified by ability and by held misconception, dealt
round-robin within blocks. Simple randomisation achieves balance only in
expectation, which leaves a study of this size under-powered.

**Outcomes.** Knowledge retention on a *transfer* test (higher-tier variants, as
the proposal specifies — re-testing the same problem measures whether a student
can repeat a fix, not whether the belief changed); recurring-error rate; and
mastery gain.

**Analysis.** Welch's *t*-test (unequal variances — the arms have no reason to
share one) reported alongside Mann-Whitney *U*, because learning gains are
bounded and skewed and reporting only whichever came out significant is the
classic forking-paths error. Hedges' *g* with a BCa bootstrap interval and
Cliff's delta. Holm–Bonferroni across the family. Achieved power and the minimum
detectable effect are reported so a null result is interpretable.

### The honest status of these numbers

**The cohort is simulated.** The response model in
`deductive_mas/simulation/student.py` encodes a pedagogical assumption drawn from
the scaffolding literature — that guidance which is on target and inside the ZPD
produces more durable learning than being handed an answer — and any experiment
run against it will reproduce that assumption. **Nothing here is evidence that
Socratic tutoring works.** The disclaimer is attached to the result object and
printed above every table.

What the simulation *does* establish, which is not circular:

1. the analysis, statistics and reporting are exercised end to end on data with
   known ground truth, so a bug that would corrupt a real study surfaces here;
2. the effect is driven by quantities the system genuinely computes — did it
   identify the right misconception, did it aim at a concept the learner actually
   lacks, was that concept reachable — so the experiment measures the system's
   **targeting quality**, converted into outcomes through an explicit and
   editable response model;
3. the design can be power-analysed before a single human is recruited.

The comparison that carries real information is `treatment-mas` against
`ablation-no-kg` on the **on-target rate**, because that quantity is measured, not
modelled.

### The design analysis

`dmas design` runs the experiment over many random cohorts (replication), under
a null response model in which the arms cannot differ (calibration: the
family-wise rejection rate must sit at the nominal level), across a sweep of the
assumed Socratic advantage (sensitivity: the effect must vanish at the null while
the measured on-target rate does not move), and across cohort sizes (empirical
power, for sizing a human study).

### The gate

`dmas gate adversarial` scores the leakage gate on guidance manufactured from
every (task, submission) pair with verdicts known by construction; `dmas gate
audit` runs the full pipeline and records what each generator proposed and what
the gate did with it.

### Running it with humans

Replace `simulation/baselines.py`'s tutors with a logging front end, keep the
same outcome measures and the same analysis code, and pre-register: arms, sample
size from `stats.power.required_sample_size`, primary outcome, and the
correction family. `Experiment._analyse` needs no changes — it consumes lists of
per-participant numbers.

## RQ3 — Robustness

> How resilient is the architecture against diverse coding styles, non-standard
> variable naming conventions, and convoluted but conceptually flawed logic?

**Instrument.** `dmas robustness` — a mutation study. Each labelled submission is
rewritten by semantics-preserving AST transformations (obfuscated identifiers,
split tuple assignments, inverted conditionals, injected dead code, expanded
augmented assignments, and all of them combined) and the diagnosis is recomputed.

**Verification.** Every mutant is checked to be behaviourally identical to its
original before it is scored. Without that check, a transformation that
accidentally changed the program's meaning would make an honest change of
diagnosis look like a failure.

**Measures.** Top-1 stability (did the headline diagnosis survive?) and Jaccard
overlap of the full finding set. Diagnoses that shifted are listed individually
in the report rather than absorbed into an average.

Note that the injected-dead-code condition legitimately *lowers* set overlap: the
mutant really does contain a dead store, and the system really should say so.
Top-1 stability is the measure to read there.

## Reproducing everything

```bash
make test          # the full suite
dmas bench --json      > rq1.json
dmas experiment --json > rq2.json      # --students 180 by default
dmas robustness --json > rq3.json
```

Every command is deterministic given `--seed`. Without an API key the reasoning
backend falls back to the offline deterministic reasoner, so results do not
depend on a network service; with `OPENAI_API_KEY` set (or `--backend openai
--base-url ... --model openai/gpt-oss-120b`, which is the default endpoint and
model) the same commands run a reasoning model and measure its contribution by
comparison.
