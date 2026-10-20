# Architecture

## The one-sentence version

A student's program is executed alongside a reference implementation; the two
runs are aligned by *behaviour* to find the earliest instant they disagree; that
instant is explained as a **belief**, not a bug; the belief is located in a
prerequisite graph; and the guidance that follows is checked, mechanically, for
answer leakage before anyone sees it.

## Layering

Dependencies point downward only. The package has no third-party dependencies at
all, so every arrow below is code in this repository.

```
                  web/ ── cli/           ui/
                            \            /
                          agents/  ──────
                          /  |  \
                         /   |   \
          knowledge/  ───    |    ─── retrieval/      llm/
                 \           |            /            /
                  \      pipeline.py ────────────────────
                   \         |
                    ──── analysis/
                             |
                          util/  domain.py  config.py  stats/
```

* **`util/`** — determinism, tokenisation, linear algebra. No domain knowledge.
* **`domain.py`** — the vocabulary every layer shares. Data only, no behaviour.
* **`analysis/`** — program analysis. Knows nothing about pedagogy: it can tell
  you a loop never terminates, not that the student misunderstands invariants.
* **`knowledge/`** — the curriculum: concept graph, misconception taxonomy,
  mastery model, ZPD selection. Written *against* analysis types.
* **`pipeline.py`** — composes the two. It sits above both, which is what keeps
  the dependency graph acyclic; putting it inside `analysis/` creates a cycle,
  and an earlier version of this project had exactly that bug.
* **`retrieval/`**, **`llm/`** — grounding evidence and reasoning backends.
* **`agents/`** — the three agents, the compliance gate, the orchestrator.
* **`ui/`**, **`cli/`** — presentation. Pure functions from results to lines,
  plus the interactive session, which owns the one thing a single call cannot:
  continuity across a learner's attempts.
* **`web/`** — the same session behind HTTP. It calls the session's rules in
  `cli/session.py` (`detect_problem`, `record_attempt`) instead of copying them,
  so the terminal and the browser cannot drift apart.

## The blackboard

The agents never call one another. Each reads and writes a shared `Blackboard`,
so the Cognitive Intervention agent depends on *a deficiency frontier existing*,
not on how the alignment agent computed one. Any stage can be ablated — and the
RQ2 experiment does exactly that, running an arm with the knowledge graph
removed to check that targeting, and not merely questioning, is what helps.

```
Submission ─▶ SocraticEvaluator ─▶ KGAlignment ─▶ CognitiveIntervention ─▶ Verifier ─▶ Report
                    │                   │                  │                  │
                 evidence            blame,             hints,            leakage,
                 diagnosis          frontier          challenge          grounding
```

The pipeline is a fixed order rather than a negotiation, for two reasons: the
information dependency is strict (you cannot align what you have not diagnosed),
and a fixed order makes a session reproducible, which the experimental protocol
requires. Any stage may fail without taking the session down — a failed stage
records a warning and later stages degrade to what they can still do.

## Evidence flow inside the evaluator

```
                ┌── static rules ─────────────┐
 submission ──▶ ├── dynamic trace divergence ─┤──▶ Dempster–Shafer ──▶ Diagnosis
                ├── cost analysis ────────────┤     (per-misconception
                └── reasoning backend ────────┘      binary frames)
```

Four sources, deliberately different in character:

| source | property | reliability |
|---|---|---|
| symbolic rules | sound but incomplete | 0.86 |
| trace divergence | precise but input-dependent | 0.92 |
| cost analysis | an upper bound, not a proof | 0.73 |
| reasoning backend | broad but fallible | 0.62 |

They are fused per misconception on a binary frame `{present, absent}` rather
than over one shared frame. That matters: a submission can use a list as a queue
*and* scan it linearly, and forcing those to share a unit of mass would make each
look weaker the more of them are present. A clean differential-testing run
contributes mass to *absent*, so "we looked hard and found nothing" is treated as
evidence rather than as silence.

## The compliance gate

The one hard guarantee the system makes:

> No guidance reaches a student without passing the answer-leakage gate.

It is enforced by replacement, not by warning. When a rung fails, the chain is:
generated text → deterministic template → template with the offending sentences
removed → a fixed question that contains no code span and asserts nothing. The
chain always terminates, so a rung is never silently dropped.

Three independent detection channels, because there are three ways to give the
game away:

| channel | catches |
|---|---|
| winnowing fingerprints | a copied passage of the reference solution |
| AST structural match | a statement of the reference the student has not written |
| edit proximity | a modified version of a line the student *has* written |

Prose repairs ("set `hi` to `len(a)`") are normalised into assignments first, so
all three channels see them — a model told not to write code reaches for that
phrasing immediately.

Quoting the student verbatim is explicitly allowed. Making someone look at what
they actually wrote is the method, not a violation of it.

## Reasoning backends

`Reasoner` resolves a backend, caches by content hash, and **degrades** rather
than failing. Agents never build prompt strings; they construct a `Task` — a
named operation with a typed payload and an expected output schema — which means
the deterministic offline reasoner implements the same interface with no prompt
parsing at all.

The offline reasoner is not a language model and does not pretend to be one. It
is a rule-based generator that exists so the system always runs, so experiments
are reproducible, and so the contribution of the reasoning model is measurable
by ablation.

## The web app

`dmas serve` puts the interactive session behind a small JSON API and a static
front end, both served by the standard library's threading HTTP server. It adds
no logic of its own to diagnosis, alignment or intervention; what it adds is what
a server needs and a terminal does not.

**One projection, by whitelist.** A `TutoringResult` is never serialised. Each
attempt is projected once, in `web/views.py`, right after the orchestrator
returns, and the client only ever sees that projection. Left out on purpose: the
reference solution and the oracle, the gate log (it keeps every rejected
candidate verbatim), the telemetry warnings verbatim (a gate rejection quotes the
text it rejected; they become fixed sentences), the challenge's expected insight
(the gate never reviews it), the teacher-facing narrative and the divergence's
reference state (both state the values rung 2 exists to reveal), and the
grounding cards. A model's chain of thought never reaches a `TutoringResult` in
the first place.

**The gate, once more on the way out.** Every free-text field that could quote
anything (rule rationales, a model's reasons, the rungs) passes the leakage
detector again at the boundary and is replaced when it fails. The pipeline's own
gate should make this a no-op, so a replacement is logged.

**The ladder, one rung per request.** Rungs are sealed in the projection and
leave through `POST /api/attempts/{n}/hints {rung}`, which reveals rung
`revealed + 1` and nothing else: asking again for a shown rung is idempotent,
asking ahead is refused, and only the latest attempt has a ladder.

**Values the student's code produced are inert.** `repr`, `==`, `str` and even
`isinstance` can dispatch to a method the student defined, which would run
outside the step budget. The projection reads exact builtin types only.

**One execution lock.** `GuardedRunner` lowers the interpreter's recursion limit
for a run and restores it afterwards; the limit is process-wide, so concurrent
runs would leave it lowered. Every execution of student code holds one lock, and
a request that waits too long for it is told the tutor is busy. The step budget
and the deadline are untouched.

**Not an attempt.** Code that does not parse, or defines no function, is refused
with its line and column instead of being diagnosed: the pipeline would report no
finding for it and the session would credit the concept as repaired.

The guarded interpreter is still not a sandbox. Work done inside one C call
(`10 ** 10 ** 8`, a huge allocation) or inside a method the student defined and
the core invokes outside the trace hook is not bounded by the step budget, so the
server listens on localhost unless told otherwise.
