# Traceability to the research proposal

Every claim the proposal makes, and where it is implemented, measured, or
qualified.

## Section 1.2 — The three agents

| Proposal | Implementation |
|---|---|
| "The Socratic Evaluator Agent … traces the student's flawed code backward … identifies the exact logical junction where the student's cognition diverged" | `agents/evaluator.py` with `analysis/align.py`. The junction is exact: it is the end of the longest common prefix of the two state trajectories, sampled at loop-invariant checkpoints. |
| "The Knowledge-Graph Alignment Agent … cross-references the diagnosed misconception with a structured CS knowledge graph, identifying foundational prerequisites" | `agents/alignment.py` over a 65-concept prerequisite DAG (`knowledge/ontology.py`), by personalised PageRank plus a deficiency-frontier extraction. |
| "The Cognitive Intervention Agent … Vygotsky's ZPD … formulates counter-factual programming challenges or targeted logic hints … rather than delivering the optimised code solution" | `agents/intervention.py`: IRT-based ZPD selection, a three-rung Socratic ladder, a predict-then-run counter-factual challenge, and a leakage gate that enforces the "rather than" mechanically. |

## Section 1.2 — Novelty claims

**"Rather than deploying the model's reasoning to solve the problem for the user,
we project it outward to unpack the user's cognitive state."**

Implemented and enforced. The reasoning backend is never asked to solve the
problem; its diagnostic task is a forced choice among named misconceptions, and
its generative tasks pass a gate that rejects any output resembling the reference
solution. The `TestComplianceGate` suite includes a backend that deliberately
emits the full solution and asserts it is overridden.

**"A training-free MA-RAG framework … systematically mitigate the risk of LLM
hallucinations."**

`retrieval/ma_rag.py` — Planner, Retriever, Verifier, no learned parameters. The
mitigation is measurable: fabricated but on-topic claims score ~0.2–0.35 against
~0.5–0.9 for paraphrases of retrieved evidence, and every assertion is gated
individually rather than on a mean.

*Qualification.* Grounding is a bag-of-words support measure, not entailment. It
reliably catches invented vocabulary and off-corpus claims; it will not catch a
sentence that is fluent, on-topic and subtly false in a way the corpus does not
contradict. That is a real residual risk and is documented rather than papered
over.

**"Strict academic compliance."** The leakage gate is the compliance mechanism,
and its guarantee is structural: the replacement chain terminates in a fixed
question that contains no code span and asserts nothing.

## Section 2 — Objectives

| Objective | Status |
|---|---|
| "design and implement a scalable Multi-Agent System … that can parse flawed source code and trace cognitive misconceptions without revealing final answers" | Done. ~40 ms per real student program on one core, no network required. |
| "build an academic Knowledge Graph mapping advanced programming concepts to structurally isolate prerequisite deficiencies" | Done. 65 concepts, 10 strata, validated acyclic at import; the "structural isolation" is the deficiency frontier. |
| "rigorously evaluate the efficacy of Socratic AI scaffolding compared to traditional generative LLMs regarding long-term knowledge retention" | **Partially.** The instrument, the arms, the statistics and the power analysis are complete and tested. The cohort is simulated. Efficacy for human learners is not established and is not claimed. |

## Section 4.1 — Methodology

| Proposal | Implementation | Deviation |
|---|---|---|
| "developed using Python" | Python 3.9+, standard library only | — |
| "managed via … CrewAI or LangChain" | A blackboard orchestrator in `agents/` | **Deviation.** A dependency-free orchestrator keeps the whole pipeline auditable and lets any stage be ablated for RQ2; a framework would add ~40 transitive dependencies and hide the control flow the experiment needs to manipulate. The agent boundaries are those the proposal specifies. |
| "DeepSeek-R1 … deployed as the core reasoning engine" | **Deviation.** The reasoning engine is an open-weight reasoning model (`openai/gpt-oss-120b`) behind an OpenAI-compatible client (`llm/openai_compatible.py`); a Claude client (`llm/claude.py`) is also provided. The architecture is backend-agnostic: the reasoning trace is captured for audit and never shown to a student, whichever model produced it. |
| "custom Knowledge RAG … data structure definitions and time-complexity matrices" | 47 curated cards in `retrieval/corpus.py`; BM25 + LSA + RRF + MMR | — |

## Section 4.2 — Evaluation

| Proposal | Implementation |
|---|---|
| "Competitive Programming datasets like Codeforces or LeetCode" | **Deviation, in part.** Two purpose-built banks (a development bank of 29 submissions and a held-out bank of 31 written after the rules were frozen), because scraped problems come without misconception labels, which is the one thing RQ1 requires. External validity is measured on 4,225 real student programs from the Refactory corpus (Hu et al., ASE 2019), whose correct/wrong labels are objective and third-party. `ProblemSpec`/`Submission` accept external data unchanged. |
| "student errors across fundamental algorithms (e.g. Binary Search boundary conditions, BFS traversal mistakes)" | Both are in the bank, with exactly those misconceptions. |
| "Control Group … vanilla LLMs; Experimental Group … our system" | `simulation/baselines.py`, plus a third ablation arm the proposal does not require but the causal claim does. |
| "t-tests and Cohen's d" | Welch's *t* and Hedges' *g* (the small-sample correction — Cohen's *d* over-estimates by several percent at these sample sizes), with non-parametric and ordinal companions. |
| "Human subjects or simulated student profiles" | Simulated, with the disclaimer attached to every result. |

## Where the implementation goes beyond the proposal

* **Behavioural variable matching** (Hungarian + DTW) — the proposal asks about
  robustness to naming in RQ3 but proposes no mechanism; this is one.
* **Delta-debugging minimisation** — a 1-minimal failing input makes a hint
  concrete, which is what makes the contradiction rung persuasive.
* **Dempster-Shafer fusion** with explicit conflict — lets the system say
  "contested" instead of publishing a confident but disputed claim.
* **The answer-leakage gate** — the proposal states the no-answer constraint as
  an intention; this makes it enforceable and testable.
* **Static cost inference** — the proposal names complexity miscalculation as a
  target misconception; diagnosing it requires the system to have its own opinion
  about cost, which `analysis/complexity.py` provides.
* **The interactive session** (`cli/session.py`) — reveals hints one rung at a
  time and carries the mastery model across attempts, so an edit-and-retry cycle
  measures whether the belief was repaired rather than the symptom patched. The
  proposal describes the intervention; the session is what makes it a dialogue.

## Open work

1. A held-out, expert-annotated corpus of real submissions (RQ1 generalisation).
2. A human trial (RQ2 efficacy). The harness is ready; the cohort is not.
3. Entailment-grade grounding, to close the residual hallucination risk above.
4. Languages beyond Python. The analysis layer is AST-specific; the knowledge,
   retrieval, belief and statistics layers are not.
