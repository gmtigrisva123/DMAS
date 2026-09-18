# Algorithms

Every routine below is implemented in this repository from first principles.
There are no third-party dependencies, so each result is auditable end to end.

## Program analysis

**Control-flow graph** — `analysis/cfg.py`. Basic blocks with a loop stack for
`break`/`continue`; iterative dominator fixpoint (Cooper–Harvey–Kennedy); back
edges and their natural loops; cyclomatic complexity.

**Dataflow** — `analysis/dataflow.py`. Reaching definitions (forward, may) and
live variables (backward, may) on one worklist engine, in reverse post-order.
Derived: def-use chains, possibly-uninitialised uses, dead stores, loop-carried
variables, and loops whose guard reads nothing the body writes.

Two subtleties that a naive implementation gets wrong, and that produced real
bugs here before they were fixed:

* names bound by a nested lambda or comprehension are not free variables of the
  enclosing scope — missing this reports `sorted(xs, key=lambda p: p[1])` as a
  use of an undefined `p`;
* an in-place method call is loop progress — `queue.popleft()` is a *call*, not
  an assignment, so without it every correct worklist loop looks divergent.

**Asymptotic cost** — `analysis/complexity.py`. Syntax-directed abstract
interpretation over the semiring `O(n^p log^q n)` with a top element for
exponential growth: sequential blocks take a maximum, nested loops multiply,
`while` loops are classified by how their guard variables evolve (halving,
doubling, constant step, or no progress at all). Recursion is resolved by
extracting `T(n) = a T(n/b) + f(n)` and applying the master theorem, or
`T(n) = a T(n−1) + f(n)` by unrolling. Memoisation is detected — a guarded read
plus a write on the same subscripted container, or an `lru_cache` decorator — and
changes the answer from a branching recurrence to *states × work per state*.

Two refinements that matter pedagogically:

* a flow-insensitive container-type inference distinguishes an `O(1)` hashed
  membership test from an `O(n)` scan, which is the most consequential
  complexity misconception in practice;
* the worklist/adjacency pattern is recognised and amortised, so BFS is `O(V+E)`
  rather than `O(V·E)` — and the contrast between a `deque`+`set` BFS and a
  `list`+`list` BFS survives, which is exactly the lesson.

**Guarded execution** — `analysis/tracer.py`. `sys.settrace` records per-line
value snapshots under three independent limits: an AST denylist, a bytecode-step
budget and a wall-clock deadline. The hook is installed even when not recording,
because it is the only thing that bounds a non-terminating submission. A research
instrument, not a security sandbox.

## Locating the divergence

**Behavioural variable matching** — `analysis/varmatch.py`. Each variable becomes
a time series of the values it took. Series are compared on two channels — a
magnitude channel that tells `lo` from `hi`, and a z-scored shape channel that
recognises the same role over a different range — using **dynamic time warping**,
which tolerates a loop running a different number of iterations. The globally
optimal correspondence is found with the **Hungarian algorithm** (shortest
augmenting paths with potentials, `O(n²m)`), not greedy nearest-neighbour, so one
badly-behaved variable cannot cascade. Parameters are pinned positionally, since
the signature fixes them.

This is what answers RQ3 on naming: a student writing `left`/`right`/`centre` is
matched to `lo`/`hi`/`mid` from behaviour alone.

The assignment problem always *has* a solution, which is a trap: two programs
implementing different algorithms still produce a best-fit correspondence. A
per-role confidence bar therefore gates what may be quoted back to a student —
the report says "your `hi` holds 0 but should be 1", so it is that specific
correspondence that must be trustworthy, not the average across all of them.
Roles below the bar leave the comparison basis; when none survive, the
divergence is reported at the output level instead.

**Trajectory alignment** — `analysis/align.py`. Both runs are projected onto the
matched roles and sampled at **loop-invariant checkpoints** — each evaluation of a
loop guard, each call entry, each return. Sampling there rather than at every
line is what makes `lo, hi = 0, n` and two separate assignments compare equal.

The divergence *location* is exact: because the executions agree on every state
up to it, it is the end of their longest common prefix, and no alignment
heuristic can move it. Alignment is used to *classify* what happened — a
substitution (wrong value), a student-only transition (an extra iteration) or a
reference-only transition (a skipped one) — by comparing three continuations over
a short lookahead window. Needleman–Wunsch is retained for the trace-similarity
metric.

**Counterexample search** — `analysis/counterexample.py`. Property-based search
(declared tests first, then randomised inputs, with the reference as oracle),
then **delta debugging** (`ddmin`, generalised to a typed structural shrinker) to
a 1-minimal witness. Every candidate passes the problem's precondition, so the
shrinker can never "find" a counterexample outside the valid input space.

## Belief

**Dempster–Shafer fusion** — `analysis/belief.py`. Sources assign mass to a
hypothesis *or to ignorance*, which is what "the static rules have nothing to say
here" actually means. Shafer's discounting turns a per-source reliability into a
principled transfer of mass to ignorance; the normalisation constant measures
conflict, which the orchestrator surfaces as *contested* rather than averaging
away. Mass lives on singletons plus `Θ`, so combination is `O(n)` per pair.

## Knowledge

**Blame propagation** — `knowledge/graph.py`. Personalised PageRank over the
*reversed* prerequisite graph. Influence decays smoothly with path length rather
than falling off a fixed-depth cliff, and a prerequisite implicated by several
failing concepts outranks one reached by a single path.

**Deficiency frontier** — the unmastered concepts whose own prerequisites are
already secure: the root causes, and the only places where teaching can succeed.
The subtlety is that blame flows *down* the prerequisite chain, so testing the
prerequisites against those same lowered beliefs counts one piece of evidence
twice and walks the frontier to the floor of the curriculum. The frontier
therefore tests prerequisites against *background* belief and the concept itself
against post-diagnosis belief.

**Mastery** — `knowledge/mastery.py`. Bayesian knowledge tracing with two
extensions the setting forces: soft evidence (a diagnosis arrives with a belief,
not a grade, and is applied as a convex interpolation that recovers textbook BKT
at weight 1) and a weakest-link ceiling swept in topological order, so one pass
reaches the fixpoint.

**ZPD selection** — `knowledge/zpd.py`. A 3PL response model with Fisher
information; EAP ability estimation over a grid (preferred to MLE, which is
undefined for an all-correct or all-incorrect pattern — the common case at the
start of a session); and greedy maximisation of a submodular utility that trades
ZPD fit, information and concept relevance against redundancy, giving the
standard `1 − 1/e` guarantee.

## Retrieval

**BM25** — `retrieval/bm25.py`. Okapi with the `+1` inside the logarithm, so IDF
stays non-negative for common terms.

**Latent semantic indexing** — `retrieval/lsa.py`, `util/linalg.py`. TF-IDF
term-document matrix, truncated by **randomised SVD** (Halko–Martinsson–Tropp:
Gaussian sketch, power iterations, re-orthonormalisation, then a dense
decomposition of the small projection). The dense step is a **one-sided Jacobi
SVD** applied to the *Gram matrix* of the projection rather than to the
projection itself: for a symmetric positive semi-definite matrix the SVD is the
eigendecomposition, and decomposing the `w × w` Gram matrix costs `O(w³)` per
sweep where the tall `n × w` projection costs `O(n w²)`. That single change took
the index build — and therefore the tutor's start-up — from about five seconds to
under one, with singular values matching the exact decomposition to three
decimals. Reconstruction error of the underlying Jacobi routine is ~1e-15 in the
tests.

**Fusion** — `retrieval/fusion.py`. Reciprocal rank fusion, because BM25 scores
and cosine similarities live on incomparable scales; then maximal marginal
relevance for diversity. Two calibrations that mattered: the canonical RRF
constant of 60 flattens the head of a 47-card ranking to within rounding, so it
is set to the order of the result-set size; and redundancy is discounted across
card *kinds*, so a procedure card and a pitfall card about the same topic are not
treated as paraphrases.

**Query expansion** — out-of-vocabulary query terms are matched to the vocabulary
by character-trigram overlap, which bridges the morphological near-misses a
conservative stemmer leaves behind (`revisit` → `visit`). Orthographic
normalisation folds `-ize`/`-ise`.

**Grounding** — `retrieval/ma_rag.py`. Every *assertive* sentence must be
supported by a retrieved card. Three refinements make the check meaningful:

* terms the corpus has never seen enter the denominator at maximum weight and can
  never enter the numerator — otherwise invented vocabulary is free;
* the lexical and semantic channels are blended rather than maximised, because
  latent similarity measures *topic*, and a fabricated claim about binary search
  is on-topic by construction;
* questions, statements about the student's own code, and *reflective*
  imperatives are exempt — they have no truth value. Code-action verbs ("use",
  "set", "replace") are deliberately *not* exempt.

## Answer-leakage detection

`analysis/leakage.py`. **Winnowing** (Schleimer–Wilkerson–Aiken, the algorithm
behind MOSS): hash every `k`-gram of the alpha-normalised token stream, slide a
window, keep each window's minimum. Any shared substring of at least `w + k − 1`
tokens is detected. Identifiers are normalised first, since "set the upper bound
to `mid`" leaks exactly as much as `hi = mid`.

Novelty discounting scores `containment(hint, reference) × (1 − containment(hint,
student))`: a passage the student already wrote reveals nothing.

## Statistics

`stats/`. Regularised incomplete beta and gamma by continued fraction with
Lentz's algorithm, giving Student's *t*, χ², and Snedecor's *F*; Acklam's normal
inverse with a Halley refinement. Welch's *t*-test (not the pooled version — the
arms have no reason to share a variance), Mann–Whitney *U* with tie correction,
Wilcoxon signed-rank, χ² independence. Hedges' *g* with the small-sample
correction, Cliff's delta, and **BCa bootstrap** intervals (bias correction from
the replicate distribution, acceleration by jackknife) rather than percentile
intervals, which mis-cover for bounded scores. Holm–Bonferroni and
Benjamini–Hochberg. Power and sample-size planning. Cohen's κ and Krippendorff's
α for the agreement measure RQ1 needs.

All validated against published reference values in `tests/test_stats.py` —
including the standard planning table (`n = 64` per arm for `d = 0.5` at 80%
power, `n = 26` for `d = 0.8`).
