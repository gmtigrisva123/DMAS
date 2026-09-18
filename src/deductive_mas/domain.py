"""Shared data types (the stuff written on the blackboard). No logic in here so
the other packages can import it without cycles.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# enums
class EvidenceKind(enum.Enum):
    """Where a piece of evidence came from."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    COUNTEREXAMPLE = "counterexample"
    COMPLEXITY = "complexity"
    MODEL = "model"

    def __str__(self) -> str:
        return self.value


class Severity(enum.Enum):
    INFO = 1
    MINOR = 2
    MAJOR = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name.lower()


class HintKind(enum.Enum):
    """Hint levels, from least to most direct."""

    ORIENTING = "orienting"          # points at a region of the code
    CONTRADICTION = "contradiction"  # shows a concrete contradiction
    CONCEPTUAL = "conceptual"        # names the idea to revisit
    PROCEDURAL = "procedural"        # suggests a repair strategy, never the code

    def __str__(self) -> str:
        return self.value


class DivergenceKind(enum.Enum):
    STATE_MISMATCH = "state_mismatch"
    CONTROL_MISMATCH = "control_mismatch"
    EARLY_TERMINATION = "early_termination"
    NON_TERMINATION = "non_termination"
    EXCEPTION = "exception"
    OUTPUT_MISMATCH = "output_mismatch"

    def __str__(self) -> str:
        return self.value


# source level stuff
@dataclass(frozen=True)
class SourceSpan:
    """A region in the student's code."""

    line: int
    end_line: int
    col: int = 0
    end_col: int = 0
    snippet: str = ""

    def __str__(self) -> str:
        if self.end_line != self.line:
            return f"L{self.line}-{self.end_line}"
        return f"L{self.line}"


@dataclass(frozen=True)
class TestCase:
    """One input / expected output pair."""

    args: Tuple[Any, ...]
    expected: Any
    label: str = ""

    def __str__(self) -> str:
        return self.label or repr(self.args)


@dataclass(frozen=True)
class ProblemSpec:
    """A problem. reference_solution is never shown to the student, it is only used
    for trace alignment and for the leakage check.
    """

    pid: str
    title: str
    statement: str
    entry_point: str
    parameters: Tuple[str, ...]
    concepts: Tuple[str, ...]
    reference_solution: str
    tests: Tuple[TestCase, ...]
    sampler: Optional[Callable[[Any], Tuple[Any, ...]]] = None
    # checks that an argument tuple is valid, the shrinker must not step outside
    # the valid inputs (a "counterexample" on an unsorted array would be wrong)
    validator: Optional[Callable[[Tuple[Any, ...]], bool]] = None
    difficulty: float = 0.0          # IRT b (logits)
    discrimination: float = 1.2      # IRT a
    guessing: float = 0.05           # IRT c
    tags: Tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.pid} ({self.title})"


@dataclass(frozen=True)
class Submission:
    """A student attempt at a problem."""

    sid: str
    problem_id: str
    source: str
    author: str = "student"
    # gold labels, only set in the labelled bank
    gold_misconceptions: Tuple[str, ...] = ()
    # True if the program gives the right output on every valid input. A program
    # can be correct and still have a misconception (linear scan on sorted data,
    # list used as a queue, shared mutable cache...)
    functionally_correct: bool = False

    @property
    def lines(self) -> List[str]:
        return self.source.splitlines()


# knowledge layer
@dataclass(frozen=True)
class Concept:
    """A node in the knowledge graph."""

    cid: str
    name: str
    stratum: str
    summary: str
    prerequisites: Tuple[str, ...] = ()
    difficulty: float = 0.0
    # BKT overrides, None = use the global default
    prior: Optional[float] = None
    learn_rate: Optional[float] = None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Misconception:
    """A named misconception. concepts = the graph nodes it points at, blame gets
    propagated from there to the prerequisites.
    """

    mid: str
    name: str
    description: str
    concepts: Tuple[str, ...]
    severity: Severity = Severity.MAJOR
    # the wrong belief in the student's own words ("I think that ...")
    student_voice: str = ""
    remediation_focus: str = ""


# evidence + diagnosis
@dataclass(frozen=True)
class Evidence:
    """One piece of evidence for (or against) a misconception."""

    misconception_id: str
    kind: EvidenceKind
    belief: float
    rationale: str
    span: Optional[SourceSpan] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.misconception_id}<{self.kind}:{self.belief:.2f}>"


@dataclass(frozen=True)
class DivergencePoint:
    """First point where the student trace leaves the reference trace."""

    kind: DivergenceKind
    step: int
    student_line: Optional[int]
    reference_line: Optional[int]
    student_state: Mapping[str, Any] = field(default_factory=dict)
    reference_state: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    span: Optional[SourceSpan] = None

    def differing_roles(self) -> List[str]:
        keys = set(self.student_state) | set(self.reference_state)
        return sorted(k for k in keys if self.student_state.get(k) != self.reference_state.get(k))


@dataclass(frozen=True)
class Counterexample:
    """A small input where the student program gives the wrong answer."""

    args: Tuple[Any, ...]
    student_output: Any
    reference_output: Any
    student_error: Optional[str] = None
    shrink_steps: int = 0

    @property
    def crashed(self) -> bool:
        return self.student_error is not None


@dataclass(frozen=True)
class MisconceptionHit:
    """A fused, reportable misconception with its evidence attached."""

    misconception_id: str
    belief: float
    severity: Severity
    span: Optional[SourceSpan]
    rationale: str
    supports: Tuple[Evidence, ...] = ()

    @property
    def kinds(self) -> Tuple[EvidenceKind, ...]:
        return tuple(sorted({e.kind for e in self.supports}, key=lambda k: k.value))


@dataclass
class Diagnosis:
    """Output of the evaluator agent."""

    submission_id: str
    problem_id: str
    hits: Tuple[MisconceptionHit, ...] = ()
    divergence: Optional[DivergencePoint] = None
    counterexample: Optional[Counterexample] = None
    student_complexity: str = "unknown"
    reference_complexity: str = "unknown"
    complexity_gap: bool = False
    narrative: str = ""
    inconclusive: bool = False
    # how many of the tried inputs passed
    checks_passed: int = 0
    checks_total: int = 0
    # which backend answered and whether it fell back to the offline reasoner
    # half way. A degraded row does not measure that backend.
    backend: str = "offline"
    degraded: bool = False

    @property
    def check_summary(self) -> str:
        if not self.checks_total:
            return "no checks run"
        return f"{self.checks_passed}/{self.checks_total} inputs pass"

    @property
    def primary(self) -> Optional[MisconceptionHit]:
        return self.hits[0] if self.hits else None

    @property
    def confidence(self) -> float:
        return self.hits[0].belief if self.hits else 0.0


@dataclass
class AlignmentReport:
    """Output of the alignment agent."""

    blame: Dict[str, float] = field(default_factory=dict)
    deficiency_frontier: Tuple[str, ...] = ()
    prerequisite_paths: Tuple[Tuple[str, ...], ...] = ()
    mastery_before: Dict[str, float] = field(default_factory=dict)
    mastery_after: Dict[str, float] = field(default_factory=dict)
    root_cause: Optional[str] = None

    def top_blame(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.blame.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


# intervention layer
@dataclass(frozen=True)
class Hint:
    level: int
    kind: HintKind
    text: str
    concept_id: Optional[str] = None


@dataclass(frozen=True)
class Challenge:
    """A small "what if" task that shows the student a contradiction."""

    prompt: str
    trace_question: str
    expected_insight: str
    args: Optional[Tuple[Any, ...]] = None


@dataclass
class Intervention:
    """Output of the intervention agent."""

    target_concept: Optional[str] = None
    hints: Tuple[Hint, ...] = ()
    challenge: Optional[Challenge] = None
    next_item: Optional[str] = None
    zpd_probability: float = 0.0
    leakage: float = 0.0
    grounding: float = 0.0
    citations: Tuple[str, ...] = ()
    regenerations: int = 0


@dataclass
class Telemetry:
    """Timings, model calls and warnings for one session."""

    stages: List[Tuple[str, float]] = field(default_factory=list)
    model_calls: int = 0
    model_tokens: int = 0
    cache_hits: int = 0
    warnings: List[str] = field(default_factory=list)
    backend: str = "offline"

    def record(self, stage: str, seconds: float):
        self.stages.append((stage, seconds))

    def warn(self, message: str):
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def total_seconds(self) -> float:
        return sum(s for _, s in self.stages)


@dataclass
class TutoringResult:
    """Everything one orchestrator pass produced."""

    submission: Submission
    problem: ProblemSpec
    diagnosis: Diagnosis
    alignment: AlignmentReport
    intervention: Intervention
    telemetry: Telemetry = field(default_factory=Telemetry)
    evidence: Tuple[Evidence, ...] = ()
    # mastery posterior after this session as a plain dict, so the next attempt
    # can feed it back in
    mastery_posterior: Dict[str, float] = field(default_factory=dict)
    # how many observations each concept has had
    mastery_observations: Dict[str, int] = field(default_factory=dict)
    # what the generator proposed per rung and what the gate did with it
    gate_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Dict version for --json and the tests."""
        return {
            "submission": self.submission.sid,
            "problem": self.problem.pid,
            "diagnosis": {
                "primary": self.diagnosis.primary.misconception_id if self.diagnosis.primary else None,
                "confidence": round(self.diagnosis.confidence, 4),
                "hits": [
                    {
                        "id": h.misconception_id,
                        "belief": round(h.belief, 4),
                        "severity": str(h.severity),
                        "span": str(h.span) if h.span else None,
                        "evidence": [str(k) for k in h.kinds],
                    }
                    for h in self.diagnosis.hits
                ],
                "divergence": (
                    {
                        "kind": str(self.diagnosis.divergence.kind),
                        "step": self.diagnosis.divergence.step,
                        "student_line": self.diagnosis.divergence.student_line,
                        "roles": self.diagnosis.divergence.differing_roles(),
                    }
                    if self.diagnosis.divergence
                    else None
                ),
                "counterexample": (
                    {
                        "args": repr(self.diagnosis.counterexample.args),
                        "student": repr(self.diagnosis.counterexample.student_output),
                        "reference": repr(self.diagnosis.counterexample.reference_output),
                        "shrink_steps": self.diagnosis.counterexample.shrink_steps,
                    }
                    if self.diagnosis.counterexample
                    else None
                ),
                "complexity": {
                    "student": self.diagnosis.student_complexity,
                    "reference": self.diagnosis.reference_complexity,
                    "gap": self.diagnosis.complexity_gap,
                },
            },
            "alignment": {
                "root_cause": self.alignment.root_cause,
                "frontier": list(self.alignment.deficiency_frontier),
                "blame": {k: round(v, 4) for k, v in self.alignment.top_blame(8)},
            },
            "intervention": {
                "target": self.intervention.target_concept,
                "hints": [
                    {"level": h.level, "kind": str(h.kind), "text": h.text}
                    for h in self.intervention.hints
                ],
                "challenge": (
                    {
                        "prompt": self.intervention.challenge.prompt,
                        "trace_question": self.intervention.challenge.trace_question,
                    }
                    if self.intervention.challenge
                    else None
                ),
                "leakage": round(self.intervention.leakage, 4),
                "grounding": round(self.intervention.grounding, 4),
                "zpd_probability": round(self.intervention.zpd_probability, 4),
                "citations": list(self.intervention.citations),
            },
            "telemetry": {
                "backend": self.telemetry.backend,
                "seconds": round(self.telemetry.total_seconds, 4),
                "model_calls": self.telemetry.model_calls,
                "warnings": list(self.telemetry.warnings),
            },
        }
