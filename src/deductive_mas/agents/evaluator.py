"""Socratic evaluator agent.

Instead of just correct / incorrect it tries to find the exact point where the
student's reasoning went wrong. Three things happen here:

1. run the deterministic analysis engine (symbolic, dynamic and cost evidence
   plus the divergence point)
2. ask the model for hypotheses, but only from the taxonomy. The model picks
   among named misconceptions, it cannot invent new ones, which limits
   hallucination at the output level
3. fuse everything with Dempster's rule so independent sources that agree
   reinforce each other and disagreement shows up as conflict
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..analysis.belief import MassFunction, fuse
from ..pipeline import AnalysisEngine, AnalysisResult
from ..config import Config
from ..domain import (
    Diagnosis,
    Evidence,
    EvidenceKind,
    MisconceptionHit,
    Severity,
    SourceSpan,
)
from ..knowledge.misconceptions import (
    MISCONCEPTIONS,
    is_correctness_misconception,
    is_subsumed,
)
from ..llm.base import Task
from ..llm.parsing import clamp_unit
from ..llm.registry import Reasoner
from .base import Agent, Blackboard

_SYSTEM = (
    "You are a diagnostic agent in an intelligent tutoring system for algorithmic programming. "
    "You are given the results of a static and dynamic analysis of a student's submission. "
    "Your task is to name the student's underlying MISCONCEPTION — the flawed belief that "
    "generated the defect — not to fix the code. You must choose only from the supplied "
    "taxonomy. If the supplied evidence already explains the defect, return an empty list "
    "rather than adding speculative hypotheses. Never reveal or restate the correct solution."
)

_INSTRUCTION = (
    "Decide which misconceptions from the taxonomy this evidence supports, and with what "
    "confidence in [0, 1]. Return JSON with keys 'misconceptions' (a list of objects with "
    "'id', 'confidence' and 'reason') and 'summary' (one sentence, no code)."
)

# reply schemas, enforced server side by backends that support it
_DIAGNOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "misconceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["misconceptions", "summary"],
    "additionalProperties": False,
}

_NARRATE_SCHEMA = {
    "type": "object",
    "properties": {"narrative": {"type": "string"}},
    "required": ["narrative"],
    "additionalProperties": False,
}

# evidence kind -> independent belief source for fusion
_SOURCE_OF: Mapping[EvidenceKind, str] = {
    EvidenceKind.STATIC: "symbolic",
    EvidenceKind.COMPLEXITY: "cost",
    EvidenceKind.DYNAMIC: "dynamic",
    EvidenceKind.COUNTEREXAMPLE: "dynamic",
    EvidenceKind.MODEL: "model",
}


class SocraticEvaluatorAgent(Agent):
    """Builds a Diagnosis from evidence, never from the code alone."""

    name = "socratic-evaluator"
    role = "cognitive diagnosis"

    def __init__(
        self,
        config: Optional[Config] = None,
        reasoner: Optional[Reasoner] = None,
        engine: Optional[AnalysisEngine] = None,
    ):
        self.config = config or Config()
        self.reasoner = reasoner or Reasoner(self.config.llm)
        self.engine = engine or AnalysisEngine(self.config)

    # run
    def run(self, board: Blackboard) -> Blackboard:
        analysis = self.engine.analyse(board.problem, board.submission)
        board.analysis = analysis
        board.evidence = list(analysis.evidence)
        for note in analysis.notes:
            board.note(note)

        model_evidence = self._consult_model(board, analysis)
        board.evidence.extend(model_evidence)

        diagnosis = self._fuse(board, analysis, board.evidence)
        if self.config.diagnosis.narrate:
            diagnosis.narrative = self._narrate(board, analysis, diagnosis)
        diagnosis.backend = board.telemetry.backend
        diagnosis.degraded = any("degraded" in w for w in board.telemetry.warnings)
        board.diagnosis = diagnosis
        return board

    # fusion
    def _fuse(
        self,
        board: Blackboard,
        analysis: AnalysisResult,
        evidence: Sequence[Evidence],
    ) -> Diagnosis:
        cfg = self.config.diagnosis
        reliability = {
            "symbolic": cfg.static_reliability,
            "cost": cfg.static_reliability * 0.85,
            "dynamic": cfg.dynamic_reliability,
            "model": cfg.llm_reliability,
        }

        # several rules in one source can point at the same misconception, take the
        # max not the sum (they are different symptoms of one belief, not
        # independent trials)
        allowed = set(cfg.sources)
        evidence = [e for e in evidence if _SOURCE_OF.get(e.kind, "symbolic") in allowed]
        grouped: Dict[str, Dict[str, float]] = {}
        for item in evidence:
            source = _SOURCE_OF.get(item.kind, "symbolic")
            bucket = grouped.setdefault(source, {})
            bucket[item.misconception_id] = max(bucket.get(item.misconception_id, 0.0), item.belief)

        # Dempster's rule wants mutually exclusive hypotheses and misconceptions
        # are not (you can use a list as a queue AND scan it linearly). Sharing one
        # unit of mass would make each one look weaker the more there are. So every
        # misconception gets its own {present, absent} frame and the sources are
        # combined inside it. Conflict then means: a rule says present, execution
        # says the program never misbehaves.
        candidates = sorted({item.misconception_id for item in evidence})
        functionally_correct = analysis.counterexample is None and analysis.search.attempts > 0
        absent = "~absent~"

        hits: List[MisconceptionHit] = []
        conflicts: List[float] = []
        uncertainties: List[float] = []
        for mid in candidates:
            masses: List[Tuple[str, MassFunction]] = []
            for source in ("symbolic", "dynamic", "cost", "model"):
                support = grouped.get(source, {}).get(mid, 0.0)
                refutation = 0.0
                if (
                    source == "dynamic"
                    and functionally_correct
                    and is_correctness_misconception(mid)
                ):
                    # differential testing tried and found nothing wrong. That is evidence
                    # against a correctness misconception, not just silence.
                    refutation = cfg.dynamic_refutation
                if support <= 0.0 and refutation <= 0.0:
                    continue
                masses.append(
                    (
                        source,
                        MassFunction.from_scores(
                            {mid: support, absent: refutation},
                            reliability=reliability[source],
                        ),
                    )
                )
            if not masses:
                continue
            if cfg.fusion_rule == "dempster":
                fusion = fuse(masses)
                belief = fusion.mass.belief(mid)
                conflicts.append(fusion.max_conflict)
                uncertainties.append(fusion.mass.uncertainty)
            else:
                belief = _naive_fusion(cfg.fusion_rule, masses, mid, absent)
                conflicts.append(0.0)
                uncertainties.append(1.0 - belief)
            if belief < cfg.report_threshold:
                continue
            entry = MISCONCEPTIONS.get(mid)
            supports = tuple(e for e in evidence if e.misconception_id == mid)
            span = next((e.span for e in supports if e.span is not None), None)
            hits.append(
                MisconceptionHit(
                    misconception_id=mid,
                    belief=belief,
                    severity=entry.severity if entry else Severity.MAJOR,
                    span=span,
                    rationale=_best_rationale(supports),
                    supports=supports,
                )
            )

        # if the program is observably wrong, a belief that explains the wrong
        # answer beats one that is only about cost, even with slightly lower
        # belief. Without a counterexample this does nothing and belief decides.
        failing = analysis.counterexample is not None
        present = [h.misconception_id for h in hits]
        hits.sort(
            key=lambda h: (
                is_subsumed(h.misconception_id, present),
                not (failing and is_correctness_misconception(h.misconception_id)),
                -h.belief,
                -h.severity.value,
                h.misconception_id,
            )
        )
        hits = hits[: cfg.max_reported]

        max_conflict = max(conflicts) if conflicts else 0.0
        contested = bool(hits) and max_conflict > 0.55
        if contested:
            board.note(
                f"evidence sources conflict (K={max_conflict:.2f}): a structural rule implicates "
                "a belief that execution did not confirm"
            )
        weak = bool(hits) and hits[0].belief < cfg.primary_threshold
        if weak and not contested:
            board.note(
                f"the leading hypothesis reaches only {hits[0].belief:.2f}, below the "
                f"{cfg.primary_threshold:.2f} needed for a settled diagnosis; it is reported "
                "as provisional"
            )

        diagnosis = Diagnosis(
            submission_id=board.submission.sid,
            problem_id=board.problem.pid,
            hits=tuple(hits),
            divergence=analysis.divergence,
            counterexample=analysis.counterexample,
            student_complexity=str(analysis.complexity.total),
            reference_complexity=str(analysis.reference_complexity.total),
            complexity_gap=analysis.complexity_gap,
            inconclusive=not hits or contested or weak,
            checks_passed=analysis.search.passed,
            checks_total=analysis.search.attempts,
        )
        board.extras["fusion_conflict"] = max_conflict
        board.extras["fusion_uncertainty"] = (
            sum(uncertainties) / len(uncertainties) if uncertainties else 1.0
        )
        board.extras["fusion_sources"] = tuple(sorted(grouped))
        return diagnosis

    # model
    def _consult_model(self, board: Blackboard, analysis: AnalysisResult) -> List[Evidence]:
        payload = self._diagnose_payload(board, analysis)
        task = Task(
            name="diagnose",
            system=_SYSTEM,
            instruction=_INSTRUCTION,
            payload=payload,
            schema={"misconceptions": list, "summary": str},
            json_schema=_DIAGNOSE_SCHEMA,
        )
        completion = self.reasoner.run(task)
        board.telemetry.model_calls += 1
        board.telemetry.model_tokens += completion.total_tokens
        board.telemetry.backend = completion.backend
        if completion.degraded:
            board.telemetry.warn(f"reasoning backend degraded: {completion.note}")

        out: List[Evidence] = []
        for item in completion.data.get("misconceptions") or []:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("id") or "").strip()
            if mid not in MISCONCEPTIONS:
                board.telemetry.warn(f"model proposed an unknown misconception: {mid!r}")
                continue
            belief = clamp_unit(item.get("confidence"), 0.3)
            if belief <= 0.0:
                continue
            out.append(
                Evidence(
                    misconception_id=mid,
                    kind=EvidenceKind.MODEL,
                    belief=belief,
                    rationale=str(item.get("reason") or "proposed by the reasoning engine"),
                    payload={"backend": completion.backend},
                )
            )
        return out

    def _diagnose_payload(self, board: Blackboard, analysis: AnalysisResult) -> Dict[str, Any]:
        candidates = sorted({e.misconception_id for e in analysis.evidence})
        counterexample = analysis.counterexample
        divergence = analysis.divergence
        return {
            "problem_title": board.problem.title,
            "problem_statement": board.problem.statement,
            "problem_tags": list(board.problem.tags),
            "problem_concepts": list(board.problem.concepts),
            "student_code": board.submission.source,
            "candidates": candidates,
            "correctness_candidates": [c for c in candidates if is_correctness_misconception(c)],
            "counterexample": (
                {
                    "args": repr(counterexample.args),
                    "student": repr(counterexample.student_output),
                    "reference": repr(counterexample.reference_output),
                    "error": counterexample.student_error,
                }
                if counterexample
                else None
            ),
            "divergence": (
                {
                    "kind": str(divergence.kind),
                    "student_line": divergence.student_line,
                    "description": divergence.description,
                }
                if divergence
                else None
            ),
            "complexity_gap": analysis.complexity_gap,
            "student_complexity": str(analysis.complexity.total),
            "reference_complexity": str(analysis.reference_complexity.total),
            "sorts_input": any(c.name in ("sorted", "sort") for c in analysis.features.calls),
            "uses_table": bool(analysis.features.memo_arity) or any(
                index.stores for index in analysis.features.indexes
            ),
            "linear_membership": any(
                e.misconception_id in ("cx.membership-in-list", "graph.list-as-visited")
                for e in analysis.evidence
            ),
            "taxonomy": [
                {"id": m.mid, "name": m.name, "description": m.description}
                for m in MISCONCEPTIONS.values()
            ],
        }

    # narrative
    def _narrate(self, board: Blackboard, analysis: AnalysisResult, diagnosis: Diagnosis) -> str:
        primary = diagnosis.primary
        entry = MISCONCEPTIONS.get(primary.misconception_id) if primary else None
        counterexample = analysis.counterexample
        divergence = analysis.divergence
        task = Task(
            name="narrate",
            system=(
                "You explain a diagnosis to a teacher, factually and without prescribing a fix. "
                "Never include code."
            ),
            instruction=(
                "Write two or three sentences describing what the evidence shows. Return JSON "
                "with the key 'narrative'."
            ),
            payload={
                "misconception_name": entry.name if entry else None,
                "misconception_description": entry.description if entry else None,
                "divergence": (
                    {
                        "kind": str(divergence.kind),
                        "step": divergence.step,
                        "student_line": divergence.student_line,
                        "description": divergence.description,
                    }
                    if divergence
                    else None
                ),
                "counterexample": (
                    {
                        "args": repr(counterexample.args),
                        "student": repr(counterexample.student_output),
                        "reference": repr(counterexample.reference_output),
                        "error": counterexample.student_error,
                    }
                    if counterexample
                    else None
                ),
                "complexity": {
                    "student": diagnosis.student_complexity,
                    "reference": diagnosis.reference_complexity,
                    "gap": diagnosis.complexity_gap,
                },
            },
            schema={"narrative": str},
            json_schema=_NARRATE_SCHEMA,
        )
        completion = self.reasoner.run(task)
        board.telemetry.model_calls += 1
        board.telemetry.model_tokens += completion.total_tokens
        return str(completion.data.get("narrative") or "").strip()


def _naive_fusion(
    rule: str, masses: Sequence[Tuple[str, MassFunction]], mid: str, absent: str
) -> float:
    """Ablation combiners: max, mean and noisy-or.

    They get the same discounted per-source supports as Dempster's rule and apply
    the refutation as a multiplicative discount, so only the combination rule
    changes.
    """
    supports = [m.belief(mid) for _, m in masses if m.belief(mid) > 0.0]
    refutations = [m.belief(absent) for _, m in masses if m.belief(absent) > 0.0]
    if not supports:
        return 0.0
    if rule == "max":
        belief = max(supports)
    elif rule == "mean":
        belief = sum(supports) / len(supports)
    elif rule == "noisy-or":
        product = 1.0
        for value in supports:
            product *= 1.0 - value
        belief = 1.0 - product
    else:
        raise ValueError(f"unknown fusion rule {rule!r}")
    for value in refutations:
        belief *= 1.0 - value
    return belief


_RAW_SYSTEM = (
    "You are an expert programming tutor. You are shown a problem statement and a student's "
    "submission. Name the student's underlying MISCONCEPTION — the flawed belief that generated "
    "the defect — choosing only from the supplied taxonomy. If the submission is correct and "
    "exhibits no flawed belief, return an empty list. Never reveal or restate the correct solution."
)

_RAW_INSTRUCTION = (
    "Read the code carefully and decide which misconceptions from the taxonomy it exhibits, with "
    "a confidence in [0, 1] for each, most likely first. Return JSON with keys 'misconceptions' "
    "(a list of objects with 'id', 'confidence' and 'reason') and 'summary' (one sentence, no code)."
)


class ModelOnlyEvaluator(Agent):
    """Baseline: the model reads the code on its own.

    No differential testing, no trace alignment, no rules, no fusion. The model
    gets the problem statement, the code and the taxonomy and its ranked answer
    is the diagnosis. The difference to the full evaluator is what the deductive
    part adds.
    """

    name = "model-only-evaluator"
    role = "cognitive diagnosis (baseline)"

    def __init__(self, config: Optional[Config] = None, reasoner: Optional[Reasoner] = None):
        self.config = config or Config()
        self.reasoner = reasoner or Reasoner(self.config.llm)

    def run(self, board: Blackboard) -> Blackboard:
        task = Task(
            name="diagnose_raw",
            system=_RAW_SYSTEM,
            instruction=_RAW_INSTRUCTION,
            payload={
                "problem_title": board.problem.title,
                "problem_statement": board.problem.statement,
                "student_code": board.submission.source,
                "taxonomy": [
                    {"id": m.mid, "name": m.name, "description": m.description}
                    for m in MISCONCEPTIONS.values()
                ],
            },
            schema={"misconceptions": list, "summary": str},
            json_schema=_DIAGNOSE_SCHEMA,
        )
        completion = self.reasoner.run(task)
        board.telemetry.model_calls += 1
        board.telemetry.model_tokens += completion.total_tokens
        board.telemetry.backend = completion.backend
        if completion.degraded:
            board.telemetry.warn(f"reasoning backend degraded: {completion.note}")

        hits: List[MisconceptionHit] = []
        for item in completion.data.get("misconceptions") or []:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("id") or "").strip()
            if mid not in MISCONCEPTIONS:
                continue
            belief = clamp_unit(item.get("confidence"), 0.3)
            if belief < self.config.diagnosis.report_threshold:
                continue
            entry = MISCONCEPTIONS[mid]
            evidence = Evidence(
                misconception_id=mid,
                kind=EvidenceKind.MODEL,
                belief=belief,
                rationale=str(item.get("reason") or "proposed by the reasoning engine"),
            )
            board.evidence.append(evidence)
            hits.append(
                MisconceptionHit(
                    misconception_id=mid, belief=belief, severity=entry.severity, span=None,
                    rationale=evidence.rationale, supports=(evidence,),
                )
            )
        hits.sort(key=lambda h: (-h.belief, h.misconception_id))
        board.diagnosis = Diagnosis(
            submission_id=board.submission.sid,
            problem_id=board.problem.pid,
            hits=tuple(hits[: self.config.diagnosis.max_reported]),
            narrative=str(completion.data.get("summary") or ""),
            inconclusive=not hits,
            backend=completion.backend,
            degraded=completion.degraded,
        )
        return board


def _best_rationale(supports: Sequence[Evidence]) -> str:
    if not supports:
        return ""
    ranked = sorted(supports, key=lambda e: (-e.belief, str(e.kind)))
    return ranked[0].rationale
