"""What the browser is allowed to see.

The web app never serialises a TutoringResult. Each attempt is projected
here once, right after the orchestrator returns, by explicit whitelist: a
field that is not copied below does not exist as far as the client is
concerned. Left out on purpose:

- the reference solution, the declared tests and the oracle (ProblemSpec)
- the gate log: it keeps every rejected candidate verbatim
- the telemetry warnings verbatim: a gate rejection quotes the text it
  rejected and a degraded backend quotes its own error. They become fixed
  sentences in `notes`, never the original string
- the challenge's expected insight: the gate never reviews it
- the narrative: written for a teacher, never gated, and it states the
  reference's values at the divergence, which is what rung 2 is for
- the divergence's reference state and description, for the same reason
- the ladder: rungs leave through `hint_view`, one per request
- the grounding cards: the corpus has worked examples in it

Free text that could quote anything (rule rationales, a model's reasons,
the rungs themselves) goes through the answer-leakage gate once more on the
way out and is replaced when it fails.

Values a student's function produced are never passed to repr(), str(), ==
or isinstance(): each of those can dispatch to a method the student wrote,
and that method would run outside the step budget. `render_value` reads
exact builtin types only.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..agents.evaluator import SocraticEvaluatorAgent
from ..agents.intervention import _FALLBACK_RUNGS
from ..agents.verifier import ComplianceVerifier
from ..cli.session import Carry
from ..config import Config
from ..domain import Challenge, Hint, ProblemSpec, TutoringResult
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState
from ..knowledge.misconceptions import MISCONCEPTIONS

# replaces any evidence sentence that fails the gate on its way out
WITHHELD = "withheld: this note failed the answer-leakage check"

# telemetry warning prefix -> (kind, sentence the learner may read)
_WARNING_NOTES: Tuple[Tuple[str, str, str], ...] = (
    ("hint rung", "gate",
     "The answer-leakage gate replaced a generated question with a safer one."),
    ("counter-factual challenge replaced", "gate",
     "The answer-leakage gate replaced the generated challenge with a safer one."),
    ("final compliance check rejected", "gate",
     "The final compliance check withheld part of the guidance."),
    ("reasoning backend degraded", "backend",
     "The reasoning model was unavailable, so the deterministic engine wrote this guidance."),
)
_STAGE_FAILED = (
    "Part of the analysis did not complete; what is shown is what the remaining "
    "stages could establish."
)


# values produced by student code
def render_value(value: Any, *, limit: int = 160) -> str:
    """A repr() of a value the student's function returned, built without
    calling any method the student could have defined.
    """
    text = _render(value, 0)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render(value: Any, depth: int) -> str:
    kind = type(value)
    if value is None or kind is bool or kind is float:
        return repr(value)
    if kind is int:
        # repr of a huge int is slow, and refused past 4300 digits
        bits = value.bit_length()
        return repr(value) if bits <= 200 else f"<int of {bits} bits>"
    if kind is str:
        return repr(value) if len(value) <= 80 else repr(value[:80]) + "…"
    if depth >= 4:
        return "…"
    if kind is list or kind is tuple:
        items = [_render(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"… {len(value) - 12} more")
        if kind is tuple:
            return "(" + ", ".join(items) + ("," if len(value) == 1 else "") + ")"
        return "[" + ", ".join(items) + "]"
    if kind is dict:
        pairs = []
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                pairs.append(f"… {len(value) - 12} more")
                break
            pairs.append(f"{_render(key, depth + 1)}: {_render(item, depth + 1)}")
        return "{" + ", ".join(pairs) + "}"
    if kind is set or kind is frozenset:
        if not value:
            return f"{kind.__name__}()"
        items = []
        for index, item in enumerate(value):
            if index >= 12:
                items.append(f"… {len(value) - 12} more")
                break
            items.append(_render(item, depth + 1))
        body = "{" + ", ".join(items) + "}"
        return body if kind is set else f"frozenset({body})"
    # a class whose metaclass is plain `type` cannot intercept __name__
    if type(kind) is type:
        return f"<{kind.__name__} object>"
    return "<object>"


def safe_equivalent(a: Any, b: Any) -> bool:
    """The output comparison of the counterexample search, restricted to exact
    builtin types so that it never runs the student's __eq__.
    """
    ka, kb = type(a), type(b)
    if ka in (list, tuple) and kb in (list, tuple):
        return len(a) == len(b) and all(safe_equivalent(x, y) for x, y in zip(a, b))
    if ka is bool or kb is bool:
        return ka is kb and a == b
    numeric = (int, float)
    if ka in numeric and kb in numeric:
        if (ka is float and not math.isfinite(a)) or (kb is float and not math.isfinite(b)):
            return a == b
        return abs(a - b) < 1e-9
    plain = (type(None), str)
    if ka in plain and kb in plain:
        return a == b
    if ka is dict and kb is dict:
        return a.keys() == b.keys() and all(safe_equivalent(a[k], b[k]) for k in a)
    return False


# the gate, once more on the way out
class Boundary:
    """The answer-leakage gate for one (reference, attempt) pair."""

    def __init__(self, verifier: ComplianceVerifier, problem: ProblemSpec, source: str, limit: float) -> None:
        self.context = verifier.context_for(problem.reference_solution, source)
        self.limit = limit
        self.replaced = 0

    def passes(self, text: str) -> bool:
        return not self.context.score(text or "").rejected(self.limit)

    def clean(self, text: str, replacement: str) -> str:
        if self.passes(text):
            return text
        self.replaced += 1
        return replacement


# small pieces
def _num(value: Optional[float], places: int = 4) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return round(value, places) if math.isfinite(value) else None


def concept_ref(cid: Optional[str], graph: KnowledgeGraph) -> Optional[Dict[str, Any]]:
    if not cid:
        return None
    return {"id": cid, "name": graph.name(cid) if cid in graph else cid}


def misconception_ref(mid: str) -> Dict[str, Any]:
    entry = MISCONCEPTIONS.get(mid)
    return {"id": mid, "name": entry.name if entry else mid}


def task_view(problem: ProblemSpec, graph: KnowledgeGraph) -> Dict[str, Any]:
    """A task as the learner may see it: the statement, never the oracle."""
    if problem.pid == "freeform":
        return {
            "id": "freeform",
            "title": "Any function",
            "freeform": True,
            "entry_point": problem.entry_point,
        }
    return {
        "id": problem.pid,
        "title": problem.title,
        "statement": problem.statement,
        "entry_point": problem.entry_point,
        "parameters": list(problem.parameters),
        "signature": f"def {problem.entry_point}({', '.join(problem.parameters)}):",
        "difficulty": _num(problem.difficulty, 2),
        "concepts": [concept_ref(cid, graph) for cid in problem.concepts],
        "freeform": False,
    }


def notes_from_warnings(warnings: Iterable[str]) -> List[Dict[str, str]]:
    """Telemetry warnings become fixed sentences. The original text never
    leaves: a gate rejection quotes the text it rejected.
    """
    notes: List[Dict[str, str]] = []
    seen = set()
    for warning in warnings:
        note: Optional[Tuple[str, str]] = None
        for prefix, kind, sentence in _WARNING_NOTES:
            if warning.startswith(prefix):
                note = (kind, sentence)
                break
        if note is None and " failed: " in warning:
            note = ("stage", _STAGE_FAILED)
        if note is not None and note not in seen:
            seen.add(note)
            notes.append({"kind": note[0], "text": note[1]})
    return notes


# one attempt
def verdict_status(result: TutoringResult) -> str:
    diagnosis = result.diagnosis
    if any(w.startswith(f"{SocraticEvaluatorAgent.name} failed") for w in result.telemetry.warnings):
        # the default empty diagnosis is not "no finding": nothing was compared
        return "incomplete"
    if diagnosis.hits:
        return "provisional" if diagnosis.inconclusive else "diagnosis"
    if diagnosis.counterexample is not None:
        # the failing input is found and localised but no rule names the belief
        return "failing"
    return "clean" if diagnosis.checks_total else "structural-clean"


def attempt_view(
    result: TutoringResult,
    *,
    index: int,
    source: str,
    config: Config,
    graph: KnowledgeGraph,
    boundary: Boundary,
) -> Dict[str, Any]:
    """Everything about one attempt the learner may see before asking for
    anything. The ladder and the challenge are not in it.
    """
    diagnosis = result.diagnosis
    primary = diagnosis.primary
    has_reference = bool(result.problem.reference_solution.strip())

    findings = [
        {
            **misconception_ref(hit.misconception_id),
            "belief": _num(hit.belief),
            "severity": str(hit.severity),
            "line": hit.span.line if hit.span else None,
            "sources": [str(kind) for kind in hit.kinds],
        }
        for hit in diagnosis.hits
    ]

    belief = None
    evidence: List[Dict[str, Any]] = []
    if primary is not None:
        entry = MISCONCEPTIONS.get(primary.misconception_id)
        if entry is not None:
            belief = {"voice": entry.student_voice, "description": entry.description}
        for support in sorted(primary.supports, key=lambda e: -e.belief)[:4]:
            evidence.append(
                {
                    "source": str(support.kind),
                    "text": boundary.clean(support.rationale, WITHHELD),
                    "line": support.span.line if support.span else None,
                }
            )

    witness = None
    counterexample = diagnosis.counterexample
    if counterexample is not None:
        witness = {
            "input": render_value(counterexample.args),
            "yours": None if counterexample.crashed else render_value(counterexample.student_output),
            "required": render_value(counterexample.reference_output),
            "error": _short(counterexample.student_error),
            "steps": counterexample.shrink_steps,
        }

    divergence = None
    point = diagnosis.divergence
    if point is not None:
        divergence = {
            "kind": str(point.kind),
            "line": point.student_line,
            "checkpoint": point.step,
            "snippet": point.span.snippet.strip() if point.span else None,
        }

    complexity = {
        "yours": diagnosis.student_complexity,
        "target": diagnosis.reference_complexity if has_reference else None,
        "gap": bool(diagnosis.complexity_gap) if has_reference else False,
    }

    intervention = result.intervention
    zpd_low, zpd_high = config.intervention.zpd_low, config.intervention.zpd_high
    alignment = result.alignment
    root = alignment.root_cause
    target = intervention.target_concept
    view_alignment = {
        "root_cause": (
            {**concept_ref(root, graph), "depth": graph.depth(root) if root in graph else None}
            if root else None
        ),
        "target": concept_ref(target, graph),
        "probability": _num(intervention.zpd_probability),
        "in_zpd": zpd_low <= intervention.zpd_probability <= zpd_high,
        "paths": [
            [concept_ref(cid, graph) for cid in path]
            for path in alignment.prerequisite_paths[:2]
        ],
        "blame": [
            {
                **concept_ref(cid, graph),
                "mass": _num(mass),
                "before": _num(alignment.mastery_before.get(cid)),
                "after": _num(alignment.mastery_after.get(cid)),
            }
            for cid, mass in alignment.top_blame(6)
        ],
    }

    return {
        "index": index,
        "task": task_view(result.problem, graph),
        "source": source,
        "status": verdict_status(result),
        "checks": (
            {"passed": diagnosis.checks_passed, "total": diagnosis.checks_total}
            if diagnosis.checks_total else None
        ),
        "findings": findings,
        "belief": belief,
        "evidence": evidence,
        "witness": witness,
        "divergence": divergence,
        "complexity": complexity,
        "alignment": view_alignment,
        "gate": {
            "leakage": _num(intervention.leakage),
            "grounding": _num(intervention.grounding),
            "limit": config.intervention.max_leakage,
            "replaced": intervention.regenerations,
        },
        "ladder": {"total": len(intervention.hints)},
        "challenge": {"available": intervention.challenge is not None},
        "notes": notes_from_warnings(result.telemetry.warnings),
        "backend": result.telemetry.backend,
    }


def hint_view(hint: Hint, *, rung: int, total: int, graph: KnowledgeGraph, boundary: Boundary) -> Dict[str, Any]:
    """One rung. The gate already approved it inside the pipeline; it is
    checked again here and replaced by the pipeline's own last-resort rung if
    anything slipped through.
    """
    fallback = _FALLBACK_RUNGS[min(rung - 1, len(_FALLBACK_RUNGS) - 1)]
    return {
        "rung": rung,
        "total": total,
        "kind": str(hint.kind),
        "text": boundary.clean(hint.text, fallback),
        "concept": concept_ref(hint.concept_id, graph),
        "last": rung == total,
    }


_FALLBACK_PROMPT = (
    "Before running anything, write down what you predict your function returns for the "
    "smallest input you can think of, and why. Then run it and compare."
)


def challenge_view(challenge: Challenge, *, boundary: Boundary) -> Dict[str, Any]:
    """The predict-then-run task without its expected insight, which the gate
    never sees.
    """
    combined = f"{challenge.prompt} {challenge.trace_question}"
    if boundary.passes(combined):
        prompt, question = challenge.prompt, challenge.trace_question
    else:
        boundary.replaced += 1
        prompt, question = _FALLBACK_PROMPT, "Which of your predictions did the run contradict?"
    return {
        "prompt": prompt,
        "question": question,
        "input": render_value(challenge.args) if challenge.args is not None else None,
        "runnable": challenge.args is not None,
    }


def diff_view(previous: Carry, current: Carry, graph: KnowledgeGraph) -> Dict[str, Any]:
    """Attempt n against attempt n-1, by the rules of the terminal's diff."""
    before = set(previous.findings)
    after = set(current.findings)

    checks = None
    if previous.checks and current.checks:
        checks = {
            "before": {"passed": previous.checks[0], "total": previous.checks[1]},
            "after": {"passed": current.checks[0], "total": current.checks[1]},
        }

    # the concept the learner was working on, not whatever the new diagnosis
    # points at
    focus = previous.target_concept or previous.root_cause or current.target_concept
    mastery = None
    if focus and focus in previous.posterior and focus in current.posterior:
        mastery = {
            **concept_ref(focus, graph),
            "before": _num(previous.posterior[focus]),
            "after": _num(current.posterior[focus]),
        }
    return {
        "resolved": [misconception_ref(mid) for mid in sorted(before - after)],
        "persists": [misconception_ref(mid) for mid in sorted(before & after)],
        "new": [misconception_ref(mid) for mid in sorted(after - before)],
        # a finding that disappears while an input still fails is not a repair
        "still_failing": current.failing,
        "checks": checks,
        "mastery": mastery,
    }


def summary_view(index: int, result: TutoringResult) -> Dict[str, Any]:
    """One row of the session's history."""
    diagnosis = result.diagnosis
    primary = diagnosis.primary
    return {
        "index": index,
        "task": result.problem.pid,
        "status": verdict_status(result),
        "finding": misconception_ref(primary.misconception_id) if primary else None,
        "belief": _num(primary.belief) if primary else None,
        "checks": (
            {"passed": diagnosis.checks_passed, "total": diagnosis.checks_total}
            if diagnosis.checks_total else None
        ),
    }


def mastery_view(state: Optional[MasteryState], graph: KnowledgeGraph, threshold: float) -> Dict[str, Any]:
    """Only the concepts this session has evidence about. The cold-start prior
    covers all of them, and listing Dijkstra as "not yet" for someone who only
    tried binary search is noise.
    """
    concepts: List[Dict[str, Any]] = []
    if state is not None:
        for cid, count in state.observations.items():
            if count <= 0 or cid not in state.posterior:
                continue
            value = state.posterior[cid]
            concepts.append(
                {
                    **concept_ref(cid, graph),
                    "value": _num(value),
                    "secure": value >= threshold,
                    "observations": count,
                    "stratum": graph.concept(cid).stratum if cid in graph else None,
                }
            )
    concepts.sort(key=lambda c: (c["value"], c["id"]))
    return {
        "threshold": threshold,
        "concepts": concepts,
        "below": sum(1 for c in concepts if not c["secure"]),
    }


def concept_view(
    cid: str,
    graph: KnowledgeGraph,
    *,
    mastery: Optional[MasteryState] = None,
    depth: int = 3,
) -> Dict[str, Any]:
    concept = graph.concept(cid)

    def tree(node: str, level: int) -> Dict[str, Any]:
        ref = concept_ref(node, graph)
        if level < depth:
            ref["prerequisites"] = [tree(p, level + 1) for p in graph.prerequisites(node)]
        else:
            ref["prerequisites"] = []
            ref["more"] = bool(graph.prerequisites(node))
        return ref

    observed = None
    if mastery is not None and mastery.observations.get(cid, 0) > 0:
        observed = _num(mastery.posterior.get(cid))
    return {
        "id": cid,
        "name": concept.name,
        "stratum": concept.stratum,
        "depth": graph.depth(cid),
        "summary": concept.summary,
        "prerequisites": [tree(p, 1) for p in graph.prerequisites(cid)],
        "unlocks": [concept_ref(d, graph) for d in graph.dependents(cid)],
        "mastery": observed,
    }


def _short(text: Optional[str], limit: int = 200) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"
