"""What runs inside the sandbox child, and nothing else.

A diagnosis and a challenge run are the only two things that execute a
student's code. Both happen in a short-lived child process (see sandbox.py),
so this module is the one place student code is actually run on the server.

Each function takes a plain dict and returns a plain dict. Everything that
crosses back to the parent is built here by `views`, which reads student
values through `render_value` only and never calls their methods, so the
result is composed of built-in types the parent can trust. The raw
TutoringResult, the reasoning model's chain of thought, the gate's log of
rejected candidates and the reference solution all live and die in this
process; none of them is ever put into what is returned.
"""

import contextlib
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..agents.orchestrator import DeductiveOrchestrator
from ..agents.verifier import ComplianceVerifier
from ..analysis.tracer import GuardedRunner
from ..cli.session import Carry, freeform_problem, settle_mastery
from ..config import Config
from ..data.problems import problem as get_problem
from ..domain import ProblemSpec, Submission
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState
from ..pipeline import resolve_entry
from . import views
from .views import Boundary


@dataclass
class SandboxContext:
    """The static, prebuilt machinery a job needs. Created once in the parent
    and inherited by the child through fork, so a job pays nothing to set up.
    """

    config: Config
    orchestrator: DeductiveOrchestrator
    graph: KnowledgeGraph
    verifier: ComplianceVerifier
    runner: GuardedRunner


def build_context(
    config: Config, orchestrator: Optional[DeductiveOrchestrator] = None
) -> SandboxContext:
    orchestrator = orchestrator or DeductiveOrchestrator(config)
    return SandboxContext(
        config=config,
        orchestrator=orchestrator,
        graph=orchestrator.graph,
        verifier=orchestrator.verification.verifier,
        runner=GuardedRunner(config.execution),
    )


def _spec(ctx: SandboxContext, task: Tuple[str, str]) -> ProblemSpec:
    kind, value = task
    return get_problem(value) if kind == "known" else freeform_problem(value)


def _mastery(data: Optional[Dict[str, Any]]) -> Optional[MasteryState]:
    if not data:
        return None
    return MasteryState(
        posterior=dict(data.get("posterior", {})),
        observations=dict(data.get("observations", {})),
    )


def _carry(data: Optional[Dict[str, Any]]) -> Optional[Carry]:
    if not data:
        return None
    return Carry(
        findings=tuple(data["findings"]),
        target_concept=data["target_concept"],
        root_cause=data["root_cause"],
        checks=tuple(data["checks"]) if data["checks"] is not None else None,
        failing=data["failing"],
        posterior=dict(data["posterior"]),
    )


def carry_data(carry: Carry) -> Dict[str, Any]:
    """A Carry as a plain dict, for the trip to the child and back."""
    return {
        "findings": list(carry.findings),
        "target_concept": carry.target_concept,
        "root_cause": carry.root_cause,
        "checks": list(carry.checks) if carry.checks is not None else None,
        "failing": carry.failing,
        "posterior": dict(carry.posterior),
    }


def run_diagnose(ctx: SandboxContext, payload: Dict[str, Any]) -> Dict[str, Any]:
    """One full diagnose -> align -> intervene pass, projected to what the
    browser may see. The student's code runs here, never in the parent.
    """
    source = payload["source"]
    spec = _spec(ctx, payload["task"])
    index = payload["index"]
    mastery = _mastery(payload.get("mastery"))
    previous = _carry(payload.get("previous"))
    notes = payload.get("notes", [])

    submission = Submission(
        sid=f"editor#{index}", problem_id=spec.pid, source=source, author="session"
    )
    with _quiet():
        result = ctx.orchestrator.tutor(spec, submission, mastery=mastery)
    mastery = settle_mastery(ctx.orchestrator, mastery, result, previous, ctx.graph)

    boundary = Boundary(
        ctx.verifier, result.problem, source, ctx.config.intervention.max_leakage
    )
    view = views.attempt_view(
        result, index=index, source=source, config=ctx.config, graph=ctx.graph,
        boundary=boundary,
    )
    view["notes"] = list(notes) + view["notes"]

    hints = result.intervention.hints
    sealed = [
        views.hint_view(hint, rung=rung, total=len(hints), graph=ctx.graph, boundary=boundary)
        for rung, hint in enumerate(hints, start=1)
    ]
    challenge = result.intervention.challenge
    counterexample = result.diagnosis.counterexample
    current = Carry.of(result)

    return {
        "view": view,
        "summary": views.summary_view(index, result),
        "hints": sealed,
        "challenge": (
            views.challenge_view(challenge, boundary=boundary) if challenge is not None else None
        ),
        "challenge_args": (
            list(challenge.args)
            if challenge is not None and challenge.args is not None else None
        ),
        "has_required": counterexample is not None,
        "diff": (
            views.diff_view(previous, current, ctx.graph) if previous is not None else None
        ),
        "carry": carry_data(current),
        "mastery": (
            {"posterior": dict(mastery.posterior), "observations": dict(mastery.observations)}
            if mastery is not None else None
        ),
        "replaced": boundary.replaced,
    }


def run_challenge(ctx: SandboxContext, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the editor's code on the challenge's input, and -- for a task with
    a reference -- the reference too, so the comparison is exact and the
    reference's value is computed here rather than carried around.
    """
    source = payload["source"]
    spec = _spec(ctx, payload["task"])
    entry = resolve_entry(source, payload.get("entry") or spec.entry_point)
    args = list(payload.get("args") or [])

    with _quiet():
        outcome = ctx.runner.run(source, entry, _copy(args), trace=False)

        required: Any = None
        has_required = False
        if payload.get("has_required") and spec.reference_solution.strip():
            reference = ctx.runner.run(
                spec.reference_solution, spec.entry_point, _copy(args), trace=False
            )
            if reference.ok:
                required, has_required = reference.value, True

    run = {
        "input": views.render_value(tuple(args)),
        "ok": outcome.ok,
        "value": views.render_value(outcome.value) if outcome.ok else None,
        "error": None if outcome.ok else _error_line(outcome.error_type, outcome.error),
        "budget": bool(outcome.timed_out or outcome.recursion_overflow),
        "required": views.render_value(required) if has_required else None,
        "agrees": (
            views.safe_equivalent(outcome.value, required)
            if outcome.ok and has_required else None
        ),
    }
    return {"run": run}


class _Discard(io.TextIOBase):
    """Where a submission's print() goes: not into the server's log."""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Silence anything the student's code prints, at the Python level. In a
    fork worker a file-descriptor redirect also catches writes from C.
    """
    with contextlib.redirect_stdout(_Discard()):
        yield


def _copy(args: List[Any]) -> List[Any]:
    import copy

    return [copy.deepcopy(arg) for arg in args]


def _error_line(kind: Optional[str], message: Optional[str]) -> str:
    text = ": ".join(part for part in (kind, message) if part)
    return text if len(text) <= 200 else text[:199] + "…"
