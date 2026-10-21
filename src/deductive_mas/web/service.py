"""The terminal session's rules behind an API.

Thin by design. Diagnosis, mastery and the ladder belong to the
orchestrator and to cli/session.py, whose rules (task detection, the repair
credit) are called here rather than copied. This module adds what a server
needs and a terminal does not: many learners at once, input checking, and a
hard boundary around the code it runs.

That boundary is a process. The orchestrator and the student's code run in a
short-lived child (see sandbox.py); the HTTP process holds only a session's
plain, already-projected state and never executes student code itself. So
the reasoning model's chain of thought, the gate's rejected candidates and
the reference solution exist only in a child and die with it, and a
submission that overruns its CPU or memory is killed without touching
anyone else. The step budget and the deadline inside the interpreter are
unchanged; the process ceilings sit outside them.
"""

import ast
import secrets
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..cli.session import Carry, MAX_SOURCE_BYTES, detect_problem
from ..config import Config
from ..data.problems import all_problems, problem as get_problem
from ..data.submissions import all_submissions
from ..domain import ProblemSpec
from ..knowledge.mastery import MasteryState
from ..version import __version__
from . import views
from .jobs import SandboxContext, build_context, carry_data
from .sandbox import Outcome, make_sandbox

# analyses (diagnoses and challenge runs) one session may start per minute
ANALYSES_PER_MINUTE = 20
MAX_ATTEMPTS = 200


class WebError(Exception):
    """A request the service refuses, with what the client may be told."""

    def __init__(self, status: int, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail

    def payload(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, **self.detail}}


@dataclass
class Projection:
    """One attempt as the client may see it, built once from what the child
    returned. Everything here is plain data; the raw result never existed in
    this process.
    """

    view: Dict[str, Any]
    summary: Dict[str, Any]
    hints: Tuple[Dict[str, Any], ...] = ()
    challenge: Optional[Dict[str, Any]] = None
    challenge_args: Optional[List[Any]] = None
    has_required: bool = False
    diff: Optional[Dict[str, Any]] = None


@dataclass
class AttemptRecord:
    """What the session keeps about one diagnosed attempt: its projection for
    the client, and the plain summary of beliefs the next diff and the repair
    credit need.
    """

    index: int
    source: str
    carry: Carry
    projection: Projection


@dataclass
class WebSession:
    """One learner: the small amount of plain state the server keeps between
    requests. No TutoringResult and no student value is among it.
    """

    token: str
    mastery: Optional[MasteryState] = None
    problem: Optional[ProblemSpec] = None
    source: str = ""
    attempts: List[AttemptRecord] = field(default_factory=list)
    revealed: int = 0
    challenge_shown: bool = False
    task_mode: Optional[str] = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    recent: Deque[float] = field(default_factory=deque, repr=False)

    @property
    def latest(self) -> Optional[AttemptRecord]:
        return self.attempts[-1] if self.attempts else None

    @property
    def previous_carry(self) -> Optional[Carry]:
        return self.attempts[-1].carry if self.attempts else None

    def spend(self, now: float, *, limit: int = ANALYSES_PER_MINUTE, window: float = 60.0) -> None:
        while self.recent and now - self.recent[0] > window:
            self.recent.popleft()
        if len(self.recent) >= limit:
            wait = int(window - (now - self.recent[0])) + 1
            raise WebError(
                429, "slow_down",
                "That is a lot of runs in a minute. Take a breath, read the last result, "
                "and try again shortly.",
                retry_after=wait,
            )
        self.recent.append(now)


class SessionStore:
    """In-memory sessions behind an opaque token, with an idle timeout and a cap."""

    def __init__(self, *, ttl: float = 8 * 3600.0, capacity: int = 500) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self._sessions: Dict[str, WebSession] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, token: Optional[str]) -> Optional[WebSession]:
        if not token:
            return None
        now = time.time()
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if now - session.touched > self.ttl:
                del self._sessions[token]
                return None
            session.touched = now
            return session

    def create(self) -> WebSession:
        now = time.time()
        with self._lock:
            for token in [t for t, s in self._sessions.items() if now - s.touched > self.ttl]:
                del self._sessions[token]
            while len(self._sessions) >= self.capacity:
                self._evict_one()
            session = WebSession(token=secrets.token_urlsafe(32))
            self._sessions[session.token] = session
            return session

    def _evict_one(self) -> None:
        """Drop one session at capacity, oldest first, but never one whose lock
        is held: that session is mid-request (a blocked diagnosis freezes its
        `touched`, which is exactly what would make it look evictable), and
        evicting it would lose a learner's history under their feet. Fall back
        to the plain oldest only if every session is busy.
        """
        for session in sorted(self._sessions.values(), key=lambda s: s.touched):
            if session.lock.acquire(blocking=False):
                try:
                    del self._sessions[session.token]
                finally:
                    session.lock.release()
                return
        oldest = min(self._sessions.values(), key=lambda s: s.touched)
        del self._sessions[oldest.token]


class TutorService:
    """Everything the HTTP layer can ask for."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        orchestrator: Optional[Any] = None,
        store: Optional[SessionStore] = None,
        demo: bool = False,
        isolate: bool = True,
    ) -> None:
        self.config = config or Config()
        context: SandboxContext = build_context(self.config, orchestrator)
        self.orchestrator = context.orchestrator  # held for reference data only
        self.graph = context.graph
        self.sessions = store or SessionStore()
        self.demo = demo
        self.sandbox = make_sandbox(context, self.config, isolate=isolate)
        self.isolated = isolate and type(self.sandbox).__name__ == "ForkSandbox"
        # rungs the boundary had to replace: the pipeline's own gate should
        # make this stay at zero, so anything else is worth a line in the log
        self.boundary_replacements = 0

    def close(self) -> None:
        self.sandbox.close()

    # reference data
    def meta(self) -> Dict[str, Any]:
        return {
            "version": __version__,
            "backend": self.orchestrator.reasoner.name,
            "demo": self.demo,
            "isolated": self.isolated,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "mastery_threshold": self.config.knowledge.mastery_threshold,
            "zpd": [self.config.intervention.zpd_low, self.config.intervention.zpd_high],
            "leakage_limit": self.config.intervention.max_leakage,
        }

    def tasks(self) -> List[Dict[str, Any]]:
        return [views.task_view(spec, self.graph) for spec in all_problems()]

    def examples(self) -> List[Dict[str, Any]]:
        """Flawed attempts from the bank, for a demonstration. Off unless the
        server was started with --demo: a correct program from the bank is a
        solution, and a flawed one is a near-solution of a task the learner
        may be working on.
        """
        if not self.demo:
            raise WebError(404, "not_found", "examples are only available in demo mode")
        dev = {spec.pid for spec in all_problems()}
        return [
            {"id": sub.sid, "task": sub.problem_id, "source": sub.source}
            for sub in all_submissions()
            if not sub.functionally_correct and sub.problem_id in dev
        ]

    def concept(self, cid: str, session: Optional[WebSession] = None) -> Dict[str, Any]:
        if cid not in self.graph:
            raise WebError(404, "unknown_concept", f"there is no concept {cid!r} in the curriculum")
        mastery = session.mastery if session is not None else None
        return views.concept_view(cid, self.graph, mastery=mastery)

    # the session
    def snapshot(self, session: WebSession) -> Dict[str, Any]:
        with session.lock:  # a GET /api/session may race a POST that is mutating it
            return self._snapshot(session)

    def _snapshot(self, session: WebSession) -> Dict[str, Any]:
        current = None
        latest = session.latest
        if latest is not None:
            projection = latest.projection
            current = dict(projection.view)
            current["ladder"] = {
                "total": len(projection.hints),
                "revealed": list(projection.hints[: session.revealed]),
            }
            challenge: Dict[str, Any] = {
                "available": projection.challenge is not None,
                "shown": session.challenge_shown,
            }
            if session.challenge_shown and projection.challenge is not None:
                challenge.update(projection.challenge)
            current["challenge"] = challenge
            current["diff"] = projection.diff
        return {
            "attempts": [record.projection.summary for record in session.attempts],
            "current": current,
            "mastery": views.mastery_view(
                session.mastery, self.graph, self.config.knowledge.mastery_threshold
            ),
            "task": views.task_view(session.problem, self.graph) if session.problem else None,
            "task_mode": session.task_mode,
        }

    def submit(self, session: WebSession, source: Any, task: Any) -> Dict[str, Any]:
        """Load and retry from the terminal, decided by whether this is the
        task the last attempt answered.
        """
        if not isinstance(task, str) or not task:
            raise WebError(400, "bad_request", "say which task this code answers, or 'auto'")
        with session.lock:
            if len(session.attempts) >= MAX_ATTEMPTS:
                raise WebError(
                    409, "attempt_limit",
                    "This session has reached its attempt limit. Start over to keep going.",
                )
            session.spend(time.time())
            source = check_source(source)

            # "auto" detects the task from the function on every submission:
            # the editor is not one file, a learner can paste anything into it.
            # The same function again is still a retry of the same task.
            matches: Optional[Tuple[ProblemSpec, ...]] = None
            if task == "auto":
                spec, matches = detect_problem(source)
            else:
                spec = _known_task(task)

            previous_spec = session.problem
            retry = (
                bool(session.attempts)
                and previous_spec is not None
                and spec.pid == previous_spec.pid
                and spec.entry_point == previous_spec.entry_point
            )
            notes = _detection_notes(spec, matches) if matches is not None and not retry else []
            session.task_mode = task
            if retry and source == session.source:
                return {
                    "unchanged": True,
                    "message": (
                        "The code has not changed since the last attempt. Edit it and retry, "
                        "or ask for another question."
                    ),
                    "session": self.snapshot(session),
                }

            index = len(session.attempts) + 1
            previous = session.previous_carry if retry else None
            payload = {
                "source": source,
                "task": _task_ref(spec),
                "index": index,
                "mastery": _mastery_data(session.mastery),
                "previous": carry_data(previous) if previous is not None else None,
                "notes": notes,
            }
            result = self._demand(self.sandbox.diagnose(payload), diagnosing=True)

            projection = self._build_projection(result)
            if result.get("mastery"):
                session.mastery = MasteryState(
                    posterior=dict(result["mastery"]["posterior"]),
                    observations=dict(result["mastery"]["observations"]),
                )
            session.problem, session.source = spec, source
            session.attempts.append(AttemptRecord(
                index=index, source=source, carry=_carry(result["carry"]), projection=projection
            ))
            session.revealed = 0
            session.challenge_shown = False
            self._log_replacements(index, result)
            return {"unchanged": False, "retry": retry, "session": self.snapshot(session)}

    def reveal(self, session: WebSession, index: Any, rung: Any) -> Dict[str, Any]:
        """The next rung, and only the next rung. Asking again for a rung
        already shown returns it unchanged, so a double click costs nothing;
        asking past the next one is refused.
        """
        if not isinstance(rung, int) or isinstance(rung, bool) or rung < 1:
            raise WebError(400, "bad_request", "rung must be a positive integer")
        with session.lock:
            projection = self._latest(session, index)
            total = len(projection.hints)
            if not total:
                return {
                    "status": "none",
                    "message": "There is no guidance for this attempt: nothing was diagnosed.",
                    "session": self.snapshot(session),
                }
            if rung <= session.revealed:
                return {"status": "revealed", "hint": projection.hints[rung - 1],
                        "session": self.snapshot(session)}
            if rung > total:
                return {
                    "status": "exhausted",
                    "message": "That was the last question. Try the challenge, or edit and retry.",
                    "session": self.snapshot(session),
                }
            if rung != session.revealed + 1:
                raise WebError(
                    409, "out_of_order",
                    "Questions open one at a time.",
                    next=session.revealed + 1,
                )
            session.revealed = rung
            return {"status": "revealed", "hint": projection.hints[rung - 1],
                    "session": self.snapshot(session)}

    def open_challenge(self, session: WebSession, index: Any) -> Dict[str, Any]:
        with session.lock:
            projection = self._latest(session, index)
            if projection.challenge is None:
                return {
                    "status": "none",
                    "message": "No counter-factual challenge was produced for this attempt.",
                    "session": self.snapshot(session),
                }
            session.challenge_shown = True
            return {"status": "shown", "challenge": projection.challenge,
                    "session": self.snapshot(session)}

    def run_challenge(self, session: WebSession, index: Any, source: Any) -> Dict[str, Any]:
        """Predict, then run: the code in the editor on the challenge's input,
        under the same guarded runner and the same budgets as a diagnosis, in
        the same kind of child process.
        """
        with session.lock:
            projection = self._latest(session, index)
            if not session.challenge_shown or projection.challenge is None:
                raise WebError(409, "challenge_closed", "Open the challenge before running it.")
            if projection.challenge_args is None:
                raise WebError(409, "not_runnable", "This challenge has no input to run.")
            session.spend(time.time())
            spec = session.problem
            assert spec is not None
            source = check_source(source)
            payload = {
                "source": source,
                "task": _task_ref(spec),
                "entry": spec.entry_point,
                "args": list(projection.challenge_args),
                "has_required": projection.has_required,
            }
            result = self._demand(self.sandbox.challenge(payload), diagnosing=False)
            return {"run": result["run"]}

    def reset(self, session: WebSession) -> Dict[str, Any]:
        """Forget the mastery estimate and the history, like the terminal's reset."""
        with session.lock:
            session.mastery = None
            session.attempts = []
            session.revealed = 0
            session.challenge_shown = False
            return {"session": self.snapshot(session)}

    # internals
    def _demand(self, outcome: Outcome, *, diagnosing: bool) -> Dict[str, Any]:
        """Turn a sandbox outcome into either the plain result or the right
        refusal. A diagnosis that overran its ceiling is not an attempt; a
        challenge run that overran comes back as a run that hit the budget; a
        fault in the analyser itself is a server error, not the learner's.
        """
        if outcome.ok and outcome.result is not None:
            return outcome.result
        if outcome.status == "down":
            raise WebError(
                503, "unavailable",
                "The analysis service is not running. Ask whoever is hosting the pilot to restart it.",
            )
        if outcome.status == "error":
            # a caught exception inside the analyser, not an overrun: a server
            # fault. The type name carries no student content; it is logged,
            # never shown, and the HTTP layer turns the raise into a fixed 500.
            sys.stderr.write(f"dmas web: the analyser failed with {outcome.message or 'an error'}\n")
            raise WebError(500, "internal", "the server failed to handle the request")
        if outcome.status == "timeout" and diagnosing:
            # the wall-clock, not the CPU ceiling, killed the worker: it was
            # blocked (in practice on the reasoning model), not spinning in the
            # learner's code. Say so honestly instead of blaming their program.
            sys.stderr.write("dmas web: an analysis passed its wall-clock deadline (backend slow?)\n")
            raise WebError(
                503, "overloaded",
                "The tutor took too long to respond this time; the reasoning service may be busy. "
                "Your code was not the problem — please try again in a moment.",
            )
        if not diagnosing:
            # the editor's code blew its ceiling (limit) or died (crash) on the
            # challenge input; either way the run hit the wall. A challenge
            # makes no model calls, so a wall "timeout" here is near-impossible.
            return {"run": {
                "input": None, "ok": False, "value": None,
                "error": "The code used more time or memory than the tutor allows.",
                "budget": True, "required": None, "agrees": None,
            }}
        if outcome.status == "limit":
            raise WebError(
                422, "did_not_finish",
                "Your program used more time or memory than the tutor allows, so it could not be "
                "analysed. Look for a loop that never ends, unbounded recursion, or a value that "
                "grows without limit.",
            )
        raise WebError(
            422, "did_not_finish",
            "The analysis could not be completed for this code. Simplify it and try again.",
        )

    def _latest(self, session: WebSession, index: Any) -> Projection:
        if not isinstance(index, int) or isinstance(index, bool):
            raise WebError(400, "bad_request", "attempt must be an integer")
        latest = session.latest
        if latest is None:
            raise WebError(409, "no_attempt", "Diagnose some code first.")
        if index != latest.index:
            raise WebError(
                409, "stale_attempt",
                "Guidance belongs to the latest attempt only.",
                latest=latest.index,
            )
        return latest.projection

    def _build_projection(self, result: Dict[str, Any]) -> Projection:
        return Projection(
            view=result["view"],
            summary=result["summary"],
            hints=tuple(result["hints"]),
            challenge=result["challenge"],
            challenge_args=result["challenge_args"],
            has_required=result["has_required"],
            diff=result["diff"],
        )

    def _log_replacements(self, index: int, result: Dict[str, Any]) -> None:
        replaced = int(result.get("replaced", 0))
        if replaced:
            self.boundary_replacements += replaced
            backend = result.get("view", {}).get("backend", "?")
            sys.stderr.write(
                f"dmas web: the answer-leakage check on the way out replaced {replaced} "
                f"item(s) of attempt {index} ({backend})\n"
            )


# helpers
def check_source(source: Any) -> str:
    """Refuse what is not an attempt before it reaches the pipeline.

    A file that does not parse is not evidence about a belief: the pipeline
    would report no finding, and the session would credit the concept as
    repaired. So it never becomes an attempt; the learner gets the line and
    column instead, which is what an editor owes them anyway. This runs in the
    HTTP process because parsing and compiling execute nothing.
    """
    if not isinstance(source, str):
        raise WebError(400, "bad_request", "source must be text")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise WebError(413, "too_large", f"The code is larger than {MAX_SOURCE_BYTES // 1024} KB.")
    if not source.strip():
        raise WebError(422, "empty", "The file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
            # the compiler finds what the parser lets through (return outside
            # a function, break outside a loop); it does not run anything
            compile(tree, "<attempt>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise WebError(
            422, "syntax_error", exc.msg or "invalid syntax",
            line=exc.lineno, column=exc.offset,
            end_line=getattr(exc, "end_lineno", None), end_column=getattr(exc, "end_offset", None),
        )
    except (RecursionError, MemoryError, ValueError, OverflowError):
        raise WebError(422, "syntax_error", "The code is too deeply nested to analyse.")
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)):
        raise WebError(
            422, "no_function",
            "Define a function: the tutor calls it with inputs and compares the answers.",
        )
    return source


def _task_ref(spec: ProblemSpec) -> Tuple[str, str]:
    """How the child is told which task this is, without shipping the spec
    (and its reference solution): rebuild it there from the id.
    """
    if spec.pid == "freeform":
        return ("freeform", spec.entry_point)
    return ("known", spec.pid)


def _mastery_data(state: Optional[MasteryState]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {"posterior": dict(state.posterior), "observations": dict(state.observations)}


def _carry(data: Dict[str, Any]) -> Carry:
    return Carry(
        findings=tuple(data["findings"]),
        target_concept=data["target_concept"],
        root_cause=data["root_cause"],
        checks=tuple(data["checks"]) if data["checks"] is not None else None,
        failing=data["failing"],
        posterior=dict(data["posterior"]),
    )


def _known_task(task: str) -> ProblemSpec:
    try:
        return get_problem(task)
    except KeyError:
        raise WebError(404, "unknown_task", f"there is no task {task!r}")


def _detection_notes(spec: ProblemSpec, matches: Tuple[ProblemSpec, ...]) -> List[Dict[str, str]]:
    if len(matches) > 1:
        return [{
            "kind": "task",
            "text": "Several tasks define {}(): {}. Using “{}”; pick the task explicitly to change it."
                    .format(spec.entry_point, ", ".join(m.title for m in matches), spec.title),
        }]
    if not matches:
        return [{
            "kind": "task",
            "text": "No built-in task defines {}(), so the tutor reports structural findings only: "
                    "no counterexample search and no complexity comparison."
                    .format(spec.entry_point),
        }]
    return [{"kind": "task", "text": f"Detected the task “{spec.title}” from {spec.entry_point}()."}]
