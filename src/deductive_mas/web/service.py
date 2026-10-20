"""The terminal session's rules behind an API.

Thin by design. Diagnosis, mastery and the ladder belong to the
orchestrator and to cli/session.py, whose rules (task detection, the repair
credit, the ladder reset) are called here rather than copied. This module
adds what a server needs and a terminal does not: many learners at once,
input checking, and one lock.

The lock: GuardedRunner lowers the interpreter's recursion limit for the
length of a run and restores it afterwards. That limit belongs to the whole
process, so two runs on two threads restore each other's value and leave
the process at the lowered one. Every execution of student code holds
`execution`: the whole orchestrator pass, and a challenge run. The step
budget and the deadline inside the runner are untouched.
"""

import ast
import contextlib
import copy
import io
import secrets
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from ..agents.orchestrator import DeductiveOrchestrator
from ..analysis.tracer import GuardedRunner
from ..cli.session import MAX_SOURCE_BYTES, SessionState, detect_problem, record_attempt
from ..config import Config
from ..data.problems import all_problems, problem as get_problem
from ..data.submissions import all_submissions
from ..domain import ProblemSpec
from ..pipeline import resolve_entry
from ..version import __version__
from . import views
from .views import Boundary

# analyses (diagnoses and challenge runs) one session may start per minute.
# They are serialised behind one lock, so a runaway client would otherwise
# queue everyone else behind it.
ANALYSES_PER_MINUTE = 20
MAX_ATTEMPTS = 200
# how long a request queues for the execution lock before it is told the
# tutor is busy, instead of piling up behind a run that will not finish
EXECUTION_WAIT = 45.0


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
    """One attempt as the client may see it, computed once. The ladder and the
    challenge are sealed in here and leave one request at a time.
    """

    view: Dict[str, Any]
    summary: Dict[str, Any]
    hints: Tuple[Dict[str, Any], ...] = ()
    challenge: Optional[Dict[str, Any]] = None
    challenge_args: Optional[Tuple[Any, ...]] = None
    required: Any = None
    has_required: bool = False
    diff: Optional[Dict[str, Any]] = None


@dataclass
class WebSession:
    """One learner: the terminal's SessionState plus what the web adds."""

    token: str
    state: SessionState = field(default_factory=SessionState)
    projections: Dict[int, Projection] = field(default_factory=dict)
    # the task the learner asked for last: a task id, or "auto"
    task_mode: Optional[str] = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    recent: Deque[float] = field(default_factory=deque, repr=False)

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
                oldest = min(self._sessions.values(), key=lambda s: s.touched)
                del self._sessions[oldest.token]
            session = WebSession(token=secrets.token_urlsafe(32))
            self._sessions[session.token] = session
            return session


class _Discard(io.TextIOBase):
    """Where a submission's print() goes: not into the server's log."""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)


class TutorService:
    """Everything the HTTP layer can ask for."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        orchestrator: Optional[DeductiveOrchestrator] = None,
        store: Optional[SessionStore] = None,
        demo: bool = False,
    ) -> None:
        self.config = config or Config()
        self.orchestrator = orchestrator or DeductiveOrchestrator(self.config)
        self.graph = self.orchestrator.graph
        self.verifier = self.orchestrator.verification.verifier
        self.runner = GuardedRunner(self.config.execution)
        self.sessions = store or SessionStore()
        self.demo = demo
        self.execution = threading.Lock()
        # rungs the boundary had to replace: the pipeline's own gate should
        # make this stay at zero, so anything else is worth a line in the log
        self.boundary_replacements = 0

    # reference data
    def meta(self) -> Dict[str, Any]:
        return {
            "version": __version__,
            "backend": self.orchestrator.reasoner.name,
            "demo": self.demo,
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
        mastery = session.state.mastery if session is not None else None
        return views.concept_view(cid, self.graph, mastery=mastery)

    # the session
    def snapshot(self, session: WebSession) -> Dict[str, Any]:
        state = session.state
        current = None
        latest = state.latest
        if latest is not None:
            projection = session.projections[latest.index]
            current = dict(projection.view)
            current["ladder"] = {
                "total": len(projection.hints),
                "revealed": list(projection.hints[: state.revealed]),
            }
            challenge: Dict[str, Any] = {
                "available": projection.challenge is not None,
                "shown": state.challenge_shown,
            }
            if state.challenge_shown and projection.challenge is not None:
                challenge.update(projection.challenge)
            current["challenge"] = challenge
            current["diff"] = projection.diff
        return {
            "attempts": [session.projections[a.index].summary for a in state.attempts],
            "current": current,
            "mastery": views.mastery_view(
                state.mastery, self.graph, self.config.knowledge.mastery_threshold
            ),
            "task": views.task_view(state.problem, self.graph) if state.problem else None,
            "task_mode": session.task_mode,
        }

    def submit(self, session: WebSession, source: Any, task: Any) -> Dict[str, Any]:
        """load and retry from the terminal, decided by whether this is the
        task the last attempt answered.
        """
        if not isinstance(task, str) or not task:
            raise WebError(400, "bad_request", "say which task this code answers, or 'auto'")
        with session.lock:
            state = session.state
            if len(state.attempts) >= MAX_ATTEMPTS:
                raise WebError(
                    409, "attempt_limit",
                    "This session has reached its attempt limit. Start over to keep going.",
                )
            session.spend(time.time())
            with self._exclusive():
                source = check_source(source)

            # "auto" detects the task from the function on every submission:
            # the editor is not one file, a learner can paste anything into it.
            # The same function again is still a retry of the same task.
            matches: Optional[Tuple[ProblemSpec, ...]] = None
            if task == "auto":
                spec, matches = detect_problem(source)
            else:
                spec = _known_task(task)

            previous_spec = state.problem
            retry = (
                bool(state.attempts)
                and previous_spec is not None
                and spec.pid == previous_spec.pid
                and spec.entry_point == previous_spec.entry_point
            )
            notes = _detection_notes(spec, matches) if matches is not None and not retry else []
            session.task_mode = task
            if retry and source == state.source:
                return {
                    "unchanged": True,
                    "message": (
                        "The code has not changed since the last attempt. Edit it and retry, "
                        "or ask for another question."
                    ),
                    "session": self.snapshot(session),
                }

            # all or nothing: an attempt that cannot be projected must not stay
            # in the session, or every later snapshot would trip over it
            saved = (state.problem, state.source, list(state.attempts), state.mastery,
                     state.revealed, state.challenge_shown)
            state.problem, state.source = spec, source
            try:
                with self._exclusive(), _quiet():
                    attempt = record_attempt(
                        self.orchestrator, state, spec, label="editor", graph=self.graph
                    )
                previous = state.previous if retry else None
                projection = self._project(attempt.index, attempt.result, source, previous, notes)
            except BaseException:
                (state.problem, state.source, state.attempts, state.mastery,
                 state.revealed, state.challenge_shown) = saved
                raise
            session.projections[attempt.index] = projection
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
            state = session.state
            total = len(projection.hints)
            if not total:
                return {
                    "status": "none",
                    "message": "There is no guidance for this attempt: nothing was diagnosed.",
                    "session": self.snapshot(session),
                }
            if rung <= state.revealed:
                return {"status": "revealed", "hint": projection.hints[rung - 1],
                        "session": self.snapshot(session)}
            if rung > total:
                return {
                    "status": "exhausted",
                    "message": "That was the last question. Try the challenge, or edit and retry.",
                    "session": self.snapshot(session),
                }
            if rung != state.revealed + 1:
                raise WebError(
                    409, "out_of_order",
                    "Questions open one at a time.",
                    next=state.revealed + 1,
                )
            state.revealed = rung
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
            session.state.challenge_shown = True
            return {"status": "shown", "challenge": projection.challenge,
                    "session": self.snapshot(session)}

    def run_challenge(self, session: WebSession, index: Any, source: Any) -> Dict[str, Any]:
        """Predict, then run: the code in the editor on the challenge's input,
        under the same guarded runner and the same budgets as a diagnosis.
        """
        with session.lock:
            projection = self._latest(session, index)
            state = session.state
            if not state.challenge_shown or projection.challenge is None:
                raise WebError(409, "challenge_closed", "Open the challenge before running it.")
            if projection.challenge_args is None:
                raise WebError(409, "not_runnable", "This challenge has no input to run.")
            session.spend(time.time())
            spec = state.problem
            assert spec is not None
            with self._exclusive():
                source = check_source(source)
                entry = resolve_entry(source, spec.entry_point)
                with _quiet():
                    outcome = self.runner.run(
                        source, entry, copy.deepcopy(list(projection.challenge_args)), trace=False
                    )
            run: Dict[str, Any] = {
                "input": views.render_value(projection.challenge_args),
                "ok": outcome.ok,
                "value": views.render_value(outcome.value) if outcome.ok else None,
                "error": None if outcome.ok else _error_line(outcome.error_type, outcome.error),
                "budget": bool(outcome.timed_out or outcome.recursion_overflow),
                "required": (
                    views.render_value(projection.required) if projection.has_required else None
                ),
                "agrees": (
                    views.safe_equivalent(outcome.value, projection.required)
                    if outcome.ok and projection.has_required else None
                ),
            }
            return {"run": run}

    def reset(self, session: WebSession) -> Dict[str, Any]:
        """Forget the mastery estimate and the history, like the terminal's reset."""
        with session.lock:
            state = session.state
            state.mastery = None
            state.attempts = []
            state.revealed = 0
            state.challenge_shown = False
            session.projections.clear()
            return {"session": self.snapshot(session)}

    # internals
    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self.execution.acquire(timeout=EXECUTION_WAIT):
            raise WebError(
                503, "busy",
                "The tutor is busy with other learners' code. Try again in a moment.",
            )
        try:
            yield
        finally:
            self.execution.release()

    def _latest(self, session: WebSession, index: Any) -> Projection:
        if not isinstance(index, int) or isinstance(index, bool):
            raise WebError(400, "bad_request", "attempt must be an integer")
        latest = session.state.latest
        if latest is None:
            raise WebError(409, "no_attempt", "Diagnose some code first.")
        if index != latest.index:
            raise WebError(
                409, "stale_attempt",
                "Guidance belongs to the latest attempt only.",
                latest=latest.index,
            )
        return session.projections[latest.index]

    def _project(
        self,
        index: int,
        result: Any,
        source: str,
        previous: Any,
        notes: List[Dict[str, str]],
    ) -> Projection:
        boundary = Boundary(
            self.verifier, result.problem, source, self.config.intervention.max_leakage
        )
        view = views.attempt_view(
            result, index=index, source=source, config=self.config, graph=self.graph,
            boundary=boundary,
        )
        view["notes"] = notes + view["notes"]
        hints = result.intervention.hints
        sealed = tuple(
            views.hint_view(hint, rung=rung, total=len(hints), graph=self.graph, boundary=boundary)
            for rung, hint in enumerate(hints, start=1)
        )
        challenge = result.intervention.challenge
        counterexample = result.diagnosis.counterexample
        projection = Projection(
            view=view,
            summary=views.summary_view(index, result),
            hints=sealed,
            challenge=(
                views.challenge_view(challenge, boundary=boundary) if challenge is not None else None
            ),
            challenge_args=(
                tuple(challenge.args) if challenge is not None and challenge.args is not None else None
            ),
            required=counterexample.reference_output if counterexample is not None else None,
            has_required=counterexample is not None,
            diff=(
                views.diff_view(previous.result, result, self.graph) if previous is not None else None
            ),
        )
        if boundary.replaced:
            self.boundary_replacements += boundary.replaced
            sys.stderr.write(
                f"dmas web: the answer-leakage check on the way out replaced {boundary.replaced} "
                f"item(s) of attempt {index} ({result.telemetry.backend})\n"
            )
        return projection


# helpers
def check_source(source: Any) -> str:
    """Refuse what is not an attempt before it reaches the pipeline.

    A file that does not parse is not evidence about a belief: the pipeline
    would report no finding, and the session would credit the concept as
    repaired. So it never becomes an attempt; the learner gets the line and
    column instead, which is what an editor owes them anyway.
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


def _error_line(kind: Optional[str], message: Optional[str]) -> str:
    text = ": ".join(part for part in (kind, message) if part)
    return text if len(text) <= 200 else text[:199] + "…"


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    with contextlib.redirect_stdout(_Discard()):
        yield
