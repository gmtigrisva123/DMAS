"""Guarded, traced execution of student code.

Static analysis can only say something looks suspicious, the trace says when
it went wrong. We record a value trace of the student program and of the
reference on the same input so the alignment step can point at the first
disagreement.

Not a security sandbox. Three limits (AST denylist, step budget, wall clock
deadline) which is fine for classroom code, do not run adversarial code in
here.
"""

import ast
import sys
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import ExecutionConfig
from ..errors import ExecutionTimeout, UnsafeCodeError
from ..util.timing import Deadline

# names a submission can never call
BANNED_NAMES = frozenset(
    """eval exec compile open input __import__ globals locals vars breakpoint exit quit
    memoryview setattr delattr getattr help dir""".split()
)

# attributes that expose interpreter internals
BANNED_ATTRIBUTE_PREFIXES = ("__",)

SYNTHETIC_FILENAME = "<dmas-submission>"


@dataclass(frozen=True)
class TraceEvent:
    """One observed state, taken right before a line runs."""

    step: int
    lineno: int
    function: str
    state: Mapping[str, Any] = field(default_factory=dict)

    def projected(self, names: Sequence[str]) -> Tuple[Any, ...]:
        return tuple(self.state.get(name) for name in names)

    def __str__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in sorted(self.state.items()))
        return f"#{self.step} L{self.lineno} {{{body}}}"


@dataclass
class ExecutionResult:
    """Outcome of one guarded run."""

    ok: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_line: Optional[int] = None
    events: Tuple[TraceEvent, ...] = ()
    steps: int = 0
    truncated: bool = False
    timed_out: bool = False
    recursion_overflow: bool = False

    @property
    def diverged(self) -> bool:
        """True if the run hit a limit instead of finishing."""
        return self.timed_out or self.recursion_overflow

    def variable_series(self, minimum_length: int = 2) -> Dict[str, List[float]]:
        """Numeric time series per variable, for behavioural matching."""
        series: Dict[str, List[float]] = {}
        for event in self.events:
            for name, value in event.state.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    series.setdefault(name, []).append(float(value))
        return {k: v for k, v in series.items() if len(v) >= minimum_length}


# static safety gate
def assert_safe(tree: ast.AST, allowed_imports: Sequence[str]):
    """Raise UnsafeCodeError if the tree uses something banned."""
    allowed = set(allowed_imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    raise UnsafeCodeError(f"import of {alias.name!r} is not permitted")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed:
                raise UnsafeCodeError(f"import from {node.module!r} is not permitted")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise UnsafeCodeError(f"use of {node.id!r} is not permitted")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(BANNED_ATTRIBUTE_PREFIXES):
                raise UnsafeCodeError(f"access to {node.attr!r} is not permitted")


# the runner
class GuardedRunner:
    """Compiles and runs a submission with step / time / construct limits."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    # compiling
    def compile_module(self, source: str, *, filename: str = SYNTHETIC_FILENAME) -> Dict[str, Any]:
        """Exec source at module level and return the namespace."""
        tree = ast.parse(source, filename=filename)
        assert_safe(tree, self.config.allowed_imports)
        code = compile(tree, filename, "exec")
        namespace: Dict[str, Any] = {
            "__name__": "dmas_submission",
            "__builtins__": _safe_builtins(self.config.allowed_imports),
        }
        exec(code, namespace)
        return namespace

    # running
    def run(
        self,
        source: str,
        entry_point: str,
        args: Sequence[Any] = (),
        *,
        trace: bool = True,
        filename: str = SYNTHETIC_FILENAME,
    ) -> ExecutionResult:
        """Run entry_point(*args) and record the value trace."""
        try:
            namespace = self.compile_module(source, filename=filename)
        except UnsafeCodeError as exc:
            return ExecutionResult(ok=False, error=str(exc), error_type="UnsafeCodeError")
        except SyntaxError as exc:
            return ExecutionResult(
                ok=False, error=f"{exc.msg}", error_type="SyntaxError", error_line=exc.lineno
            )
        except Exception as exc:  # module level failure
            return ExecutionResult(ok=False, error=str(exc), error_type=type(exc).__name__)

        function = namespace.get(entry_point)
        if not callable(function):
            return ExecutionResult(
                ok=False,
                error=f"entry point {entry_point!r} is not defined",
                error_type="MissingEntryPoint",
            )
        return self._invoke(function, args, filename=filename, trace=trace)

    def run_callable(self, function: Callable[..., Any], args: Sequence[Any]) -> ExecutionResult:
        """Run an already compiled callable without recording state.
        Used as the differential testing oracle where only the return value
        matters. Step and time limits still apply (an infinite loop must not hang
        the diagnosis) but no locals are snapshotted, about 10x cheaper.
        """
        return self._invoke(function, args, filename=SYNTHETIC_FILENAME, trace=False)

    # internals
    def _invoke(
        self,
        function: Callable[..., Any],
        args: Sequence[Any],
        *,
        filename: str,
        trace: bool,
    ) -> ExecutionResult:
        collector = _Collector(self.config, filename, record=trace)
        collector.deadline = Deadline(self.config.max_seconds)

        previous_trace = sys.gettrace()
        previous_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(min(previous_limit, self.config.max_recursion + 120))
        # hook is installed even when not recording, it is the only thing that
        # stops a non terminating submission
        sys.settrace(collector.dispatch)
        try:
            value = function(*args)
            return ExecutionResult(
                ok=True,
                value=value,
                events=tuple(collector.events),
                steps=collector.steps,
                truncated=collector.truncated,
            )
        except _StepBudgetExceeded:
            return ExecutionResult(
                ok=False,
                error="execution exceeded its step/time budget (possible infinite loop)",
                error_type="ExecutionTimeout",
                events=tuple(collector.events),
                steps=collector.steps,
                truncated=collector.truncated,
                timed_out=True,
            )
        except RecursionError:
            return ExecutionResult(
                ok=False,
                error="maximum recursion depth exceeded (missing or unreachable base case)",
                error_type="RecursionError",
                events=tuple(collector.events),
                steps=collector.steps,
                truncated=collector.truncated,
                recursion_overflow=True,
            )
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                error=str(exc) or type(exc).__name__,
                error_type=type(exc).__name__,
                error_line=_failing_line(exc, filename),
                events=tuple(collector.events),
                steps=collector.steps,
                truncated=collector.truncated,
            )
        finally:
            sys.settrace(previous_trace)
            sys.setrecursionlimit(previous_limit)


class _StepBudgetExceeded(Exception):
    """Internal signal raised from the trace hook."""


class _Collector:
    """The sys.settrace hook, snapshots locals on every line."""

    __slots__ = ("config", "filename", "events", "steps", "truncated", "deadline", "record", "_check")

    def __init__(self, config: ExecutionConfig, filename: str, *, record: bool = True):
        self.config = config
        self.filename = filename
        self.record = record
        self.events: List[TraceEvent] = []
        self.steps = 0
        self.truncated = False
        self.deadline: Optional[Deadline] = None
        self._check = 0

    def dispatch(self, frame: FrameType, event: str, arg: Any):
        # only trace the submission itself, library frames stay untraced
        if frame.f_code.co_filename != self.filename:
            return None
        return self.local(frame, event, arg)

    def local(self, frame: FrameType, event: str, arg: Any):
        if event != "line":
            return self.local
        self.steps += 1
        self._check += 1
        if self.steps > self.config.max_steps:
            raise _StepBudgetExceeded()
        if self._check >= 256:
            self._check = 0
            if self.deadline is not None and self.deadline.expired:
                raise _StepBudgetExceeded()
        if not self.record:
            return self.local
        if len(self.events) < self.config.max_trace_events:
            self.events.append(
                TraceEvent(
                    step=self.steps,
                    lineno=frame.f_lineno,
                    function=frame.f_code.co_name,
                    state=_snapshot(frame, self.config.max_container_preview),
                )
            )
        else:
            self.truncated = True
        return self.local


def _snapshot(frame: FrameType, preview: int) -> Dict[str, Any]:
    """Project a frame's locals onto comparable, hashable values."""
    out: Dict[str, Any] = {}
    for name, value in frame.f_locals.items():
        if name.startswith("_"):
            continue
        rendered = _project(value, preview)
        if rendered is not _SKIP:
            out[name] = rendered
    return out


class _Skip:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<skip>"


_SKIP = _Skip()


def _project(value: Any, preview: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 64 else f"<str len={len(value)}>"
    if isinstance(value, (list, tuple)):
        if len(value) <= preview and all(
            v is None or isinstance(v, (bool, int, float, str)) for v in value
        ):
            return tuple(value)
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, (set, frozenset)):
        if len(value) <= preview and all(isinstance(v, (bool, int, float, str)) for v in value):
            return tuple(sorted(value, key=repr))
        return f"<set len={len(value)}>"
    if isinstance(value, dict):
        if len(value) <= preview and all(
            isinstance(k, (bool, int, float, str)) and isinstance(v, (bool, int, float, str))
            for k, v in value.items()
        ):
            return tuple(sorted(value.items(), key=repr))
        return f"<dict len={len(value)}>"
    if callable(value) or isinstance(value, type):
        return _SKIP
    return f"<{type(value).__name__}>"


def _failing_line(exc: BaseException, filename: str) -> Optional[int]:
    """Deepest traceback line that belongs to the submission."""
    tb = exc.__traceback__
    line: Optional[int] = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == filename:
            line = tb.tb_lineno
        tb = tb.tb_next
    return line


def _safe_builtins(allowed_imports: Sequence[str] = ()) -> Dict[str, Any]:
    """Reduced __builtins__ with only what algorithmic code needs.
    __import__ is there but limited to the allowlist, so 'from collections
    import deque' works and anything else is refused at runtime too (the AST
    gate cannot see an import through an alias it did not expect).
    """
    import builtins

    permitted = frozenset(allowed_imports)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in permitted:
            raise UnsafeCodeError(f"import of {name!r} is not permitted")
        return builtins.__import__(name, globals, locals, fromlist, level)

    allowed = """abs all any bin bool bytes callable chr complex dict divmod enumerate filter
    float format frozenset hash hex int isinstance issubclass iter len list map max min next
    object oct ord pow print range repr reversed round set slice sorted str sum tuple type zip
    True False None Exception ValueError TypeError IndexError KeyError ZeroDivisionError
    StopIteration RecursionError OverflowError ArithmeticError AssertionError
    NotImplementedError RuntimeError AttributeError""".split()
    namespace = {name: getattr(builtins, name) for name in allowed if hasattr(builtins, name)}
    namespace["__build_class__"] = builtins.__build_class__
    namespace["__import__"] = guarded_import
    namespace["__name__"] = "builtins"
    return namespace
