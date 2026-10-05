"""Trace alignment: find the first point where the student's execution leaves
the reference's.

1. project both traces onto the shared roles from varmatch.py -> two state
   trajectories (consecutive duplicates collapsed)
2. align them with Needleman-Wunsch. Alignment instead of index by index
   comparison is what makes this robust to an extra / missing iteration
3. report the first column that is not a match plus the operation
   (substitution / insertion / deletion) that says how the student left the
   correct path
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..domain import DivergenceKind, DivergencePoint, SourceSpan
from .tracer import ExecutionResult, TraceEvent
from .varmatch import RoleMapping

MATCH = "match"
SUBSTITUTE = "substitute"
STUDENT_ONLY = "student-only"      # a step the reference never takes
REFERENCE_ONLY = "reference-only"  # a step the student skipped


@dataclass(frozen=True)
class StateSample:
    """One point of a state trajectory."""

    step: int
    lineno: int
    values: Tuple[Any, ...]
    roles: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return dict(zip(self.roles, self.values))

    def __str__(self) -> str:
        body = ", ".join(f"{r}={v!r}" for r, v in zip(self.roles, self.values))
        return f"L{self.lineno} {{{body}}}"


@dataclass(frozen=True)
class AlignedColumn:
    op: str
    student: Optional[StateSample]
    reference: Optional[StateSample]


@dataclass
class Alignment:
    columns: Tuple[AlignedColumn, ...] = ()
    cost: float = 0.0
    roles: Tuple[str, ...] = ()

    @property
    def identity(self) -> float:
        if not self.columns:
            return 1.0
        return sum(1 for c in self.columns if c.op == MATCH) / len(self.columns)

    def first_mismatch(self) -> Optional[AlignedColumn]:
        for column in self.columns:
            if column.op != MATCH:
                return column
        return None


# trajectory extraction
def trajectory(
    result: ExecutionResult,
    roles: Sequence[str],
    rename: Optional[Mapping[str, str]] = None,
    *,
    sync_lines: Optional[Sequence[int]] = None,
    limit: int = 600,
) -> List[StateSample]:
    """Collapse a value trace into a list of distinct abstract states.

    sync_lines limits sampling to loop checkpoints (see cfg.synchronisation_lines).
    Without it every line is sampled and states where a role is still unbound
    get dropped.
    """
    rename = rename or {}
    keys = tuple(roles)
    checkpoints = set(sync_lines) if sync_lines else None
    out: List[StateSample] = []
    previous: Optional[Tuple[Any, ...]] = None
    for event in result.events:
        if checkpoints is not None and event.lineno not in checkpoints:
            continue
        mapped: Dict[str, Any] = {}
        for name, value in event.state.items():
            target = rename.get(name, name)
            if target in keys:
                mapped[target] = value
        values = tuple(mapped.get(key) for key in keys)
        if checkpoints is None and any(v is None for v in values):
            continue                       # partially initialised, skip
        if values == previous:
            continue
        previous = values
        out.append(StateSample(step=event.step, lineno=event.lineno, values=values, roles=keys))
        if len(out) >= limit:
            break
    return out


# Needleman-Wunsch
def needleman_wunsch(
    student: Sequence[StateSample],
    reference: Sequence[StateSample],
    *,
    gap_cost: float = 1.0,
    mismatch_cost: float = 1.0,
) -> Alignment:
    """Global alignment of two trajectories, O(len(a) * len(b)).
    Ties go to the diagonal so a real value error shows up as a substitution
    at the earliest position instead of hiding behind two gaps.
    """
    n, m = len(student), len(reference)
    roles = student[0].roles if student else (reference[0].roles if reference else ())
    if n == 0 and m == 0:
        return Alignment(columns=(), cost=0.0, roles=roles)

    # dp[i][j] = cost of aligning student[:i] with reference[:j]
    dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        back[i][0] = 1                                    # consume student
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        back[0][j] = 2                                    # consume reference

    for i in range(1, n + 1):
        srow = student[i - 1].values
        dp_i, dp_prev, back_i = dp[i], dp[i - 1], back[i]
        for j in range(1, m + 1):
            same = srow == reference[j - 1].values
            diagonal = dp_prev[j - 1] + (0.0 if same else mismatch_cost)
            up = dp_prev[j] + gap_cost
            left = dp_i[j - 1] + gap_cost
            best, choice = diagonal, 0
            if up < best:
                best, choice = up, 1
            if left < best:
                best, choice = left, 2
            dp_i[j] = best
            back_i[j] = choice

    columns: List[AlignedColumn] = []
    i, j = n, m
    while i > 0 or j > 0:
        choice = back[i][j]
        if i > 0 and j > 0 and choice == 0:
            same = student[i - 1].values == reference[j - 1].values
            columns.append(
                AlignedColumn(MATCH if same else SUBSTITUTE, student[i - 1], reference[j - 1])
            )
            i, j = i - 1, j - 1
        elif i > 0 and (choice == 1 or j == 0):
            columns.append(AlignedColumn(STUDENT_ONLY, student[i - 1], None))
            i -= 1
        else:
            columns.append(AlignedColumn(REFERENCE_ONLY, None, reference[j - 1]))
            j -= 1
    columns.reverse()
    return Alignment(columns=tuple(columns), cost=dp[n][m], roles=roles)


# divergence localisation
def locate_divergence(
    student: ExecutionResult,
    reference: ExecutionResult,
    mapping: RoleMapping,
    *,
    source_lines: Sequence[str] = (),
    student_sync: Optional[Sequence[int]] = None,
    reference_sync: Optional[Sequence[int]] = None,
) -> Optional[DivergencePoint]:
    """Find the first point where the student's execution left the correct path."""
    # hard failures first, nothing to align if the program crashed
    if student.timed_out:
        return DivergencePoint(
            kind=DivergenceKind.NON_TERMINATION,
            step=student.steps,
            student_line=student.events[-1].lineno if student.events else None,
            reference_line=None,
            description="the program never terminates: no loop guard variable makes progress",
            span=_span(student.events[-1].lineno if student.events else None, source_lines),
        )
    if student.recursion_overflow:
        return DivergencePoint(
            kind=DivergenceKind.NON_TERMINATION,
            step=student.steps,
            student_line=student.events[-1].lineno if student.events else None,
            reference_line=None,
            description="recursion never reaches a base case",
            span=_span(student.events[-1].lineno if student.events else None, source_lines),
        )
    if not student.ok and student.error:
        return DivergencePoint(
            kind=DivergenceKind.EXCEPTION,
            step=student.steps,
            student_line=student.error_line,
            reference_line=None,
            description=f"{student.error_type}: {student.error}",
            span=_span(student.error_line, source_lines),
        )

    roles = mapping.roles
    if not roles:
        return _output_divergence(student, reference, source_lines)

    rename = dict(mapping.pairs)
    student_path = trajectory(student, roles, rename, sync_lines=student_sync)
    reference_path = trajectory(reference, roles, sync_lines=reference_sync)
    if not student_path or not reference_path:
        # no shared checkpoint reached (e.g. loop free solution), sample every line
        student_path = trajectory(student, roles, rename)
        reference_path = trajectory(reference, roles)

    return classify_divergence(student, reference, student_path, reference_path, roles, source_lines)


LOOKAHEAD = 8


def classify_divergence(
    student: ExecutionResult,
    reference: ExecutionResult,
    student_path: Sequence[StateSample],
    reference_path: Sequence[StateSample],
    roles: Sequence[str],
    source_lines: Sequence[str] = (),
) -> Optional[DivergencePoint]:
    """Find and classify the first disagreement between two trajectories.

    The location is exact: both runs agree on every state before it, so it is
    just the end of the longest common prefix. The classification is what the
    alignment is for: at position k we compare three continuations over a short
    lookahead window and keep the one that explains the rest best:

      substitution    both continue, wrong value at k
      student-only    the student did a transition the reference never does
                      (extra iteration, duplicated update)
      reference-only  the student skipped a transition (missing iteration,
                      early exit)
    """
    k = 0
    while (
        k < len(student_path)
        and k < len(reference_path)
        and student_path[k].values == reference_path[k].values
    ):
        k += 1

    if k >= len(student_path) and k >= len(reference_path):
        return _output_divergence(student, reference, source_lines)

    if k >= len(student_path):                      # student ran out of states
        sample = reference_path[k]
        anchor = student_path[-1].lineno if student_path else None
        return DivergencePoint(
            kind=DivergenceKind.EARLY_TERMINATION,
            step=sample.step,
            student_line=anchor,
            reference_line=sample.lineno,
            reference_state=sample.as_dict(),
            description=(
                "the program stops after "
                f"{len(student_path)} checkpoint(s) while the correct solution needs "
                f"{len(reference_path)}"
            ),
            span=_span(anchor, source_lines),
        )

    if k >= len(reference_path):                    # student kept going
        sample = student_path[k]
        return DivergencePoint(
            kind=DivergenceKind.CONTROL_MISMATCH,
            step=sample.step,
            student_line=sample.lineno,
            reference_line=None,
            student_state=sample.as_dict(),
            description="the program keeps iterating after the correct solution has finished",
            span=_span(sample.lineno, source_lines),
        )

    substitution = _continuation_score(student_path, k + 1, reference_path, k + 1)
    inserted = _continuation_score(student_path, k + 1, reference_path, k)
    skipped = _continuation_score(student_path, k, reference_path, k + 1)

    if inserted > substitution and inserted >= skipped:
        sample = student_path[k]
        return DivergencePoint(
            kind=DivergenceKind.CONTROL_MISMATCH,
            step=sample.step,
            student_line=sample.lineno,
            reference_line=reference_path[k].lineno,
            student_state=sample.as_dict(),
            reference_state=reference_path[k].as_dict(),
            description="the program performs an extra state transition the correct solution never performs",
            span=_span(sample.lineno, source_lines),
        )
    if skipped > substitution and skipped > inserted:
        sample = reference_path[k]
        anchor = student_path[k].lineno
        return DivergencePoint(
            kind=DivergenceKind.EARLY_TERMINATION,
            step=sample.step,
            student_line=anchor,
            reference_line=sample.lineno,
            student_state=student_path[k].as_dict(),
            reference_state=sample.as_dict(),
            description="the program skips a state transition the correct solution must perform",
            span=_span(anchor, source_lines),
        )

    student_state = student_path[k].as_dict()
    reference_state = reference_path[k].as_dict()
    # a role that is unbound on one side at this checkpoint is just where the
    # two programs put their statements, not something to report
    differing = [
        r for r in roles
        if student_state.get(r) != reference_state.get(r)
        and student_state.get(r) is not None
        and reference_state.get(r) is not None
    ] or [r for r in roles if student_state.get(r) != reference_state.get(r)]
    return DivergencePoint(
        kind=DivergenceKind.STATE_MISMATCH,
        step=student_path[k].step,
        student_line=student_path[k].lineno,
        reference_line=reference_path[k].lineno,
        student_state=student_state,
        reference_state=reference_state,
        description=(
            f"after {k} matching checkpoint(s) the executions disagree: "
            + ", ".join(
                f"{r} is {student_state.get(r)!r} but should be {reference_state.get(r)!r}"
                for r in differing[:3]
            )
        ),
        span=_span(student_path[k].lineno, source_lines),
    )


def _continuation_score(
    student_path: Sequence[StateSample],
    i: int,
    reference_path: Sequence[StateSample],
    j: int,
) -> float:
    """Fraction of matching states over a short lookahead window."""
    window = min(LOOKAHEAD, len(student_path) - i, len(reference_path) - j)
    if window <= 0:
        return 0.0
    hits = sum(
        1 for offset in range(window)
        if student_path[i + offset].values == reference_path[j + offset].values
    )
    return hits / float(window)


def _output_divergence(
    student: ExecutionResult, reference: ExecutionResult, source_lines: Sequence[str]
) -> Optional[DivergencePoint]:
    if student.ok and reference.ok and student.value != reference.value:
        line = student.events[-1].lineno if student.events else None
        return DivergencePoint(
            kind=DivergenceKind.OUTPUT_MISMATCH,
            step=student.steps,
            student_line=line,
            reference_line=None,
            student_state={"return": student.value},
            reference_state={"return": reference.value},
            description=(
                f"the internal states agree throughout, but the returned value "
                f"{student.value!r} differs from {reference.value!r}"
            ),
            span=_span(line, source_lines),
        )
    return None


def _nearest_student_line(alignment: Alignment, target: AlignedColumn) -> Optional[int]:
    seen: Optional[int] = None
    for column in alignment.columns:
        if column is target:
            break
        if column.student is not None:
            seen = column.student.lineno
    return seen


def _span(line: Optional[int], source_lines: Sequence[str]) -> Optional[SourceSpan]:
    if not line or line < 1:
        return None
    snippet = source_lines[line - 1].rstrip() if line - 1 < len(source_lines) else ""
    return SourceSpan(line=line, end_line=line, snippet=snippet)
