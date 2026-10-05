"""Misconception taxonomy + the symbolic rules that detect them.

A misconception is not a bug, it is the belief that produces the bug. "Your
loop is wrong" is a bug report, "you treat hi as inclusive while the guard
treats it as exclusive" is a diagnosis and predicts the next mistake.

Each entry has the graph concepts it implicates, the belief in the student's
own words (used to open the dialogue) and a remediation focus. Most have a
rule: a short predicate over the analysis facts that fires with a calibrated
belief. Some have none on purpose (greedy.local-optimum-assumed cannot be
decided from syntax) and can only come from model evidence.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..analysis.ast_features import CodeFeatures, LoopFact
from ..analysis.cfg import ControlFlowGraph
from ..analysis.complexity import ComplexityReport
from ..analysis.dataflow import DataflowFacts
from ..analysis.tracer import ExecutionResult
from ..domain import (
    Counterexample,
    DivergenceKind,
    DivergencePoint,
    EvidenceKind,
    Misconception,
    ProblemSpec,
    Severity,
)

# the taxonomy
_TAXONOMY: Tuple[Misconception, ...] = (
    Misconception(
        mid="bs.interval-convention-mismatch",
        name="Mixed interval conventions",
        description=(
            "The loop guard and the bound initialisation follow different conventions: one "
            "treats the upper bound as inclusive, the other as exclusive."
        ),
        concepts=("half-open-intervals", "boundary-conditions", "binary-search"),
        severity=Severity.MAJOR,
        student_voice="I think the upper bound is just 'the end of the array' — which end does not matter.",
        remediation_focus="Choose one convention and derive every line from it.",
    ),
    Misconception(
        mid="bs.no-progress-update",
        name="Search interval does not shrink",
        description=(
            "A bound is moved to the midpoint itself rather than past it, so on some inputs "
            "the interval stops shrinking and the loop cannot terminate."
        ),
        concepts=("loop-termination", "loop-invariant", "binary-search"),
        severity=Severity.CRITICAL,
        student_voice="I think moving a bound to the middle is enough to make progress.",
        remediation_focus="Identify the decreasing measure and check it strictly decreases.",
    ),
    Misconception(
        mid="loop.non-termination",
        name="Loop guard never changes",
        description="No variable the loop guard reads is ever written in the loop body.",
        concepts=("loop-termination", "loop-invariant"),
        severity=Severity.CRITICAL,
        student_voice="I think the loop will stop on its own once the condition becomes false.",
        remediation_focus="Name the quantity that must decrease and show where it decreases.",
    ),
    Misconception(
        mid="loop.off-by-one-index",
        name="Index runs past the end",
        description=(
            "The loop ranges over every index while the body reads a neighbouring index, so "
            "the final iteration addresses one element past the end."
        ),
        concepts=("sequence-indexing", "boundary-conditions"),
        severity=Severity.MAJOR,
        student_voice="I think looking at the next element is always safe inside the loop.",
        remediation_focus="Count how many valid pairs exist, then compare that with your loop's bound.",
    ),
    Misconception(
        mid="rec.missing-base-case",
        name="No base case",
        description="Every path through the recursive function calls itself again.",
        concepts=("base-case", "recursion-basics"),
        severity=Severity.CRITICAL,
        student_voice="I think the recursion will bottom out by itself.",
        remediation_focus="State the smallest instance and answer it without recursing.",
    ),
    Misconception(
        mid="rec.unreachable-base-case",
        name="Base case is never reached",
        description=(
            "A base case exists but the argument does not converge to it, so the call stack "
            "grows without bound."
        ),
        concepts=("base-case", "recursive-decomposition", "call-stack"),
        severity=Severity.CRITICAL,
        student_voice="I think any smaller argument will eventually hit my base case.",
        remediation_focus="Check that the argument strictly approaches the base case on every path.",
    ),
    Misconception(
        mid="rec.result-discarded",
        name="Recursive result thrown away",
        description="The function calls itself but ignores the returned value.",
        concepts=("recursion-basics", "recursive-decomposition"),
        severity=Severity.MAJOR,
        student_voice="I think the recursive call updates things by itself.",
        remediation_focus="Treat the recursive call as a value to combine rather than as a side effect.",
    ),
    Misconception(
        mid="dp.recomputed-subproblems",
        name="Overlapping subproblems recomputed",
        description=(
            "A branching recursion re-solves the same subproblems exponentially often because "
            "no result is cached."
        ),
        concepts=("overlapping-subproblems", "memoisation", "tail-vs-tree-recursion"),
        severity=Severity.MAJOR,
        student_voice="I think each recursive call solves something new.",
        remediation_focus="Count distinct states, then compare that with the number of calls.",
    ),
    Misconception(
        mid="dp.transition-order",
        name="States read before they are written",
        description=(
            "A table entry is read from a position the loop order has not filled yet, so the "
            "transition consumes an uninitialised value."
        ),
        concepts=("dp-transition-order", "tabulation", "dag-properties"),
        severity=Severity.MAJOR,
        student_voice="I think the table order does not matter as long as the recurrence is right.",
        remediation_focus="Draw the dependency arrows, then iterate in a topological order.",
    ),
    Misconception(
        mid="graph.visited-on-dequeue",
        name="Vertices marked when dequeued",
        description=(
            "A vertex is marked visited when it leaves the worklist rather than when it "
            "enters, so it can be enqueued many times before it is ever marked."
        ),
        concepts=("visited-set", "bfs", "graph-traversal"),
        severity=Severity.MAJOR,
        student_voice="I think marking a vertex when I process it is the same as marking it when I find it.",
        remediation_focus="Ask how many times one vertex can enter the worklist.",
    ),
    Misconception(
        mid="graph.list-as-visited",
        name="Linear-time membership test",
        description="The visited collection is a list, so each membership test scans it.",
        concepts=("visited-set", "hash-table", "cost-model"),
        severity=Severity.MINOR,
        student_voice="I think checking membership costs the same whatever the container is.",
        remediation_focus="Compare the cost model of a list with that of a set.",
    ),
    Misconception(
        mid="graph.list-as-queue",
        name="List used as a queue",
        description="Removing from the front of a list shifts every remaining element.",
        concepts=("queue-deque", "cost-model", "bfs"),
        severity=Severity.MINOR,
        student_voice="I think removing the first element is as cheap as removing the last.",
        remediation_focus="Work out the cost of one front-removal on a contiguous array.",
    ),
    Misconception(
        mid="cx.membership-in-list",
        name="Membership scan inside a loop",
        description="A linear membership test sits inside a loop, multiplying the cost.",
        concepts=("cost-model", "hash-table"),
        severity=Severity.MINOR,
        student_voice="I think `in` is a single cheap operation.",
        remediation_focus="Substitute the cost of the inner operation into the loop's cost.",
    ),
    Misconception(
        mid="cx.asymptotic-gap",
        name="Asymptotically slower than required",
        description="The submission's growth rate is strictly worse than the intended solution's.",
        concepts=("asymptotic-notation", "cost-model"),
        severity=Severity.MAJOR,
        student_voice="I think a solution that returns the right answer is fast enough.",
        remediation_focus="Derive the cost from the loop structure before writing more code.",
    ),
    Misconception(
        mid="cx.sort-inside-loop",
        name="Re-sorting inside a loop",
        description="A sort is performed on every iteration although the data changes little.",
        concepts=("comparison-sorting", "asymptotic-notation"),
        severity=Severity.MINOR,
        student_voice="I think sorting again is the simplest way to keep things ordered.",
        remediation_focus="Ask what the sort costs multiplied by the number of iterations.",
    ),
    Misconception(
        mid="search.linear-scan-when-ordered",
        name="Ignoring available order",
        description=(
            "The input carries an order the algorithm never exploits, so a logarithmic search "
            "is performed linearly."
        ),
        concepts=("binary-search", "linear-search", "monotone-predicate"),
        severity=Severity.MAJOR,
        student_voice="I think I have to look at every element to be sure.",
        remediation_focus="Identify the monotone property, then halve the search space.",
    ),
    Misconception(
        mid="py.mutable-default-argument",
        name="Mutable default argument",
        description=(
            "A default argument object is created once at definition time and shared by every "
            "call, so state leaks between invocations."
        ),
        concepts=("mutability-aliasing", "var-binding"),
        severity=Severity.MAJOR,
        student_voice="I think the default value is rebuilt on every call.",
        remediation_focus="Ask when the default expression is evaluated.",
    ),
    Misconception(
        mid="py.float-index",
        name="Non-integer index arithmetic",
        description="An index expression uses true division, producing a float that cannot index.",
        concepts=("integer-arithmetic", "sequence-indexing"),
        severity=Severity.MAJOR,
        student_voice="I think dividing by two gives me the middle position.",
        remediation_focus="Separate the arithmetic value from the positional value.",
    ),
    Misconception(
        mid="py.identity-vs-equality",
        name="Identity used for equality",
        description="`is` compares object identity, not value, so equal values can compare false.",
        concepts=("boolean-logic", "var-binding"),
        severity=Severity.MAJOR,
        student_voice="I think `is` and `==` mean the same thing.",
        remediation_focus="Distinguish 'the same object' from 'the same value'.",
    ),
    Misconception(
        mid="py.aliasing-copy",
        name="Assignment mistaken for a copy",
        description="Binding a second name to a container shares it; mutating one mutates both.",
        concepts=("mutability-aliasing",),
        severity=Severity.MAJOR,
        student_voice="I think assigning to a new name gives me an independent copy.",
        remediation_focus="Draw the two names and the one object they point at.",
    ),
    Misconception(
        mid="py.mutate-during-iteration",
        name="Container mutated while iterated",
        description="Elements are added to or removed from a container during its own traversal.",
        concepts=("iteration-mutation-safety", "iteration"),
        severity=Severity.MAJOR,
        student_voice="I think the loop re-reads the container each time round.",
        remediation_focus="Iterate over a snapshot, or build the result separately.",
    ),
    Misconception(
        mid="var.uninitialised",
        name="Value used before it is defined",
        description="A name is read on a path where no assignment to it can have executed.",
        concepts=("var-binding",),
        severity=Severity.CRITICAL,
        student_voice="I think naming a variable is enough to give it a value.",
        remediation_focus="Follow the path the program actually takes to that line.",
    ),
    Misconception(
        mid="var.dead-store",
        name="Computed value never used",
        description="A variable is assigned and then never read on any path.",
        concepts=("var-binding",),
        severity=Severity.INFO,
        student_voice="I think this line is doing something.",
        remediation_focus="Ask what depends on this value.",
    ),
    Misconception(
        mid="run.index-out-of-range",
        name="Index outside the valid range",
        description="Execution addressed a position the sequence does not have.",
        concepts=("sequence-indexing", "boundary-conditions"),
        severity=Severity.CRITICAL,
        student_voice="I think the index cannot leave the array if the logic is right.",
        remediation_focus="Work out the largest value the index expression can take over the whole loop.",
    ),
    Misconception(
        mid="run.type-error",
        name="Operation applied to the wrong type",
        description="Execution combined values whose types do not support the operation.",
        concepts=("var-binding", "integer-arithmetic"),
        severity=Severity.MAJOR,
        student_voice="I think the value here is the kind of thing I expect.",
        remediation_focus="Track what each name actually holds at that point in the run.",
    ),
    Misconception(
        mid="ds.wrong-container-choice",
        name="Container mismatched to the access pattern",
        description="The chosen data structure does not support the dominant operation cheaply.",
        concepts=("cost-model", "hash-table", "queue-deque"),
        severity=Severity.MINOR,
        student_voice="I think any container will do as long as it holds the data.",
        remediation_focus="List the operations by frequency and pick for the most frequent.",
    ),
    Misconception(
        mid="greedy.local-optimum-assumed",
        name="Greedy choice assumed safe",
        description=(
            "A locally optimal choice is taken as globally optimal without an exchange "
            "argument to justify it."
        ),
        concepts=("greedy-choice-property", "greedy-exchange-argument"),
        severity=Severity.MAJOR,
        student_voice="I think taking the best option now must lead to the best result overall.",
        remediation_focus="Try to construct a counterexample before trusting the greedy rule.",
    ),
    Misconception(
        mid="logic.wrong-comparison-boundary",
        name="Strict versus non-strict comparison",
        description="A comparison excludes (or includes) the boundary value it should include.",
        concepts=("boundary-conditions", "ordering-relations"),
        severity=Severity.MAJOR,
        student_voice="I think the boundary case behaves like the general case.",
        remediation_focus="Evaluate the condition by hand exactly at the boundary.",
    ),
)

MISCONCEPTIONS: Dict[str, Misconception] = {m.mid: m for m in _TAXONOMY}


def misconception(mid: str) -> Optional[Misconception]:
    return MISCONCEPTIONS.get(mid)


# rule plumbing
@dataclass
class RuleContext:
    """Everything a rule can look at for one submission."""

    problem: ProblemSpec
    features: CodeFeatures
    complexity: ComplexityReport
    reference_complexity: ComplexityReport
    # False if the task has no reference, which makes every comparative rule
    # inapplicable
    has_reference: bool = True
    cfg: Optional[ControlFlowGraph] = None
    dataflow: Optional[DataflowFacts] = None
    execution: Optional[ExecutionResult] = None
    divergence: Optional[DivergencePoint] = None
    counterexample: Optional[Counterexample] = None

    def guard_loops(self) -> List[LoopFact]:
        return [loop for loop in self.features.loops if loop.kind == "while"]

    def initialiser_of(self, name: str) -> Optional[str]:
        """The expression a variable is first given, in source order.
        Order matters: looking for an exact target match first finds 'hi = mid' in
        the loop before 'lo, hi = 0, len(a)' above it, and then thinks the bound
        was initialised to a midpoint. So collect candidates from both the direct
        and the tuple form and take the earliest line.
        """
        candidates: List[Tuple[int, str]] = []
        for assign in self.features.assignments:
            if assign.target == name:
                candidates.append((assign.line, assign.value))
                continue
            targets = [t.strip() for t in _unwrap(assign.target).split(",")]
            if name not in targets:
                continue
            values = _split_top_level(_unwrap(assign.value))
            if len(targets) == len(values):
                candidates.append((assign.line, values[targets.index(name)]))
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[0])[1]

    def has_midpoint(self, loop: LoopFact) -> bool:
        """True if the loop computes a midpoint from its own bounds."""
        lo, hi = loop.body_lines
        return any(
            lo <= assign.line <= hi and re.search(r"//\s*2\s*$|/\s*2\s*$", assign.value or "")
            for assign in self.features.assignments
        )


@dataclass(frozen=True)
class RuleHit:
    """A fired rule: calibrated belief + human readable reason."""

    belief: float
    rationale: str
    line: Optional[int] = None


Detector = Callable[[RuleContext], Optional[RuleHit]]


@dataclass(frozen=True)
class Rule:
    misconception_id: str
    kind: EvidenceKind
    detect: Detector

    def __call__(self, context: RuleContext) -> Optional[RuleHit]:
        return self.detect(context)


def _unwrap(expression: str) -> str:
    """Strip one pair of parentheses: (lo, hi) -> lo, hi."""
    text = expression.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _split_top_level(expression: str) -> List[str]:
    """Split '0, len(a) - 1' at the top level commas."""
    parts: List[str] = []
    depth = 0
    current = ""
    for char in expression:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


_GUARD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


# detectors
def _interval_convention(ctx: RuleContext) -> Optional[RuleHit]:
    for loop in ctx.guard_loops():
        match = _GUARD_RE.match(loop.guard or "")
        if not match:
            continue
        left, op, right = match.groups()
        if not ctx.has_midpoint(loop):
            # without a midpoint the loop is a two pointer scan and 'lo < hi' with an
            # inclusive bound is totally fine
            continue
        upper = ctx.initialiser_of(right)
        if upper is None or "len(" not in upper:
            continue
        inclusive_bound = bool(re.search(r"-\s*1\s*$", upper))
        if op == "<=" and not inclusive_bound:
            return RuleHit(
                0.58,
                f"the guard `{left} {op} {right}` treats `{right}` as an inclusive bound, "
                f"but it is initialised to `{upper}`, one past the last index",
                loop.line,
            )
        if op == "<" and inclusive_bound:
            return RuleHit(
                0.52,
                f"the guard `{left} {op} {right}` treats `{right}` as an exclusive bound, "
                f"but it is initialised to `{upper}`, the last valid index",
                loop.line,
            )
    return None


def _no_progress_update(ctx: RuleContext) -> Optional[RuleHit]:
    midpoints = {
        assign.target
        for assign in ctx.features.assignments
        if re.search(r"//\s*2\s*$", assign.value) or re.search(r"/\s*2\s*$", assign.value)
    }
    if not midpoints:
        return None
    for loop in ctx.guard_loops():
        match = _GUARD_RE.match(loop.guard or "")
        if not match or match.group(2) != "<=":
            continue
        bounds = {match.group(1), match.group(3)}
        for assign in ctx.features.assignments:
            lo, hi = loop.body_lines
            if not (lo <= assign.line <= hi):
                continue
            if assign.target in bounds and assign.value.strip() in midpoints:
                return RuleHit(
                    0.66,
                    f"inside a `{match.group(2)}` loop the bound `{assign.target}` is set to the "
                    f"midpoint `{assign.value.strip()}` itself, so the interval can stop shrinking",
                    assign.line,
                )
    return None


def _loop_non_termination(ctx: RuleContext) -> Optional[RuleHit]:
    if ctx.dataflow is None or ctx.cfg is None:
        return None
    stalled = ctx.dataflow.non_terminating_loops()
    if not stalled:
        return None
    header = sorted(stalled)[0]
    line = ctx.cfg.blocks[header].lines[0] if header in ctx.cfg.blocks else None
    names = ", ".join(sorted(stalled[header])[:3])
    return RuleHit(
        0.72,
        f"the loop guard reads {names} but the body never writes any of them",
        line,
    )


def _off_by_one_index(ctx: RuleContext) -> Optional[RuleHit]:
    for loop in ctx.features.loops:
        if loop.kind != "for" or not loop.iterable:
            continue
        match = re.match(r"range\(\s*len\(([A-Za-z_][A-Za-z0-9_]*)\)\s*\)\s*$", loop.iterable)
        if not match:
            continue
        container = match.group(1)
        lo, hi = loop.body_lines
        for index in ctx.features.indexes:
            if index.container != container or index.line < lo or index.line > hi:
                continue
            if index.offset is not None and index.offset > 0:
                return RuleHit(
                    0.60,
                    f"the loop covers every index of `{container}` while the body reads "
                    f"`{index.source}`, which is out of range on the last iteration",
                    index.line,
                )
    return None


def _missing_base_case(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    if features.is_recursive and not features.has_base_case:
        return RuleHit(
            0.88,
            "every `return` in the function contains a call to the function itself",
            features.self_call_lines[0] if features.self_call_lines else None,
        )
    return None


def _unreachable_base_case(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    execution = ctx.execution
    if not (features.is_recursive and features.has_base_case):
        return None
    if execution is not None and execution.recursion_overflow:
        return RuleHit(
            0.78,
            "a base case exists, but execution still exhausted the call stack, so the "
            "argument does not converge to it on every path",
            features.self_call_lines[0] if features.self_call_lines else None,
        )
    return None


def _result_discarded(ctx: RuleContext) -> Optional[RuleHit]:
    lines = ctx.features.self_calls_discarded
    if lines:
        return RuleHit(
            0.70,
            "the function calls itself as a statement, so the value it returns is discarded",
            lines[0],
        )
    return None


def _recomputed_subproblems(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    if not features.is_recursive or features.memo_arity:
        return None
    branching = features.self_call_count
    in_loop = features.self_call_in_loop
    if branching < 2 and not in_loop:
        return None
    if not ctx.complexity.total.is_exponential:
        return None
    if in_loop and branching < 2:
        detail = "a recursive call inside a loop makes the branching factor data-dependent"
    else:
        detail = f"{branching} recursive calls per invocation"
    return RuleHit(
        0.74,
        f"{detail} with no cache gives {ctx.complexity.total}, while the number of distinct "
        "states the recursion can reach is only polynomial",
        features.self_call_lines[0] if features.self_call_lines else None,
    )


def _transition_order(ctx: RuleContext) -> Optional[RuleHit]:
    """Detect a table entry read from a direction the loop has not filled yet."""
    function = ctx.features.function
    if function is None:
        return None
    ascending: Dict[str, bool] = {}
    for loop in ctx.features.loops:
        if loop.kind == "for" and loop.target and loop.iterable and loop.iterable.startswith("range("):
            ascending[loop.target] = "-1" not in loop.iterable and "reversed" not in loop.iterable

    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        base, written = _subscript_signature(target)
        if base is None:
            continue
        for sub in ast.walk(node.value):
            if not isinstance(sub, ast.Subscript):
                continue
            read_base, read = _subscript_signature(sub)
            if read_base != base or len(read) != len(written):
                continue
            for (name, offset), (_, read_offset) in zip(written, read):
                if name is None or offset is None or read_offset is None:
                    continue
                forward = read_offset > offset
                if forward and ascending.get(name, False):
                    return RuleHit(
                        0.68,
                        f"`{ast.unparse(target)}` is written while reading `{ast.unparse(sub)}`, "
                        f"but `{name}` increases, so that entry has not been filled yet",
                        getattr(node, "lineno", None),
                    )
    return None


def _subscript_signature(node: ast.Subscript) -> Tuple[Optional[str], List[Tuple[Optional[str], Optional[int]]]]:
    """dp[i][j+1] -> ("dp", [("i", 0), ("j", 1)])."""
    dims: List[Tuple[Optional[str], Optional[int]]] = []
    current: ast.expr = node
    while isinstance(current, ast.Subscript):
        dims.append(_dimension(current.slice))
        current = current.value
    dims.reverse()
    if isinstance(current, ast.Name):
        return current.id, dims
    return None, dims


def _dimension(index: ast.expr) -> Tuple[Optional[str], Optional[int]]:
    """i -> (i, 0); i + 1 -> (i, +1); i - coin -> (i, -1).
    A non constant shift (i + coin) still has a direction and that is all the
    transition order check needs.
    """
    if isinstance(index, ast.Name):
        return index.id, 0
    if isinstance(index, ast.BinOp) and isinstance(index.op, (ast.Add, ast.Sub)):
        if not isinstance(index.left, ast.Name):
            return None, None
        sign = 1 if isinstance(index.op, ast.Add) else -1
        right = index.right
        if isinstance(right, ast.Constant) and isinstance(right.value, int):
            return index.left.id, sign * right.value
        if isinstance(right, (ast.Name, ast.Call, ast.Subscript)):
            return index.left.id, sign          # unknown size, known direction
    return None, None


def _visited_on_dequeue(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    popped: Dict[str, int] = {}
    for assign in features.assignments:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.(pop|popleft)\(", assign.value or ""):
            popped[assign.target] = assign.line
    if not popped:
        return None
    marks: List[Tuple[str, int]] = []
    for call in features.calls:
        if call.name not in ("add", "append") or not call.args:
            continue
        argument = call.args[0].strip()
        if argument in popped and call.receiver and call.receiver not in _queue_names(features):
            marks.append((argument, call.line))
    if not marks:
        return None
    # if neighbours are also marked at enqueue time the algorithm is fine
    neighbour_marks = _marks_at_enqueue(features)
    if neighbour_marks:
        return None
    name, line = marks[0]
    return RuleHit(
        0.66,
        f"`{name}` is marked visited only after it is removed from the worklist, so it can be "
        "added many times before it is ever marked",
        line,
    )


def _queue_names(features: CodeFeatures) -> set:
    names = set()
    for assign in features.assignments:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.(pop|popleft)\(", assign.value or ""):
            names.add((assign.value or "").split(".")[0])
    return names


def _marks_at_enqueue(features: CodeFeatures) -> List[int]:
    """Marks applied to a neighbour when it is discovered.
    queue.append(neighbour) is an enqueue not a mark, so calls on the worklist
    itself are excluded, otherwise every correct BFS looks like it marks on
    insertion and the rule never fires.
    """
    targets = {loop.target for loop in features.loops if loop.kind == "for" and loop.target}
    worklists = _queue_names(features)
    return [
        call.line
        for call in features.calls
        if call.name in ("add", "append")
        and call.args
        and call.args[0].strip() in targets
        and call.receiver not in worklists
    ]


def _lifo_worklist_for_bfs(ctx: RuleContext) -> Optional[RuleHit]:
    """A stack where the task needs a queue.
    Swapping the worklist discipline silently turns BFS into DFS. It still
    terminates and visits everything so tests on connected graphs often pass,
    which is why this needs a structural rule and not a test.
    """
    required = set(ctx.problem.concepts) | set(ctx.problem.tags)
    if not ({"bfs", "shortest-path-unweighted"} & required):
        return None
    features = ctx.features
    for assign in features.assignments:
        value = (assign.value or "").strip()
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.pop\(\s*\)$", value)
        if not match:
            continue
        container = match.group(1)
        if features.container_kinds.get(container) == "deque":
            continue
        return RuleHit(
            0.58,
            f"`{value}` removes the most recently added item, so `{container}` behaves as a "
            "stack; a breadth-first traversal needs first-in-first-out order",
            assign.line,
        )
    return None


def _list_as_visited(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    for comparison in features.comparisons:
        if comparison.op not in ("in", "not in"):
            continue
        container = comparison.right.strip()
        if features.container_kinds.get(container) != "list":
            continue
        if not _queue_names(features):
            continue                      # no worklist, not a traversal
        loop = features.loop_containing(comparison.line)
        if loop is None:
            continue
        return RuleHit(
            0.55,
            f"`{comparison.source}` scans the list `{container}` on every iteration; a set "
            "would answer the same question in expected constant time",
            comparison.line,
        )
    return None


def _list_as_queue(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    for call in features.calls:
        if call.name != "pop" or not call.args or call.args[0].strip() != "0":
            continue
        if features.container_kinds.get(call.receiver or "") == "deque":
            continue
        return RuleHit(
            0.58,
            f"`{call.source}` removes from the front of a list, which shifts every remaining "
            "element and costs O(n) per call",
            call.line,
        )
    return None


def _membership_in_list(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    for comparison in features.comparisons:
        if comparison.op not in ("in", "not in"):
            continue
        container = comparison.right.strip()
        kind = features.container_kinds.get(container)
        if kind not in ("list", None):
            continue
        if kind is None and container not in features.parameters:
            continue
        loop = features.loop_containing(comparison.line)
        if loop is None or loop.depth < 1:
            continue
        return RuleHit(
            0.42,
            f"`{comparison.source}` performs a linear membership test inside a loop",
            comparison.line,
        )
    return None


def _asymptotic_gap(ctx: RuleContext) -> Optional[RuleHit]:
    if not ctx.has_reference:
        return None
    student = ctx.complexity.total
    reference = ctx.reference_complexity.total
    if not student.dominates(reference):
        return None
    belief = 0.68 if student.is_exponential else 0.52
    return RuleHit(
        belief,
        f"the submission runs in {student} where the intended solution runs in {reference}",
        ctx.features.loops[0].line if ctx.features.loops else None,
    )


def _sort_inside_loop(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    for call in features.calls:
        if call.name not in ("sorted", "sort"):
            continue
        loop = features.loop_containing(call.line)
        if loop is None or call.line <= loop.line:
            continue                      # in the header, evaluated once
        return RuleHit(
            0.50,
            f"`{call.source}` runs on every iteration of the loop starting at line {loop.line}",
            call.line,
        )
    return None


def _linear_scan_when_ordered(ctx: RuleContext) -> Optional[RuleHit]:
    if not ctx.has_reference:
        return None
    reference = ctx.reference_complexity.total
    student = ctx.complexity.total
    if reference.log_power == 0 or reference.poly > 0.0:
        return None                                  # target is not logarithmic
    if not student.dominates(reference):
        return None
    for loop in ctx.features.loops:
        if loop.kind == "for" and loop.iterable:
            return RuleHit(
                0.56,
                f"the intended solution is {reference} because the input is ordered, but the "
                f"submission scans it with `for ... in {loop.iterable}`",
                loop.line,
            )
    return None


def _mutable_default(ctx: RuleContext) -> Optional[RuleHit]:
    defaults = ctx.features.mutable_defaults
    if not defaults:
        return None
    name, line = defaults[0]
    if ctx.features.memo_arity and ctx.features.is_recursive:
        # a shared dict used on purpose as a memo table is less bad but it still
        # leaks state across calls, report with lower belief
        return RuleHit(
            0.34,
            f"`{name}` is a mutable default used as a cache: it is created once and shared by "
            "every call, so results leak between separate invocations",
            line,
        )
    return RuleHit(
        0.72,
        f"`{name}` has a mutable default, created once at definition time and shared by all calls",
        line,
    )


_TRUE_DIVISION_RE = re.compile(r"(?<!/)/(?!/)")


def _float_index(ctx: RuleContext) -> Optional[RuleHit]:
    """Flag index expressions that are (or come from) true division."""
    for index in ctx.features.indexes:
        if _TRUE_DIVISION_RE.search(index.index):
            return RuleHit(
                0.74,
                f"`{index.source}` indexes with `{index.index}`, and `/` yields a float",
                index.line,
            )
    # an index is usually a variable, so follow it back one step to where it was
    # computed: 'mid = (lo + hi) / 2' is the real defect, a[mid] just the symptom
    named = {i.index.strip(): i for i in ctx.features.indexes}
    for assign in ctx.features.assignments:
        target = assign.target.strip()
        if target in named and _TRUE_DIVISION_RE.search(assign.value or ""):
            return RuleHit(
                0.74,
                f"`{target}` is computed as `{assign.value}`, and `/` yields a float, "
                f"which cannot index `{named[target].container}`",
                assign.line,
            )
    return None


def _identity_vs_equality(ctx: RuleContext) -> Optional[RuleHit]:
    for comparison in ctx.features.comparisons:
        if comparison.op not in ("is", "is not"):
            continue
        if "None" in (comparison.left, comparison.right):
            continue
        return RuleHit(
            0.70,
            f"`{comparison.source}` compares identity; equal values may still be distinct objects",
            comparison.line,
        )
    return None


def _aliasing_copy(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    mutators = {"append", "extend", "insert", "pop", "remove", "sort", "clear", "add", "update"}
    for assign in features.assignments:
        value = assign.value.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", assign.target):
            continue
        if features.container_kinds.get(value) not in ("list", "dict", "set", "deque"):
            if value not in features.parameters:
                continue
        mutated = [
            call for call in features.calls
            if call.receiver == assign.target and call.name in mutators
        ]
        if mutated:
            return RuleHit(
                0.54,
                f"`{assign.target} = {value}` binds a second name to the same object, and "
                f"`{mutated[0].source}` then mutates it through that name",
                assign.line,
            )
    return None


def _mutate_during_iteration(ctx: RuleContext) -> Optional[RuleHit]:
    features = ctx.features
    mutators = {"append", "remove", "insert", "pop", "clear", "extend"}
    for loop in features.loops:
        if loop.kind != "for" or not loop.iterable:
            continue
        container = loop.iterable.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", container):
            continue
        lo, hi = loop.body_lines
        for call in features.calls:
            if call.receiver == container and call.name in mutators and lo <= call.line <= hi:
                return RuleHit(
                    0.66,
                    f"`{call.source}` modifies `{container}` while the loop at line {loop.line} "
                    "is iterating over it",
                    call.line,
                )
    return None


def _uninitialised(ctx: RuleContext) -> Optional[RuleHit]:
    if ctx.dataflow is None or not ctx.dataflow.uninitialised:
        return None
    known = set(ctx.features.identifiers)
    candidates = [
        (name, line) for name, line in sorted(ctx.dataflow.uninitialised)
        if name in known
    ]
    if not candidates:
        return None
    name, line = candidates[0]
    return RuleHit(0.62, f"`{name}` is read at line {line} before any assignment can reach it", line)


def _dead_store(ctx: RuleContext) -> Optional[RuleHit]:
    if ctx.dataflow is None or not ctx.dataflow.dead_stores:
        return None
    name, _, line = sorted(ctx.dataflow.dead_stores)[0]
    return RuleHit(0.28, f"`{name}` is assigned at line {line} but never read afterwards", line)


def _runtime_error(mid: str, *error_types: str) -> Detector:
    wanted = set(error_types)

    def detect(ctx: RuleContext) -> Optional[RuleHit]:
        execution = ctx.execution
        counterexample = ctx.counterexample
        error = None
        line = None
        if execution is not None and execution.error_type in wanted:
            error, line = execution.error, execution.error_line
        elif counterexample is not None and counterexample.student_error:
            for candidate in wanted:
                if counterexample.student_error.startswith(candidate) or candidate in counterexample.student_error:
                    error = counterexample.student_error
                    break
        if error is None:
            return None
        return RuleHit(0.80, f"execution raised {error}", line)

    return detect


def _wrong_comparison_boundary(ctx: RuleContext) -> Optional[RuleHit]:
    divergence = ctx.divergence
    if divergence is None or divergence.kind not in (
        DivergenceKind.EARLY_TERMINATION,
        DivergenceKind.OUTPUT_MISMATCH,
    ):
        return None
    line = divergence.student_line
    if line is None:
        return None
    nearby = [
        comparison for comparison in ctx.features.comparisons
        if comparison.op in ("<", "<=", ">", ">=") and abs(comparison.line - line) <= 1
    ]
    if not nearby:
        return None
    comparison = nearby[0]
    return RuleHit(
        0.45,
        f"the executions part company at `{comparison.source}`, a comparison whose strictness "
        "decides whether the boundary value is examined",
        comparison.line,
    )


# the rule set
RULES: Tuple[Rule, ...] = (
    Rule("bs.interval-convention-mismatch", EvidenceKind.STATIC, _interval_convention),
    Rule("bs.no-progress-update", EvidenceKind.STATIC, _no_progress_update),
    Rule("loop.non-termination", EvidenceKind.STATIC, _loop_non_termination),
    Rule("loop.off-by-one-index", EvidenceKind.STATIC, _off_by_one_index),
    Rule("rec.missing-base-case", EvidenceKind.STATIC, _missing_base_case),
    Rule("rec.unreachable-base-case", EvidenceKind.DYNAMIC, _unreachable_base_case),
    Rule("rec.result-discarded", EvidenceKind.STATIC, _result_discarded),
    Rule("dp.recomputed-subproblems", EvidenceKind.COMPLEXITY, _recomputed_subproblems),
    Rule("dp.transition-order", EvidenceKind.STATIC, _transition_order),
    Rule("graph.visited-on-dequeue", EvidenceKind.STATIC, _visited_on_dequeue),
    Rule("ds.wrong-container-choice", EvidenceKind.STATIC, _lifo_worklist_for_bfs),
    Rule("graph.list-as-visited", EvidenceKind.COMPLEXITY, _list_as_visited),
    Rule("graph.list-as-queue", EvidenceKind.COMPLEXITY, _list_as_queue),
    Rule("cx.membership-in-list", EvidenceKind.COMPLEXITY, _membership_in_list),
    Rule("cx.asymptotic-gap", EvidenceKind.COMPLEXITY, _asymptotic_gap),
    Rule("cx.sort-inside-loop", EvidenceKind.COMPLEXITY, _sort_inside_loop),
    Rule("search.linear-scan-when-ordered", EvidenceKind.COMPLEXITY, _linear_scan_when_ordered),
    Rule("py.mutable-default-argument", EvidenceKind.STATIC, _mutable_default),
    Rule("py.float-index", EvidenceKind.STATIC, _float_index),
    Rule("py.identity-vs-equality", EvidenceKind.STATIC, _identity_vs_equality),
    Rule("py.aliasing-copy", EvidenceKind.STATIC, _aliasing_copy),
    Rule("py.mutate-during-iteration", EvidenceKind.STATIC, _mutate_during_iteration),
    Rule("var.uninitialised", EvidenceKind.STATIC, _uninitialised),
    Rule("var.dead-store", EvidenceKind.STATIC, _dead_store),
    Rule("run.index-out-of-range", EvidenceKind.DYNAMIC, _runtime_error("run.index-out-of-range", "IndexError", "KeyError")),
    Rule("run.type-error", EvidenceKind.DYNAMIC, _runtime_error("run.type-error", "TypeError", "ValueError", "AttributeError")),
    Rule("logic.wrong-comparison-boundary", EvidenceKind.DYNAMIC, _wrong_comparison_boundary),
)

# misconceptions about cost or hygiene, not correctness. A submission can
# have these and still be right on every input, so they never compete with
# an explanation of a wrong answer.
COST_ONLY: frozenset = frozenset(
    {
        "cx.asymptotic-gap",
        "cx.membership-in-list",
        "cx.sort-inside-loop",
        "graph.list-as-queue",
        "graph.list-as-visited",
        "search.linear-scan-when-ordered",
        "var.dead-store",
        "py.mutable-default-argument",
    }
)


# cx.asymptotic-gap says THAT a submission is too slow, these say WHY. When
# one of them is present the gap is its symptom and should not be the
# headline ("you re-solve the same subproblem" teaches more than "your
# solution is O(n^2)").
SUBSUMES: Mapping[str, frozenset] = {
    "cx.asymptotic-gap": frozenset(
        {
            "dp.recomputed-subproblems",
            "graph.list-as-queue",
            "graph.list-as-visited",
            "cx.membership-in-list",
            "cx.sort-inside-loop",
            "search.linear-scan-when-ordered",
            "ds.wrong-container-choice",
        }
    ),
}


def is_subsumed(mid: str, present: Sequence[str]) -> bool:
    """True if a more specific finding already explains mid."""
    causes = SUBSUMES.get(mid)
    return bool(causes and causes.intersection(present))


def is_correctness_misconception(mid: str) -> bool:
    """True if this misconception can explain an observably wrong answer."""
    return mid not in COST_ONLY


# rules only the model can give evidence for
MODEL_ONLY: Tuple[str, ...] = ("greedy.local-optimum-assumed",)


def validate_taxonomy(concept_ids: Sequence[str]) -> List[str]:
    """Referential integrity problems in the taxonomy (empty if fine)."""
    known = set(concept_ids)
    problems: List[str] = []
    for entry in _TAXONOMY:
        unknown = [c for c in entry.concepts if c not in known]
        if unknown:
            problems.append(f"{entry.mid} references unknown concepts: {unknown}")
        if not entry.concepts:
            problems.append(f"{entry.mid} implicates no concept")
    for rule in RULES:
        if rule.misconception_id not in MISCONCEPTIONS:
            problems.append(f"rule targets unknown misconception: {rule.misconception_id}")
    return problems
