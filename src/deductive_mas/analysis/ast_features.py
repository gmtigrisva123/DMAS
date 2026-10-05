"""Structural features from the AST.

The misconception rules are written against the facts defined here, not
against raw AST nodes, so each rule stays a short readable predicate.
"""

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..domain import SourceSpan
from .complexity import _container_kinds  # reuse the local type inference


# fact records
@dataclass(frozen=True)
class ComparisonFact:
    line: int
    op: str
    left: str
    right: str
    source: str


@dataclass(frozen=True)
class IndexFact:
    line: int
    container: str
    index: str
    # constant offset in the index (a[i + 1] -> +1)
    offset: Optional[int]
    stores: bool
    source: str


@dataclass(frozen=True)
class AssignFact:
    line: int
    target: str
    value: str
    augmented: bool
    # names read on the right hand side
    reads: Tuple[str, ...]


@dataclass(frozen=True)
class CallFact:
    line: int
    name: str
    receiver: Optional[str]
    args: Tuple[str, ...]
    source: str


@dataclass(frozen=True)
class LoopFact:
    line: int
    kind: str                      # "for" or "while"
    target: Optional[str]
    iterable: Optional[str]
    guard: Optional[str]
    depth: int
    body_lines: Tuple[int, int]
    # names assigned anywhere in the loop body
    writes: Tuple[str, ...]
    # names read by the guard / iterable
    reads: Tuple[str, ...]


@dataclass
class CodeFeatures:
    """Everything the rule layer needs to know about the program structure."""

    source: str
    entry_point: str
    tree: Optional[ast.Module] = None
    function: Optional[ast.FunctionDef] = None
    parse_error: Optional[str] = None
    parse_error_line: Optional[int] = None

    node_counts: Counter = field(default_factory=Counter)
    max_depth: int = 0
    parameters: Tuple[str, ...] = ()
    identifiers: Set[str] = field(default_factory=set)
    container_kinds: Dict[str, str] = field(default_factory=dict)

    comparisons: Tuple[ComparisonFact, ...] = ()
    indexes: Tuple[IndexFact, ...] = ()
    assignments: Tuple[AssignFact, ...] = ()
    calls: Tuple[CallFact, ...] = ()
    loops: Tuple[LoopFact, ...] = ()

    returns: Tuple[int, ...] = ()
    is_recursive: bool = False
    self_call_lines: Tuple[int, ...] = ()
    # number of self call sites (two calls on one line = two)
    self_call_count: int = 0
    # True if a self call sits inside a loop (branching factor depends on data)
    self_call_in_loop: bool = False
    self_calls_discarded: Tuple[int, ...] = ()
    has_base_case: bool = False
    mutable_defaults: Tuple[Tuple[str, int], ...] = ()
    memo_arity: int = 0

    @property
    def parsed(self) -> bool:
        return self.function is not None

    @property
    def loop_depth(self) -> int:
        return max((loop.depth for loop in self.loops), default=0)

    def span(self, line: Optional[int]) -> Optional[SourceSpan]:
        if not line or line < 1:
            return None
        lines = self.source.splitlines()
        snippet = lines[line - 1].rstrip() if line - 1 < len(lines) else ""
        return SourceSpan(line=line, end_line=line, snippet=snippet)

    def calls_named(self, *names: str) -> List[CallFact]:
        wanted = set(names)
        return [c for c in self.calls if c.name in wanted]

    def loop_containing(self, line: int) -> Optional[LoopFact]:
        best: Optional[LoopFact] = None
        for loop in self.loops:
            lo, hi = loop.body_lines
            if lo <= line <= hi and (best is None or loop.depth > best.depth):
                best = loop
        return best


# extraction
_OP_NAMES = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not",
    ast.In: "in", ast.NotIn: "not in",
}


def _src(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _names(node: Optional[ast.AST]) -> Tuple[str, ...]:
    if node is None:
        return ()
    return tuple(sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)}))


def _constant_offset(index: ast.AST) -> Optional[int]:
    """i + 1 -> 1, i - 1 -> -1, i -> 0, anything else -> None."""
    if isinstance(index, ast.Name):
        return 0
    if isinstance(index, ast.BinOp) and isinstance(index.op, (ast.Add, ast.Sub)):
        if isinstance(index.right, ast.Constant) and isinstance(index.right.value, int):
            sign = 1 if isinstance(index.op, ast.Add) else -1
            return sign * index.right.value
    return None


def _depth(node: ast.AST, current: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return current
    return max(_depth(child, current + 1) for child in children)


def extract(source: str, entry_point: str) -> CodeFeatures:
    """Parse source and summarise the structure of entry_point."""
    features = CodeFeatures(source=source, entry_point=entry_point)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        features.parse_error = exc.msg
        features.parse_error_line = exc.lineno
        return features
    except (ValueError, MemoryError, RecursionError) as exc:
        features.parse_error = str(exc)
        return features

    features.tree = tree
    function = _find_function(tree, entry_point)
    if function is None:
        features.parse_error = f"function {entry_point!r} is not defined"
        return features
    features.function = function

    features.node_counts = Counter(type(n).__name__ for n in ast.walk(function))
    features.max_depth = _depth(function)
    features.parameters = tuple(a.arg for a in function.args.args)
    features.identifiers = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    features.container_kinds = _container_kinds(function)

    comparisons: List[ComparisonFact] = []
    indexes: List[IndexFact] = []
    assignments: List[AssignFact] = []
    calls: List[CallFact] = []
    returns: List[int] = []
    self_call_lines: List[int] = []
    discarded: List[int] = []

    for node in ast.walk(function):
        if isinstance(node, ast.Compare):
            left = _src(node.left)
            for op, right in zip(node.ops, node.comparators):
                comparisons.append(
                    ComparisonFact(
                        line=getattr(node, "lineno", 0),
                        op=_OP_NAMES.get(type(op), type(op).__name__),
                        left=left,
                        right=_src(right),
                        source=_src(node),
                    )
                )
                left = _src(right)
        elif isinstance(node, ast.Subscript):
            base = node.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and not isinstance(node.slice, ast.Slice):
                indexes.append(
                    IndexFact(
                        line=getattr(node, "lineno", 0),
                        container=base.id,
                        index=_src(node.slice),
                        offset=_constant_offset(node.slice),
                        stores=isinstance(node.ctx, ast.Store),
                        source=_src(node),
                    )
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assignments.append(
                    AssignFact(
                        line=getattr(node, "lineno", 0),
                        target=_src(target),
                        value=_src(node.value),
                        augmented=False,
                        reads=_names(node.value),
                    )
                )
        elif isinstance(node, ast.AugAssign):
            assignments.append(
                AssignFact(
                    line=getattr(node, "lineno", 0),
                    target=_src(node.target),
                    value=_src(node.value),
                    augmented=True,
                    reads=_names(node.value),
                )
            )
        elif isinstance(node, ast.Call):
            name, receiver = _call_identity(node)
            calls.append(
                CallFact(
                    line=getattr(node, "lineno", 0),
                    name=name or "",
                    receiver=receiver,
                    args=tuple(_src(a) for a in node.args),
                    source=_src(node),
                )
            )
            if name == function.name:
                self_call_lines.append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.Return):
            returns.append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name, _ = _call_identity(node.value)
            if name == function.name:
                discarded.append(getattr(node, "lineno", 0))

    features.comparisons = tuple(comparisons)
    features.indexes = tuple(indexes)
    features.assignments = tuple(assignments)
    features.calls = tuple(calls)
    features.returns = tuple(sorted(set(returns)))
    features.self_call_lines = tuple(sorted(set(self_call_lines)))
    features.self_call_count = len(self_call_lines)
    features.self_calls_discarded = tuple(sorted(set(discarded)))
    features.is_recursive = bool(self_call_lines)
    features.has_base_case = _has_base_case(function)
    features.mutable_defaults = _mutable_defaults(function)
    features.loops = tuple(_loops(function))
    features.self_call_in_loop = any(
        loop.body_lines[0] <= line <= loop.body_lines[1]
        for line in features.self_call_lines
        for loop in features.loops
    )

    from .complexity import _memoisation_arity  # local import, avoids a cycle

    features.memo_arity = _memoisation_arity(function)
    return features


def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if candidates:
        return candidates[0]
    # fall back to the first top level function so a renamed entry point still
    # gets analysed
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _call_identity(node: ast.Call) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(node.func, ast.Name):
        return node.func.id, None
    if isinstance(node.func, ast.Attribute):
        receiver = node.func.value
        return node.func.attr, receiver.id if isinstance(receiver, ast.Name) else _src(receiver)
    return None, None


def _has_base_case(function: ast.FunctionDef) -> bool:
    """True if some return is reachable without a recursive call."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if value is None:
            return True
        calls_self = any(
            isinstance(sub, ast.Call) and _call_identity(sub)[0] == function.name
            for sub in ast.walk(value)
        )
        if not calls_self:
            return True
    return False


def _mutable_defaults(function: ast.FunctionDef) -> Tuple[Tuple[str, int], ...]:
    out: List[Tuple[str, int]] = []
    args = function.args
    positional = list(args.args)
    defaults = list(args.defaults)
    paired = list(zip(positional[len(positional) - len(defaults) :], defaults))
    paired += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    for arg, default in paired:
        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
            out.append((arg.arg, getattr(default, "lineno", function.lineno)))
        elif isinstance(default, ast.Call):
            name, _ = _call_identity(default)
            if name in ("list", "dict", "set", "defaultdict", "Counter", "deque"):
                out.append((arg.arg, getattr(default, "lineno", function.lineno)))
    return tuple(out)


def _loops(function: ast.FunctionDef) -> List[LoopFact]:
    """Collect loops with nesting depth and body extent."""
    out: List[LoopFact] = []

    def visit(node: ast.AST, depth: int):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                body = list(getattr(child, "body", []))
                # nodes like ast.Load have no lineno, letting their 0 into the min makes
                # the loop look like it spans the whole file and every "inside a loop?"
                # check becomes true
                lines = [n for n in (getattr(s, "lineno", 0) for s in ast.walk(child)) if n]
                extent = (min(lines), max(lines)) if lines else (child.lineno, child.lineno)
                writes: Set[str] = set()
                for stmt in body:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            writes.add(sub.id)
                        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                            writes.add(sub.target.id)
                if isinstance(child, ast.While):
                    fact = LoopFact(
                        line=child.lineno, kind="while", target=None, iterable=None,
                        guard=_src(child.test), depth=depth + 1, body_lines=extent,
                        writes=tuple(sorted(writes)), reads=_names(child.test),
                    )
                else:
                    fact = LoopFact(
                        line=child.lineno, kind="for", target=_src(child.target),
                        iterable=_src(child.iter), guard=None, depth=depth + 1,
                        body_lines=extent, writes=tuple(sorted(writes)),
                        reads=_names(child.iter),
                    )
                out.append(fact)
                visit(child, depth + 1)
            else:
                visit(child, depth)

    visit(function, 0)
    return out
