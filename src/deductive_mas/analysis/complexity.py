"""Static time complexity estimate.

We need our own opinion about the cost of a program to compare it with the
reference. Syntax directed, over the cost semiring O(n^p log^q n) plus a top
element for exponential:

    cost(s1; s2)         = max(cost(s1), cost(s2))
    cost(if t: A else B) = cost(t) + max(cost(A), cost(B))
    cost(for/while)      = trip_count * cost(body)
    cost(f(x))           = table lookup, or the inferred cost of f

Recursive functions: pull a recurrence T(n) = a T(n/b) + f(n) (or
T(n) = a T(n-1) + f(n)) from the call sites and solve it with the master
theorem / unrolling.

This is an upper bound heuristic, not a proof. It over-approximates data
dependent loops on purpose, and the dynamic layer double checks anything
that ends up in a diagnosis.
"""

import ast
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


# the cost lattice
@dataclass(frozen=True, order=False)
class Asymptotic:
    """An element of { n^poly * log^log_power n } u { base^n }."""

    poly: float = 0.0
    log_power: int = 0
    exponential_base: Optional[float] = None

    # constructors
    @staticmethod
    def constant() -> "Asymptotic":
        return Asymptotic(0.0, 0)

    @staticmethod
    def logarithmic(power: int = 1) -> "Asymptotic":
        return Asymptotic(0.0, power)

    @staticmethod
    def linear() -> "Asymptotic":
        return Asymptotic(1.0, 0)

    @staticmethod
    def linearithmic() -> "Asymptotic":
        return Asymptotic(1.0, 1)

    @staticmethod
    def polynomial(degree: float) -> "Asymptotic":
        return Asymptotic(float(degree), 0)

    @staticmethod
    def exponential(base: float = 2.0) -> "Asymptotic":
        return Asymptotic(0.0, 0, base)

    # algebra
    @property
    def is_exponential(self) -> bool:
        return self.exponential_base is not None

    def _key(self) -> Tuple[int, float, float]:
        if self.is_exponential:
            return (1, float(self.exponential_base or 2.0), 0.0)
        return (0, self.poly, float(self.log_power))

    def __mul__(self, other: "Asymptotic") -> "Asymptotic":
        if self.is_exponential or other.is_exponential:
            bases = [b for b in (self.exponential_base, other.exponential_base) if b]
            return Asymptotic.exponential(max(bases) if bases else 2.0)
        return Asymptotic(self.poly + other.poly, self.log_power + other.log_power)

    def __lt__(self, other: "Asymptotic") -> bool:
        return self._key() < other._key()

    def __le__(self, other: "Asymptotic") -> bool:
        return self._key() <= other._key()

    def dominates(self, other: "Asymptotic") -> bool:
        return other._key() < self._key()

    @staticmethod
    def max(*items: "Asymptotic") -> "Asymptotic":
        best = Asymptotic.constant()
        for item in items:
            if item.dominates(best):
                best = item
        return best

    # display
    def __str__(self) -> str:
        if self.is_exponential:
            base = self.exponential_base or 2.0
            shown = int(base) if float(base).is_integer() else round(base, 2)
            return f"O({shown}^n)"
        parts: List[str] = []
        p = self.poly
        if abs(p - 1.0) < 1e-9:
            parts.append("n")
        elif abs(p - 0.5) < 1e-9:
            parts.append("sqrt(n)")
        elif p > 1e-9:
            shown = int(p) if float(p).is_integer() else round(p, 2)
            parts.append(f"n^{shown}")
        if self.log_power == 1:
            parts.append("log n")
        elif self.log_power > 1:
            parts.append(f"log^{self.log_power} n")
        if not parts:
            return "O(1)"
        return "O(" + " ".join(parts) + ")"

    __repr__ = __str__


# library cost table
_LINEAR = Asymptotic.linear()
_LOG = Asymptotic.logarithmic()
_NLOGN = Asymptotic.linearithmic()
_CONST = Asymptotic.constant()

# cost of a call, keyed by the bare function / method name
LIBRARY_COST: Dict[str, Asymptotic] = {
    "sorted": _NLOGN, "sort": _NLOGN,
    "sum": _LINEAR, "min": _LINEAR, "max": _LINEAR, "any": _LINEAR, "all": _LINEAR,
    "count": _LINEAR, "index": _LINEAR, "remove": _LINEAR, "reverse": _LINEAR,
    "copy": _LINEAR, "join": _LINEAR, "split": _LINEAR, "extend": _LINEAR,
    "list": _LINEAR, "set": _LINEAR, "tuple": _LINEAR, "dict": _LINEAR,
    "reversed": _LINEAR, "enumerate": _CONST, "zip": _CONST, "map": _CONST,
    "filter": _CONST, "range": _CONST, "len": _CONST, "abs": _CONST,
    "append": _CONST, "add": _CONST, "popleft": _CONST, "appendleft": _CONST,
    "heappush": _LOG, "heappop": _LOG, "heapify": _LINEAR,
    "bisect_left": _LOG, "bisect_right": _LOG, "insort": _LINEAR,
    "sqrt": _CONST, "log": _CONST, "pow": _CONST, "gcd": _LOG,
    "deque": _LINEAR, "Counter": _LINEAR, "defaultdict": _CONST,
}

# methods that are linear because of a known pitfall, with an explanation
PITFALL_CALLS: Dict[str, str] = {
    "pop(0)": "list.pop(0) shifts every remaining element: O(n) per call, not O(1)",
    "insert(0)": "list.insert(0, x) shifts every element: O(n) per call",
}


# reports
@dataclass
class LoopFinding:
    line: int
    kind: str                      # "for" or "while"
    trip_count: Asymptotic
    reason: str
    depth: int


@dataclass
class ComplexityReport:
    function: str
    total: Asymptotic = field(default_factory=Asymptotic.constant)
    loops: List[LoopFinding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    recurrence: Optional[str] = None
    max_loop_depth: int = 0
    is_recursive: bool = False

    def __str__(self) -> str:
        return str(self.total)


# the estimator
class ComplexityEstimator:
    """Cost inference for one module."""

    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.functions: Dict[str, ast.FunctionDef] = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self._cache: Dict[str, Asymptotic] = {}
        self._in_progress: Set[str] = set()
        self._containers: Dict[str, str] = {}
        self._amortised: Set[str] = set()
        self._current = ComplexityReport(function="<module>")

    # entry api
    def report(self, name: str) -> ComplexityReport:
        func = self.functions.get(name)
        if func is None:
            return ComplexityReport(function=name, notes=[f"function {name!r} not found"])
        rep = ComplexityReport(function=name)
        self._current = rep
        rep.is_recursive = self._is_recursive(func)
        # self calls count as O(1) while we measure f(n), their cost is what the
        # recurrence is about so counting it here would be circular
        self._in_progress.add(name)
        self._containers = _container_kinds(func)
        try:
            body_cost = self._block(func.body, depth=0, params=_param_names(func))
        finally:
            self._in_progress.discard(name)
        rep.max_loop_depth = max([f.depth for f in rep.loops], default=0)

        if rep.is_recursive:
            rep.total = self._solve_recurrence(func, body_cost, rep)
        else:
            rep.total = body_cost
        return rep

    def cost_of(self, name: str) -> Asymptotic:
        """Cost of calling name. Recursion is broken with an O(1) assumption."""
        if name in self._cache:
            return self._cache[name]
        if name in self._in_progress:
            return _CONST
        func = self.functions.get(name)
        if func is None:
            return LIBRARY_COST.get(name, _CONST)
        self._in_progress.add(name)
        saved = getattr(self, "_current", None)
        saved_containers = getattr(self, "_containers", {})
        scratch = ComplexityReport(function=name)
        self._current = scratch
        self._containers = _container_kinds(func)
        cost = self._block(func.body, depth=0, params=_param_names(func))
        if self._is_recursive(func):
            cost = self._solve_recurrence(func, cost, scratch)
        self._current = saved if saved is not None else scratch
        self._containers = saved_containers
        self._in_progress.discard(name)
        self._cache[name] = cost
        return cost

    # statements
    def _block(self, body: Sequence[ast.stmt], depth: int, params: Set[str]) -> Asymptotic:
        return Asymptotic.max(*[self._statement(s, depth, params) for s in body]) if body else _CONST

    def _statement(self, stmt: ast.stmt, depth: int, params: Set[str]) -> Asymptotic:
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            trips, reason = self._for_trip_count(stmt, params)
            if self._is_amortised_adjacency(stmt):
                trips = _CONST
                reason = "adjacency list: amortised over all edges of the traversal"
            self._current.loops.append(
                LoopFinding(stmt.lineno, "for", trips, reason, depth + 1)
            )
            body = self._block(stmt.body, depth + 1, params)
            body = Asymptotic.max(body, self._expr(stmt.iter, params))
            return trips * body
        if isinstance(stmt, ast.While):
            trips, reason = self._while_trip_count(stmt, params)
            popped = _worklist_popped_vars(stmt)
            if popped:
                # classic BFS/DFS shape: the inner 'for v in adj[u]' runs once per edge
                # over the whole traversal, so the loops add up to O(V + E) instead of
                # multiplying to O(V * E)
                reason = "worklist traversal: inner adjacency loop is amortised over all edges"
                self._current.loops.append(
                    LoopFinding(stmt.lineno, "while", trips, reason, depth + 1)
                )
                self._note("graph traversal cost is O(V + E) amortised over all adjacency lists")
                previous = self._amortised
                self._amortised = previous | popped
                try:
                    body = self._block(stmt.body, depth + 1, params)
                finally:
                    self._amortised = previous
                return trips * body
            self._current.loops.append(
                LoopFinding(stmt.lineno, "while", trips, reason, depth + 1)
            )
            body = self._block(stmt.body, depth + 1, params)
            return trips * Asymptotic.max(body, self._expr(stmt.test, params))
        if isinstance(stmt, ast.If):
            branches = Asymptotic.max(
                self._block(stmt.body, depth, params), self._block(stmt.orelse, depth, params)
            )
            return Asymptotic.max(self._expr(stmt.test, params), branches)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            return self._block(stmt.body, depth, params)
        if isinstance(stmt, ast.Try):
            parts = [self._block(stmt.body, depth, params)]
            parts += [self._block(h.body, depth, params) for h in stmt.handlers]
            parts.append(self._block(stmt.orelse, depth, params))
            parts.append(self._block(stmt.finalbody, depth, params))
            return Asymptotic.max(*parts)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return _CONST
        return Asymptotic.max(*[self._expr(e, params) for e in _expressions_of(stmt)]) if _expressions_of(stmt) else _CONST

    # expressions
    def _expr(self, node: Optional[ast.expr], params: Set[str]) -> Asymptotic:
        if node is None:
            return _CONST
        worst = _CONST
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                worst = Asymptotic.max(worst, self._call(sub, params))
            elif isinstance(sub, ast.Compare):
                for op, right in zip(sub.ops, sub.comparators):
                    if not isinstance(op, (ast.In, ast.NotIn)):
                        continue
                    kind = self._container_kind(right)
                    if kind in ("set", "dict"):
                        continue                      # hashed lookup, O(1)
                    worst = Asymptotic.max(worst, _LINEAR)
                    self._note(
                        "membership test `in` over a list/tuple is O(n); "
                        "a set or dict makes it O(1)"
                    )
            elif isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
                worst = Asymptotic.max(worst, _LINEAR)
                self._note("slicing copies the sliced range: O(k)")
            elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                inner = _CONST
                for gen in sub.generators:
                    trips, _ = _iterable_trip_count(gen.iter, params)
                    inner = inner * trips
                worst = Asymptotic.max(worst, inner)
        return worst

    def _call(self, node: ast.Call, params: Set[str]) -> Asymptotic:
        name = _call_name(node)
        if name is None:
            return _CONST
        if name in _VARIADIC_REDUCERS and len(node.args) >= 2:
            return _CONST                      # max(a, b) is O(1), max(xs) is O(n)
        if name in ("pop", "insert") and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == 0:
                self._note(PITFALL_CALLS[f"{name}(0)"])
                return _LINEAR
        if name in self.functions:
            return self.cost_of(name)
        return LIBRARY_COST.get(name, _CONST)

    def _is_amortised_adjacency(self, stmt: ast.stmt) -> bool:
        """True for 'for v in adj[u]' where u came off the worklist."""
        if not self._amortised:
            return False
        iterable = getattr(stmt, "iter", None)
        if isinstance(iterable, ast.Call) and iterable.args:
            iterable = iterable.args[0]
        if isinstance(iterable, ast.Subscript):
            return bool(_names_in(iterable.slice) & self._amortised)
        if isinstance(iterable, ast.Call):
            return bool(_names_in(iterable) & self._amortised)
        return False

    def _container_kind(self, node: ast.expr) -> Optional[str]:
        """Best effort local type of a container expression."""
        containers = getattr(self, "_containers", {})
        if isinstance(node, ast.Name):
            return containers.get(node.id)
        if isinstance(node, ast.Set) or isinstance(node, ast.SetComp):
            return "set"
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(node, (ast.List, ast.ListComp, ast.Tuple)):
            return "list"
        if isinstance(node, ast.Call):
            return _CONTAINER_CONSTRUCTORS.get(_call_name(node) or "")
        return None

    def _note(self, text: str):
        current = getattr(self, "_current", None)
        if current is not None and text not in current.notes:
            current.notes.append(text)

    # trip counts
    def _for_trip_count(self, stmt: ast.stmt, params: Set[str]) -> Tuple[Asymptotic, str]:
        return _iterable_trip_count(getattr(stmt, "iter", None), params)

    def _while_trip_count(self, stmt: ast.While, params: Set[str]) -> Tuple[Asymptotic, str]:
        """Classify a while loop by how its guard variables change.

        - a guard variable halved (x //= 2) or a search interval narrowed around
          a midpoint -> O(log n)
        - doubled (x *= 2) -> O(log n)
        - incremented by a constant -> O(n)
        - nothing written at all -> divergent
        """
        guard = _names_in(stmt.test)
        updates = _updates_in(stmt.body)
        if not guard:
            return _LINEAR, "unbounded guard, assumed linear"
        touched = {name for name in guard if name in updates}
        if not touched:
            return _LINEAR, "guard variables are never updated in the body (loop may diverge)"

        midpoint = _has_midpoint_narrowing(stmt)
        for name in sorted(touched):
            kind = updates[name]
            if kind in ("halve", "double"):
                return _LOG, f"`{name}` is {'halved' if kind == 'halve' else 'doubled'} each iteration"
        if midpoint:
            return _LOG, "search interval is halved around a midpoint each iteration"
        for name in sorted(touched):
            if updates[name] == "step":
                return _LINEAR, f"`{name}` changes by a constant step each iteration"
        return _LINEAR, "guard variable updated by a data-dependent amount, assumed linear"

    # recurrences
    def _is_recursive(self, func: ast.FunctionDef) -> bool:
        return bool(_self_calls(func))

    def _solve_recurrence(
        self, func: ast.FunctionDef, body_cost: Asymptotic, rep: ComplexityReport
    ) -> Asymptotic:
        calls = _self_calls(func)
        a = len(calls)
        params = _param_names(func)
        shrink = _shrink_kind(calls, params)
        f = body_cost
        looped = _self_calls_inside_loop(func)
        if looped and a < 2:
            # 'for coin in coins: min_coins(coins, amount - coin)' makes one call per
            # iteration so the branching factor is len(coins), not 1. Any branching
            # factor > 1 makes a decrementing recursion exponential.
            a = 2

        memo_arity = _memoisation_arity(func)
        if memo_arity:
            # with a cache every distinct state is computed once, so cost =
            # (number of states) x (work per state), not a branching recurrence. This
            # is the overlapping subproblems idea behind DP.
            rep.recurrence = (
                f"memoised: states = O(n^{memo_arity}), work/state = {f}"
            )
            rep.notes.append(
                f"recursion is memoised on a {memo_arity}-dimensional key, so "
                "overlapping subproblems are solved once"
            )
            return Asymptotic.polynomial(float(memo_arity)) * f

        if shrink == "divide":
            b = 2.0
            critical = math.log(a, b) if a > 0 else 0.0
            rep.recurrence = f"T(n) = {a} T(n/2) + {f}"
            if f.is_exponential:
                return f
            if critical > f.poly + 1e-9:
                return Asymptotic.polynomial(round(critical, 4))
            if abs(critical - f.poly) < 1e-9:
                return Asymptotic(f.poly, f.log_power + 1)
            return f
        if shrink == "decrement":
            rep.recurrence = f"T(n) = {a} T(n-1) + {f}"
            if a >= 2:
                return Asymptotic.exponential(float(a))
            return _LINEAR * f
        rep.recurrence = f"T(n) = {a} T(?) + {f} (unresolved argument reduction)"
        if a >= 2:
            return Asymptotic.exponential(float(a))
        return _LINEAR * f


# AST helpers
def _param_names(func: ast.FunctionDef) -> Set[str]:
    args = func.args
    names = {a.arg for a in list(args.args) + list(args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _names_in(node: Optional[ast.AST]) -> Set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _expressions_of(stmt: ast.stmt) -> List[ast.expr]:
    out: List[ast.expr] = []
    for field_name, value in ast.iter_fields(stmt):
        if isinstance(value, ast.expr):
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, ast.expr))
    return out


def _iterable_trip_count(node: Optional[ast.expr], params: Set[str]) -> Tuple[Asymptotic, str]:
    """Trip count of 'for _ in <node>'."""
    if node is None:
        return _LINEAR, "unknown iterable"
    if isinstance(node, ast.Call) and _call_name(node) == "range":
        args = node.args
        bound = args[1] if len(args) >= 2 else (args[0] if args else None)
        if len(args) >= 3 and isinstance(args[2], ast.Constant):
            pass  # constant stride, same class
        return _range_bound(bound, params)
    if isinstance(node, ast.Call) and _call_name(node) in ("enumerate", "reversed", "sorted", "list", "set"):
        inner = node.args[0] if node.args else None
        trips, reason = _iterable_trip_count(inner, params)
        return trips, reason
    if isinstance(node, ast.Name):
        return _LINEAR, f"iterates over `{node.id}`"
    if isinstance(node, (ast.List, ast.Tuple)):
        return _CONST, "constant-size literal"
    return _LINEAR, "iterates over a collection"


def _range_bound(node: Optional[ast.expr], params: Set[str]) -> Tuple[Asymptotic, str]:
    if node is None:
        return _LINEAR, "range with unknown bound"
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return _CONST, f"constant bound {node.value}"
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "len":
            return _LINEAR, "range(len(...)) is linear in the collection"
        if name in ("isqrt", "sqrt"):
            return Asymptotic.polynomial(0.5), "range up to sqrt(n)"
        if name == "int" and node.args:
            return _range_bound(node.args[0], params)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            exponent = node.right.value
            if isinstance(exponent, (int, float)):
                left, _ = _range_bound(node.left, params)
                if abs(float(exponent) - 0.5) < 1e-9:
                    return Asymptotic.polynomial(0.5), "range up to n**0.5"
                return Asymptotic.polynomial(left.poly * float(exponent)), f"range up to n**{exponent}"
        if isinstance(node.op, ast.Mult):
            left, lr = _range_bound(node.left, params)
            right, rr = _range_bound(node.right, params)
            return left * right, f"{lr} times {rr}"
        left, reason = _range_bound(node.left, params)
        if left != _CONST:
            return left, reason
        return _range_bound(node.right, params)
    if isinstance(node, ast.Name):
        return _LINEAR, f"range up to `{node.id}`"
    return _LINEAR, "range with a data-dependent bound"


def _updates_in(body: Sequence[ast.stmt]) -> Dict[str, str]:
    """Map each variable written in body to how it changes:
    halve, double, step (constant additive) or other.
    """
    out: Dict[str, str] = {}

    def classify(target: str, value: ast.expr) -> str:
        if isinstance(value, ast.BinOp):
            op, left, right = value.op, value.left, value.right
            right_const = isinstance(right, ast.Constant) and isinstance(right.value, (int, float))
            mentions_self = target in _names_in(value)
            if isinstance(op, (ast.FloorDiv, ast.Div)) and right_const and mentions_self:
                return "halve" if float(right.value) > 1 else "other"
            if isinstance(op, ast.Mult) and right_const and mentions_self:
                return "double" if float(right.value) > 1 else "other"
            if isinstance(op, ast.RShift) and mentions_self:
                return "halve"
            if isinstance(op, ast.LShift) and mentions_self:
                return "double"
            if isinstance(op, (ast.Add, ast.Sub)) and right_const and mentions_self:
                return "step"
        return "other"

    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = _merge(out.get(target.id), classify(target.id, node.value))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                synth = ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=node.op, right=node.value)
                out[name] = _merge(out.get(name), classify(name, synth))
    return out


def _merge(existing: Optional[str], incoming: str) -> str:
    if existing is None:
        return incoming
    priority = {"halve": 3, "double": 3, "step": 2, "other": 1}
    return existing if priority[existing] >= priority[incoming] else incoming


def _has_midpoint_narrowing(stmt: ast.While) -> bool:
    """Detect the binary search shape: mid from two bounds, bounds move to mid."""
    mids: Set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
            value = node.value
            if isinstance(value.op, (ast.FloorDiv, ast.Div)) and isinstance(value.right, ast.Constant):
                if value.right.value == 2 and len(_names_in(value.left)) >= 2:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            mids.add(target.id)
            elif isinstance(value.op, ast.Add):
                # lo + (hi - lo) // 2
                right = value.right
                if (
                    isinstance(right, ast.BinOp)
                    and isinstance(right.op, (ast.FloorDiv, ast.Div))
                    and isinstance(right.right, ast.Constant)
                    and right.right.value == 2
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            mids.add(target.id)
    if not mids:
        return False
    guard = _names_in(stmt.test)
    for node in ast.walk(stmt):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in guard:
                    if mids & _names_in(node.value):
                        return True
    return False


_CONTAINER_CONSTRUCTORS: Dict[str, str] = {
    "set": "set", "frozenset": "set",
    "dict": "dict", "defaultdict": "dict", "Counter": "dict", "OrderedDict": "dict",
    "list": "list", "tuple": "list", "sorted": "list", "deque": "deque",
}


def _container_kinds(func: ast.FunctionDef) -> Dict[str, str]:
    """Infer the container type of locals from their initialisers.
    Tiny flow insensitive type inference, just enough to tell an O(1) hashed
    membership test from an O(n) scan.
    """
    kinds: Dict[str, str] = {}

    def kind_of(value: ast.expr) -> Optional[str]:
        if isinstance(value, (ast.Set, ast.SetComp)):
            return "set"
        if isinstance(value, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(value, (ast.List, ast.ListComp, ast.Tuple)):
            return "list"
        if isinstance(value, ast.Call):
            return _CONTAINER_CONSTRUCTORS.get(_call_name(value) or "")
        return None

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            inferred = kind_of(node.value)
            if inferred is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    previous = kinds.get(target.id)
                    kinds[target.id] = inferred if previous in (None, inferred) else "list"
    defaults = func.args.defaults
    positional = list(func.args.args)
    for arg, default in zip(positional[len(positional) - len(defaults):], defaults):
        inferred = kind_of(default)
        if inferred is not None:
            kinds[arg.arg] = inferred
    for arg in func.args.args:
        annotation = arg.annotation
        if isinstance(annotation, ast.Name) and annotation.id.lower() in _CONTAINER_CONSTRUCTORS:
            kinds[arg.arg] = _CONTAINER_CONSTRUCTORS[annotation.id.lower()]
    return kinds


def _memoisation_arity(func: ast.FunctionDef) -> int:
    """Memo key dimensionality, or 0 when the recursion is not cached.

    A cache = a guarded read and a write on the same subscripted container:

        if n in memo: return memo[n]     # guarded read
        memo[n] = ...                    # write
    """
    guarded: Dict[str, int] = {}
    written: Dict[str, int] = {}

    def arity(index: ast.expr) -> int:
        if isinstance(index, ast.Tuple):
            return len(index.elts)
        return 1

    for node in ast.walk(func):
        if isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(right, ast.Name):
                    guarded[right.id] = max(guarded.get(right.id, 0), arity(node.left))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            base = node.value
            depth = 1
            while isinstance(base, ast.Subscript):
                depth += 1
                base = base.value
            if isinstance(base, ast.Name):
                written[base.id] = max(written.get(base.id, 0), max(depth, arity(node.slice)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    base = target.value
                    depth = 1
                    while isinstance(base, ast.Subscript):
                        depth += 1
                        base = base.value
                    if isinstance(base, ast.Name):
                        written[base.id] = max(written.get(base.id, 0), max(depth, arity(target.slice)))

    best = 0
    for name, key_arity in written.items():
        if name in guarded:
            best = max(best, max(key_arity, guarded[name]))
    # functools.lru_cache / cache memoise on the full arg tuple
    for decorator in func.decorator_list:
        label = _call_name(decorator) if isinstance(decorator, ast.Call) else (
            decorator.attr if isinstance(decorator, ast.Attribute)
            else getattr(decorator, "id", "")
        )
        if label in ("lru_cache", "cache"):
            best = max(best, max(1, len(func.args.args)))
    return best


_VARIADIC_REDUCERS = frozenset({"max", "min", "sum", "pow", "divmod", "gcd"})


def _worklist_popped_vars(stmt: ast.While) -> Set[str]:
    """Recognise 'while queue: u = queue.pop(); for v in adj[u]: ...'.

    Returns the variables popped off the worklist (empty if the shape does not
    match) so the caller can mark the adjacency loops as amortised. Signature:
    (1) guard is a bare container, (2) body removes an element from it,
    (3) an inner for iterates something indexed by the removed element. Then
    the inner body runs once per edge in total, which is why BFS is linear.
    """
    if not isinstance(stmt.test, ast.Name):
        return set()
    container = stmt.test.id
    popped: Set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if _call_name(call) in ("pop", "popleft", "popitem", "get_nowait"):
                receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
                if isinstance(receiver, ast.Name) and receiver.id == container:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            popped.add(target.id)
    if not popped:
        return set()
    for node in ast.walk(stmt):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterable = node.iter
            if isinstance(iterable, ast.Call) and iterable.args:
                iterable = iterable.args[0]
            if isinstance(iterable, ast.Subscript) and _names_in(iterable.slice) & popped:
                return popped
            if isinstance(iterable, ast.Call) and _call_name(iterable) in ("neighbors", "adj"):
                return popped
    return set()


def _self_calls_inside_loop(func: ast.FunctionDef) -> bool:
    """True if at least one recursive call is inside a loop body."""
    for node in ast.walk(func):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _call_name(inner) == func.name:
                return True
    return False


def _self_calls(func: ast.FunctionDef) -> List[ast.Call]:
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and _call_name(node) == func.name
    ]


def _shrink_kind(calls: Sequence[ast.Call], params: Set[str]) -> str:
    """How the recursive argument shrinks: divide, decrement or unknown."""
    seen: Set[str] = set()
    for call in calls:
        for arg in list(call.args) + [kw.value for kw in call.keywords]:
            if isinstance(arg, ast.BinOp):
                if isinstance(arg.op, (ast.FloorDiv, ast.Div, ast.RShift)):
                    seen.add("divide")
                elif isinstance(arg.op, (ast.Sub, ast.Add)):
                    seen.add("decrement")
            elif isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                seen.add("divide")
            elif isinstance(arg, ast.Name) and arg.id not in params:
                seen.add("divide")
    if "divide" in seen:
        return "divide"
    if "decrement" in seen:
        return "decrement"
    return "unknown"


def estimate(source: str, entry_point: str) -> ComplexityReport:
    """Parse source and report on entry_point."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ComplexityReport(function=entry_point, notes=[f"syntax error: {exc.msg}"])
    return ComplexityEstimator(tree).report(entry_point)
