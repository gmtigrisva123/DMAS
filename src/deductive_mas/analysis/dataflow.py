"""Dataflow analyses on the CFG. Two classic ones on the same worklist engine:

- reaching definitions (forward, may): which assignments reach a point. Lets
  us say "the bound you compare against was last written before the loop and
  never updated inside it".
- live variables (backward, may): which values are still needed. Dead
  assignments usually mean the student patched a symptom.

From reaching defs we also get def-use chains, loop carried dependencies and
possibly uninitialised uses.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .cfg import ControlFlowGraph

# a definition is (variable, statement index, line)
Definition = Tuple[str, int, int]


# def / use extraction
def _target_names(node: ast.AST) -> Tuple[Set[str], Set[str]]:
    """Split an assignment target into (defined names, incidentally used names).
    a[i] = v does not define a, it mutates it, so a is both a use and a weak
    def, and i is a plain use.
    """
    defined: Set[str] = set()
    used: Set[str] = set()
    if isinstance(node, ast.Name):
        defined.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            d, u = _target_names(elt)
            defined |= d
            used |= u
    elif isinstance(node, ast.Starred):
        d, u = _target_names(node.value)
        defined |= d
        used |= u
    elif isinstance(node, ast.Subscript):
        base = node.value
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            defined.add(base.id)
            used.add(base.id)
        used |= expression_uses(node.slice)
    elif isinstance(node, ast.Attribute):
        used |= expression_uses(node.value)
    return defined, used


def expression_uses(node: Optional[ast.AST]) -> Set[str]:
    """All free names loaded by an expression.
    Names bound by a nested lambda / comprehension are not uses of the outer
    scope: sorted(xs, key=lambda p: p[1]) reads xs, not p.
    """
    if node is None:
        return set()
    out: Set[str] = set()
    bound: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Lambda):
            args = sub.args
            bound |= {a.arg for a in list(args.args) + list(args.kwonlyargs)}
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in sub.generators:
                bound |= _target_names(generator.target)[0]
        elif isinstance(sub, ast.Name) and not isinstance(sub.ctx, (ast.Store, ast.Del)):
            out.add(sub.id)
    return out - bound


def mutation_targets(stmt: ast.AST) -> Set[str]:
    """Names a statement mutates through a subscript or attribute store.

    a[i] = v writes into an object a already refers to, it cannot create a.
    Reaching defs record it as a weak def so dead store / loop update facts
    see the write, but the uninitialised use check must not count it as a
    definition, otherwise on a loop back edge it would "define" a name the loop
    reads before anything bound it.
    """
    targets: List[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets = [stmt.target]
    out: Set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Subscript):
                base = node.value
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name):
                    out.add(base.id)
    strong, _ = _strong_targets(targets)
    return out - strong


def _strong_targets(targets: Iterable[ast.AST]) -> Tuple[Set[str], Set[str]]:
    """Names bound directly (not through a subscript) by assignment targets."""
    defined: Set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
    return defined, set()


def statement_defs_uses(stmt: ast.AST) -> Tuple[Set[str], Set[str]]:
    """(defs, uses) for one statement or CFG pseudo statement."""
    defs: Set[str] = set()
    uses: Set[str] = set()

    if isinstance(stmt, ast.Assign):
        uses |= expression_uses(stmt.value)
        for target in stmt.targets:
            d, u = _target_names(target)
            defs |= d
            uses |= u
    elif isinstance(stmt, ast.AugAssign):
        uses |= expression_uses(stmt.value)
        d, u = _target_names(stmt.target)
        defs |= d
        uses |= d | u
    elif isinstance(stmt, ast.AnnAssign):
        uses |= expression_uses(stmt.value)
        d, u = _target_names(stmt.target)
        defs |= d
        uses |= u
    elif isinstance(stmt, (ast.Return, ast.Expr)):
        uses |= expression_uses(getattr(stmt, "value", None))
    elif isinstance(stmt, ast.Raise):
        uses |= expression_uses(getattr(stmt, "exc", None))
    elif isinstance(stmt, (ast.Delete,)):
        for target in stmt.targets:
            d, u = _target_names(target)
            defs |= d
            uses |= u
    elif hasattr(stmt, "test"):                     # _Condition pseudo statement
        uses |= expression_uses(getattr(stmt, "test"))
    elif hasattr(stmt, "iter"):                     # _Iteration pseudo statement
        uses |= expression_uses(getattr(stmt, "iter"))
        d, u = _target_names(getattr(stmt, "target"))
        defs |= d
        uses |= u
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs.add(stmt.name)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            uses |= expression_uses(item.context_expr)
            if item.optional_vars is not None:
                d, u = _target_names(item.optional_vars)
                defs |= d
                uses |= u
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for alias in stmt.names:
            defs.add(alias.asname or alias.name.split(".")[0])
    return defs, uses


# analyses
@dataclass
class DataflowFacts:
    """Solved dataflow state for one function."""

    reaching_in: Dict[int, Set[Definition]] = field(default_factory=dict)
    reaching_out: Dict[int, Set[Definition]] = field(default_factory=dict)
    live_in: Dict[int, Set[str]] = field(default_factory=dict)
    live_out: Dict[int, Set[str]] = field(default_factory=dict)
    def_use: Dict[Definition, Set[int]] = field(default_factory=dict)
    uninitialised: Set[Tuple[str, int]] = field(default_factory=set)
    dead_stores: Set[Definition] = field(default_factory=set)
    loop_carried: Dict[int, Set[str]] = field(default_factory=dict)
    # variables written inside a loop body, by loop header block
    loop_updated: Dict[int, Set[str]] = field(default_factory=dict)
    # variables read by a loop guard, by loop header block
    loop_guard_vars: Dict[int, Set[str]] = field(default_factory=dict)

    def stale_guard_variables(self) -> Dict[int, Set[str]]:
        """Guard variables never updated in the loop body.
        Some being stale is normal (while i < n never writes n), what matters is
        when none get updated.
        """
        out: Dict[int, Set[str]] = {}
        for header, guard_vars in self.loop_guard_vars.items():
            stale = guard_vars - self.loop_updated.get(header, set())
            if stale:
                out[header] = stale
        return out

    def non_terminating_loops(self) -> Dict[int, Set[str]]:
        """Loop headers whose guard reads no variable the body writes.
        Such a loop can only exit via break / return, and if the CFG has none of
        those it really loops forever (classic missing 'lo = mid + 1').
        """
        out: Dict[int, Set[str]] = {}
        for header, guard_vars in self.loop_guard_vars.items():
            if not guard_vars:
                continue
            if not (guard_vars & self.loop_updated.get(header, set())):
                out[header] = set(guard_vars)
        return out


def analyse(cfg: ControlFlowGraph, parameters: Sequence[str] = ()) -> DataflowFacts:
    """Solve reaching definitions and liveness for cfg."""
    facts = DataflowFacts()
    reach = cfg.reachable()
    order = [b for b in cfg.rpo() if b in reach]

    # per block gen/kill for reaching defs
    gen: Dict[int, Set[Definition]] = {}
    kill_vars: Dict[int, Set[str]] = {}
    block_uses: Dict[int, List[Tuple[str, int]]] = {}
    weak: Set[Definition] = set()
    for bid in reach:
        block = cfg.blocks[bid]
        g: Set[Definition] = set()
        killed: Set[str] = set()
        uses_seq: List[Tuple[str, int]] = []
        for index, stmt in enumerate(block.statements):
            defs, uses = statement_defs_uses(stmt)
            mutated = mutation_targets(stmt)
            line = getattr(stmt, "lineno", 0)
            for name in sorted(uses):
                uses_seq.append((name, line))
            for name in sorted(defs):
                if name in mutated:
                    # a mutation does not rebind the name and does not kill the def that
                    # bound it, it is recorded next to it
                    g.add((name, index, line))
                    weak.add((name, index, line))
                    continue
                g = {d for d in g if d[0] != name}
                g.add((name, index, line))
                killed.add(name)
        gen[bid] = g
        kill_vars[bid] = killed
        block_uses[bid] = uses_seq

    entry_defs: Set[Definition] = {(p, -1, 0) for p in parameters}

    # forward may analysis: reaching definitions
    rin: Dict[int, Set[Definition]] = {b: set() for b in reach}
    rout: Dict[int, Set[Definition]] = {b: set(gen[b]) for b in reach}
    rin[cfg.entry] = set(entry_defs)
    worklist = list(order)
    in_worklist = set(worklist)
    while worklist:
        bid = worklist.pop(0)
        in_worklist.discard(bid)
        incoming: Set[Definition] = set(entry_defs) if bid == cfg.entry else set()
        for pred in cfg.blocks[bid].predecessors:
            if pred in reach:
                incoming |= rout[pred]
        rin[bid] = incoming
        survivors = {d for d in incoming if d[0] not in kill_vars[bid]}
        new_out = survivors | gen[bid]
        if new_out != rout[bid]:
            rout[bid] = new_out
            for succ in cfg.blocks[bid].successors:
                if succ in reach and succ not in in_worklist:
                    worklist.append(succ)
                    in_worklist.add(succ)
    facts.reaching_in, facts.reaching_out = rin, rout

    # backward may analysis: live variables
    use_b: Dict[int, Set[str]] = {}
    def_b: Dict[int, Set[str]] = {}
    for bid in reach:
        upward: Set[str] = set()
        written: Set[str] = set()
        for stmt in cfg.blocks[bid].statements:
            defs, uses = statement_defs_uses(stmt)
            upward |= uses - written
            written |= defs
        use_b[bid] = upward
        def_b[bid] = written

    lin: Dict[int, Set[str]] = {b: set() for b in reach}
    lout: Dict[int, Set[str]] = {b: set() for b in reach}
    worklist = list(reversed(order))
    in_worklist = set(worklist)
    while worklist:
        bid = worklist.pop(0)
        in_worklist.discard(bid)
        outgoing: Set[str] = set()
        for succ in cfg.blocks[bid].successors:
            if succ in reach:
                outgoing |= lin[succ]
        lout[bid] = outgoing
        new_in = use_b[bid] | (outgoing - def_b[bid])
        if new_in != lin[bid]:
            lin[bid] = new_in
            for pred in cfg.blocks[bid].predecessors:
                if pred in reach and pred not in in_worklist:
                    worklist.append(pred)
                    in_worklist.add(pred)
    facts.live_in, facts.live_out = lin, lout

    # derived facts
    for bid in reach:
        # only a strong def can make a name available, a subscript store on a back
        # edge says nothing about the first iteration
        available = {d[0] for d in rin[bid] if d not in weak}
        block = cfg.blocks[bid]
        for index, stmt in enumerate(block.statements):
            defs, uses = statement_defs_uses(stmt)
            line = getattr(stmt, "lineno", 0)
            for name in sorted(uses):
                if name not in available and name not in _BUILTIN_NAMES:
                    facts.uninitialised.add((name, line))
            available |= defs - mutation_targets(stmt)

    for definition in {d for b in reach for d in gen[b]}:
        name = definition[0]
        consumers = {
            bid
            for bid in reach
            if name in use_b[bid] and definition in rin[bid]
        }
        # a def is also used if a later statement in its own block reads it
        facts.def_use[definition] = consumers
    for bid in reach:
        for definition in gen[bid]:
            name = definition[0]
            if name in lout[bid]:
                continue
            index = definition[1]
            later_use = any(
                name in statement_defs_uses(s)[1]
                for s in cfg.blocks[bid].statements[index + 1 :]
            )
            if not later_use:
                facts.dead_stores.add(definition)

    for header, body in cfg.natural_loops():
        if cfg.blocks[header].loop_kind != "while":
            continue                 # for over a finite iterable always ends
        guard_vars: Set[str] = set()
        for stmt in cfg.blocks[header].statements:
            guard_vars |= statement_defs_uses(stmt)[1]
        updated: Set[str] = set()
        for bid in body:
            if bid == header:
                continue
            updated |= def_b.get(bid, set())
            for stmt in cfg.blocks[bid].statements:
                updated |= mutated_names(stmt)
        facts.loop_guard_vars.setdefault(header, set()).update(guard_vars)
        facts.loop_updated.setdefault(header, set()).update(updated)
        carried = {
            name
            for name in use_b.get(header, set()) | guard_vars
            if name in updated
        }
        facts.loop_carried.setdefault(header, set()).update(carried)

    return facts


# methods that modify the receiver in place. 'while queue:' makes progress
# through queue.popleft() which is a call, not an assignment. Missing that
# makes every correct worklist loop look non terminating.
MUTATING_METHODS = frozenset(
    """append appendleft extend insert pop popleft popitem remove clear add discard
    update sort reverse setdefault""".split()
)


def mutated_names(stmt: ast.AST) -> Set[str]:
    """Receivers of in place mutating method calls in a statement."""
    out: Set[str] = set()
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in MUTATING_METHODS:
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Name):
            out.add(receiver.id)
    return out


_BUILTIN_NAMES = frozenset(
    """abs all any bin bool dict divmod enumerate filter float format frozenset getattr hash int
    isinstance iter len list map max min next object pow print range repr reversed round set
    sorted str sum tuple type zip math collections heapq bisect itertools functools deque
    defaultdict Counter float inf True False None""".split()
)
