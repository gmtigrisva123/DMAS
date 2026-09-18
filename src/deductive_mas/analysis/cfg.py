"""Control flow graph + dominators.

One CFG per function, from it we get: reachability (dead code after an early
return), dominators (Cooper-Harvey-Kennedy style iteration), back edges and
their natural loops (for loop carried variables), and cyclomatic complexity.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ENTRY = 0


@dataclass
class BasicBlock:
    """A straight line run of statements."""

    bid: int
    label: str = ""
    statements: List[ast.AST] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    # set for loop headers so dataflow can special case them
    is_loop_header: bool = False
    # "while" or "for" for loop headers, empty otherwise. Termination reasoning
    # only applies to while, a for over a finite iterable always ends.
    loop_kind: str = ""

    @property
    def lines(self) -> Tuple[int, int]:
        nums = [getattr(s, "lineno", 0) for s in self.statements if getattr(s, "lineno", 0)]
        return (min(nums), max(nums)) if nums else (0, 0)

    def __str__(self) -> str:
        lo, hi = self.lines
        span = f"L{lo}" if lo == hi else f"L{lo}-{hi}"
        return f"B{self.bid}[{self.label or 'seq'}]{span if lo else ''}"


@dataclass
class ControlFlowGraph:
    """Blocks plus the derived facts for one function."""

    name: str
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry: int = ENTRY
    exit: int = -1

    # topology
    def successors(self, bid: int) -> List[int]:
        return self.blocks[bid].successors

    def reachable(self) -> Set[int]:
        seen: Set[int] = set()
        stack = [self.entry]
        while stack:
            bid = stack.pop()
            if bid in seen or bid not in self.blocks:
                continue
            seen.add(bid)
            stack.extend(self.blocks[bid].successors)
        return seen

    def unreachable_lines(self) -> List[int]:
        dead = set(self.blocks) - self.reachable()
        out: List[int] = []
        for bid in sorted(dead):
            lo, hi = self.blocks[bid].lines
            if lo:
                out.extend(range(lo, hi + 1))
        return sorted(set(out))

    def rpo(self) -> List[int]:
        """Reverse post order over reachable blocks (visit order for dataflow)."""
        order: List[int] = []
        seen: Set[int] = set()

        def visit(bid: int):
            stack = [(bid, iter(self.blocks[bid].successors))]
            seen.add(bid)
            while stack:
                node, it = stack[-1]
                advanced = False
                for succ in it:
                    if succ in self.blocks and succ not in seen:
                        seen.add(succ)
                        stack.append((succ, iter(self.blocks[succ].successors)))
                        advanced = True
                        break
                if not advanced:
                    order.append(node)
                    stack.pop()

        visit(self.entry)
        return list(reversed(order))

    # dominators
    def dominators(self) -> Dict[int, Set[int]]:
        """Iterative dominator fixpoint on the reachable subgraph.
        dom(entry) = {entry}; dom(n) = {n} u intersection of dom(p) over preds p.
        """
        reach = self.reachable()
        order = [b for b in self.rpo() if b in reach]
        dom: Dict[int, Set[int]] = {b: set(reach) for b in reach}
        dom[self.entry] = {self.entry}
        changed = True
        while changed:
            changed = False
            for bid in order:
                if bid == self.entry:
                    continue
                preds = [p for p in self.blocks[bid].predecessors if p in reach]
                if not preds:
                    new = {bid}
                else:
                    new = set(dom[preds[0]])
                    for p in preds[1:]:
                        new &= dom[p]
                    new.add(bid)
                if new != dom[bid]:
                    dom[bid] = new
                    changed = True
        return dom

    def back_edges(self) -> List[Tuple[int, int]]:
        """Edges u -> v where v dominates u, i.e. loop back edges."""
        dom = self.dominators()
        edges: List[Tuple[int, int]] = []
        for bid in sorted(dom):
            for succ in self.blocks[bid].successors:
                if succ in dom and succ in dom[bid]:
                    edges.append((bid, succ))
        return edges

    def natural_loops(self) -> List[Tuple[int, Set[int]]]:
        """For each back edge the natural loop it induces (header, body set)."""
        loops: List[Tuple[int, Set[int]]] = []
        for tail, header in self.back_edges():
            body = {header, tail}
            stack = [tail]
            while stack:
                node = stack.pop()
                for pred in self.blocks[node].predecessors:
                    if pred not in body:
                        body.add(pred)
                        stack.append(pred)
            loops.append((header, body))
        return loops

    @property
    def cyclomatic_complexity(self) -> int:
        reach = self.reachable()
        edges = sum(1 for b in reach for s in self.blocks[b].successors if s in reach)
        return edges - len(reach) + 2

    def to_dot(self) -> str:
        lines = [f'digraph "{self.name}" {{', "  node [shape=box, fontname=Menlo];"]
        for bid, block in sorted(self.blocks.items()):
            lines.append(f'  {bid} [label="{block}"];')
            for succ in block.successors:
                lines.append(f"  {bid} -> {succ};")
        lines.append("}")
        return "\n".join(lines)


class _Builder:
    """Recursive descent CFG builder with a loop stack for break / continue."""

    def __init__(self, name: str):
        self.cfg = ControlFlowGraph(name=name)
        self._next = 0
        self.entry = self._new("entry")
        self.cfg.entry = self.entry
        self.exit = self._new("exit")
        self.cfg.exit = self.exit
        self._loops: List[Tuple[int, int]] = []  # (continue target, break target)

    # plumbing
    def _new(self, label: str = "") -> int:
        bid = self._next
        self._next += 1
        self.cfg.blocks[bid] = BasicBlock(bid=bid, label=label)
        return bid

    def _edge(self, src: Optional[int], dst: int):
        if src is None:
            return
        block = self.cfg.blocks[src]
        if dst not in block.successors:
            block.successors.append(dst)
            self.cfg.blocks[dst].predecessors.append(src)

    def _emit(self, bid: int, stmt: ast.AST):
        self.cfg.blocks[bid].statements.append(stmt)

    # traversal
    def build(self, body: Sequence[ast.stmt]) -> ControlFlowGraph:
        tail = self._sequence(body, self.entry)
        self._edge(tail, self.exit)
        return self.cfg

    def _sequence(self, body: Sequence[ast.stmt], current: Optional[int]) -> Optional[int]:
        for stmt in body:
            if current is None:
                current = self._new("unreachable")
            current = self._statement(stmt, current)
        return current

    def _statement(self, stmt: ast.stmt, current: int) -> Optional[int]:
        if isinstance(stmt, ast.Return):
            self._emit(current, stmt)
            self._edge(current, self.exit)
            return None
        if isinstance(stmt, ast.Break):
            self._emit(current, stmt)
            if self._loops:
                self._edge(current, self._loops[-1][1])
            return None
        if isinstance(stmt, ast.Continue):
            self._emit(current, stmt)
            if self._loops:
                self._edge(current, self._loops[-1][0])
            return None
        if isinstance(stmt, ast.If):
            return self._if(stmt, current)
        if isinstance(stmt, (ast.While, ast.For, ast.AsyncFor)):
            return self._loop(stmt, current)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            self._emit(current, stmt)
            return self._sequence(stmt.body, current)
        if isinstance(stmt, ast.Try):
            return self._try(stmt, current)
        if isinstance(stmt, ast.Raise):
            self._emit(current, stmt)
            self._edge(current, self.exit)
            return None
        self._emit(current, stmt)
        return current

    def _if(self, stmt: ast.If, current: int) -> Optional[int]:
        self._emit(current, _Condition(stmt.test, stmt.lineno))
        then_entry = self._new("then")
        self._edge(current, then_entry)
        then_tail = self._sequence(stmt.body, then_entry)

        if stmt.orelse:
            else_entry = self._new("else")
            self._edge(current, else_entry)
            else_tail = self._sequence(stmt.orelse, else_entry)
        else:
            else_tail = current

        if then_tail is None and else_tail is None:
            return None
        join = self._new("join")
        self._edge(then_tail, join)
        if else_tail is not current or not stmt.orelse:
            self._edge(else_tail, join)
        else:
            self._edge(else_tail, join)
        return join

    def _loop(self, stmt: ast.stmt, current: int) -> Optional[int]:
        header = self._new("loop")
        self.cfg.blocks[header].is_loop_header = True
        self.cfg.blocks[header].loop_kind = "while" if isinstance(stmt, ast.While) else "for"
        self._edge(current, header)
        test = getattr(stmt, "test", None)
        self._emit(header, _Condition(test, stmt.lineno) if test is not None else _Iteration(stmt))

        exit_block = self._new("loop-exit")
        body_entry = self._new("loop-body")
        self._edge(header, body_entry)
        self._edge(header, exit_block)

        self._loops.append((header, exit_block))
        body_tail = self._sequence(stmt.body, body_entry)
        self._edge(body_tail, header)
        self._loops.pop()

        orelse = getattr(stmt, "orelse", None)
        if orelse:
            else_tail = self._sequence(orelse, exit_block)
            return else_tail
        return exit_block

    def _try(self, stmt: ast.Try, current: int) -> Optional[int]:
        body_tail = self._sequence(stmt.body, current)
        tails: List[Optional[int]] = [body_tail]
        for handler in stmt.handlers:
            handler_entry = self._new("except")
            self._edge(current, handler_entry)
            tails.append(self._sequence(handler.body, handler_entry))
        if stmt.orelse:
            tails.append(self._sequence(stmt.orelse, body_tail) if body_tail is not None else None)
        alive = [t for t in tails if t is not None]
        if not alive:
            return None
        join = self._new("try-join")
        for tail in alive:
            self._edge(tail, join)
        if stmt.finalbody:
            return self._sequence(stmt.finalbody, join)
        return join


class _Condition(ast.AST):
    """Pseudo statement carrying a branch condition into a block."""

    _fields = ("test",)

    def __init__(self, test: Optional[ast.expr], lineno: int):
        self.test = test
        self.lineno = lineno
        self.end_lineno = lineno


class _Iteration(ast.AST):
    """Pseudo statement carrying a for loop's target / iterable."""

    _fields = ("target", "iter")

    def __init__(self, stmt: ast.stmt):
        self.target = getattr(stmt, "target", None)
        self.iter = getattr(stmt, "iter", None)
        self.lineno = stmt.lineno
        self.end_lineno = stmt.lineno


def synchronisation_lines(func: ast.FunctionDef, cfg: ControlFlowGraph) -> Set[int]:
    """Lines where two implementations of the same algorithm can be compared.

    Comparing every executed line is brittle: lo, hi = 0, n on one line vs two
    lines gives different intermediate states for the same algorithm. What
    really corresponds are the loop checkpoints (each evaluation of a loop
    guard) plus each entry into a call. Sampling there compares the runs
    iteration by iteration.
    """
    lines: Set[int] = set()
    for block in cfg.blocks.values():
        if block.is_loop_header:
            lo, _ = block.lines
            if lo:
                lines.add(lo)
    body = [s for s in func.body if not _is_docstring(s)]
    if body:
        lines.add(body[0].lineno)
    for node in ast.walk(func):
        if isinstance(node, ast.Return):
            lines.add(node.lineno)
    return lines


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def build_cfg(func: ast.FunctionDef) -> ControlFlowGraph:
    """Build the CFG of one function definition."""
    builder = _Builder(func.name)
    return builder.build(func.body)


def build_all(tree: ast.Module) -> Dict[str, ControlFlowGraph]:
    """One CFG per function in the module (nested ones included)."""
    out: Dict[str, ControlFlowGraph] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = build_cfg(node)
    return out
