"""Syntactic normalisation.

Before any structural comparison the program is projected onto a naming
invariant normal form: alpha rename in first occurrence order, drop
docstrings, abstract constants. Two programs that only differ in identifier
names then have the same normal form and the same structural hash.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

_PROTECTED = frozenset(
    """print len range enumerate sorted sum min max abs int float str list dict set tuple
    reversed zip map filter any all bool divmod pow round enumerate isinstance next iter
    append pop popleft appendleft add insert extend remove sort keys values items get
    deque defaultdict Counter heappush heappop bisect_left bisect_right math inf
    self None True False""".split()
)


class _AlphaRenamer(ast.NodeTransformer):
    """Rename user identifiers to v0, v1, ... in order of first appearance."""

    def __init__(self):
        self.mapping: Dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name in _PROTECTED or name.startswith("__"):
            return name
        if name not in self.mapping:
            self.mapping[name] = "v%d" % len(self.mapping)
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        new = ast.arg(arg=self._rename(node.arg), annotation=None)
        return ast.copy_location(new, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._rename(node.name)
        node.returns = None
        node.decorator_list = []
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # attribute names are library api (.popleft) so they stay, only the
        # receiver gets renamed
        node.value = self.visit(node.value)
        return node


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
                and len(body) > 1
            ):
                node.body = body[1:]
        self.generic_visit(node)
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


@dataclass(frozen=True)
class NormalForm:
    """Normal form of a submission."""

    source: str
    identifier_map: Dict[str, str]
    structural_hash: str
    shape_hash: str

    def __str__(self) -> str:
        return self.structural_hash[:12]


def _shape(node: ast.AST) -> str:
    """Bracketed serialisation of node types only (constants abstracted)."""
    parts: List[str] = [type(node).__name__]
    for field, value in ast.iter_fields(node):
        if field in {"ctx", "type_comment", "kind"}:
            continue
        if isinstance(value, list):
            inner = "".join(_shape(v) for v in value if isinstance(v, ast.AST))
            parts.append("[" + inner + "]")
        elif isinstance(value, ast.AST):
            parts.append("(" + _shape(value) + ")")
    return "".join(parts)


def normalise(source: str) -> Optional[NormalForm]:
    """Normal form, or None if the source does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    tree = _DocstringStripper().visit(tree)
    renamer = _AlphaRenamer()
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    try:
        canonical = ast.unparse(tree)
    except Exception:
        canonical = ast.dump(tree)
    shape = _shape(tree)
    return NormalForm(
        source=canonical,
        identifier_map=dict(renamer.mapping),
        structural_hash=hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest(),
        shape_hash=hashlib.blake2b(shape.encode("utf-8"), digest_size=16).hexdigest(),
    )


def structurally_equivalent(a: str, b: str) -> bool:
    """True if two programs only differ in naming, spacing and docstrings."""
    na, nb = normalise(a), normalise(b)
    return bool(na and nb and na.structural_hash == nb.structural_hash)


def shape_equivalent(a: str, b: str) -> bool:
    """True if two programs have the same AST shape (constants ignored)."""
    na, nb = normalise(a), normalise(b)
    return bool(na and nb and na.shape_hash == nb.shape_hash)
