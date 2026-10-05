"""Answer leakage detection.

Hints must never hand over the solution and that has to be enforced by code,
not by asking the model nicely. Every generated text goes through a gate
built on document fingerprinting.

Core is winnowing (Schleimer, Wilkerson & Aiken 2003, the MOSS algorithm):
hash every k-gram of the normalised token stream, slide a window of w hashes
over them and keep the min of each window (ties to the right, never reselect
the same hash). Any shared substring of >= w + k - 1 tokens is detected.

Three tweaks for the tutoring setting:

- identifier normalisation: "set the upper bound to mid" leaks as much as
  "hi = mid", so tokens are alpha normalised first
- novelty discount: text that already appears in the student's own code
  reveals nothing (quoting the student is the whole point of Socratic
  hints), so the score is containment(hint, ref) * (1 - containment(hint, student))
- edit proximity: the most common leak is a one token repair ("change it
  to hi = len(a)"), winnowing cannot see that, so a third channel flags any
  code span that is a near miss of a line the student already wrote.
  Quoting the student verbatim is fine, handing them an edited line is not.
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..util.text import code_tokens, ngrams, normalised_levenshtein, prose_tokens
from .normalize import normalise

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]{2,})`")
_CODEY_LINE_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_\[\]\.]*\s*(?:=|\+=|-=|//=|\*=)[^=]")
# an inline span that is a statement made of keywords instead of operators
# (if x not in seen:) is code too
_KEYWORD_SPAN_RE = re.compile(
    r"^(?:if|elif|else|while|for|return|def|not|yield|del|pass|break|continue)\b"
)

# repairs written in prose. "Set hi to len(a)" says as much as hi = len(a)
# and a model told not to write code will do exactly this. Both get
# normalised to an assignment so all three channels see them.
# Where the assigned value ends: punctuation, or a conjunction that starts
# a new clause ("set hi to len(a) and it will work")
_CLAUSE_END = r"(?=\s*(?:[.,;:`\n]|\b(?:and|or|so|then|which|because|instead|rather|but)\b)|$)"
_TARGET = r"`?([A-Za-z_][\w\[\]\.]*)`?"
_CONNECTOR = r"(?:equal(?:s)?\s+to|equals|to\s+be|to|=|as)"

_PROSE_ASSIGNMENT_RES = (
    re.compile(
        r"\b(?:set|change|make|initialis[ez]|update|assign)\s+" + _TARGET
        + r"\s+" + _CONNECTOR + r"\s+`?(.+?)`?" + _CLAUSE_END,
        re.IGNORECASE,
    ),
    re.compile(
        r"\breplace\s+" + _TARGET + r"\s+with\s+`?(.+?)`?" + _CLAUSE_END,
        re.IGNORECASE,
    ),
)

# a code span at most this far from a student statement is a concrete edit,
# not a question
EDIT_PROXIMITY_THRESHOLD = 0.45

# operators that make a span code like enough for the edit channel
_OPERATORISH = frozenset("= + - * / < > [ ] %".split())


@dataclass(frozen=True)
class LeakageReport:
    """Verdict of the leakage gate."""

    score: float
    spans: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    reference_containment: float = 0.0
    student_containment: float = 0.0
    structural_match: bool = False
    channel: str = "none"

    def rejected(self, threshold: float) -> bool:
        return self.score > threshold or self.structural_match

    def __str__(self) -> str:
        return f"leakage={self.score:.2f}({self.channel})"


# winnowing
def _hash(gram: Sequence[str]) -> int:
    digest = hashlib.blake2b("\x1f".join(gram).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def kgram_hashes(tokens: Sequence[str], k: int) -> List[int]:
    """Hashes of every k-gram (shorter input gives one hash)."""
    if not tokens:
        return []
    if len(tokens) < k:
        return [_hash(tokens)]
    return [_hash(gram) for gram in ngrams(list(tokens), k)]


def winnow(hashes: Sequence[int], window: int) -> Set[int]:
    """Winnowing: sparse, position independent fingerprint of hashes.
    In every window take the min, ties to the rightmost, don't reselect the
    current selection. Density is 2 / (window + 1) and the detection guarantee
    still holds.
    """
    if not hashes:
        return set()
    if window <= 1 or len(hashes) <= window:
        return set(hashes)
    selected: Set[int] = set()
    previous_index = -1
    for start in range(len(hashes) - window + 1):
        chunk = hashes[start : start + window]
        best = min(chunk)
        offset = len(chunk) - 1 - chunk[::-1].index(best)   # rightmost min
        index = start + offset
        if index != previous_index:
            selected.add(best)
            previous_index = index
    return selected


def fingerprint(source: str, *, k: int = 5, window: int = 4, code: bool = True) -> Set[int]:
    """Winnowed fingerprint of a code snippet (or prose when code=False)."""
    tokens = code_tokens(source, normalise_identifiers=True) if code else prose_tokens(source)
    return winnow(kgram_hashes(tokens, k), window)


def full_fingerprint(source: str, *, k: int = 5, code: bool = True) -> Set[int]:
    """Every k-gram hash of a snippet, the exhaustive index.

    Winnowing is meant for a corpus of thousands of files. Here we have one
    student program and one reference, a few hundred tokens, and the queries
    are single lines. On a five token query a winnowed index is noise (whether
    the query's hashes got selected depends on the text around them), so an
    exact quote of the reference can score 0. Indexing every k-gram costs
    nothing at this size and makes containment exact.
    """
    tokens = (
        code_tokens(source, normalise_identifiers=True, normalise_numbers=False)
        if code else prose_tokens(source)
    )
    return set(kgram_hashes(tokens, k))


def containment(query: Set[int], corpus: Set[int]) -> float:
    """Asymmetric overlap |Q n C| / |Q|, the usual plagiarism measure."""
    if not query:
        return 0.0
    return len(query & corpus) / float(len(query))


# getting code out of generated prose
def extract_code_spans(text: str) -> List[str]:
    """Pull fenced blocks, inline spans and assignment shaped lines out of prose."""
    spans: List[str] = []
    remainder = text
    for match in _FENCE_RE.finditer(text):
        spans.append(match.group(1).strip())
        remainder = remainder.replace(match.group(0), " ")
    for match in _INLINE_RE.finditer(remainder):
        candidate = match.group(1).strip()
        if (
            any(ch in candidate for ch in "=+-*/[]()<>")
            or " " not in candidate
            or _KEYWORD_SPAN_RE.match(candidate)
        ):
            spans.append(candidate)
    for line in remainder.splitlines():
        if _CODEY_LINE_RE.match(line):
            spans.append(line.strip())
    for pattern in _PROSE_ASSIGNMENT_RES:
        for match in pattern.finditer(remainder):
            target, value = match.group(1).strip(), match.group(2).strip()
            if target and value:
                spans.append(f"{target} = {value}")
    return [s for s in spans if s]


def _parses_as_code(snippet: str) -> Optional[ast.AST]:
    indented = "\n".join("    " + line for line in snippet.splitlines())
    for candidate in (snippet, "def _wrapper():\n" + indented):
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    return None


_STATEMENT_TYPES = (ast.Assign, ast.AugAssign, ast.If, ast.While, ast.For, ast.Return)


def _decompose(node: ast.AST) -> List[ast.AST]:
    """Yield a statement plus its element wise pieces.
    lo, hi = 0, len(a) and lo = 0 / hi = len(a) are the same program, so
    tuple assignments are split. Otherwise a hint could get past the gate just
    by packing an assignment differently.
    """
    out: List[ast.AST] = [node]
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                for element, item in zip(target.elts, value.elts):
                    piece = ast.Assign(targets=[element], value=item)
                    piece.lineno = getattr(node, "lineno", 1)
                    piece.col_offset = 0
                    out.append(piece)
    return out


def _shape_hashes(source: str) -> Set[str]:
    """Shape hashes of every statement level subtree."""
    shapes: Set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return shapes
    for node in ast.walk(tree):
        if not isinstance(node, _STATEMENT_TYPES):
            continue
        for piece in _decompose(node):
            shape = _shape_of(piece)
            if shape:
                shapes.add(shape)
    return shapes


def _shape_of(node: ast.AST) -> Optional[str]:
    """Naming invariant identity of one statement, constants included.
    Abstracting constants would make key=lambda p: p[1] equal to
    key=lambda p: p[0], and a one constant change is often the whole fix.
    """
    try:
        rendered = ast.unparse(node)
    except Exception:
        return None
    form = normalise(rendered) or normalise("def _w():\n    " + rendered)
    return form.structural_hash if form else None


def _normalised_statements(source: str) -> Tuple[str, ...]:
    """Every statement of a program, alpha normalised into a token string."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    out: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _STATEMENT_TYPES):
            continue
        for piece in _decompose(node):
            target = piece.test if isinstance(piece, (ast.If, ast.While)) else piece
            try:
                rendered = ast.unparse(target)
            except Exception:
                continue
            out.append(" ".join(code_tokens(rendered, normalise_identifiers=True, normalise_numbers=False)))
    return tuple(dict.fromkeys(s for s in out if s))


# the gate
class LeakageDetector:
    """Scores how much of the reference a piece of text reveals."""

    # k-gram ladder. Short hints have too few 5-grams to match anything, so
    # each corpus is fingerprinted at a fine and a coarse k and every query is
    # compared at its own k.
    LADDER = (3, 5)

    def __init__(self, *, kgram: int = 5, window: int = 4):
        self.kgram = kgram
        self.window = window

    def resolution_for(self, tokens: Sequence[str]) -> int:
        for k in self.LADDER:
            if len(tokens) < k + 4:
                return k
        return min(self.kgram, self.LADDER[-1])

    def build(self, reference_solution: str, student_source: str) -> "LeakageContext":
        return LeakageContext(
            detector=self,
            reference={k: full_fingerprint(reference_solution, k=k) for k in self.LADDER},
            student={k: full_fingerprint(student_source, k=k) for k in self.LADDER},
            reference_shapes=_shape_hashes(reference_solution),
            student_shapes=_shape_hashes(student_source),
            student_statements=_normalised_statements(student_source),
            reference_statements=_normalised_statements(reference_solution),
        )


@dataclass(frozen=True)
class EditProximity:
    """How far a code span is from the student's code and from the reference."""

    to_student: float
    to_reference: float

    def is_repair_instruction(self) -> bool:
        return (
            0.0 < self.to_student <= EDIT_PROXIMITY_THRESHOLD
            and self.to_reference < self.to_student
        )


@dataclass
class LeakageContext:
    """Precomputed fingerprints for one (problem, submission) pair."""

    detector: LeakageDetector
    reference: Dict[int, Set[int]] = field(default_factory=dict)
    student: Dict[int, Set[int]] = field(default_factory=dict)
    reference_shapes: Set[str] = field(default_factory=set)
    student_shapes: Set[str] = field(default_factory=set)
    student_statements: Tuple[str, ...] = ()
    reference_statements: Tuple[str, ...] = ()

    def score(self, guidance: str) -> LeakageReport:
        """Score one hint / challenge / narrative for leakage.
        The three channels catch different ways of giving it away, max wins.
        """
        reasons: List[str] = []
        offending: List[str] = []
        worst = 0.0
        channel = "none"
        worst_ref = 0.0
        worst_student = 0.0
        structural = False

        for span in extract_code_spans(guidance):
            tokens = code_tokens(span, normalise_identifiers=True, normalise_numbers=False)
            if len(tokens) < 3:
                continue

            # channel 1: k-gram fingerprint containment
            k = self.detector.resolution_for(tokens)
            marks = set(kgram_hashes(tokens, k))
            if marks:
                ref_hit = containment(marks, self.reference.get(k, set()))
                student_hit = containment(marks, self.student.get(k, set()))
                value = ref_hit * (1.0 - student_hit)
                if value > worst:
                    worst, channel = value, "fingerprint"
                    worst_ref, worst_student = ref_hit, student_hit
                if value > 0.25:
                    offending.append(span)
                    reasons.append(
                        "snippet {!r} matches {:.0%} of the reference solution and only "
                        "{:.0%} of the student's own code".format(span, ref_hit, student_hit)
                    )

            # channel 2: structural reproduction
            if self._structurally_reveals(span):
                structural = True
                if worst < 1.0:
                    worst, channel = 1.0, "structural"
                offending.append(span)
                reasons.append(
                    "snippet {!r} reproduces a statement of the reference solution that the "
                    "student has not written".format(span)
                )
                continue

            # channel 3: concrete edit of a student line
            proximity = self._edit_proximity(span, tokens)
            if proximity is not None and proximity.is_repair_instruction():
                distance = proximity.to_student
                closeness = 1.0 - distance / EDIT_PROXIMITY_THRESHOLD
                value = 0.6 + 0.4 * closeness
                if value > worst:
                    worst, channel = value, "edit"
                offending.append(span)
                reasons.append(
                    "snippet {!r} is a modified version of a line the student wrote, and the "
                    "modification moves it closer to the reference solution "
                    "(d_student={:.2f} > d_reference={:.2f}): that is a repair instruction, "
                    "not a question".format(span, proximity.to_student, proximity.to_reference)
                )

        return LeakageReport(
            score=min(1.0, worst),
            spans=tuple(dict.fromkeys(offending)),
            reasons=tuple(dict.fromkeys(reasons)),
            reference_containment=worst_ref,
            student_containment=worst_student,
            structural_match=structural,
            channel=channel,
        )

    # channels
    def _structurally_reveals(self, span: str) -> bool:
        tree = _parses_as_code(span)
        if tree is None:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, _STATEMENT_TYPES):
                continue
            for piece in _decompose(node):
                shape = _shape_of(piece)
                if shape and shape in self.reference_shapes and shape not in self.student_shapes:
                    return True
        return False

    def _edit_proximity(self, span: str, tokens: Sequence[str]) -> Optional["EditProximity"]:
        """Distance from a code span to the nearest student and reference line.

        Quoting the student's own line is fine, handing them an edited line is
        not. The two are told apart by direction: a span closer to the reference
        than to the student's code is a repair instruction.
        """
        if not self.student_statements:
            return None
        if not any(token in _OPERATORISH for token in tokens):
            return None
        query = " ".join(tokens)
        to_student = min(
            (normalised_levenshtein(query, stmt) for stmt in self.student_statements),
            default=1.0,
        )
        to_reference = min(
            (normalised_levenshtein(query, stmt) for stmt in self.reference_statements),
            default=1.0,
        )
        return EditProximity(to_student=to_student, to_reference=to_reference)
