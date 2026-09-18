"""Tokenisers. Retrieval, the leak detector and the grounding check all need
tokens and they need them to be comparable, so they all come from here.
"""

import io
import re
import token as token_module
import tokenize as tokenize_module
from typing import Optional, Iterable, Iterator, List, Sequence, Set, Tuple

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n{2,}")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

PROSE_STOPWORDS = frozenset(
    """a an the and or but if then else of to in on for with without by from as at is are was
    were be been being it its this that these those we you they he she i not no do does did can
    could should would may might will shall must have has had there here when while which who whom
    whose what how why into over under above below than so such very more most some any each""".split()
)

CODE_STOPWORDS = frozenset(
    """def return if else elif for while in range len self import from as pass none true false
    print int str list dict set tuple""".split()
)


def _split_identifier(word: str) -> List[str]:
    """lowerBound_idx -> ['lower', 'bound', 'idx'] (camel + snake aware)."""
    pieces = [p for p in _CAMEL_RE.sub(" ", word).replace("_", " ").split() if p]
    return [p.lower() for p in pieces]


def prose_tokens(text: str, *, stem: bool = True, keep_stopwords: bool = False) -> List[str]:
    """Tokenise prose into a bag of comparable terms."""
    out: List[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        for piece in _split_identifier(word) or [word.lower()]:
            if len(piece) < 2 and not piece.isdigit():
                continue
            if not keep_stopwords and piece in PROSE_STOPWORDS:
                continue
            out.append(_stem(piece) if stem else piece)
    return out


# spelling variants that must not split the vocabulary. The corpus is in
# British English, a student typing "memoization" or "analyze" must reach the
# same cards.
def _normalise_orthography(word: str) -> str:
    for suffix, replacement in (("ization", "isation"), ("ize", "ise"), ("yze", "yse")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + replacement
    return word


def _stem(word: str) -> str:
    """Very conservative suffix stripper (a subset of Porter step 1).
    Full Porter over-conflates technical words (binary -> bin), so only
    inflectional endings are removed.
    """
    word = _normalise_orthography(word)
    if len(word) <= 4:
        return word
    for suffix in ("ations", "ation", "ities", "ings", "edly", "ies", "ing", "ers", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if suffix == "ies":
                return stem + "y"
            if suffix in {"ations", "ation"}:
                return stem + "e" if not stem.endswith("e") else stem
            return stem
    return word


def code_tokens(
    source: str, *, normalise_identifiers: bool = False, normalise_numbers: Optional[bool] = None
) -> List[str]:
    """Tokenise python source with the stdlib tokenizer.
    normalise_identifiers replaces every non builtin name with ID, which the
    naming invariant fingerprint needs. Numbers follow the same setting unless
    normalise_numbers says otherwise: the leak detector keeps them literal
    because p[1] and p[0] are different programs and often the whole fix.
    """
    if normalise_numbers is None:
        normalise_numbers = normalise_identifiers
    out: List[str] = []
    try:
        stream = tokenize_module.generate_tokens(io.StringIO(source).readline)
        for tok in stream:
            if tok.type in (
                token_module.NEWLINE,
                token_module.NL,
                token_module.INDENT,
                token_module.DEDENT,
                token_module.ENDMARKER,
                tokenize_module.COMMENT,
            ):
                continue
            if tok.type == token_module.STRING:
                out.append("STR")
            elif tok.type == token_module.NUMBER:
                out.append("NUM" if normalise_numbers else tok.string)
            elif tok.type == token_module.NAME:
                if tok.string in _PY_KEYWORDS or tok.string in _PY_BUILTINS:
                    out.append(tok.string)
                else:
                    out.append("ID" if normalise_identifiers else tok.string.lower())
            else:
                stripped = tok.string.strip()
                if stripped:
                    out.append(stripped)
    except (tokenize_module.TokenError, IndentationError, SyntaxError):
        # code that does not compile is the normal case here, fall back to a regex
        # tokenisation so we never lose the signal
        out = [m.group(0).lower() for m in _WORD_RE.finditer(source)]
    return out


_PY_KEYWORDS = frozenset(
    """False None True and as assert async await break class continue def del elif else except
    finally for from global if import in is lambda nonlocal not or pass raise return try while
    with yield""".split()
)

_PY_BUILTINS = frozenset(
    """abs all any bin bool dict divmod enumerate filter float format frozenset getattr hash int
    isinstance iter len list map max min next object pow print range repr reversed round set
    setattr slice sorted str sum tuple type zip""".split()
)


def ngrams(items: Sequence[str], n: int) -> Iterator[Tuple[str, ...]]:
    """Contiguous n-grams of items."""
    if n <= 0:
        raise ValueError("n must be positive")
    for i in range(len(items) - n + 1):
        yield tuple(items[i : i + n])


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Character n-gram set, for fuzzy identifier comparison."""
    squeezed = re.sub(r"\s+", " ", text.strip().lower())
    if len(squeezed) < n:
        return {squeezed} if squeezed else set()
    return {squeezed[i : i + n] for i in range(len(squeezed) - n + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when both empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def containment(a: Iterable[str], b: Iterable[str]) -> float:
    """Asymmetric containment |A n B| / |A|, the core of the leak detector."""
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / float(len(sa))


def normalised_levenshtein(a: str, b: str) -> float:
    """Edit distance scaled to [0, 1] (0 = identical), O(len(a) * len(b))."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / float(max(len(a), len(b)))


def sentences(text: str) -> List[str]:
    """Split prose into claim sized units for the grounding check."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p and p.strip()]
    return [p for p in parts if len(p) > 2]


def truncate(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate to width display columns, ellipsis if cut."""
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)].rstrip() + ellipsis


def wrap(text: str, width: int, indent: str = "") -> List[str]:
    """Whitespace preserving greedy wrapper (textwrap mangles code spans)."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = indent
        for word in paragraph.split():
            candidate = word if current.strip() == "" else current + " " + word
            if len(candidate) > width and current.strip():
                lines.append(current)
                current = indent + word
            else:
                current = candidate if current.strip() else indent + word
        if current.strip():
            lines.append(current)
    return lines
