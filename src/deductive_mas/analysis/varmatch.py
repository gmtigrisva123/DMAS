"""Match variables by behaviour, not by name.

A student writing l / r instead of lo / hi is not making a mistake, so name
based matching is out. Each variable becomes the time series of its values
during execution, series are compared with DTW (robust to a different number
of iterations) and the global best assignment comes from the Hungarian
algorithm instead of greedy nearest neighbour, so one weird variable cannot
mess up all the others.

Parameters are matched by position since the signature fixes them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..util.linalg import dtw_distance, hungarian, zscore
from ..util.text import char_ngrams, jaccard
from .tracer import ExecutionResult

# weights of the three similarity channels
W_MAGNITUDE = 0.50
W_SHAPE = 0.35
W_NAME = 0.15


@dataclass(frozen=True)
class RoleMapping:
    """Student variable <-> reference variable correspondence."""

    pairs: Mapping[str, str] = field(default_factory=dict)
    scores: Mapping[str, float] = field(default_factory=dict)
    forced: Tuple[str, ...] = ()
    unmatched_student: Tuple[str, ...] = ()
    unmatched_reference: Tuple[str, ...] = ()

    @property
    def roles(self) -> Tuple[str, ...]:
        """Reference side role names, sorted. The shared comparison basis."""
        return tuple(sorted(self.pairs.values()))

    def inverse(self) -> Dict[str, str]:
        return {v: k for k, v in self.pairs.items()}

    def confidence(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def describe(self) -> List[str]:
        out = []
        for student, reference in sorted(self.pairs.items()):
            tag = "positional" if student in self.forced else f"{self.scores.get(student, 0.0):.2f}"
            arrow = "=" if student == reference else "->"
            out.append(f"{student} {arrow} {reference} ({tag})")
        return out


def series_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarity in [0, 1] between two numeric traces.
    Two channels: magnitude (raw values, jointly scaled) tells lo from hi,
    shape (z-scored) recognises the same role over a different range.
    """
    if not a or not b:
        return 0.0
    span = max(max(abs(x) for x in a), max(abs(x) for x in b), 1.0)
    raw_a = [x / span for x in a]
    raw_b = [x / span for x in b]
    band = max(4, abs(len(a) - len(b)))
    magnitude = 1.0 / (1.0 + dtw_distance(raw_a, raw_b, band=band) / max(len(a), len(b)))

    za, zb = zscore(list(a)), zscore(list(b))
    if not any(za) and not any(zb):
        shape = 1.0                      # both constant, perfectly consistent
    elif not any(za) or not any(zb):
        shape = 0.0                      # one constant one varying, different roles
    else:
        shape = 1.0 / (1.0 + dtw_distance(za, zb, band=band) / max(len(a), len(b)))
    return W_MAGNITUDE / (W_MAGNITUDE + W_SHAPE) * magnitude + W_SHAPE / (W_MAGNITUDE + W_SHAPE) * shape


def name_similarity(a: str, b: str) -> float:
    """Weak lexical prior on identifier names, tie breaker only."""
    if a == b:
        return 1.0
    return jaccard(char_ngrams(a, 2), char_ngrams(b, 2))


def match_variables(
    student: ExecutionResult,
    reference: ExecutionResult,
    *,
    student_params: Sequence[str] = (),
    reference_params: Sequence[str] = (),
    min_similarity: float = 0.55,
) -> RoleMapping:
    """Optimal student variable -> reference variable assignment."""
    student_series = student.variable_series()
    reference_series = reference.variable_series()

    pairs: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    forced: List[str] = []

    # parameters are fixed by the signature
    for s_name, r_name in zip(student_params, reference_params):
        pairs[s_name] = r_name
        scores[s_name] = 1.0
        forced.append(s_name)

    s_free = sorted(n for n in student_series if n not in pairs)
    r_free = sorted(n for n in reference_series if n not in set(pairs.values()))
    if not s_free or not r_free:
        return RoleMapping(
            pairs=pairs,
            scores=scores,
            forced=tuple(forced),
            unmatched_student=tuple(s_free),
            unmatched_reference=tuple(r_free),
        )

    similarity = [
        [
            (1.0 - W_NAME) * series_similarity(student_series[s], reference_series[r])
            + W_NAME * name_similarity(s, r)
            for r in r_free
        ]
        for s in s_free
    ]

    transposed = len(s_free) > len(r_free)
    cost = (
        [[1.0 - similarity[i][j] for i in range(len(s_free))] for j in range(len(r_free))]
        if transposed
        else [[1.0 - value for value in row] for row in similarity]
    )
    assignment = hungarian(cost)

    matched_student: set = set()
    matched_reference: set = set()
    for index, target in enumerate(assignment):
        if target < 0:
            continue
        s_index, r_index = (target, index) if transposed else (index, target)
        score = similarity[s_index][r_index]
        if score < min_similarity:
            continue
        pairs[s_free[s_index]] = r_free[r_index]
        scores[s_free[s_index]] = score
        matched_student.add(s_free[s_index])
        matched_reference.add(r_free[r_index])

    return RoleMapping(
        pairs=pairs,
        scores=scores,
        forced=tuple(forced),
        unmatched_student=tuple(n for n in s_free if n not in matched_student),
        unmatched_reference=tuple(n for n in r_free if n not in matched_reference),
    )
