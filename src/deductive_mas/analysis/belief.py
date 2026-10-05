"""Dempster-Shafer evidence fusion.

The evaluator has three quite different evidence sources (rules, trace
divergence, model). Averaging them is wrong (not the same quantity) and
multiplying is wrong too (not independent likelihoods). DS belief functions
fit better:

- a source can put mass on a hypothesis OR on ignorance (Theta), which is what
  "the rules have nothing to say" means
- discounting turns a per-source reliability into mass moved to Theta
- the normalisation constant K measures conflict between sources, the
  orchestrator uses it to mark a diagnosis as inconclusive

Only singletons and Theta carry mass so combining is O(n), not O(2^n).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

_EPS = 1e-12


@dataclass(frozen=True)
class MassFunction:
    """A mass function over singletons plus ignorance."""

    focal: Mapping[str, float] = field(default_factory=dict)
    theta: float = 1.0

    @staticmethod
    def vacuous() -> "MassFunction":
        """Total ignorance, the identity for combination."""
        return MassFunction(focal={}, theta=1.0)

    @staticmethod
    def from_scores(
        scores: Mapping[str, float], *, reliability: float = 1.0, cap: float = 0.98
    ) -> "MassFunction":
        """Discounted mass function from per hypothesis scores in [0, 1].
        Scores are scaled so the total never goes above cap (some mass must stay
        on Theta), then discounted by reliability.
        """
        positive = {k: max(0.0, float(v)) for k, v in scores.items() if v and v > 0.0}
        total = sum(positive.values())
        if total <= _EPS:
            return MassFunction.vacuous()
        scale = min(1.0, cap / total) * max(0.0, min(1.0, reliability))
        focal = {k: v * scale for k, v in positive.items()}
        assigned = sum(focal.values())
        return MassFunction(focal=focal, theta=max(0.0, 1.0 - assigned))

    def combine(self, other: "MassFunction") -> Tuple["MassFunction", float]:
        """Dempster's rule. Also returns the conflict K.

        With mass only on singletons and Theta:

            m(A)     = m1(A) m2(A) + m1(A) m2(Theta) + m1(Theta) m2(A)
            m(Theta) = m1(Theta) m2(Theta)
            K        = sum over A != B of m1(A) m2(B)

        Result is divided by 1 - K. If K = 1 (total conflict) we just return self
        instead of dividing by zero.
        """
        keys = set(self.focal) | set(other.focal)
        raw: Dict[str, float] = {}
        for key in keys:
            m1 = self.focal.get(key, 0.0)
            m2 = other.focal.get(key, 0.0)
            raw[key] = m1 * m2 + m1 * other.theta + self.theta * m2
        raw_theta = self.theta * other.theta

        total_1 = sum(self.focal.values())
        total_2 = sum(other.focal.values())
        agreement = sum(self.focal.get(k, 0.0) * other.focal.get(k, 0.0) for k in keys)
        conflict = total_1 * total_2 - agreement

        denominator = 1.0 - conflict
        if denominator <= _EPS:
            return self, 1.0
        combined = {k: v / denominator for k, v in raw.items() if v > _EPS}
        return MassFunction(focal=combined, theta=max(0.0, raw_theta / denominator)), conflict

    def belief(self, hypothesis: str) -> float:
        """Lower bound on P(hypothesis)."""
        return self.focal.get(hypothesis, 0.0)

    def plausibility(self, hypothesis: str) -> float:
        """Upper bound: everything not committed to a rival hypothesis."""
        return self.focal.get(hypothesis, 0.0) + self.theta

    def pignistic(self) -> Dict[str, float]:
        """Pignistic transform (Smets): split the ignorance evenly."""
        if not self.focal:
            return {}
        share = self.theta / len(self.focal)
        return {k: v + share for k, v in self.focal.items()}

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.focal.items(), key=lambda kv: (-kv[1], kv[0]))

    @property
    def uncertainty(self) -> float:
        return self.theta


@dataclass
class FusionResult:
    """Fused belief plus what the orchestrator needs to know about conflict."""

    mass: MassFunction = field(default_factory=MassFunction.vacuous)
    conflicts: Tuple[float, ...] = ()
    sources: Tuple[str, ...] = ()

    @property
    def max_conflict(self) -> float:
        return max(self.conflicts) if self.conflicts else 0.0

    @property
    def contested(self) -> bool:
        """True if sources disagree enough that a confident claim is unsafe."""
        return self.max_conflict > 0.55

    def ranked(self) -> List[Tuple[str, float]]:
        return self.mass.ranked()


def fuse(sources: Sequence[Tuple[str, MassFunction]]) -> FusionResult:
    """Combine sources in order, keeping the pairwise conflicts."""
    result = MassFunction.vacuous()
    conflicts: List[float] = []
    names: List[str] = []
    for name, mass in sources:
        if not mass.focal:
            continue
        result, conflict = result.combine(mass)
        conflicts.append(conflict)
        names.append(name)
    return FusionResult(mass=result, conflicts=tuple(conflicts), sources=tuple(names))
