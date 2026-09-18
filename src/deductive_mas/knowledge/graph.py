"""The knowledge graph (prerequisite DAG over concepts) and the algorithms on
it:

- topological order / transitive closure. A cycle is a curriculum bug so the
  constructor refuses it.
- blame propagation: personalised PageRank on the reversed graph. A
  misconception at one concept is evidence about its prerequisites with
  influence decaying by path length, random walk with restart sums that over
  all paths at once instead of a fixed depth expansion.
- deficiency frontier: among the blamed concepts, the ones whose own
  prerequisites are already mastered. These are the root causes and the only
  places where remediation can start.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..domain import Concept
from ..errors import KnowledgeGraphError


class KnowledgeGraph:
    """Immutable prerequisite DAG over Concept objects."""

    def __init__(self, concepts: Sequence[Concept]):
        self._concepts: Dict[str, Concept] = {}
        for concept in concepts:
            if concept.cid in self._concepts:
                raise KnowledgeGraphError(f"duplicate concept id: {concept.cid!r}")
            self._concepts[concept.cid] = concept

        self._prerequisites: Dict[str, Tuple[str, ...]] = {}
        self._dependents: Dict[str, List[str]] = {cid: [] for cid in self._concepts}
        for concept in self._concepts.values():
            missing = [p for p in concept.prerequisites if p not in self._concepts]
            if missing:
                raise KnowledgeGraphError(
                    f"concept {concept.cid!r} lists unknown prerequisites: {missing}"
                )
            self._prerequisites[concept.cid] = tuple(concept.prerequisites)
            for prerequisite in concept.prerequisites:
                self._dependents[prerequisite].append(concept.cid)

        self._order = self._topological_order()

    # accessors
    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, cid: object) -> bool:
        return cid in self._concepts

    def __iter__(self):
        return iter(self._concepts.values())

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self._order)

    def concept(self, cid: str) -> Concept:
        try:
            return self._concepts[cid]
        except KeyError as exc:
            raise KnowledgeGraphError(f"unknown concept: {cid!r}") from exc

    def name(self, cid: str) -> str:
        return self._concepts[cid].name if cid in self._concepts else cid

    def prerequisites(self, cid: str) -> Tuple[str, ...]:
        return self._prerequisites.get(cid, ())

    def dependents(self, cid: str) -> Tuple[str, ...]:
        return tuple(self._dependents.get(cid, ()))

    def strata(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for cid in self._order:
            out.setdefault(self._concepts[cid].stratum, []).append(cid)
        return out

    # topology
    def _topological_order(self) -> List[str]:
        """Kahn's algorithm, raises on a cycle."""
        indegree = {cid: len(self._prerequisites[cid]) for cid in self._concepts}
        ready = deque(sorted(cid for cid, degree in indegree.items() if degree == 0))
        order: List[str] = []
        while ready:
            cid = ready.popleft()
            order.append(cid)
            for dependent in sorted(self._dependents[cid]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if len(order) != len(self._concepts):
            remaining = sorted(set(self._concepts) - set(order))
            raise KnowledgeGraphError(
                "prerequisite graph contains a cycle involving: " + ", ".join(remaining[:8])
            )
        return order

    def topological_order(self) -> Tuple[str, ...]:
        return tuple(self._order)

    def ancestors(self, cid: str) -> Set[str]:
        """All transitive prerequisites of cid."""
        seen: Set[str] = set()
        stack = list(self.prerequisites(cid))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.prerequisites(current))
        return seen

    def descendants(self, cid: str) -> Set[str]:
        """Everything that depends on cid, transitively."""
        seen: Set[str] = set()
        stack = list(self.dependents(cid))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.dependents(current))
        return seen

    def depth(self, cid: str) -> int:
        """Longest prerequisite chain ending at cid (0 for a foundation)."""
        best = 0
        memo: Dict[str, int] = {}

        def visit(node: str) -> int:
            if node in memo:
                return memo[node]
            prerequisites = self.prerequisites(node)
            memo[node] = 0 if not prerequisites else 1 + max(visit(p) for p in prerequisites)
            return memo[node]

        best = visit(cid)
        return best

    def prerequisite_path(self, target: str, source: str) -> Tuple[str, ...]:
        """Shortest prerequisite chain from source up to target, in learning order
        (source first) so it reads like a study plan.
        """
        if target == source:
            return (target,)
        parent: Dict[str, Optional[str]] = {target: None}
        queue = deque([target])
        while queue:
            current = queue.popleft()
            for prerequisite in self.prerequisites(current):
                if prerequisite in parent:
                    continue
                parent[prerequisite] = current
                if prerequisite == source:
                    chain: List[str] = [source]
                    node: Optional[str] = current
                    while node is not None:
                        chain.append(node)
                        node = parent[node]
                    return tuple(chain)
                queue.append(prerequisite)
        return ()

    # blame
    def propagate_blame(
        self,
        seeds: Mapping[str, float],
        *,
        damping: float = 0.62,
        iterations: int = 64,
        tolerance: float = 1e-10,
    ) -> Dict[str, float]:
        """Personalised PageRank on the reversed prerequisite graph.

        A walker starts at a blamed concept (proportional to the belief) and at
        each step either moves to one of its prerequisites (prob damping) or
        teleports back to a seed (prob 1 - damping). The stationary distribution
        is the blame. Compared to a fixed depth expansion: influence decays
        smoothly, and a prerequisite shared by several failing concepts ranks
        above one hit by a single path.
        """
        total = sum(max(0.0, v) for v in seeds.values())
        if total <= 0.0:
            return {}
        personalisation = {
            cid: max(0.0, value) / total
            for cid, value in seeds.items()
            if cid in self._concepts and value > 0.0
        }
        if not personalisation:
            return {}

        rank: Dict[str, float] = dict(personalisation)
        for _ in range(max(1, iterations)):
            nxt: Dict[str, float] = {cid: 0.0 for cid in rank}
            dangling = 0.0
            for cid, mass in rank.items():
                targets = self.prerequisites(cid)
                if not targets:
                    dangling += mass          # foundation concept, walker restarts
                    continue
                share = damping * mass / len(targets)
                for target in targets:
                    nxt[target] = nxt.get(target, 0.0) + share
            leak = (1.0 - damping) + damping * dangling
            for cid, weight in personalisation.items():
                nxt[cid] = nxt.get(cid, 0.0) + leak * weight
            delta = sum(abs(nxt.get(cid, 0.0) - rank.get(cid, 0.0)) for cid in set(nxt) | set(rank))
            rank = nxt
            if delta < tolerance:
                break
        norm = sum(rank.values())
        if norm <= 0.0:
            return {}
        return {cid: value / norm for cid, value in rank.items() if value > 1e-9}

    # frontier
    def deficiency_frontier(
        self,
        mastery: Mapping[str, float],
        *,
        threshold: float = 0.62,
        candidates: Optional[Iterable[str]] = None,
        prerequisite_mastery: Optional[Mapping[str, float]] = None,
    ) -> Tuple[str, ...]:
        """Unmastered concepts whose prerequisites are already mastered.

        These are the root causes. Teaching a concept whose prerequisites are
        shaky just moves the failure up one level, so the frontier is where
        teaching can actually work (the ZPD read off the graph).

        prerequisite_mastery matters: blame goes DOWN the chain, so one failure
        lowers belief in a concept and in everything under it. Testing the
        prerequisites against those same lowered values counts the evidence twice
        and walks the frontier all the way down to "does not understand
        variables". Passing the background (pre diagnosis) beliefs asks the right
        question: given what we already believed, where is the highest point this
        failure could come from?
        """
        pool = list(candidates) if candidates is not None else list(self._concepts)
        background = prerequisite_mastery if prerequisite_mastery is not None else mastery
        unmastered = {
            cid for cid in pool
            if cid in self._concepts and mastery.get(cid, 0.0) < threshold
        }
        frontier = [
            cid for cid in unmastered
            if all(background.get(p, 0.0) >= threshold for p in self.prerequisites(cid))
        ]
        if not frontier and unmastered:
            # everything is shaky, fall back to the shallowest failing concepts
            shallowest = min(self.depth(cid) for cid in unmastered)
            frontier = [cid for cid in unmastered if self.depth(cid) == shallowest]
        return tuple(sorted(frontier, key=lambda cid: (self.depth(cid), cid)))

    # rendering
    def to_dot(self, highlight: Mapping[str, float] = ()) -> str:
        highlight = dict(highlight or {})
        lines = ["digraph knowledge {", "  rankdir=BT;", "  node [shape=box, fontname=Menlo];"]
        for cid in self._order:
            weight = highlight.get(cid, 0.0)
            colour = f', style=filled, fillcolor="0.02 {min(1.0, weight * 3):.2f} 1.0"' if weight else ""
            lines.append(f'  "{cid}" [label="{self._concepts[cid].name}"{colour}];')
            for prerequisite in self.prerequisites(cid):
                lines.append(f'  "{prerequisite}" -> "{cid}";')
        lines.append("}")
        return "\n".join(lines)
