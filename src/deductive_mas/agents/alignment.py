"""Knowledge graph alignment agent.

Takes the diagnosis and finds which prerequisite concept to actually teach. It
is not a lookup: a misconception points at several concepts, each has
prerequisites, and the one worth teaching is the deepest one that is blamed
and not yet mastered. Three steps:

1. blame propagation: personalised PageRank on the reversed prerequisite
   graph, seeded with the diagnosis beliefs
2. mastery update: the blame goes into BKT as soft negative evidence, then a
   weakest link pass keeps mastery consistent with the prerequisite order
3. frontier: the unmastered concepts whose prerequisites are mastered. That
   set is basically the ZPD read off the graph.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..domain import AlignmentReport
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState, MasteryTracker
from ..knowledge.misconceptions import MISCONCEPTIONS
from ..knowledge.ontology import knowledge_graph
from .base import Agent, Blackboard


class KnowledgeGraphAlignmentAgent(Agent):
    """Turns a diagnosis into a located prerequisite deficiency."""

    name = "kg-alignment"
    role = "prerequisite localisation"

    def __init__(
        self,
        config: Optional[Config] = None,
        graph: Optional[KnowledgeGraph] = None,
        tracker: Optional[MasteryTracker] = None,
    ):
        self.config = config or Config()
        self.graph = graph or knowledge_graph()
        self.tracker = tracker or MasteryTracker(self.graph, self.config.knowledge)

    def run(self, board: Blackboard) -> Blackboard:
        diagnosis = board.diagnosis
        # no history yet: assume the student is roughly at the level of the task
        # they picked, not at the bottom
        state: MasteryState = board.extras.get("mastery") or self.tracker.initial(
            ability=board.problem.difficulty
        )

        seeds = self._seed_concepts(board)
        task_concepts = [cid for cid in board.problem.concepts if cid in self.graph]
        passed_everything = bool(
            diagnosis is not None
            and diagnosis.checks_total
            and diagnosis.checks_passed == diagnosis.checks_total
        )

        if not seeds:
            # nothing diagnosed = evidence of mastery for the task concepts. (An older
            # version fed the task concepts in as blame here, so a student who fixed
            # their code saw their mastery go DOWN. Don't do that.) Blame is still kept
            # so the intervention has something to point at.
            updated = self.tracker.apply_success(state, task_concepts, weight=1.0)
            blame = self.graph.propagate_blame(
                {cid: 1.0 for cid in task_concepts},
                damping=self.config.knowledge.blame_damping,
                iterations=self.config.knowledge.blame_iterations,
                tolerance=self.config.knowledge.blame_tolerance,
            )
        else:
            blame = self.graph.propagate_blame(
                seeds,
                damping=self.config.knowledge.blame_damping,
                iterations=self.config.knowledge.blame_iterations,
                tolerance=self.config.knowledge.blame_tolerance,
            )
            updated = self.tracker.apply_blame(state, blame)
            if passed_everything:
                # right answer but a flawed belief (e.g. list used as a queue): the
                # correctness concepts still get partial credit
                implicated = set(blame)
                credit = [cid for cid in task_concepts if cid not in implicated]
                if credit:
                    updated = self.tracker.apply_success(updated, credit, weight=0.5)

        implicated = sorted(blame, key=lambda cid: -blame[cid])
        peak = max(blame.values()) if blame else 0.0
        eligible = [
            cid for cid in implicated
            if peak <= 0.0 or blame[cid] >= peak * self.config.knowledge.min_blame_ratio
        ]
        frontier = self.tracker.frontier(
            updated, candidates=eligible or implicated, background=state
        )
        root_cause = self._root_cause(frontier, blame, direct=set(seeds))

        report = AlignmentReport(
            blame=dict(blame),
            deficiency_frontier=tuple(frontier),
            prerequisite_paths=self._paths(seeds, root_cause),
            mastery_before={cid: state.get(cid) for cid in implicated},
            mastery_after={cid: updated.get(cid) for cid in implicated},
            root_cause=root_cause,
        )
        board.alignment = report
        board.extras["mastery"] = updated
        board.extras["seeded_concepts"] = seeds
        if root_cause and diagnosis is not None and diagnosis.primary is not None:
            board.note(
                f"remediation target {root_cause!r} lies "
                f"{self.graph.depth(root_cause)} prerequisite levels deep"
            )
        return board

    # internals
    def _seed_concepts(self, board: Blackboard) -> Dict[str, float]:
        """Spread each hit's belief over the concepts it implicates."""
        seeds: Dict[str, float] = {}
        diagnosis = board.diagnosis
        if diagnosis is None:
            return seeds
        for hit in diagnosis.hits:
            entry = MISCONCEPTIONS.get(hit.misconception_id)
            if entry is None or not entry.concepts:
                continue
            share = hit.belief / len(entry.concepts)
            for concept in entry.concepts:
                if concept in self.graph:
                    seeds[concept] = seeds.get(concept, 0.0) + share
        return seeds

    def _root_cause(
        self,
        frontier: Sequence[str],
        blame: Dict[str, float],
        direct: Optional[set] = None,
    ) -> Optional[str]:
        """The frontier concept the diagnosis points at most directly.

        A concept named by the misconception itself beats one that only got blame
        through propagation, otherwise a shared foundation concept collecting mass
        from several places would always win and the advice drifts away from the
        actual error.
        """
        if not frontier:
            return None
        direct = direct or set()
        return max(
            frontier,
            key=lambda cid: (cid in direct, blame.get(cid, 0.0), -self.graph.depth(cid), cid),
        )

    def _paths(self, seeds: Dict[str, float], root_cause: Optional[str]) -> Tuple[Tuple[str, ...], ...]:
        """Prerequisite chains from the root cause up to each implicated concept."""
        if root_cause is None:
            return ()
        chains: List[Tuple[str, ...]] = []
        for target in sorted(seeds, key=lambda cid: -seeds[cid])[:4]:
            chain = self.graph.prerequisite_path(target, root_cause)
            if len(chain) > 1:
                chains.append(chain)
        # a chain that is a prefix of a longer one adds nothing
        chains.sort(key=len, reverse=True)
        kept: List[Tuple[str, ...]] = []
        for chain in chains:
            if not any(longer[: len(chain)] == chain for longer in kept):
                kept.append(chain)
        return tuple(kept[:3])
