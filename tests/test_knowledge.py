import pytest

from deductive_mas.config import KnowledgeConfig
from deductive_mas.domain import Concept
from deductive_mas.errors import KnowledgeGraphError
from deductive_mas.knowledge.graph import KnowledgeGraph
from deductive_mas.knowledge.mastery import MasteryTracker
from deductive_mas.knowledge.misconceptions import (
    MISCONCEPTIONS,
    RULES,
    is_correctness_misconception,
    is_subsumed,
    validate_taxonomy,
)
from deductive_mas.knowledge.zpd import (
    Item,
    ZPDSelector,
    ability_from_mastery,
    estimate_ability,
    fisher_information,
    probability_correct,
)


class TestGraphStructure:
    def test_ontology_is_a_valid_dag(self, graph):
        assert len(graph) > 50
        assert len(graph.topological_order()) == len(graph)

    def test_prerequisites_precede_dependents(self, graph):
        position = {cid: i for i, cid in enumerate(graph.topological_order())}
        for concept in graph:
            for prerequisite in graph.prerequisites(concept.cid):
                assert position[prerequisite] < position[concept.cid]

    def test_a_cycle_is_refused(self):
        with pytest.raises(KnowledgeGraphError):
            KnowledgeGraph([
                Concept("a", "A", "s", "", ("b",)),
                Concept("b", "B", "s", "", ("a",)),
            ])

    def test_an_unknown_prerequisite_is_refused(self):
        with pytest.raises(KnowledgeGraphError):
            KnowledgeGraph([Concept("a", "A", "s", "", ("ghost",))])

    def test_a_duplicate_id_is_refused(self):
        with pytest.raises(KnowledgeGraphError):
            KnowledgeGraph([Concept("a", "A", "s", ""), Concept("a", "A2", "s", "")])

    def test_ancestors_and_descendants_are_inverse(self, graph):
        assert "sequence-indexing" in graph.ancestors("binary-search")
        assert "binary-search" in graph.descendants("sequence-indexing")

    def test_prerequisite_path_reads_in_learning_order(self, graph):
        path = graph.prerequisite_path("binary-search", "sequence-indexing")
        assert path[0] == "sequence-indexing" and path[-1] == "binary-search"

    def test_no_path_returns_empty(self, graph):
        assert graph.prerequisite_path("var-binding", "dijkstra") == ()

    def test_unknown_concept_raises(self, graph):
        with pytest.raises(KnowledgeGraphError):
            graph.concept("nope")


class TestBlamePropagation:
    def test_blame_is_a_distribution(self, graph):
        blame = graph.propagate_blame({"binary-search": 1.0})
        assert sum(blame.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in blame.values())

    def test_the_seed_carries_the_most_blame(self, graph):
        blame = graph.propagate_blame({"binary-search": 1.0})
        assert max(blame, key=lambda k: blame[k]) == "binary-search"

    def test_blame_reaches_prerequisites_and_decays(self, graph):
        blame = graph.propagate_blame({"binary-search": 1.0})
        assert blame.get("half-open-intervals", 0) > 0
        assert blame["half-open-intervals"] < blame["binary-search"]

    def test_blame_does_not_flow_to_dependents(self, graph):
        blame = graph.propagate_blame({"half-open-intervals": 1.0})
        assert blame.get("dijkstra", 0.0) == 0.0

    def test_empty_seeds_produce_nothing(self, graph):
        assert graph.propagate_blame({}) == {}
        assert graph.propagate_blame({"binary-search": 0.0}) == {}

    def test_shared_prerequisites_accumulate(self, graph):
        one = graph.propagate_blame({"bfs": 1.0})
        two = graph.propagate_blame({"bfs": 0.5, "dfs": 0.5})
        assert two.get("visited-set", 0) > 0 and one.get("visited-set", 0) > 0


class TestDeficiencyFrontier:
    def test_frontier_is_the_root_cause_not_the_foundation(self, graph):
        mastery = {cid: 0.9 for cid in graph.ids}
        for cid in ("half-open-intervals", "binary-search", "boundary-conditions", "loop-invariant"):
            mastery[cid] = 0.2
        frontier = graph.deficiency_frontier(mastery)
        assert frontier == ("loop-invariant",)

    def test_background_mastery_prevents_a_cascade(self, graph):
        background = {cid: 0.9 for cid in graph.ids}
        post = {cid: 0.2 for cid in graph.ids}
        frontier = graph.deficiency_frontier(
            post, candidates=["binary-search"], prerequisite_mastery=background
        )
        assert frontier == ("binary-search",)

    def test_everything_mastered_means_no_frontier(self, graph):
        assert graph.deficiency_frontier({cid: 0.99 for cid in graph.ids}) == ()


class TestMastery:
    def test_cold_start_respects_difficulty(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial(ability=0.0)
        assert state.get("var-binding") > state.get("dijkstra")

    def test_a_capable_learner_starts_higher(self, graph):
        tracker = MasteryTracker(graph)
        assert tracker.initial(2.0).mean() > tracker.initial(-2.0).mean()

    def test_failure_lowers_and_success_raises(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial()
        after_failure = tracker.observe(state, "binary-search", correct=False)
        after_success = tracker.observe(state, "binary-search", correct=True)
        assert after_failure.get("binary-search") < state.get("binary-search")
        assert after_success.get("binary-search") > state.get("binary-search")

    def test_soft_evidence_moves_less_than_hard_evidence(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial()
        soft = tracker.observe(state, "binary-search", correct=False, weight=0.2)
        hard = tracker.observe(state, "binary-search", correct=False, weight=1.0)
        assert hard.get("binary-search") < soft.get("binary-search")

    def test_zero_weight_is_a_no_op(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial()
        assert tracker.observe(state, "binary-search", correct=False, weight=0.0) is state

    def test_weakest_link_ceiling_propagates_upward(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial(2.0)
        state.posterior["loop-invariant"] = 0.05
        regularised = tracker.regularise(state)
        assert regularised.get("half-open-intervals") < state.get("half-open-intervals")

    def test_blame_becomes_soft_negative_evidence(self, graph):
        tracker = MasteryTracker(graph)
        state = tracker.initial(0.5)
        blame = graph.propagate_blame({"binary-search": 1.0})
        updated = tracker.apply_blame(state, blame)
        assert updated.get("binary-search") < state.get("binary-search")

    def test_invalid_bkt_parameters_are_refused(self, graph):
        with pytest.raises(ValueError):
            MasteryTracker(graph, KnowledgeConfig(bkt_slip=0.9))


class TestZPD:
    def test_response_probability_is_monotone_in_ability(self):
        item = Item("i", difficulty=0.0)
        assert probability_correct(-2, item) < probability_correct(0, item) < probability_correct(2, item)

    def test_guessing_is_the_floor(self):
        item = Item("i", difficulty=0.0, guessing=0.25)
        assert probability_correct(-40, item) == pytest.approx(0.25, abs=1e-6)

    def test_information_peaks_near_the_difficulty(self):
        item = Item("i", difficulty=0.5, guessing=0.0)
        grid = [t / 10 for t in range(-30, 31)]
        peak = max(grid, key=lambda t: fisher_information(t, item))
        assert peak == pytest.approx(0.5, abs=0.2)

    def test_ability_estimate_without_responses_is_the_prior(self):
        estimate = estimate_ability([])
        assert estimate.mean == 0.0 and estimate.sd == 1.0

    def test_ability_estimate_tracks_the_responses(self):
        easy, hard = Item("e", -1.5), Item("h", 1.5)
        strong = estimate_ability([(easy, True), (hard, True)])
        weak = estimate_ability([(easy, False), (hard, False)])
        assert strong.mean > weak.mean

    def test_mastery_projects_onto_ability(self):
        difficulties = {"a": -2.0, "b": 0.0, "c": 2.0}
        high = ability_from_mastery({"a": 0.95, "b": 0.9, "c": 0.8}, difficulties)
        low = ability_from_mastery({"a": 0.3, "b": 0.1, "c": 0.05}, difficulties)
        assert high > low

    def test_band_weight_peaks_inside_the_zpd(self):
        selector = ZPDSelector()
        assert selector.band_weight(0.65) == 1.0
        assert selector.band_weight(0.99) < 1.0
        assert selector.band_weight(0.05) < 1.0

    def test_selection_prefers_relevant_reachable_items(self):
        items = [
            Item("easy", -3.0, concepts=("x",)),
            Item("fit", 0.0, concepts=("target",)),
            Item("hard", 3.5, concepts=("y",)),
        ]
        chosen = ZPDSelector().select(items, 0.0, {"target": 1.0}, k=1)
        assert chosen and chosen[0].item.iid == "fit"

    def test_selection_diversifies_concepts(self):
        items = [
            Item("a", 0.0, concepts=("p", "q")),
            Item("b", 0.0, concepts=("p", "q")),
            Item("c", 0.0, concepts=("r",)),
        ]
        chosen = ZPDSelector().select(items, 0.0, {"p": 1.0, "q": 1.0, "r": 0.9}, k=2)
        assert {s.item.iid for s in chosen} == {"a", "c"}

    def test_selection_of_nothing_is_empty(self):
        assert ZPDSelector().select([], 0.0, {}) == []


class TestTaxonomy:
    def test_taxonomy_is_referentially_sound(self, graph):
        assert validate_taxonomy(graph.ids) == []

    def test_every_rule_targets_a_known_misconception(self):
        assert all(rule.misconception_id in MISCONCEPTIONS for rule in RULES)

    def test_every_misconception_has_a_student_voice(self):
        assert all(entry.student_voice for entry in MISCONCEPTIONS.values())

    def test_cost_only_findings_are_classified(self):
        assert not is_correctness_misconception("cx.asymptotic-gap")
        assert is_correctness_misconception("rec.missing-base-case")

    def test_a_symptom_is_subsumed_by_its_cause(self):
        assert is_subsumed("cx.asymptotic-gap", ["dp.recomputed-subproblems"])
        assert not is_subsumed("cx.asymptotic-gap", ["py.float-index"])
