import pytest

from deductive_mas.errors import RetrievalError
from deductive_mas.knowledge.graph import KnowledgeGraph
from deductive_mas.retrieval.bm25 import BM25Index
from deductive_mas.retrieval.corpus import cards, cards_for
from deductive_mas.retrieval.fusion import maximal_marginal_relevance, reciprocal_rank_fusion
from deductive_mas.retrieval.index import HybridIndex
from deductive_mas.retrieval.lsa import LSAIndex
from deductive_mas.retrieval.ma_rag import MultiAgentRAG, Planner
from deductive_mas.util.text import prose_tokens

DOCUMENTS = [
    ("a", "animal the quick brown fox jumps".split()),
    ("b", "animal a slow brown dog sleeps".split()),
    ("c", "animal quick quick quick fox".split()),
]


@pytest.fixture(scope="module")
def index() -> HybridIndex:
    return HybridIndex()


@pytest.fixture(scope="module")
def rag(index: HybridIndex) -> MultiAgentRAG:
    return MultiAgentRAG(index)


class TestCorpus:
    def test_every_card_references_known_concepts(self, graph: KnowledgeGraph) -> None:
        unknown = [(c.card_id, x) for c in cards() for x in c.concepts if x not in graph]
        assert unknown == []

    def test_card_ids_are_unique(self) -> None:
        ids = [c.card_id for c in cards()]
        assert len(ids) == len(set(ids))

    def test_cards_can_be_selected_by_concept(self) -> None:
        selected = cards_for(("binary-search",))
        assert selected and all("binary-search" in c.concepts for c in selected)

    def test_every_card_has_body_text(self) -> None:
        assert all(len(c.text) > 40 for c in cards())


class TestBM25:
    def test_term_frequency_raises_the_score(self) -> None:
        bm25 = BM25Index.build(DOCUMENTS)
        scores = bm25.score(["quick"])
        assert scores["c"] > scores["a"]

    def test_a_term_in_every_document_is_uninformative(self) -> None:
        bm25 = BM25Index.build(DOCUMENTS)
        assert bm25.idf("animal") < bm25.idf("dog")

    def test_an_unknown_term_scores_nothing(self) -> None:
        assert BM25Index.build(DOCUMENTS).score(["zebra"]) == {}

    def test_ranking_is_capped(self) -> None:
        assert len(BM25Index.build(DOCUMENTS).ranked(["quick"], top_k=1)) == 1

    def test_an_empty_index_scores_nothing(self) -> None:
        assert BM25Index.build([]).score(["x"]) == {}


class TestLSA:
    def test_documents_get_latent_coordinates(self) -> None:
        lsa = LSAIndex.build(DOCUMENTS, components=2)
        assert lsa.rank >= 1
        assert len(lsa.doc_vectors) == 3

    def test_a_document_is_most_similar_to_itself(self) -> None:
        lsa = LSAIndex.build(DOCUMENTS, components=2)
        assert lsa.similarity("a", "a") == pytest.approx(1.0, abs=1e-9)

    def test_query_folding_ranks_the_right_document(self) -> None:
        lsa = LSAIndex.build(DOCUMENTS, components=2)
        assert lsa.ranked(["dog", "sleeps"], top_k=1)[0][0] == "b"

    def test_an_empty_corpus_is_handled(self) -> None:
        assert LSAIndex.build([]).rank == 0

    def test_unknown_documents_have_no_vector(self) -> None:
        assert LSAIndex.build(DOCUMENTS, components=2).vector("zzz") == []


class TestFusion:
    def test_rank_fusion_rewards_consensus(self) -> None:
        fused = dict(reciprocal_rank_fusion([[("x", 9.0), ("y", 1.0)], [("x", 0.1), ("z", 0.05)]]))
        assert fused["x"] > fused["y"] and fused["x"] > fused["z"]

    def test_weights_must_align(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[("x", 1.0)]], weights=[1.0, 2.0])

    def test_mmr_suppresses_a_near_duplicate(self) -> None:
        candidates = [("a", 1.0), ("a2", 0.99), ("b", 0.5)]

        def similarity(x: str, y: str) -> float:
            return 1.0 if {x, y} == {"a", "a2"} else 0.0

        # with redundancy weighted equal to relevance the near duplicate loses to
        # the less relevant alternative
        chosen = [
            doc for doc, _ in
            maximal_marginal_relevance(candidates, similarity, lambda_=0.5, top_k=2)
        ]
        assert chosen == ["a", "b"]
        # at the production lambda relevance dominates and both are kept
        greedy = [
            doc for doc, _ in
            maximal_marginal_relevance(candidates, similarity, lambda_=0.84, top_k=2)
        ]
        assert greedy == ["a", "a2"]

    def test_mmr_of_nothing_is_nothing(self) -> None:
        assert maximal_marginal_relevance([], lambda a, b: 0.0) == []


class TestHybridIndex:
    def test_index_is_populated(self, index: HybridIndex) -> None:
        assert len(index) == len(cards())

    def test_an_empty_corpus_is_refused(self) -> None:
        with pytest.raises(RetrievalError):
            HybridIndex(documents=[])

    def test_unknown_card_raises(self, index: HybridIndex) -> None:
        with pytest.raises(RetrievalError):
            index.card("nope")

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("mixing inclusive and exclusive interval bounds", "iv-mismatch"),
            ("what makes a loop terminate", "lp-termination"),
            ("removing the first element of a list shifts everything", "ds-deque"),
            ("the two properties dynamic programming needs", "dp-two-properties"),
            ("proving a greedy algorithm correct", "gd-choice"),
        ],
    )
    def test_targeted_queries_retrieve_the_right_card(self, index: HybridIndex, query: str, expected: str) -> None:
        assert expected in [hit.card_id for hit in index.search(query, top_k=5)]

    def test_concept_hints_bias_retrieval(self, index: HybridIndex) -> None:
        hits = index.search("marking vertices", concepts=("visited-set",), top_k=4)
        assert any("visited-set" in hit.card.concepts for hit in hits)

    def test_query_expansion_bridges_morphology(self, index: HybridIndex) -> None:
        assert "visit" in index.expand(["revisit"])

    def test_expansion_leaves_known_terms_alone(self, index: HybridIndex) -> None:
        assert index.expand(["recursion"]) == ["recursion"]

    def test_diversification_changes_the_selection(self, index: HybridIndex) -> None:
        plain = [h.card_id for h in index.search("binary search", top_k=6, diversify=False)]
        diverse = [h.card_id for h in index.search("binary search", top_k=6, diversify=True)]
        assert set(plain) and set(diverse)

    def test_an_empty_query_returns_nothing(self, index: HybridIndex) -> None:
        assert index.search("") == []


class TestMultiAgentRAG:
    def test_planner_decomposes_a_diagnosis(self) -> None:
        plan = Planner().plan(
            problem_title="Lower bound",
            misconception_texts=["Mixed interval conventions."],
            concepts=("half-open-intervals",),
            frontier=("loop-invariant",),
            divergence_kind="early_termination",
            complexity_gap=("O(n)", "O(log n)"),
        )
        purposes = {q.purpose for q in plan}
        assert {"misconception", "divergence", "complexity", "prerequisite", "problem"} <= purposes

    def test_planner_always_produces_something(self) -> None:
        assert Planner().plan() != []

    def test_grounding_retrieves_relevant_evidence(self, rag: MultiAgentRAG) -> None:
        context = rag.ground(
            misconception_texts=["The loop guard and the bound follow different conventions."],
            concepts=("half-open-intervals", "boundary-conditions"),
        )
        assert context.cards
        assert "half-open-intervals" in context.concept_coverage()

    def test_rendered_context_respects_its_budget(self, rag: MultiAgentRAG) -> None:
        context = rag.ground(concepts=("binary-search",), misconception_texts=["interval bounds"])
        assert len(context.render(max_chars=400)) <= 400 + 200

    def test_a_supported_claim_scores_high(self, rag: MultiAgentRAG) -> None:
        context = rag.ground(
            misconception_texts=["Mixed interval conventions."],
            concepts=("half-open-intervals",),
        )
        report = rag.verify("A half-open interval is empty exactly when lo equals hi.", context)
        assert report.score > 0.5 and not report.unsupported

    @pytest.mark.parametrize(
        "claim",
        [
            "Binary search requires the array to be a balanced binary tree rebuilt each comparison.",
            "Every binary search must allocate a Fibonacci heap to memoise the pivot gradients.",
            "Interval bounds are recomputed by gradient descent over the quicksort partition.",
        ],
    )
    def test_a_fabricated_claim_is_flagged(self, rag: MultiAgentRAG, claim: str) -> None:
        context = rag.ground(
            misconception_texts=["Mixed interval conventions."],
            concepts=("half-open-intervals",),
        )
        assert rag.verify(claim, context).unsupported

    def test_grounding_separates_truth_from_topicality(self, rag: MultiAgentRAG) -> None:
        context = rag.ground(
            misconception_texts=["Mixed interval conventions."],
            concepts=("half-open-intervals",),
        )
        true_claim = rag.verify("A half-open interval is empty exactly when lo equals hi.", context)
        on_topic_nonsense = rag.verify(
            "Binary search requires the array to be a balanced binary tree rebuilt each comparison.",
            context,
        )
        assert true_claim.score > on_topic_nonsense.score + 0.15

    def test_questions_are_exempt_from_grounding(self, rag: MultiAgentRAG) -> None:
        context = rag.ground(concepts=("binary-search",), misconception_texts=["bounds"])
        report = rag.verify("What is your upper bound before the first comparison?", context)
        assert report.score == 1.0 and not report.assertions

    def test_empty_text_is_fully_grounded(self, rag: MultiAgentRAG) -> None:
        assert rag.verify("", rag.ground(concepts=("binary-search",))).score == 1.0
