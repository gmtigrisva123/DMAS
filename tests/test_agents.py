import dataclasses

import pytest

from deductive_mas.agents.base import Blackboard
from deductive_mas.agents.evaluator import SocraticEvaluatorAgent
from deductive_mas.agents.intervention import CognitiveInterventionAgent
from deductive_mas.agents.orchestrator import DeductiveOrchestrator
from deductive_mas.config import Config, LLMConfig
from deductive_mas.data.problems import problem
from deductive_mas.data.submissions import all_submissions, correct_submissions, labelled, submission
from deductive_mas.domain import HintKind
from deductive_mas.llm.base import Completion, Task
from deductive_mas.llm.registry import Reasoner

REFERENCE_LEAK = (
    "def lower_bound(a, t):\n"
    "    lo, hi = 0, len(a)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if a[mid] < t:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid\n"
    "    return lo\n"
)


class _LeakingBackend:
    """A backend that does exactly what the gate is there to stop."""

    name = "leaking"

    def available(self):
        return True

    def run(self, task: Task) -> Completion:
        if task.name == "hints":
            return Completion(
                backend=self.name,
                data={
                    "hints": [
                        {"kind": "orienting", "text": "Change the initialisation to `hi = len(a)`."},
                        {"kind": "contradiction", "text": "Here is the fix:\n```python\n" + REFERENCE_LEAK + "```"},
                        {"kind": "conceptual", "text": "Just write `lo, hi = 0, len(a)` instead."},
                    ]
                },
            )
        if task.name == "challenge":
            return Completion(
                backend=self.name,
                data={"prompt": "Replace your initialisation with `hi = len(a)` and keep "
                                "`while lo < hi:` as it is.",
                      "trace_question": "", "expected_insight": ""},
            )
        if task.name == "narrate":
            return Completion(backend=self.name, data={"narrative": "It is broken."})
        return Completion(backend=self.name, data={"misconceptions": [], "summary": ""})


class _HallucinatingBackend(_LeakingBackend):
    name = "hallucinating"

    def run(self, task: Task) -> Completion:
        if task.name == "hints":
            return Completion(
                backend=self.name,
                data={
                    "hints": [
                        {"kind": "orienting",
                         "text": "Binary search requires rebuilding a Fibonacci heap on every "
                                 "comparison to keep the pivot gradients balanced."},
                    ]
                },
            )
        if task.name == "diagnose":
            return Completion(
                backend=self.name,
                data={"misconceptions": [{"id": "not-a-real-misconception", "confidence": 0.9,
                                          "reason": "invented"}], "summary": ""},
            )
        return super().run(task)


@pytest.fixture(scope="module")
def evaluator(config):
    return SocraticEvaluatorAgent(config)


def _board(sid):
    sub = submission(sid)
    return Blackboard(problem=problem(sub.problem_id), submission=sub)


class TestSocraticEvaluator:
    def test_recovers_every_gold_label(self, evaluator):
        misses = []
        for sub in labelled():
            board = evaluator(Blackboard(problem=problem(sub.problem_id), submission=sub))
            found = {hit.misconception_id for hit in board.diagnosis.hits}
            if not (set(sub.gold_misconceptions) & found):
                misses.append(sub.sid)
        assert misses == []

    def test_ranks_a_gold_label_first(self, evaluator):
        wrong = []
        for sub in labelled():
            board = evaluator(Blackboard(problem=problem(sub.problem_id), submission=sub))
            primary = board.diagnosis.primary
            if not primary or primary.misconception_id not in sub.gold_misconceptions:
                wrong.append(sub.sid)
        assert wrong == []

    def test_never_accuses_a_correct_submission(self, evaluator):
        flagged = []
        for sub in correct_submissions():
            board = evaluator(Blackboard(problem=problem(sub.problem_id), submission=sub))
            if board.diagnosis.hits:
                flagged.append((sub.sid, [h.misconception_id for h in board.diagnosis.hits]))
        assert flagged == []

    def test_localises_the_divergence_for_a_wrong_answer(self, evaluator):
        board = evaluator(_board("lb_inclusive_bound"))
        assert board.diagnosis.divergence is not None
        assert board.diagnosis.divergence.student_line

    def test_attaches_a_minimal_counterexample(self, evaluator):
        board = evaluator(_board("lb_inclusive_bound"))
        counterexample = board.diagnosis.counterexample
        assert counterexample is not None and len(counterexample.args[0]) <= 2

    def test_reports_a_complexity_gap(self, evaluator):
        board = evaluator(_board("ts_quadratic"))
        assert board.diagnosis.complexity_gap
        assert board.diagnosis.student_complexity != board.diagnosis.reference_complexity

    def test_evidence_carries_multiple_independent_kinds(self, evaluator):
        board = evaluator(_board("lb_inclusive_bound"))
        assert len(board.diagnosis.primary.kinds) >= 2

    def test_simultaneous_misconceptions_are_not_forced_to_compete(self, evaluator):
        board = evaluator(_board("hc_list_queue"))
        found = {hit.misconception_id for hit in board.diagnosis.hits}
        assert {"graph.list-as-queue", "graph.list-as-visited"} <= found

    def test_an_unparsable_submission_does_not_crash(self, evaluator, config):
        from deductive_mas.domain import Submission

        board = Blackboard(
            problem=problem("lower_bound"),
            submission=Submission(sid="broken", problem_id="lower_bound", source="def f(:\n"),
        )
        evaluator(board)
        assert board.diagnosis is not None and board.notes

    def test_a_hallucinated_misconception_is_discarded(self, config):
        agent = SocraticEvaluatorAgent(
            config, Reasoner(config.llm, backend=_HallucinatingBackend())
        )
        board = agent(_board("lb_inclusive_bound"))
        assert all(
            hit.misconception_id != "not-a-real-misconception" for hit in board.diagnosis.hits
        )
        assert any("unknown misconception" in w for w in board.telemetry.warnings)

    def test_diagnosis_is_deterministic(self, evaluator):
        first = evaluator(_board("mc_transition_order")).diagnosis
        second = evaluator(_board("mc_transition_order")).diagnosis
        assert [h.misconception_id for h in first.hits] == [h.misconception_id for h in second.hits]
        assert first.confidence == pytest.approx(second.confidence)


class TestKnowledgeGraphAlignment:
    def test_locates_a_root_cause_inside_the_graph(self, orchestrator, graph):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.alignment.root_cause in graph

    def test_blame_is_a_distribution_over_implicated_concepts(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert sum(result.alignment.blame.values()) == pytest.approx(1.0)

    def test_the_diagnosed_concept_carries_the_most_blame(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        top = result.alignment.top_blame(1)[0][0]
        assert top in ("half-open-intervals", "boundary-conditions", "binary-search")

    def test_mastery_falls_on_the_implicated_concepts(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        dropped = [
            cid for cid, after in result.alignment.mastery_after.items()
            if after < result.alignment.mastery_before.get(cid, 1.0)
        ]
        assert dropped

    def test_the_root_cause_is_not_the_curriculum_floor(self, orchestrator, graph):
        # someone who wrote a working loop knows what a variable is. If the
        # diagnosis lands on var-binding for an interval error the blame cascade
        # ran away.
        offenders = []
        for sid in ("lb_inclusive_bound", "lb_no_progress", "hc_mark_on_dequeue",
                    "mc_transition_order", "mi_sort_by_start"):
            sub = submission(sid)
            result = orchestrator.tutor(problem(sub.problem_id), sub)
            if graph.depth(result.alignment.root_cause or "var-binding") == 0:
                offenders.append((sid, result.alignment.root_cause))
        assert offenders == []

    def test_a_learning_path_is_offered(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.alignment.prerequisite_paths
        assert all(len(path) > 1 for path in result.alignment.prerequisite_paths)


class TestCognitiveIntervention:
    def test_produces_a_full_socratic_ladder(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        hints = result.intervention.hints
        assert len(hints) == 3
        assert [h.level for h in hints] == [1, 2, 3]
        assert hints[0].kind is HintKind.ORIENTING

    def test_guidance_cites_the_concrete_counterexample(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        joined = " ".join(h.text for h in result.intervention.hints)
        assert repr(result.diagnosis.counterexample.args) in joined

    def test_a_counter_factual_challenge_is_offered(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.intervention.challenge is not None
        assert result.intervention.challenge.prompt

    def test_guidance_is_grounded_in_cited_evidence(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.intervention.citations

    def test_the_target_lies_in_or_near_the_zpd(self, orchestrator):
        outside = []
        for sub in labelled():
            result = orchestrator.tutor(problem(sub.problem_id), sub)
            probability = result.intervention.zpd_probability
            if not 0.30 <= probability <= 0.99:
                outside.append((sub.sid, probability))
        assert outside == []


class TestComplianceGate:
    """Nothing reaches the student without passing the gate."""

    def test_no_guidance_in_the_whole_bank_leaks(self, orchestrator, config):
        offenders = []
        for sub in all_submissions():
            result = orchestrator.tutor(problem(sub.problem_id), sub)
            if result.intervention.leakage > config.intervention.max_leakage:
                offenders.append((sub.sid, result.intervention.leakage))
        assert offenders == []

    def test_a_leaking_backend_is_overridden_not_trusted(self, config):
        reasoner = Reasoner(config.llm, backend=_LeakingBackend())
        orchestrator = DeductiveOrchestrator(config, reasoner=reasoner)
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))

        assert result.intervention.leakage <= config.intervention.max_leakage
        assert result.intervention.regenerations >= 1
        joined = " ".join(h.text for h in result.intervention.hints)
        assert "len(a)" not in joined or "?" in joined
        assert "```" not in joined
        assert any("replaced" in w for w in result.telemetry.warnings)

    def test_a_leaking_challenge_is_replaced(self, config):
        reasoner = Reasoner(config.llm, backend=_LeakingBackend())
        orchestrator = DeductiveOrchestrator(config, reasoner=reasoner)
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.intervention.challenge is not None
        # the fix is hi = len(a). 'while lo < hi' is the student's own line and
        # quoting it leaks nothing, so it is the fix that must not show up.
        assert "len(a)" not in result.intervention.challenge.prompt

    def test_a_hallucinating_backend_is_overridden(self, config):
        reasoner = Reasoner(config.llm, backend=_HallucinatingBackend())
        orchestrator = DeductiveOrchestrator(config, reasoner=reasoner)
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        joined = " ".join(h.text for h in result.intervention.hints)
        assert "Fibonacci heap" not in joined
        assert result.intervention.hints

    def test_the_gate_still_yields_usable_guidance(self, config):
        reasoner = Reasoner(config.llm, backend=_LeakingBackend())
        orchestrator = DeductiveOrchestrator(config, reasoner=reasoner)
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert len(result.intervention.hints) == 3
        assert all(len(h.text) > 40 for h in result.intervention.hints)


class TestOrchestrator:
    def test_a_full_session_is_serialisable(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        payload = result.to_dict()
        assert payload["diagnosis"]["primary"] == "bs.interval-convention-mismatch"
        assert payload["intervention"]["hints"]
        assert payload["telemetry"]["backend"]

    def test_telemetry_records_every_stage(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        stages = {name for name, _ in result.telemetry.stages}
        assert {"socratic-evaluator", "kg-alignment", "cognitive-intervention"} <= stages

    def test_a_failing_stage_does_not_abort_the_session(self, config):
        orchestrator = DeductiveOrchestrator(config)

        class _Exploding:
            name = "kg-alignment"

            def __call__(self, board):
                raise RuntimeError("stage failure")

        orchestrator.alignment = _Exploding()
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert result.diagnosis.primary is not None
        assert any("kg-alignment failed" in w for w in result.telemetry.warnings)

    def test_diagnose_only_shortcut(self, orchestrator):
        diagnosis = orchestrator.diagnose(problem("lower_bound"), submission("lb_no_progress"))
        assert diagnosis.primary.misconception_id == "bs.no-progress-update"

    def test_a_correct_submission_is_reported_as_such(self, orchestrator):
        result = orchestrator.tutor(problem("lower_bound"), submission("lb_correct"))
        assert result.diagnosis.hits == ()
        assert result.diagnosis.inconclusive

    def test_sessions_are_reproducible(self, config):
        first = DeductiveOrchestrator(config).tutor(
            problem("hop_counts"), submission("hc_mark_on_dequeue")
        )
        second = DeductiveOrchestrator(config).tutor(
            problem("hop_counts"), submission("hc_mark_on_dequeue")
        )
        assert first.to_dict()["diagnosis"] == second.to_dict()["diagnosis"]
        assert first.to_dict()["intervention"] == second.to_dict()["intervention"]

    def test_prior_mastery_is_honoured(self, config, graph):
        orchestrator = DeductiveOrchestrator(config)
        state = orchestrator.tracker.initial(ability=2.5)
        result = orchestrator.tutor(
            problem("lower_bound"), submission("lb_inclusive_bound"), mastery=state
        )
        assert result.alignment.mastery_before
