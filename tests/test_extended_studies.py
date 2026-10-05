import json
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest

from deductive_mas.agents.base import Blackboard
from deductive_mas.agents.evaluator import ModelOnlyEvaluator, _naive_fusion
from deductive_mas.agents.orchestrator import DeductiveOrchestrator
from deductive_mas.analysis.belief import MassFunction
from deductive_mas.config import Config
from deductive_mas.data.heldout import (
    HELDOUT_PROBLEMS,
    all_heldout_problems,
    all_heldout_submissions,
    heldout_labelled,
)
from deductive_mas.data.problems import problem
from deductive_mas.data.submissions import submission
from deductive_mas.knowledge.misconceptions import MISCONCEPTIONS
from deductive_mas.knowledge.ontology import knowledge_graph
from deductive_mas.llm.base import Completion, Task
from deductive_mas.llm.registry import Reasoner
from deductive_mas.simulation.benchmark import DiagnosticBenchmark
from deductive_mas.simulation.design import DesignAnalysis, interpolated_response_model, null_response_model
from deductive_mas.simulation.gate_study import GateAudit, GateBenchmark, manufacture
from deductive_mas.simulation.refactory import RefactoryStudy, load_refactory
from deductive_mas.simulation.student import ResponseModel


# held-out bank
class TestHeldOutBank:
    def test_problems_resolve_through_the_shared_lookup(self):
        for spec in all_heldout_problems():
            assert problem(spec.pid) is spec

    @pytest.mark.parametrize("spec", all_heldout_problems(), ids=lambda s: s.pid)
    def test_reference_passes_its_own_tests_and_sampler_respects_validator(self, spec):
        namespace: dict = {}
        exec(spec.reference_solution, namespace)
        fn = namespace[spec.entry_point]
        for case in spec.tests:
            args = [list(a) if isinstance(a, list) else a for a in case.args]
            assert fn(*args) == case.expected, case.label
        rng = Random(3)
        assert all(spec.validator(spec.sampler(rng)) for _ in range(100))

    def test_labels_are_in_the_taxonomy_and_concepts_in_the_graph(self):
        graph = knowledge_graph()
        for sub in all_heldout_submissions():
            assert all(m in MISCONCEPTIONS for m in sub.gold_misconceptions), sub.sid
        for spec in all_heldout_problems():
            assert all(c in graph for c in spec.concepts), spec.pid

    def test_bank_composition(self):
        subs = all_heldout_submissions()
        assert len(subs) == 31 and len(heldout_labelled()) == 25
        assert sum(1 for s in subs if not s.gold_misconceptions) == 6

    def test_no_clean_held_out_submission_is_accused(self, config, orchestrator):
        result = DiagnosticBenchmark(config, orchestrator).run(all_heldout_submissions())
        assert result.false_positive_rate == 0.0
        assert result.report.top1_accuracy >= 0.7   # the frozen rule number is 0.76


# fusion ablations + model only baseline
class TestFusionAblations:
    def test_naive_rules_agree_on_a_single_source(self):
        masses = [("symbolic", MassFunction.from_scores({"m": 0.7, "~absent~": 0.0}, reliability=0.86))]
        for rule in ("max", "mean", "noisy-or"):
            assert abs(_naive_fusion(rule, masses, "m", "~absent~") - 0.7 * 0.86) < 1e-9

    def test_noisy_or_exceeds_max_with_two_sources(self):
        masses = [
            ("symbolic", MassFunction.from_scores({"m": 0.5}, reliability=1.0)),
            ("dynamic", MassFunction.from_scores({"m": 0.5}, reliability=1.0)),
        ]
        assert _naive_fusion("noisy-or", masses, "m", "~absent~") > _naive_fusion("max", masses, "m", "~absent~")

    def test_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            _naive_fusion("median", [("s", MassFunction.from_scores({"m": 0.5}))], "m", "~absent~")

    def test_removing_dynamic_evidence_loses_the_runtime_diagnoses(self, config):
        limited = replace(config, diagnosis=replace(config.diagnosis, sources=("symbolic",), narrate=False))
        diagnosis = DeductiveOrchestrator(limited).diagnose(problem("min_coins"), submission("mc_table_off_by_one"))
        assert "run.index-out-of-range" not in [h.misconception_id for h in diagnosis.hits]

    def test_model_only_baseline_uses_the_backend_verbatim(self, config):
        class _Opinionated:
            name = "opinionated"

            def available(self):
                return True

            def run(self, task: Task) -> Completion:
                assert task.name == "diagnose_raw" and "student_code" in task.payload
                return Completion(backend=self.name, data={
                    "misconceptions": [{"id": "greedy.local-optimum-assumed", "confidence": 0.9, "reason": "greedy"},
                                       {"id": "not.a.real.id", "confidence": 0.9, "reason": "?"}],
                    "summary": "greedy",
                })

        reasoner = Reasoner(replace(config.llm, cache_dir=None), backend=_Opinionated())
        orchestrator = DeductiveOrchestrator(config, reasoner=reasoner)
        evaluator = ModelOnlyEvaluator(config, reasoner)
        diagnosis = orchestrator.diagnose(problem("min_coins"), submission("mc_greedy"), evaluator=evaluator)
        assert [h.misconception_id for h in diagnosis.hits] == ["greedy.local-optimum-assumed"]
        assert diagnosis.divergence is None and diagnosis.backend == "opinionated"

    def test_a_degraded_backend_is_recorded_on_the_diagnosis(self, config):
        class _Down:
            name = "down"

            def available(self):
                return True

            def run(self, task):
                from deductive_mas.errors import BackendError
                raise BackendError("offline for the day")

        reasoner = Reasoner(replace(config.llm, cache_dir=None), backend=_Down())
        diagnosis = DeductiveOrchestrator(config, reasoner=reasoner).diagnose(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert diagnosis.degraded

    def test_a_backend_that_fails_only_on_hints_still_marks_the_row_degraded(self, config):
        """The audit reruns degraded rows, a fallback rung must not count as a model rung."""
        from deductive_mas.llm.offline import OfflineReasoner

        class _HintsDown:
            name = "hintsdown"

            def __init__(self):
                self.inner = OfflineReasoner()

            def available(self):
                return True

            def run(self, task):
                from deductive_mas.errors import BackendError
                if task.name in ("hints", "challenge"):
                    raise BackendError("rate limited")
                completion = self.inner.run(task)
                completion.backend = self.name
                return completion

        reasoner = Reasoner(replace(config.llm, cache_dir=None), backend=_HintsDown())
        result = DeductiveOrchestrator(config, reasoner=reasoner).tutor(problem("lower_bound"), submission("lb_inclusive_bound"))
        assert any("degraded" in w for w in result.telemetry.warnings)
        assert all(record["candidate_backend"] == "offline" for record in result.gate_log)


# gate studies
class TestGateStudies:
    def test_manufactured_candidates_cover_every_category(self):
        candidates = manufacture(problem("lower_bound"), submission("lb_inclusive_bound"))
        categories = {c.category for c in candidates}
        assert {"reference-line", "reference-line-renamed", "full-solution", "prose-repair",
                "student-line", "question", "fallback-rung"} <= categories
        assert all(c.should_reject for c in candidates if c.category.startswith("reference"))
        assert not any(c.should_reject for c in candidates if c.category == "student-line")

    def test_correct_submissions_get_no_leak_categories(self):
        candidates = manufacture(problem("lower_bound"), submission("lb_correct"))
        assert not any(c.should_reject for c in candidates)

    def test_benchmark_on_a_slice_rejects_leaks_and_keeps_quotations(self, config):
        result = GateBenchmark(config).run([submission("lb_inclusive_bound"), submission("mc_greedy")])
        summary = result.summary()
        assert summary["reference-line"]["rejection_rate"] == 1.0
        assert summary["student-line"]["rejection_rate"] == 0.0
        assert summary["question"]["rejection_rate"] == 0.0

    def test_audit_records_every_rung_and_the_challenge(self, config, orchestrator):
        result = GateAudit(config, orchestrator).run([submission("lb_inclusive_bound")])
        items = [r["item"] for r in result.records]
        assert items == ["hint-1", "hint-2", "hint-3", "challenge"]
        assert all(r["final_source"] in ("generated", "template", "sanitised", "fallback") for r in result.records)
        assert result.summary()["hint-1"]["n"] == 1


# Refactory adapter, on a tiny synthetic copy of its layout
class TestRefactoryAdapter:
    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        q = tmp_path / "question_1"
        (q / "ans").mkdir(parents=True)
        (q / "code" / "reference").mkdir(parents=True)
        (q / "code" / "correct").mkdir()
        (q / "code" / "wrong").mkdir()
        (q / "code" / "reference" / "reference.py").write_text(
            "def search(x, seq):\n    for i in range(len(seq)):\n        if x <= seq[i]:\n            return i\n    return len(seq)\n"
        )
        (q / "ans" / "input_001.txt").write_text("search(5, (1, 5, 10))\n")
        (q / "ans" / "output_001.txt").write_text("1\n")
        (q / "ans" / "input_002.txt").write_text("search(100, [])\n")
        (q / "ans" / "output_002.txt").write_text("0\n")
        (q / "code" / "correct" / "correct_1_001.py").write_text(
            "def search(x, seq):\n    for i, e in enumerate(seq):\n        if x <= e:\n            return i\n    return len(seq)\n"
        )
        (q / "code" / "wrong" / "wrong_1_001.py").write_text(
            "def search(x, seq):\n    for i, e in enumerate(seq):\n        if x < e:\n            return i\n    return len(seq)\n"
        )
        return tmp_path

    def test_loader_builds_tasks_with_course_tests(self, corpus):
        tasks = load_refactory(corpus)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.spec.entry_point == "search" and len(task.spec.tests) == 2
        assert {s.functionally_correct for s in task.submissions} == {True, False}

    def test_study_detects_the_wrong_program_and_spares_the_right_one(self, corpus, config):
        study = RefactoryStudy(replace(config, diagnosis=replace(config.diagnosis, narrate=False)))
        payload = study.run(load_refactory(corpus)).to_dict()
        overall = payload["overall"]
        assert overall["n_correct"] == 1 and overall["n_wrong"] == 1
        assert overall["wrong__counterexample_rate"] == 1.0
        assert overall["correct__false_accusation_rate"] == 0.0


# design analysis
class TestDesignAnalysis:
    def test_null_model_removes_every_socratic_advantage(self):
        null = null_response_model()
        assert null.socratic_on_target == null.direct_answer
        assert null.socratic_transfer_retention == null.direct_transfer_retention
        assert null.repair_on_target == null.repair_direct
        assert null.zpd_bonus == 1.0 and null.zpd_penalty == 1.0

    def test_interpolation_recovers_both_ends(self):
        assert interpolated_response_model(1.0) == ResponseModel()
        zero = interpolated_response_model(0.0)
        assert zero.socratic_on_target == null_response_model().socratic_on_target

    def test_a_tiny_run_produces_every_section(self, config):
        result = DesignAnalysis(config).run(seeds=2, null_seeds=2, students=30, strengths=(0.0, 1.0),
                                            cohort_sizes=(30,), power_seeds=2)
        payload = result.to_dict()
        assert payload["replication"]["seeds"] == 2
        assert payload["null_calibration"]["seeds"] == 2
        assert [s["strength"] for s in payload["sensitivity"]] == [0.0, 1.0]
        assert payload["power"][0]["n_per_arm"] == 10
        json.dumps(payload)   # serialisable
