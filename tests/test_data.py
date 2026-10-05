import dataclasses
import random

import pytest

from deductive_mas.analysis.counterexample import CounterexampleSearch
from deductive_mas.analysis.tracer import GuardedRunner
from deductive_mas.config import ExecutionConfig
from deductive_mas.data.problems import all_problems, problem
from deductive_mas.data.submissions import (
    all_submissions,
    correct_submissions,
    labelled,
    submission,
)
from deductive_mas.knowledge.misconceptions import MISCONCEPTIONS


@pytest.fixture(scope="module")
def searcher():
    config = dataclasses.replace(ExecutionConfig(), max_steps=12_000, max_seconds=0.4)
    return CounterexampleSearch(GuardedRunner(config))


class TestProblemBank:
    def test_bank_is_populated(self):
        assert len(all_problems()) >= 6

    def test_unknown_problem_raises_with_a_helpful_message(self):
        with pytest.raises(KeyError) as info:
            problem("nope")
        assert "known problems" in str(info.value)

    @pytest.mark.parametrize("spec", all_problems(), ids=lambda s: s.pid)
    def test_reference_solution_passes_its_own_tests(self, spec):
        namespace = GuardedRunner().compile_module(spec.reference_solution)
        function = namespace[spec.entry_point]
        for test in spec.tests:
            args = [list(a) if isinstance(a, list) else a for a in test.args]
            assert function(*args) == test.expected, test.label

    @pytest.mark.parametrize("spec", all_problems(), ids=lambda s: s.pid)
    def test_concepts_exist_in_the_ontology(self, spec, graph):
        assert all(concept in graph for concept in spec.concepts)

    @pytest.mark.parametrize("spec", all_problems(), ids=lambda s: s.pid)
    def test_sampler_always_satisfies_the_validator(self, spec):
        rng = random.Random(3)
        for _ in range(120):
            assert spec.validator(spec.sampler(rng))

    @pytest.mark.parametrize("spec", all_problems(), ids=lambda s: s.pid)
    def test_declared_tests_satisfy_the_validator(self, spec):
        assert all(spec.validator(test.args) for test in spec.tests)

    @pytest.mark.parametrize("spec", all_problems(), ids=lambda s: s.pid)
    def test_boundary_cases_are_covered(self, spec):
        assert len(spec.tests) >= 4


class TestSubmissionBank:
    def test_bank_contains_clean_and_flawed_work(self):
        assert len(labelled()) >= 20
        assert len(correct_submissions()) >= 5

    def test_unknown_submission_raises(self):
        with pytest.raises(KeyError):
            submission("nope")

    def test_gold_labels_are_in_the_taxonomy(self):
        unknown = [
            label for s in all_submissions()
            for label in s.gold_misconceptions
            if label not in MISCONCEPTIONS
        ]
        assert unknown == []

    def test_every_submission_names_a_real_problem(self):
        assert all(problem(s.problem_id) for s in all_submissions())

    def test_clean_submissions_carry_no_labels(self):
        assert all(not s.gold_misconceptions for s in correct_submissions())
        assert all(s.functionally_correct for s in correct_submissions())

    @pytest.mark.parametrize("sub", all_submissions(), ids=lambda s: s.sid)
    def test_declared_correctness_matches_observed_behaviour(self, sub, searcher):
        spec = problem(sub.problem_id)
        outcome = searcher.search(spec, sub.source, rng=random.Random(11), random_attempts=45)
        assert outcome.found == (not sub.functionally_correct), (
            f"{sub.sid}: declared functionally_correct={sub.functionally_correct} "
            f"but the differential tester {'found' if outcome.found else 'found no'} counterexample"
        )
