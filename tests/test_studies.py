import pytest

from deductive_mas.data.submissions import all_submissions, labelled
from deductive_mas.simulation.benchmark import DiagnosticBenchmark
from deductive_mas.simulation.experiment import Experiment
from deductive_mas.simulation.robustness import COMPOSITE, RobustnessStudy, TRANSFORMS, mutate
from deductive_mas.simulation.student import ResponseModel, StudentPopulation


class TestRQ1Benchmark:
    @pytest.fixture(scope="class")
    def result(self, config, orchestrator):
        return DiagnosticBenchmark(config, orchestrator).run()

    def test_every_labelled_submission_is_scored(self, result):
        assert result.report.support == len(labelled())

    def test_top_one_accuracy_is_complete(self, result):
        assert result.report.top1_accuracy == 1.0

    def test_no_correct_submission_is_accused(self, result):
        assert result.false_positive_rate == 0.0

    def test_agreement_is_chance_corrected_and_high(self, result):
        assert result.report.kappa > 0.9

    def test_it_beats_a_test_runner(self, result):
        assert result.report.top1_accuracy > result.baseline_top1

    def test_most_failures_are_localised_to_a_line(self, result):
        assert result.localisation_rate > 0.5

    def test_latency_is_interactive(self, result):
        assert result.mean_seconds < 1.0

    def test_the_caveat_travels_with_the_numbers(self, result):
        payload = result.to_dict()
        assert "generalis" in payload["caveat"] or "generaliz" in payload["caveat"]

    def test_per_label_scores_are_reported(self, result):
        assert result.to_dict()["per_label"]


class TestRQ3Robustness:
    @pytest.fixture(scope="class")
    def result(self, config, orchestrator):
        return RobustnessStudy(config, orchestrator).run(
            submissions=labelled()[:8],
            transforms=["rename-identifiers", "split-assignments", "inject-noise", COMPOSITE],
        )

    def test_every_transformation_produces_verified_mutants(self, result):
        assert result.overall()["n"] > 0
        assert all(outcome.verified for outcome in result.outcomes)

    def test_diagnosis_survives_renaming(self, result):
        stability = result.by_transform()["rename-identifiers"]["top1_stability"]
        assert stability >= 0.8

    def test_overall_stability_is_high(self, result):
        assert result.overall()["top1_stability"] >= 0.85

    def test_regressions_are_reported_not_hidden(self, result):
        payload = result.to_dict()
        assert "regressions" in payload
        stable = payload["overall"]["top1_stability"]
        assert (stable == 1.0) == (payload["regressions"] == [])

    @pytest.mark.parametrize("name", sorted(TRANSFORMS))
    def test_each_transformation_preserves_parseability(self, name):
        import random

        source = all_submissions()[0].source
        mutated = mutate(source, name, random.Random(1))
        assert mutated is not None
        compile(mutated, "<mutant>", "exec")

    def test_renaming_actually_renames(self):
        import random

        source = "def f(alpha, beta):\n    gamma = alpha + beta\n    return gamma\n"
        mutated = mutate(source, "rename-identifiers", random.Random(1))
        assert "alpha" not in mutated and "def f(" in mutated   # entry point survives

    def test_an_unknown_transformation_is_refused(self):
        import random

        assert mutate("def f():\n    pass\n", "teleport", random.Random(1)) is None

    def test_unparsable_source_is_refused(self):
        import random

        assert mutate("def f(:\n", COMPOSITE, random.Random(1)) is None


class TestRQ2Experiment:
    @pytest.fixture(scope="class")
    def result(self, config):
        return Experiment(config).run(students=48, seed=20260909, sessions=2)

    def test_all_three_arms_are_populated(self, result):
        assert set(result.arms) == {"control-direct", "treatment-mas", "ablation-no-kg"}
        assert all(arm.n > 0 for arm in result.arms.values())

    def test_randomisation_balances_the_arms(self, result):
        sizes = [arm.n for arm in result.arms.values()]
        assert max(sizes) - min(sizes) <= 2

    def test_the_system_aims_at_the_right_concept_far_more_often(self, result):
        treatment = result.arms["treatment-mas"].summary()["on_target_rate"]
        ablation = result.arms["ablation-no-kg"].summary()["on_target_rate"]
        assert treatment > ablation + 0.2

    def test_only_the_control_arm_discloses_the_answer(self, result):
        assert result.arms["treatment-mas"].summary()["max_leakage"] < 0.18
        assert result.arms["control-direct"].summary()["max_leakage"] == 1.0

    def test_comparisons_report_both_a_test_and_an_effect_size(self, result):
        assert result.comparisons
        for comparison in result.comparisons:
            row = comparison.as_row()
            assert "t(" in row["welch"] and "Hedges" in row["hedges_g"]

    def test_multiple_comparisons_are_corrected(self, result):
        assert result.corrections
        assert all(c.adjusted >= c.raw for c in result.corrections)

    def test_power_is_reported_alongside_the_effect(self, result):
        assert {"n_per_arm", "observed_effect", "achieved_power"} <= set(result.power)

    def test_the_simulation_disclaimer_is_attached(self, result):
        assert "SIMULATED" in result.to_dict()["disclaimer"]

    def test_the_experiment_is_reproducible(self, config):
        a = Experiment(config).run(students=24, seed=7, sessions=1)
        b = Experiment(config).run(students=24, seed=7, sessions=1)
        assert a.arms["treatment-mas"].retention == b.arms["treatment-mas"].retention


class TestStudentModel:
    def test_cohorts_are_reproducible(self, graph, config):
        from deductive_mas.knowledge.mastery import MasteryTracker

        population = StudentPopulation(graph, MasteryTracker(graph, config.knowledge))
        pool = ["bs.interval-convention-mismatch", "rec.missing-base-case"]
        a = population.sample(10, seed=3, misconception_pool=pool)
        b = population.sample(10, seed=3, misconception_pool=pool)
        assert [s.ability for s in a] == [s.ability for s in b]

    def test_misconceptions_depress_the_relevant_concepts(self, graph, config):
        from deductive_mas.knowledge.mastery import MasteryTracker

        population = StudentPopulation(graph, MasteryTracker(graph, config.knowledge))
        student = population.sample(
            1, seed=1, misconception_pool=["bs.interval-convention-mismatch"]
        )[0]
        assert student.mastery.get("half-open-intervals") < 0.3

    def test_an_empty_pool_is_refused(self, graph, config):
        from deductive_mas.knowledge.mastery import MasteryTracker

        population = StudentPopulation(graph, MasteryTracker(graph, config.knowledge))
        with pytest.raises(ValueError):
            population.sample(3, seed=1, misconception_pool=[])

    def test_on_target_socratic_guidance_teaches_more_than_a_direct_answer(self, graph, config):
        import random

        from deductive_mas.knowledge.mastery import MasteryTracker

        population = StudentPopulation(graph, MasteryTracker(graph, config.knowledge))
        pool = ["bs.interval-convention-mismatch"]
        socratic = population.sample(1, seed=5, misconception_pool=pool)[0]
        direct = population.sample(1, seed=5, misconception_pool=pool)[0]

        good = population.apply_intervention(
            socratic, misconception_id=pool[0], diagnosed=True,
            target_concept="half-open-intervals", zpd_probability=0.6, socratic=True,
            rng=random.Random(1),
        )
        plain = population.apply_intervention(
            direct, misconception_id=pool[0], diagnosed=False,
            target_concept=None, zpd_probability=0.0, socratic=False,
            rng=random.Random(1),
        )
        assert good["gain"] > plain["gain"]
        assert good["transfer_retention"] > plain["transfer_retention"]

    def test_the_response_model_is_an_explicit_parameter_set(self):
        model = ResponseModel()
        assert model.socratic_on_target > model.direct_answer
        assert model.socratic_transfer_retention > model.direct_transfer_retention
