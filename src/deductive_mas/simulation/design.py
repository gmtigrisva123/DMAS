"""Design analysis for the RQ2 study.

The cohort is simulated and one simulated cohort proves little (the response
model encodes the assumption, so one run just reproduces it). What a
simulation can legitimately show is done here:

- replication: same design, many random cohorts -> sampling distribution of
  the effect sizes and how often the family wise procedure declares an effect
- null calibration: response model made indifferent between arms. Anything
  "found" then is a false positive, so the rejection rate must be at the
  nominal level. This checks the stats / stratified assignment / reporting
  do not manufacture results.
- sensitivity: sweep the assumed Socratic advantage from 0 to the default.
  The effect should vanish at the null and grow monotonically, while the
  on-target rate does not move (it depends on the diagnosis, not on the
  response model)
- power: empirical detection rate across cohort sizes next to the analytic
  power curve, to size a human study before recruiting
"""

import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.orchestrator import DeductiveOrchestrator
from ..config import Config
from ..stats.effect import hedges_g
from ..stats.power import two_sample_power
from ..stats.tests import mean, stdev
from .baselines import DeductiveTutor, DirectAnswerTutor, UntargetedSocraticTutor
from .experiment import Experiment
from .student import ResponseModel


def null_response_model() -> ResponseModel:
    """Response model where the arms cannot differ."""
    base = ResponseModel()
    return ResponseModel(
        socratic_on_target=base.direct_answer,
        socratic_off_target=base.direct_answer,
        zpd_bonus=1.0,
        zpd_penalty=1.0,
        direct_answer=base.direct_answer,
        direct_transfer_retention=base.direct_transfer_retention,
        socratic_transfer_retention=base.direct_transfer_retention,
        repair_on_target=base.repair_direct,
        repair_direct=base.repair_direct,
        decay=base.decay,
    )


def interpolated_response_model(strength: float) -> ResponseModel:
    """Blend the null model (0) into the default model (1)."""
    null, full = null_response_model(), ResponseModel()

    def mix(a: float, b: float) -> float:
        return a + strength * (b - a)

    return ResponseModel(
        socratic_on_target=mix(null.socratic_on_target, full.socratic_on_target),
        socratic_off_target=mix(null.socratic_off_target, full.socratic_off_target),
        zpd_bonus=mix(null.zpd_bonus, full.zpd_bonus),
        zpd_penalty=mix(null.zpd_penalty, full.zpd_penalty),
        direct_answer=full.direct_answer,
        direct_transfer_retention=full.direct_transfer_retention,
        socratic_transfer_retention=mix(null.socratic_transfer_retention, full.socratic_transfer_retention),
        repair_on_target=mix(null.repair_on_target, full.repair_on_target),
        repair_direct=full.repair_direct,
        decay=full.decay,
    )


@dataclass
class SeedOutcome:
    seed: int
    students: int
    retention_g: float
    repeat_error_g: float
    mastery_g: float
    retention_p: float
    repeat_error_p: float
    mastery_p: float
    any_rejected: bool
    rejected: Tuple[str, ...]
    on_target: Dict[str, float]
    in_zpd: Dict[str, float]
    retention_means: Dict[str, float]
    repeat_error_means: Dict[str, float]


@dataclass
class DesignResult:
    replication: List[SeedOutcome] = field(default_factory=list)
    null: List[SeedOutcome] = field(default_factory=list)
    sensitivity: List[Dict[str, Any]] = field(default_factory=list)
    power: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _describe(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "mean": round(mean(list(values)), 4),
            "sd": round(stdev(list(values)), 4) if len(values) > 1 else 0.0,
            "q05": round(ordered[max(0, int(0.05 * (len(ordered) - 1)))], 4),
            "q50": round(ordered[int(0.5 * (len(ordered) - 1))], 4),
            "q95": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        }

    def _summarise(self, outcomes: Sequence[SeedOutcome]) -> Dict[str, Any]:
        if not outcomes:
            return {}
        arms = sorted({arm for o in outcomes for arm in o.on_target})
        return {
            "seeds": len(outcomes),
            "students": outcomes[0].students,
            "retention_g": self._describe([o.retention_g for o in outcomes]),
            "repeat_error_g": self._describe([o.repeat_error_g for o in outcomes]),
            "mastery_g": self._describe([o.mastery_g for o in outcomes]),
            "family_wise_rejection_rate": round(
                sum(1 for o in outcomes if o.any_rejected) / len(outcomes), 4
            ),
            "retention_rejection_rate": round(
                sum(1 for o in outcomes if "treatment-mas/knowledge retention" in o.rejected) / len(outcomes), 4
            ),
            "repeat_error_rejection_rate": round(
                sum(1 for o in outcomes if "treatment-mas/recurring error rate" in o.rejected) / len(outcomes), 4
            ),
            "on_target": {arm: self._describe([o.on_target[arm] for o in outcomes]) for arm in arms},
            "in_zpd": {arm: self._describe([o.in_zpd[arm] for o in outcomes]) for arm in arms},
            "retention_means": {arm: self._describe([o.retention_means[arm] for o in outcomes]) for arm in arms},
            "repeat_error_means": {arm: self._describe([o.repeat_error_means[arm] for o in outcomes]) for arm in arms},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replication": self._summarise(self.replication),
            "null_calibration": self._summarise(self.null),
            "sensitivity": self.sensitivity,
            "power": self.power,
            "per_seed": {
                "replication": [o.__dict__ for o in self.replication],
                "null": [o.__dict__ for o in self.null],
            },
        }


class DesignAnalysis:
    """Runs the four analyses above on the RQ2 harness."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.orchestrator = DeductiveOrchestrator(self.config)

    def _one(self, seed: int, students: int, model: Optional[ResponseModel]) -> SeedOutcome:
        experiment = Experiment(self.config, orchestrator=self.orchestrator, response_model=model)
        result = experiment.run(students=students, seed=seed, sessions=2, include_ablation=True)
        control = result.arms[DirectAnswerTutor.arm]
        treatment = result.arms[DeductiveTutor.arm]
        comparisons = {
            (c.treatment, c.measure): c for c in result.comparisons if c.treatment == DeductiveTutor.arm
        }
        rejected = tuple(a.label for a in result.corrections if a.rejected)

        def g(measure: str) -> float:
            comparison = comparisons.get((DeductiveTutor.arm, measure))
            return comparison.effect.value if comparison is not None else 0.0

        def p(measure: str) -> float:
            comparison = comparisons.get((DeductiveTutor.arm, measure))
            return comparison.parametric.p_value if comparison is not None else 1.0

        return SeedOutcome(
            seed=seed,
            students=students,
            retention_g=g("knowledge retention"),
            repeat_error_g=g("recurring error rate"),
            mastery_g=g("mastery gain"),
            retention_p=p("knowledge retention"),
            repeat_error_p=p("recurring error rate"),
            mastery_p=p("mastery gain"),
            any_rejected=any(a.rejected and a.label.startswith(DeductiveTutor.arm) for a in result.corrections),
            rejected=rejected,
            on_target={name: arm.summary()["on_target_rate"] for name, arm in result.arms.items()},
            in_zpd={name: arm.summary()["in_zpd_rate"] for name, arm in result.arms.items()},
            retention_means={name: arm.summary()["retention_mean"] for name, arm in result.arms.items()},
            repeat_error_means={name: arm.summary()["repeat_error_mean"] for name, arm in result.arms.items()},
        )

    def run(
        self,
        *,
        seeds: int = 20,
        null_seeds: int = 100,
        students: int = 180,
        strengths: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        cohort_sizes: Sequence[int] = (60, 120, 180, 240),
        power_seeds: int = 20,
        base_seed: int = 20260909,
        progress=None,
    ) -> DesignResult:
        result = DesignResult()
        rng = random.Random(base_seed)
        replication_seeds = [rng.randrange(1, 2**31) for _ in range(seeds)]
        null_seed_list = [rng.randrange(1, 2**31) for _ in range(null_seeds)]
        power_seed_list = [rng.randrange(1, 2**31) for _ in range(power_seeds)]

        for index, seed in enumerate(replication_seeds):
            result.replication.append(self._one(seed, students, None))
            if progress:
                progress(f"replication {index + 1}/{seeds}")
        null_model = null_response_model()
        for index, seed in enumerate(null_seed_list):
            result.null.append(self._one(seed, students, null_model))
            if progress and (index + 1) % 10 == 0:
                progress(f"null calibration {index + 1}/{null_seeds}")
        for strength in strengths:
            model = interpolated_response_model(strength)
            outcomes = [self._one(seed, students, model) for seed in replication_seeds[: max(5, seeds // 2)]]
            result.sensitivity.append(
                {
                    "strength": strength,
                    "retention_g": self._describe([o.retention_g for o in outcomes]),
                    "repeat_error_g": self._describe([o.repeat_error_g for o in outcomes]),
                    "family_wise_rejection_rate": round(
                        sum(1 for o in outcomes if o.any_rejected) / len(outcomes), 4
                    ),
                    "on_target_treatment": self._describe([o.on_target[DeductiveTutor.arm] for o in outcomes]),
                    "on_target_ablation": self._describe([o.on_target[UntargetedSocraticTutor.arm] for o in outcomes]),
                }
            )
            if progress:
                progress(f"sensitivity strength={strength}")
        for size in cohort_sizes:
            outcomes = [self._one(seed, size, None) for seed in power_seed_list]
            observed = mean([abs(o.retention_g) for o in outcomes])
            result.power.append(
                {
                    "students": size,
                    "n_per_arm": size // 3,
                    "empirical_power_retention": round(
                        sum(1 for o in outcomes if "treatment-mas/knowledge retention" in o.rejected) / len(outcomes), 4
                    ),
                    "empirical_power_repeat_error": round(
                        sum(1 for o in outcomes if "treatment-mas/recurring error rate" in o.rejected) / len(outcomes), 4
                    ),
                    "mean_abs_retention_g": round(observed, 4),
                    "analytic_power_at_mean_g": round(two_sample_power(observed, size // 3), 4),
                }
            )
            if progress:
                progress(f"power cohort={size}")
        return result

    _describe = staticmethod(DesignResult._describe)
