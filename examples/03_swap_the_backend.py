"""What does the model add? Run the same submission through the offline
reasoner and through Claude (if ANTHROPIC_API_KEY is set) and diff the two
reports. Everything else in the pipeline is identical.

    ANTHROPIC_API_KEY=... python examples/03_swap_the_backend.py
"""

import dataclasses
import os

from deductive_mas import Config, DeductiveOrchestrator, problem, submission
from deductive_mas.config import LLMConfig


def run(backend: str):
    config = dataclasses.replace(
        Config(), llm=LLMConfig(backend=backend, cache_dir=None)
    )
    attempt = submission("mc_greedy")
    return DeductiveOrchestrator(config).tutor(problem(attempt.problem_id), attempt)


def describe(label, result):
    primary = result.diagnosis.primary
    print(f"\n=== {label} ({result.telemetry.backend}) " + "=" * 30)
    print("misconception :", primary.misconception_id if primary else "(none)")
    print("belief        :", round(result.diagnosis.confidence, 3))
    print("target        :", result.intervention.target_concept)
    print("leakage       :", round(result.intervention.leakage, 3))
    for hint in result.intervention.hints:
        print(f"  {hint.level}. {hint.text[:96]}")
    for warning in result.telemetry.warnings:
        print("  ! ", warning)


def main():
    describe("offline", run("offline"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        describe("Claude", run("claude"))
    else:
        print("\nSet ANTHROPIC_API_KEY to run the live arm of this comparison.")
        print("The offline arm above is the ablation baseline it is measured against.")


if __name__ == "__main__":
    main()
