"""Diagnose one submission and read the result from code.

    python examples/01_diagnose_a_submission.py
"""

from deductive_mas import DeductiveOrchestrator, knowledge_graph, problem, submission
from deductive_mas.ui.ansi import Style
from deductive_mas.ui.render import render_tutoring_result

SOURCE = """
def lower_bound(a, t):
    lo, hi = 0, len(a) - 1        # <- an inclusive bound ...
    while lo < hi:                # <- ... under an exclusive guard
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
"""


def main():
    from deductive_mas.domain import Submission

    attempt = Submission(sid="demo", problem_id="lower_bound", source=SOURCE)
    result = DeductiveOrchestrator().tutor(problem("lower_bound"), attempt)

    print("\n".join(render_tutoring_result(Style(), result, knowledge_graph())))

    print("\n--- programmatic access " + "-" * 46)
    print("misconception :", result.diagnosis.primary.misconception_id)
    print("belief        :", round(result.diagnosis.confidence, 3))
    print("diverges at   : line", result.diagnosis.divergence.student_line)
    print("witness       :", result.diagnosis.counterexample.args)
    print("root cause    :", result.alignment.root_cause)
    print("leakage       :", result.intervention.leakage, "(gate limit 0.18)")


if __name__ == "__main__":
    main()
