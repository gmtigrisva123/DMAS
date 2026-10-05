"""Diagnose a whole cohort and summarise what the class gets wrong (the
instructor's view).

    python examples/02_grade_a_cohort.py
"""

from collections import Counter

from deductive_mas import DeductiveOrchestrator, all_submissions, knowledge_graph, problem
from deductive_mas.knowledge.misconceptions import MISCONCEPTIONS


def main():
    orchestrator = DeductiveOrchestrator()
    graph = knowledge_graph()

    misconceptions: Counter = Counter()
    root_causes: Counter = Counter()
    clean = 0

    for attempt in all_submissions():
        result = orchestrator.tutor(problem(attempt.problem_id), attempt)
        primary = result.diagnosis.primary
        if primary is None:
            clean += 1
            continue
        misconceptions[primary.misconception_id] += 1
        if result.alignment.root_cause:
            root_causes[result.alignment.root_cause] += 1

    total = len(all_submissions())
    print(f"cohort of {total}: {clean} clean, {total - clean} carrying a misconception\n")

    print("most common flawed beliefs")
    for mid, count in misconceptions.most_common(6):
        entry = MISCONCEPTIONS[mid]
        print(f"  {count:2d}x  {entry.name}")
        print(f"       “{entry.student_voice}”")

    print("\nwhere to teach next (deficiency frontier across the cohort)")
    for cid, count in root_causes.most_common(5):
        print(f"  {count:2d}x  {graph.name(cid):32s} depth {graph.depth(cid)}")


if __name__ == "__main__":
    main()
