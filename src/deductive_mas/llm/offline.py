"""The offline reasoner.

NOT a language model. A rule based text generator that implements the same
task interface as the live backends. Reasons it exists:

1. the system must always run, also without network / api key
2. experiments must be reproducible on any machine
3. it is the ablation baseline: full pipeline vs this = what the model adds

Its hints are templated from the structured diagnosis (divergence point,
shrunk counterexample, frontier) so they are still concrete. It does not
invent diagnostic evidence: where the symbolic and dynamic layers are silent
it stays silent.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import Completion, Task

# Socratic probes for the concepts students most often get stuck on
CONCEPT_PROBES: Mapping[str, str] = {
    "half-open-intervals": (
        "If your interval is written [lo, hi), how many elements does it contain, and which "
        "single value of hi makes it empty?"
    ),
    "boundary-conditions": (
        "Which inputs sit exactly on the boundary of your loop's range, and what does your code "
        "do on each of them?"
    ),
    "loop-invariant": (
        "Write down the property you believe holds at the top of every iteration. Does it still "
        "hold immediately after your update lines run?"
    ),
    "loop-termination": (
        "Name a non-negative quantity that must get strictly smaller on every iteration. Does "
        "yours always decrease, or can it stay the same?"
    ),
    "base-case": (
        "What is the smallest input your function can be given, and which line answers it "
        "without calling itself again?"
    ),
    "recursive-decomposition": (
        "On which strictly smaller instance does each recursive call operate, and how do you "
        "know it always gets closer to your base case?"
    ),
    "overlapping-subproblems": (
        "How many *distinct* arguments can your function ever be called with, and how many calls "
        "does it actually make?"
    ),
    "memoisation": (
        "If the same argument arrives twice, what does your function do the second time — and "
        "what would you like it to do?"
    ),
    "dp-transition-order": (
        "Draw an arrow from each table entry to the entries it reads. Does your loop order ever "
        "read an entry it has not yet written?"
    ),
    "visited-set": (
        "Between discovering a vertex and processing it, how many times could it be added to "
        "your worklist?"
    ),
    "bfs": (
        "At the moment you take a vertex out of the queue, what do you already know about its "
        "distance from the source?"
    ),
    "cost-model": (
        "Take the single most frequent operation in your inner loop. What does it cost on the "
        "container you chose?"
    ),
    "asymptotic-notation": (
        "Multiply your loop's trip count by the cost of its body. What growth rate does that give?"
    ),
    "sequence-indexing": (
        "For a sequence of length n, which index values are valid, and what is the largest index "
        "your code can produce?"
    ),
    "mutability-aliasing": (
        "Draw the names in your function and the objects they point at. How many objects are "
        "there really?"
    ),
    "integer-arithmetic": (
        "What type does your index expression evaluate to, and what types may be used to index a "
        "sequence?"
    ),
    "greedy-choice-property": (
        "Assume some optimal answer disagrees with your first choice. Can you always swap your "
        "choice in without making it worse?"
    ),
    "monotone-predicate": (
        "Is the property you are testing false for a prefix of the range and true for the rest? "
        "If not, what breaks?"
    ),
    "iteration-mutation-safety": (
        "While the loop is running, does the container it walks over change length? What does the "
        "loop's position mean after that?"
    ),
    "hash-table": (
        "What does a membership test cost on your container, and what would it cost on one that "
        "hashes its elements?"
    ),
    "queue-deque": (
        "Removing the element at the front — how many other elements have to move?"
    ),
}

_GENERIC_PROBE = (
    "State the defining property of {concept} in one sentence, then check line by line whether "
    "your code actually maintains it."
)


class OfflineReasoner:
    """Deterministic implementation of the reasoning tasks."""

    name = "offline"

    def available(self) -> bool:
        return True

    # dispatch
    def run(self, task: Task) -> Completion:
        handler = {
            "diagnose": self._diagnose,
            "narrate": self._narrate,
            "hints": self._hints,
            "challenge": self._challenge,
        }.get(task.name)
        if handler is None:
            return Completion(backend=self.name, data={}, note=f"unsupported task {task.name!r}")
        data = handler(dict(task.payload))
        return Completion(backend=self.name, model="deterministic-template", data=data)

    # handlers
    def _diagnose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only contribute what a deterministic rule can defend.
        The symbolic and dynamic layers already cover everything decidable from
        syntax. What is left are judgement calls, and instead of making them up
        this fires two narrow heuristics and otherwise abstains.
        """
        hypotheses: List[Dict[str, Any]] = []
        tags = set(payload.get("problem_tags") or ())
        has_counterexample = bool(payload.get("counterexample"))
        # only correctness hypotheses compete with a reasoning level explanation,
        # a complexity finding does not explain a wrong answer
        symbolic = list(payload.get("correctness_candidates") or ())

        # a greedy shape is a property of the submission, not the task: the usual
        # form is applying a greedy rule to a problem that needs DP, and the
        # problem tags say nothing about greed
        greedy_shape = "greedy" in tags or (
            bool(payload.get("sorts_input")) and not payload.get("uses_table")
        )
        if greedy_shape and has_counterexample and not symbolic:
            hypotheses.append(
                {
                    "id": "greedy.local-optimum-assumed",
                    "confidence": 0.45,
                    "reason": (
                        "the submission commits to a locally optimal choice and fails on a small "
                        "input, with no structural defect that would explain the wrong answer — "
                        "the signature of a greedy rule adopted without an exchange argument"
                    ),
                }
            )
        if payload.get("complexity_gap") and payload.get("linear_membership"):
            hypotheses.append(
                {
                    "id": "ds.wrong-container-choice",
                    "confidence": 0.30,
                    "reason": "the dominant inner-loop operation is linear on the chosen container",
                }
            )
        return {"misconceptions": hypotheses, "summary": ""}

    def _narrate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Factual account of the diagnosis built from the structured fields."""
        parts: List[str] = []
        misconception = payload.get("misconception_name")
        divergence = payload.get("divergence") or {}
        counterexample = payload.get("counterexample") or {}
        complexity = payload.get("complexity") or {}

        if misconception:
            parts.append(f"The evidence points to one belief: {misconception.lower()}.")
        if divergence.get("description"):
            step = divergence.get("step")
            location = f" at line {divergence['student_line']}" if divergence.get("student_line") else ""
            prefix = f"Tracing both executions{location}"
            if step:
                prefix += f" (checkpoint {step})"
            parts.append(f"{prefix}: {divergence['description']}.")
        if counterexample.get("args") is not None:
            student = counterexample.get("student")
            expected = counterexample.get("reference")
            if counterexample.get("error"):
                parts.append(
                    f"On the smallest failing input {counterexample['args']} the program raises "
                    f"{counterexample['error']}."
                )
            else:
                parts.append(
                    f"On the smallest failing input {counterexample['args']} it produces "
                    f"{student} where {expected} is required."
                )
        if complexity.get("gap"):
            parts.append(
                f"Its growth rate is {complexity.get('student')} against a target of "
                f"{complexity.get('reference')}."
            )
        if not parts:
            parts.append(
                "No decisive evidence was found: the submission matches the reference on every "
                "input examined and shows no structural warning sign."
            )
        return {"narrative": " ".join(parts)}

    def _hints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Three rung Socratic ladder grounded in the concrete evidence."""
        divergence = payload.get("divergence") or {}
        counterexample = payload.get("counterexample") or {}
        concept_id = payload.get("target_concept") or ""
        concept_name = payload.get("target_concept_name") or concept_id.replace("-", " ")
        focus = payload.get("remediation_focus") or ""
        roles = list(divergence.get("roles") or ())
        line = divergence.get("student_line")
        args = counterexample.get("args")

        hints: List[Dict[str, str]] = []

        # rung 1: point at the region, claim nothing
        if args is not None and line:
            first = (
                f"Take the input {args} and step through your own function by hand, writing down "
                f"the values at line {line} each time you reach it. Stop at the first moment a "
                f"value surprises you."
            )
        elif args is not None:
            first = (
                f"Work through your function on {args} with pencil and paper, recording every "
                f"value you compute. Where does the paper trace stop matching what you intended?"
            )
        else:
            first = (
                "Pick the smallest input you can think of for this problem and hand-execute your "
                "function on it, writing down each variable as it changes."
            )
        hints.append({"kind": "orienting", "text": first})

        # rung 2: show the contradiction, still no repair
        if roles and divergence.get("student_state") and divergence.get("reference_state"):
            role = roles[0]
            observed = divergence["student_state"].get(role)
            required = divergence["reference_state"].get(role)
            if role == "return":
                # internal states agreed all the way, only the result differs. No variable
                # to ask about, so ask about the decision.
                second = (
                    f"Every intermediate value matched a correct solution, yet your function "
                    f"returns {observed!r} where {required!r} is required. So the difference is "
                    f"not in a value you computed but in a decision you made — which condition "
                    f"in your code decides that answer, and what does it assume?"
                )
            else:
                second = (
                    f"At that point your `{role}` holds {observed!r}, but for the answer to be "
                    f"reachable it would have to be {required!r}. Which line last wrote `{role}`, "
                    f"and what did you assume about the range it was allowed to take?"
                )
        elif counterexample.get("error"):
            error = str(counterexample["error"])
            if "budget" in error or "infinite" in error or "recursion" in error:
                # a run that never finishes has no failing expression, so ask about
                # progress instead
                second = (
                    "Your function never finishes on that input. Name the quantity you expected "
                    "to get strictly smaller on every pass — then check, on paper, what it "
                    "actually does."
                )
            else:
                second = (
                    f"On that input your function raises {error}. What value did the failing "
                    "expression have, and which earlier line allowed it to take that value?"
                )
        elif args is not None:
            second = (
                f"Your function returns {counterexample.get('student')!r} on {args}, while the "
                f"specification requires {counterexample.get('reference')!r}. Which decision in "
                "your code produced the difference, and what were you assuming when you made it?"
            )
        else:
            second = (
                "Compare what your code does with what you meant it to do, one statement at a "
                "time. At which statement do the two accounts first disagree?"
            )
        hints.append({"kind": "contradiction", "text": second})

        # rung 3: name the prerequisite concept and ask its defining question
        probe = CONCEPT_PROBES.get(concept_id) or _GENERIC_PROBE.format(concept=concept_name)
        third = probe
        if focus:
            third = f"{probe} {focus}"
        hints.append({"kind": "conceptual", "text": third})

        return {"hints": hints}

    def _challenge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Counterfactual task: predict before you run."""
        counterexample = payload.get("counterexample") or {}
        divergence = payload.get("divergence") or {}
        args = counterexample.get("args")
        concept_name = payload.get("target_concept_name") or "the idea under test"

        if args is not None:
            prompt = (
                f"Before running anything, write down what you predict your function returns for "
                f"{args}, and why. Then run it and compare."
            )
            trace_question = (
                "If the prediction and the result differ, the misconception lives in the step "
                "where your reasoning and your code parted company — find that step and describe "
                "it in one sentence."
            )
            insight = (
                "A prediction that does not match execution localises the flawed belief far more "
                "precisely than reading the code again."
            )
        elif divergence.get("student_line"):
            prompt = (
                f"Construct the smallest input you can that forces line "
                f"{divergence['student_line']} to execute at least twice, and predict the values "
                "it will see on each pass."
            )
            trace_question = "Which of your predictions did the run contradict?"
            insight = "Deliberately targeting a line exposes the assumption it encodes."
        else:
            prompt = (
                f"Write two inputs: one where you are confident your function is right, and one "
                f"where you are not. What distinguishes them, in terms of {concept_name}?"
            )
            trace_question = "What property does the second input have that the first does not?"
            insight = "Naming the distinguishing property is naming the concept under test."

        return {
            "prompt": prompt,
            "trace_question": trace_question,
            "expected_insight": insight,
        }
