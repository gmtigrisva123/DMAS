"""Cognitive intervention agent.

ZPD + Socratic method: instead of giving the fixed code it produces hints and
small counterfactual challenges that expose the contradiction in the student's
code. Every constraint is enforced by a mechanism, not by asking nicely:

- what to teach: ZPD constrained item selection over the deficiency frontier
- what can be said: grounded in retrieved knowledge cards
- what must not be said: the leakage gate. A hint that fails is replaced by a
  template rung, never just flagged, so the output is gate clean by
  construction.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..domain import Challenge, Hint, HintKind, Intervention
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.mastery import MasteryState, MasteryTracker
from ..knowledge.misconceptions import MISCONCEPTIONS
from ..knowledge.ontology import knowledge_graph
from ..knowledge.zpd import ZPDSelector, ability_from_mastery, items_from_concepts
from ..llm.base import Task
from ..llm.offline import OfflineReasoner
from ..llm.parsing import as_str_list
from ..llm.registry import Reasoner
from ..retrieval.ma_rag import MultiAgentRAG
from ..util.text import sentences
from .base import Agent, Blackboard
from .verifier import ComplianceVerifier

_HINT_SYSTEM = (
    "You are a Socratic tutor for algorithmic programming. You never provide, quote, paraphrase "
    "or hint at the corrected code. You never say what to change a line to. You ask questions "
    "that make the student notice a contradiction in their own reasoning, and you refer only to "
    "code the student has already written. Every claim you make about how algorithms behave must "
    "be supported by the supplied evidence cards."
)

_HINT_INSTRUCTION = (
    "Write a ladder of {levels} hints, from least to most directive, about the diagnosed "
    "misconception. Rung 1 orients attention to a concrete input and location. Rung 2 exhibits "
    "the contradiction the evidence establishes. Rung 3 names the prerequisite concept and asks "
    "its defining question. Return JSON with the key 'hints': a list of objects with 'kind' "
    "(orienting | contradiction | conceptual) and 'text'."
)

_CHALLENGE_SYSTEM = (
    "You design counter-factual micro-tasks that make a student's flawed belief testable. You "
    "never reveal the correct solution or the required repair."
)

_CHALLENGE_INSTRUCTION = (
    "Design one short task the student can perform on their own code to expose the contradiction. "
    "Return JSON with keys 'prompt', 'trace_question' and 'expected_insight'."
)

_FALLBACK_CHALLENGE = (
    "Before running anything, write down what you predict your function returns for the "
    "smallest input you can think of, and why. Then run it and compare."
)

_KIND_ORDER = (HintKind.ORIENTING, HintKind.CONTRADICTION, HintKind.CONCEPTUAL, HintKind.PROCEDURAL)

# reply schemas, enforced server side by backends that support it
_HINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": [k.value for k in _KIND_ORDER]},
                    "text": {"type": "string"},
                },
                "required": ["kind", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hints"],
    "additionalProperties": False,
}

_CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "trace_question": {"type": "string"},
        "expected_insight": {"type": "string"},
    },
    "required": ["prompt", "trace_question", "expected_insight"],
    "additionalProperties": False,
}

# last resort per rung: only questions and reflective imperatives, no code
# span and no claims, so they pass both gates no matter what
_FALLBACK_RUNGS = (
    "Pick the smallest input you can construct for this problem and hand-execute your "
    "function on it, writing down every variable as it changes. Where does your paper "
    "trace stop matching what you intended?",
    "Compare what your code does with what you meant it to do, one statement at a time. "
    "At which statement do the two accounts first disagree?",
    "State the property you believe holds every time your loop reaches its top. Does it "
    "still hold at the very first and the very last iteration?",
    "Describe, in one sentence, the assumption this code relies on. How would you test "
    "whether that assumption actually holds?",
)


class CognitiveInterventionAgent(Agent):
    """Picks a ZPD target and produces gate clean Socratic hints."""

    name = "cognitive-intervention"
    role = "scaffolded remediation"

    def __init__(
        self,
        config: Optional[Config] = None,
        reasoner: Optional[Reasoner] = None,
        rag: Optional[MultiAgentRAG] = None,
        graph: Optional[KnowledgeGraph] = None,
        verifier: Optional[ComplianceVerifier] = None,
    ):
        self.config = config or Config()
        self.reasoner = reasoner or Reasoner(self.config.llm)
        self.rag = rag or MultiAgentRAG(config=self.config.retrieval)
        self.graph = graph or knowledge_graph()
        self.verifier = verifier or ComplianceVerifier(self.config, self.rag)
        self.selector = ZPDSelector(self.config.intervention)
        self.fallback = OfflineReasoner()

    # run
    def run(self, board: Blackboard) -> Blackboard:
        target, probability = self._select_target(board)
        context = self._ground(board, target)
        board.context = context

        payload = self._payload(board, target)
        hints = self._hints(board, payload, context)
        challenge = self._challenge(board, payload, context)

        board.intervention = Intervention(
            target_concept=target,
            hints=tuple(hints),
            challenge=challenge,
            next_item=board.extras.get("next_item"),
            zpd_probability=probability,
            citations=context.citations,
            regenerations=board.extras.get("regenerations", 0),
        )
        return board

    # targeting
    def _select_target(self, board: Blackboard) -> Tuple[Optional[str], float]:
        alignment = board.alignment
        mastery: MasteryState = board.extras.get("mastery")
        if alignment is None:
            return None, 0.0

        candidates = list(alignment.deficiency_frontier) or [
            cid for cid, _ in alignment.top_blame(4)
        ]
        if not candidates:
            return alignment.root_cause, 0.0

        difficulties = {c.cid: c.difficulty for c in self.graph}
        ability = (
            ability_from_mastery(mastery.posterior, difficulties) if mastery is not None else 0.0
        )
        items = items_from_concepts(difficulties, candidates)
        if not items:
            return alignment.root_cause, 0.0

        # concepts named directly by the diagnosis are worth more than concepts
        # that only inherited blame
        direct = set(board.extras.get("seeded_concepts") or ())
        relevance = {
            cid: mass * (1.6 if cid in direct else 1.0) for cid, mass in alignment.blame.items()
        }
        chosen = self.selector.select(items, ability, relevance, k=2)
        if not chosen:
            return alignment.root_cause, 0.0
        best = chosen[0]
        board.extras["ability"] = ability
        board.extras["zpd_candidates"] = [
            (s.item.concepts[0], round(s.probability, 3), round(s.utility, 4)) for s in chosen
        ]
        if len(chosen) > 1:
            board.extras["next_item"] = chosen[1].item.concepts[0]
        if not best.in_band:
            board.note(
                f"no frontier concept lies inside the ZPD at ability {ability:+.2f}; "
                f"the closest ({best.item.concepts[0]}) has p={best.probability:.2f}"
            )
        return best.item.concepts[0], best.probability

    def _ground(self, board: Blackboard, target: Optional[str]):
        diagnosis = board.diagnosis
        alignment = board.alignment
        entries = [
            MISCONCEPTIONS[h.misconception_id]
            for h in (diagnosis.hits if diagnosis else ())
            if h.misconception_id in MISCONCEPTIONS
        ]
        concepts: List[str] = []
        for entry in entries[:2]:
            concepts.extend(entry.concepts)
        if target:
            concepts.insert(0, target)
        return self.rag.ground(
            problem_title=board.problem.title,
            misconception_ids=[e.mid for e in entries],
            misconception_texts=[e.description for e in entries],
            concepts=tuple(dict.fromkeys(concepts)),
            frontier=tuple(alignment.deficiency_frontier[:3]) if alignment else (),
            divergence_kind=str(diagnosis.divergence.kind) if diagnosis and diagnosis.divergence else None,
            complexity_gap=(
                (diagnosis.student_complexity, diagnosis.reference_complexity)
                if diagnosis and diagnosis.complexity_gap
                else None
            ),
        )

    # generation
    def _payload(self, board: Blackboard, target: Optional[str]) -> Dict[str, Any]:
        diagnosis = board.diagnosis
        primary = diagnosis.primary if diagnosis else None
        entry = MISCONCEPTIONS.get(primary.misconception_id) if primary else None
        divergence = diagnosis.divergence if diagnosis else None
        counterexample = diagnosis.counterexample if diagnosis else None
        return {
            "problem_title": board.problem.title,
            "problem_statement": board.problem.statement,
            "student_code": board.submission.source,
            "misconception_id": entry.mid if entry else None,
            "misconception_name": entry.name if entry else None,
            "student_voice": entry.student_voice if entry else None,
            "remediation_focus": entry.remediation_focus if entry else None,
            "target_concept": target,
            "target_concept_name": self.graph.name(target) if target and target in self.graph else target,
            "target_concept_summary": (
                self.graph.concept(target).summary if target and target in self.graph else None
            ),
            "divergence": (
                {
                    "kind": str(divergence.kind),
                    "step": divergence.step,
                    "student_line": divergence.student_line,
                    "roles": divergence.differing_roles(),
                    "student_state": dict(divergence.student_state),
                    "reference_state": dict(divergence.reference_state),
                    "description": divergence.description,
                }
                if divergence
                else None
            ),
            "counterexample": (
                {
                    "args": repr(counterexample.args),
                    "student": repr(counterexample.student_output),
                    "reference": repr(counterexample.reference_output),
                    "error": counterexample.student_error,
                }
                if counterexample
                else None
            ),
        }

    def _hints(self, board: Blackboard, payload: Dict[str, Any], context) -> List[Hint]:
        leakage_context = self.verifier.context_for(
            board.problem.reference_solution, board.submission.source
        )
        levels = self.config.intervention.hint_levels
        task = Task(
            name="hints",
            system=_HINT_SYSTEM,
            instruction=_HINT_INSTRUCTION.format(levels=levels),
            payload={**payload, "evidence": context.render()},
            schema={"hints": list},
            json_schema=_HINTS_SCHEMA,
        )
        raw = self._generate_hints(board, task)
        safe_template = self.fallback.run(task).data.get("hints") or []

        accepted: List[Hint] = []
        replacements = 0
        gate_log: List[Dict[str, Any]] = board.extras.setdefault("gate_log", [])
        for index in range(levels):
            candidate = raw[index]["text"] if index < len(raw) else ""
            template = safe_template[index]["text"] if index < len(safe_template) else ""
            kind_label = (
                raw[index].get("kind") if index < len(raw)
                else (safe_template[index].get("kind") if index < len(safe_template) else None)
            )

            text, rejection = self._first_clean(
                [candidate, template], leakage_context, context, fallback=_FALLBACK_RUNGS[index]
            )
            if rejection is not None:
                board.telemetry.warn(
                    "hint rung {} replaced: {}".format(
                        index + 1,
                        rejection.reasons[0] if rejection.reasons else "failed the compliance gate",
                    )
                )
                replacements += 1
            # audit record of what the generator proposed and what the gate did.
            # This is the raw data for the gate study.
            verdict = self.verifier.review(candidate, leakage_context, context) if candidate else None
            gate_log.append(
                {
                    "item": f"hint-{index + 1}",
                    "candidate": candidate,
                    "candidate_backend": board.telemetry.backend,
                    "accepted": bool(verdict and verdict.accepted),
                    "leakage": round(verdict.leakage.score, 4) if verdict else None,
                    "leakage_channel": verdict.leakage.channel if verdict else None,
                    "structural_match": bool(verdict and verdict.leakage.structural_match),
                    "unsupported_claims": (
                        [c.claim for c in verdict.grounding.unsupported] if verdict else []
                    ),
                    "reasons": list(verdict.reasons) if verdict else [],
                    "final": text,
                    "final_source": (
                        "generated" if text == candidate.strip()
                        else "template" if text == template.strip()
                        else "fallback" if text == _FALLBACK_RUNGS[index]
                        else "sanitised"
                    ),
                }
            )
            if not text:
                continue
            accepted.append(
                Hint(
                    level=len(accepted) + 1,
                    kind=_kind_for(kind_label, index),
                    text=text,
                    concept_id=payload.get("target_concept"),
                )
            )
        board.extras["regenerations"] = replacements
        return accepted

    # gate
    def _first_clean(self, candidates, leakage_context, context, *, fallback: str):
        """First candidate that passes the gate, else a safe rung.

        The chain always ends: generated text -> template -> template with the bad
        sentences removed -> a fixed question with no code and no claims. Dropping
        the rung (what an earlier version did) is the one thing that must not
        happen because it deletes the content while reporting success.
        """
        rejection = None
        for text in candidates:
            text = (text or "").strip()
            if not text:
                continue
            verdict = self.verifier.review(text, leakage_context, context)
            if verdict.accepted:
                return text, rejection
            rejection = rejection or verdict
            cleaned = self._sanitise(text, leakage_context, context)
            if cleaned:
                return cleaned, rejection
        return fallback, rejection

    def _sanitise(self, text: str, leakage_context, context) -> str:
        """Remove the sentences that fail, keep the rest."""
        parts = sentences(text)
        if len(parts) < 2:
            return ""
        kept = [
            part for part in parts
            if self.verifier.review(part, leakage_context, context).accepted
        ]
        if not kept or len(kept) == len(parts):
            return ""
        return " ".join(kept)

    def _generate_hints(self, board: Blackboard, task: Task) -> List[Dict[str, Any]]:
        completion = self.reasoner.run(task)
        board.telemetry.model_calls += 1
        board.telemetry.model_tokens += completion.total_tokens
        board.telemetry.backend = completion.backend
        if completion.degraded:
            # a rung written by the fallback does not measure the model, so the audit
            # counts the row as degraded and reruns it (same as the evaluator does)
            board.telemetry.warn(f"reasoning backend degraded: {completion.note}")
        out: List[Dict[str, Any]] = []
        for item in completion.data.get("hints") or []:
            if isinstance(item, dict) and str(item.get("text") or "").strip():
                out.append({"kind": item.get("kind"), "text": str(item["text"]).strip()})
            elif isinstance(item, str) and item.strip():
                out.append({"kind": None, "text": item.strip()})
        return out

    def _challenge(self, board: Blackboard, payload: Dict[str, Any], context) -> Optional[Challenge]:
        leakage_context = self.verifier.context_for(
            board.problem.reference_solution, board.submission.source
        )
        task = Task(
            name="challenge",
            system=_CHALLENGE_SYSTEM,
            instruction=_CHALLENGE_INSTRUCTION,
            payload={**payload, "evidence": context.render()},
            schema={"prompt": str},
            json_schema=_CHALLENGE_SCHEMA,
        )
        completion = self.reasoner.run(task)
        board.telemetry.model_calls += 1
        board.telemetry.model_tokens += completion.total_tokens
        board.telemetry.backend = completion.backend
        if completion.degraded:
            board.telemetry.warn(f"reasoning backend degraded: {completion.note}")
        data = completion.data or {}
        prompt = str(data.get("prompt") or "").strip()
        trace_question = str(data.get("trace_question") or "").strip()
        insight = str(data.get("expected_insight") or "").strip()

        generated = f"{prompt} {trace_question}".strip()
        record: Dict[str, Any] = {
            "item": "challenge",
            "candidate": generated,
            "candidate_backend": board.telemetry.backend,
            "accepted": False,
            "leakage": None,
            "leakage_channel": None,
            "structural_match": False,
            "unsupported_claims": [],
            "reasons": [],
            "final": "",
            "final_source": "fallback",
        }
        if prompt:
            verdict = self.verifier.review(generated, leakage_context, context)
            record.update(
                accepted=verdict.accepted,
                leakage=round(verdict.leakage.score, 4),
                leakage_channel=verdict.leakage.channel,
                structural_match=verdict.leakage.structural_match,
                unsupported_claims=[c.claim for c in verdict.grounding.unsupported],
                reasons=list(verdict.reasons),
            )
            if not verdict.accepted:
                board.telemetry.warn(
                    "counter-factual challenge replaced: "
                    + (verdict.reasons[0] if verdict.reasons else "failed the compliance gate")
                )
                prompt = ""
        if not prompt:
            data = self.fallback.run(task).data
            prompt = str(data.get("prompt") or "").strip()
            trace_question = str(data.get("trace_question") or "").strip()
            insight = str(data.get("expected_insight") or "").strip()
            if prompt and not self.verifier.review(
                f"{prompt} {trace_question}", leakage_context, context
            ).accepted:
                prompt = ""
        if not prompt:
            prompt = _FALLBACK_CHALLENGE
            trace_question = "Which of your predictions did the run contradict?"
            insight = "A prediction that execution refutes localises the flawed belief exactly."

        final = f"{prompt} {trace_question}".strip()
        record["final"] = final
        record["final_source"] = (
            "generated" if record["accepted"] and final == generated
            else "fallback" if prompt == _FALLBACK_CHALLENGE
            else "template"
        )
        board.extras.setdefault("gate_log", []).append(record)

        counterexample = board.diagnosis.counterexample if board.diagnosis else None
        return Challenge(
            prompt=prompt,
            trace_question=trace_question,
            expected_insight=insight,
            args=counterexample.args if counterexample else None,
        )


def _kind_for(label: Optional[str], index: int) -> HintKind:
    if isinstance(label, str):
        for kind in HintKind:
            if kind.value == label.strip().lower():
                return kind
    return _KIND_ORDER[min(index, len(_KIND_ORDER) - 1)]
