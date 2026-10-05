import json

import pytest

from deductive_mas.config import LLMConfig
from deductive_mas.errors import BackendError, ConfigurationError
from deductive_mas.llm.base import Completion, Task, validate
from deductive_mas.llm.cache import ResponseCache
from deductive_mas.llm import claude as claude_backend
from deductive_mas.llm.claude import ClaudeBackend
from deductive_mas.llm.openai_compatible import OpenAICompatibleBackend, _unpack
from deductive_mas.llm.offline import OfflineReasoner
from deductive_mas.llm.parsing import (
    clamp_unit,
    as_str_list,
    find_json_object,
    parse_structured,
    repair_json,
    split_reasoning,
)
from deductive_mas.llm.registry import Reasoner, make_backend

HINT_PAYLOAD = {
    "divergence": {
        "student_line": 4,
        "roles": ["hi"],
        "student_state": {"hi": 8},
        "reference_state": {"hi": 7},
    },
    "counterexample": {"args": "([0], 1)", "student": 0, "reference": 1},
    "target_concept": "half-open-intervals",
    "target_concept_name": "Half-open intervals",
    "remediation_focus": "Fix one convention and derive every line from it.",
}


def _task(name="hints", schema=None):
    return Task(name=name, system="s", instruction="i", payload=HINT_PAYLOAD,
                schema=schema or {"hints": list})


class TestParsing:
    def test_reasoning_is_separated_from_the_answer(self):
        reasoning, answer = split_reasoning('<think>chain of thought</think>{"a": 1}')
        assert reasoning == "chain of thought" and answer == '{"a": 1}'

    def test_text_without_reasoning_is_untouched(self):
        assert split_reasoning('{"a": 1}') == ("", '{"a": 1}')

    def test_balanced_object_is_located_amid_prose(self):
        blob = find_json_object('prelude {"a": {"b": 1}} epilogue')
        assert blob and json.loads(blob) == {"a": {"b": 1}}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        blob = find_json_object('{"a": "}{", "b": 2}')
        assert blob and json.loads(blob)["b"] == 2

    def test_no_object_returns_none(self):
        assert find_json_object("no braces here") is None
        assert find_json_object("") is None

    def test_common_deviations_are_repaired(self):
        assert json.loads(repair_json('{"a": 1, "b": [1, 2,],}'))["b"] == [1, 2]

    def test_python_literals_are_accepted(self):
        parsed = parse_structured("```json\n{'a': 1, 'ok': True, 'x': None}\n```")
        assert parsed == {"a": 1, "ok": True, "x": None}

    def test_unparsable_output_returns_none(self):
        assert parse_structured("total nonsense") is None
        assert parse_structured("") is None

    def test_confidence_is_clamped(self):
        assert clamp_unit(1.7) == 1.0
        assert clamp_unit(-3) == 0.0
        assert clamp_unit("nope", 0.25) == 0.25
        assert clamp_unit(float("nan"), 0.5) == 0.5

    def test_string_lists_are_coerced(self):
        assert as_str_list("one") == ["one"]
        assert as_str_list([{"text": "a"}, "b", 3]) == ["a", "b"]
        assert as_str_list(None) == []


class TestSchemaValidation:
    def test_missing_key_is_reported(self):
        ok, why = validate({}, {"hints": list})
        assert not ok and "hints" in why

    def test_wrong_type_is_reported(self):
        ok, why = validate({"hints": "x"}, {"hints": list})
        assert not ok and "list" in why

    def test_int_satisfies_float(self):
        assert validate({"x": 1}, {"x": float})[0]

    def test_tuple_satisfies_list(self):
        assert validate({"x": ()}, {"x": list})[0]


class TestOfflineReasoner:
    def test_produces_a_three_rung_ladder(self):
        hints = OfflineReasoner().run(_task()).data["hints"]
        assert len(hints) == 3
        assert [h["kind"] for h in hints] == ["orienting", "contradiction", "conceptual"]

    def test_hints_quote_the_concrete_evidence(self):
        hints = OfflineReasoner().run(_task()).data["hints"]
        joined = " ".join(h["text"] for h in hints)
        assert "([0], 1)" in joined and "line 4" in joined

    def test_hints_never_contain_an_assignment(self):
        for hint in OfflineReasoner().run(_task()).data["hints"]:
            assert "=" not in hint["text"] or "?" in hint["text"]

    def test_a_known_concept_gets_its_own_probe(self):
        third = OfflineReasoner().run(_task()).data["hints"][2]["text"]
        assert "empty" in third            # the half open interval probe

    def test_an_unknown_concept_falls_back_to_a_generic_probe(self):
        payload = dict(HINT_PAYLOAD, target_concept="nonexistent", target_concept_name="Nonsense")
        hints = OfflineReasoner().run(Task("hints", "s", "i", payload, {"hints": list})).data["hints"]
        assert "Nonsense" in hints[2]["text"]

    def test_challenge_asks_for_a_prediction(self):
        data = OfflineReasoner().run(_task("challenge", {"prompt": str})).data
        assert "predict" in data["prompt"].lower() and data["trace_question"]

    def test_narrate_summarises_the_evidence(self):
        payload = {
            "misconception_name": "Mixed interval conventions",
            "divergence": {"description": "hi is 8 but should be 7", "student_line": 4, "step": 2},
            "counterexample": {"args": "([0], 1)", "student": 0, "reference": 1},
            "complexity": {"gap": False},
        }
        narrative = OfflineReasoner().run(Task("narrate", "s", "i", payload, {"narrative": str}))
        assert "([0], 1)" in narrative.data["narrative"]

    def test_narrate_handles_an_empty_diagnosis(self):
        result = OfflineReasoner().run(Task("narrate", "s", "i", {}, {"narrative": str}))
        assert "No decisive evidence" in result.data["narrative"]

    def test_diagnose_abstains_without_a_trigger(self):
        result = OfflineReasoner().run(Task("diagnose", "s", "i", {}, {"misconceptions": list}))
        assert result.data["misconceptions"] == []

    def test_diagnose_recognises_a_greedy_shape(self):
        payload = {
            "problem_tags": ["dynamic-programming"],
            "counterexample": {"args": "([2, 7], 15)"},
            "correctness_candidates": [],
            "sorts_input": True,
            "uses_table": False,
        }
        result = OfflineReasoner().run(Task("diagnose", "s", "i", payload, {"misconceptions": list}))
        assert result.data["misconceptions"][0]["id"] == "greedy.local-optimum-assumed"

    def test_a_structural_candidate_suppresses_the_heuristic(self):
        payload = {
            "problem_tags": ["greedy"],
            "counterexample": {"args": "(x,)"},
            "correctness_candidates": ["rec.missing-base-case"],
            "sorts_input": True,
        }
        result = OfflineReasoner().run(Task("diagnose", "s", "i", payload, {"misconceptions": list}))
        assert result.data["misconceptions"] == []

    def test_an_unknown_task_is_reported_not_raised(self):
        result = OfflineReasoner().run(Task("nope", "s", "i", {}, {}))
        assert result.data == {} and "unsupported" in result.note

    def test_output_is_deterministic(self):
        first = OfflineReasoner().run(_task()).data
        second = OfflineReasoner().run(_task()).data
        assert first == second


class TestCache:
    def test_round_trip(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        key = cache.key(_task(), "m", 0.2)
        assert cache.get(key) is None
        cache.put(key, Completion(text="hello", data={"hints": []}, backend="offline"))
        restored = cache.get(key)
        assert restored is not None and restored.text == "hello" and restored.cached

    def test_key_depends_on_the_payload_and_the_model(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        other = Task("hints", "s", "i", {"different": 1}, {"hints": list})
        assert cache.key(_task(), "m", 0.2) != cache.key(other, "m", 0.2)
        assert cache.key(_task(), "m", 0.2) != cache.key(_task(), "other", 0.2)

    def test_key_is_stable_across_dict_ordering(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        a = Task("t", "s", "i", {"x": 1, "y": 2}, {})
        b = Task("t", "s", "i", {"y": 2, "x": 1}, {})
        assert cache.key(a, "m", 0.2) == cache.key(b, "m", 0.2)

    def test_disabled_cache_stores_nothing(self):
        cache = ResponseCache(None)
        key = cache.key(_task(), "m", 0.2)
        cache.put(key, Completion(text="x"))
        assert cache.get(key) is None

    def test_clear_removes_entries(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        cache.put(cache.key(_task(), "m", 0.2), Completion(text="x"))
        assert cache.clear() == 1


class TestBackendSelection:
    def test_offline_is_selected_explicitly(self):
        assert make_backend(LLMConfig(backend="offline")).name == "offline"

    def test_unknown_backend_is_refused(self):
        with pytest.raises(ConfigurationError):
            make_backend(LLMConfig(backend="martian"))

    def test_claude_without_a_key_is_refused(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DMAS_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            make_backend(LLMConfig(backend="claude"))

    def test_openai_compatible_without_a_key_is_refused(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DMAS_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            make_backend(LLMConfig(backend="openai", base_url="https://example.org/v1"))

    def test_a_localhost_endpoint_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAICompatibleBackend(LLMConfig(base_url="http://localhost:11434/v1"))
        assert backend.available()

    def test_auto_falls_back_to_offline(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DMAS_API_KEY", raising=False)
        assert make_backend(LLMConfig(backend="auto")).name == "offline"

    def test_auto_selects_the_openai_compatible_backend_when_a_key_is_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
        assert make_backend(LLMConfig(backend="auto")).name == "openai"

    def test_defaults_match_the_paper(self):
        config = LLMConfig()
        assert config.model == "openai/gpt-oss-120b"
        assert config.base_url == "https://api.groq.com/openai/v1"

    def test_claude_gets_its_own_host_unless_told_otherwise(self):
        assert ClaudeBackend(LLMConfig(backend="claude")).config.base_url == "https://api.anthropic.com"
        custom = LLMConfig(backend="claude", base_url="https://proxy.example/v1")
        assert ClaudeBackend(custom).config.base_url == "https://proxy.example/v1"

    def test_prompt_rendering_carries_the_payload_and_schema(self):
        messages = OpenAICompatibleBackend(LLMConfig()).render(_task())
        assert messages[0].role == "system"
        assert "hints" in messages[1].content and "half-open-intervals" in messages[1].content

    def test_response_envelope_is_unpacked(self):
        payload = {
            "choices": [{"message": {"content": "{}", "reasoning_content": "why"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        unpacked = _unpack(payload)
        assert unpacked["reasoning"] == "why" and unpacked["completion_tokens"] == 4

    def test_an_empty_envelope_is_an_error(self):
        with pytest.raises(BackendError):
            _unpack({"choices": []})


class TestReasonerDegradation:
    class _Broken:
        name = "broken"

        def available(self):
            return True

        def run(self, task):
            raise BackendError("endpoint unreachable")

    class _Exploding:
        name = "exploding"

        def available(self):
            return True

        def run(self, task):
            raise RuntimeError("kaboom")

    def test_a_backend_error_degrades_to_the_offline_reasoner(self):
        reasoner = Reasoner(LLMConfig(cache_dir=None), backend=self._Broken())
        completion = reasoner.run(_task())
        assert completion.degraded and completion.data["hints"]
        assert reasoner.degradations

    def test_an_unexpected_exception_also_degrades(self):
        reasoner = Reasoner(LLMConfig(cache_dir=None), backend=self._Exploding())
        completion = reasoner.run(_task())
        assert completion.degraded and completion.data["hints"]

    def test_degraded_responses_are_not_cached(self, tmp_path):
        reasoner = Reasoner(LLMConfig(cache_dir=str(tmp_path)), backend=self._Broken())
        reasoner.run(_task())
        assert list(tmp_path.glob("*.json")) == []

    def test_a_cache_hit_avoids_the_backend(self, tmp_path):
        reasoner = Reasoner(LLMConfig(cache_dir=str(tmp_path)), backend=OfflineReasoner())
        reasoner.run(_task())
        before = reasoner.calls
        reasoner.run(_task())
        assert reasoner.cache_hits == 1 and reasoner.calls == before

    def test_backends_never_share_a_cache_entry(self, tmp_path):
        """An offline answer must never be served to a session running a live model."""

        class _Live:
            name = "claude"

            def available(self):
                return True

            def run(self, task):
                return Completion(text="{}", data={"hints": [{"kind": "orienting", "text": "live"}]},
                                  backend="claude")

        config = LLMConfig(cache_dir=str(tmp_path))
        Reasoner(config, backend=OfflineReasoner()).run(_task())
        live = Reasoner(config, backend=_Live()).run(_task())
        assert live.backend == "claude" and not live.cached


class TestClaudeBackend:
    """Messages API client against a stubbed transport."""

    def _config(self):
        return LLMConfig(backend="claude", model="claude-opus-5", api_key="sk-ant-test",
                         cache_dir=None, max_retries=1)

    def test_request_body_uses_adaptive_thinking_and_the_task_schema(self):
        task = Task(name="narrate", system="s", instruction="i", payload={"a": 1},
                    schema={"narrative": str},
                    json_schema={"type": "object", "properties": {"narrative": {"type": "string"}},
                                 "required": ["narrative"], "additionalProperties": False})
        body = ClaudeBackend(self._config()).body(task)
        assert body["model"] == "claude-opus-5"
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"]["effort"] == "medium"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert "temperature" not in body and "top_p" not in body
        assert body["system"] == "s" and body["messages"][0]["role"] == "user"

    def test_legacy_models_omit_thinking_and_effort(self):
        config = LLMConfig(backend="claude", api_key="k", model="claude-haiku-4-5", cache_dir=None)
        body = ClaudeBackend(config).body(_task())
        assert "thinking" not in body and "effort" not in body.get("output_config", {})

    def test_envelope_unpacks_text_thinking_and_usage(self):
        payload = {
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "considered the bounds"},
                {"type": "text", "text": '{"narrative": "ok"}'},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 2},
        }
        unpacked = claude_backend._unpack(payload)
        assert unpacked["content"] == '{"narrative": "ok"}'
        assert unpacked["reasoning"] == "considered the bounds"
        assert unpacked["prompt_tokens"] == 12 and unpacked["completion_tokens"] == 4

    def test_a_refusal_is_a_backend_error(self):
        with pytest.raises(BackendError):
            claude_backend._unpack({"stop_reason": "refusal", "content": [],
                                    "stop_details": {"category": "x"}})

    def test_truncation_is_a_backend_error(self):
        with pytest.raises(BackendError):
            claude_backend._unpack({"stop_reason": "max_tokens",
                                    "content": [{"type": "text", "text": "{"}]})

    def test_run_validates_and_repairs_once(self, monkeypatch):
        backend = ClaudeBackend(self._config())
        replies = iter([
            {"content": "not json at all", "reasoning": "", "model": "m",
             "prompt_tokens": 1, "completion_tokens": 1},
            {"content": '{"hints": [{"kind": "orienting", "text": "look at line 4"}]}',
             "reasoning": "", "model": "m", "prompt_tokens": 1, "completion_tokens": 1},
        ])
        seen = []

        def fake_post(body):
            seen.append(body)
            return next(replies)

        monkeypatch.setattr(backend, "_post", fake_post)
        completion = backend.run(_task())
        assert completion.data["hints"][0]["text"] == "look at line 4"
        assert len(seen) == 2 and "could not be parsed" in seen[1]["messages"][0]["content"]

    def test_unstructured_fallback_when_the_endpoint_rejects_the_schema(self, monkeypatch):
        backend = ClaudeBackend(self._config())
        calls = []

        def fake_post(body):
            calls.append(body)
            if "format" in body.get("output_config", {}):
                raise BackendError("HTTP 400: output_config.format unsupported", status=400)
            return {"content": '{"hints": []}', "reasoning": "", "model": "m",
                    "prompt_tokens": 0, "completion_tokens": 0}

        monkeypatch.setattr(backend, "_post", fake_post)
        task = Task(name="hints", system="s", instruction="i", payload={}, schema={"hints": list},
                    json_schema={"type": "object"})
        assert backend.run(task).data == {"hints": []}
        assert len(calls) == 2 and "format" not in calls[1].get("output_config", {})

    def test_persistent_garbage_degrades(self, monkeypatch):
        backend = ClaudeBackend(self._config())
        monkeypatch.setattr(backend, "_post", lambda body: {
            "content": "???", "reasoning": "", "model": "m", "prompt_tokens": 0, "completion_tokens": 0})
        with pytest.raises(BackendError):
            backend.run(_task())
