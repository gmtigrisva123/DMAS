import dataclasses
import http.client
import json
import socket
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from deductive_mas.config import Config
from deductive_mas.data.problems import all_problems, problem
from deductive_mas.data.submissions import submission
from deductive_mas.domain import Submission
from deductive_mas.llm.base import Completion, Task
from deductive_mas.llm.registry import Reasoner
from deductive_mas.agents.orchestrator import DeductiveOrchestrator
from deductive_mas.web import TutorService, make_server
from deductive_mas.web.jobs import build_context, run_diagnose
from deductive_mas.web.service import SessionStore, WebError, WebSession
from deductive_mas.web.views import notes_from_warnings, render_value, safe_equivalent

BROKEN = submission("lb_inclusive_bound").source
FIXED = BROKEN.replace("len(a) - 1", "len(a)")
REFERENCE = problem("lower_bound").reference_solution


class _LeakingBackend:
    """Does what the gate is there to stop, in every task it is given."""

    name = "leaking"

    def available(self) -> bool:
        return True

    def run(self, task: Task) -> Completion:
        if task.name == "hints":
            return Completion(backend=self.name, data={"hints": [
                {"kind": "orienting", "text": "Change the initialisation to `hi = len(a)`."},
                {"kind": "contradiction", "text": "Here is the fix:\n```python\n" + REFERENCE + "\n```"},
                {"kind": "conceptual", "text": "Just write `lo, hi = 0, len(a)` instead."},
            ]})
        if task.name == "challenge":
            return Completion(backend=self.name, data={
                "prompt": "Replace your initialisation with `hi = len(a)` and keep `while lo < hi:`.",
                "trace_question": "", "expected_insight": "SECRET-INSIGHT hi must be len(a)",
            })
        if task.name == "narrate":
            return Completion(backend=self.name, data={"narrative": "SECRET-NARRATIVE: set hi = len(a)"})
        return Completion(backend=self.name, data={
            "misconceptions": [{"id": "bs.interval-convention-mismatch", "confidence": 0.9,
                                "reason": "the fix is `lo, hi = 0, len(a)`"}],
            "summary": "",
        })


# a server on an ephemeral port, spoken to over real sockets
class Client:
    def __init__(self, port: int) -> None:
        self.port = port
        self.cookie: Optional[str] = None
        self.bodies: List[str] = []

    def raw(self, method: str, path: str, body: Optional[bytes] = None,
            headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], str]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            sent = dict(headers or {})
            if self.cookie and "Cookie" not in sent:
                sent["Cookie"] = f"dmas_session={self.cookie}"
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            text = response.read().decode("utf-8")
            found = {k.lower(): v for k, v in response.getheaders()}
            cookie = found.get("set-cookie")
            if cookie and cookie.startswith("dmas_session="):
                self.cookie = cookie.split(";")[0].split("=", 1)[1]
            self.bodies.append(text)
            return response.status, found, text
        finally:
            connection.close()

    def get(self, path: str) -> Tuple[int, Any]:
        status, _headers, text = self.raw("GET", path)
        return status, json.loads(text)

    def post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        data = json.dumps(payload or {}).encode("utf-8")
        status, _headers, text = self.raw(
            "POST", path, data, {"Content-Type": "application/json", "Content-Length": str(len(data))}
        )
        return status, json.loads(text)


def _serve(service: TutorService) -> Iterator[Tuple[TutorService, int]]:
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield service, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def served(config: Config) -> Iterator[Tuple[TutorService, int]]:
    yield from _serve(TutorService(config))


@pytest.fixture()
def client(served: Tuple[TutorService, int]) -> Client:
    return Client(served[1])


def _without_source(body: str) -> str:
    """A response with the learner's own code taken out: echoing what they
    wrote back to them is the point, and it may contain anything."""
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "source"}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    try:
        return json.dumps(strip(json.loads(body)))
    except ValueError:
        return body


def _reference_lines(learner: str = "") -> List[str]:
    """Every statement of every reference, except the ones the learner wrote
    too: quoting their own line back to them is allowed."""
    own = {line.strip() for line in learner.splitlines()}
    lines = []
    for spec in all_problems():
        for line in spec.reference_solution.splitlines():
            text = line.strip()
            if len(text) > 12 and text not in own and not text.startswith(("def ", "return", "from ", "else")):
                lines.append(text)
    return lines


class TestWhatTheBrowserMaySee:
    def test_task_list_has_the_statement_and_never_the_oracle(self, client: Client) -> None:
        status, body = client.get("/api/tasks")
        assert status == 200 and len(body["tasks"]) == len(all_problems())
        raw = json.dumps(body)
        for line in _reference_lines():
            assert line not in raw
        assert "reference" not in raw and "tests" not in raw

    def test_a_diagnosis_carries_the_evidence_but_not_the_ladder(self, client: Client) -> None:
        status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        assert status == 200 and body["unchanged"] is False
        current = body["session"]["current"]
        assert current["status"] == "diagnosis"
        assert current["findings"][0]["id"] == "bs.interval-convention-mismatch"
        assert current["witness"] == {"input": "([0], 1)", "yours": "0", "required": "1",
                                      "error": None, "steps": 3}
        assert current["divergence"]["line"] == 4
        assert current["ladder"] == {"total": 3, "revealed": []}
        assert current["challenge"] == {"available": True, "shown": False}
        raw = json.dumps(body)
        for forbidden in ("reference_state", "should be", "narrative", "gate_log", "expected_insight",
                          "Take the input", "At that point", "lp-invariant"):
            assert forbidden not in raw, forbidden

    def test_rungs_open_one_at_a_time(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        status, body = client.post("/api/attempts/1/hints", {"rung": 2})
        assert status == 409 and body["error"]["next"] == 1
        status, first = client.post("/api/attempts/1/hints", {"rung": 1})
        assert status == 200 and first["hint"]["rung"] == 1 and first["hint"]["kind"] == "orienting"
        assert "At that point" not in json.dumps(first)
        # asking again for a shown rung changes nothing
        _status, again = client.post("/api/attempts/1/hints", {"rung": 1})
        assert again["hint"] == first["hint"]
        assert len(again["session"]["current"]["ladder"]["revealed"]) == 1
        assert client.post("/api/attempts/1/hints", {"rung": 3})[0] == 409
        client.post("/api/attempts/1/hints", {"rung": 2})
        _status, last = client.post("/api/attempts/1/hints", {"rung": 3})
        assert last["hint"]["last"] is True
        _status, past = client.post("/api/attempts/1/hints", {"rung": 4})
        assert past["status"] == "exhausted"

    @pytest.mark.parametrize("rung", [0, -1, True, "1", None, 1.0])
    def test_a_rung_must_be_a_positive_integer(self, client: Client, rung: Any) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        assert client.post("/api/attempts/1/hints", {"rung": rung})[0] == 400

    def test_the_challenge_opens_on_request_and_runs_the_editor(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        status, refused = client.post("/api/attempts/1/challenge/run", {"source": BROKEN})
        assert status == 409 and refused["error"]["code"] == "challenge_closed"
        _status, opened = client.post("/api/attempts/1/challenge")
        challenge = opened["challenge"]
        assert challenge["runnable"] and challenge["input"] == "([0], 1)"
        assert "expected_insight" not in json.dumps(opened)
        _status, ran = client.post("/api/attempts/1/challenge/run", {"source": BROKEN})
        assert ran["run"] == {"input": "([0], 1)", "ok": True, "value": "0", "error": None,
                              "budget": False, "required": "1", "agrees": False}
        _status, ran = client.post("/api/attempts/1/challenge/run", {"source": FIXED})
        assert ran["run"]["value"] == "1" and ran["run"]["agrees"] is True
        # a run is not an attempt
        assert len(client.get("/api/session")[1]["session"]["attempts"]) == 1

    def test_guidance_belongs_to_the_latest_attempt(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        client.post("/api/attempts", {"source": FIXED, "task": "lower_bound"})
        status, body = client.post("/api/attempts/1/hints", {"rung": 1})
        assert status == 409 and body["error"] == {
            "code": "stale_attempt", "message": "Guidance belongs to the latest attempt only.", "latest": 2}


class TestTheSessionContract:
    """The terminal session's behaviour, through HTTP."""

    def test_an_unchanged_retry_is_not_an_attempt(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        assert status == 200 and body["unchanged"] is True
        assert len(body["session"]["attempts"]) == 1

    def test_a_fix_is_recognised_and_credited(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        client.post("/api/attempts/1/hints", {"rung": 1})
        _status, body = client.post("/api/attempts", {"source": FIXED, "task": "lower_bound"})
        assert body["retry"] is True
        current = body["session"]["current"]
        assert current["status"] == "clean" and current["checks"] == {"passed": 87, "total": 87}
        diff = current["diff"]
        assert [m["id"] for m in diff["resolved"]] == ["bs.interval-convention-mismatch"]
        assert diff["still_failing"] is False
        assert diff["checks"]["before"] == {"passed": 5, "total": 7}
        assert diff["mastery"]["id"] == "loop-invariant"
        assert diff["mastery"]["after"] > diff["mastery"]["before"]
        # the new attempt starts its own ladder
        assert current["ladder"]["revealed"] == []

    def test_mastery_lists_only_what_the_session_saw(self, client: Client) -> None:
        _status, body = client.get("/api/session")
        assert body["session"]["mastery"]["concepts"] == []
        _status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        concepts = body["session"]["mastery"]["concepts"]
        assert 0 < len(concepts) < 65
        assert all(c["observations"] > 0 for c in concepts)
        assert concepts == sorted(concepts, key=lambda c: (c["value"], c["id"]))

    def test_auto_detects_the_task_and_falls_back_to_structure(self, client: Client) -> None:
        _status, body = client.post("/api/attempts", {"source": BROKEN, "task": "auto"})
        current = body["session"]["current"]
        assert current["task"]["id"] == "lower_bound"
        assert current["notes"][0]["kind"] == "task"
        unknown = "def walk(graph, start):\n    seen = []\n    for n in graph[start]:\n        if n not in seen:\n            seen.append(n)\n    return seen\n"
        _status, body = client.post("/api/attempts", {"source": unknown, "task": "auto"})
        current = body["session"]["current"]
        assert current["task"] == {"id": "freeform", "title": "Any function", "freeform": True,
                                   "entry_point": "walk"}
        assert current["witness"] is None and current["complexity"]["target"] is None

    def test_reset_forgets_attempts_and_mastery(self, client: Client) -> None:
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        _status, body = client.post("/api/session/reset")
        session = body["session"]
        assert session["attempts"] == [] and session["current"] is None
        assert session["mastery"]["concepts"] == []

    def test_a_still_failing_retry_is_not_called_a_repair(self, client: Client) -> None:
        looping = FIXED.replace("lo = mid + 1", "lo = mid")
        _status, first = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        root = first["session"]["current"]["alignment"]["root_cause"]["id"]
        _status, body = client.post("/api/attempts", {"source": looping, "task": "lower_bound"})
        current = body["session"]["current"]
        assert current["status"] == "failing"
        assert current["diff"]["still_failing"] is True
        # and the belief is not credited as repaired: the concept stays unsecured
        concepts = {c["id"]: c for c in body["session"]["mastery"]["concepts"]}
        assert root in concepts and concepts[root]["secure"] is False


class TestWhatIsNotAnAttempt:
    def test_code_that_does_not_parse(self, client: Client) -> None:
        status, body = client.post("/api/attempts", {
            "source": "def lower_bound(a, t):\n    while a\n        pass\n", "task": "lower_bound"})
        assert status == 422
        assert body["error"]["code"] == "syntax_error" and body["error"]["line"] == 2
        assert client.get("/api/session")[1]["session"]["attempts"] == []

    def test_what_only_the_compiler_rejects(self, client: Client) -> None:
        status, body = client.post("/api/attempts", {"source": "def f():\n    pass\nreturn 1\n", "task": "auto"})
        assert status == 422 and body["error"]["line"] == 3

    @pytest.mark.parametrize("source, code", [("", "empty"), ("   \n", "empty"), ("x = 1\n", "no_function")])
    def test_nothing_to_call(self, client: Client, source: str, code: str) -> None:
        status, body = client.post("/api/attempts", {"source": source, "task": "auto"})
        assert status == 422 and body["error"]["code"] == code

    def test_too_large(self, client: Client) -> None:
        source = "def f():\n    return 1\n" + "# " + "x" * (300 * 1024) + "\n"
        status, _headers, _text = client.raw("POST", "/api/attempts", json.dumps(
            {"source": source, "task": "auto"}).encode(), {"Content-Type": "application/json"})
        assert status == 413

    def test_an_unknown_task(self, client: Client) -> None:
        status, body = client.post("/api/attempts", {"source": BROKEN, "task": "nope"})
        assert status == 404 and body["error"]["code"] == "unknown_task"


class TestTheGateHoldsAtTheBoundary:
    """A backend that hands over the answer at every opportunity. Nothing it
    says may reach any response, and the guidance must still be usable."""

    @pytest.fixture()
    def leaky(self, config: Config) -> Iterator[Tuple[TutorService, int]]:
        orchestrator = DeductiveOrchestrator(config, reasoner=Reasoner(config.llm, backend=_LeakingBackend()))
        yield from _serve(TutorService(config, orchestrator=orchestrator))

    def test_nothing_leaked_reaches_any_response(self, leaky: Tuple[TutorService, int]) -> None:
        service, port = leaky
        client = Client(port)
        client.get("/api/session")
        _status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        notes = body["session"]["current"]["notes"]
        assert any(note["kind"] == "gate" for note in notes)
        for rung in (1, 2, 3):
            status, revealed = client.post("/api/attempts/1/hints", {"rung": rung})
            assert status == 200 and len(revealed["hint"]["text"]) > 40
        client.post("/api/attempts/1/challenge")
        client.post("/api/attempts/1/challenge/run", {"source": BROKEN})
        client.get("/api/session")

        # the leaking backend really produced this attempt; the raw result,
        # its gate log and the rejected candidates never existed in this
        # process -- they lived and died in the analysis child.
        assert service.meta()["backend"] == "leaking"
        assert body["session"]["current"]["backend"] == "leaking"

        everything = "\n".join(_without_source(body) for body in client.bodies)
        for leaked in (
            "hi = len(a)", "lo, hi = 0, len(a)", "```", "Here is the fix", "Just write",
            "SECRET-INSIGHT", "SECRET-NARRATIVE", "the fix is", "expected_insight",
            "gate_log", "candidate", "exceeds the limit", "reproduces a statement",
        ):
            assert leaked not in everything, leaked
        for line in _reference_lines(BROKEN):
            assert line not in everything, line

    def test_a_model_reason_is_replaced_not_shown(self, leaky: Tuple[TutorService, int]) -> None:
        _service, port = leaky
        client = Client(port)
        _status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        evidence = body["session"]["current"]["evidence"]
        model = [item for item in evidence if item["source"] == "model"]
        assert model and all(item["text"].startswith("withheld") for item in model)


class TestHostileValues:
    def test_render_value_never_calls_student_methods(self) -> None:
        class Loud:
            def __repr__(self) -> str:
                raise AssertionError("repr was called")

            def __eq__(self, other: object) -> bool:
                raise AssertionError("eq was called")

            __hash__ = object.__hash__

        assert render_value(Loud()) == "<Loud object>"
        assert render_value([1, (2, "x"), {"k": Loud()}]) == "[1, (2, 'x'), {'k': <Loud object>}]"
        assert render_value((1,)) == "(1,)" and render_value(set()) == "set()"
        assert render_value(10 ** 400).startswith("<int of")
        assert not safe_equivalent(Loud(), 1) and not safe_equivalent(1, Loud())
        assert safe_equivalent([1, 2.0], (1, 2)) and not safe_equivalent(True, 1)
        # a dict keyed by a student object must compare without hashing or
        # eq-ing that key (both raise here); it is simply reported unequal
        assert safe_equivalent({"k": 1, 2: "v"}, {2: "v", "k": 1}) is True
        assert safe_equivalent({"k": 1}, {"k": 2}) is False
        assert safe_equivalent({Loud(): 1}, {Loud(): 1}) is False
        assert safe_equivalent({"k": 1}, {Loud(): 1}) is False

    def test_an_exception_text_never_leaves_the_server(self, client: Client) -> None:
        # the core reprs a returned value outside the guard; a raising __repr__
        # fails that stage. What the learner sees is a fixed sentence.
        source = ("class Loud:\n    def __repr__(self):\n        raise ValueError('SECRET-TRACE')\n"
                  "def lower_bound(a, t):\n    return Loud()\n")
        status, body = client.post("/api/attempts", {"source": source, "task": "lower_bound"})
        assert status == 200
        assert "SECRET-TRACE" not in _without_source(json.dumps(body))
        current = body["session"]["current"]
        assert current["status"] == "incomplete"
        assert any(note["kind"] == "stage" for note in current["notes"])

    def test_print_does_not_reach_the_server_log(self, client: Client, capsys: pytest.CaptureFixture[str]) -> None:
        source = "def lower_bound(a, t):\n    print('NOISE-FROM-STUDENT')\n    return 0\n"
        client.post("/api/attempts", {"source": source, "task": "lower_bound"})
        assert "NOISE-FROM-STUDENT" not in capsys.readouterr().out

    def test_concurrent_learners_leave_the_recursion_limit_alone(self, served: Tuple[TutorService, int]) -> None:
        before = sys.getrecursionlimit()
        failures: List[int] = []

        def learner() -> None:
            client = Client(served[1])
            status, _body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
            if status != 200:
                failures.append(status)

        threads = [threading.Thread(target=learner) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert failures == [] and sys.getrecursionlimit() == before


class TestHttp:
    def test_the_session_cookie(self, served: Tuple[TutorService, int]) -> None:
        client = Client(served[1])
        _status, headers, _text = client.raw("GET", "/api/session")
        cookie = headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
        assert client.cookie and len(client.cookie) >= 32
        # a known cookie is not reissued, an unknown one is replaced
        assert "set-cookie" not in client.raw("GET", "/api/session")[1]
        client.cookie = "x" * 43
        client.raw("GET", "/api/session")
        assert client.cookie != "x" * 43

    def test_security_headers_everywhere(self, client: Client) -> None:
        for path in ("/", "/static/app.css", "/api/meta", "/api/nope"):
            _status, headers, _text = client.raw("GET", path)
            assert headers["content-security-policy"].startswith("default-src 'none'")
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["referrer-policy"] == "no-referrer"
            assert headers["server"] == "dmas"

    def test_cross_site_and_non_json_posts_are_refused(self, client: Client) -> None:
        body = json.dumps({"source": BROKEN, "task": "auto"}).encode()
        assert client.raw("POST", "/api/attempts", body, {"Content-Type": "text/plain"})[0] == 415
        assert client.raw("POST", "/api/attempts", body, {
            "Content-Type": "application/json", "Origin": "https://evil.example"})[0] == 403
        assert client.raw("POST", "/api/attempts", body, {
            "Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"})[0] == 403
        assert client.raw("POST", "/api/attempts", b"[1, 2]", {"Content-Type": "application/json"})[0] == 400
        assert client.raw("POST", "/api/attempts", b"{nope", {"Content-Type": "application/json"})[0] == 400

    @pytest.mark.parametrize("path", [
        "/static/../server.py", "/static/%2e%2e/server.py", "/static/..%2fserver.py",
        "/static/js/../../service.py", "/static/.secret", "/static/", "/server.py", "/static/js",
    ])
    def test_static_files_stay_inside_the_static_directory(self, served: Tuple[TutorService, int], path: str) -> None:
        with socket.create_connection(("127.0.0.1", served[1]), timeout=10) as raw:
            raw.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
            response = b""
            while chunk := raw.recv(65536):
                response += chunk
        assert response.startswith(b"HTTP/1.1 404")
        assert b"def " not in response

    def test_the_front_end_is_served(self, client: Client) -> None:
        status, headers, text = client.raw("GET", "/")
        assert status == 200 and headers["content-type"].startswith("text/html")
        assert '<script type="module" src="/static/js/main.js">' in text
        status, headers, _text = client.raw("GET", "/static/js/main.js")
        assert status == 200 and headers["content-type"].startswith("text/javascript")

    def test_a_refused_post_does_not_poison_the_connection(self, served: Tuple[TutorService, int]) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", served[1], timeout=10)
        try:
            body = b'{"source": "def f(): pass", "task": "auto"}'
            connection.request("POST", "/api/nope", body=body, headers={"Content-Type": "application/json"})
            first = connection.getresponse()
            first.read()
            assert first.status == 404 and first.getheader("Connection") == "close"
        finally:
            connection.close()

    def test_a_failed_projection_leaves_no_half_attempt(self, served: Tuple[TutorService, int],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        service, port = served
        client = Client(port)

        def broken(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("projection failed")

        monkeypatch.setattr(service, "_build_projection", broken)
        assert client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})[0] == 500
        monkeypatch.undo()
        status, body = client.get("/api/session")
        assert status == 200 and body["session"]["attempts"] == []
        assert body["session"]["mastery"]["concepts"] == []
        assert client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})[0] == 200

    def test_unknown_endpoints_and_methods(self, client: Client) -> None:
        assert client.get("/api/nope")[0] == 404
        assert client.raw("GET", "/api/attempts")[0] == 405
        assert client.raw("PUT", "/api/session")[0] == 405

    def test_an_internal_error_is_a_fixed_sentence(self, served: Tuple[TutorService, int],
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
        service, port = served

        def explode(_session: WebSession) -> Dict[str, Any]:
            raise RuntimeError("SECRET " + REFERENCE)

        monkeypatch.setattr(service, "snapshot", explode)
        status, _headers, text = Client(port).raw("GET", "/api/session")
        assert status == 500 and "SECRET" not in text and "len(a)" not in text
        assert json.loads(text)["error"]["code"] == "internal"
        assert "SECRET" in capsys.readouterr().err

    def test_examples_only_in_demo_mode(self, client: Client, config: Config) -> None:
        assert client.get("/api/examples")[0] == 404
        examples = TutorService(config, demo=True, isolate=False).examples()
        assert examples and all(not submission(e["id"]).functionally_correct for e in examples)

    def test_concepts(self, client: Client) -> None:
        status, body = client.get("/api/concepts/loop-invariant")
        assert status == 200 and body["concept"]["name"] == "Loop invariants"
        assert client.get("/api/concepts/not-a-concept")[0] == 404


class TestUnits:
    def test_a_gate_rejection_never_quotes_the_rejected_text(self) -> None:
        notes = notes_from_warnings([
            "hint rung 2 replaced: unsupported claim(s) below the grounding floor of 0.40: SECRET",
            "reasoning backend degraded: HTTP 500 SECRET body",
            "socratic-evaluator failed: ValueError: SECRET",
            "model proposed an unknown misconception: 'SECRET'",
        ])
        assert [note["kind"] for note in notes] == ["gate", "backend", "stage"]
        assert "SECRET" not in json.dumps(notes)

    def test_the_rate_limit(self) -> None:
        from deductive_mas.web.service import WebError

        session = WebSession(token="t")
        for second in range(20):
            session.spend(1000.0 + second * 0.1)
        with pytest.raises(WebError) as refused:
            session.spend(1002.5)
        assert refused.value.status == 429 and refused.value.detail["retry_after"] >= 1
        session.spend(1061.0)

    def test_the_serve_command_is_registered(self) -> None:
        from deductive_mas.cli.main import build_parser

        args = build_parser().parse_args(["serve", "--port", "0", "--demo"])
        assert args.port == 0 and args.demo and args.host == "127.0.0.1"
        assert args.no_sandbox is False
        assert build_parser().parse_args(["serve", "--no-sandbox"]).no_sandbox is True

    def test_a_wall_timeout_reads_as_a_service_error_not_a_learner_fault(self, config: Config) -> None:
        from deductive_mas.web.sandbox import Outcome

        service = TutorService(config, isolate=False)
        try:
            with pytest.raises(WebError) as refused:
                service._demand(Outcome("timeout"), diagnosing=True)
            assert refused.value.status == 503 and refused.value.code == "overloaded"
            assert "not the problem" in refused.value.message
            run = service._demand(Outcome("timeout"), diagnosing=False)
            assert run["run"]["budget"] is True
        finally:
            service.close()

    def test_the_store_does_not_evict_a_session_in_flight(self) -> None:
        store = SessionStore(capacity=2)
        a, b = store.create(), store.create()
        a.touched = time.time() - 100  # oldest, as a blocked diagnosis would be, but not TTL-expired
        held, release = threading.Event(), threading.Event()

        def hold_a() -> None:  # a request thread mid-diagnosis on A
            with a.lock:
                held.set()
                release.wait(5)

        worker = threading.Thread(target=hold_a)
        worker.start()
        held.wait(5)
        try:
            store.create()  # at capacity: must skip the locked A and evict B
            assert store.get(a.token) is a
            assert store.get(b.token) is None
        finally:
            release.set()
            worker.join(5)


# code that the step budget and the deadline inside the interpreter cannot
# stop: an operation that runs to completion in one C call, and a method the
# student defined that the pipeline calls outside the trace hook. On a shared
# server these must be contained by the operating system, in a child that can
# be killed, without disturbing anyone else.
HOSTILE_REPR = (
    "class B:\n"
    "    def __repr__(self):\n"
    "        while True:\n"
    "            pass\n\n"
    "def lower_bound(a, x):\n"
    "    return B()\n"
)
CPU_BOMB = "def lower_bound(a, x):\n    return 10 ** (10 ** 8)\n"
MEM_BOMB = "def lower_bound(a, x):\n    big = [0] * (10 ** 10)\n    return len(big)\n"


def _hardened_config(config: Config) -> Config:
    """Small ceilings so the kills happen quickly in the test."""
    return dataclasses.replace(config, execution=dataclasses.replace(
        config.execution,
        sandbox_cpu_seconds=2.0, sandbox_wall_seconds=3.0,
        sandbox_address_space_mb=512, sandbox_workers=3,
    ))


@pytest.fixture(scope="module")
def hardened(config: Config) -> Iterator[Tuple[TutorService, int]]:
    yield from _serve(TutorService(_hardened_config(config)))


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
class TestTheSandbox:
    def test_it_really_is_isolated(self, hardened: Tuple[TutorService, int]) -> None:
        service, port = hardened
        assert service.isolated is True
        assert Client(port).get("/api/meta")[1]["isolated"] is True

    def test_a_c_level_cpu_bomb_is_killed_and_is_not_an_attempt(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        client = Client(port)
        start = time.monotonic()
        status, body = client.post("/api/attempts", {"source": CPU_BOMB, "task": "lower_bound"})
        assert status == 422 and body["error"]["code"] == "did_not_finish"
        assert time.monotonic() - start < 8.0  # the ~2s ceiling, not left to run
        assert client.get("/api/session")[1]["session"]["attempts"] == []

    def test_a_hostile_repr_outside_the_trace_is_killed(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        client = Client(port)
        start = time.monotonic()
        status, body = client.post("/api/attempts", {"source": HOSTILE_REPR, "task": "lower_bound"})
        assert status == 422 and body["error"]["code"] == "did_not_finish"
        assert time.monotonic() - start < 8.0

    def test_a_memory_bomb_cannot_exhaust_the_host(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        client = Client(port)
        # either refused as over-budget or caught as a failing run; never a
        # host out-of-memory, and the server keeps serving.
        status, _body = client.post("/api/attempts", {"source": MEM_BOMB, "task": "lower_bound"})
        assert status in (200, 422)
        assert client.get("/api/meta")[0] == 200

    def test_a_runaway_does_not_block_other_learners(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        running = threading.Event()

        def runaway() -> None:
            running.set()
            Client(port).post("/api/attempts", {"source": HOSTILE_REPR, "task": "lower_bound"})

        thread = threading.Thread(target=runaway, daemon=True)
        thread.start()
        running.wait(5)
        time.sleep(0.2)  # let the runaway take a worker and start spinning
        start = time.monotonic()
        status, _body = Client(port).post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        elapsed = time.monotonic() - start
        assert status == 200
        assert elapsed < 2.0  # answered while the runaway is still being killed
        thread.join(timeout=10)

    def test_a_challenge_run_that_overruns_is_a_budget_hit_not_a_crash(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        client = Client(port)
        client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        client.post("/api/attempts/1/challenge")
        status, ran = client.post("/api/attempts/1/challenge/run", {"source": CPU_BOMB})
        assert status == 200
        assert ran["run"]["ok"] is False and ran["run"]["budget"] is True

    def test_the_server_keeps_serving_after_a_worker_is_killed(
        self, hardened: Tuple[TutorService, int]
    ) -> None:
        _service, port = hardened
        client = Client(port)
        assert client.post("/api/attempts", {"source": CPU_BOMB, "task": "lower_bound"})[0] == 422
        status, body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
        assert status == 200 and body["session"]["current"]["status"] == "diagnosis"

    def test_jobs_queued_behind_the_worker_cap_still_succeed(self, config: Config) -> None:
        # with one worker, concurrent analyses serialise in the zygote's queue;
        # a queued-but-valid job must come back as a real result, never a false
        # time-limit (the parent no longer imposes its own deadline).
        hardened = dataclasses.replace(config, execution=dataclasses.replace(
            config.execution, sandbox_workers=1))
        service = TutorService(hardened)
        port = None
        gen = _serve(service)
        _service, port = next(gen)
        try:
            results: List[int] = []
            lock = threading.Lock()

            def learner() -> None:
                client = Client(port)
                status, _body = client.post("/api/attempts", {"source": BROKEN, "task": "lower_bound"})
                with lock:
                    results.append(status)

            threads = [threading.Thread(target=learner) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            assert results == [200] * 5
        finally:
            try:
                next(gen)
            except StopIteration:
                pass

    def test_a_wall_clock_kill_is_a_service_timeout_not_the_learners_fault(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a worker stuck NOT on CPU (here a sleep standing in for a slow model
        # call) is killed by the wall, not RLIMIT_CPU. That must read as a
        # service timeout, never as the learner's infinite loop.
        import deductive_mas.web.sandbox as sandbox_module
        from deductive_mas.web.jobs import build_context

        def sleeper(_ctx: Any, _payload: Dict[str, Any]) -> Dict[str, Any]:
            import time as _time
            _time.sleep(8)
            return {}

        monkeypatch.setitem(sandbox_module._DISPATCH, "diagnose", sleeper)
        cfg = dataclasses.replace(
            config,
            llm=dataclasses.replace(config.llm, timeout=0.0),
            execution=dataclasses.replace(
                config.execution, sandbox_cpu_seconds=1, sandbox_wall_seconds=2, sandbox_workers=2),
        )
        sandbox = sandbox_module.ForkSandbox(build_context(cfg), cfg)
        try:
            outcome = sandbox.diagnose({"source": "x"})
            assert outcome.status == "timeout"
        finally:
            sandbox.close()

    def test_a_result_landing_near_the_deadline_is_delivered_not_killed(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the deadline sweep must drain a finished worker's pipe before killing
        # it: a job that returns shortly before its wall comes back as a real
        # result, never a spurious timeout.
        import deductive_mas.web.sandbox as sandbox_module
        from deductive_mas.web.jobs import build_context

        def almost(_ctx: Any, _payload: Dict[str, Any]) -> Dict[str, Any]:
            import time as _time
            _time.sleep(3)
            return {"landed": True}

        monkeypatch.setitem(sandbox_module._DISPATCH, "diagnose", almost)
        cfg = dataclasses.replace(
            config,
            llm=dataclasses.replace(config.llm, timeout=0.0),
            execution=dataclasses.replace(
                config.execution, sandbox_cpu_seconds=1, sandbox_wall_seconds=2, sandbox_workers=2),
        )  # wall = max(2, 0 + 1 + 5) = 6s; the job returns at ~3s
        sandbox = sandbox_module.ForkSandbox(build_context(cfg), cfg)
        try:
            outcome = sandbox.diagnose({"source": "x"})
            assert outcome.status == "ok" and outcome.result == {"landed": True}
        finally:
            sandbox.close()

    def test_a_down_sandbox_is_reported_not_hidden(self, config: Config) -> None:
        service = TutorService(_hardened_config(config))
        try:
            service.sandbox.close()
            session = service.sessions.create()
            with pytest.raises(WebError) as refused:
                service.submit(session, BROKEN, "lower_bound")
            assert refused.value.status == 503 and refused.value.code == "unavailable"
        finally:
            service.close()


class TestTheInProcessFallback:
    def test_it_runs_the_same_pipeline_without_a_child(self, config: Config) -> None:
        service = TutorService(config, isolate=False)
        try:
            assert service.isolated is False and service.meta()["isolated"] is False
            session = service.sessions.create()
            body = service.submit(session, BROKEN, "lower_bound")
            assert body["session"]["current"]["status"] == "diagnosis"
        finally:
            service.close()


class TestTheChildIsTheBoundary:
    def test_a_leaking_result_becomes_a_clean_projection_in_the_child(self, config: Config) -> None:
        orchestrator = DeductiveOrchestrator(
            config, reasoner=Reasoner(config.llm, backend=_LeakingBackend())
        )
        context = build_context(config, orchestrator)
        spec = problem("lower_bound")

        # the raw result really does carry the answer and the gate's log of
        # rejected candidates -- this is what must never leave the child.
        raw = orchestrator.tutor(
            spec, Submission(sid="x", problem_id="lower_bound", source=BROKEN, author="t")
        )
        assert any(entry["candidate"] for entry in raw.gate_log)

        out = run_diagnose(context, {
            "source": BROKEN, "task": ("known", "lower_bound"), "index": 1,
            "mastery": None, "previous": None, "notes": [],
        })
        blob = _without_source(json.dumps(out))
        for leaked in (
            "hi = len(a)", "lo, hi = 0, len(a)", "SECRET-INSIGHT", "SECRET-NARRATIVE",
            "gate_log", "candidate", "expected_insight", "narrative",
        ):
            assert leaked not in blob, leaked
        for line in _reference_lines(BROKEN):
            assert line not in blob, line
