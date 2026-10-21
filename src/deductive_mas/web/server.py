"""HTTP for the tutoring service.

The standard library's threading server, a JSON API under /api and the
static front end. No framework, in keeping with the rest of the package:
`./dmas serve` works from a fresh checkout with nothing installed.

Every response carries a strict content security policy (everything from
this origin, nothing from anywhere else, so a learner's browser never talks
to a third party), and an error never carries more than a fixed sentence:
exception text stays in the server's log.
"""

import json
import re
import sys
import time
import traceback
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple
from urllib.parse import unquote, urlsplit

from ..cli.session import MAX_SOURCE_BYTES
from ..config import Config
from ..version import __version__
from .service import TutorService, WebError, WebSession

COOKIE = "dmas_session"
STATIC = Path(__file__).resolve().with_name("static")
MAX_BODY = MAX_SOURCE_BYTES + 64 * 1024

_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
_STATIC_NAME = re.compile(r"^[A-Za-z0-9_\-./]{1,200}$")
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webmanifest": "application/manifest+json",
}
_CSP = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "font-src 'self'",
        "manifest-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)
_SECURITY_HEADERS = (
    ("Content-Security-Policy", _CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()"),
)


def _log(message: str) -> None:
    sys.stderr.write(time.strftime("%H:%M:%S ") + message + "\n")
    sys.stderr.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: Tuple[str, int], service: TutorService, *, secure_cookie: bool) -> None:
        super().__init__(address, Handler)
        self.service = service
        self.secure_cookie = secure_cookie


Route = Tuple[str, Pattern[str], str]
_ROUTES: Tuple[Route, ...] = tuple(
    (method, re.compile(f"^{pattern}$"), name)
    for method, pattern, name in (
        ("GET", r"/api/health", "health"),
        ("GET", r"/api/meta", "meta"),
        ("GET", r"/api/tasks", "tasks"),
        ("GET", r"/api/examples", "examples"),
        ("GET", r"/api/session", "session"),
        ("GET", r"/api/concepts/([a-z0-9-]{1,64})", "concept"),
        ("POST", r"/api/session/reset", "reset"),
        ("POST", r"/api/attempts", "submit"),
        ("POST", r"/api/attempts/(\d{1,6})/hints", "hint"),
        ("POST", r"/api/attempts/(\d{1,6})/challenge", "challenge"),
        ("POST", r"/api/attempts/(\d{1,6})/challenge/run", "run"),
    )
)


class Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = "dmas"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # a client that stops sending mid-request does not hold a thread forever
    timeout = 30
    _cookie: Optional[str] = None
    _body_read = False

    # dispatch
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._refuse_method()

    do_DELETE = do_PATCH = do_PUT

    def _refuse_method(self) -> None:
        self._json(HTTPStatus.METHOD_NOT_ALLOWED,
                   {"error": {"code": "method", "message": "method not allowed"}})

    def _dispatch(self, method: str) -> None:
        self._cookie = None
        self._body_read = False
        path = urlsplit(self.path).path
        try:
            self._route(method, path)
        finally:
            # a body left unread would be parsed as the next request on this
            # keep-alive connection
            if method == "POST" and not self._body_read:
                self.close_connection = True

    def _route(self, method: str, path: str) -> None:
        try:
            if not path.startswith("/api/"):
                if method != "GET":
                    raise WebError(405, "method", "method not allowed")
                self._static(path)
                return
            for route_method, pattern, name in _ROUTES:
                match = pattern.match(path)
                if match is None:
                    continue
                if route_method != method:
                    continue
                body = self._body() if method == "POST" else {}
                payload = getattr(self, "api_" + name)(body, *match.groups())
                self._json(HTTPStatus.OK, payload)
                return
            if any(pattern.match(path) for _m, pattern, _n in _ROUTES):
                raise WebError(405, "method", "method not allowed")
            raise WebError(404, "not_found", "no such endpoint")
        except WebError as exc:
            extra = []
            if exc.status == 429 and "retry_after" in exc.detail:
                extra.append(("Retry-After", str(exc.detail["retry_after"])))
            self._json(exc.status, exc.payload(), extra)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            # the full story goes to the server's log, never to the client
            _log(f"error in {method} {path}\n" + traceback.format_exc())
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"code": "internal",
                           "message": "Something went wrong on our side. Your code is still in "
                                      "the editor; try again in a moment."}},
            )

    # api
    def api_health(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "version": __version__}

    def api_meta(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.server.service.meta()

    def api_tasks(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"tasks": self.server.service.tasks()}

    def api_examples(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"examples": self.server.service.examples()}

    def api_session(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"session": self.server.service.snapshot(self._session())}

    def api_concept(self, body: Dict[str, Any], cid: str) -> Dict[str, Any]:
        return {"concept": self.server.service.concept(cid, self._session(create=False))}

    def api_reset(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.server.service.reset(self._session())

    def api_submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.server.service.submit(self._session(), body.get("source"), body.get("task"))

    def api_hint(self, body: Dict[str, Any], index: str) -> Dict[str, Any]:
        return self.server.service.reveal(self._session(), int(index), body.get("rung"))

    def api_challenge(self, body: Dict[str, Any], index: str) -> Dict[str, Any]:
        return self.server.service.open_challenge(self._session(), int(index))

    def api_run(self, body: Dict[str, Any], index: str) -> Dict[str, Any]:
        return self.server.service.run_challenge(self._session(), int(index), body.get("source"))

    # request
    def _body(self) -> Dict[str, Any]:
        """A JSON object, from this origin, of bounded size."""
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise WebError(415, "content_type", "send JSON")
        # a cross-site form or fetch is refused even before the SameSite cookie is
        site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if site and site not in ("same-origin", "none"):
            raise WebError(403, "cross_site", "cross-site requests are refused")
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != (self.headers.get("Host") or ""):
            raise WebError(403, "cross_site", "cross-site requests are refused")
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise WebError(411, "length_required", "send a Content-Length")
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            self.close_connection = True
            raise WebError(411, "length_required", "send a Content-Length")
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            raise WebError(413, "too_large", f"The code is larger than {MAX_SOURCE_BYTES // 1024} KB.")
        raw = self.rfile.read(length)
        self._body_read = True
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError):
            raise WebError(400, "bad_json", "the request body is not JSON")
        if not isinstance(payload, dict):
            raise WebError(400, "bad_json", "the request body must be a JSON object")
        return payload

    def _session(self, *, create: bool = True) -> WebSession:
        store = self.server.service.sessions
        token = self._read_cookie()
        session = store.get(token)
        if session is None and create:
            session = store.create()
        if session is not None and session.token != token:
            self._cookie = session.token
        if session is None:
            # only a read that needs no state asks for create=False
            return WebSession(token="")
        return session

    def _read_cookie(self) -> Optional[str]:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            jar = SimpleCookie()
            jar.load(raw)
        except CookieError:
            return None
        morsel = jar.get(COOKIE)
        if morsel is None or not _TOKEN.match(morsel.value):
            return None
        return morsel.value

    # responses
    def _json(self, status: int, payload: Any, extra: Optional[List[Tuple[str, str]]] = None) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self.command == "POST" and not self._body_read:
            self.send_header("Connection", "close")
        for name, value in extra or ():
            self.send_header(name, value)
        if self._cookie:
            parts = [f"{COOKIE}={self._cookie}", "Path=/", "HttpOnly", "SameSite=Strict",
                     f"Max-Age={int(self.server.service.sessions.ttl)}"]
            if self.server.secure_cookie:
                parts.append("Secure")
            self.send_header("Set-Cookie", "; ".join(parts))
        self._security()
        self.end_headers()
        self.wfile.write(data)

    def _static(self, path: str) -> None:
        name = "index.html" if path in ("/", "/index.html") else None
        if name is None and path.startswith("/static/"):
            name = unquote(path[len("/static/"):])
        target = _static_file(name)
        if target is None:
            body = b"not found"
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._security()
            self.end_headers()
            self.wfile.write(body)
            return
        stat = target.stat()
        tag = f'"{int(stat.st_mtime)}-{stat.st_size}"'
        if self.headers.get("If-None-Match") == tag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", tag)
            self.send_header("Content-Length", "0")
            self._security()
            self.end_headers()
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _TYPES[target.suffix])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("ETag", tag)
        self._security()
        self.end_headers()
        self.wfile.write(data)

    def _security(self) -> None:
        for name, value in _SECURITY_HEADERS:
            self.send_header(name, value)

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: Any) -> None:
        _log(f"{self.address_string()} {format % args}")


def _static_file(name: Optional[str]) -> Optional[Path]:
    """A file inside the static directory, or None. No traversal, no dotfiles,
    only the types listed above.
    """
    if not name or not _STATIC_NAME.match(name):
        return None
    parts = name.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        return None
    target = (STATIC / name).resolve()
    if STATIC not in target.parents or not target.is_file() or target.suffix not in _TYPES:
        return None
    return target


def make_server(
    service: TutorService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    secure_cookie: bool = False,
) -> _Server:
    return _Server((host, port), service, secure_cookie=secure_cookie)


def serve(
    config: Optional[Config] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    demo: bool = False,
    secure_cookie: bool = False,
    isolate: bool = True,
    announce: Callable[[str], None] = print,
) -> int:
    service = TutorService(config or Config(), demo=demo, isolate=isolate)
    server = make_server(service, host=host, port=port, secure_cookie=secure_cookie)
    bound_host, bound_port = server.server_address[:2]
    shown = "127.0.0.1" if bound_host in ("0.0.0.0", "") else bound_host
    announce("")
    announce(f"  DMAS web  ·  http://{shown}:{bound_port}")
    announce(f"  reasoning {service.orchestrator.reasoner.name}  ·  "
             f"demo {'on' if demo else 'off'}  ·  "
             f"isolation {'on' if service.isolated else 'off'}  ·  ctrl-c to stop")
    if not service.isolated:
        announce(
            "  note: process isolation is off. The guarded interpreter is a step budget and a "
            "deadline, not a sandbox, so a crafted submission can hang or exhaust this process. "
            "Run like this only on a trusted network you are watching."
        )
    if bound_host not in ("127.0.0.1", "localhost", "::1"):
        announce(
            "  note: listening beyond this machine; see the README before exposing it to a network."
        )
    announce("")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        announce("\n  stopped")
    finally:
        server.server_close()
        service.close()
    return 0
