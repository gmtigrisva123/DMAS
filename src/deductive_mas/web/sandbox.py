"""Running student code in a child that can be killed.

The guarded interpreter stops a loop it can see: it counts the lines a
submission executes and checks a deadline on each one. Two things run where
it cannot count them -- an operation that finishes inside a single C call
(``10 ** 10 ** 8``, a huge allocation) and a method the student defined that
the pipeline calls directly (``__repr__``, ``__eq__``). In the terminal that
is the student's own machine and their own problem. On a shared server it is
not: one such submission would spin a core or exhaust memory for everyone.

So the server runs every analysis in a child process with hard ceilings the
operating system enforces, not the interpreter:

- a long-lived *zygote* is forked once, while the parent is single-threaded,
  and holds the built orchestrator. It runs no student code itself.
- for each job the zygote forks a *worker*, which sets RLIMIT_CPU and
  RLIMIT_AS on itself, runs the one job, writes the result to a pipe and
  exits. A worker that overruns is sent SIGKILL by the zygote; its death
  frees a core and its memory and touches nothing else.
- the HTTP process never forks (forking a threaded process is unsafe); it
  speaks to the zygote over a socket and correlates replies by job id.

Only plain data crosses the socket, built by jobs.py/views.py, which read
student values without calling their methods. The reasoning model's chain of
thought, the gate's rejected candidates and the reference solution never
leave the worker.

Where fork is not available, or the operator asks for it, an in-process
fallback runs the same jobs under a lock. That mode is honest about what it
is: no ceiling can be enforced, so it is for a supervised, trusted-network
pilot only.
"""

import os
import pickle
import select
import signal
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from ..config import Config
from .jobs import SandboxContext, run_challenge, run_diagnose

_HEADER = struct.Struct(">I")
_MAX_FRAME = 8 * 1024 * 1024
_DISPATCH = {"diagnose": run_diagnose, "challenge": run_challenge}


@dataclass
class Outcome:
    """What one job came back as. `ok` carries the job's plain-data result;
    `limit` means the worker was killed for overrunning its CPU or memory
    ceiling; `crash` means it died some other way; `down` means the sandbox
    itself is gone.
    """

    status: str  # "ok" | "limit" | "crash" | "down"
    result: Optional[Dict[str, Any]] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class _FrameReader:
    """Reassembles length-prefixed pickles from a byte stream."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._need: Optional[int] = None

    def feed(self, chunk: bytes) -> List[Any]:
        self._buf.extend(chunk)
        out: List[Any] = []
        while True:
            if self._need is None:
                if len(self._buf) < _HEADER.size:
                    break
                (self._need,) = _HEADER.unpack(self._buf[: _HEADER.size])
                del self._buf[: _HEADER.size]
                if self._need > _MAX_FRAME:
                    raise ValueError("frame too large")
            if len(self._buf) < self._need:
                break
            body = bytes(self._buf[: self._need])
            del self._buf[: self._need]
            self._need = None
            out.append(pickle.loads(body))
        return out


def _encode(obj: Any) -> bytes:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(body)) + body


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


# the sandbox the HTTP process holds
class ForkSandbox:
    """The parent-side handle: forks the zygote and talks to it."""

    def __init__(self, context: SandboxContext, config: Config) -> None:
        execution = config.execution
        # RLIMIT_CPU is the ceiling that matters for student code, which is
        # purely CPU-bound here: it can neither sleep nor do I/O (no `time`,
        # no sockets, only an allowlist of pure imports). The wall-clock is a
        # backstop for a worker stuck *not* burning CPU -- in practice only a
        # slow reasoning-model HTTP call, trusted code -- so it must sit above
        # that call's own timeout or it would kill a legitimate diagnosis.
        self.wall = max(
            execution.sandbox_wall_seconds,
            config.llm.timeout + execution.sandbox_cpu_seconds + 5.0,
        )
        limits = {
            "cpu": execution.sandbox_cpu_seconds,
            "address_space": execution.sandbox_address_space_mb * 1024 * 1024,
            "wall": self.wall,
            "workers": max(1, execution.sandbox_workers),
        }
        parent_sock, child_sock = _socketpair()
        pid = os.fork()
        if pid == 0:  # the zygote
            try:
                parent_sock.close()
                _Zygote(child_sock, context, limits).run()
            finally:
                os._exit(0)
        child_sock.close()
        self._sock = parent_sock
        self._pid = pid
        self._counter = 0
        self._alive = True
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._reader = threading.Thread(target=self._read_loop, name="dmas-sandbox", daemon=True)
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._alive

    def diagnose(self, payload: Dict[str, Any]) -> Outcome:
        return self._submit("diagnose", payload)

    def challenge(self, payload: Dict[str, Any]) -> Outcome:
        return self._submit("challenge", payload)

    def _submit(self, kind: str, payload: Dict[str, Any]) -> Outcome:
        if not self._alive:
            return Outcome("down", message="the analysis sandbox is not running")
        event = threading.Event()
        with self._lock:
            if not self._alive:
                return Outcome("down", message="the analysis sandbox is not running")
            self._counter += 1
            job_id = self._counter
            self._pending[job_id] = {"event": event, "outcome": None}
        try:
            with self._write_lock:
                self._sock.sendall(_encode({"id": job_id, "kind": kind, "payload": payload}))
        except OSError:
            self._fail_all("the analysis sandbox stopped responding")
            return Outcome("down", message="the analysis sandbox is not running")
        # the zygote enforces its own wall-clock deadline and always replies;
        # this wait is only a backstop against the channel itself wedging.
        if not event.wait(self.wall + 10.0):
            with self._lock:
                self._pending.pop(job_id, None)
            return Outcome("limit", message="the analysis took too long")
        with self._lock:
            slot = self._pending.pop(job_id, None)
        return (slot or {}).get("outcome") or Outcome("down", message="no reply")

    def _read_loop(self) -> None:
        reader = _FrameReader()
        try:
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                for frame in reader.feed(chunk):
                    self._resolve(frame)
        except (OSError, ValueError, pickle.PickleError):
            pass
        self._fail_all("the analysis sandbox stopped responding")

    def _resolve(self, frame: Dict[str, Any]) -> None:
        with self._lock:
            slot = self._pending.get(frame.get("id"))
            if slot is None:
                return
            slot["outcome"] = Outcome(
                frame.get("status", "crash"),
                result=frame.get("result"),
                message=frame.get("message", ""),
            )
            slot["event"].set()

    def _fail_all(self, message: str) -> None:
        with self._lock:
            self._alive = False
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            if slot.get("outcome") is None:
                slot["outcome"] = Outcome("down", message=message)
            slot["event"].set()

    def close(self) -> None:
        with self._lock:
            self._alive = False
        import socket

        # shut the socket down both ways first: a recv() blocked in the reader
        # thread would otherwise keep the peer from seeing the close, and the
        # zygote waits on exactly that EOF to exit.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._pid:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    done, _ = os.waitpid(self._pid, os.WNOHANG)
                except OSError:
                    break
                if done:
                    break
                time.sleep(0.02)
            else:
                self._force(self._pid)
            self._pid = 0

    def _force(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass


@dataclass
class _Worker:
    job_id: int
    pid: int
    fd: int
    deadline: float
    reader: _FrameReader = field(default_factory=_FrameReader)


class _Zygote:
    """Single-threaded forever; the only place the server forks, and it only
    forks workers, never touching student code itself.
    """

    def __init__(self, sock: Any, context: SandboxContext, limits: Dict[str, float]) -> None:
        self._sock = sock  # kept alive so its fd is not closed under us
        self._fd = sock.fileno()
        self._ctx = context
        self._limits = limits
        self._cap = int(limits["workers"])
        self._wall = float(limits["wall"])
        self._workers: Dict[int, _Worker] = {}  # keyed by result-pipe read fd
        self._incoming = _FrameReader()
        self._pending_requests: Deque[Dict[str, Any]] = deque()

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _die_with_parent()
        try:
            self._loop()
        except _ParentGone:
            pass
        finally:
            for worker in list(self._workers.values()):
                self._kill(worker.pid)
                self._reap(worker.pid)

    def _loop(self) -> None:
        while True:
            now = time.monotonic()
            timeout = self._enforce_deadlines(now)
            readers = [self._fd] + list(self._workers)
            try:
                ready, _, _ = select.select(readers, [], [], timeout)
            except InterruptedError:
                continue
            for fd in ready:
                if fd == self._fd:
                    self._on_control()
                elif fd in self._workers:
                    self._on_worker(self._workers[fd])
            self._pump_queue_from_control()

    # control channel (to the HTTP process)
    def _on_control(self) -> None:
        try:
            chunk = os.read(self._fd, 65536)
        except OSError:
            raise _ParentGone()
        if not chunk:
            raise _ParentGone()
        for request in self._incoming.feed(chunk):
            self._pending_requests.append(request)

    def _pump_queue_from_control(self) -> None:
        while self._pending_requests and len(self._workers) < self._cap:
            self._start(self._pending_requests.popleft())

    def _send(self, frame: Dict[str, Any]) -> None:
        try:
            _write_all(self._fd, _encode(frame))
        except OSError:
            raise _ParentGone()

    # workers
    def _start(self, request: Dict[str, Any]) -> None:
        job_id = request.get("id")
        handler = _DISPATCH.get(request.get("kind"))
        if handler is None:
            self._send({"id": job_id, "status": "crash", "message": "unknown job"})
            return
        try:
            read_fd, write_fd = os.pipe()
        except OSError:
            self._send({"id": job_id, "status": "crash", "message": "cannot start a worker"})
            return
        try:
            pid = os.fork()
        except OSError:
            os.close(read_fd)  # the fork failed; the pipe we just made must not leak
            os.close(write_fd)
            self._send({"id": job_id, "status": "crash", "message": "cannot start a worker"})
            return
        if pid == 0:  # the worker
            os.close(read_fd)
            # drop the read ends of sibling workers inherited through the fork;
            # they are the zygote's to hold, not ours
            for other in list(self._workers):
                try:
                    os.close(other)
                except OSError:
                    pass
            self._run_worker(write_fd, handler, request.get("payload", {}))
        os.close(write_fd)
        self._workers[read_fd] = _Worker(
            job_id=job_id, pid=pid, fd=read_fd, deadline=time.monotonic() + self._wall
        )

    def _run_worker(self, write_fd: int, handler: Any, payload: Dict[str, Any]) -> None:
        exit_code = 0
        try:
            self._sock.close()
            _mute_stdout()
            _apply_limits(self._limits)
            try:
                frame = {"status": "ok", "result": handler(self._ctx, payload)}
            except Exception as exc:  # a fault in the job, not an overrun
                frame = {"status": "error", "message": type(exc).__name__}
            _write_all(write_fd, _encode(frame))
        except BaseException:
            exit_code = 1
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass
            os._exit(exit_code)

    def _on_worker(self, worker: _Worker) -> None:
        try:
            chunk = os.read(worker.fd, 65536)
        except OSError:
            chunk = b""
        if chunk:
            try:
                frames = worker.reader.feed(chunk)
            except (ValueError, pickle.PickleError):
                frames = []
            if frames:
                frame = frames[0]
                self._finish(worker.fd, worker.pid, {
                    "id": worker.job_id,
                    "status": frame.get("status", "crash"),
                    "result": frame.get("result"),
                    "message": frame.get("message", ""),
                }, reap=True)
            return
        # EOF with no complete frame: the worker died before replying
        verdict = self._postmortem(worker.pid)
        self._finish(worker.fd, worker.pid, {"id": worker.job_id, **verdict}, reap=False)

    def _enforce_deadlines(self, now: float) -> Optional[float]:
        soonest: Optional[float] = None
        for worker in list(self._workers.values()):
            if now >= worker.deadline:
                self._kill(worker.pid)
                self._finish(worker.fd, worker.pid, {
                    "id": worker.job_id, "status": "limit",
                    "message": "the analysis ran past its time limit",
                }, reap=True)
            else:
                soonest = worker.deadline if soonest is None else min(soonest, worker.deadline)
        return None if soonest is None else max(0.0, soonest - now)

    def _postmortem(self, pid: int) -> Dict[str, Any]:
        status = self._reap(pid)
        if status is not None and os.WIFSIGNALED(status):
            if os.WTERMSIG(status) in (
                signal.SIGXCPU, signal.SIGKILL, signal.SIGSEGV, signal.SIGABRT
            ):
                return {"status": "limit", "message": "the analysis exceeded its CPU or memory limit"}
        return {"status": "crash", "message": "the analysis could not be completed"}

    def _finish(self, fd: int, pid: int, frame: Dict[str, Any], *, reap: bool) -> None:
        self._workers.pop(fd, None)
        try:
            os.close(fd)
        except OSError:
            pass
        if reap:
            self._reap(pid)
        self._send(frame)

    def _kill(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    def _reap(self, pid: int) -> Optional[int]:
        try:
            _, status = os.waitpid(pid, 0)
            return status
        except OSError:
            return None


class InlineSandbox:
    """Fallback: runs jobs in this process under a lock. No ceiling can be
    enforced here; for a supervised, trusted pilot only.
    """

    def __init__(self, context: SandboxContext) -> None:
        self._ctx = context
        self._lock = threading.Lock()
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    def diagnose(self, payload: Dict[str, Any]) -> Outcome:
        return self._run(run_diagnose, payload)

    def challenge(self, payload: Dict[str, Any]) -> Outcome:
        return self._run(run_challenge, payload)

    def _run(self, handler: Any, payload: Dict[str, Any]) -> Outcome:
        with self._lock:
            try:
                return Outcome("ok", result=handler(self._ctx, payload))
            except Exception as exc:
                return Outcome("crash", message=type(exc).__name__)

    def close(self) -> None:
        self._alive = False


def make_sandbox(context: SandboxContext, config: Config, *, isolate: bool = True) -> Any:
    """A ForkSandbox where fork is available and wanted, else the in-process
    fallback.
    """
    if isolate and hasattr(os, "fork"):
        return ForkSandbox(context, config)
    return InlineSandbox(context)


# helpers
class _ParentGone(Exception):
    pass


def _socketpair() -> Any:
    import socket

    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


def _apply_limits(limits: Dict[str, float]) -> None:
    try:
        import resource
    except ImportError:
        return
    cpu = int(limits["cpu"])
    address = int(limits["address_space"])
    for what, soft, hard in (
        (resource.RLIMIT_CPU, cpu, cpu + 1),
        (resource.RLIMIT_AS, address, address),
        (resource.RLIMIT_CORE, 0, 0),
    ):
        try:
            resource.setrlimit(what, (soft, hard))
        except (OSError, ValueError):
            pass


def _mute_stdout() -> None:
    """Point the worker's stdout at the void, at the file-descriptor level, so
    a student's print() -- or any write from a C routine -- cannot reach the
    server's own stdout, which the worker inherited.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        if devnull > 2:
            os.close(devnull)
    except OSError:
        pass


def _die_with_parent() -> None:
    """Ask the kernel to kill this process if the HTTP process dies, so a
    crashed server never leaves the zygote behind (Linux only).
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass
