"""The animation that runs while the engine thinks.

A diagnosis against a hosted model takes seconds, and a terminal that prints
nothing for seconds looks broken. The spinner says the same thing the rest of
the interface says with ✻: the tool is talking about itself. It writes nothing
at all unless stdout is a terminal, so transcripts and tests stay clean.
"""

import threading
import time
from typing import Optional

from .ansi import Style, _OutputStream

# the three agents take their turn: evidence, then the prerequisite, then the
# question. Same three colours as the pills under the wordmark.
FRAMES = (
    ("●", "·", "·"),
    ("·", "●", "·"),
    ("·", "·", "●"),
    ("·", "●", "·"),
)
HUES = ("amber", "sky", "teal")
INTERVAL = 0.16


class Spinner:
    """A one line "still working" animation, used as a context manager."""

    def __init__(self, style: Style, output: _OutputStream, label: str = "Working") -> None:
        self.style = style
        self.output = output
        self.label = label
        self.live = style.enabled and bool(getattr(output, "isatty", lambda: False)())
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        if self.live:
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            self.output.write("\r\x1b[K")
            self.output.flush()
        return False

    def _spin(self) -> None:
        started = time.time()
        frame = 0
        while not self.done.is_set():
            elapsed = time.time() - started
            dots = "".join(
                self.style.paint(glyph, HUES[i])
                for i, glyph in enumerate(FRAMES[frame % len(FRAMES)])
            )
            self.output.write(
                "\r\x1b[K  "
                + dots
                + "  "
                + self.style.paint(self.label + "…", "text")
                + self.style.paint(f"  ctrl-c to interrupt · {elapsed:.0f}s", "dim")
            )
            self.output.flush()
            frame += 1
            self.done.wait(INTERVAL)
