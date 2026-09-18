"""Drive the interactive tutor from code.
The session reads commands from any stream, so a script can walk a file
through the same dialogue a person would have, including edit and retry.

    PYTHONPATH=src python3 examples/04_scripted_session.py
"""

import io
import tempfile
from pathlib import Path

from deductive_mas.cli.session import TutorSession
from deductive_mas.ui.ansi import Style

BROKEN = """\
def lower_bound(a, t):
    lo, hi = 0, len(a) - 1     # inclusive bound ...
    while lo < hi:             # ... under an exclusive guard
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
"""
FIXED = BROKEN.replace("len(a) - 1", "len(a)    ")


class Script:
    """Commands, mixed with callables that play the learner's editor."""

    def __init__(self, *items):
        self.items = list(items)

    def isatty(self):
        return False

    def readline(self):
        while self.items:
            item = self.items.pop(0)
            if callable(item):
                item()
                continue
            return item + "\n"
        return ""


def main():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "attempt.py"
        path.write_text(BROKEN, encoding="utf-8")

        script = Script(
            f"load {path}",
            "hint",                                            # first rung only
            "challenge",
            lambda: path.write_text(FIXED, encoding="utf-8"),  # the learner edits
            "retry",                                           # was the belief repaired?
            "mastery",
            "quit",
        )
        output = io.StringIO()
        TutorSession(style=Style(), input_stream=script, output_stream=output).run()
        print(output.getvalue())


if __name__ == "__main__":
    main()
