"""ANSI colours and the colour theme.

The terminal is the interface so a few rules: colour means something
(severity, belief, pass/fail), nothing else is coloured. NO_COLOR, a pipe,
a dumb terminal or --color never all fall back to plain text with the same
layout. Every widget measures width with display_width which ignores
escape codes, so columns line up.
"""

import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from typing import Optional

_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x07]*\x07")

RESET = "\x1b[0m"


# capability detection
def supports_colour(mode: str = "auto", stream=None) -> bool:
    """Should we emit escape codes at all."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def supports_truecolour() -> bool:
    return os.environ.get("COLORTERM", "") in ("truecolor", "24bit")


def terminal_width(default: int = 96, maximum: int = 108) -> int:
    try:
        width = shutil.get_terminal_size((default, 24)).columns
    except OSError:
        width = default
    return max(52, min(width - 1, maximum))


def display_width(text: str) -> int:
    """Printable width, ignoring escapes, wide glyphs count as two."""
    stripped = _ESCAPE_RE.sub("", text)
    width = 0
    for char in stripped:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def strip(text: str) -> str:
    return _ESCAPE_RE.sub("", text)


# theme
@dataclass(frozen=True)
class Colour:
    """A colour with a 24 bit and a 256 colour version."""

    rgb: tuple
    xterm: int

    def fg(self, truecolour: bool) -> str:
        if truecolour:
            r, g, b = self.rgb
            return f"\x1b[38;2;{r};{g};{b}m"
        return f"\x1b[38;5;{self.xterm}m"

    def bg(self, truecolour: bool) -> str:
        if truecolour:
            r, g, b = self.rgb
            return f"\x1b[48;2;{r};{g};{b}m"
        return f"\x1b[48;5;{self.xterm}m"


# palette that reads on light and dark terminals
PALETTE = {
    "primary":  Colour((94, 176, 214), 74),    # blue: structure, headings
    "accent":   Colour((186, 148, 232), 141),  # violet: knowledge layer
    "success":  Colour((126, 191, 121), 108),  # green: ok, passing
    "warning":  Colour((222, 179, 92), 179),   # amber: contested, degraded
    "danger":   Colour((219, 108, 112), 167),  # red: critical
    "muted":    Colour((128, 135, 145), 245),  # grey: secondary text
    "text":     Colour((214, 218, 224), 252),
    "code":     Colour((162, 194, 155), 151),  # source listings
}

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"


class Style:
    """Renders styled text, or plain text when colour is off."""

    def __init__(self, mode: str = "auto", stream=None):
        self.enabled = supports_colour(mode, stream)
        self.truecolour = self.enabled and supports_truecolour()
        self.width = terminal_width()

    # basics
    def paint(self, text: str, colour: Optional[str] = None, *, bold: bool = False,
              dim: bool = False, italic: bool = False, underline: bool = False) -> str:
        if not self.enabled or not text:
            return text
        prefix = ""
        if colour and colour in PALETTE:
            prefix += PALETTE[colour].fg(self.truecolour)
        if bold:
            prefix += BOLD
        if dim:
            prefix += DIM
        if italic:
            prefix += ITALIC
        if underline:
            prefix += UNDERLINE
        return f"{prefix}{text}{RESET}" if prefix else text

    # shortcuts
    def primary(self, text: str, **kwargs) -> str:
        return self.paint(text, "primary", **kwargs)

    def accent(self, text: str, **kwargs) -> str:
        return self.paint(text, "accent", **kwargs)

    def success(self, text: str, **kwargs) -> str:
        return self.paint(text, "success", **kwargs)

    def warning(self, text: str, **kwargs) -> str:
        return self.paint(text, "warning", **kwargs)

    def danger(self, text: str, **kwargs) -> str:
        return self.paint(text, "danger", **kwargs)

    def muted(self, text: str, **kwargs) -> str:
        return self.paint(text, "muted", **kwargs)

    def code(self, text: str, **kwargs) -> str:
        return self.paint(text, "code", **kwargs)

    def heading(self, text: str) -> str:
        return self.paint(text, "primary", bold=True)

    # extras
    def severity(self, level: str) -> str:
        colour = {
            "critical": "danger", "major": "danger",
            "minor": "warning", "info": "muted",
        }.get(level.lower(), "muted")
        return self.paint(level.upper(), colour, bold=level.lower() in ("critical", "major"))

    def belief(self, value: float) -> str:
        """Colour a confidence by how much weight it carries."""
        colour = "success" if value >= 0.66 else "warning" if value >= 0.4 else "muted"
        return self.paint(f"{value:.2f}", colour, bold=value >= 0.66)

    def link(self, text: str, url: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b]8;;{url}\x07{text}\x1b]8;;\x07"
