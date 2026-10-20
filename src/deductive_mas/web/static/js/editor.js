// The code editor: a transparent textarea over a highlighted copy of its
// own text. The textarea keeps everything a browser already does well
// (undo, selection, input methods, screen readers, mobile keyboards); the
// layer underneath carries colour and the tutor's marks: the line where the
// execution first diverged, the lines a rule flagged, a syntax error.

import { clear, h } from "./dom.js";
import { highlight } from "./python.js";

const INDENT = "    ";
const LH = 22;
const PAD_Y = 14;

export class CodeEditor {
  constructor(host, { onChange, onSubmit, onCaret } = {}) {
    this.onChange = onChange;
    this.onSubmit = onSubmit;
    this.onCaret = onCaret;
    this.marks = { divergence: null, findings: [], error: null, stale: false };
    this.lineEls = [];
    this.gutterEls = [];
    this.current = 0;
    this.escaped = false;

    this.gutter = h("div", { class: "gutter", "aria-hidden": "true" });
    this.layer = h("pre", { class: "code-layer", "aria-hidden": "true" });
    this.input = h("textarea", {
      class: "code-input",
      id: "editor-input",
      spellcheck: "false",
      autocapitalize: "off",
      autocomplete: "off",
      autocorrect: "off",
      wrap: "off",
      "aria-label": "Your Python code",
      "aria-describedby": "editor-help",
    });
    this.code = h("div", { class: "code" }, this.layer, this.input);
    this.scroller = h("div", { class: "editor-scroll" }, h("div", { class: "editor-body" }, this.gutter, this.code));
    this.host = host;
    clear(host).append(
      this.scroller,
      h("p", { id: "editor-help", class: "sr-only" },
        "Tab indents. Press Escape and then Tab to leave the editor. " +
        "Command or Control with Enter diagnoses the code."),
    );

    this.input.addEventListener("input", () => {
      this.render();
      this.caret();
      this.onChange?.(this.value);
    });
    this.input.addEventListener("keydown", (event) => this.key(event));
    for (const type of ["keyup", "click", "focus", "select"]) {
      this.input.addEventListener(type, () => this.caret());
    }
    document.addEventListener("selectionchange", () => {
      if (document.activeElement === this.input) this.caret();
    });
    this.render();
  }

  get value() {
    return this.input.value;
  }

  set value(text) {
    this.input.value = text;
    this.render();
    this.caret(false);
  }

  focus() {
    this.input.focus({ preventScroll: true });
  }

  setMarks(marks) {
    this.marks = { divergence: null, findings: [], error: null, stale: false, ...marks };
    this.host.classList.toggle("stale", Boolean(this.marks.stale));
    this.render();
  }

  // drawing
  render() {
    const lines = highlight(this.input.value);
    const { divergence, error } = this.marks;
    const findings = new Set(this.marks.findings || []);
    const fragment = document.createDocumentFragment();
    this.lineEls = lines.map((tokens, index) => {
      const n = index + 1;
      const el = document.createElement("span");
      el.className = "code-line";
      let range = null;
      if (error && error.line === n) {
        el.classList.add("mark-error");
        if (error.column) {
          const from = error.column - 1;
          const to = error.endLine === n && error.endColumn > error.column ? error.endColumn - 1 : from + 1;
          range = [from, to];
        }
      }
      if (divergence && divergence.line === n) el.classList.add("mark-divergence");
      paint(el, tokens, range);
      if (divergence && divergence.line === n && divergence.label) {
        el.append(h("span", { class: "ghost ghost-divergence" }, "◂ ", divergence.label));
      }
      if (error && error.line === n) {
        el.append(h("span", { class: "ghost ghost-error" }, "◂ ", error.message));
      }
      fragment.append(el);
      return el;
    });
    clear(this.layer).append(fragment);

    const gutter = document.createDocumentFragment();
    this.gutterEls = lines.map((_, index) => {
      const n = index + 1;
      const el = h("div", { class: "gutter-line" }, String(n));
      if (findings.has(n)) el.classList.add("mark-finding");
      if (divergence && divergence.line === n) el.classList.add("mark-divergence");
      if (error && error.line === n) el.classList.add("mark-error");
      gutter.append(el);
      return el;
    });
    clear(this.gutter).append(gutter);
    this.highlightCurrent(this.current);
  }

  highlightCurrent(index) {
    this.lineEls[this.current]?.classList.remove("is-current");
    this.gutterEls[this.current]?.classList.remove("is-current");
    this.current = Math.min(index, this.lineEls.length - 1);
    if (document.activeElement === this.input) {
      this.lineEls[this.current]?.classList.add("is-current");
      this.gutterEls[this.current]?.classList.add("is-current");
    }
  }

  caret(follow = true) {
    const { selectionStart, value } = this.input;
    const before = value.slice(0, selectionStart);
    const line = before.split("\n").length - 1;
    const column = selectionStart - (before.lastIndexOf("\n") + 1);
    this.highlightCurrent(line);
    this.onCaret?.(line + 1, column + 1);
    if (follow && document.activeElement === this.input) this.follow(line, column);
  }

  // keep the caret in view: the textarea never scrolls itself, its frame does
  follow(line, column) {
    const scroller = this.scroller;
    const top = PAD_Y + line * LH;
    if (top < scroller.scrollTop + 4) scroller.scrollTop = top - 8;
    else if (top + LH > scroller.scrollTop + scroller.clientHeight - 8) {
      scroller.scrollTop = top + LH - scroller.clientHeight + 24;
    }
    const width = this.charWidth();
    const gutter = this.gutter.offsetWidth;
    const x = gutter + 18 + column * width;
    if (x > scroller.scrollLeft + scroller.clientWidth - 24) scroller.scrollLeft = x - scroller.clientWidth + 48;
    else if (x < scroller.scrollLeft + gutter + 8) scroller.scrollLeft = Math.max(0, x - gutter - 32);
  }

  charWidth() {
    if (!this._charWidth) {
      const probe = h("span", null, "0".repeat(40));
      probe.style.visibility = "hidden";
      probe.style.position = "absolute";
      this.layer.append(probe);
      this._charWidth = probe.getBoundingClientRect().width / 40 || 8;
      probe.remove();
    }
    return this._charWidth;
  }

  showLine(n, { hover = false } = {}) {
    const el = this.lineEls[n - 1];
    if (!el) return;
    if (!hover) {
      const top = PAD_Y + (n - 1) * LH;
      const scroller = this.scroller;
      if (top < scroller.scrollTop || top > scroller.scrollTop + scroller.clientHeight - 2 * LH) {
        scroller.scrollTo({ top: Math.max(0, top - scroller.clientHeight / 3), behavior: "smooth" });
      }
    }
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
    clearTimeout(el._flash);
    el._flash = setTimeout(() => el.classList.remove("flash"), 1200);
  }

  // editing
  key(event) {
    const mod = event.metaKey || event.ctrlKey;
    if (mod && event.key === "Enter") {
      event.preventDefault();
      this.onSubmit?.();
      return;
    }
    if (event.key === "Escape") {
      // the next Tab leaves the editor instead of indenting
      this.escaped = true;
      return;
    }
    if (event.key === "Tab") {
      if (this.escaped || mod || event.altKey) {
        this.escaped = false;
        return;
      }
      event.preventDefault();
      if (event.shiftKey) this.outdent();
      else this.indent();
      return;
    }
    this.escaped = false;
    if (event.isComposing) return;
    if (event.key === "Enter" && !mod && !event.altKey && !event.shiftKey) {
      event.preventDefault();
      this.newline();
    } else if (event.key === "Backspace" && !mod && !event.altKey) {
      if (this.unindent()) event.preventDefault();
    } else if (mod && event.key === "/") {
      event.preventDefault();
      this.comment();
    }
  }

  replace(from, to, text, select) {
    const input = this.input;
    input.focus();
    input.setSelectionRange(from, to);
    let done = false;
    try {
      done = document.execCommand(text ? "insertText" : "delete", false, text);
    } catch {
      done = false;
    }
    if (!done) {
      input.setRangeText(text, from, to, "end");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (select) input.setSelectionRange(select[0], select[1]);
  }

  block() {
    const { selectionStart: start, selectionEnd: end, value } = this.input;
    const from = value.lastIndexOf("\n", start - 1) + 1;
    let stopAt = end;
    if (end > start && value[end - 1] === "\n") stopAt -= 1;
    const next = value.indexOf("\n", stopAt);
    const to = next === -1 ? value.length : next;
    return { from, to, start, end, text: value.slice(from, to) };
  }

  indent() {
    const { selectionStart: start, selectionEnd: end, value } = this.input;
    if (start === end || !value.slice(start, end).includes("\n")) {
      const column = start - (value.lastIndexOf("\n", start - 1) + 1);
      this.replace(start, end, " ".repeat(4 - (column % 4)));
      return;
    }
    const { from, to, text } = this.block();
    const out = text.split("\n").map((line) => (line.trim() ? INDENT + line : line)).join("\n");
    this.replace(from, to, out, [from, from + out.length]);
  }

  outdent() {
    const { from, to, text, start, end } = this.block();
    let removedFirst = 0;
    const out = text.split("\n").map((line, index) => {
      const cut = /^( {1,4}|\t)/.exec(line);
      if (!cut) return line;
      if (index === 0) removedFirst = cut[0].length;
      return line.slice(cut[0].length);
    }).join("\n");
    if (out === text) return;
    if (start === end) {
      const caret = Math.max(from, start - removedFirst);
      this.replace(from, to, out, [caret, caret]);
    } else {
      this.replace(from, to, out, [from, from + out.length]);
    }
  }

  newline() {
    const { selectionStart: start, selectionEnd: end, value } = this.input;
    const line = value.slice(value.lastIndexOf("\n", start - 1) + 1, start);
    let indent = /^[ \t]*/.exec(line)[0];
    const code = line.replace(/#.*$/, "").trimEnd();
    if (code.endsWith(":")) indent += INDENT;
    else if (/^\s*(return|pass|break|continue|raise)\b/.test(line)) indent = indent.slice(0, Math.max(0, indent.length - 4));
    this.replace(start, end, "\n" + indent);
  }

  unindent() {
    const { selectionStart: start, selectionEnd: end, value } = this.input;
    if (start !== end) return false;
    const from = value.lastIndexOf("\n", start - 1) + 1;
    const before = value.slice(from, start);
    if (!before || /[^ ]/.test(before)) return false;
    const width = ((before.length - 1) % 4) + 1;
    this.replace(start - width, start, "");
    return true;
  }

  comment() {
    const { from, to, text } = this.block();
    const lines = text.split("\n");
    const live = lines.filter((line) => line.trim());
    if (!live.length) return;
    const commented = live.every((line) => /^\s*#/.test(line));
    const margin = Math.min(...live.map((line) => /^ */.exec(line)[0].length));
    const out = lines.map((line) => {
      if (!line.trim()) return line;
      if (commented) return line.replace(/^(\s*)# ?/, "$1");
      return line.slice(0, margin) + "# " + line.slice(margin);
    }).join("\n");
    this.replace(from, to, out, [from, from + out.length]);
  }
}

function paint(el, tokens, range) {
  let position = 0;
  for (const [cls, text] of tokens) {
    const start = position;
    const end = position + text.length;
    position = end;
    if (!range || end <= range[0] || start >= range[1]) {
      el.append(token(cls, text));
      continue;
    }
    const a = Math.max(range[0], start) - start;
    const b = Math.min(range[1], end) - start;
    if (a > 0) el.append(token(cls, text.slice(0, a)));
    el.append(token(`${cls} squiggle`.trim(), text.slice(a, b)));
    if (b < text.length) el.append(token(cls, text.slice(b)));
  }
  if (range && range[0] >= position) {
    // the parser points past the end of the line: underline a space there
    el.append(token("squiggle", " "));
  }
}

function token(cls, text) {
  if (!cls) return document.createTextNode(text);
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = text;
  return span;
}
