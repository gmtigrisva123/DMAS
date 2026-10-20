// The app: state, the actions the margin and the sheets call, and the parts
// of the page that are not the margin (the top bar, the task card, the
// editor's status line). The server is the only authority on the session;
// every action replaces `state.session` with the snapshot it returns.

import { api } from "./api.js";
import { announce, clear, h, rich, toast } from "./dom.js";
import { CodeEditor } from "./editor.js";
import { renderMargin } from "./margin.js";
import {
  confirmReset, difficultyDots, pickTask, showConcept, showHistory, showMastery,
} from "./sheets.js";

const storage = {
  get(key) {
    try { return window.localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { window.localStorage.setItem(key, value); } catch { /* private mode: nothing to keep */ }
  },
};

const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const $ = (id) => document.getElementById(id);

const app = {
  api,
  modKey: isMac ? "⌘" : "Ctrl",
  state: {
    meta: null,
    tasks: [],
    taskById: new Map(),
    examples: [],
    session: null,
    selectedTask: null,
    busy: { submit: false, hint: false, challenge: false, run: false },
    fresh: { margin: false, rung: null },
    open: { evidence: false, blame: false },
    run: null,
    prediction: "",
    syntaxError: null,
    message: null,
    taskCollapsed: storage.get("dmas:task-collapsed") === "1",
    view: "code",
  },
  actions: {},
};
const { state } = app;

const editor = new CodeEditor($("editor"), {
  onChange: (value) => {
    if (state.syntaxError) {
      state.syntaxError = null;
      setMessage(null);
    }
    saveBuffer(value);
    renderEditorState();
  },
  onSubmit: () => app.actions.submit(),
  onCaret: (line, column) => {
    $("caret-pos").textContent = `Ln ${line}, Col ${column}`;
  },
});

// buffers: one per task, kept in this browser only
let saveTimer = null;
function saveBuffer(value) {
  clearTimeout(saveTimer);
  const task = state.selectedTask;
  saveTimer = setTimeout(() => storage.set(`dmas:buffer:${task}`, value), 250);
}

function starter(taskId) {
  if (taskId === "auto") {
    return "# Any Python function. The tutor matches it to a task by its name,\n" +
      "# or checks its structure when it has no reference for it.\n\n";
  }
  const task = state.taskById.get(taskId);
  return task ? `${task.signature}\n    ` : "";
}

function bufferFor(taskId) {
  const saved = storage.get(`dmas:buffer:${taskId}`);
  if (saved !== null) return saved;
  const current = state.session && state.session.current;
  if (current && current.task.id === taskId) return current.source;
  return starter(taskId);
}

// what the editor shows about the latest attempt
function currentForEditor() {
  const current = state.session && state.session.current;
  if (!current) return null;
  if (state.selectedTask === "auto" || current.task.id === state.selectedTask) return current;
  return null;
}

function renderEditorState() {
  const current = currentForEditor();
  const frame = $("editor-frame");
  const label = $("editor-state");
  const edited = current ? editor.value !== current.source : false;
  frame.dataset.state = current ? (edited ? "edited" : "diagnosed") : "new";
  clear(label);
  if (current && edited) {
    label.append("edited since attempt ", h("b", null, String(current.index)), ` · ${app.modKey} ↵ to retry`);
  } else if (current) {
    label.append(`diagnosed · attempt ${current.index}`);
  } else {
    label.append("not diagnosed yet");
  }
  $("diagnose-label").textContent = current ? "Retry" : "Diagnose";

  if (state.syntaxError) {
    editor.setMarks({ error: state.syntaxError });
  } else if (current) {
    editor.setMarks({
      divergence: current.divergence && current.divergence.line
        ? { line: current.divergence.line, label: current.divergence.kind === "exception" ? "raises here" : "first divergence" }
        : null,
      findings: current.findings.map((f) => f.line).filter(Boolean),
      stale: edited,
    });
  } else {
    editor.setMarks({});
  }
}

function setMessage(text, kind = "") {
  state.message = text ? { text, kind } : null;
  const el = $("editor-message");
  el.className = `editor-message ${kind}`;
  clear(el);
  if (text) el.append(text);
}

// the top bar and the task card
function renderTopbar() {
  const task = state.taskById.get(state.selectedTask);
  $("task-switch-title").textContent = task ? task.title : "Any function";

  const attempts = (state.session && state.session.attempts) || [];
  const history = $("history-button");
  history.hidden = !attempts.length;
  clear($("history-dots")).append(...attempts.slice(-6).map((a) => h("i", { dataset: { status: a.status } })));
  $("history-label").textContent = `${attempts.length} attempt${attempts.length === 1 ? "" : "s"}`;

  const mastery = state.session && state.session.mastery;
  const count = $("mastery-count");
  count.hidden = !(mastery && mastery.below);
  count.textContent = mastery ? String(mastery.below) : "";
  count.title = mastery ? `${mastery.below} concepts below the mastery line` : "";
}

function renderTaskCard() {
  const card = $("task-card");
  card.classList.toggle("collapsed", state.taskCollapsed);
  const toggle = h("button", {
    type: "button", class: "task-toggle", "aria-expanded": String(!state.taskCollapsed),
    on: {
      click: () => {
        state.taskCollapsed = !state.taskCollapsed;
        storage.set("dmas:task-collapsed", state.taskCollapsed ? "1" : "0");
        renderTaskCard();
      },
    },
  }, state.taskCollapsed ? "Show the task" : "Hide");

  const task = state.taskById.get(state.selectedTask);
  if (!task) {
    const detected = state.session && state.session.task_mode === "auto" && state.session.task;
    clear(card).append(
      h("div", { class: "task-card-head" }, h("div", { class: "eyebrow" }, "Task"), toggle),
      h("h1", { class: "task-title" }, "Any function"),
      h("p", { class: "task-statement" },
        "Write or paste a Python function. If its name matches a task the tutor has a reference for, ",
        "it is diagnosed against that task; otherwise you get structural findings only."),
      detected ? h("div", { class: "task-meta" }, h("span", { class: "chip" },
        detected.freeform ? "No matching task: structural findings" : `Detected: ${detected.title}`)) : null);
    return;
  }
  clear(card).append(
    h("div", { class: "task-card-head" }, h("div", { class: "eyebrow" }, "Task"), toggle),
    h("h1", { class: "task-title" }, task.title),
    h("p", { class: "task-statement prose" }, rich(task.statement)),
    h("div", { class: "task-meta" },
      h("span", { class: "signature" }, task.signature),
      task.concepts.map((c) => h("button", {
        type: "button", class: "chip", on: { click: () => app.actions.openConcept(c.id) },
      }, c.name)),
      difficultyDots(task.difficulty)));
}

function renderMarginOnly() {
  renderMargin($("margin"), app);
}

function render() {
  renderTopbar();
  renderTaskCard();
  renderEditorState();
  renderMarginOnly();
  const button = $("diagnose-button");
  button.disabled = state.busy.submit;
  button.classList.toggle("busy", state.busy.submit);
  $("mobile-badge").hidden = !(state.view === "code" && state.session && state.session.current && state.fresh.badge);
}

function setView(view) {
  state.view = view;
  document.body.dataset.view = view;
  $("show-code").setAttribute("aria-pressed", String(view === "code"));
  $("show-tutor").setAttribute("aria-pressed", String(view === "tutor"));
  if (view === "tutor") state.fresh.badge = false;
  $("mobile-badge").hidden = !(view === "code" && state.fresh.badge);
}

const narrow = () => window.matchMedia("(max-width: 860px)").matches;

// actions
function failure(error, fallback) {
  if (error.code === "stale_attempt" || error.code === "out_of_order" || error.code === "no_attempt") {
    refresh();
    toast("The tutor had moved on; this view is now up to date.");
    return;
  }
  toast(error.message || fallback);
}

async function refresh() {
  try {
    state.session = (await api.session()).session;
  } catch (error) {
    toast(error.message);
  }
  render();
}

app.actions.submit = async () => {
  if (state.busy.submit) return;
  state.busy.submit = true;
  setMessage(null);
  render();
  try {
    const result = await api.submit(editor.value, state.selectedTask);
    state.session = result.session;
    if (result.unchanged) {
      setMessage(result.message, "info");
    } else {
      state.run = null;
      state.prediction = "";
      state.open.evidence = false;
      state.fresh.margin = true;
      state.fresh.badge = true;
      const current = state.session.current;
      announce(verdictSentence(current));
      window.scrollTo({ top: 0, behavior: "smooth" });
      if (narrow()) setView("tutor");
    }
  } catch (error) {
    if (error.code === "syntax_error" && error.detail.line) {
      state.syntaxError = {
        line: error.detail.line,
        column: error.detail.column,
        endLine: error.detail.end_line,
        endColumn: error.detail.end_column,
        message: error.message,
      };
      setMessage(`Line ${error.detail.line}: ${error.message}. Fix this first; code that does not parse is not diagnosed.`, "error");
      if (narrow()) setView("code");
      editor.showLine(error.detail.line);
    } else {
      setMessage(error.message, "error");
    }
  } finally {
    state.busy.submit = false;
    render();
  }
};

app.actions.nextHint = async () => {
  const current = state.session && state.session.current;
  if (!current || state.busy.hint) return;
  state.busy.hint = true;
  renderMarginOnly();
  let shown = null;
  try {
    const result = await api.hint(current.index, current.ladder.revealed.length + 1);
    state.session = result.session;
    if (result.status === "revealed") {
      shown = result.hint.rung;
      state.fresh.rung = shown;
      announce(`Question ${shown} of ${result.hint.total}. ${result.hint.text}`);
    } else if (result.message) {
      toast(result.message);
    }
  } catch (error) {
    failure(error, "Could not open the next question.");
  } finally {
    state.busy.hint = false;
    render();
  }
  if (shown) {
    const rung = document.getElementById(`rung-${shown}`);
    if (rung) {
      rung.focus({ preventScroll: true });
      rung.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
};

app.actions.openChallenge = async () => {
  const current = state.session && state.session.current;
  if (!current || state.busy.challenge) return;
  state.busy.challenge = true;
  renderMarginOnly();
  try {
    const result = await api.challenge(current.index);
    state.session = result.session;
    if (result.status !== "shown" && result.message) toast(result.message);
  } catch (error) {
    failure(error, "Could not open the experiment.");
  } finally {
    state.busy.challenge = false;
    render();
  }
  $("prediction")?.focus({ preventScroll: true });
};

app.actions.runChallenge = async (prediction) => {
  const current = state.session && state.session.current;
  if (!current || state.busy.run) return;
  state.prediction = prediction;
  state.busy.run = true;
  renderMarginOnly();
  try {
    const result = await api.run(current.index, editor.value);
    state.run = { attempt: current.index, prediction, result: result.run };
    announce(result.run.ok ? `It returned ${result.run.value}.` : `It stopped: ${result.run.error}.`);
  } catch (error) {
    if (error.code === "syntax_error") {
      toast(`The code in the editor does not parse (line ${error.detail.line}: ${error.message}).`);
    } else {
      failure(error, "Could not run the code.");
    }
  } finally {
    state.busy.run = false;
    renderMarginOnly();
  }
};

app.actions.showLine = (line, options = {}) => {
  if (!options.hover && narrow()) setView("code");
  editor.showLine(line, options);
};

app.actions.openConcept = (id) => showConcept(app, id);

app.actions.selectTask = (id) => {
  if (id === state.selectedTask) return;
  clearTimeout(saveTimer);
  storage.set(`dmas:buffer:${state.selectedTask}`, editor.value);
  state.selectedTask = id;
  storage.set("dmas:task", id);
  state.syntaxError = null;
  setMessage(null);
  editor.value = bufferFor(id);
  render();
  editor.focus();
};

app.actions.loadExample = (example) => {
  if (state.selectedTask !== example.task && state.selectedTask !== "auto") {
    storage.set(`dmas:buffer:${state.selectedTask}`, editor.value);
    state.selectedTask = example.task;
    storage.set("dmas:task", example.task);
  }
  editor.value = example.source;
  storage.set(`dmas:buffer:${state.selectedTask}`, example.source);
  setMessage(`Loaded the bank attempt “${example.id}”. Diagnose it with ${app.modKey} ↵.`, "info");
  if (narrow()) setView("code");
  render();
  editor.focus();
};

app.actions.reset = async () => {
  try {
    state.session = (await api.reset()).session;
    state.run = null;
    state.prediction = "";
    setMessage("Started over: no attempts, no mastery estimate.", "info");
  } catch (error) {
    toast(error.message);
  }
  render();
};

function verdictSentence(current) {
  if (!current) return "";
  const primary = current.findings[0];
  const checks = current.checks ? ` ${current.checks.passed} of ${current.checks.total} inputs pass.` : "";
  switch (current.status) {
    case "diagnosis": return `Diagnosis: ${primary.name}.${checks}`;
    case "provisional": return `Provisional diagnosis: ${primary.name}.${checks}`;
    case "failing": return `Your function fails on an input.${checks}`;
    case "clean": return `No finding.${checks}`;
    case "incomplete": return "The analysis could not finish.";
    default: return "No structural warning sign.";
  }
}

// theme
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function effectiveTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// wiring
function wire() {
  $("diagnose-button").addEventListener("click", () => app.actions.submit());
  $("task-switch").addEventListener("click", () => pickTask(app));
  $("mastery-button").addEventListener("click", () => showMastery(app));
  $("history-button").addEventListener("click", () => showHistory(app));
  $("menu-button").addEventListener("click", () => confirmReset(app));
  $("theme-button").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    storage.set("dmas:theme", next);
  });
  $("show-code").addEventListener("click", () => setView("code"));
  $("show-tutor").addEventListener("click", () => setView("tutor"));
  document.addEventListener("keydown", (event) => {
    const mod = event.metaKey || event.ctrlKey;
    if (!mod || event.key !== "Enter") return;
    if (event.target === editor.input || event.target.closest("dialog")) return;
    if (event.target.id === "prediction") {
      event.preventDefault();
      app.actions.runChallenge(event.target.value);
      return;
    }
    event.preventDefault();
    app.actions.submit();
  });
  $("diagnose-kbd").textContent = `${app.modKey} ↵`;
}

async function boot() {
  applyTheme(storage.get("dmas:theme"));
  wire();
  setView("code");
  try {
    const [meta, tasks, session] = await Promise.all([api.meta(), api.tasks(), api.session()]);
    state.meta = meta;
    state.tasks = tasks.tasks;
    state.taskById = new Map(tasks.tasks.map((t) => [t.id, t]));
    state.session = session.session;
    if (meta.demo) {
      try {
        state.examples = (await api.examples()).examples;
      } catch {
        state.examples = [];
      }
    }
  } catch (error) {
    clear($("margin")).append(h("section", { class: "card" },
      h("div", { class: "card-kicker" }, "Offline"),
      h("p", { class: "verdict-caption" }, error.message)));
    return;
  }
  const remembered = storage.get("dmas:task");
  const valid = (id) => id === "auto" || state.taskById.has(id);
  const fromSession = state.session.current ? state.session.current.task.id : null;
  state.selectedTask = valid(remembered) ? remembered
    : valid(state.session.task_mode) ? state.session.task_mode
      : valid(fromSession) ? fromSession
        : state.tasks[0].id;
  editor.value = bufferFor(state.selectedTask);
  render();
}

boot();
