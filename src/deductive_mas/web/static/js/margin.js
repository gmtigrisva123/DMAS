// The margin: everything the tutor says about the latest attempt, in the
// order of the reasoning. What happened (the verdict and the failing
// input), why (the belief and its evidence), what to do about it (one
// question at a time, then an experiment), and where it comes from (the
// prerequisite chain). Built fresh from the session snapshot on every
// change; nothing here is fetched on its own.

import { fixed, h, icon, pct, rich } from "./dom.js";

const STATUS = {
  diagnosis: { tag: "Diagnosis", cls: "tag-evidence" },
  provisional: { tag: "Provisional", cls: "tag-evidence-outline" },
  failing: { tag: "Failing input", cls: "tag-danger" },
  clean: { tag: "No finding", cls: "tag-ok" },
  "structural-clean": { tag: "No warning sign", cls: "tag-ok-outline" },
  incomplete: { tag: "Incomplete", cls: "tag-plain" },
};

const DIVERGENCE = {
  state_mismatch: "This is where the values in your run first differ from a correct run's.",
  control_mismatch: "Here your run takes a step that a correct run never takes.",
  early_termination: "Around here your run stops before a correct run would.",
  non_termination: "Your run never finishes. This is where it was when the step budget ran out.",
  exception: "Your run raises an error here.",
  output_mismatch: "Every intermediate value matched a correct run; only the value returned differs.",
};

const KIND = {
  orienting: "Where to look",
  contradiction: "What doesn't add up",
  conceptual: "The idea underneath",
  procedural: "A way forward",
};

const SOURCES = {
  static: "static",
  dynamic: "dynamic",
  complexity: "cost",
  counterexample: "execution",
  model: "model",
};

export function renderMargin(root, app) {
  const { state } = app;
  const session = state.session;
  const current = session && session.current;
  const blocks = [];

  if (state.busy.submit) blocks.push(working());
  if (current && state.selectedTask !== "auto" && current.task.id !== state.selectedTask) {
    blocks.push(otherTask(app, current));
  }
  if (!current) {
    blocks.push(welcome(app));
  } else {
    blocks.push(verdict(app, current));
    if (current.diff) blocks.push(diff(current));
    if (current.notes.length) blocks.push(notes(current.notes));
    if (current.witness || current.divergence) blocks.push(witness(app, current));
    if (current.belief) blocks.push(belief(app, current));
    blocks.push(ladder(app, current));
    if (current.challenge.available) blocks.push(challenge(app, current));
    const clean = ["clean", "structural-clean", "incomplete"].includes(current.status);
    if (!clean && (current.alignment.root_cause || current.alignment.blame.length)) blocks.push(alignment(app, current));
    blocks.push(provenance(current));
  }

  const fresh = state.fresh.margin;
  root.replaceChildren(...blocks);
  root.classList.toggle("is-busy", Boolean(state.busy.submit));
  if (fresh) {
    blocks.forEach((el, index) => {
      el.classList.add("enter");
      el.style.setProperty("--i", String(index));
    });
    state.fresh.margin = false;
  }
}

// pieces
function card(kicker, agent, body, aside) {
  return h("section", { class: `card ${agent ? "agent-" + agent : ""}` },
    h("div", { class: "card-head" },
      h("h2", { class: "card-kicker" }, kicker),
      aside ? h("div", { class: "card-aside" }, aside) : null),
    body);
}

function working() {
  return h("section", { class: "card working", "aria-busy": "true" },
    h("div", { class: "card-kicker" }, "Working"),
    h("ol", { class: "working-steps" },
      ["Running your code against a reference, shrinking the failing input",
        "Tracing the mistake back through the prerequisite graph",
        "Writing questions and checking them for answer leakage"]
        .map((text, i) => h("li", { vars: { "--i": i } }, h("i"), text))));
}

function otherTask(app, current) {
  const selected = app.state.taskById.get(app.state.selectedTask);
  return h("div", { class: "banner" },
    icon("info"),
    h("span", null, "This analysis is for “", current.task.title || "your file",
      "”. Diagnose to analyse “", selected ? selected.title : "this task", "”."));
}

function welcome(app) {
  const { meta, examples } = app.state;
  const body = h("section", { class: "card welcome" },
    h("h1", { class: "welcome-title" }, "Diagnosing the belief, ", h("em", null, "not the bug.")),
    h("p", { class: "welcome-lede" },
      "Write your solution on the left and diagnose it. The tutor runs it against a reference on " +
      "many inputs, finds the smallest one it gets wrong, and asks you one question at a time. " +
      "It never hands you the answer."),
    h("ol", { class: "agents" },
      h("li", null, h("span", { class: "num" }, "1"), h("b", null, "Find where it breaks"),
        h("p", null, "The smallest failing input, and the line where your run first parts from a correct one.")),
      h("li", null, h("span", { class: "num" }, "2"), h("b", null, "Find why"),
        h("p", null, "The belief behind the mistake, traced back to the concept that made it possible.")),
      h("li", null, h("span", { class: "num" }, "3"), h("b", null, "Ask, don't tell"),
        h("p", null, "One question at a time. Each passes an answer-leakage gate before you see it."))),
    h("p", { class: "welcome-foot" }, h("span", { class: "kbd" }, app.modKey, " ↵"), "diagnoses from the editor"));

  if (meta && meta.demo && examples.length) {
    const forTask = examples.filter((ex) => app.state.selectedTask === "auto" || ex.task === app.state.selectedTask);
    if (forTask.length) {
      body.append(h("div", { class: "examples" },
        h("div", { class: "eyebrow" }, "Demo · flawed attempts from the bank"),
        h("div", { class: "examples-list" },
          forTask.map((ex) => h("button", {
            type: "button", class: "chip", on: { click: () => app.actions.loadExample(ex) },
          }, ex.id)))));
    }
  }
  return body;
}

function verdict(app, current) {
  const { status } = current;
  const style = STATUS[status];
  const primary = current.findings[0];
  const line = (primary && primary.line) || (current.divergence && current.divergence.line);

  let title;
  let caption;
  if (status === "diagnosis" || status === "provisional") {
    title = primary.name;
    caption = status === "provisional"
      ? "The evidence supports this ranking but does not settle it. Treat it as a hypothesis to test, not a verdict."
      : agreeing(primary.sources);
  } else if (status === "failing") {
    title = "Your function fails on an input";
    caption = "No belief in the tutor's taxonomy explains it yet, but the failing input and the line below still say where to look.";
  } else if (status === "incomplete") {
    title = "The analysis could not finish";
    caption = "Something in this attempt stopped the evaluator before it could compare your code with a reference, so nothing is concluded from it. Try returning a plain value.";
  } else if (status === "clean") {
    title = "Every input agrees with the reference";
    caption = "No structural warning sign either.";
  } else {
    title = "No warning sign in the structure";
    caption = "There is no reference for this function, so nothing was run against one: only its structure, dataflow and cost were checked.";
  }

  const metrics = [];
  if (primary) {
    metrics.push(metric("Belief",
      h("div", { class: "meter", vars: { "--value": primary.belief } }, h("i")),
      h("strong", null, fixed(primary.belief)),
      h("span", { class: "muted small" }, primary.severity)));
  }
  if (current.checks) {
    metrics.push(metric("Inputs", segments(current.checks),
      h("strong", null, `${current.checks.passed}/${current.checks.total}`), h("span", { class: "muted small" }, "pass")));
  }
  if (line) {
    metrics.push(metric("Located",
      h("button", { type: "button", class: "line-ref", on: { click: () => app.actions.showLine(line) } }, `line ${line}`)));
  }
  const cx = current.complexity;
  if (cx && cx.yours && cx.yours !== "unknown") {
    metrics.push(metric("Growth",
      h("code", { class: "inline" }, cx.yours),
      cx.target && cx.gap ? h("span", { class: "muted small nowrap" }, "target ", h("code", { class: "inline" }, cx.target)) : null,
      cx.target && !cx.gap ? h("span", { class: "muted small nowrap" }, "as required") : null,
      cx.gap ? h("span", { class: "tag tag-evidence-outline" }, "gap") : null));
  }

  const others = current.findings.slice(1);
  return h("section", { class: "card verdict", dataset: { status } },
    h("div", { class: "verdict-top" },
      h("span", { class: `tag ${style.cls}` }, style.tag),
      primary && primary.severity === "critical" ? h("span", { class: "tag tag-danger" }, "critical") : null,
      h("span", { class: "verdict-attempt" },
        `Attempt ${current.index}`, current.task.freeform ? "" : ` · ${current.task.title}`)),
    h("h2", { class: "verdict-title" }, title),
    h("p", { class: "verdict-caption" }, caption),
    metrics.length ? h("div", { class: "verdict-metrics" }, metrics) : null,
    others.length ? h("p", { class: "also" }, "Also considered: ",
      others.map((f, i) => [i ? ", " : "", f.name, h("span", { class: "muted" }, ` (${fixed(f.belief)})`)])) : null);
}

function metric(label, ...value) {
  return h("div", null, h("div", { class: "metric-label" }, label), h("div", { class: "metric-value" }, value));
}

function segments({ passed, total }) {
  if (total > 24) {
    return h("div", { class: "segments continuous", vars: { "--value": total ? passed / total : 0 } }, h("i"));
  }
  return h("div", { class: "segments", role: "img", "aria-label": `${passed} of ${total} inputs pass` },
    Array.from({ length: total }, (_, i) => h("i", { class: i < passed ? "pass" : "fail", vars: { "--i": i } })));
}

function diff(current) {
  const d = current.diff;
  const rows = [];
  const resolvedLabel = d.still_failing ? "no longer seen" : "resolved";
  for (const m of d.resolved) rows.push(diffRow(resolvedLabel, d.still_failing ? "plain" : "resolved", m.name));
  for (const m of d.persists) rows.push(diffRow("persists", "persists", m.name));
  for (const m of d.new) rows.push(diffRow("new", "new", m.name));
  if (!rows.length) rows.push(diffRow("findings", "plain", "No change in the findings"));
  if (d.checks) {
    const before = d.checks.before.passed / d.checks.before.total;
    const after = d.checks.after.passed / d.checks.after.total;
    rows.push(diffRow("inputs", "plain",
      h("span", { class: "delta" },
        `${d.checks.before.passed}/${d.checks.before.total}`,
        h("span", { class: `arrow ${after > before ? "up" : after < before ? "down" : "flat"}` }, "→"),
        h("strong", null, `${d.checks.after.passed}/${d.checks.after.total}`))));
  }
  if (d.mastery) {
    const up = d.mastery.after > d.mastery.before + 1e-6;
    const down = d.mastery.after < d.mastery.before - 1e-6;
    rows.push(diffRow("mastery", "plain",
      h("span", { class: "delta" }, d.mastery.name, " ",
        h("span", { class: "muted" }, fixed(d.mastery.before)),
        h("span", { class: `arrow ${up ? "up" : down ? "down" : "flat"}` }, up ? "↑" : down ? "↓" : "·"),
        h("strong", null, fixed(d.mastery.after)))));
  }
  return h("section", { class: "card diff agent-question" },
    h("div", { class: "card-head" },
      h("h2", { class: "card-kicker" }, `Since attempt ${current.index - 1}`),
      d.still_failing && d.resolved.length
        ? h("div", { class: "card-aside" }, "an input still fails")
        : null),
    h("div", { class: "diff-rows" }, rows));
}

function diffRow(key, cls, value) {
  return h("div", { class: "diff-row" }, h("span", { class: `diff-key ${cls}` }, key), h("span", null, value));
}

function notes(list) {
  return h("div", { class: "notes" },
    list.map((note) => h("div", { class: `note ${note.kind}` },
      icon(note.kind === "gate" ? "shield" : "info"), h("span", null, note.text))));
}

function witness(app, current) {
  const w = current.witness;
  const body = [];
  if (w) {
    const rows = [
      h("div", { class: "witness-key" }, "Smallest failing input"),
      h("div", { class: "witness-value input" }, h("span", { class: "pill neutral" }, w.input)),
    ];
    if (w.error) {
      rows.push(h("div", { class: "witness-key" }, "Your function"),
        h("div", { class: "witness-value" }, h("span", { class: "pill yours" }, w.error)));
    } else {
      rows.push(h("div", { class: "witness-key" }, "Your function returns"),
        h("div", { class: "witness-value" }, h("span", { class: "pill yours" }, w.yours)));
    }
    rows.push(h("div", { class: "witness-key" }, "A correct one returns"),
      h("div", { class: "witness-value" }, h("span", { class: "pill required" }, w.required)));
    body.push(h("div", { class: "witness-grid" }, rows));
    body.push(h("p", { class: "witness-note" },
      "Found by running your code beside a reference",
      w.steps ? `, then shrunk to this size in ${w.steps} delta-debugging step${w.steps === 1 ? "" : "s"}.` : "."));
  }
  const d = current.divergence;
  if (d && d.line) {
    body.push(h("div", { class: "divergence" },
      h("span", { class: "divergence-icon" }, icon("split")),
      h("div", { class: "divergence-text" }, DIVERGENCE[d.kind] || "This is where the runs part.", " ",
        h("button", { type: "button", class: "line-ref", on: { click: () => app.actions.showLine(d.line) } }, `line ${d.line}`),
        d.checkpoint && d.kind !== "non_termination" && d.kind !== "exception"
          ? h("span", { class: "muted" }, ` · step ${d.checkpoint} of the trace`)
          : null),
      d.snippet ? h("div", { class: "divergence-snippet" }, d.snippet) : null));
  }
  return card("What happened", "evidence", body);
}

function belief(app, current) {
  const evidence = current.evidence;
  const on = (n, opts) => app.actions.showLine(n, opts);
  const why = evidence.length
    ? h("details", { class: "disclosure", open: app.state.open.evidence || undefined,
      on: { toggle: (e) => { app.state.open.evidence = e.target.open; } } },
      h("summary", null, icon("chevron"), "Why the tutor thinks so"),
      h("ul", { class: "evidence-list" },
        evidence.map((item) => h("li", null,
          h("span", { class: "source-tag" }, `[${SOURCES[item.source] || item.source}]`),
          h("span", { class: "prose" }, rich(item.text, { onLine: on }))))))
    : null;
  return card("The belief", "evidence", [
    h("blockquote", { class: "belief-quote" }, `“${current.belief.voice}”`),
    h("p", { class: "belief-description" }, current.belief.description),
    why,
  ]);
}

function ladder(app, current) {
  const { total, revealed } = current.ladder;
  const quiet = current.status === "clean" || current.status === "structural-clean";
  const values = current.witness ? [current.witness.input] : [];
  const on = (n, opts) => app.actions.showLine(n, opts);
  const fresh = app.state.fresh.rung;

  const progress = total
    ? h("div", { class: "ladder-progress", "aria-label": `${revealed.length} of ${total} questions shown` },
      Array.from({ length: total }, (_, i) => h("i", { class: i < revealed.length ? "on" : "" })))
    : null;

  if (!total) {
    return card("Questions", "question", h("p", { class: "muted" },
      "There are no questions for this attempt: nothing was diagnosed."));
  }

  const rungs = revealed.map((hint) => h("li", {
    class: `rung${fresh === hint.rung ? " fresh" : ""}`, id: `rung-${hint.rung}`, tabindex: "-1",
  },
  h("span", { class: "rung-num" }, String(hint.rung)),
  h("div", { class: "rung-meta" },
    KIND[hint.kind] || hint.kind,
    hint.concept ? h("button", {
      type: "button", class: "chip chip-concept", on: { click: () => app.actions.openConcept(hint.concept.id) },
    }, hint.concept.name) : null),
  h("p", { class: "rung-text prose" }, rich(hint.text, { onLine: on, values }))));
  app.state.fresh.rung = null;

  for (let rung = revealed.length + 1; rung <= total; rung += 1) {
    rungs.push(h("li", { class: "rung locked" },
      h("span", { class: "rung-num" }, icon("lock")),
      h("p", { class: "rung-text" }, h("strong", null, `Question ${rung}`), " · ",
        rung === revealed.length + 1 ? "opens when you ask for it" : "opens after the one before it")));
  }

  const done = revealed.length >= total;
  const foot = h("div", { class: "ladder-foot" },
    done ? null : h("button", {
      type: "button",
      class: `button button-question${app.state.busy.hint ? " busy" : ""}`,
      disabled: app.state.busy.hint || undefined,
      on: { click: () => app.actions.nextHint() },
    }, h("span", { class: "button-label" },
      revealed.length ? "I'm stuck — next question" : "Ask the first question")),
    h("span", { class: "hint-text" },
      done
        ? "That was the last question. Try the experiment below, then edit your code and diagnose again."
        : revealed.length ? "Sit with the last one first." : "One at a time, never the answer."));

  const body = [
    h("ol", { class: "rungs" }, rungs),
    foot,
  ];
  if (quiet) {
    return card("Questions", "question", h("details", { class: "disclosure" },
      h("summary", null, icon("chevron"), `Your code passed. ${total} reflective questions remain if you want them.`),
      h("div", { class: "stack" }, body)), progress);
  }
  return h("section", { class: "card ladder-card agent-question" },
    h("div", { class: "card-head" }, h("h2", { class: "card-kicker" }, "Questions"), progress),
    body);
}

function challenge(app, current) {
  const c = current.challenge;
  if (!c.shown) {
    return card("Predict, then run", "question", [
      h("p", { class: "muted" }, "A small experiment on your own code: say what it will do, then watch it."),
      h("div", { class: "ladder-foot" },
        h("button", {
          type: "button", class: `button button-ghost${app.state.busy.challenge ? " busy" : ""}`,
          disabled: app.state.busy.challenge || undefined, on: { click: () => app.actions.openChallenge() },
        }, h("span", { class: "button-label" }, "Open the experiment"))),
    ]);
  }
  const run = app.state.run && app.state.run.attempt === current.index ? app.state.run : null;
  const values = c.input ? [c.input] : [];
  const field = h("input", {
    class: "field", id: "prediction", type: "text", autocomplete: "off", spellcheck: "false",
    placeholder: "e.g. 0", value: app.state.prediction || "",
    on: { input: (e) => { app.state.prediction = e.target.value; } },
  });
  const parts = [
    h("p", { class: "challenge-prompt prose" }, rich(c.prompt, { values, onLine: (n, o) => app.actions.showLine(n, o) })),
    c.question ? h("p", { class: "challenge-question" }, c.question) : null,
  ];
  if (c.runnable) {
    parts.push(h("div", { class: "predict" },
      h("label", { for: "prediction" }, "Your prediction"),
      h("div", { class: "predict-row" }, field,
        h("button", {
          type: "button", class: `button button-primary${app.state.busy.run ? " busy" : ""}`,
          disabled: app.state.busy.run || undefined,
          on: { click: () => app.actions.runChallenge(field.value) },
        }, icon("play"), h("span", { class: "button-label" }, "Run it"))),
      h("p", { class: "muted small" }, "Runs the code in the editor now, on ", h("code", { class: "inline" }, c.input),
        ", under the same limits as a diagnosis.")));
  }
  if (run) parts.push(runResult(run));
  return card("Predict, then run", "question", parts);
}

function runResult(run) {
  const r = run.result;
  const rows = [];
  if (run.prediction.trim()) {
    rows.push(h("div", { class: "witness-key" }, "You predicted"),
      h("div", { class: "witness-value" }, h("span", { class: "pill neutral" }, run.prediction.trim())));
  }
  rows.push(h("div", { class: "witness-key" }, r.ok ? "It returned" : "It"),
    h("div", { class: "witness-value" }, h("span", { class: `pill ${r.agrees ? "required" : "yours"}` }, r.ok ? r.value : r.error)));
  if (r.required !== null && r.required !== undefined) {
    rows.push(h("div", { class: "witness-key" }, "A correct one returns"),
      h("div", { class: "witness-value" }, h("span", { class: "pill required" }, r.required)));
  }
  const predicted = normalise(run.prediction);
  let sentence;
  if (!predicted) sentence = "Next time, commit to a prediction first: the surprise is the point.";
  else if (r.ok && predicted === normalise(r.value)) {
    sentence = r.agrees
      ? "You predicted your code exactly, and it now returns what the task requires here."
      : "You predicted your code exactly. Now: is that what the task asks for, and which line decides it?";
  } else {
    sentence = "Your code did something you did not predict. That gap is where the belief lives: find the step where your trace and the run part.";
  }
  return h("div", { class: "run-result", role: "status" },
    h("div", { class: "witness-grid" }, rows),
    h("p", { class: "run-verdict" }, sentence));
}

function alignment(app, current) {
  const a = current.alignment;
  const mastery = new Map();
  for (const c of (app.state.session.mastery || {}).concepts || []) mastery.set(c.id, c.value);
  for (const b of a.blame) if (b.after !== null) mastery.set(b.id, b.after);
  const threshold = (app.state.meta && app.state.meta.mastery_threshold) || 0.62;
  const node = (ref, isRoot) => {
    const value = mastery.get(ref.id);
    return h("button", {
      type: "button", class: `node${isRoot ? " is-root" : ""}`,
      title: value !== undefined ? `${ref.name}: mastery ${fixed(value)}` : ref.name,
      on: { click: () => app.actions.openConcept(ref.id) },
    }, h("span", { class: `ring${value !== undefined && value < threshold ? " weak" : ""}`, vars: { "--value": value || 0 } }), ref.name);
  };

  const parts = [];
  if (a.root_cause) {
    parts.push(h("div", { class: "root" },
      h("div", null,
        h("div", { class: "eyebrow" }, "Root cause"),
        h("div", { class: "root-name" }, a.root_cause.name),
        h("p", { class: "root-sub" },
          a.root_cause.depth !== null ? `${a.root_cause.depth} levels deep in the curriculum. ` : "",
          "The deepest concept implicated whose own prerequisites are already secure.")),
      h("button", { type: "button", class: "button button-ghost button-small", on: { click: () => app.actions.openConcept(a.root_cause.id) } },
        "Explore")));
  }
  if (a.paths.length) {
    parts.push(h("div", { class: "eyebrow" }, "Learning path"));
    for (const path of a.paths) {
      parts.push(h("div", { class: "path" },
        path.map((ref, i) => [i ? h("span", { class: "path-sep", "aria-hidden": "true" }, "→") : null, node(ref, i === 0)])));
    }
  }
  if (a.target && a.probability !== null) {
    const [low, high] = (app.state.meta && app.state.meta.zpd) || [0.45, 0.85];
    parts.push(h("div", { class: "zpd" },
      h("div", { class: "eyebrow" }, "Focus now"),
      h("p", { class: "zpd-text" },
        h("strong", null, a.target.name), ": ",
        `about ${pct(a.probability)} likely to click from where you are, `,
        a.in_zpd ? "inside your zone of proximal development." : `outside the ${pct(low)}–${pct(high)} band the tutor aims for.`),
      h("div", { class: "zpd-track", role: "img", "aria-label": `success probability ${pct(a.probability)}; target band ${pct(low)} to ${pct(high)}` },
        h("div", { class: "zpd-band", vars: { "--low": low, "--high": high } }),
        h("div", { class: "zpd-marker", vars: { "--value": a.probability } })),
      h("div", { class: "zpd-scale", "aria-hidden": "true" },
        h("span", null, "too hard"),
        h("span", { class: "zpd-zone", vars: { "--at": (low + high) / 2 } }, "your zone"),
        h("span", null, "too easy")),
      a.root_cause && a.target.id !== a.root_cause.id
        ? h("p", { class: "muted small" }, `The root cause is ${a.root_cause.name}; this focus was chosen because it is closer to your current reach.`)
        : null));
  }
  if (a.blame.length) {
    const top = Math.max(...a.blame.map((b) => b.mass || 0)) || 1;
    parts.push(h("details", { class: "disclosure", open: app.state.open.blame || undefined,
      on: { toggle: (e) => { app.state.open.blame = e.target.open; } } },
      h("summary", null, icon("chevron"), "Every implicated concept"),
      h("div", { class: "blame" },
        a.blame.map((b) => h("button", { type: "button", class: "blame-row", on: { click: () => app.actions.openConcept(b.id) } },
          h("span", { class: `blame-name${a.root_cause && b.id === a.root_cause.id ? " is-root" : ""}` }, b.name),
          h("span", { class: "blame-bar" }, h("i", { vars: { "--value": (b.mass || 0) / top } })),
          h("span", { class: "blame-shift" },
            b.before !== null && b.after !== null
              ? [fixed(b.before), " ", h("span", { class: b.after < b.before - 1e-6 ? "down" : "" }, b.after < b.before - 1e-6 ? "↓" : "·"), " ", fixed(b.after)]
              : "—"))))));
  }
  return card("Where it comes from", "concept", h("div", { class: "stack" }, parts));
}

function provenance(current) {
  const g = current.gate;
  const ok = g.leakage !== null && g.leakage <= g.limit;
  return h("footer", { class: "provenance" },
    h("span", { class: "gate-shield" }, icon("shield"),
      "answer leakage ", h("strong", { class: ok ? "ok" : "warn" }, fixed(g.leakage)), h("span", null, ` / limit ${fixed(g.limit)}`)),
    g.grounding !== null ? h("span", null, `grounding ${fixed(g.grounding)}`) : null,
    g.replaced ? h("span", null, `${g.replaced} rung${g.replaced === 1 ? "" : "s"} replaced by the gate`) : null,
    h("span", null, `reasoning: ${current.backend}`));
}

// helpers
function agreeing(sources) {
  const words = [...new Set(sources.map((s) => SOURCES[s] || s))];
  if (words.length < 2) return `Settled by ${words[0] || "the"} evidence.`;
  return `Settled by independent evidence: ${words.slice(0, -1).join(", ")} and ${words[words.length - 1]} agree.`;
}

function normalise(text) {
  return String(text || "").replace(/\s+/g, "").replace(/^['"](.*)['"]$/, "$1");
}
