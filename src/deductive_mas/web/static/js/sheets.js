// Side sheets and dialogs: what the session believes you know, one concept
// of the curriculum, the history of attempts, the task picker. All native
// <dialog>, so focus, Escape and the backdrop behave without help.

import { clear, fixed, h, icon } from "./dom.js";

const sheet = () => document.getElementById("sheet");
const modal = () => document.getElementById("modal");

for (const id of ["sheet", "modal"]) {
  const dialog = document.getElementById(id);
  dialog.addEventListener("click", (event) => {
    // a click on the backdrop lands on the dialog element itself
    if (event.target === dialog || event.target.closest("[data-close]")) dialog.close();
  });
}

function openSheet(title, ...body) {
  const dialog = sheet();
  document.getElementById("sheet-title").textContent = title;
  clear(document.getElementById("sheet-body")).append(...body.flat().filter(Boolean));
  if (!dialog.open) dialog.showModal();
  document.getElementById("sheet-body").scrollTop = 0;
}

function openModal(title, ...body) {
  const dialog = modal();
  document.getElementById("modal-title").textContent = title;
  clear(document.getElementById("modal-body")).append(...body.flat().filter(Boolean));
  if (!dialog.open) dialog.showModal();
}

export function closeModal() {
  if (modal().open) modal().close();
}

export function showMastery(app) {
  const mastery = app.state.session && app.state.session.mastery;
  const concepts = (mastery && mastery.concepts) || [];
  const threshold = (mastery && mastery.threshold) || 0.62;
  if (!concepts.length) {
    openSheet("What the tutor believes you know",
      h("p", { class: "empty" }, "Nothing tracked yet. Diagnose some code and the estimate starts here."));
    return;
  }
  openSheet("What the tutor believes you know",
    h("p", { class: "mastery-intro" },
      "An estimate from your attempts in this session (Bayesian knowledge tracing on the prerequisite graph). ",
      "Only the concepts your attempts gave evidence about are listed. ",
      h("strong", null, `${mastery.below} of ${concepts.length}`), " are below the line at ", fixed(threshold), "."),
    h("ul", { class: "mastery-list" },
      concepts.map((c) => h("li", null,
        h("button", { type: "button", class: "mastery-row", on: { click: () => app.actions.openConcept(c.id) } },
          h("div", { class: "mastery-row-top" },
            h("span", null, c.name),
            h("span", { class: `state ${c.secure ? "secure" : "not-yet"}` }, c.secure ? "secure" : "not yet"),
            h("span", { class: "value" }, fixed(c.value))),
          h("div", { class: "mastery-bar", vars: { "--value": c.value, "--threshold": threshold } },
            h("i", { class: c.secure ? "secure" : "" }), h("b"))))),
    ));
}

export async function showConcept(app, id) {
  openSheet("Concept", h("p", { class: "empty" }, "Loading…"));
  let concept;
  try {
    concept = (await app.api.concept(id)).concept;
  } catch (error) {
    openSheet("Concept", h("p", { class: "empty" }, error.message));
    return;
  }
  const tree = (nodes) => h("ul", null, nodes.map((n) => h("li", null,
    h("button", { type: "button", class: "chip", on: { click: () => showConcept(app, n.id) } }, n.name),
    n.prerequisites && n.prerequisites.length ? tree(n.prerequisites) : null,
    n.more ? h("span", { class: "muted small" }, " …") : null)));
  openSheet("Concept",
    h("div", { class: "concept-name" }, concept.name),
    h("div", { class: "concept-sub" },
      h("span", { class: "tag tag-concept" }, concept.stratum.replace(/-/g, " ")),
      h("span", { class: "tag tag-plain" }, `depth ${concept.depth}`),
      concept.mastery !== null
        ? h("span", { class: "tag tag-question" }, `your mastery ${fixed(concept.mastery)}`)
        : null),
    h("p", { class: "concept-summary" }, concept.summary),
    h("section", { class: "concept-section" },
      h("div", { class: "eyebrow" }, "Rests on"),
      concept.prerequisites.length
        ? h("div", { class: "tree" }, tree(concept.prerequisites))
        : h("p", { class: "muted small" }, "Nothing: this is a foundation concept.")),
    concept.unlocks.length ? h("section", { class: "concept-section" },
      h("div", { class: "eyebrow" }, "Unlocks"),
      h("div", { class: "chips" }, concept.unlocks.map((u) => h("button", {
        type: "button", class: "chip", on: { click: () => showConcept(app, u.id) },
      }, u.name)))) : null);
}

export function showHistory(app) {
  const session = app.state.session;
  const attempts = (session && session.attempts) || [];
  const latest = attempts.length ? attempts[attempts.length - 1].index : null;
  const titles = app.state.taskById;
  openSheet("This session",
    attempts.length
      ? h("ol", { class: "history-list" }, attempts.slice().reverse().map((a) => h("li", {
        class: `history-row${a.index === latest ? " is-current" : ""}`,
      },
      h("span", { class: "history-num", dataset: { status: a.status } }, String(a.index)),
      h("div", null,
        h("div", { class: "history-title" },
          a.finding ? a.finding.name
            : a.status === "failing" ? "Failing input, no belief named"
              : a.status === "incomplete" ? "Analysis incomplete" : "No finding"),
        h("div", { class: "history-sub" },
          titles.get(a.task) ? titles.get(a.task).title : "Any function",
          a.belief !== null ? ` · belief ${fixed(a.belief)}` : "")),
      h("span", { class: "history-checks" }, a.checks ? `${a.checks.passed}/${a.checks.total}` : "—"))))
      : h("p", { class: "empty" }, "No attempts yet."),
    attempts.length ? h("p", { class: "muted small" },
      "Questions and the experiment always belong to the latest attempt.") : null);
}

export function pickTask(app) {
  const current = app.state.selectedTask;
  const option = (id, title, text, signature, difficulty) => h("li", null,
    h("button", {
      type: "button", class: "task-option", "aria-current": id === current ? "true" : "false",
      on: { click: () => { closeModal(); app.actions.selectTask(id); } },
    },
    h("span", { class: "task-option-title" }, title),
    signature ? h("span", { class: "signature" }, signature) : h("span"),
    h("span", { class: "task-option-text" }, text),
    difficulty !== null && difficulty !== undefined ? difficultyDots(difficulty) : null));
  openModal("Choose a task",
    h("ul", { class: "task-list" },
      app.state.tasks.map((t) => option(t.id, t.title, t.statement, t.signature, t.difficulty)),
      option("auto", "Any function",
        "Paste or write any Python function. The tutor matches it to a task by its name, or reports " +
        "structural findings when it has no reference for it.", null, null)));
}

export function difficultyDots(value) {
  // IRT difficulty in logits, roughly -1 to +1.5 on this bank
  const level = Math.max(1, Math.min(5, Math.round((value + 1) * 2) + 1));
  return h("span", { class: "difficulty", title: `IRT difficulty ${value >= 0 ? "+" : ""}${value}` },
    h("span", { class: "difficulty-label" }, "difficulty"),
    Array.from({ length: 5 }, (_, i) => h("i", { class: i < level ? "on" : "", "aria-hidden": "true" })),
    h("span", { class: "sr-only" }, `${level} of 5`));
}

export function confirmReset(app) {
  openModal("Start over?",
    h("p", { class: "confirm-text" },
      "This forgets the attempts and the mastery estimate of this session. The code in the editor stays."),
    h("div", { class: "confirm-actions" },
      h("button", { type: "button", class: "button button-ghost", on: { click: closeModal } }, "Cancel"),
      h("button", {
        type: "button", class: "button button-danger",
        on: { click: () => { closeModal(); app.actions.reset(); } },
      }, icon("check"), "Start over")));
}
