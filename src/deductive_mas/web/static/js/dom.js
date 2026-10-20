// Building DOM without innerHTML. Every string that reaches the page goes
// through a text node: guidance can come from a language model, and code
// comes from the learner, so neither is ever parsed as markup.

const SVG = "http://www.w3.org/2000/svg";

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) apply(el, props);
  append(el, children);
  return el;
}

function apply(el, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "on") for (const [type, fn] of Object.entries(value)) el.addEventListener(type, fn);
    else if (key === "vars") for (const [name, v] of Object.entries(value)) el.style.setProperty(name, String(v));
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

const PATHS = {
  arrow: ["M4 10h12", "m11.5 5.5 4.5 4.5-4.5 4.5"],
  chevron: ["m8 5.5 4.5 4.5L8 14.5"],
  down: ["M5.5 7.5 10 12l4.5-4.5"],
  lock: ["M6 9V7a4 4 0 0 1 8 0v2", "M5 9h10v7H5z"],
  check: ["m4.5 10.5 3.5 3.5 7.5-8"],
  split: ["M4 4v4a4 4 0 0 0 4 4h8", "M16 16v-4", "m12.5 8.5 3.5 3.5-3.5 3.5"],
  play: ["M7 5.5v9l7-4.5z"],
  shield: ["M10 2.8 4.5 5v4.6c0 3.6 2.4 6.2 5.5 7.6 3.1-1.4 5.5-4 5.5-7.6V5z", "m7.5 10 1.8 1.8 3.3-3.6"],
  info: ["M10 9v5", "M10 6.2v.1", "M10 17.5a7.5 7.5 0 1 0 0-15 7.5 7.5 0 0 0 0 15Z"],
  graph: ["M6 6.5a2 2 0 1 0 0-.01", "M14 6.5a2 2 0 1 0 0-.01", "M10 15a2 2 0 1 0 0-.01", "m7.2 8 2 4.8M12.8 8l-2 4.8"],
  spark: ["M10 3v3M10 14v3M3 10h3M14 10h3M5.2 5.2l2 2M12.8 12.8l2 2M5.2 14.8l2-2M12.8 7.2l2-2"],
};

export function icon(name, cls = "icon") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 20 20");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  for (const d of PATHS[name] || []) {
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

// Prose with `code` spans. "line 4" becomes a button that points the editor
// at that line, and any of the given literals (the failing input, say)
// becomes a value chip so the eye can find it again in the code panel.
export function rich(text, { onLine, values = [] } = {}) {
  const fragment = document.createDocumentFragment();
  const parts = String(text || "").split(/(`[^`\n]+`)/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      fragment.append(h("code", { class: "inline" }, part.slice(1, -1)));
      continue;
    }
    fragment.append(prose(part, onLine, values.filter(Boolean)));
  }
  return fragment;
}

function prose(text, onLine, values) {
  const fragment = document.createDocumentFragment();
  const patterns = [];
  if (onLine) patterns.push("\\bline (\\d+)\\b");
  for (const value of values) patterns.push(escapeRegExp(value));
  if (!patterns.length) {
    fragment.append(text);
    return fragment;
  }
  const re = new RegExp(patterns.map((p) => `(${p})`).join("|"), "g");
  let last = 0;
  for (const match of text.matchAll(re)) {
    if (match.index > last) fragment.append(text.slice(last, match.index));
    const lineMatch = onLine ? /^line (\d+)$/.exec(match[0]) : null;
    if (lineMatch) {
      const n = Number(lineMatch[1]);
      fragment.append(
        h("button", {
          type: "button",
          class: "line-ref",
          title: `Show line ${n} in the editor`,
          on: { click: () => onLine(n), mouseenter: () => onLine(n, { hover: true }) },
        }, `line ${n}`),
      );
    } else {
      fragment.append(h("span", { class: "value-ref" }, match[0]));
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) fragment.append(text.slice(last));
  return fragment;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

export function fixed(value, places = 2) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(places);
}

export function announce(message) {
  const region = document.getElementById("announcer");
  if (!region) return;
  region.textContent = "";
  // a fresh node so the same sentence twice is read twice
  requestAnimationFrame(() => {
    region.textContent = message;
  });
}

let toastTimer = null;
export function toast(message, ms = 4200) {
  document.querySelectorAll(".toast").forEach((el) => el.remove());
  const el = h("div", { class: "toast", role: "status" }, message);
  document.body.append(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), ms);
}
