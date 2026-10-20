// A small Python tokenizer for highlighting. It only has to colour text,
// never to understand it, so it is a single pass of one sticky regex and
// tolerates half-typed code (an unterminated string runs to the line end).

const KEYWORDS = new Set(
  ("and as assert async await break class continue def del elif else except finally for " +
    "from global if import in is lambda nonlocal not or pass raise return try while with yield")
    .split(" "),
);
const CONSTANTS = new Set(["True", "False", "None"]);
const BUILTINS = new Set(
  ("abs all any bin bool bytes callable chr dict divmod enumerate filter float format " +
    "frozenset hash hex int isinstance issubclass iter len list map max min next object oct ord " +
    "pow print range repr reversed round set slice sorted str sum super tuple type zip " +
    "Exception ValueError TypeError IndexError KeyError ZeroDivisionError StopIteration " +
    "RecursionError deque heappush heappop heapify bisect_left bisect_right defaultdict Counter " +
    "lru_cache cache inf")
    .split(" "),
);

const TOKEN = new RegExp(
  [
    /(#[^\n]*)/.source,
    /((?:[rRbBuUfF]{1,2})?(?:'''[\s\S]*?(?:'''|$)|"""[\s\S]*?(?:"""|$)|'(?:\\.|[^'\\\n])*(?:'|(?=\n)|$)|"(?:\\.|[^"\\\n])*(?:"|(?=\n)|$)))/.source,
    /((?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?[jJ]?))/.source,
    /([\p{L}_][\p{L}\p{N}_]*)/.source,
    /(\s+)/.source,
    /([^\s])/.source,
  ].join("|"),
  "yu",
);

// -> lines, each a list of [className, text]
export function highlight(source) {
  const tokens = [];
  let previous = "";
  let lineStart = true;
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < source.length) {
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(source);
    if (!m) {
      tokens.push(["", source.slice(at)]);
      break;
    }
    const [text, comment, string, number, name, space] = m;
    let cls = "";
    if (comment !== undefined) cls = "t-com";
    else if (string !== undefined) cls = "t-str";
    else if (number !== undefined) cls = "t-num";
    else if (name !== undefined) {
      if (CONSTANTS.has(name)) cls = "t-const";
      else if (KEYWORDS.has(name)) cls = "t-kw";
      else if (previous === "def") cls = "t-fn";
      else if (previous === "class") cls = "t-cls";
      else if (name === "self" || name === "cls") cls = "t-self";
      else if (BUILTINS.has(name)) cls = "t-bi";
    } else if (space === undefined) {
      cls = text === "@" && lineStart ? "t-deco" : "t-op";
    }
    tokens.push([cls, text]);
    if (space === undefined) {
      previous = name !== undefined ? name : text;
      lineStart = false;
    } else if (text.includes("\n")) {
      lineStart = true;
    }
  }

  const lines = [[]];
  for (const [cls, text] of tokens) {
    const pieces = text.split("\n");
    pieces.forEach((piece, index) => {
      if (index > 0) lines.push([]);
      if (piece) lines[lines.length - 1].push([cls, piece]);
    });
  }
  return lines;
}
