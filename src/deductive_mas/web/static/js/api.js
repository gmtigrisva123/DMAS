// The tutoring API. The session lives in an HttpOnly cookie this script
// cannot read; every call is same-origin JSON.

export class ApiError extends Error {
  constructor(status, code, message, detail = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function call(method, path, body) {
  let response;
  try {
    response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? { Accept: "application/json" } : {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", "Can't reach the tutor. Check the connection and try again.");
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const error = (payload && payload.error) || {};
    const { code, message, ...detail } = error;
    throw new ApiError(
      response.status,
      code || "http",
      message || "The tutor could not answer that request.",
      detail,
    );
  }
  return payload;
}

export const api = {
  meta: () => call("GET", "/api/meta"),
  tasks: () => call("GET", "/api/tasks"),
  examples: () => call("GET", "/api/examples"),
  session: () => call("GET", "/api/session"),
  concept: (id) => call("GET", `/api/concepts/${encodeURIComponent(id)}`),
  submit: (source, task) => call("POST", "/api/attempts", { source, task }),
  hint: (attempt, rung) => call("POST", `/api/attempts/${attempt}/hints`, { rung }),
  challenge: (attempt) => call("POST", `/api/attempts/${attempt}/challenge`, {}),
  run: (attempt, source) => call("POST", `/api/attempts/${attempt}/challenge/run`, { source }),
  reset: () => call("POST", "/api/session/reset", {}),
};
