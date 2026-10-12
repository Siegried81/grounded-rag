// Thin fetch wrappers around the FastAPI backend (see api/main.py).
// `uiLang` is sent as the `ui_lang` query parameter so server-generated texts
// (limitations, pipeline steps, refusal explanations) come back in the
// interface language; a backend that does not know the parameter ignores it.

// Turn a non-2xx response into an Error carrying FastAPI's `detail`.
async function httpError(resp) {
  let detail = `${resp.status} ${resp.statusText}`;
  try {
    const body = await resp.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
  } catch {
    /* non-JSON error body: keep the status line */
  }
  return new Error(detail);
}

async function request(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) throw await httpError(resp);
  return resp.json();
}

function withQuery(path, params) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v) qs.set(k, v);
  const s = qs.toString();
  return s ? `${path}?${s}` : path;
}

export const getConfig = (corpus, uiLang) =>
  request(withQuery("/api/config", { corpus, ui_lang: uiLang }));

export const getCorpora = (uiLang) => request(withQuery("/api/corpora", { ui_lang: uiLang }));

// One cited document in full, so a reader can check a citation in context.
// `question` is passed as `q` only to highlight the same terms as the source
// card; the document is the same whatever it is.
export const getDocument = (corpus, source, question) =>
  request(withQuery("/api/document", { corpus, source, q: question }));

// Upper bound on one /api/ask round trip. Free-tier LLM calls can take a
// minute or two under rate limiting, but a request that never returns would
// otherwise keep the UI locked (busy) forever.
export const ASK_TIMEOUT_MS = 120000;

// /api/ask reads `ui_lang` from the JSON body (it localises the refusal and
// verification explanations it builds), so it travels there, not in the query.
export const ask = (body, uiLang) =>
  request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(uiLang ? { ...body, ui_lang: uiLang } : body),
    signal: AbortSignal.timeout(ASK_TIMEOUT_MS),
  });

// Split a server-sent-events buffer into complete events. Returns the parsed
// events ({ event, data }) and the unfinished tail, which the caller keeps
// until the next chunk arrives: a network chunk can end in the middle of an
// event, or of a multi-byte character (the decoder handles the latter).
export function parseSse(buffer) {
  const events = [];
  const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = blocks.pop();
  for (const block of blocks) {
    let event = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")) });
  }
  return { events, rest };
}

// POST /api/ask/stream: same body and same final result as `ask`, but the
// answer arrives while the model writes it. `onEvent` receives the `sources`
// and `delta` events for display; the promise resolves with the `done`
// payload, which is exactly what /api/ask would have returned. A stream that
// ends without `done` is an error, never a partial result.
export async function askStream(body, uiLang, onEvent) {
  const resp = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(uiLang ? { ...body, ui_lang: uiLang } : body),
    signal: AbortSignal.timeout(ASK_TIMEOUT_MS),
  });
  // Validation and unknown-corpus errors come back as JSON, as for /api/ask.
  if (!resp.ok) throw await httpError(resp);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    for (const ev of parsed.events) {
      if (ev.event === "done") return ev.data;
      onEvent(ev);
    }
  }
  throw new Error("The answer stream ended before the answer was complete.");
}
