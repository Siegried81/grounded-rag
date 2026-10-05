// Thin fetch wrappers around the FastAPI backend (see api/main.py).
// `uiLang` is sent as the `ui_lang` query parameter so server-generated texts
// (limitations, pipeline steps, refusal explanations) come back in the
// interface language; a backend that does not know the parameter ignores it.

async function request(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch {
      /* non-JSON error body: keep the status line */
    }
    throw new Error(detail);
  }
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

// /api/ask reads `ui_lang` from the JSON body (it localises the refusal and
// verification explanations it builds), so it travels there, not in the query.
export const ask = (body, uiLang) =>
  request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(uiLang ? { ...body, ui_lang: uiLang } : body),
  });
