import React, { useCallback, useEffect, useRef, useState } from "react";
import { ASK_TIMEOUT_MS, ask, getConfig, getCorpora } from "./api.js";
import { useT } from "./i18n.js";
import { Inline } from "./markdown.jsx";
import Header from "./components/Header.jsx";
import Sidebar from "./components/Sidebar.jsx";
import EmptyState from "./components/EmptyState.jsx";
import Composer from "./components/Composer.jsx";
import Turn from "./components/Turn.jsx";

let nextId = 1;

export default function App() {
  const { t, lang: uiLang } = useT();
  const [config, setConfig] = useState(null);
  const [corpora, setCorpora] = useState([]);
  const [corpus, setCorpus] = useState(null);
  const [settings, setSettings] = useState({ mode: "hybrid", top_k: 5, use_mmr: true });
  // Language the LLM answers in; "auto" means the language of the sources.
  const [answerLang, setAnswerLang] = useState("en");
  // One conversation per corpus, like the Streamlit app's session history.
  const [histories, setHistories] = useState({});
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Last turn that finished ({ id, status }): drives the scroll to its answer
  // and the screen-reader announcement, both of which happen once per turn.
  const [finished, setFinished] = useState(null);
  const bottomRef = useRef(null);
  // Server defaults are applied once; later config fetches (corpus or UI
  // language change) must not overwrite what the user has chosen since.
  const defaultsApplied = useRef(false);
  const corporaLoaded = useRef(false);

  const applyConfig = (cfg) => {
    setConfig(cfg);
    if (defaultsApplied.current) return;
    defaultsApplied.current = true;
    setSettings({ mode: cfg.retrieval_mode, top_k: cfg.top_k, use_mmr: cfg.use_mmr });
    if (cfg.default_language) setAnswerLang(cfg.default_language);
  };

  // Corpus list, re-fetched when the UI language changes because example
  // questions and other descriptions are localized server-side.
  // `live` discards a reply that lands after the effect was superseded, so
  // a quick language switch cannot leave the older language on screen.
  useEffect(() => {
    let live = true;
    getCorpora(uiLang)
      .then((data) => {
        if (!live) return;
        corporaLoaded.current = true;
        setCorpora(data.corpora);
        setCorpus((c) =>
          c && data.corpora.some((x) => x.name === c) ? c : data.corpora[0]?.name ?? null
        );
        // Without a corpus the config effect below never runs; fetch the
        // global config so the header and "no corpus" banner can render.
        if (!data.corpora.length) {
          return getConfig(null, uiLang).then((cfg) => live && applyConfig(cfg));
        }
      })
      .catch((e) => {
        if (live && !corporaLoaded.current) setLoadError(e.message);
      });
    return () => {
      live = false;
    };
  }, [uiLang]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = corpora.find((c) => c.name === corpus);

  // Per-corpus config: limitations depend on whether the corpus has BM25, and
  // limitations/pipeline steps are written in the UI language. Same `live`
  // guard: switching corpus twice must not let the first reply land last.
  useEffect(() => {
    if (!corpus) return;
    let live = true;
    getConfig(corpus, uiLang)
      .then((cfg) => live && applyConfig(cfg))
      .catch((e) => {
        if (live && !defaultsApplied.current) setLoadError(e.message);
      });
    return () => {
      live = false;
    };
  }, [corpus, uiLang]); // eslint-disable-line react-hooks/exhaustive-deps

  // The mode must be one the selected corpus supports.
  useEffect(() => {
    setSettings((s) =>
      current && !current.modes.includes(s.mode) ? { ...s, mode: current.modes[0] } : s
    );
  }, [current]);

  const turns = (corpus && histories[corpus]) || [];

  // A new question scrolls to the bottom so the loading indicator is visible.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  // A finished turn scrolls to the top of its answer instead: scrolling to
  // the bottom would land below the source cards, past the answer itself.
  useEffect(() => {
    if (!finished) return;
    document
      .getElementById(`turn-${finished.id}-assistant`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [finished]);

  const updateTurn = (c, id, patch) =>
    setHistories((h) => ({
      ...h,
      [c]: (h[c] || []).map((x) => (x.id === id ? { ...x, ...patch } : x)),
    }));

  const send = useCallback(
    async (question) => {
      if (!corpus || busy) return;
      const id = nextId++;
      const c = corpus;
      // Kept on the turn so the badge is right even when the backend does not
      // echo the language back yet.
      const requestedLanguage = answerLang;
      setHistories((h) => ({
        ...h,
        [c]: [...(h[c] || []), { id, question, status: "loading", requestedLanguage }],
      }));
      setBusy(true);
      try {
        const result = await ask(
          { corpus: c, question, ...settings, language: requestedLanguage },
          uiLang
        );
        updateTurn(c, id, { status: "done", result });
        setFinished({ id, status: result.refused ? "refused" : "done" });
      } catch (e) {
        // AbortSignal.timeout rejects with TimeoutError (AbortError in older
        // browsers); its message is not meant for users, so translate it.
        const timedOut = e.name === "TimeoutError" || e.name === "AbortError";
        const errorMessage = timedOut
          ? t("requestTimedOut", { s: ASK_TIMEOUT_MS / 1000 })
          : e.message;
        updateTurn(c, id, { status: "failed", errorMessage });
        setFinished({ id, status: "failed" });
      } finally {
        setBusy(false);
      }
    },
    [corpus, busy, settings, answerLang, uiLang, t]
  );

  const clear = () => setHistories((h) => ({ ...h, [corpus]: [] }));

  if (loadError) {
    return (
      <div className="fatal">
        <h1>{t("appTitle")}</h1>
        <div className="banner error" role="alert">
          <strong>{t("apiUnreachable")}</strong>
          <p>
            <Inline text={t("apiStartHint", { error: loadError })} />
          </p>
        </div>
      </div>
    );
  }

  // Example questions follow the UI language; older backends only send `examples`.
  const examples = current
    ? current.examples_by_language?.[uiLang] || current.examples || []
    : [];

  const announcement = finished
    ? t({ done: "announceReady", refused: "announceRefused", failed: "announceFailed" }[finished.status])
    : "";

  return (
    <div className="layout">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        corpora={corpora}
        corpus={corpus}
        onCorpus={(c) => {
          setCorpus(c);
          setSidebarOpen(false);
        }}
        settings={settings}
        onSettings={setSettings}
        config={config}
        onClear={clear}
        canClear={turns.length > 0 && !busy}
      />
      <main className="main">
        <Header
          config={config}
          corpus={corpus}
          mode={settings.mode}
          answerLang={answerLang}
          onAnswerLang={setAnswerLang}
          onToggleSidebar={() => setSidebarOpen(true)}
        />
        {/* The thread itself is not a live region: that would make screen
            readers read every source card. One status line announces the
            outcome of a turn instead; the loading indicator has its own. */}
        <div className="sr-only" role="status">{announcement}</div>
        <div className="thread">
          {config && corpora.length === 0 && (
            <div className="banner warn">
              <Inline text={t("noCorpus")} />
            </div>
          )}
          {!config && <div className="loading" role="status">{t("loading")}</div>}
          {current && turns.length === 0 && (
            <EmptyState examples={examples} onPick={send} disabled={busy} />
          )}
          {turns.map((x) => <Turn key={x.id} turn={x} />)}
          <div ref={bottomRef} />
        </div>
        <Composer onSend={send} busy={busy} disabled={busy || !corpus} />
      </main>
    </div>
  );
}
