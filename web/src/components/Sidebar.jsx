import React, { useEffect, useRef, useState } from "react";
import { useT } from "../i18n.js";
import { Inline } from "../markdown.jsx";

// Must match the drawer breakpoint in styles.css.
const DRAWER_QUERY = "(max-width: 860px)";

/** True when the sidebar is an off-canvas drawer (narrow screens). */
function useIsDrawer() {
  const [narrow, setNarrow] = useState(() => window.matchMedia(DRAWER_QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(DRAWER_QUERY);
    const onChange = (e) => setNarrow(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

export default function Sidebar({
  open, onClose, corpora, corpus, onCorpus, settings, onSettings, config, onClear, canClear,
}) {
  const { t } = useT();
  const current = corpora.find((c) => c.name === corpus);
  const set = (patch) => onSettings({ ...settings, ...patch });
  const isDrawer = useIsDrawer();
  const closeRef = useRef(null);

  // Drawer behaviour: the closed drawer is only translated off-screen, so
  // `inert` keeps Tab from walking through its hidden controls; Escape closes
  // it and focus lands on the close button when it opens. On wide screens the
  // sidebar is always visible and none of this applies.
  // The focus effect depends only on `open`: `onClose` is a fresh arrow on
  // every App render and re-running it would steal focus from a control the
  // user is adjusting inside the open drawer.
  useEffect(() => {
    if (isDrawer && open) closeRef.current?.focus();
  }, [isDrawer, open]);

  useEffect(() => {
    if (!isDrawer || !open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isDrawer, open, onClose]);

  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} aria-hidden="true" />
      {/* React 18 only forwards `inert` as a string attribute, hence "" / undefined. */}
      <aside
        className={`sidebar ${open ? "open" : ""}`}
        aria-label={t("settings")}
        inert={isDrawer && !open ? "" : undefined}
      >
        <div className="sidebar-head">
          <span className="brand">{t("settings")}</span>
          <button
            ref={closeRef}
            type="button"
            className="icon-btn close-btn"
            onClick={onClose}
            aria-label={t("closeSettings")}
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>

        <section className="panel">
          <label htmlFor="corpus" className="label">{t("corpus")}</label>
          <select id="corpus" value={corpus || ""} onChange={(e) => onCorpus(e.target.value)}>
            {corpora.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
          {current && (
            <>
              <div className="stats">
                <div className="stat"><span className="stat-n">{current.documents}</span><span>{t("documents")}</span></div>
                <div className="stat"><span className="stat-n">{current.chunks}</span><span>{t("chunks")}</span></div>
              </div>
              <p className="caption">
                {t("detectedLanguage", { value: current.languages.join(", ") || t("unknown") })}
                {current.pages ? t("pdfPages", { n: current.pages }) : ""}
              </p>
              <details className="disclosure">
                <summary>{t("indexedFiles", { n: current.sources.length })}</summary>
                <ul className="files">
                  {current.sources.map((s) => <li key={s}><code>{s}</code></li>)}
                </ul>
              </details>
            </>
          )}
        </section>

        <section className="panel">
          <h2 className="panel-title">{t("retrieval")}</h2>
          <fieldset className="segmented">
            <legend className="label">{t("mode")}</legend>
            <div className="segmented-row">
              {(current?.modes || ["dense"]).map((m) => (
                <label key={m} className={settings.mode === m ? "on" : ""}>
                  <input
                    type="radio"
                    name="mode"
                    value={m}
                    checked={settings.mode === m}
                    onChange={() => set({ mode: m })}
                  />
                  {m}
                </label>
              ))}
            </div>
          </fieldset>
          <p className="caption">
            {current && !current.has_bm25 ? t("noBm25") : t("modesHelp")}
          </p>

          <label htmlFor="topk" className="label">
            {t("topK")} <strong>{settings.top_k}</strong>
          </label>
          <input
            id="topk"
            type="range"
            min="1"
            max="10"
            value={settings.top_k}
            onChange={(e) => set({ top_k: Number(e.target.value) })}
          />

          <label className="check">
            <input type="checkbox" checked={settings.use_mmr} onChange={(e) => set({ use_mmr: e.target.checked })} />
            {t("useMmr")}
          </label>
          {config && (
            <p className="caption">
              <Inline text={t("thresholdHelp", { value: config.refusal_threshold.toFixed(2) })} />
            </p>
          )}
          <button type="button" className="btn secondary wide" onClick={onClear} disabled={!canClear}>
            {t("clearConversation")}
          </button>
        </section>

        {config && (
          <>
            <details className="panel disclosure">
              <summary>{t("howItWorks")}</summary>
              <ol className="steps">
                {(config.pipeline_steps || []).map((s, i) => <li key={i}><Inline text={s} /></li>)}
              </ol>
            </details>
            <details className="panel disclosure">
              <summary>{t("limitations")}</summary>
              <ul className="limits">
                {(config.limitations || []).map((s, i) => <li key={i}><Inline text={s} /></li>)}
              </ul>
            </details>
          </>
        )}
      </aside>
    </>
  );
}
