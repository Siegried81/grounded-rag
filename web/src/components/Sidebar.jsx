import React from "react";
import { useT } from "../i18n.js";
import { Inline } from "../markdown.jsx";

export default function Sidebar({
  open, onClose, corpora, corpus, onCorpus, settings, onSettings, config, onClear, canClear,
}) {
  const { t } = useT();
  const current = corpora.find((c) => c.name === corpus);
  const set = (patch) => onSettings({ ...settings, ...patch });

  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label={t("settings")}>
        <div className="sidebar-head">
          <span className="brand">{t("settings")}</span>
          <button type="button" className="icon-btn close-btn" onClick={onClose} aria-label={t("closeSettings")}>
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
