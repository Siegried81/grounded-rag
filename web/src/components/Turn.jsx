import React, { useRef, useState } from "react";
import { useT } from "../i18n.js";
import { Markdown } from "../markdown.jsx";
import DocumentModal from "./DocumentModal.jsx";

function SourceCard({ src, domId, flash, onOpenDocument }) {
  const { t } = useT();
  return (
    <article
      id={domId}
      className={`src ${src.cited ? "cited" : ""} ${flash ? "flash" : ""}`}
      tabIndex={-1}
      aria-label={t("sourceLabel", { sid: src.sid, location: src.location })}
    >
      <header className="src-head">
        <span className="src-id">S{src.sid}</span>
        <button
          type="button"
          className="src-file src-link"
          title={t("openDocument", { source: src.source })}
          onClick={() => onOpenDocument(src.source)}
        >
          {src.source}
        </button>
        {src.page != null && <span className="src-page">{t("pageShort", { n: src.page })}</span>}
        <span className="src-score" title={t("scoreTitle")}>
          {src.score.toFixed(3)}
        </span>
        {src.cited && <span className="src-cited">{t("cited")}</span>}
      </header>
      {/* excerpt_html is escaped server-side by ui.highlight_and_linkify; only <mark>
          tags and http(s) <a> links are added. */}
      <p className="src-body" dangerouslySetInnerHTML={{ __html: src.excerpt_html }} />
    </article>
  );
}

function Sources({ title, sources, turnId, flashSid, onOpenDocument }) {
  const { t } = useT();
  if (!sources.length) return null;
  return (
    <section className="sources" aria-label={title}>
      <h3 className="section-title">{title}</h3>
      <div className="src-grid">
        {sources.map((s) => (
          <SourceCard
            key={s.sid}
            src={s}
            domId={`src-${turnId}-${s.sid}`}
            flash={flashSid === s.sid}
            onOpenDocument={onOpenDocument}
          />
        ))}
      </div>
      <p className="caption">{t("scoreNote")}</p>
    </section>
  );
}

function Verification({ v }) {
  const { t } = useT();
  return (
    <section className={`verify ${v.ok ? "ok" : "bad"}`} aria-label={t("verification")}>
      <div className="verify-head">
        <span className="verify-badge">{v.ok ? t("verified") : t("issuesFound")}</span>
        <span>
          {t("grounding")} <strong>{v.grounding_score.toFixed(2)}</strong>
        </span>
      </div>
      <p className="caption">{v.explanation}</p>
      {v.invalid_citations.length > 0 && (
        <p className="verify-issue">
          {t("invalidCitations", { list: v.invalid_citations.map((n) => `[S${n}]`).join(", ") })}
        </p>
      )}
      {v.uncited_sentences.length > 0 && (
        <div className="verify-issue">
          {t("uncitedClaims")}
          <ul>
            {v.uncited_sentences.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
      )}
    </section>
  );
}

function Latency({ latency }) {
  const { t } = useT();
  if (!latency) return null;
  return (
    <div className="latency" aria-label={t("timings")}>
      {latency.retrieval_s != null && (
        <span className="chip">{t("retrievalTime", { s: latency.retrieval_s.toFixed(2) })}</span>
      )}
      {latency.answer_s != null && (
        <span className="chip">{t("answerTime", { s: latency.answer_s.toFixed(2) })}</span>
      )}
    </div>
  );
}

/** "Answered in English" badge. Prefers the language the server reports and
 * falls back to the one requested, for backends that do not echo it. */
function AnswerLanguage({ language }) {
  const { t } = useT();
  if (!language) return null;
  const label =
    language === "auto"
      ? t("answeredInSourceLanguage")
      : t("answeredIn", { language: t(`languageName.${language}`) });
  return <span className="chip lang-chip">{label}</span>;
}

function ErrorBanner({ title, hint }) {
  return (
    <div className="banner error" role="alert">
      <strong>{title}</strong>
      {hint && <Markdown text={hint} />}
    </div>
  );
}

export default function Turn({ turn }) {
  const { t } = useT();
  const [flashSid, setFlashSid] = useState(null);
  const [openSource, setOpenSource] = useState(null);
  const timer = useRef(null);
  const r = turn.result;

  // Clicking a citation chip scrolls its source card into view and pulses it.
  const focusSource = (n) => {
    const el = document.getElementById(`src-${turn.id}-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    el.focus({ preventScroll: true });
    setFlashSid(n);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setFlashSid(null), 1600);
  };

  const renderCite = (n) => {
    const valid = r && n >= 1 && n <= r.sources.length;
    return valid ? (
      <button type="button" className="cite" onClick={() => focusSource(n)} aria-label={t("showSource", { n })}>
        S{n}
      </button>
    ) : (
      <span className="cite-bad" title={t("badCitation")}>S{n}</span>
    );
  };

  return (
    <div className="turn">
      <div className="msg user">
        <div className="bubble">{turn.question}</div>
      </div>
      <div className="msg assistant">
        <div className="avatar" aria-hidden="true">S</div>
        <div className="assistant-body">
          {turn.status === "loading" && (
            <div className="loading" role="status">
              <span className="dots" aria-hidden="true"><i /><i /><i /></span>
              {t("thinking")}
            </div>
          )}

          {turn.status === "failed" && (
            <ErrorBanner
              title={t("requestFailed")}
              hint={t("requestFailedHint", { error: turn.errorMessage })}
            />
          )}

          {r && r.error && <ErrorBanner title={r.error.message} hint={r.error.hint} />}

          {r && r.refused && (
            <>
              <div className="refusal" role="note">
                <strong>{t("refusalTitle")}</strong>
                <p>{r.refusal.explanation}</p>
                {r.refusal.best_score != null && (
                  <div className="meter" aria-hidden="true">
                    <div className="meter-bar" style={{ width: `${Math.max(0, Math.min(1, r.refusal.best_score)) * 100}%` }} />
                    <div className="meter-gate" style={{ left: `${r.refusal.threshold * 100}%` }} />
                  </div>
                )}
              </div>
              <Sources
                title={t("closestPassages")}
                sources={r.refusal.closest}
                turnId={turn.id}
                flashSid={flashSid}
                onOpenDocument={setOpenSource}
              />
            </>
          )}

          {r && !r.refused && (
            <>
              {r.answer && (
                <div className="answer">
                  <Markdown text={r.answer} renderCite={renderCite} />
                </div>
              )}
              {r.verification && <Verification v={r.verification} />}
              <Sources
                title={t("sources")}
                sources={r.sources}
                turnId={turn.id}
                flashSid={flashSid}
                onOpenDocument={setOpenSource}
              />
            </>
          )}

          {r && (
            <div className="latency-row">
              {r.answer && !r.refused && <AnswerLanguage language={r.language || turn.requestedLanguage} />}
              <Latency latency={r.latency} />
            </div>
          )}
        </div>
      </div>
      {openSource && (
        <DocumentModal
          corpus={turn.corpus}
          source={openSource}
          question={turn.question}
          onClose={() => setOpenSource(null)}
        />
      )}
    </div>
  );
}
