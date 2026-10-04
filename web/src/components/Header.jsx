import React from "react";
import { FALLBACK_ANSWER_LANGUAGES, UI_LANGUAGES, useT } from "../i18n.js";
import { Inline } from "../markdown.jsx";

/** Interface language toggle and "Answer in" select. They are separate on
 * purpose: someone may want an English interface over a French corpus and
 * still read answers in French, or the other way round. */
function LanguageControls({ config, answerLang, onAnswerLang }) {
  const { t, lang, setLang } = useT();
  const answerLanguages = config?.languages?.length ? config.languages : FALLBACK_ANSWER_LANGUAGES;
  return (
    <div className="lang-controls" role="group" aria-label={t("languageControls")}>
      <div className="lang-toggle" role="group" aria-label={t("uiLanguage")}>
        {UI_LANGUAGES.map((l) => (
          <button
            key={l}
            type="button"
            className={l === lang ? "on" : ""}
            aria-pressed={l === lang}
            lang={l}
            title={t(`uiLanguageName.${l}`)}
            onClick={() => setLang(l)}
          >
            {t(`uiLanguageShort.${l}`)}
          </button>
        ))}
      </div>
      <label className="answer-lang">
        <span>{t("answerIn")}</span>
        <select value={answerLang} onChange={(e) => onAnswerLang(e.target.value)}>
          {answerLanguages.map((l) => (
            <option key={l} value={l}>{t(`answerLanguageOption.${l}`)}</option>
          ))}
        </select>
      </label>
    </div>
  );
}

export default function Header({ config, corpus, mode, answerLang, onAnswerLang, onToggleSidebar }) {
  const { t } = useT();
  return (
    <header className="header">
      <div className="header-top">
        <button
          type="button"
          className="icon-btn sidebar-toggle"
          onClick={onToggleSidebar}
          aria-label={t("openSettings")}
        >
          <span aria-hidden="true">☰</span>
        </button>
        <div className="header-text">
          <h1>{t("appTitle")}</h1>
          <p className="rule">
            <Inline text={t("headerRule")} />
          </p>
        </div>
        <LanguageControls config={config} answerLang={answerLang} onAnswerLang={onAnswerLang} />
      </div>
      {config && (
        <ul className="badges" aria-label={t("activeConfig")}>
          {corpus && <li>{t("badgeCorpus", { value: corpus })}</li>}
          <li>{t("badgeRetrieval", { value: mode })}</li>
          <li>{t("badgeLlm", { provider: config.llm_provider, model: config.llm_model })}</li>
          <li>{t("badgeEmbeddings", { value: config.embed_provider })}</li>
          <li>{t("badgeRefusal", { value: config.refusal_threshold.toFixed(2) })}</li>
          {!config.llm_key_configured && <li className="warn">{t("badgeNoKey")}</li>}
        </ul>
      )}
    </header>
  );
}
