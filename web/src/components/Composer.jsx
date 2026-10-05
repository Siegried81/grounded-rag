import React, { useState } from "react";
import { useT } from "../i18n.js";

/** Question input. `busy` (a request is in flight) is separate from
 * `disabled` (also no corpus, or still loading) so the button only says
 * "Working..." when something is actually being worked on. */
export default function Composer({ onSend, busy, disabled }) {
  const { t } = useT();
  const [text, setText] = useState("");
  const submit = (e) => {
    e.preventDefault();
    const q = text.trim();
    if (!q || disabled) return;
    onSend(q);
    setText("");
  };
  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-row">
        <label htmlFor="question" className="sr-only">{t("question")}</label>
        <textarea
          id="question"
          rows={1}
          value={text}
          placeholder={t("composerPlaceholder")}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter inserts a newline (chat convention).
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          maxLength={2000}
        />
        <button type="submit" className="btn primary" disabled={disabled || !text.trim()}>
          {busy ? t("working") : t("ask")}
        </button>
      </div>
      {/* Always visible, not a tooltip: AI-generated content must be disclosed
          to the reader (EU AI Act transparency obligation). */}
      <p className="ai-notice">{t("aiDisclosure")}</p>
    </form>
  );
}
