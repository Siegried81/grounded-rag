import React, { useState } from "react";
import { useT } from "../i18n.js";

export default function Composer({ onSend, disabled }) {
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
        {disabled ? t("working") : t("ask")}
      </button>
    </form>
  );
}
