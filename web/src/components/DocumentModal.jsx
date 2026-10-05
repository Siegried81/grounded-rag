// Full-document viewer opened from a source card.
//
// A source card only shows a window around the match, which is not enough to
// check a citation: this reads the whole file the passage came from, with the
// question's words highlighted by the server. It scrolls to the first highlight
// so the reader lands on the relevant part of a long document instead of its
// first line.
import React, { useEffect, useRef, useState } from "react";
import { getDocument } from "../api.js";
import { useT } from "../i18n.js";

export default function DocumentModal({ corpus, source, question, onClose }) {
  const { t } = useT();
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const body = useRef(null);
  const closeButton = useRef(null);

  useEffect(() => {
    // `cancelled` guards against a late response from a document the reader has
    // already closed or switched away from overwriting the current one.
    let cancelled = false;
    setDoc(null);
    setError(null);
    getDocument(corpus, source, question)
      .then((d) => !cancelled && setDoc(d))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [corpus, source, question]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Focus goes to the close button so Escape and Tab work without a click, and
  // the view jumps to the first match once the text is in the DOM.
  useEffect(() => {
    closeButton.current?.focus();
  }, []);
  useEffect(() => {
    if (!doc) return;
    const hit = body.current?.querySelector("mark");
    hit?.scrollIntoView({ block: "center" });
  }, [doc]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal doc-modal"
        role="dialog"
        aria-modal="true"
        aria-label={source}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="doc-head">
          <h2 className="doc-title">{source}</h2>
          {doc && <span className="doc-meta">{t("documentChars", { n: doc.chars })}</span>}
          <button type="button" className="doc-close" onClick={onClose} ref={closeButton}>
            {t("close")}
          </button>
        </header>
        <p className="caption">{t("documentHint")}</p>
        {error && <p className="verify-issue">{t("documentFailed", { error })}</p>}
        {!doc && !error && <p className="caption">{t("documentLoading")}</p>}
        {doc && (
          <>
            {doc.truncated && (
              <p className="verify-issue">{t("documentTruncated", { n: doc.shown_chars })}</p>
            )}
            {/* text_html is escaped server-side by ui.highlight_and_linkify; only <mark> tags
                and http(s) <a> links are added. */}
            <pre
              className="doc-body"
              ref={body}
              dangerouslySetInnerHTML={{ __html: doc.text_html }}
            />
          </>
        )}
      </div>
    </div>
  );
}
