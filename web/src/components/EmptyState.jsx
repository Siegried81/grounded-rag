import React from "react";
import { useT } from "../i18n.js";

export default function EmptyState({ examples, onPick, disabled }) {
  const { t } = useT();
  return (
    <div className="empty">
      <div className="empty-card">
        <h2>{t("emptyTitle")}</h2>
        <p>{t("emptyBody")}</p>
      </div>
      {examples.length > 0 && (
        <>
          <h3 className="examples-title">{t("tryThese")}</h3>
          <div className="examples">
            {examples.map((q) => (
              <button key={q} type="button" className="example" onClick={() => onPick(q)} disabled={disabled}>
                {q}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
