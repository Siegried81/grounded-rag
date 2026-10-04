// Minimal, safe Markdown renderer for LLM answers and help text.
// Supports paragraphs, "-"/"*" and "1." lists, **bold**, `code` and [S#]
// citation chips. Everything is rendered as React text nodes, never as raw
// HTML, so answer text cannot inject markup.
import React from "react";

const INLINE_RE = /(\*\*[^*]+\*\*|`[^`]+`|\[S\d+\])/g;

export function Inline({ text, renderCite }) {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    if (!part) return null;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    const cite = /^\[S(\d+)\]$/.exec(part);
    if (cite && renderCite) return <React.Fragment key={i}>{renderCite(Number(cite[1]))}</React.Fragment>;
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

export function Markdown({ text, renderCite }) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list);
    list = null;
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) {
        flush();
        list = { type, items: [] };
      }
      list.items.push((bullet || numbered)[1]);
    } else if (!line) {
      flush();
    } else {
      flush();
      // Strip heading markers; headings render as bold paragraphs in a chat bubble.
      const heading = /^#{1,6}\s+(.*)$/.exec(line);
      blocks.push({ type: heading ? "h" : "p", text: heading ? heading[1] : line });
    }
  }
  flush();
  return blocks.map((b, i) => {
    if (b.type === "ul" || b.type === "ol") {
      const Tag = b.type;
      return (
        <Tag key={i}>
          {b.items.map((it, j) => (
            <li key={j}>
              <Inline text={it} renderCite={renderCite} />
            </li>
          ))}
        </Tag>
      );
    }
    return (
      <p key={i} className={b.type === "h" ? "md-heading" : undefined}>
        <Inline text={b.text} renderCite={renderCite} />
      </p>
    );
  });
}
