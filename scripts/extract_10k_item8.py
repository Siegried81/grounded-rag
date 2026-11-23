"""Extract Item 8 (Financial Statements and Supplementary Data) of a 10-K into text sections.

The six sections in data/filings_sections/ cover the narrative Items; the
numbers live in Item 8, which is HTML tables in the raw SEC submission. This
script reads the primary 10-K document of that submission, walks its HTML with
the standard library only, and writes one .txt file per statement or note group
into its own corpus, data/filings_statements/ (kept apart from the narrative
corpus, see docs/decisions.md, 2026-11-23).

Why the output looks the way it does: a table is flattened to one line per row,
`label: 2025 | 2024 | 2023`, so a person can read it and the line-aware chunker
(rag/smart_chunk.py, `chunk_contextual_text`) never cuts a number from its label
or year. Each file starts with a `Context:` line, which that chunker puts on top
of every chunk, then the statement's title, the unit stated in the filing ("in
millions, except number of shares, which are reflected in thousands, and
per-share amounts") and the year order, because a chunk retrieved alone has to
say what its numbers mean. Text is copied from the filing, never rewritten.

Usage:
    python scripts/extract_10k_item8.py                       # default paths
    python scripts/extract_10k_item8.py --raw X.txt --out data/filings_statements
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "filings" / "raw" / "apple_fy2025_10k_sec-submission.txt"
DEFAULT_OUT = ROOT / "data" / "filings_statements"

# (output file, heading that opens the section). A section runs from its heading
# to the heading of the next entry; the last one runs to the end of Item 8. Every
# note is kept, not only the ones the eval questions use: Item 8 is one part of
# the filing and a reader looking for leases or stock plans should find them.
SECTIONS = [
    ("07_item8_income_statement.txt", "CONSOLIDATED STATEMENTS OF OPERATIONS"),
    ("07_item8_comprehensive_income.txt", "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"),
    ("07_item8_balance_sheet.txt", "CONSOLIDATED BALANCE SHEETS"),
    ("07_item8_shareholders_equity.txt", "CONSOLIDATED STATEMENTS OF SHAREHOLDERS"),
    ("07_item8_cash_flows.txt", "CONSOLIDATED STATEMENTS OF CASH FLOWS"),
    ("07_item8_note_1_accounting_policies.txt", "Note 1 "),
    ("07_item8_note_2_revenue.txt", "Note 2 "),
    ("07_item8_note_3_earnings_per_share.txt", "Note 3 "),
    ("07_item8_note_4_financial_instruments.txt", "Note 4 "),
    ("07_item8_note_5_property_plant_equipment.txt", "Note 5 "),
    ("07_item8_note_6_statement_details.txt", "Note 6 "),
    ("07_item8_note_7_income_taxes.txt", "Note 7 "),
    ("07_item8_note_8_leases.txt", "Note 8 "),
    ("07_item8_note_9_debt.txt", "Note 9 "),
    ("07_item8_note_10_shareholders_equity.txt", "Note 10 "),
    ("07_item8_note_11_share_based_compensation.txt", "Note 11 "),
    ("07_item8_note_12_commitments_contingencies.txt", "Note 12 "),
    ("07_item8_note_13_segment_information.txt", "Note 13 "),
    ("07_item8_auditor_reports.txt", "Report of Independent Registered Public Accounting Firm"),
]
# Files whose tables get the column name on every value (see label_columns).
PAIRED = {"07_item8_note_13_segment_information.txt"}
END_MARKER = "Reports of Independent Registered Public Accounting Firm"

BLOCK_TAGS = {"div", "p", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}
CELL_TAGS = {"td", "th"}


class _Flattener(HTMLParser):
    """HTML to a list of lines, tables flattened to `label: v1 | v2 | v3`.

    The hidden inline-XBRL header is skipped (it is a block of tagged facts, not
    text a reader sees). Cells are collected per row; empty cells and the
    separate "$", "%" and ")" cells SEC filings use for alignment are folded into
    the number they belong to.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._hidden = 0
        self._stack: list[bool] = []

    def handle_starttag(self, tag, attrs):
        hides = tag == "ix:header" or (
            tag == "div" and "display:none" in (dict(attrs).get("style") or "").replace(" ", "")
        )
        if tag == "br" and self._cell is not None:
            self._cell.append(" ")  # "Greater<br>China" must not become "GreaterChina"
        if tag not in ("br", "meta", "link", "img", "hr"):
            self._stack.append(hides)
        self._hidden += hides
        if self._hidden:
            return
        if tag == "tr":
            self._flush()
            self._row = []
        elif tag in CELL_TAGS and self._row is not None:
            self._cell = []
        elif tag in BLOCK_TAGS and self._row is None:
            self._flush()

    def handle_endtag(self, tag):
        if self._stack:
            self._hidden -= self._stack.pop()
        if self._hidden:
            return
        if tag in CELL_TAGS and self._row is not None and self._cell is not None:
            self._row.append(_clean("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            line = _row_to_line(self._row)
            if line:
                self.lines.append(line)
            self._row = None
        elif tag in BLOCK_TAGS and self._row is None:
            self._flush()

    def handle_data(self, data):
        if self._hidden:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._row is None:
            self._buf.append(data)

    def _flush(self) -> None:
        text = _clean("".join(self._buf))
        self._buf = []
        if text:
            self.lines.append(text)


def _clean(text: str) -> str:
    """Collapse whitespace and non-breaking spaces, keep the characters."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _row_to_line(cells: list[str]) -> str:
    """Turn one table row into `label: v1 | v2 | v3` (or a plain line)."""
    if cells and not cells[0] and len([c for c in cells if c]) > 1:
        # No row label: a column-header row (fiscal year dates, "Level 1"...).
        return " | ".join(re.sub(r"([A-Z][a-z]+ \d{1,2}),(\d{4})", lambda m: f"{m[1]}, {m[2]}", c) for c in cells if c)
    merged: list[str] = []
    for c in cells:
        if not c:
            continue
        if c in ("$", "(") and merged is not None:
            merged.append(c)
            continue
        if c in (")", "%", ")%") and merged:
            merged[-1] += c
            continue
        if merged and merged[-1] in ("$", "("):
            merged[-1] = (merged[-1] + c) if merged[-1] == "(" else c
            continue
        merged.append(c)
    merged = [m for m in merged if m != "$"]
    if not merged:
        return ""
    if len(merged) == 1:
        return merged[0]
    return f"{merged[0]}: " + " | ".join(merged[1:])


def primary_document(raw: str) -> str:
    """The 10-K document inside the SEC full-submission text file."""
    start = raw.index("<TYPE>10-K")
    return raw[start:raw.index("</DOCUMENT>", start)]


def item8_lines(raw: str) -> list[str]:
    """Flattened lines of Part II Item 8, from its heading to the start of Item 9."""
    parser = _Flattener()
    parser.feed(primary_document(raw))
    parser._flush()
    footer = re.compile(r"Apple Inc\.( \| \d{4} Form 10-K \| \d+)?$")
    lines = [ln for ln in parser.lines if not footer.match(ln)]
    heads = [i for i, ln in enumerate(lines) if re.match(r"Item 8\.\s*Financial Statements", ln)]
    if not heads:
        raise ValueError("Item 8 heading not found")
    start = heads[-1]  # the first match is the table of contents
    end = next(
        i for i in range(start + 1, len(lines)) if re.match(r"Item 9\.\s*Changes in and Disagreements", lines[i])
    )
    return lines[start:end]


def split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Cut Item 8 into the SECTIONS files, each with the Item 8 caption on top."""
    # Heading positions, searched after the end of the auditor's report so the
    # index at the top of Item 8 and the report text are not matched.
    floor = next((i for i, ln in enumerate(lines) if ln.startswith(END_MARKER)), 0)
    positions: list[tuple[int, str]] = []
    cursor = floor
    for name, marker in SECTIONS:
        pat = marker.strip().upper() if marker.isupper() else marker
        hit = next(
            (
                i
                for i in range(cursor, len(lines))
                if (lines[i].upper().startswith(pat) if marker.isupper() else lines[i].startswith(pat))
                and len(lines[i]) < 120
            ),
            None,
        )
        if hit is None:
            raise ValueError(f"heading not found for {name}: {marker!r}")
        positions.append((hit, name))
        cursor = hit + 1
    out: dict[str, list[str]] = {}
    for n, (pos, name) in enumerate(positions):
        stop = positions[n + 1][0] if n + 1 < len(positions) else len(lines)
        out[name] = lines[pos:stop]
    return out


def label_columns(lines: list[str]) -> list[str]:
    """Put the column name on each value of a table whose columns are not years.

    The segment note has one table per year with columns Americas | Europe | ...;
    a chunk cut in the middle of it would hold numbers with no idea which region
    they belong to. Rows become `label (2025): Americas 178,353 | Europe 111,032`.
    Year-only headers (`2025 | 2024 | 2023`) are left alone: the header on top of
    every statement already says it, and the lines stay short.
    """
    out: list[str] = []
    year = ""
    header: list[str] = []
    for ln in lines:
        if re.fullmatch(r"[0-9]{4}", ln):
            year, header = ln, []
            out.append(ln)
            continue
        cols = [c.strip() for c in ln.split("|")]
        is_header = ":" not in ln and len(cols) > 2 and not any(re.search(r"[0-9]{3}", c) for c in cols)
        if is_header:
            header = [re.sub(r"([a-z])([A-Z])", lambda m: f"{m[1]} {m[2]}", c) for c in cols]
            out.append(" | ".join(header))
            continue
        label, sep, values = ln.partition(": ")
        vals = [v.strip() for v in values.split("|")] if sep else []
        if header and sep and len(vals) == len(header):
            tag = f" ({year})" if year else ""
            out.append(f"{label}{tag}: " + " | ".join(f"{h} {v}" for h, v in zip(header, vals, strict=True)))
        else:
            out.append(ln)
    return out


def render(name: str, body: list[str]) -> str:
    """One file's text: a `Context:` header line, then the body lines."""
    header = (
        "Context: Apple Inc. Form 10-K, fiscal year ended September 27, 2025, "
        "Item 8 (amounts in millions of U.S. dollars unless a line says otherwise)."
    )
    return header + "\n\n" + "\n".join(label_columns(body) if name in PAIRED else body) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    raw = args.raw.read_text(encoding="utf-8", errors="replace")
    sections = split_sections(item8_lines(raw))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in sections.items():
        text = render(name, body)
        (args.out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"{name}: {len(body)} lines, {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
