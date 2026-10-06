# AI Act corpus

The full text of **Regulation (EU) 2024/1689** (the Artificial Intelligence Act),
English language version, as published in the Official Journal
(ELI: `http://data.europa.eu/eli/reg/2024/1689/oj`).

```bash
python cli.py ingest --corpus ai_act_sections
python scripts/run_eval.py --corpus ai_act_sections \
    --eval-file eval/ai_act_sections_eval.jsonl --k 3
```

Only top-level files of this folder are ingested, and `README.md` is skipped, so
this file is documentation and not evidence.

## Granularity: one file per unit of the regulation

308 files, 578 426 characters of section text, named so that a plain filename
sort reproduces reading order (this README is not counted; it is not ingested):

| Prefix          | Files | Unit                                                |
| --------------- | ----: | --------------------------------------------------- |
| `01_title_…`    |     1 | Title block and the seven citations ("Having regard…") |
| `02_recital_NNN`|   180 | One recital each                                    |
| `03_article_NNN`|   113 | One article each, headed by its chapter (and section) title |
| `04_final_part` |     1 | Binding clause, place and date, signatures          |
| `05_annex_NN`   |    13 | One annex each (`annex_03` is Annex III)            |

The unit is the unit the regulation itself cites: you ask about "Article 50" or
"Annex III", not about byte range 412000-412900. That is what makes the retrieval
evaluation meaningful — each eval question is labelled with the one or two files
that answer it, so `recall@k` and `MRR` measure whether retrieval routes to the
right article rather than saturating at 1.0 on a single blob. Each article file
repeats its chapter heading on the first line so the opening chunk carries that
context; the regulation number itself appears only in `01_title_and_citations.txt`,
so a question about it has exactly one answering file.

Annex files are numbered `01`-`13` rather than by Roman numeral because
`annex_ix` sorts before `annex_v`.

## Provenance

Extracted from the saved EUR-Lex HTML of the EN version. EUR-Lex lays numbered
lists out as nested two-column tables (marker cell, text cell), so the extraction
folds each marker into the first paragraph of its cell; footnote call-outs are
dropped (their targets are OJ references to other legislation, printed outside the
articles), and the one real superscript is kept as `10^25` in Article 51(2) — the
FLOP threshold reads "1025" if it is flattened.

EUR-Lex answers scripted requests with HTTP 202 and an empty body, so the HTML
has to be saved from a browser; re-running the extraction needs that saved page
and `lxml` (which is not a project dependency — the extraction is a one-off, its
output is what the repo ships).

## Why there is no longer an `ai_act` corpus

`data/ai_act/` used to hold a single document and `data/ai_act_sections/` four
files split from it at its own `##` headings — the same text twice (measured: 207
of 207 unique words shared), which made "two corpora" a presentation of one.
Keeping a whole-regulation copy beside the per-unit split would reproduce exactly
that, so the single-document corpus is retired and this is the only AI Act corpus.
The contrast the README advertises is between this regulation and the SEC 10-K in
`../filings_sections/`: a legal text with articles and annexes against a company
filing with Item sections.
