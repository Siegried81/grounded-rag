# Filings corpus

This folder is the provenance of the two 10-K corpora, not a corpus itself: it
holds the raw SEC submission under `raw/` and ingests nothing (there is no
`filings` corpus). To use another filing, put its raw submission in `raw/`, cut it
into Items and Item 8 files in a new `data/<name>/` folder and ingest that:

```bash
python cli.py ingest --corpus <name>
```

Where to get a filing (all public, free):

- **SEC EDGAR full-text search**: https://efts.sec.gov/LATEST/search-index?q=...
- A company's filing page, e.g. Apple's 10-K:
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K

## What ships as a worked example

- `raw/apple_fy2025_10k_sec-submission.txt`: the **raw SEC full-submission** text
  file for Apple Inc.'s FY2025 10-K (accession 0000320193-25-000079). It is kept as
  provenance only: it is SGML/HTML with 90 attached documents and XBRL, so it is
  **not** usable as a RAG corpus directly (and lives in `raw/` so ingestion skips
  it).
- `../filings_sections/`: the primary 10-K document extracted from that raw file,
  stripped of markup and split verbatim into six Item sections (Business, Risk
  Factors, Cybersecurity, Legal Proceedings, MD&A, Market Risk). This is the
  ready-to-query corpus for questions on the text:

  ```bash
  python cli.py ingest --corpus filings_sections
  python scripts/run_eval.py --corpus filings_sections \
      --eval-file eval/filings_sections_eval.jsonl --k 3
  ```

- `../filings_statements/`: Item 8 (Financial Statements and Supplementary Data) in
  19 files: the five statements, the 13 notes and the auditor's reports. It is a
  corpus of its own on purpose: next to the narrative Items its table chunks pushed
  the text questions down, and the text pushed the figures down
  (`docs/decisions.md`, 2026-11-23). It comes from `scripts/extract_10k_item8.py`,
  which flattens each table to `label: 2025 | 2024 | 2023` lines (millions of
  dollars) and writes `07_item8_*.txt`; rerun it to regenerate them.

  ```bash
  python cli.py ingest --corpus filings_statements
  python scripts/run_eval.py --corpus filings_statements \
      --eval-file eval/filings_statements_answers_eval.jsonl --k 5
  ```

Splitting one filing into its Item sections is what makes retrieval evaluation
meaningful: each eval question is labelled with the section that answers it (two
for the one question both answer), so `recall@k` / `MRR` measure whether retrieval routes to the right section
rather than saturating at 1.0 on a single blob.
