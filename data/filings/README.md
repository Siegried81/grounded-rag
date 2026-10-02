# Filings corpus

Drop a SEC 10-K here as `.txt` or `.pdf`, then build the index:

```bash
python cli.py ingest --corpus filings
```

Only top-level files of this folder are ingested, so `raw/` (see below) is left
out of the `filings` corpus.

Where to get a filing (all public, free):

- **SEC EDGAR full-text search** — https://efts.sec.gov/LATEST/search-index?q=...
- A company's filing page, e.g. Apple's 10-K:
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K

## What ships as a worked example

- `raw/apple_fy2025_10k_sec-submission.txt` — the **raw SEC full-submission** text
  file for Apple Inc.'s FY2025 10-K (accession 0000320193-25-000079). It is kept as
  provenance only: it is SGML/HTML with 90 attached documents and XBRL, so it is
  **not** usable as a RAG corpus directly (and lives in `raw/` so ingestion skips
  it).
- `../filings_sections/` — the primary 10-K document extracted from that raw file,
  stripped of markup and split verbatim into six Item sections (Business, Risk
  Factors, Cybersecurity, Legal Proceedings, MD&A, Market Risk). This is the
  ready-to-query corpus:

  ```bash
  python cli.py ingest --corpus filings_sections
  python scripts/run_eval.py --corpus filings_sections \
      --eval-file eval/filings_sections_eval.jsonl --k 3
  ```

Splitting one filing into its Item sections is what makes retrieval evaluation
meaningful: each eval question is labelled with the single section that answers
it, so `recall@k` / `MRR` measure whether retrieval routes to the right section
rather than saturating at 1.0 on a single blob.
