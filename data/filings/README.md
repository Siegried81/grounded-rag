# Filings corpus

Drop a SEC 10-K here as `.txt` or `.pdf`, then build the index:

```bash
python cli.py ingest --corpus filings
```

Where to get one (all public, free):

- **SEC EDGAR full-text search** — https://efts.sec.gov/LATEST/search-index?q=...
- A company's filing page, e.g. Apple's 10-K:
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K

Save the document text (the "Risk Factors" / "Management's Discussion" sections
are the most interesting to query) into this folder. Nothing is committed: this
folder is empty on purpose so the repo stays light and carries no third-party
filing text.
