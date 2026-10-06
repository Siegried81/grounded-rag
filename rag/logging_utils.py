"""Lightweight structured query logging as append-only JSONL.

One JSON object per line makes it trivial to append safely and to analyse later
(refusal rate, score drift) without a database.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def log_query(
    path: Path,
    *,
    question: str,
    corpus: str,
    n_retrieved: int,
    top_score: float | None,
    refused: bool,
    extra: dict | None = None,
) -> None:
    """Append one query record, with a UTC ISO timestamp, to the JSONL file at `path`.

    Creates parent directories. Never raises: logging is a side concern and must not
    break answering, so any failure (read-only disk, bad path, unserialisable extra)
    is swallowed.
    """
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "corpus": corpus,
            "n_retrieved": n_retrieved,
            "top_score": top_score,
            "refused": refused,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def read_log(path: Path) -> list[dict]:
    """Read all records from a JSONL log; return [] if missing, skipping bad lines.

    Tolerant on purpose so a truncated last line never hides the rest of the log.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def never_crash_on_console_encoding() -> None:
    """Make stdout/stderr drop unencodable characters instead of raising.

    The eval scripts print question and answer text verbatim. A Windows console
    is cp1252, EUR-Lex and the models emit U+2011 (non-breaking hyphen), U+202F
    (narrow no-break space) and U+2019 (curly apostrophe), and `print` then dies
    with a UnicodeEncodeError — after the LLM calls for that run have already
    been made and paid for. Replacing the character costs one glyph of a printed
    line; raising costs the run.

    `backslashreplace` rather than `replace` so the lost character is still
    identifiable in the transcript. The results JSONL is written with an explicit
    utf-8 encoding and is unaffected either way.
    """
    for stream in (sys.stdout, sys.stderr):
        # A replaced stream (pytest's capture, a notebook) may not be a
        # TextIOWrapper, and a console that cannot be reconfigured is not a
        # reason to fail before the run starts.
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass
